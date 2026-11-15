"""End-to-end training pipeline for MailGuard ML.

Usage (from the repository root):
    python -m training.train               # download corpus (if needed), train, evaluate, save artifacts
    python -m training.train --skip-download

Outputs (in models/):
    mailguard_model.joblib  - selected model (SpamModel)
    model_card.json         - metadata, test metrics, artifact checksum, limitations
    evaluation.json         - comparison of all candidates, curves, threshold & error analysis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data import (  # noqa: E402
    CORPUS_ARCHIVES,
    CORPUS_BASE_URL,
    deduplicate,
    describe_split,
    download_corpus,
    extract_corpus,
    frame_for,
    group_stratified_split,
    load_corpus,
    validate_labels,
)
from src.features import STRUCTURED_FEATURES  # noqa: E402
from src.metrics import (  # noqa: E402
    apply_threshold,
    calibration_points,
    classification_report_dict,
    compute_classification_metrics,
    pr_points,
    roc_points,
    threshold_table,
)
from src.model import default_candidates, save_model  # noqa: E402
from src.trainer import CandidateResult, fit_candidates, select_best  # noqa: E402

DATASET_INFO = {
    "name": "Apache SpamAssassin Public Mail Corpus",
    "url": CORPUS_BASE_URL,
    "archives": list(CORPUS_ARCHIVES),
    "license_note": (
        "No formal open-source license. Per the corpus readme, messages were posted to public fora or "
        "sent with knowledge they may be made public, and copyright of message text remains with the "
        "original senders. The raw corpus is downloaded at training time and is not redistributed here."
    ),
    "attribution": "SpamAssassin public corpus, curated by Justin Mason / Apache SpamAssassin project.",
}

LIMITATIONS = [
    "Training data is 2002-2005 email (SpamAssassin corpus). Modern spam/phishing styles, languages and "
    "obfuscation techniques are under-represented, so real-world performance will be lower than test metrics.",
    "Ham comes largely from technical mailing lists and newsletters received by a few people; business, "
    "personal and non-English mail is under-represented, which raises false-positive risk on such mail.",
    "Test metrics come from the same corpus (random group-aware split), not from a later time period or a "
    "different organisation; they are optimistic estimates of deployment performance.",
    "Header text is intentionally excluded from TF-IDF to avoid learning collection artefacts; only a few "
    "header properties are used. Pasting a body without headers disables those header signals.",
    "No sender reputation, authentication (SPF/DKIM/DMARC), URL reputation or attachment scanning is performed.",
    "Explanations are per-feature contributions in score space (log-odds/margin); they describe the model, "
    "not ground truth, and character n-gram signals can be hard to interpret.",
    "Educational project: not a substitute for a production mail-security gateway.",
]


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _versions() -> dict[str, str]:
    import joblib
    import lightgbm
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }


def _single_email_latency_ms(result: CandidateResult, X: pd.DataFrame, n: int = 50) -> float:
    """Median latency of predicting one email at a time (feature frame already built)."""
    times = []
    for i in range(min(n, len(X))):
        row = X.iloc[[i]]
        t0 = time.perf_counter()
        result.model.predict_proba(row)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def _evaluate_candidate(result: CandidateResult, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    t0 = time.perf_counter()
    test_prob = result.model.predict_proba(X_test)
    batch_ms = (time.perf_counter() - t0) * 1000
    thr = result.model.threshold
    return {
        "name": result.spec.name,
        "display_name": result.spec.display_name,
        "description": result.spec.description,
        "best_params": result.best_params,
        "grid_results": result.grid_results,
        "calibrated": result.model.calibrator is not None,
        "threshold": thr,
        "n_features": result.extra["n_features"],
        "fit_seconds": round(result.fit_seconds, 2),
        "test_batch_ms_per_email": round(batch_ms / len(X_test), 4),
        "single_email_latency_ms": round(_single_email_latency_ms(result, X_test), 3),
        "validation": result.val_metrics,
        "test": compute_classification_metrics(y_test, test_prob, thr),
        "test_at_0_5": compute_classification_metrics(y_test, test_prob, 0.5),
        "curves": {
            "roc": roc_points(y_test, test_prob),
            "pr": pr_points(y_test, test_prob),
            "calibration": calibration_points(y_test, test_prob),
        },
        "_test_prob": test_prob,  # removed before serialization
    }


def _error_analysis(test_df: pd.DataFrame, test_prob: np.ndarray, threshold: float, model) -> dict:
    y = test_df["label"].to_numpy()
    pred = apply_threshold(test_prob, threshold)
    errors = []
    for pos in np.flatnonzero(pred != y):
        row = test_df.iloc[pos]
        frame = frame_for(test_df.iloc[[pos]])
        top = model.explain(frame, top_k=3)["contributions"]
        errors.append(
            {
                "type": "false_positive" if pred[pos] == 1 else "false_negative",
                "file": row["file"],
                "subset": row["subset"],
                "true_label": "spam" if y[pos] == 1 else "ham",
                "spam_probability": round(float(test_prob[pos]), 4),
                "subject": row["subject"][:120],
                "sender_domain": row["sender_domain"],
                "snippet": row["snippet"][:160],
                "top_signals": [f"{c['feature']} ({c['contribution']:+.2f})" for c in top],
            }
        )
    errors.sort(key=lambda e: (e["type"], -abs(e["spam_probability"] - 0.5)))

    by_subset = []
    for subset, idx in test_df.groupby("subset").indices.items():
        sub_y, sub_pred = y[idx], pred[idx]
        by_subset.append(
            {
                "subset": subset,
                "n": int(len(idx)),
                "true_label": "spam" if sub_y[0] == 1 else "ham",
                "errors": int((sub_y != sub_pred).sum()),
                "error_rate": float((sub_y != sub_pred).mean()),
            }
        )

    # Probability distribution for the selected model (histogram data, 20 bins).
    edges = np.linspace(0, 1, 21)
    hist = {
        "edges": edges.round(2).tolist(),
        "ham": np.histogram(test_prob[y == 0], bins=edges)[0].tolist(),
        "spam": np.histogram(test_prob[y == 1], bins=edges)[0].tolist(),
    }
    return {"errors": errors, "by_subset": sorted(by_subset, key=lambda r: r["subset"]), "probability_histogram": hist}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate MailGuard ML spam classifiers.")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.MODELS_DIR)
    parser.add_argument("--skip-download", action="store_true", help="use already-downloaded archives")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--beta", type=float, default=0.5, help="F-beta used for threshold tuning (default 0.5)")
    parser.add_argument("--small", action="store_true", help="tiny vocabularies for a fast smoke run")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    raw_dir, extract_dir = args.data_dir / "raw", args.data_dir / "extracted"

    # 1. Data acquisition --------------------------------------------------------------
    if args.skip_download:
        missing = [a for a in CORPUS_ARCHIVES if not (raw_dir / a).exists()]
        if missing:
            _log(f"ERROR: missing archives {missing}; run without --skip-download")
            return 2
        checksums = {a: hashlib.sha256((raw_dir / a).read_bytes()).hexdigest() for a in CORPUS_ARCHIVES}
    else:
        checksums = download_corpus(raw_dir, log=_log)
    extract_corpus(raw_dir, extract_dir, log=_log)

    # 2. Parsing, label validation, deduplication --------------------------------------
    df, load_report = load_corpus(extract_dir, log=_log)
    validate_labels(df)
    df, dedup_report = deduplicate(df)
    validate_labels(df)
    _log(f"Deduplication: {dedup_report}")

    # 3. Leakage-safe split ------------------------------------------------------------
    y_all = df["label"].to_numpy()
    train_idx, val_idx, test_idx = group_stratified_split(y_all, df["group_key"].to_numpy(), seed=args.seed)
    train_df, val_df, test_df = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
    n_groups = df["group_key"].nunique()
    _log(f"Split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)} ({n_groups} near-dup groups)")

    X_train, X_val, X_test = frame_for(train_df), frame_for(val_df), frame_for(test_df)
    y_train, y_val, y_test = (d["label"].to_numpy() for d in (train_df, val_df, test_df))
    struct_train = X_train[list(STRUCTURED_FEATURES)]
    constant_features = [f for f in STRUCTURED_FEATURES if struct_train[f].nunique() <= 1]
    feature_prevalence = {f: float((struct_train[f] != 0).mean()) for f in STRUCTURED_FEATURES}
    if constant_features:
        _log(f"Structured features constant in training data (no learnable signal): {constant_features}")

    # 4. Fit & select ---------------------------------------------------------------
    specs = default_candidates(small=args.small)
    results = fit_candidates(
        specs, X_train, y_train, X_val, y_val, seed=args.seed, beta=args.beta, version=config.MODEL_VERSION, log=_log
    )
    best = select_best(results)
    _log(f"Selected model: {best.spec.display_name} (by validation F{args.beta:g}, then PR-AUC)")

    # 5. Final evaluation on the untouched test split --------------------------------
    evaluations = [_evaluate_candidate(r, X_test, y_test) for r in results]
    for ev in evaluations:
        t = ev["test"]
        _log(
            f"TEST {ev['name']:<20} acc={t['accuracy']:.4f} P={t['precision']:.4f} R={t['recall']:.4f} "
            f"F1={t['f1']:.4f} ROC-AUC={t['roc_auc']:.4f} PR-AUC={t['pr_auc']:.4f} thr={ev['threshold']:.2f}"
        )
    best_eval = next(e for e in evaluations if e["name"] == best.spec.name)
    test_prob = best_eval.pop("_test_prob")
    for ev in evaluations:
        ev.pop("_test_prob", None)

    selected_pred = apply_threshold(test_prob, best.model.threshold)
    errors = _error_analysis(test_df, test_prob, best.model.threshold, best.model)

    # 6. Persist artifacts ------------------------------------------------------------
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    versions = _versions()
    best.model.metadata = {
        "trained_at": trained_at,
        "dataset": DATASET_INFO["name"],
        "threshold_objective": f"F{args.beta:g} on validation",
        **{f"{k}_version": v for k, v in versions.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = save_model(best.model, args.output_dir / config.MODEL_FILENAME)
    _log(f"Saved {artifact['path']} ({artifact['size_bytes'] / 1e6:.2f} MB, sha256={artifact['sha256'][:12]}...)")

    splits = {"train": describe_split(train_df), "validation": describe_split(val_df), "test": describe_split(test_df)}
    data_report = {
        "load": load_report,
        "deduplication": dedup_report,
        "near_duplicate_groups": int(n_groups),
        "archive_sha256": checksums,
        "split_method": "StratifiedGroupKFold on near-duplicate groups (~71/14/14), seed=%d" % args.seed,
        "splits": splits,
    }

    evaluation = {
        "generated_at": trained_at,
        "selected_model": best.spec.name,
        "selection_rule": f"Highest validation F{args.beta:g} at the tuned threshold (3 dp), tie-break validation PR-AUC",
        "threshold_objective": f"F{args.beta:g} (precision-weighted) on validation probabilities",
        "models": evaluations,
        "selected": {
            "classification_report": classification_report_dict(y_test, selected_pred),
            "threshold_table_test": threshold_table(y_test, test_prob),
            "threshold_table_validation": threshold_table(y_val, best.val_prob),
            "global_importance": best.model.global_importance(top_k=20),
            **errors,
        },
        "data": data_report,
    }

    model_card = {
        "model_name": "MailGuard Spam Classifier",
        "algorithm": best.spec.display_name,
        "algorithm_key": best.spec.name,
        "version": config.MODEL_VERSION,
        "trained_at": trained_at,
        "hyperparameters": best.best_params,
        "calibration": "Platt scaling fit on validation split" if best.model.calibrator is not None else "none (native probabilities)",
        "decision_threshold": best.model.threshold,
        "threshold_objective": f"maximize F{args.beta:g} on validation",
        "dataset": {**DATASET_INFO, "archive_sha256": checksums, "splits": splits, "deduplication": dedup_report},
        "features": {
            "text": "word TF-IDF (1-2 grams) + char_wb TF-IDF (3-5 grams) on normalized subject+body",
            "structured": list(STRUCTURED_FEATURES),
            "n_features": best.extra["n_features"],
            "constant_in_training": constant_features,
            "prevalence_in_training": feature_prevalence,
        },
        "validation_metrics": best.val_metrics,
        "test_metrics": best_eval["test"],
        "artifact": artifact,
        "environment": versions,
        "training_seconds": round(time.perf_counter() - started, 1),
        "intended_use": "Educational demonstration of an end-to-end spam classification system.",
        "limitations": LIMITATIONS
        + (
            [
                "These structured features never vary in the training data, so the model cannot use them even "
                "though they are extracted and displayed: " + ", ".join(constant_features) + "."
            ]
            if constant_features
            else []
        ),
    }

    with open(args.output_dir / config.EVALUATION_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(evaluation, fh, indent=1, default=_json_default)
    with open(args.output_dir / config.MODEL_CARD_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(model_card, fh, indent=2, default=_json_default)
    _log(f"Wrote {config.EVALUATION_FILENAME} and {config.MODEL_CARD_FILENAME}; total {time.perf_counter() - started:.1f}s")
    return 0


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


if __name__ == "__main__":
    raise SystemExit(main())
