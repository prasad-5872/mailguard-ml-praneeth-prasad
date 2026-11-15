"""Leakage-safe model fitting and selection.

Protocol
--------
1. Preprocessors (TF-IDF vocabularies/IDF weights, scalers) are fit on the TRAIN split only.
2. Each candidate's hyper-parameters are chosen by validation PR-AUC.
3. Probability calibration (Platt scaling, for NB and SVM) is fit on the VALIDATION split.
   Platt scaling is monotone, so it does not change ranking metrics.
4. The decision threshold is tuned on the (calibrated) VALIDATION probabilities.
5. The best candidate is selected on validation metrics. The TEST split is only used
   to report final metrics; it never influences any fitted parameter or choice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.metrics import compute_classification_metrics, find_optimal_threshold
from src.model import CandidateSpec, SpamModel, build_preprocessor, fit_platt_calibrator


@dataclass
class CandidateResult:
    spec: CandidateSpec
    model: SpamModel
    best_params: dict[str, Any]
    grid_results: list[dict[str, Any]]
    val_prob: np.ndarray
    val_metrics: dict[str, Any]
    threshold_objective: float
    fit_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)


def fit_candidates(
    specs: list[CandidateSpec],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    seed: int = 42,
    beta: float = 0.5,
    version: str = "0.0.0",
    log=print,
) -> list[CandidateResult]:
    """Fit every candidate following the leakage-safe protocol described in the module docstring."""
    y_train = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)
    fitted_pre: dict[Any, tuple[Any, Any, Any, float]] = {}
    results: list[CandidateResult] = []

    for spec in specs:
        t0 = time.perf_counter()
        key = spec.preprocessor
        if key not in fitted_pre:
            tp = time.perf_counter()
            pre = build_preprocessor(spec.preprocessor)
            Xtr = pre.fit_transform(X_train)  # fit on TRAIN only
            Xva = pre.transform(X_val)
            fitted_pre[key] = (pre, Xtr, Xva, time.perf_counter() - tp)
            log(f"  preprocessor {key.struct_scaling}: {Xtr.shape[1]:,} features ({fitted_pre[key][3]:.1f}s)")
        pre, Xtr, Xva, pre_seconds = fitted_pre[key]

        grid_results = []
        best = None
        for params in spec.param_grid:
            est = spec.factory(params, seed)
            est.fit(Xtr, y_train)
            probe = SpamModel(spec.name, spec.display_name, pre, est, spec.explainer, 0.5)
            val_scores = probe._scores_from_matrix(Xva)
            ap = float(average_precision_score(y_val, val_scores))
            grid_results.append({"params": params, "val_pr_auc": ap})
            log(f"  {spec.name} {params}: val PR-AUC={ap:.4f}")
            if best is None or ap > best[0] + 1e-9:
                best = (ap, params, est, val_scores)

        assert best is not None
        _, params, est, val_scores = best
        calibrator = fit_platt_calibrator(val_scores, y_val) if spec.calibrate else None
        model = SpamModel(
            name=spec.name,
            display_name=spec.display_name,
            preprocessor=pre,
            estimator=est,
            explainer=spec.explainer,
            threshold=0.5,
            calibrator=calibrator,
            version=version,
            params=dict(params),
        )
        val_prob = model.scores_to_proba(val_scores)
        threshold, objective = find_optimal_threshold(y_val, val_prob, beta=beta)
        model.threshold = threshold
        val_metrics = compute_classification_metrics(y_val, val_prob, threshold)
        fit_seconds = time.perf_counter() - t0  # includes preprocessor fit if first user of this config
        log(
            f"  -> {spec.name}: best={params} threshold={threshold:.2f} "
            f"val F{beta:g}={objective:.4f} val PR-AUC={val_metrics['pr_auc']:.4f}"
        )
        results.append(
            CandidateResult(
                spec, model, dict(params), grid_results, val_prob, val_metrics, objective, fit_seconds,
                extra={"preprocessor_fit_seconds": pre_seconds, "n_features": int(Xtr.shape[1])},
            )
        )
    return results


def select_best(results: list[CandidateResult]) -> CandidateResult:
    """Select by validation F-beta at the tuned threshold (the deployment objective, rounded to 3 dp),
    tie-broken by validation PR-AUC, then validation F1. The test split is never consulted."""
    if not results:
        raise ValueError("no candidate results to select from")

    def key(r: CandidateResult):
        return (round(r.threshold_objective, 3), r.val_metrics["pr_auc"] or 0.0, r.val_metrics["f1"])

    return max(results, key=key)
