"""Page 4 - Model Information (model card)."""

from __future__ import annotations

import platform

import pandas as pd
import streamlit as st

from src import config
from src.app_resources import get_model_bundle, load_artifact_json
from src.features import FEATURE_DESCRIPTIONS

st.title("Model Information")

card = load_artifact_json(config.MODEL_CARD_PATH)
model, model_error = get_model_bundle()

if model_error:
    st.error(f"**Model failed to load:** {model_error}", icon=":material/error:")
if not card:
    st.warning("`models/model_card.json` not found. Run `python -m training.train` to generate the artifacts.")
    st.stop()

t = card["test_metrics"]
c = st.columns(4)
c[0].metric("Model", card["algorithm"])
c[1].metric("Version", card["version"])
c[2].metric("Trained (UTC)", card["trained_at"][:10])
c[3].metric("Decision threshold", f"{card['decision_threshold']:.2f}")

st.subheader("Overview")
overview = pd.DataFrame(
    [
        ("Name", card["model_name"]),
        ("Algorithm", f"{card['algorithm']} ({card['algorithm_key']})"),
        ("Version", card["version"]),
        ("Training date", card["trained_at"]),
        ("Hyper-parameters", ", ".join(f"{k}={v}" for k, v in card["hyperparameters"].items())),
        ("Calibration", card["calibration"]),
        ("Threshold objective", card["threshold_objective"]),
        ("Input features", f"{card['features']['n_features']:,} ({card['features']['text']}; "
                           f"{len(card['features']['structured'])} structured)"),
        ("Training time", f"{card['training_seconds']} s (full pipeline incl. parsing)"),
        ("Intended use", card["intended_use"]),
    ],
    columns=["field", "value"],
)
st.dataframe(overview, hide_index=True, width="stretch")

st.subheader("Test metrics")
st.caption(f"Held-out test split ({t['n_samples']:,} emails: {t['n_spam']:,} spam, {t['n_ham']:,} ham) at threshold {t['threshold']:.2f}.")
m = st.columns(6)
for col, (name, key) in zip(m, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                                 ("F1", "f1"), ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc")]):
    col.metric(name, f"{t[key]:.4f}")
cm = t["confusion_matrix"]
st.caption(f"Confusion matrix: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']} · Brier score {t['brier']:.4f} · "
           f"false positive rate {t['false_positive_rate']:.4f}")

st.subheader("Dataset")
ds = card["dataset"]
st.markdown(f"**{ds['name']}** - [{ds['url']}]({ds['url']})")
st.markdown(f"*Attribution:* {ds['attribution']}")
st.markdown(f"*License / usage:* {ds['license_note']}")
dd = ds["deduplication"]
st.markdown(
    f"Archives used: {', '.join(f'`{a}`' for a in ds['archives'])}. "
    f"{dd['rows_before']:,} parsed messages → {dd['exact_duplicates_dropped']:,} exact duplicates and "
    f"{dd['conflicting_label_rows_dropped']:,} conflicting-label rows removed → {dd['rows_after']:,} unique emails."
)
st.dataframe(
    pd.DataFrame([{"split": k, "emails": v["n"], "spam": v["n_spam"], "ham": v["n_ham"], "spam ratio": round(v["spam_ratio"], 3)}
                  for k, v in ds["splits"].items()]),
    hide_index=True, width="stretch",
)

st.subheader("Artifact")
art = card["artifact"]
loaded_ok = model is not None
st.dataframe(
    pd.DataFrame(
        [
            ("File", f"models/{art['path']}"),
            ("Size", f"{art['size_bytes'] / 1e6:.2f} MB"),
            ("SHA-256", art["sha256"]),
            ("Integrity check", "verified at load" if loaded_ok else "FAILED / not loaded"),
            ("Format", "joblib (pickle) - load only artifacts you trust"),
        ],
        columns=["field", "value"],
    ),
    hide_index=True, width="stretch",
)
if model is not None and model.metadata.get("load_warnings"):
    with st.expander("Warnings raised while loading the artifact"):
        for w in model.metadata["load_warnings"]:
            st.code(w, language=None)

st.markdown("**Environment** (training vs. this app)")
import lightgbm  # noqa: E402
import numpy  # noqa: E402
import sklearn  # noqa: E402

runtime = {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
           "numpy": numpy.__version__, "pandas": pd.__version__}
env_rows = [{"package": k, "training": card["environment"].get(k, "-"), "runtime": v,
             "match": "yes" if card["environment"].get(k) == v else "differs"} for k, v in runtime.items()]
st.dataframe(pd.DataFrame(env_rows), hide_index=True, width="stretch")
if any(r["match"] == "differs" and r["package"] == "scikit-learn" for r in env_rows):
    st.warning("scikit-learn differs from the training version; pickled models may behave differently. Pin versions or retrain.")

st.subheader("Features")
st.dataframe(
    pd.DataFrame([{"structured feature": f, "description": FEATURE_DESCRIPTIONS.get(f, "")} for f in card["features"]["structured"]]),
    hide_index=True, width="stretch",
)
if model is not None:
    imp = model.global_importance(top_k=15)
    if "spam_indicators" in imp:
        a, b = st.columns(2)
        a.markdown("**Strongest spam indicators** (weight in score space)")
        a.dataframe(pd.DataFrame(imp["spam_indicators"]), hide_index=True, width="stretch")
        b.markdown("**Strongest ham indicators**")
        b.dataframe(pd.DataFrame(imp["ham_indicators"]), hide_index=True, width="stretch")
        st.caption("Text features are normalized: URLs → `urltoken`, email addresses → `emailtoken`, digits → `0`.")
    else:
        st.markdown("**Top features by LightGBM gain**")
        st.dataframe(pd.DataFrame(imp["importance"]), hide_index=True, width="stretch")

st.subheader("Known limitations")
for item in card["limitations"]:
    st.markdown(f"- {item}")
st.caption("Monitoring metrics are in-memory only and reset when the app restarts.")
