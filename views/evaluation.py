"""Page 2 - Model Evaluation (all numbers come from models/evaluation.json written by training)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.app_resources import load_artifact_json, theme_is_dark
from src.charts import (
    calibration_chart,
    confusion_matrix_chart,
    curve_chart,
    probability_histogram,
    threshold_chart,
)

dark = theme_is_dark()
st.title("Model Evaluation")

ev = load_artifact_json(config.EVALUATION_PATH)
if not ev:
    st.warning("`models/evaluation.json` not found. Run `python -m training.train` to generate it.")
    st.stop()

models = ev["models"]
by_name = {m["name"]: m for m in models}
selected = ev["selected_model"]
sel = ev["selected"]
splits = ev["data"]["splits"]

st.caption(
    f"Generated {ev['generated_at']} · test split: {splits['test']['n']:,} emails "
    f"({splits['test']['n_spam']:,} spam / {splits['test']['n_ham']:,} ham). "
    "The test split was not used for fitting, calibration, threshold tuning or model selection."
)
st.info(f"**Selected model:** {by_name[selected]['display_name']}. **Selection rule:** {ev['selection_rule']}. "
        f"**Threshold objective:** {ev['threshold_objective']}.", icon=":material/rule:")

# ---------------------------------------------------------------- comparison table
st.subheader("Model comparison")
rows = []
for m in models:
    t, v = m["test"], m["validation"]
    rows.append(
        {
            "Model": m["display_name"] + (" ★" if m["name"] == selected else ""),
            "Calibrated": "Platt" if m["calibrated"] else "native",
            "Threshold": m["threshold"],
            "Accuracy": t["accuracy"],
            "Precision": t["precision"],
            "Recall": t["recall"],
            "F1": t["f1"],
            "ROC-AUC": t["roc_auc"],
            "PR-AUC": t["pr_auc"],
            "Brier": t["brier"],
            "FP": t["confusion_matrix"]["fp"],
            "FN": t["confusion_matrix"]["fn"],
            "Val F0.5": v["f0_5"],
            "Val PR-AUC": v["pr_auc"],
            "Fit (s)": m["fit_seconds"],
            "Latency (ms/email)": m["single_email_latency_ms"],
        }
    )
fmt4 = st.column_config.NumberColumn(format="%.4f")
st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    width="stretch",
    column_config={k: fmt4 for k in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier", "Val F0.5", "Val PR-AUC")}
    | {"Threshold": st.column_config.NumberColumn(format="%.2f")},
)
st.caption("Test metrics at each model's own tuned threshold. Latency = median single-email predict time on the "
           "training machine (feature extraction excluded). ★ = deployed model.")

with st.expander("Hyper-parameter search (validation PR-AUC)"):
    grid_rows = [
        {"Model": m["display_name"], "Params": str(g["params"]), "Val PR-AUC": g["val_pr_auc"]}
        for m in models for g in m["grid_results"]
    ]
    st.dataframe(pd.DataFrame(grid_rows), hide_index=True, width="stretch", column_config={"Val PR-AUC": fmt4})

# ---------------------------------------------------------------- curves
st.subheader("ROC and precision-recall curves (test split)")
c1, c2 = st.columns(2)
with c1:
    roc = [{"name": m["display_name"], "x": m["curves"]["roc"]["fpr"], "y": m["curves"]["roc"]["tpr"], "auc": m["test"]["roc_auc"]} for m in models]
    st.plotly_chart(curve_chart(roc, "roc", dark), width="stretch")
with c2:
    pr = [{"name": m["display_name"], "x": m["curves"]["pr"]["recall"], "y": m["curves"]["pr"]["precision"], "auc": m["test"]["pr_auc"]} for m in models]
    st.plotly_chart(curve_chart(pr, "pr", dark), width="stretch")
st.caption("Tip: curves are near the top-left/top-right corners; use the plot zoom to compare models.")

# ---------------------------------------------------------------- per-model detail
st.subheader("Per-model detail")
names = [m["name"] for m in models]
choice = st.selectbox("Model", names, index=names.index(selected), format_func=lambda n: by_name[n]["display_name"])
m = by_name[choice]
d1, d2 = st.columns([1, 1])
with d1:
    st.markdown(f"**Confusion matrix** (test, threshold {m['threshold']:.2f})")
    st.plotly_chart(confusion_matrix_chart(m["test"]["confusion_matrix"], dark), width="stretch")
with d2:
    t, t5 = m["test"], m["test_at_0_5"]
    metric_df = pd.DataFrame(
        {
            "metric": ["Accuracy", "Precision", "Recall", "F1", "F0.5", "Specificity", "False positive rate", "ROC-AUC", "PR-AUC", "Brier score"],
            f"tuned ({m['threshold']:.2f})": [t["accuracy"], t["precision"], t["recall"], t["f1"], t["f0_5"], t["specificity"], t["false_positive_rate"], t["roc_auc"], t["pr_auc"], t["brier"]],
            "at 0.50": [t5["accuracy"], t5["precision"], t5["recall"], t5["f1"], t5["f0_5"], t5["specificity"], t5["false_positive_rate"], t5["roc_auc"], t5["pr_auc"], t5["brier"]],
        }
    )
    st.markdown("**Classification metrics** (test)")
    st.dataframe(metric_df, hide_index=True, width="stretch", column_config={c: fmt4 for c in metric_df.columns[1:]})
    st.caption(m["description"])

st.markdown(f"**Classification report** - {by_name[selected]['display_name']} (deployed model, test, tuned threshold)")
report = sel["classification_report"]
rep_rows = [
    {"class": k, "precision": v["precision"], "recall": v["recall"], "f1-score": v["f1-score"], "support": int(v["support"])}
    for k, v in report.items() if isinstance(v, dict)
]
st.dataframe(pd.DataFrame(rep_rows), hide_index=True, width="stretch",
             column_config={c: fmt4 for c in ("precision", "recall", "f1-score")})

st.markdown("**Probability calibration** (test, reliability diagram)")
cal = [{"name": mm["display_name"], "x": mm["curves"]["calibration"]["mean_predicted"],
        "y": mm["curves"]["calibration"]["fraction_positive"], "count": mm["curves"]["calibration"]["count"]} for mm in models]
st.plotly_chart(calibration_chart(cal, dark), width="stretch")
st.caption("Probability bins holding fewer than 5 test emails are hidden (too noisy). Brier score (lower is better) summarises calibration + sharpness. NB and SVM scores are Platt-scaled on validation data.")

# ---------------------------------------------------------------- threshold analysis
st.subheader("Threshold analysis - deployed model")
split_choice = st.radio("Split", ["test", "validation"], horizontal=True,
                        help="The tuned threshold was chosen on validation; the test view shows how it generalises.")
table = sel["threshold_table_test"] if split_choice == "test" else sel["threshold_table_validation"]
sel_thr = by_name[selected]["threshold"]
st.plotly_chart(threshold_chart(table, sel_thr, dark), width="stretch")
st.dataframe(
    pd.DataFrame(table).rename(columns={"false_positive_rate": "FPR", "false_positives": "FP", "false_negatives": "FN"}),
    hide_index=True, width="stretch",
    column_config={c: fmt4 for c in ("precision", "recall", "f1", "f0_5", "FPR")} | {"threshold": st.column_config.NumberColumn(format="%.2f")},
)

# ---------------------------------------------------------------- error analysis
st.subheader("Error analysis - deployed model (test split)")
e1, e2 = st.columns([1, 1])
with e1:
    st.markdown("**Error rate by corpus subset**")
    st.dataframe(pd.DataFrame(sel["by_subset"]), hide_index=True, width="stretch",
                 column_config={"error_rate": st.column_config.NumberColumn(format="%.4f")})
    st.caption("`hard_ham` is legitimate mail that deliberately resembles spam (HTML, marketing language) - expect more false positives there.")
with e2:
    st.markdown("**Predicted probability by true class**")
    st.plotly_chart(probability_histogram(sel["probability_histogram"], dark), width="stretch")

errors = sel["errors"]
fp = [e for e in errors if e["type"] == "false_positive"]
fn = [e for e in errors if e["type"] == "false_negative"]
st.markdown(f"**Misclassified test emails** - {len(fp)} false positives (ham → spam), {len(fn)} false negatives (spam → ham)")
if errors:
    err_df = pd.DataFrame(errors)
    err_df["top_signals"] = err_df["top_signals"].map(lambda s: "; ".join(s))
    kind = st.segmented_control("Show", ["false_positive", "false_negative"], default="false_positive",
                                format_func=lambda k: "False positives" if k == "false_positive" else "False negatives")
    if kind:
        err_df = err_df[err_df["type"] == kind]
    st.dataframe(
        err_df[["subset", "spam_probability", "subject", "sender_domain", "top_signals", "snippet", "file"]],
        hide_index=True, width="stretch",
        column_config={"spam_probability": st.column_config.ProgressColumn("spam prob.", min_value=0.0, max_value=1.0, format="%.3f")},
    )
    st.caption("Subjects/snippets are from the public SpamAssassin corpus (copyright remains with original senders), shown truncated for error analysis.")

# ---------------------------------------------------------------- data
with st.expander("Dataset preparation report"):
    data = ev["data"]
    st.json({"load": data["load"], "deduplication": data["deduplication"],
             "near_duplicate_groups": data["near_duplicate_groups"], "split_method": data["split_method"]})
    st.dataframe(
        pd.DataFrame([{"split": k, **{kk: vv for kk, vv in v.items() if kk != "by_subset"}, **v["by_subset"]} for k, v in splits.items()]),
        hide_index=True, width="stretch",
    )
