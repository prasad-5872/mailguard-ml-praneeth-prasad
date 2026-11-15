"""Page 3 - Monitoring dashboard (live, in-memory, updated by real inference requests)."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.app_resources import get_metrics_store, theme_is_dark
from src.charts import class_distribution_chart, latency_chart
from src.logging_utils import get_log_buffer

store = get_metrics_store()
dark = theme_is_dark()

st.title("Monitoring")
st.caption(
    f"In-memory metrics shared by all users of this app instance. Counters are cumulative since the process "
    f"started; latency p95, charts and history cover the last {store.window_size:,} requests, and the log view keeps "
    f"the last {get_log_buffer().capacity:,} log entries. **Everything resets when the app restarts** "
    "(redeploy, crash, or Streamlit Community Cloud putting the app to sleep)."
)

auto = st.toggle("Auto-refresh every 5 seconds", value=False)


def _fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f} ms"


@st.fragment(run_every="5s" if auto else None)
def dashboard() -> None:
    snap = store.snapshot()
    st.caption(f"Process started {snap['started_at']} (UTC)")
    k = st.columns(6)
    k[0].metric("Total requests", f"{snap['total_requests']:,}")
    k[1].metric("Successful", f"{snap['successful_requests']:,}")
    k[2].metric("Errors", f"{snap['error_count']:,}")
    k[3].metric("Error rate", f"{snap['error_rate']:.1%}")
    k[4].metric("Avg latency", _fmt_ms(snap["avg_latency_ms"]), help="Mean over all requests since process start")
    k[5].metric("p95 latency", _fmt_ms(snap["p95_latency_ms"]),
                help=f"95th percentile over the last {snap['records_in_window']} requests (success + error)")

    records = store.recent()
    if not records:
        st.info("No requests yet. Analyze an email on the **Email Analyzer** page and come back - every request "
                "(including rejected input) is counted here.", icon=":material/info:")
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Prediction distribution** (successful requests)")
        st.plotly_chart(class_distribution_chart(snap["spam_count"], snap["ham_count"], dark), width="stretch",
                        config={"displayModeBar": False})
        if snap["p95_success_latency_ms"] is not None:
            st.caption(f"p95 latency of successful requests: {snap['p95_success_latency_ms']:.1f} ms")
    with c2:
        st.markdown("**Latency per request**")
        st.plotly_chart(latency_chart(records, dark), width="stretch")

    st.markdown("**Request history** (newest first)")
    hist = pd.DataFrame(records)
    st.dataframe(
        hist[["timestamp", "request_id", "status", "source", "prediction", "spam_probability", "threshold",
              "latency_ms", "input_bytes", "error_type", "error_message"]],
        hide_index=True, width="stretch",
        column_config={
            "spam_probability": st.column_config.NumberColumn(format="%.4f"),
            "latency_ms": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    logs = get_log_buffer().records(200)
    st.markdown("**Recent structured logs** (newest first, raw email content is never logged)")
    if logs:
        log_df = pd.DataFrame(logs)
        preferred = [c for c in ("timestamp", "level", "message", "request_id", "event", "prediction",
                                 "spam_probability", "latency_ms", "error_type", "error") if c in log_df.columns]
        st.dataframe(log_df[preferred], hide_index=True, width="stretch")
        st.download_button("Download logs (JSON Lines)", "\n".join(json.dumps(r, default=str) for r in logs),
                           file_name="mailguard_logs.jsonl", mime="application/json", icon=":material/download:")


dashboard()
