"""MailGuard ML - Streamlit entry point.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="MailGuard ML", page_icon=":material/shield:", layout="wide")

from src.app_resources import get_model_bundle, get_service  # noqa: E402
from src.styles import inject_styles  # noqa: E402

inject_styles()

pages = [
    st.Page("views/analyzer.py", title="Email spam analyser", icon=":material/mail:", default=True),
    st.Page("views/evaluation.py", title="Model Evaluator", icon=":material/analytics:"),
    st.Page("views/monitoring.py", title="Monitoring", icon=":material/monitor_heart:"),
    st.Page("views/model_info.py", title="Statistics", icon=":material/info:"),
]

get_service()  # warm up model + metrics store once per process
model, model_error = get_model_bundle()

with st.sidebar:
    st.markdown("### :material/shield: MailGuard ML")
    if model is not None:
        st.success(f"Model ready: {model.display_name} v{model.version}", icon=":material/check_circle:")
    else:
        st.error("Model unavailable - see Model Information.", icon=":material/error:")
    st.caption(
        "Educational spam-detection demo trained on the public SpamAssassin corpus. "
        "Emails you analyse are processed in memory and are not stored or logged."
    )

st.navigation(pages).run()
