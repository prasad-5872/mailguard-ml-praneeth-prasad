"""Headless UI tests: run the real Streamlit app script with streamlit.testing.AppTest."""

import pytest

from src import config

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (config.MODEL_PATH.exists() and config.EVALUATION_PATH.exists() and config.MODEL_CARD_PATH.exists()),
    reason="trained artifacts not present; run training first",
)

APP = str(config.ROOT_DIR / "app.py")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_analyzer_page_runs_sample_and_updates_monitoring():
    at = _fresh_app()
    assert at.title[0].value == "Email Analyzer"

    from src.app_resources import get_metrics_store

    before = get_metrics_store().snapshot()["total_requests"]
    at.selectbox[0].set_value("spam_prize_html.eml").run()
    next(b for b in at.button if b.label == "Analyze sample").click().run()
    assert not at.exception, [e.value for e in at.exception]
    verdict = next(m for m in at.metric if m.label == "Verdict")
    assert verdict.value == "SPAM"
    assert get_metrics_store().snapshot()["total_requests"] == before + 1

    # Moving the threshold re-labels locally without a new request.
    at.slider(key="threshold").set_value(1.0).run()
    assert next(m for m in at.metric if m.label == "Verdict").value in {"HAM", "SPAM"}
    assert get_metrics_store().snapshot()["total_requests"] == before + 1


def test_analyzer_rejects_empty_paste_and_counts_error():
    at = _fresh_app()
    from src.app_resources import get_metrics_store

    errors_before = get_metrics_store().snapshot()["error_count"]
    next(b for b in at.button if b.label == "Analyze pasted email").click().run()
    assert any("empty" in w.value.lower() for w in at.warning)
    assert get_metrics_store().snapshot()["error_count"] == errors_before + 1


def test_analyzer_body_only_paste():
    at = _fresh_app()
    at.text_area[0].set_value("Hi team, the review moved to Thursday at 3pm. Agenda attached in the doc.").run()
    next(b for b in at.button if b.label == "Analyze pasted email").click().run()
    assert not at.exception
    assert any("No email headers were detected" in w.value for w in at.warning)


@pytest.mark.parametrize(
    "page,title",
    [
        ("views/evaluation.py", "Model Evaluation"),
        ("views/monitoring.py", "Monitoring"),
        ("views/model_info.py", "Model Information"),
    ],
)
def test_other_pages_render(page, title):
    at = _fresh_app()
    at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == title
