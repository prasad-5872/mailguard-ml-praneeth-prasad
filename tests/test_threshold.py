import pytest

from src.inference import InferenceService
from src.metrics import apply_threshold
from src.validation import ValidationError


def test_apply_threshold_boundary_is_spam():
    assert apply_threshold([0.49, 0.5, 0.51], 0.5).tolist() == [0, 1, 1]


def test_apply_threshold_extremes():
    assert apply_threshold([0.0, 1.0], 0.0).tolist() == [1, 1]
    assert apply_threshold([0.0, 0.999], 1.0).tolist() == [0, 0]


def test_apply_threshold_scalar_and_invalid():
    assert apply_threshold(0.7, 0.5).tolist() == [1]
    with pytest.raises(ValueError):
        apply_threshold([0.5], 1.5)


def test_service_uses_model_threshold_by_default(tiny_model, metrics_store, test_logger, sample_bytes):
    service = InferenceService(tiny_model, metrics_store, test_logger)
    result = service.analyze(sample_bytes("ham_meeting_notes.eml"))
    assert result.threshold == tiny_model.threshold


def test_custom_threshold_changes_label(tiny_model, metrics_store, test_logger, sample_bytes):
    service = InferenceService(tiny_model, metrics_store, test_logger)
    raw = sample_bytes("spam_prize_html.eml")
    assert service.analyze(raw, threshold=0.0).label == "spam"
    result = service.analyze(raw, threshold=1.0)
    assert result.label == ("spam" if result.spam_probability >= 1.0 else "ham")


def test_relabel_without_new_request(tiny_model, metrics_store, test_logger, sample_bytes):
    service = InferenceService(tiny_model, metrics_store, test_logger)
    result = service.analyze(sample_bytes("spam_prize_html.eml"), threshold=0.5)
    before = metrics_store.snapshot()["total_requests"]
    p = result.spam_probability
    assert result.label_at(max(0.0, p - 0.01)) == "spam"
    if p < 1.0:
        assert result.label_at(min(1.0, p + 0.01)) == "ham"
    assert metrics_store.snapshot()["total_requests"] == before
    with pytest.raises(ValidationError):
        result.label_at(2.0)


def test_invalid_threshold_is_rejected_and_recorded(tiny_model, metrics_store, test_logger, sample_bytes):
    service = InferenceService(tiny_model, metrics_store, test_logger)
    with pytest.raises(ValidationError):
        service.analyze(sample_bytes("ham_meeting_notes.eml"), threshold=1.5)
    snap = metrics_store.snapshot()
    assert snap["error_count"] == 1 and snap["successful_requests"] == 0
    assert metrics_store.recent(1)[0]["error_type"] == "validation_error"
