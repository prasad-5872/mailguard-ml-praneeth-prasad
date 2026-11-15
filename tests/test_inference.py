import json
import logging

import numpy as np
import pytest
import scipy.sparse as sp

from src.features import STRUCTURED_FEATURES, build_feature_frame
from src.inference import InferenceError, InferenceService, ModelUnavailableError
from src.logging_utils import JsonFormatter, RingBufferHandler
from src.parsing import parse_email
from src.validation import ValidationError


@pytest.fixture
def ring_logger(test_logger):
    ring = RingBufferHandler(100)
    ring.setFormatter(JsonFormatter())
    test_logger.addHandler(ring)
    return test_logger, ring


def test_all_candidates_produce_valid_probabilities(models_by_name, synthetic_split):
    _, _, X_val, y_val = synthetic_split
    assert set(models_by_name) == {"naive_bayes", "logistic_regression", "linear_svm", "lightgbm"}
    for name, model in models_by_name.items():
        prob = model.predict_proba(X_val)
        assert prob.shape == (len(X_val),), name
        assert np.all((prob >= 0) & (prob <= 1)), name
        # synthetic data is easily separable: every model should rank spam above ham
        assert prob[y_val == 1].mean() > prob[y_val == 0].mean(), name
        assert 0.0 <= model.threshold <= 1.0
        assert set(np.unique(model.predict(X_val))) <= {0, 1}


def test_calibration_applied_where_required(models_by_name):
    assert models_by_name["naive_bayes"].calibrator is not None
    assert models_by_name["linear_svm"].calibrator is not None
    assert models_by_name["logistic_regression"].calibrator is None
    assert models_by_name["lightgbm"].calibrator is None


def test_successful_inference_result(tiny_model, metrics_store, ring_logger, sample_bytes):
    logger, ring = ring_logger
    service = InferenceService(tiny_model, metrics_store, logger)
    result = service.analyze(sample_bytes("spam_account_phishing.eml"), source="upload")
    assert result.label in {"spam", "ham"}
    assert 0.0 <= result.spam_probability <= 1.0
    assert result.latency_ms > 0
    assert len(result.request_id) == 12
    assert set(result.features) == set(STRUCTURED_FEATURES)
    assert result.parsed.subject.startswith("URGENT")
    assert result.explanation["contributions"]

    snap = metrics_store.snapshot()
    assert snap["total_requests"] == 1 and snap["successful_requests"] == 1 and snap["error_count"] == 0
    assert snap["spam_count"] + snap["ham_count"] == 1
    rec = metrics_store.recent(1)[0]
    assert rec["request_id"] == result.request_id and rec["source"] == "upload"

    log = ring.records(1)[0]
    assert log["message"] == "inference_success" and log["request_id"] == result.request_id
    assert log["latency_ms"] > 0


def test_raw_content_is_not_logged(tiny_model, metrics_store, ring_logger):
    logger, ring = ring_logger
    secret = "TOPSECRETBODYMARKER"
    service = InferenceService(tiny_model, metrics_store, logger)
    service.analyze(f"From: a@example.com\nSubject: {secret} subject\n\nbody with {secret}")
    with pytest.raises(ValidationError):
        service.analyze("   ")
    dumped = json.dumps(ring.records())
    assert secret not in dumped


def test_validation_error_recorded(tiny_model, metrics_store, ring_logger):
    logger, ring = ring_logger
    service = InferenceService(tiny_model, metrics_store, logger)
    with pytest.raises(ValidationError):
        service.analyze("")
    snap = metrics_store.snapshot()
    assert snap["error_count"] == 1 and snap["error_rate"] == 1.0
    assert ring.records(1)[0]["level"] == "WARNING"


def test_model_unavailable(metrics_store, test_logger, sample_bytes):
    service = InferenceService(None, metrics_store, test_logger, model_error="artifact missing")
    assert not service.ready
    with pytest.raises(ModelUnavailableError, match="artifact missing"):
        service.analyze(sample_bytes("ham_meeting_notes.eml"))
    assert metrics_store.recent(1)[0]["error_type"] == "model_unavailable"


def test_internal_error_wrapped_with_request_id(tiny_model, metrics_store, ring_logger, sample_bytes, monkeypatch):
    logger, ring = ring_logger
    service = InferenceService(tiny_model, metrics_store, logger)

    def boom(_frame):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(tiny_model, "predict_proba", boom)
    with pytest.raises(InferenceError) as info:
        service.analyze(sample_bytes("ham_meeting_notes.eml"))
    assert metrics_store.recent(1)[0]["request_id"] == info.value.request_id
    assert metrics_store.recent(1)[0]["error_type"] == "internal_error"
    entry = ring.records(1)[0]
    assert entry["level"] == "ERROR" and "exception" in entry


@pytest.mark.parametrize("name", ["naive_bayes", "logistic_regression", "linear_svm"])
def test_linear_explanations_are_additive(models_by_name, sample_bytes, name):
    model = models_by_name[name]
    frame = build_feature_frame([parse_email(sample_bytes("spam_prize_html.eml"))])
    exp = model.explain(frame, top_k=10**6)
    total = exp["base_value"] + sum(c["contribution"] for c in exp["contributions"])
    assert total == pytest.approx(exp["score"], rel=1e-4, abs=1e-4)
    assert exp["score"] == pytest.approx(float(model.decision_scores(frame)[0]), rel=1e-6)
    contribs = [abs(c["contribution"]) for c in exp["contributions"]]
    assert contribs == sorted(contribs, reverse=True)
    # linear contributions only come from text that is present in the email
    assert all(c["value"] > 0 for c in exp["contributions"] if c["kind"] != "structured")


def test_tree_shap_explanations_are_additive(models_by_name, sample_bytes):
    model = models_by_name["lightgbm"]
    frame = build_feature_frame([parse_email(sample_bytes("spam_account_phishing.eml"))])
    exp = model.explain(frame, top_k=10**6)
    assert exp["method"].startswith("TreeSHAP")
    total = exp["base_value"] + sum(c["contribution"] for c in exp["contributions"])
    assert total == pytest.approx(exp["score"], abs=1e-6)
    for c in exp["contributions"]:
        if c["kind"] != "structured":
            assert ("absent" in c["description"]) == (c["value"] == 0)


def test_tree_shap_matches_shap_library(models_by_name, sample_bytes):
    shap = pytest.importorskip("shap")
    model = models_by_name["lightgbm"]
    frame = build_feature_frame([parse_email(sample_bytes("spam_account_phishing.eml"))])
    X = model.transform(frame)
    native = model.estimator.predict(X, pred_contrib=True)
    native = native.toarray() if sp.issparse(native) else np.asarray(native)
    values = shap.TreeExplainer(model.estimator.booster_).shap_values(X.toarray())
    values = values[1] if isinstance(values, list) else values
    np.testing.assert_allclose(np.ravel(values), native.ravel()[:-1], atol=1e-6)


def test_global_importance(models_by_name):
    lin = models_by_name["logistic_regression"].global_importance(top_k=5)
    assert len(lin["spam_indicators"]) == 5 and len(lin["ham_indicators"]) == 5
    assert lin["spam_indicators"][0]["weight"] >= lin["spam_indicators"][-1]["weight"]
    tree = models_by_name["lightgbm"].global_importance(top_k=5)
    assert 0 < len(tree["importance"]) <= 5


def test_missing_columns_rejected(tiny_model):
    import pandas as pd

    with pytest.raises(ValueError, match="missing columns"):
        tiny_model.transform(pd.DataFrame({"text": ["hello"]}))


def test_logging_setup_is_idempotent():
    from src.logging_utils import get_log_buffer, get_logger

    a, b = get_logger(), get_logger()
    assert a is b
    assert len([h for h in a.handlers if isinstance(h, RingBufferHandler)]) == 1
    assert get_log_buffer(a).capacity > 0
    assert a.level == logging.INFO
