import json

import joblib
import numpy as np
import pytest

from src import config
from src.features import build_feature_frame
from src.model import ModelLoadError, file_sha256, load_model, save_model
from src.parsing import parse_email


def test_save_load_roundtrip(tiny_model, synthetic_split, tmp_path):
    _, _, X_val, _ = synthetic_split
    path = tmp_path / "model.joblib"
    info = save_model(tiny_model, path)
    assert info["sha256"] == file_sha256(path) and info["size_bytes"] > 0
    loaded = load_model(path, expected_sha256=info["sha256"])
    np.testing.assert_allclose(loaded.predict_proba(X_val), tiny_model.predict_proba(X_val))
    assert loaded.threshold == tiny_model.threshold
    assert loaded.feature_names is not None
    assert isinstance(loaded.metadata["load_warnings"], list)


def test_missing_artifact(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        load_model(tmp_path / "nope.joblib")


def test_checksum_mismatch(tiny_model, tmp_path):
    path = tmp_path / "model.joblib"
    save_model(tiny_model, path)
    with pytest.raises(ModelLoadError, match="checksum mismatch"):
        load_model(path, expected_sha256="0" * 64)


def test_corrupt_artifact(tmp_path):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"this is not a joblib file")
    with pytest.raises(ModelLoadError, match="Could not load"):
        load_model(path)


def test_wrong_object_type(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump({"not": "a model"}, path)
    with pytest.raises(ModelLoadError, match="does not contain a SpamModel"):
        load_model(path)


def test_schema_version_mismatch(tiny_model, tmp_path):
    import copy

    clone = copy.copy(tiny_model)
    clone.schema_version = 999
    path = tmp_path / "model.joblib"
    joblib.dump(clone, path)
    with pytest.raises(ModelLoadError, match="schema version"):
        load_model(path)


def test_feature_set_mismatch(tiny_model, tmp_path):
    import copy

    clone = copy.copy(tiny_model)
    clone.structured_features = ("something_else",)
    path = tmp_path / "model.joblib"
    joblib.dump(clone, path)
    with pytest.raises(ModelLoadError, match="structured feature set"):
        load_model(path)


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="trained artifact not present; run training first")
def test_shipped_artifact_matches_model_card_and_classifies_samples(sample_bytes):
    card = json.loads(config.MODEL_CARD_PATH.read_text(encoding="utf-8"))
    model = load_model(config.MODEL_PATH, expected_sha256=card["artifact"]["sha256"])
    assert model.version == card["version"]
    assert model.threshold == pytest.approx(card["decision_threshold"])
    names = ["ham_meeting_notes.eml", "ham_mailing_list_reply.eml", "spam_prize_html.eml", "spam_account_phishing.eml"]
    frame = build_feature_frame([parse_email(sample_bytes(n)) for n in names])
    prob = model.predict_proba(frame)
    assert max(prob[:2]) < min(prob[2:]), dict(zip(names, prob.round(3)))
