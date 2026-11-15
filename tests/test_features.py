import math

import pytest

from src.config import MAX_MODEL_TEXT_CHARS
from src.features import (
    STRUCTURED_FEATURES,
    TEXT_COLUMN,
    build_feature_frame,
    extract_structured_features,
    normalize_text,
)
from src.parsing import parse_email


def test_normalize_text_masks_urls_emails_digits():
    out = normalize_text("Visit HTTP://Evil.example.com/x?id=9 or mail Bob@Example.com, call 555-1234 NOW")
    assert out == "visit urltoken or mail emailtoken , call 000-0000 now"


def test_normalize_text_truncates():
    assert len(normalize_text("a" * (MAX_MODEL_TEXT_CHARS + 500))) == MAX_MODEL_TEXT_CHARS


def test_structured_features_complete_and_finite(sample_bytes):
    feats = extract_structured_features(parse_email(sample_bytes("ham_meeting_notes.eml")))
    assert tuple(feats) == STRUCTURED_FEATURES
    assert all(isinstance(v, float) and math.isfinite(v) for v in feats.values())


def test_spam_signals_detected(sample_bytes):
    f = extract_structured_features(parse_email(sample_bytes("spam_prize_html.eml")))
    assert f["reply_to_differs"] == 1.0
    assert f["ip_url_count"] >= 1
    assert f["link_text_mismatch"] == 1.0  # visible trusted-bank URL, href to payout domain
    assert f["html_only"] == 1.0 and f["has_html"] == 1.0
    assert f["subject_exclamations"] == 3.0
    assert f["subject_upper_ratio"] == pytest.approx(22 / 45)  # 22 upper-case of 45 letters
    assert f["currency_symbols"] >= 3
    assert f["spam_phrase_count"] >= 5
    assert f["missing_message_id"] == 1.0 and f["missing_date"] == 1.0


def test_phishing_signals_detected(sample_bytes):
    f = extract_structured_features(parse_email(sample_bytes("spam_account_phishing.eml")))
    assert f["shortener_url_count"] == 1.0
    assert f["html_form_count"] == 1.0
    assert f["link_text_mismatch"] == 1.0
    # bit.ly, 203.0.113.45 and the visible (spoofed) anchor URL www.example-bank.com
    assert f["unique_url_domains"] == 3.0


def test_ham_has_no_spam_header_signals(sample_bytes):
    f = extract_structured_features(parse_email(sample_bytes("ham_mailing_list_reply.eml")))
    assert f["subject_is_reply"] == 1.0
    for key in ("reply_to_differs", "ip_url_count", "link_text_mismatch", "missing_message_id", "missing_date"):
        assert f[key] == 0.0, key


def test_body_only_input_disables_header_features():
    f = extract_structured_features(parse_email("just a short note without headers"))
    assert f["missing_message_id"] == 0.0 and f["missing_date"] == 0.0 and f["reply_to_differs"] == 0.0


def test_from_display_name_domain_mismatch():
    raw = 'From: "service@bank.example.com" <x@attacker.example.net>\nSubject: hi\n\nbody'
    assert extract_structured_features(parse_email(raw))["from_name_domain_mismatch"] == 1.0


def test_build_feature_frame_shape_and_dtypes(sample_bytes):
    parsed = [parse_email(sample_bytes(n)) for n in ("ham_meeting_notes.eml", "spam_prize_html.eml")]
    frame = build_feature_frame(parsed)
    assert list(frame.columns) == [TEXT_COLUMN, *STRUCTURED_FEATURES]
    assert len(frame) == 2
    assert frame[TEXT_COLUMN].iloc[0].startswith("notes from monday")
    assert all(str(frame[c].dtype) == "float64" for c in STRUCTURED_FEATURES)


def test_build_feature_frame_empty():
    frame = build_feature_frame([])
    assert frame.empty and list(frame.columns) == [TEXT_COLUMN, *STRUCTURED_FEATURES]
