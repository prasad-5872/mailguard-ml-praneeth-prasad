import math

import pytest

from src.parsing import parse_email
from src.validation import (
    ValidationError,
    validate_parsed_email,
    validate_raw_email,
    validate_threshold,
    validate_upload_filename,
)


@pytest.mark.parametrize("raw", [None, "", "   \n\t", b"", b"  \n"])
def test_empty_input_rejected(raw):
    with pytest.raises(ValidationError):
        validate_raw_email(raw)


@pytest.mark.parametrize("raw", [123, 4.5, ["a"], {"a": 1}])
def test_wrong_type_rejected(raw):
    with pytest.raises(ValidationError):
        validate_raw_email(raw)


def test_too_large_rejected():
    with pytest.raises(ValidationError, match="too large"):
        validate_raw_email("a" * 101, max_bytes=100)
    # multi-byte characters are measured in bytes, not characters
    with pytest.raises(ValidationError):
        validate_raw_email("é" * 60, max_bytes=100)


def test_binary_input_rejected():
    with pytest.raises(ValidationError, match="binary"):
        validate_raw_email(b"PK\x03\x04\x00\x00binary")


def test_valid_input_returned_unchanged():
    assert validate_raw_email("Subject: hi\n\nbody") == "Subject: hi\n\nbody"
    assert validate_raw_email(b"hello") == b"hello"


@pytest.mark.parametrize("value,expected", [(0, 0.0), (1, 1.0), (0.37, 0.37), ("0.5", 0.5)])
def test_valid_thresholds(value, expected):
    assert validate_threshold(value) == expected


@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan, math.inf, "abc", None, True, [0.5]])
def test_invalid_thresholds(value):
    with pytest.raises(ValidationError):
        validate_threshold(value)


@pytest.mark.parametrize("name", ["mail.eml", "MAIL.EML", "message.txt"])
def test_allowed_upload_names(name):
    assert validate_upload_filename(name) == name


@pytest.mark.parametrize("name", [None, "", "virus.exe", "mail.eml.zip", "noext"])
def test_rejected_upload_names(name):
    with pytest.raises(ValidationError):
        validate_upload_filename(name)


def test_parsed_email_without_content_rejected():
    with pytest.raises(ValidationError, match="No readable"):
        validate_parsed_email(parse_email("From: a@example.com\nSubject:\n\n   \n"))


def test_parsed_email_with_subject_only_accepted():
    parsed = parse_email("From: a@example.com\nSubject: hello\n\n")
    assert validate_parsed_email(parsed) is parsed
