"""Input validation for inference requests."""

from __future__ import annotations

import math

from src.config import ALLOWED_UPLOAD_EXTENSIONS, MAX_EMAIL_BYTES
from src.parsing import ParsedEmail


class ValidationError(ValueError):
    """Raised when user input is rejected. The message is safe to show to the user."""


def validate_raw_email(raw: object, max_bytes: int = MAX_EMAIL_BYTES) -> str | bytes:
    """Validate raw email input (pasted text or uploaded bytes) and return it unchanged."""
    if raw is None:
        raise ValidationError("No email content was provided.")
    if not isinstance(raw, (str, bytes)):
        raise ValidationError(f"Email content must be text or bytes, not {type(raw).__name__}.")

    size = len(raw.encode("utf-8", errors="replace")) if isinstance(raw, str) else len(raw)
    if size == 0 or not raw.strip():
        raise ValidationError("Email content is empty.")
    if size > max_bytes:
        raise ValidationError(
            f"Email is too large ({size:,} bytes). The maximum supported size is {max_bytes:,} bytes."
        )

    sample = raw[:8192]
    nul_count = sample.count(b"\x00") if isinstance(sample, bytes) else sample.count("\x00")
    if nul_count > 0:
        raise ValidationError("Input looks like a binary file, not an email (contains NUL bytes).")
    return raw


def validate_threshold(threshold: object) -> float:
    """Validate a decision threshold; must be a finite number in [0, 1]."""
    if isinstance(threshold, bool):
        raise ValidationError("Threshold must be a number between 0 and 1.")
    try:
        value = float(threshold)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValidationError("Threshold must be a number between 0 and 1.") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValidationError(f"Threshold must be between 0 and 1 (got {threshold!r}).")
    return value


def validate_upload_filename(filename: str | None) -> str:
    """Validate the extension of an uploaded file."""
    if not filename:
        raise ValidationError("Uploaded file has no name.")
    lowered = filename.lower()
    if not lowered.endswith(ALLOWED_UPLOAD_EXTENSIONS):
        allowed = ", ".join(ALLOWED_UPLOAD_EXTENSIONS)
        raise ValidationError(f"Unsupported file type for '{filename}'. Allowed: {allowed}.")
    return filename


def validate_parsed_email(parsed: ParsedEmail) -> ParsedEmail:
    """Reject messages with no analysable content (no subject and no body text)."""
    if not parsed.subject.strip() and not parsed.body.strip():
        raise ValidationError("No readable subject or body text was found in the email.")
    return parsed
