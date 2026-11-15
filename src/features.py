"""Feature engineering: text normalization for TF-IDF and structured (hand-crafted) email features.

Leakage note: raw header text is intentionally NOT fed to the text vectorizers.
In the SpamAssassin corpus, ham and spam were collected through different
channels (mailing lists vs. spam traps), so header tokens such as list names,
relay hostnames or ``spamassassin.taint.org`` would let a model learn the
collection source instead of spam characteristics. Only subject + body text and
a small set of generalizable header *properties* (e.g. Reply-To/From domain
mismatch) are used.
"""

from __future__ import annotations

import ipaddress
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import MAX_MODEL_TEXT_CHARS
from src.parsing import ParsedEmail, domain_of_address, url_host

TEXT_COLUMN = "text"

SHORTENER_DOMAINS = frozenset(
    {
        "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
        "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "t.ly", "s.id", "lnkd.in", "v.gd",
    }
)

SPAM_PHRASES = (
    "click here", "act now", "limited time", "order now", "risk free", "risk-free", "no obligation",
    "100% free", "free gift", "winner", "congratulations", "you have been selected", "earn money",
    "make money", "work from home", "lowest price", "best price", "special promotion", "dear friend",
    "this is not spam", "remove me", "to be removed", "unsubscribe", "credit card", "no cost",
    "money back", "guarantee", "cash bonus", "double your", "million dollars", "urgent",
    "verify your account", "account suspended", "password", "wire transfer", "viagra", "mortgage",
)

_URL_TOKEN_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_EMAIL_TOKEN_RE = re.compile(r"(?i)\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_DIGIT_RE = re.compile(r"\d")
_WS_RE = re.compile(r"\s+")
_CURRENCY_RE = re.compile(r"[$£€¥]")
_EMBEDDED_EMAIL_RE = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")

# Order matters: it defines the column order of the structured feature block.
STRUCTURED_FEATURES: tuple[str, ...] = (
    "subject_length",
    "subject_upper_ratio",
    "subject_exclamations",
    "subject_is_reply",
    "body_length_log",
    "body_upper_ratio",
    "body_exclamations_per_1k",
    "body_digit_ratio",
    "currency_symbols",
    "spam_phrase_count",
    "url_count",
    "unique_url_domains",
    "ip_url_count",
    "shortener_url_count",
    "link_text_mismatch",
    "has_html",
    "html_only",
    "html_image_count",
    "html_form_count",
    "html_script_count",
    "attachment_count",
    "reply_to_differs",
    "from_name_domain_mismatch",
    "missing_message_id",
    "missing_date",
)

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "subject_length": "Number of characters in the subject",
    "subject_upper_ratio": "Share of letters in the subject that are upper-case",
    "subject_exclamations": "Number of '!' in the subject",
    "subject_is_reply": "Subject starts with Re:/Fw:/Fwd:",
    "body_length_log": "log(1 + body length in characters)",
    "body_upper_ratio": "Share of letters in the body that are upper-case",
    "body_exclamations_per_1k": "'!' per 1,000 body characters",
    "body_digit_ratio": "Share of body characters that are digits",
    "currency_symbols": "Count of $, £, €, ¥ symbols",
    "spam_phrase_count": "Occurrences of common spam phrases",
    "url_count": "Number of distinct URLs",
    "unique_url_domains": "Number of distinct URL hosts",
    "ip_url_count": "URLs whose host is a raw IP address",
    "shortener_url_count": "URLs using a known link shortener",
    "link_text_mismatch": "HTML links whose visible URL text points to a different host than the href",
    "has_html": "Message has an HTML body",
    "html_only": "HTML body with no plain-text alternative",
    "html_image_count": "Number of <img> tags",
    "html_form_count": "Number of <form> tags",
    "html_script_count": "Number of <script> tags",
    "attachment_count": "Number of attachments / non-text parts",
    "reply_to_differs": "Reply-To domain differs from From domain",
    "from_name_domain_mismatch": "From display name contains an address with a different domain",
    "missing_message_id": "Headers present but no Message-ID",
    "missing_date": "Headers present but no Date",
}


def normalize_text(text: str) -> str:
    """Normalize text for TF-IDF: lower-case, mask URLs/emails/digits, collapse whitespace.

    Masking replaces specific URLs, addresses and numbers by generic tokens so the
    model learns *patterns* (e.g. "has a URL", "price-like number") rather than
    memorizing individual domains or phone numbers from the training corpus.
    """
    text = (text or "")[:MAX_MODEL_TEXT_CHARS]
    text = _URL_TOKEN_RE.sub(" urltoken ", text)
    text = _EMAIL_TOKEN_RE.sub(" emailtoken ", text)
    text = _DIGIT_RE.sub("0", text)
    text = _WS_RE.sub(" ", text).strip().lower()
    return text


def _upper_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _is_ip_host(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _link_text_mismatches(link_pairs: Iterable[tuple[str, str]]) -> int:
    count = 0
    for anchor, href in link_pairs:
        anchor_urls = _URL_TOKEN_RE.findall(anchor) or (
            [anchor] if re.match(r"(?i)^[a-z0-9.-]+\.[a-z]{2,}(/\S*)?$", anchor.strip()) else []
        )
        if not anchor_urls:
            continue
        anchor_host = url_host(anchor_urls[0] if "://" in anchor_urls[0] else "http://" + anchor_urls[0])
        href_host = url_host(href)
        if anchor_host and href_host and anchor_host.removeprefix("www.") != href_host.removeprefix("www."):
            count += 1
    return count


def extract_structured_features(parsed: ParsedEmail) -> dict[str, float]:
    """Compute the structured feature vector for one parsed email (keys = STRUCTURED_FEATURES)."""
    subject = parsed.subject or ""
    body = parsed.body or ""
    full_text = f"{subject}\n{body}"
    lower_text = full_text.lower()
    body_len = len(body)

    hosts = [url_host(u) for u in parsed.urls]
    hosts = [h for h in hosts if h]

    sender_domain = parsed.sender_domain
    reply_domain = parsed.reply_to_domain
    reply_to_differs = bool(parsed.has_headers and reply_domain and sender_domain and reply_domain != sender_domain)

    name_mismatch = False
    if parsed.has_headers and parsed.sender_name and sender_domain:
        embedded = _EMBEDDED_EMAIL_RE.search(parsed.sender_name)
        if embedded and embedded.group(1).lower() != sender_domain:
            name_mismatch = True

    features = {
        "subject_length": float(len(subject)),
        "subject_upper_ratio": _upper_ratio(subject),
        "subject_exclamations": float(subject.count("!")),
        "subject_is_reply": float(bool(re.match(r"(?i)^\s*(re|fw|fwd)\s*:", subject))),
        "body_length_log": math.log1p(body_len),
        "body_upper_ratio": _upper_ratio(body),
        "body_exclamations_per_1k": (body.count("!") / body_len * 1000.0) if body_len else 0.0,
        "body_digit_ratio": (sum(c.isdigit() for c in body) / body_len) if body_len else 0.0,
        "currency_symbols": float(len(_CURRENCY_RE.findall(full_text))),
        "spam_phrase_count": float(sum(lower_text.count(p) for p in SPAM_PHRASES)),
        "url_count": float(len(parsed.urls)),
        "unique_url_domains": float(len(set(hosts))),
        "ip_url_count": float(sum(1 for h in hosts if _is_ip_host(h))),
        "shortener_url_count": float(sum(1 for h in hosts if h.removeprefix("www.") in SHORTENER_DOMAINS)),
        "link_text_mismatch": float(_link_text_mismatches(parsed.link_pairs)),
        "has_html": float(parsed.has_html),
        "html_only": float(parsed.html_only),
        "html_image_count": float(parsed.html_stats.images),
        "html_form_count": float(parsed.html_stats.forms),
        "html_script_count": float(parsed.html_stats.scripts),
        "attachment_count": float(len(parsed.attachments)),
        "reply_to_differs": float(reply_to_differs),
        "from_name_domain_mismatch": float(name_mismatch),
        "missing_message_id": float(parsed.has_headers and not parsed.message_id.strip()),
        "missing_date": float(parsed.has_headers and not parsed.date.strip()),
    }
    # Guard against NaN/inf from pathological inputs.
    return {k: (v if math.isfinite(v) else 0.0) for k, v in features.items()}


def build_feature_frame(parsed_emails: Iterable[ParsedEmail]) -> pd.DataFrame:
    """Build the model input frame: one normalized text column + structured feature columns."""
    rows = []
    for parsed in parsed_emails:
        row: dict[str, object] = {TEXT_COLUMN: normalize_text(parsed.model_text)}
        row.update(extract_structured_features(parsed))
        rows.append(row)
    frame = pd.DataFrame(rows, columns=[TEXT_COLUMN, *STRUCTURED_FEATURES])
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].astype(object)
    frame[list(STRUCTURED_FEATURES)] = frame[list(STRUCTURED_FEATURES)].astype(np.float64)
    return frame


def sender_domain_of(parsed: ParsedEmail) -> str:
    """Convenience accessor used by error analysis/UI."""
    return domain_of_address(parsed.sender_address)
