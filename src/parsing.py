"""Email parsing for raw RFC 5322 / MIME messages (``.eml`` files or pasted source).

The parser is deliberately defensive: real-world spam is frequently malformed
(unknown charsets, broken MIME boundaries, bogus encoded words), so every
decoding step falls back to a lossy-but-safe alternative instead of raising.
"""

from __future__ import annotations

import email
import re
from dataclasses import dataclass, field
from email import policy
from email.header import decode_header, make_header
from email.message import Message
from email.utils import getaddresses, parseaddr
from html.parser import HTMLParser
from urllib.parse import urlsplit

from src.config import MAX_MIME_PARTS, MAX_URLS

# RFC 5322 field names: printable ASCII except colon and space.
_HEADER_LINE_RE = re.compile(r"^[!-9;-~]{1,76}:[ \t]*\S?")
_MBOX_FROM_RE = re.compile(r"^From \S+")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"'(){}\[\]|\\^`]+")
_TRAILING_PUNCT = ".,;:!?)]}'\""
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")

MAX_HEADER_VALUE_CHARS = 2_000


@dataclass
class HtmlStats:
    """Structural counts collected while converting HTML to text."""

    links: int = 0
    images: int = 0
    forms: int = 0
    scripts: int = 0
    iframes: int = 0


@dataclass
class ParsedEmail:
    """Normalized view of an email message used by feature extraction and the UI."""

    subject: str = ""
    sender: str = ""
    sender_name: str = ""
    sender_address: str = ""
    reply_to: str = ""
    reply_to_address: str = ""
    to: str = ""
    date: str = ""
    message_id: str = ""
    headers: list[tuple[str, str]] = field(default_factory=list)
    received_count: int = 0
    body_text: str = ""
    body_html: str = ""
    html_text: str = ""
    urls: list[str] = field(default_factory=list)
    link_pairs: list[tuple[str, str]] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)
    content_types: list[str] = field(default_factory=list)
    html_stats: HtmlStats = field(default_factory=HtmlStats)
    has_headers: bool = False
    defects: list[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        """Best available body text: plain text if present, otherwise text derived from HTML."""
        if self.body_text.strip():
            return self.body_text
        return self.html_text

    @property
    def has_html(self) -> bool:
        return bool(self.body_html.strip())

    @property
    def html_only(self) -> bool:
        return self.has_html and not self.body_text.strip()

    @property
    def model_text(self) -> str:
        """Text representation fed to the TF-IDF vectorizers (subject + body)."""
        return f"{self.subject}\n{self.body}".strip()

    @property
    def sender_domain(self) -> str:
        return domain_of_address(self.sender_address)

    @property
    def reply_to_domain(self) -> str:
        return domain_of_address(self.reply_to_address)

    def header(self, name: str) -> str:
        """Return the first header value with ``name`` (case-insensitive), or ''."""
        lname = name.lower()
        for key, value in self.headers:
            if key.lower() == lname:
                return value
        return ""


class _HTMLToText(HTMLParser):
    """Minimal HTML-to-text converter that also records links and structural counts."""

    _BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "hr"}
    _SKIP_TAGS = {"script", "style", "head", "title", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.stats = HtmlStats()
        self.link_pairs: list[tuple[str, str]] = []
        self._skip_depth = 0
        self._current_href: str | None = None
        self._current_anchor: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            if tag == "script":
                self.stats.scripts += 1
            return
        if tag == "a":
            self.stats.links += 1
            self._current_href = attr_map.get("href", "").strip()
            self._current_anchor = []
        elif tag == "img":
            self.stats.images += 1
        elif tag == "form":
            self.stats.forms += 1
        elif tag == "iframe":
            self.stats.iframes += 1
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() == "a":
            self.handle_endtag("a")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "a" and self._current_href is not None:
            anchor = " ".join("".join(self._current_anchor).split())
            self.link_pairs.append((anchor, self._current_href))
            self._current_href = None
            self._current_anchor = []
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self.parts.append(data)
        if self._current_href is not None:
            self._current_anchor.append(data)

    def text(self) -> str:
        return clean_whitespace("".join(self.parts))


def clean_whitespace(text: str) -> str:
    """Collapse runs of horizontal whitespace and excessive blank lines."""
    lines = [_WS_RE.sub(" ", line).strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return _MULTI_NL_RE.sub("\n\n", "\n".join(lines)).strip()


def html_to_text(html: str) -> tuple[str, HtmlStats, list[tuple[str, str]]]:
    """Convert HTML to readable text; returns (text, structural stats, (anchor text, href) pairs)."""
    parser = _HTMLToText()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # pragma: no cover - HTMLParser is very lenient; guard anyway
        pass
    return parser.text(), parser.stats, parser.link_pairs


def extract_urls(text: str) -> list[str]:
    """Extract http(s)/www URLs from free text, stripping trailing punctuation."""
    urls = []
    for match in URL_RE.finditer(text or ""):
        url = match.group(0).rstrip(_TRAILING_PUNCT)
        if len(url) > 4:
            urls.append(url)
    return urls


def url_host(url: str) -> str:
    """Return the lower-cased host of a URL ('' if it cannot be parsed)."""
    candidate = url.strip()
    if candidate.lower().startswith("www."):
        candidate = "http://" + candidate
    try:
        host = urlsplit(candidate).hostname or ""
    except ValueError:
        return ""
    return host.lower().strip(".")


def domain_of_address(address: str) -> str:
    """Return the lower-cased domain part of an email address ('' if absent)."""
    if not address or "@" not in address:
        return ""
    return address.rsplit("@", 1)[1].strip().strip(">").lower()


def decode_header_value(value: object) -> str:
    """Decode RFC 2047 encoded words; never raises."""
    if value is None:
        return ""
    raw = str(value)
    try:
        decoded = str(make_header(decode_header(raw)))
    except Exception:
        decoded = raw
    decoded = " ".join(decoded.split())
    return decoded[:MAX_HEADER_VALUE_CHARS]


def _decode_bytes(payload: bytes, charset: str | None) -> str:
    candidates = [c for c in (charset, "utf-8") if c]
    for enc in candidates:
        try:
            return payload.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return payload.decode("latin-1", errors="replace")


def _decode_part(part: Message) -> str:
    """Decode a non-multipart MIME part's payload into text."""
    try:
        payload = part.get_payload(decode=True)
    except Exception:
        payload = None
    if isinstance(payload, bytes):
        try:
            charset = part.get_content_charset()
        except Exception:
            charset = None
        return _decode_bytes(payload, charset)
    raw = part.get_payload()
    return raw if isinstance(raw, str) else ""


_KNOWN_HEADERS = {
    "from", "to", "cc", "subject", "date", "received", "message-id", "return-path", "reply-to",
    "mime-version", "content-type", "delivered-to", "sender", "x-mailer", "content-transfer-encoding",
}


def looks_like_raw_email(text: str) -> bool:
    """Heuristic: does the text start with an RFC 5322 header block (or an mbox 'From ' line)?

    The first non-empty line must be a header-shaped line (or mbox separator) and
    the leading header block must contain at least one well-known header name.
    This stops a pasted body that starts with e.g. ``Note: ...`` from being
    mistaken for headers.
    """
    lines = text.lstrip("﻿").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return False
    if _MBOX_FROM_RE.match(lines[0]):
        return True
    if not _HEADER_LINE_RE.match(lines[0]):
        return False
    for line in lines[:100]:
        if not line.strip():
            break
        if line[0] in " \t":  # folded continuation line
            continue
        if not _HEADER_LINE_RE.match(line):
            return False
        if line.split(":", 1)[0].strip().lower() in _KNOWN_HEADERS:
            return True
    return False


def _unique(items: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
            if len(out) >= limit:
                break
    return out


def _parse_message(msg: Message) -> ParsedEmail:
    parsed = ParsedEmail(has_headers=True)

    headers: list[tuple[str, str]] = []
    for key, value in msg.items():
        headers.append((str(key), decode_header_value(value)))
    parsed.headers = headers
    parsed.received_count = sum(1 for k, _ in headers if k.lower() == "received")

    parsed.subject = decode_header_value(msg.get("Subject"))
    parsed.sender = decode_header_value(msg.get("From"))
    name, addr = parseaddr(parsed.sender)
    parsed.sender_name, parsed.sender_address = name.strip(), addr.strip().lower()
    parsed.reply_to = decode_header_value(msg.get("Reply-To"))
    reply_addrs = getaddresses([parsed.reply_to]) if parsed.reply_to else []
    parsed.reply_to_address = reply_addrs[0][1].strip().lower() if reply_addrs else ""
    parsed.to = decode_header_value(msg.get("To"))
    parsed.date = decode_header_value(msg.get("Date"))
    parsed.message_id = decode_header_value(msg.get("Message-ID"))
    parsed.defects = [type(d).__name__ for d in getattr(msg, "defects", [])]

    plain_parts: list[str] = []
    html_parts: list[str] = []
    for index, part in enumerate(msg.walk()):
        if index >= MAX_MIME_PARTS:
            parsed.defects.append("TooManyMimeParts")
            break
        if part.is_multipart():
            continue
        try:
            ctype = part.get_content_type()
        except Exception:
            ctype = "text/plain"
        parsed.content_types.append(ctype)
        disposition = str(part.get("Content-Disposition", "")).lower()
        try:
            filename = part.get_filename()
        except Exception:
            filename = None
        if filename or disposition.startswith("attachment"):
            parsed.attachments.append(decode_header_value(filename) if filename else "(unnamed)")
            continue
        if ctype == "text/html":
            html_parts.append(_decode_part(part))
        elif ctype.startswith("text/") or ctype == "message/rfc822":
            plain_parts.append(_decode_part(part))
        elif not ctype.startswith("multipart/"):
            parsed.attachments.append(f"(inline {ctype})")

    parsed.body_text = clean_whitespace("\n\n".join(p for p in plain_parts if p))
    parsed.body_html = "\n".join(p for p in html_parts if p)
    if parsed.body_html:
        parsed.html_text, parsed.html_stats, parsed.link_pairs = html_to_text(parsed.body_html)
    _collect_urls(parsed)
    return parsed


def _collect_urls(parsed: ParsedEmail) -> None:
    hrefs = [href for _, href in parsed.link_pairs if href.lower().startswith(("http://", "https://", "www."))]
    text_urls = extract_urls(parsed.subject) + extract_urls(parsed.body_text) + extract_urls(parsed.html_text)
    parsed.urls = _unique(hrefs + text_urls, MAX_URLS)


def parse_email(raw: str | bytes) -> ParsedEmail:
    """Parse raw email source (bytes from an ``.eml`` file or pasted text).

    If the input does not start with something that looks like a header block,
    it is treated as a body-only message (``has_headers=False``) so users can
    paste just the text of an email.
    """
    if isinstance(raw, bytes):
        probe = raw[:4096].decode("utf-8", errors="replace")
        if looks_like_raw_email(probe):
            return _parse_message(email.message_from_bytes(raw, policy=policy.compat32))
        text = _decode_bytes(raw, "utf-8")
    elif isinstance(raw, str):
        if looks_like_raw_email(raw):
            return _parse_message(email.message_from_string(raw.lstrip("﻿"), policy=policy.compat32))
        text = raw
    else:
        raise TypeError(f"raw email must be str or bytes, got {type(raw).__name__}")

    parsed = ParsedEmail(has_headers=False)
    if re.search(r"(?is)<\s*(html|body|div|p|a|table|br)\b", text):
        parsed.body_html = text
        parsed.html_text, parsed.html_stats, parsed.link_pairs = html_to_text(text)
    else:
        parsed.body_text = clean_whitespace(text)
    parsed.content_types = ["text/html" if parsed.body_html else "text/plain"]
    _collect_urls(parsed)
    return parsed
