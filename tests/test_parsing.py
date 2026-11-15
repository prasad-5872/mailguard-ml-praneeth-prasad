import base64

import pytest

from src.parsing import extract_urls, html_to_text, looks_like_raw_email, parse_email, url_host


def test_parse_plain_eml(sample_bytes):
    parsed = parse_email(sample_bytes("ham_meeting_notes.eml"))
    assert parsed.has_headers
    assert parsed.subject == "Notes from Monday's pipeline review"
    assert parsed.sender_name == "Priya Raman"
    assert parsed.sender_address == "priya.raman@example.org"
    assert parsed.sender_domain == "example.org"
    assert parsed.message_id.startswith("<20240910091355")
    assert parsed.received_count == 1
    assert "nightly ingestion job" in parsed.body
    assert not parsed.has_html
    assert parsed.header("content-type").startswith("text/plain")


def test_parse_multipart_extracts_plain_html_urls_and_forms(sample_bytes):
    parsed = parse_email(sample_bytes("spam_account_phishing.eml"))
    assert parsed.reply_to_address == "support-desk@mailbox-example.ru"
    assert parsed.reply_to_domain == "mailbox-example.ru"
    assert "suspended" in parsed.body_text
    assert parsed.has_html and not parsed.html_only
    assert parsed.html_stats.forms == 1
    assert "http://bit.ly/3xVerify" in parsed.urls
    assert "http://203.0.113.45/secure/login" in parsed.urls
    assert ("https://www.example-bank.com/verify", "http://203.0.113.45/secure/login") in parsed.link_pairs
    assert set(parsed.content_types) == {"text/plain", "text/html"}


def test_parse_html_only(sample_bytes):
    parsed = parse_email(sample_bytes("spam_prize_html.eml"))
    assert parsed.html_only
    assert parsed.html_stats.images == 1
    assert parsed.html_stats.links == 2
    assert "CONGRATULATIONS DEAR FRIEND" in parsed.body
    assert "<p>" not in parsed.body


def test_html_to_text_skips_script_and_style():
    text, stats, links = html_to_text(
        "<html><head><style>.x{color:red}</style></head><body><script>alert(1)</script>"
        "<p>Hello <a href='http://a.example.com'>there</a></p></body></html>"
    )
    assert "alert" not in text and "color" not in text
    assert "Hello there" in text
    assert stats.scripts == 1 and stats.links == 1
    assert links == [("there", "http://a.example.com")]


def test_rfc2047_encoded_subject_and_sender():
    raw = (
        "From: =?utf-8?B?" + base64.b64encode("Zoë Müller".encode()).decode() + "?= <zoe@example.de>\n"
        "Subject: =?utf-8?Q?Gr=C3=BC=C3=9Fe_aus_Berlin?=\n\nHallo!"
    )
    parsed = parse_email(raw)
    assert parsed.subject == "Grüße aus Berlin"
    assert parsed.sender_name == "Zoë Müller"


def test_base64_and_quoted_printable_bodies():
    b64 = base64.b64encode("Café meeting at noon".encode("utf-8")).decode()
    raw = (
        "From: a@example.com\nSubject: t\nMIME-Version: 1.0\n"
        'Content-Type: multipart/mixed; boundary="B"\n\n'
        "--B\nContent-Type: text/plain; charset=utf-8\nContent-Transfer-Encoding: base64\n\n" + b64 + "\n"
        "--B\nContent-Type: text/plain; charset=utf-8\nContent-Transfer-Encoding: quoted-printable\n\n"
        "Price: 10=E2=82=AC only\n--B--\n"
    )
    parsed = parse_email(raw)
    assert "Café meeting at noon" in parsed.body
    assert "10€ only" in parsed.body


def test_unknown_charset_does_not_raise():
    raw = b"From: a@example.com\nSubject: x\nContent-Type: text/plain; charset=x-totally-unknown\n\nhello \xe9\xe8"
    parsed = parse_email(raw)
    assert parsed.body.startswith("hello")


def test_attachment_detected():
    raw = (
        "From: a@example.com\nSubject: invoice\nMIME-Version: 1.0\n"
        'Content-Type: multipart/mixed; boundary="B"\n\n'
        "--B\nContent-Type: text/plain\n\nsee attached\n"
        '--B\nContent-Type: application/pdf\nContent-Disposition: attachment; filename="inv.pdf"\n'
        "Content-Transfer-Encoding: base64\n\nJVBERi0x\n--B--\n"
    )
    parsed = parse_email(raw)
    assert parsed.attachments == ["inv.pdf"]
    assert parsed.body == "see attached"


def test_body_only_paste_is_not_treated_as_headers():
    parsed = parse_email("Note: the deadline moved.\nPlease update the plan.")
    assert not parsed.has_headers
    assert parsed.subject == ""
    assert parsed.body.startswith("Note: the deadline moved.")


def test_pasted_html_body_without_headers():
    parsed = parse_email("<p>Win <a href='http://x.example.com'>now</a></p>")
    assert not parsed.has_headers
    assert parsed.html_only
    assert parsed.urls == ["http://x.example.com"]


def test_mbox_from_line_is_accepted():
    raw = "From someone@example.com  Thu Aug 22 13:17:22 2002\nFrom: someone@example.com\nSubject: hi\n\nbody text"
    assert looks_like_raw_email(raw)
    parsed = parse_email(raw)
    assert parsed.subject == "hi" and parsed.body == "body text"


def test_bytes_and_str_inputs_agree(sample_bytes):
    raw = sample_bytes("ham_mailing_list_reply.eml")
    a, b = parse_email(raw), parse_email(raw.decode("utf-8"))
    assert (a.subject, a.body, a.sender_address) == (b.subject, b.body, b.sender_address)


def test_extract_urls_strips_trailing_punctuation():
    assert extract_urls("see http://a.example.com/x. and (www.b.example.org), ok") == [
        "http://a.example.com/x",
        "www.b.example.org",
    ]


def test_url_host():
    assert url_host("HTTP://Sub.Example.COM:8080/x") == "sub.example.com"
    assert url_host("www.example.org/path") == "www.example.org"
    assert url_host("http://[bad") == ""


def test_parse_email_rejects_wrong_type():
    with pytest.raises(TypeError):
        parse_email(123)  # type: ignore[arg-type]
