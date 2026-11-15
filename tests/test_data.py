import numpy as np
import pandas as pd
import pytest

from src.data import (
    deduplicate,
    exact_dedup_key,
    group_stratified_split,
    iter_corpus_files,
    near_dup_group_key,
    validate_labels,
)
from src.parsing import parse_email


def _row(text: str, label: int, subset: str) -> dict:
    parsed = parse_email(f"From: a@example.com\nSubject: s\n\n{text}")
    return {"label": label, "subset": subset, "dedup_key": exact_dedup_key(parsed)}


def test_exact_dedup_ignores_case_and_whitespace():
    a = parse_email("Subject: Hello\n\nBuy   NOW")
    b = parse_email("Subject: hello\n\nbuy now")
    assert exact_dedup_key(a) == exact_dedup_key(b)


def test_deduplicate_drops_duplicates_and_conflicts():
    df = pd.DataFrame(
        [
            _row("same text here", 0, "easy_ham"),
            _row("same   TEXT here", 0, "easy_ham"),
            _row("conflict text", 0, "easy_ham"),
            _row("conflict text", 1, "spam"),
            _row("unique spam", 1, "spam"),
        ]
    )
    out, report = deduplicate(df)
    assert len(out) == 2
    assert report["exact_duplicates_dropped"] == 1
    assert report["conflicting_label_keys"] == 1 and report["conflicting_label_rows_dropped"] == 2
    assert out["dedup_key"].is_unique


def test_near_dup_key_groups_campaign_variants():
    body = "Dear {name}, you won ${amt}! Claim at http://x{n}.example.com/claim before it expires. " * 2
    a = parse_email(body.format(name="Ann", amt=500, n=1))
    b = parse_email(body.format(name="Ann", amt=9000, n=2))
    c = parse_email("A completely different message about the quarterly planning meeting agenda and notes.")
    assert near_dup_group_key(a) == near_dup_group_key(b)
    assert near_dup_group_key(a) != near_dup_group_key(c)


def test_validate_labels():
    ok = pd.DataFrame({"label": [0, 1], "subset": ["easy_ham", "spam"]})
    validate_labels(ok)
    with pytest.raises(ValueError, match="does not match"):
        validate_labels(pd.DataFrame({"label": [1, 1], "subset": ["easy_ham", "spam"]}))
    with pytest.raises(ValueError, match="both"):
        validate_labels(pd.DataFrame({"label": [0], "subset": ["easy_ham"]}))
    with pytest.raises(ValueError, match="empty"):
        validate_labels(pd.DataFrame({"label": [], "subset": []}))
    with pytest.raises(ValueError, match="0 \\(ham\\) or 1"):
        validate_labels(pd.DataFrame({"label": [0, 2], "subset": ["easy_ham", "spam"]}))


def test_group_split_has_no_group_leakage_and_is_stratified():
    rng = np.random.default_rng(0)
    n = 700
    labels = (rng.random(n) < 0.3).astype(int)
    groups = np.array([f"g{i // 3}" for i in range(n)])  # groups of 3 identical-label items
    labels = np.array([labels[(i // 3) * 3] for i in range(n)])
    tr, va, te = group_stratified_split(labels, groups, seed=1)
    assert len(tr) + len(va) + len(te) == n
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not set(groups[a]) & set(groups[b])
    for split in (tr, va, te):
        assert abs(labels[split].mean() - labels.mean()) < 0.06
    assert 0.10 < len(te) / n < 0.20 and 0.10 < len(va) / n < 0.20


def test_iter_corpus_files_skips_cmds_and_unexpected_dirs(tmp_path):
    root = tmp_path / "20030228_spam" / "spam"
    root.mkdir(parents=True)
    (root / "0001.abc").write_text("From: a@example.com\n\nx")
    (root / "cmds").write_text("mv a b")
    stray = tmp_path / "20030228_spam" / "other"
    stray.mkdir()
    (stray / "0002.abc").write_text("x")
    files = list(iter_corpus_files(tmp_path))
    assert [(f.path.name, f.subset, f.label) for f in files] == [("0001.abc", "spam", 1)]
