"""SpamAssassin public corpus: download, extraction, label validation, deduplication and splitting.

Dataset: Apache SpamAssassin Public Mail Corpus
  https://spamassassin.apache.org/old/publiccorpus/
Messages keep their full original headers. Per the corpus readme, copyright of the
message text remains with the original senders, so the raw corpus is NOT
redistributed in this repository; it is downloaded at training time.
"""

from __future__ import annotations

import hashlib
import re
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.features import STRUCTURED_FEATURES, build_feature_frame, normalize_text
from src.parsing import ParsedEmail, parse_email

CORPUS_BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"

# archive -> (subset directory inside archive, label). 1 = spam, 0 = ham.
# The 2003/2005 snapshots supersede the 2002 ones (which are subsets). 20050311_spam_2
# is the latest spam_2 snapshot, with one known mislabelled ham removed by the curator.
CORPUS_ARCHIVES: dict[str, tuple[str, int]] = {
    "20030228_easy_ham.tar.bz2": ("easy_ham", 0),
    "20030228_easy_ham_2.tar.bz2": ("easy_ham_2", 0),
    "20030228_hard_ham.tar.bz2": ("hard_ham", 0),
    "20030228_spam.tar.bz2": ("spam", 1),
    "20050311_spam_2.tar.bz2": ("spam_2", 1),
}
SUBSET_LABELS = {subset: label for subset, label in CORPUS_ARCHIVES.values()}
NON_MESSAGE_FILES = {"cmds"}  # helper scripts shipped inside each corpus folder

_HTTP_HEADERS = {"User-Agent": "mailguard-ml-training/1.0 (educational project)"}


def download_corpus(raw_dir: Path, force: bool = False, log=print) -> dict[str, str]:
    """Download the corpus archives (skips existing files). Returns {archive: sha256}."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    for archive in CORPUS_ARCHIVES:
        target = raw_dir / archive
        if force or not target.exists() or target.stat().st_size == 0:
            url = CORPUS_BASE_URL + archive
            log(f"Downloading {url}")
            request = urllib.request.Request(url, headers=_HTTP_HEADERS)
            tmp = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as fh:
                while chunk := response.read(1 << 16):
                    fh.write(chunk)
            tmp.replace(target)
        checksums[archive] = hashlib.sha256(target.read_bytes()).hexdigest()
    return checksums


def extract_corpus(raw_dir: Path, extract_dir: Path, log=print) -> None:
    """Extract each archive into ``extract_dir/<archive stem>/`` using tarfile's safe 'data' filter."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    for archive in CORPUS_ARCHIVES:
        dest = extract_dir / archive.replace(".tar.bz2", "")
        if dest.exists() and any(dest.rglob("*")):
            continue
        log(f"Extracting {archive}")
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(raw_dir / archive, "r:bz2") as tar:
            tar.extractall(dest, filter="data")


@dataclass(frozen=True)
class CorpusFile:
    path: Path
    archive: str
    subset: str
    label: int


def iter_corpus_files(extract_dir: Path) -> Iterator[CorpusFile]:
    for archive, (subset, label) in CORPUS_ARCHIVES.items():
        root = extract_dir / archive.replace(".tar.bz2", "")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name in NON_MESSAGE_FILES:
                continue
            # The archive directory names the subset; refuse files from unexpected folders.
            if path.parent.name != subset:
                continue
            yield CorpusFile(path, archive, subset, label)


_MASK_RE = re.compile(r"[\W_\d]+")  # punctuation, whitespace and digits


def exact_dedup_key(parsed: ParsedEmail) -> str:
    """Key for exact duplicates: normalized subject+body (whitespace/case-insensitive)."""
    return hashlib.sha256(normalize_text(parsed.model_text).encode("utf-8")).hexdigest()


def near_dup_group_key(parsed: ParsedEmail) -> str:
    """Coarse key that groups near-duplicates (e.g. the same spam campaign with different
    numbers, URLs or recipient names) so they never straddle train/test splits.

    Uses the normalized text with digits/URLs/emails masked and all punctuation removed,
    truncated to the first 300 characters.
    """
    text = _MASK_RE.sub("", normalize_text(parsed.body or parsed.subject))[:300]
    if len(text) < 40:  # too short to be meaningful; fall back to the exact key
        return "x" + exact_dedup_key(parsed)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_corpus(extract_dir: Path, log=print) -> tuple[pd.DataFrame, dict[str, object]]:
    """Parse every corpus message, validate labels and drop unusable records."""
    rows = []
    report: dict[str, object] = {"files_seen": 0, "parse_errors": 0, "empty_content": 0, "label_errors": 0}
    for cf in iter_corpus_files(extract_dir):
        report["files_seen"] = int(report["files_seen"]) + 1
        if SUBSET_LABELS.get(cf.subset) != cf.label or cf.label not in (0, 1):
            report["label_errors"] = int(report["label_errors"]) + 1
            continue
        try:
            parsed = parse_email(cf.path.read_bytes())
        except Exception:
            report["parse_errors"] = int(report["parse_errors"]) + 1
            continue
        if not parsed.model_text.strip():
            report["empty_content"] = int(report["empty_content"]) + 1
            continue
        rows.append(
            {
                "file": f"{cf.subset}/{cf.path.name}",
                "subset": cf.subset,
                "label": cf.label,
                "parsed": parsed,
                "dedup_key": exact_dedup_key(parsed),
                "group_key": near_dup_group_key(parsed),
                "subject": parsed.subject[:200],
                "sender_domain": parsed.sender_domain,
                "snippet": " ".join(parsed.body.split())[:200],
            }
        )
    df = pd.DataFrame(rows)
    report["parsed_ok"] = len(df)
    log(f"Parsed {len(df):,} messages ({report})")
    return df, report


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove exact duplicates; drop every copy of a message whose duplicates carry conflicting labels."""
    labels_per_key = df.groupby("dedup_key")["label"].nunique()
    conflicting = set(labels_per_key[labels_per_key > 1].index)
    before = len(df)
    conflict_rows = int(df["dedup_key"].isin(conflicting).sum())
    clean = df[~df["dedup_key"].isin(conflicting)]
    deduped = clean.drop_duplicates(subset="dedup_key", keep="first").reset_index(drop=True)
    return deduped, {
        "rows_before": before,
        "conflicting_label_keys": len(conflicting),
        "conflicting_label_rows_dropped": conflict_rows,
        "exact_duplicates_dropped": len(clean) - len(deduped),
        "rows_after": len(deduped),
    }


def validate_labels(df: pd.DataFrame) -> None:
    """Fail loudly if labels are not binary, disagree with their subset, or a class is missing."""
    if df.empty:
        raise ValueError("dataset is empty")
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError("labels must be 0 (ham) or 1 (spam)")
    expected = df["subset"].map(SUBSET_LABELS)
    if expected.isna().any() or (expected != df["label"]).any():
        raise ValueError("label does not match its corpus subset for some rows")
    if df["label"].nunique() < 2:
        raise ValueError("dataset must contain both ham and spam")


def group_stratified_split(
    labels: np.ndarray, groups: np.ndarray, seed: int, test_folds: int = 7, val_folds: int = 6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train/val/test (~71/14/14) keeping each near-duplicate group in one split."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    idx = np.arange(len(labels))
    outer = StratifiedGroupKFold(n_splits=test_folds, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(outer.split(idx, labels, groups))
    inner = StratifiedGroupKFold(n_splits=val_folds, shuffle=True, random_state=seed + 1)
    tr_rel, va_rel = next(inner.split(trainval_idx, labels[trainval_idx], groups[trainval_idx]))
    train_idx, val_idx = trainval_idx[tr_rel], trainval_idx[va_rel]

    for a, b in ((train_idx, val_idx), (train_idx, test_idx), (val_idx, test_idx)):
        if set(groups[a]) & set(groups[b]):
            raise AssertionError("group leakage between splits")
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def frame_for(df: pd.DataFrame) -> pd.DataFrame:
    """Model input frame for the rows of a corpus DataFrame."""
    frame = build_feature_frame(df["parsed"].tolist())
    frame.index = df.index
    return frame


def describe_split(df: pd.DataFrame) -> dict[str, object]:
    return {
        "n": int(len(df)),
        "n_spam": int(df["label"].sum()),
        "n_ham": int((1 - df["label"]).sum()),
        "spam_ratio": float(df["label"].mean()) if len(df) else 0.0,
        "by_subset": {k: int(v) for k, v in df["subset"].value_counts().sort_index().items()},
    }


__all__ = [
    "CORPUS_ARCHIVES",
    "CORPUS_BASE_URL",
    "STRUCTURED_FEATURES",
    "deduplicate",
    "describe_split",
    "download_corpus",
    "extract_corpus",
    "frame_for",
    "group_stratified_split",
    "load_corpus",
    "validate_labels",
]
