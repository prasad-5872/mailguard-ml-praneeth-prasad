from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import build_feature_frame  # noqa: E402
from src.model import default_candidates  # noqa: E402
from src.monitoring import MetricsStore  # noqa: E402
from src.parsing import parse_email  # noqa: E402
from src.trainer import fit_candidates  # noqa: E402

SAMPLES_DIR = ROOT / "samples"

HAM_SENTENCES = [
    "the meeting is moved to thursday afternoon",
    "please review the attached design document",
    "the build passed on all platforms after the fix",
    "can you check the database migration script",
    "thanks for the notes from the code review",
    "the patch fixes the memory leak in the parser",
    "we should update the documentation before the release",
    "the linker error only happens on the arm toolchain",
    "here is the agenda for the planning session",
    "i pushed the branch with the refactored tests",
]
SPAM_SENTENCES = [
    "click here to claim your free prize now",
    "limited time offer act now and save",
    "earn money from home guaranteed income",
    "you have been selected as our lucky winner",
    "lowest price on watches order now",
    "risk free investment double your money fast",
    "verify your account password immediately",
    "congratulations a cash bonus is waiting for you",
    "no obligation free gift for every customer",
    "special promotion unsubscribe at any time",
]


def make_raw_email(label: int, rng: np.random.Generator, i: int) -> str:
    pool = SPAM_SENTENCES if label else HAM_SENTENCES
    body_lines = [pool[j] for j in rng.choice(len(pool), size=4, replace=False)]
    if label:
        subject = f"{pool[int(rng.integers(len(pool)))].upper()}!!!"
        headers = (
            f"From: Promo Team <promo{i}@deals-example.biz>\n"
            f"Reply-To: claims{i}@payout-example.info\n"
            f"To: user@example.com\nSubject: {subject}\n"
        )
        if i % 2:
            links = f'<a href="http://198.51.100.{i % 250}/x">http://www.shop-example.com</a>'
            body = "<html><body><p>" + "</p><p>".join(body_lines) + f"</p>{links}</body></html>"
            return headers + "MIME-Version: 1.0\nContent-Type: text/html\n\n" + body
        body = "\n".join(body_lines) + f"\nhttp://bit.ly/x{i} ${i * 10}"
    else:
        subject = f"Re: {pool[int(rng.integers(len(pool)))]}"
        headers = (
            f"From: Dev {i} <dev{i}@example.org>\nTo: team@example.org\nSubject: {subject}\n"
            f"Date: Tue, 10 Sep 2024 09:13:55 +0000\nMessage-ID: <{i}@example.org>\n"
        )
        body = "\n".join(body_lines) + "\n\nthanks,\ndev"
    return headers + "Content-Type: text/plain\n\n" + body


def make_synthetic_corpus(n_per_class: int = 60, seed: int = 0) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    raws, labels = [], []
    for i in range(n_per_class):
        for label in (0, 1):
            raws.append(make_raw_email(label, rng, i))
            labels.append(label)
    return raws, np.array(labels)


@pytest.fixture(scope="session")
def synthetic_split():
    raws, labels = make_synthetic_corpus(60)
    frame = build_feature_frame(parse_email(r) for r in raws)
    idx = np.arange(len(raws))
    train, val = idx[idx % 3 != 0], idx[idx % 3 == 0]
    return (
        frame.iloc[train].reset_index(drop=True),
        labels[train],
        frame.iloc[val].reset_index(drop=True),
        labels[val],
    )


@pytest.fixture(scope="session")
def fitted_results(synthetic_split):
    X_tr, y_tr, X_va, y_va = synthetic_split
    return fit_candidates(default_candidates(small=True), X_tr, y_tr, X_va, y_va, seed=0, log=lambda *_: None)


@pytest.fixture(scope="session")
def models_by_name(fitted_results):
    return {r.spec.name: r.model for r in fitted_results}


@pytest.fixture(scope="session")
def tiny_model(models_by_name):
    return models_by_name["logistic_regression"]


@pytest.fixture
def metrics_store():
    return MetricsStore(window_size=50)


@pytest.fixture
def test_logger():
    logger = logging.getLogger("mailguard-test")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


@pytest.fixture
def sample_bytes():
    def _load(name: str) -> bytes:
        return (SAMPLES_DIR / name).read_bytes()

    return _load
