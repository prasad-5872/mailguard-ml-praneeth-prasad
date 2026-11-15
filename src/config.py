"""Central configuration: paths, limits and constants shared by training and the app."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = Path(os.environ.get("MAILGUARD_MODELS_DIR", ROOT_DIR / "models"))
MODEL_FILENAME = "mailguard_model.joblib"
MODEL_CARD_FILENAME = "model_card.json"
EVALUATION_FILENAME = "evaluation.json"

MODEL_PATH = MODELS_DIR / MODEL_FILENAME
MODEL_CARD_PATH = MODELS_DIR / MODEL_CARD_FILENAME
EVALUATION_PATH = MODELS_DIR / EVALUATION_FILENAME

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTRACTED_DATA_DIR = DATA_DIR / "extracted"

SAMPLES_DIR = ROOT_DIR / "samples"

MODEL_VERSION = "1.0.0"
RANDOM_SEED = 42

# Input limits (defensive bounds for a public demo).
MAX_EMAIL_BYTES = 1_000_000  # 1 MB of raw email source
MAX_MODEL_TEXT_CHARS = 20_000  # text passed to the vectorizers is truncated to this length
MAX_MIME_PARTS = 200
MAX_URLS = 500
ALLOWED_UPLOAD_EXTENSIONS = (".eml", ".txt")

# Monitoring bounds (in-memory only; reset on process restart).
METRICS_WINDOW_SIZE = 1_000
LOG_BUFFER_SIZE = 500

# Default decision threshold if the model artifact does not provide a tuned one.
DEFAULT_THRESHOLD = 0.5
