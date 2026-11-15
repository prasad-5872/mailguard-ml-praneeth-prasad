"""Process-wide resources for the Streamlit app (model, metrics store, logger, artifacts).

``st.cache_resource`` makes these singletons per server process, so the metrics
store is shared by all sessions and is reset when the process restarts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import streamlit as st

from src import config
from src.inference import InferenceService
from src.logging_utils import get_logger, log_event
from src.model import ModelLoadError, SpamModel, load_model
from src.monitoring import MetricsStore


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def _cached_json(path_str: str, mtime: float) -> dict[str, Any] | None:
    return read_json(Path(path_str))


def load_artifact_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON artifact, re-reading it if the file changes."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return _cached_json(str(path), mtime)


@st.cache_resource(show_spinner="Loading model artifact...")
def get_model_bundle() -> tuple[SpamModel | None, str | None]:
    logger = get_logger()
    card = read_json(config.MODEL_CARD_PATH) or {}
    expected = (card.get("artifact") or {}).get("sha256")
    try:
        model = load_model(config.MODEL_PATH, expected_sha256=expected)
    except ModelLoadError as exc:
        log_event(logger, logging.ERROR, "model_load_failed", event="model_load", error=str(exc))
        return None, str(exc)
    except Exception as exc:  # defensive: never crash the UI on startup
        log_event(logger, logging.ERROR, "model_load_failed", event="model_load", error=type(exc).__name__)
        return None, f"Unexpected error while loading the model: {type(exc).__name__}"
    log_event(
        logger,
        logging.INFO,
        "model_loaded",
        event="model_load",
        model=model.name,
        model_version=model.version,
        checksum_verified=bool(expected),
        warnings=len(model.metadata.get("load_warnings", [])),
    )
    return model, None


@st.cache_resource
def get_metrics_store() -> MetricsStore:
    return MetricsStore(window_size=config.METRICS_WINDOW_SIZE)


@st.cache_resource
def get_service() -> InferenceService:
    model, error = get_model_bundle()
    return InferenceService(model, get_metrics_store(), get_logger(), model_error=error)


def theme_is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False
