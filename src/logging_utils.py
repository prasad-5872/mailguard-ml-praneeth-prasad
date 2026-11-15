"""Structured (JSON) logging with an in-memory ring buffer for the monitoring page.

Privacy: callers must never pass raw email content to the logger. The inference
service only logs request metadata (request id, sizes, label, probability,
latency, error type).
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from collections import deque
from datetime import datetime, timezone

from src.config import LOG_BUFFER_SIZE

LOGGER_NAME = "mailguard"
_SETUP_LOCK = threading.Lock()


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class RingBufferHandler(logging.Handler):
    """Keep the most recent N formatted records in memory (bounded)."""

    def __init__(self, capacity: int = LOG_BUFFER_SIZE) -> None:
        super().__init__()
        self._records: deque[dict[str, object]] = deque(maxlen=capacity)
        self._lock_records = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._records.maxlen or 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = json.loads(self.format(record))
        except Exception:  # pragma: no cover - formatting failures must never break inference
            return
        with self._lock_records:
            self._records.append(entry)

    def records(self, limit: int | None = None) -> list[dict[str, object]]:
        with self._lock_records:
            items = list(self._records)
        items.reverse()  # newest first
        return items[:limit] if limit else items

    def clear(self) -> None:
        with self._lock_records:
            self._records.clear()


def get_logger(buffer_capacity: int = LOG_BUFFER_SIZE) -> logging.Logger:
    """Return the configured application logger (idempotent; safe under Streamlit reruns)."""
    logger = logging.getLogger(LOGGER_NAME)
    with _SETUP_LOCK:
        if not getattr(logger, "_mailguard_configured", False):
            logger.setLevel(logging.INFO)
            logger.propagate = False
            formatter = JsonFormatter()
            stream = logging.StreamHandler(sys.stdout)
            stream.setFormatter(formatter)
            ring = RingBufferHandler(buffer_capacity)
            ring.setFormatter(formatter)
            logger.addHandler(stream)
            logger.addHandler(ring)
            logger._mailguard_configured = True  # type: ignore[attr-defined]
            logger._mailguard_ring = ring  # type: ignore[attr-defined]
    return logger


def get_log_buffer(logger: logging.Logger | None = None) -> RingBufferHandler:
    logger = logger or get_logger()
    return logger._mailguard_ring  # type: ignore[attr-defined]


def log_event(logger: logging.Logger, level: int, message: str, **fields: object) -> None:
    """Log a message with structured fields."""
    logger.log(level, message, extra={"fields": fields})
