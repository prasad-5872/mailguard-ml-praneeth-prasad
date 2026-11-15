"""In-memory, bounded request metrics.

All metrics live in process memory: they are shared by every user session of a
running Streamlit app, and they are RESET whenever the app process restarts
(redeploy, sleep/wake on Streamlit Community Cloud, crash). Counters are
cumulative since process start; latency percentiles and history use a bounded
window of the most recent requests.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from src.config import METRICS_WINDOW_SIZE
from src.metrics import percentile


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    timestamp: str
    status: str  # "success" | "error"
    latency_ms: float
    source: str
    input_bytes: int
    prediction: str | None = None
    spam_probability: float | None = None
    threshold: float | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class MetricsStore:
    """Thread-safe bounded store of inference request records and counters."""

    def __init__(self, window_size: int = METRICS_WINDOW_SIZE) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self._lock = threading.Lock()
        self._records: deque[RequestRecord] = deque(maxlen=window_size)
        self.started_at = utc_now_iso()
        self._total = 0
        self._success = 0
        self._errors = 0
        self._spam = 0
        self._ham = 0
        self._latency_sum_ms = 0.0

    @property
    def window_size(self) -> int:
        return self._records.maxlen or 0

    def record(self, rec: RequestRecord) -> None:
        with self._lock:
            self._records.append(rec)
            self._total += 1
            self._latency_sum_ms += rec.latency_ms
            if rec.status == "success":
                self._success += 1
                if rec.prediction == "spam":
                    self._spam += 1
                elif rec.prediction == "ham":
                    self._ham += 1
            else:
                self._errors += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            records = list(self._records)
            total, success, errors = self._total, self._success, self._errors
            spam, ham, latency_sum = self._spam, self._ham, self._latency_sum_ms
        window_latencies = [r.latency_ms for r in records]
        window_success = [r.latency_ms for r in records if r.status == "success"]
        return {
            "started_at": self.started_at,
            "total_requests": total,
            "successful_requests": success,
            "error_count": errors,
            "error_rate": (errors / total) if total else 0.0,
            "avg_latency_ms": (latency_sum / total) if total else None,
            "p95_latency_ms": percentile(window_latencies, 95),
            "p95_success_latency_ms": percentile(window_success, 95),
            "spam_count": spam,
            "ham_count": ham,
            "window_size": self.window_size,
            "records_in_window": len(records),
        }

    def recent(self, limit: int | None = None) -> list[dict[str, object]]:
        """Most recent records first."""
        with self._lock:
            items = list(self._records)
        items.reverse()
        if limit is not None:
            items = items[:limit]
        return [r.to_dict() for r in items]

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._total = self._success = self._errors = self._spam = self._ham = 0
            self._latency_sum_ms = 0.0
            self.started_at = utc_now_iso()
