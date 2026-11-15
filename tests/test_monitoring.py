import json
import logging
import threading

import pytest

from src.logging_utils import JsonFormatter, RingBufferHandler, log_event
from src.monitoring import MetricsStore, RequestRecord, utc_now_iso


def _rec(i: int, status: str = "success", latency: float = 10.0, prediction: str | None = "ham") -> RequestRecord:
    return RequestRecord(
        request_id=f"r{i}",
        timestamp=utc_now_iso(),
        status=status,
        latency_ms=latency,
        source="test",
        input_bytes=100,
        prediction=prediction if status == "success" else None,
        error_type=None if status == "success" else "validation_error",
    )


def test_empty_snapshot():
    snap = MetricsStore(10).snapshot()
    assert snap["total_requests"] == 0 and snap["error_rate"] == 0.0
    assert snap["avg_latency_ms"] is None and snap["p95_latency_ms"] is None


def test_counters_and_rates():
    store = MetricsStore(100)
    for i in range(8):
        store.record(_rec(i, latency=float(i + 1), prediction="spam" if i % 2 else "ham"))
    store.record(_rec(8, status="error", latency=9.0))
    store.record(_rec(9, status="error", latency=10.0))
    snap = store.snapshot()
    assert snap["total_requests"] == 10
    assert snap["successful_requests"] == 8 and snap["error_count"] == 2
    assert snap["error_rate"] == pytest.approx(0.2)
    assert snap["spam_count"] == 4 and snap["ham_count"] == 4
    assert snap["avg_latency_ms"] == pytest.approx(5.5)
    assert snap["p95_latency_ms"] == pytest.approx(9.55)
    assert snap["p95_success_latency_ms"] == pytest.approx(7.65)


def test_window_is_bounded_but_counters_cumulative():
    store = MetricsStore(window_size=5)
    for i in range(12):
        store.record(_rec(i))
    snap = store.snapshot()
    assert snap["records_in_window"] == 5 and snap["total_requests"] == 12
    recent = store.recent()
    assert [r["request_id"] for r in recent] == ["r11", "r10", "r9", "r8", "r7"]
    assert len(store.recent(2)) == 2


def test_reset():
    store = MetricsStore(5)
    store.record(_rec(1))
    store.reset()
    assert store.snapshot()["total_requests"] == 0 and store.recent() == []


def test_invalid_window():
    with pytest.raises(ValueError):
        MetricsStore(0)


def test_thread_safety():
    store = MetricsStore(window_size=10_000)

    def worker(offset: int) -> None:
        for i in range(500):
            store.record(_rec(offset + i))

    threads = [threading.Thread(target=worker, args=(k * 1000,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = store.snapshot()
    assert snap["total_requests"] == 4000 and snap["records_in_window"] == 4000


def test_ring_buffer_bounded_and_json():
    logger = logging.getLogger("mailguard-ring-test")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    ring = RingBufferHandler(3)
    ring.setFormatter(JsonFormatter())
    logger.addHandler(ring)
    for i in range(5):
        log_event(logger, logging.INFO, "evt", request_id=f"id{i}", latency_ms=1.5)
    records = ring.records()
    assert [r["request_id"] for r in records] == ["id4", "id3", "id2"]
    assert records[0]["level"] == "INFO" and "timestamp" in records[0]
    json.dumps(records)
    ring.clear()
    assert ring.records() == []
