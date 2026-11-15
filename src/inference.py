"""Inference service: validation -> parsing -> features -> prediction -> explanation, with
request IDs, latency measurement, structured logging and metrics recording.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.features import STRUCTURED_FEATURES, build_feature_frame
from src.logging_utils import log_event
from src.metrics import apply_threshold
from src.model import SpamModel
from src.monitoring import MetricsStore, RequestRecord, utc_now_iso
from src.parsing import ParsedEmail, parse_email
from src.validation import ValidationError, validate_parsed_email, validate_raw_email, validate_threshold


class ModelUnavailableError(RuntimeError):
    """The model artifact could not be loaded; inference is disabled."""


class InferenceError(RuntimeError):
    """Unexpected failure during inference (details are logged, not shown verbatim)."""

    def __init__(self, message: str, request_id: str) -> None:
        super().__init__(message)
        self.request_id = request_id


@dataclass
class PredictionResult:
    request_id: str
    label: str
    spam_probability: float
    threshold: float
    latency_ms: float
    model_name: str
    model_version: str
    parsed: ParsedEmail
    features: dict[str, float]
    explanation: dict[str, Any] = field(default_factory=dict)

    def label_at(self, threshold: float) -> str:
        """Re-label the stored probability at a different threshold (no new inference request)."""
        t = validate_threshold(threshold)
        return "spam" if int(apply_threshold(self.spam_probability, t)[0]) == 1 else "ham"


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


class InferenceService:
    def __init__(
        self,
        model: SpamModel | None,
        metrics: MetricsStore,
        logger: logging.Logger,
        model_error: str | None = None,
    ) -> None:
        self.model = model
        self.metrics = metrics
        self.logger = logger
        self.model_error = model_error

    @property
    def ready(self) -> bool:
        return self.model is not None

    def analyze(
        self,
        raw: str | bytes,
        threshold: float | None = None,
        source: str = "paste",
        top_k: int = 12,
    ) -> PredictionResult:
        request_id = new_request_id()
        start = time.perf_counter()
        input_bytes = len(raw) if isinstance(raw, bytes) else len(str(raw or "").encode("utf-8", errors="replace"))
        thr_for_log: float | None = None
        try:
            if self.model is None:
                raise ModelUnavailableError(self.model_error or "Model is not loaded.")
            thr = validate_threshold(self.model.threshold if threshold is None else threshold)
            thr_for_log = thr
            validated = validate_raw_email(raw)
            parsed = validate_parsed_email(parse_email(validated))
            frame = build_feature_frame([parsed])
            prob = float(self.model.predict_proba(frame)[0])
            label = "spam" if prob >= thr else "ham"
            explanation = self.model.explain(frame, top_k=top_k)
            latency_ms = (time.perf_counter() - start) * 1000.0
        except ValidationError as exc:
            self._record_error(request_id, start, source, input_bytes, thr_for_log, "validation_error", str(exc))
            raise
        except ModelUnavailableError as exc:
            self._record_error(request_id, start, source, input_bytes, thr_for_log, "model_unavailable", str(exc))
            raise
        except Exception as exc:
            self._record_error(
                request_id, start, source, input_bytes, thr_for_log, "internal_error", type(exc).__name__, exc_info=True
            )
            raise InferenceError("Unexpected error while analysing the email.", request_id) from exc

        self.metrics.record(
            RequestRecord(
                request_id=request_id,
                timestamp=utc_now_iso(),
                status="success",
                latency_ms=latency_ms,
                source=source,
                input_bytes=input_bytes,
                prediction=label,
                spam_probability=prob,
                threshold=thr,
            )
        )
        # Metadata only: raw email content is never logged.
        log_event(
            self.logger,
            logging.INFO,
            "inference_success",
            event="inference",
            request_id=request_id,
            source=source,
            input_bytes=input_bytes,
            has_headers=parsed.has_headers,
            url_count=len(parsed.urls),
            prediction=label,
            spam_probability=round(prob, 6),
            threshold=thr,
            latency_ms=round(latency_ms, 3),
            model=self.model.name,
            model_version=self.model.version,
        )
        return PredictionResult(
            request_id=request_id,
            label=label,
            spam_probability=prob,
            threshold=thr,
            latency_ms=latency_ms,
            model_name=self.model.display_name,
            model_version=self.model.version,
            parsed=parsed,
            features={k: float(frame.iloc[0][k]) for k in STRUCTURED_FEATURES},
            explanation=explanation,
        )

    def _record_error(
        self,
        request_id: str,
        start: float,
        source: str,
        input_bytes: int,
        threshold: float | None,
        error_type: str,
        message: str,
        exc_info: bool = False,
    ) -> None:
        latency_ms = (time.perf_counter() - start) * 1000.0
        self.metrics.record(
            RequestRecord(
                request_id=request_id,
                timestamp=utc_now_iso(),
                status="error",
                latency_ms=latency_ms,
                source=source,
                input_bytes=input_bytes,
                threshold=threshold,
                error_type=error_type,
                error_message=message[:200],
            )
        )
        level = logging.ERROR if error_type == "internal_error" else logging.WARNING
        self.logger.log(
            level,
            "inference_failed",
            exc_info=exc_info,
            extra={
                "fields": {
                    "event": "inference",
                    "request_id": request_id,
                    "source": source,
                    "input_bytes": input_bytes,
                    "error_type": error_type,
                    "error": message[:200],
                    "latency_ms": round(latency_ms, 3),
                }
            },
        )
