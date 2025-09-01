from __future__ import annotations

import json
import logging
from typing import Any, Dict

try:
    from opentelemetry.trace import get_current_span as otel_get_current_span
except ImportError:  # pragma: no cover
    otel_get_current_span = None


def get_current_span():
    """Get the current OpenTelemetry span, or None if not available."""
    if otel_get_current_span is not None:
        return otel_get_current_span()
    return None


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        # Attach trace/span ids if present
        try:
            span = get_current_span()
            ctx = span.get_span_context() if span else None
            if ctx:
                base["trace_id"] = f"{ctx.trace_id:032x}"
                base["span_id"] = f"{ctx.span_id:016x}"
        except Exception:
            pass
        return json.dumps(base, ensure_ascii=False)


def configure_json_logging(level: str = "info") -> None:
    logging.root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    logging.root.addHandler(handler)
    logging.root.setLevel(level.upper())
