from __future__ import annotations

import json
import logging
from typing import Any, Dict

try:
    from opentelemetry.trace import get_current_span
except Exception:  # pragma: no cover

    def get_current_span():  # type: ignore
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
