from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Dict, Iterator, Optional

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    from opentelemetry.trace import get_current_span
except Exception:  # pragma: no cover
    trace = None  # type: ignore[assignment]
    Resource = object  # type: ignore[assignment]
    TracerProvider = object  # type: ignore[assignment]
    BatchSpanProcessor = object  # type: ignore[assignment]
    OTLPSpanExporter = object  # type: ignore[assignment]
    ParentBased = object  # type: ignore[assignment]
    TraceIdRatioBased = object  # type: ignore[assignment]
    get_current_span = lambda: None  # type: ignore


def init_tracing(
    service_name: str, sampling_ratio: float = 1.0, otlp_endpoint: Optional[str] = None
) -> None:
    """Initialize OpenTelemetry tracing with optional OTLP exporter."""
    if trace is None:  # pragma: no cover
        return
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(
        resource=resource, sampler=ParentBased(TraceIdRatioBased(float(sampling_ratio)))
    )
    if otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=str(otlp_endpoint), insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


@contextmanager
def start_span(name: str, **attrs: Dict[str, Any]) -> Iterator[Any]:
    """Synchronous context manager for creating spans."""
    if trace is None:  # pragma: no cover
        yield None
        return
    tracer = trace.get_tracer("modelmuxer")
    with tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            if v is not None:
                try:
                    span.set_attribute(k, v)
                except Exception:
                    pass
        yield span


@asynccontextmanager
async def start_span_async(name: str, **attrs: Dict[str, Any]) -> AsyncIterator[Any]:
    """Asynchronous context manager for creating spans."""
    if trace is None:  # pragma: no cover
        yield None
        return
    tracer = trace.get_tracer("modelmuxer")
    with tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            if v is not None:
                try:
                    span.set_attribute(k, v)
                except Exception:
                    pass
        yield span


def get_trace_id() -> Optional[str]:
    """Return current trace id in hex, if any."""
    try:
        span = get_current_span()
        ctx = span.get_span_context() if span else None
        if ctx and getattr(ctx, "trace_id", 0):
            return f"{ctx.trace_id:032x}"
    except Exception:  # pragma: no cover
        return None
    return None
