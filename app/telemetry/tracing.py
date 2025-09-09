# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterator,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    # Only import for type checking to avoid runtime dependency issues
    from opentelemetry import trace as trace_module
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as OtelOTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource as OtelResource
    from opentelemetry.sdk.trace import TracerProvider as OtelTracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor as OtelBatchSpanProcessor,
    )
    from opentelemetry.sdk.trace.sampling import ParentBased as OtelParentBased
    from opentelemetry.sdk.trace.sampling import (
        TraceIdRatioBased as OtelTraceIdRatioBased,
    )
    from opentelemetry.trace import Span as OtelSpan
else:
    trace_module = None
    OtelResource = None
    OtelTracerProvider = None
    OtelBatchSpanProcessor = None
    OtelOTLPSpanExporter = None
    OtelParentBased = None
    OtelTraceIdRatioBased = None
    OtelSpan = None

# Runtime imports with proper error handling
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    from opentelemetry.trace import get_current_span

    OPENTELEMETRY_AVAILABLE = True
except ImportError:  # pragma: no cover
    # Create proper stub objects that don't cause type issues
    trace = None  # type: ignore
    Resource = None  # type: ignore
    TracerProvider = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
    OTLPSpanExporter = None  # type: ignore
    ParentBased = None  # type: ignore
    TraceIdRatioBased = None  # type: ignore
    get_current_span = None  # type: ignore
    OPENTELEMETRY_AVAILABLE = False


def init_tracing(
    service_name: str, sampling_ratio: float = 1.0, otlp_endpoint: str | None = None
) -> None:
    """Initialize OpenTelemetry tracing with optional OTLP exporter."""
    if not OPENTELEMETRY_AVAILABLE:
        return

    # At this point we know OpenTelemetry is available and all imports succeeded
    # The type checker may not understand this, but runtime guarantees these are not None
    assert trace is not None
    assert Resource is not None
    assert TracerProvider is not None
    assert BatchSpanProcessor is not None
    assert OTLPSpanExporter is not None
    assert ParentBased is not None
    assert TraceIdRatioBased is not None

    resource = Resource.create({"service.name": service_name})
    sampler = ParentBased(TraceIdRatioBased(float(sampling_ratio)))
    provider = TracerProvider(resource=resource, sampler=sampler)

    if otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=str(otlp_endpoint), insecure=True)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)


@runtime_checkable
class SpanProtocol(Protocol):
    def set_attribute(self, key: str, value: object) -> None: ...

    def add_event(self, name: str, attributes: dict[str, object] | None = None) -> None: ...


@contextmanager
def start_span(name: str, **attrs: Any) -> Iterator[SpanProtocol | None]:
    """Synchronous context manager for creating spans."""
    if not OPENTELEMETRY_AVAILABLE or trace is None:
        yield None
        return

    tracer = trace.get_tracer("modelmuxer")
    with tracer.start_as_current_span(name) as span:
        # Set attributes on the span
        for k, v in attrs.items():
            if v is not None:
                try:
                    span.set_attribute(k, v)
                except Exception:
                    pass

        # Create a wrapper that implements SpanProtocol
        class SpanWrapper:
            def __init__(self, otel_span: Any) -> None:
                self._span = otel_span

            def set_attribute(self, key: str, value: object) -> None:
                try:
                    self._span.set_attribute(key, value)
                except Exception:
                    pass

            def add_event(self, name: str, attributes: dict[str, object] | None = None) -> None:
                try:
                    self._span.add_event(name, attributes or {})
                except Exception:
                    pass

        yield SpanWrapper(span)


@asynccontextmanager
async def start_span_async(name: str, **attrs: Any) -> AsyncIterator[SpanProtocol | None]:
    """Asynchronous context manager for creating spans."""
    if not OPENTELEMETRY_AVAILABLE or trace is None:
        yield None
        return

    tracer = trace.get_tracer("modelmuxer")
    with tracer.start_as_current_span(name) as span:
        # Set attributes on the span
        for k, v in attrs.items():
            if v is not None:
                try:
                    span.set_attribute(k, v)
                except Exception:
                    pass

        # Create a wrapper that implements SpanProtocol
        class SpanWrapper:
            def __init__(self, otel_span: Any) -> None:
                self._span = otel_span

            def set_attribute(self, key: str, value: object) -> None:
                try:
                    self._span.set_attribute(key, value)
                except Exception:
                    pass

            def add_event(self, name: str, attributes: dict[str, object] | None = None) -> None:
                try:
                    self._span.add_event(name, attributes or {})
                except Exception:
                    pass

        yield SpanWrapper(span)


def get_trace_id() -> str | None:
    """Return current trace id in hex, if any."""
    if not OPENTELEMETRY_AVAILABLE or get_current_span is None:
        return None

    try:
        span = get_current_span()
        if span is None:
            return None
        ctx = span.get_span_context()
        if ctx and hasattr(ctx, "trace_id") and ctx.trace_id:
            return f"{ctx.trace_id:032x}"
    except Exception:  # pragma: no cover
        pass
    return None
