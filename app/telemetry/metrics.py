# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Prometheus metrics for ModelMuxer telemetry.

This module provides metrics collection for monitoring application performance,
routing decisions, and system health. Uses graceful fallbacks when prometheus_client
is not available.
"""

import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# Define protocol types unconditionally so annotations are always available
@runtime_checkable
class _CounterProtocol(Protocol):
    def labels(self, *label_values: str, **label_kwargs: str) -> "_CounterProtocol": ...

    def inc(self, amount: float = 1.0) -> None: ...


@runtime_checkable
class _HistogramProtocol(Protocol):
    def labels(self, *label_values: str, **label_kwargs: str) -> "_HistogramProtocol": ...

    def observe(self, amount: float, exemplar: Dict[str, str] | None = None) -> None: ...


@runtime_checkable
class _GaugeProtocol(Protocol):
    def labels(self, *label_values: str, **label_kwargs: str) -> "_GaugeProtocol": ...

    def set(self, value: float) -> None: ...


try:
    from prometheus_client import Counter as _PromCounter
    from prometheus_client import Gauge as _PromGauge
    from prometheus_client import Histogram as _PromHistogram
    from prometheus_client import Summary as _PromSummary
except ImportError:
    logger.warning("prometheus_client not available - using no-op metrics")

    class _NoopMetric:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, *label_values: str, **label_kwargs: str) -> "_NoopMetric":
            return self

        def inc(self, amount: float = 1.0) -> None:
            pass

        def observe(self, amount: float, exemplar: Dict[str, str] | None = None) -> None:
            pass

        def set(self, value: float) -> None:
            pass

        def time(self) -> "_NoopContext":
            return _NoopContext()

    class _NoopContext:
        def __enter__(self) -> "_NoopContext":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    Counter = _NoopMetric
    Histogram = _NoopMetric
    Gauge = _NoopMetric
    Summary = _NoopMetric
else:
    # Re-export concrete classes for construction, but we will type the module-level
    # metrics using the Protocols above to avoid UnknownMemberType in strict mode.
    Counter = _PromCounter  # type: ignore
    Histogram = _PromHistogram  # type: ignore
    Gauge = _PromGauge  # type: ignore
    Summary = _PromSummary  # type: ignore

"""
Common metrics for both app and tests. We expose both names used across tests and dashboards.
"""

# Request metrics (legacy names used by tests/dashboards)
HTTP_REQUESTS_TOTAL: _CounterProtocol = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["route", "method", "status"],
)

HTTP_LATENCY: _HistogramProtocol = Histogram(
    "http_request_duration_milliseconds",
    "HTTP request duration in milliseconds",
    ["route", "method"],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
)

# App-prefixed equivalents for internal use
REQUESTS_TOTAL: _CounterProtocol = Counter(
    "modelmuxer_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
)

REQUEST_DURATION: _HistogramProtocol = Histogram(
    "modelmuxer_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0],
)

# Router metrics
ROUTER_REQUESTS: _CounterProtocol = Counter(
    "modelmuxer_router_requests_total", "Total router requests by route type", ["route"]
)

ROUTER_DECISION_LATENCY: _HistogramProtocol = Histogram(
    "modelmuxer_router_decision_latency_ms",
    "Router decision latency in milliseconds",
    ["route"],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

ROUTER_FALLBACKS: _CounterProtocol = Counter(
    "modelmuxer_router_fallbacks_total",
    "Router fallback events by route and reason",
    ["route", "reason"],
)

ROUTER_INTENT_TOTAL: _CounterProtocol = Counter(
    "modelmuxer_router_intent_total", "Intent classification results", ["label"]
)

# Cost estimation and budget metrics
LLM_ROUTER_COST_ESTIMATE_USD_SUM: _CounterProtocol = Counter(
    "modelmuxer_router_cost_estimate_usd_sum",
    (
        "Total estimated cost in USD for router decisions. The 'within_budget' label indicates whether "
        "the model estimate was within the configured budget threshold ('true') or exceeded it ('false'), "
        "enabling analysis of budget gating effectiveness."
    ),
    ["route", "model", "within_budget"],
)

LLM_ROUTER_ETA_MS_BUCKET: _HistogramProtocol = Histogram(
    "modelmuxer_router_eta_ms_bucket",
    "Estimated latency distribution in milliseconds",
    ["route", "model"],
    buckets=[50, 100, 250, 500, 1000, 2000, 5000, 10000],
)

LLM_ROUTER_BUDGET_EXCEEDED_TOTAL: _CounterProtocol = Counter(
    "modelmuxer_router_budget_exceeded_total",
    "Total budget exceeded events by route and reason",
    ["route", "reason"],
)

LLM_ROUTER_DOWN_ROUTE_TOTAL: _CounterProtocol = Counter(
    "modelmuxer_router_down_route_total",
    "Total down-routing events from expensive to cheaper models",
    ["route", "from_model", "to_model"],
)

LLM_ROUTER_UNPRICED_MODELS_SKIPPED: _CounterProtocol = Counter(
    "modelmuxer_router_unpriced_models_skipped_total",
    "Total unpriced models skipped during routing",
    ["route"],
)

LLM_ROUTER_SELECTED_COST_ESTIMATE_USD: _CounterProtocol = Counter(
    "modelmuxer_router_selected_cost_estimate_usd_sum",
    "Estimated cost for selected model",
    ["route", "model"],
)

# Provider metrics
PROVIDER_REQUESTS: _CounterProtocol = Counter(
    "modelmuxer_provider_requests_total",
    "Provider requests by provider, model, and outcome",
    ["provider", "model", "outcome"],
)

PROVIDER_LATENCY: _HistogramProtocol = Histogram(
    "modelmuxer_provider_latency_seconds",
    "Provider response latency in seconds",
    ["provider", "model"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0],
)

PROVIDER_ERRORS: _CounterProtocol = Counter(
    "modelmuxer_provider_errors_total",
    "Provider errors by provider and error type",
    ["provider", "error_type"],
)

# Cache metrics
CACHE_HITS: _CounterProtocol = Counter("modelmuxer_cache_hits_total", "Cache hits by cache type", ["cache_type"])

CACHE_MISSES: _CounterProtocol = Counter("modelmuxer_cache_misses_total", "Cache misses by cache type", ["cache_type"])

# System metrics
ACTIVE_CONNECTIONS: _GaugeProtocol = Gauge("modelmuxer_active_connections", "Number of active connections")

MEMORY_USAGE: _GaugeProtocol = Gauge("modelmuxer_memory_usage_bytes", "Memory usage in bytes")

# Health check metrics
HEALTH_CHECK_STATUS: _GaugeProtocol = Gauge(
    "modelmuxer_health_check_status", "Health check status (1=healthy, 0=unhealthy)", ["component"]
)

# Cost tracking metrics
COST_TOTAL_USD: _CounterProtocol = Counter(
    "modelmuxer_cost_total_usd", "Total cost in USD by provider and model", ["provider", "model"]
)

TOKENS_TOTAL: _CounterProtocol = Counter(
    "modelmuxer_tokens_total",
    "Total tokens by provider and model",
    ["provider", "model", "type"],
)

# Policy metrics
POLICY_VIOLATIONS: _CounterProtocol = Counter(
    "modelmuxer_policy_violations_total", "Policy violations by type", ["type"]
)

POLICY_REDACTIONS: _CounterProtocol = Counter(
    "modelmuxer_policy_redactions_total", "Policy redactions by PII type", ["pii_type"]
)
