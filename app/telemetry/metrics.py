from __future__ import annotations

from typing import Any

try:
    from prometheus_client import Counter, Histogram
except Exception:  # pragma: no cover

    class _Noop:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, *args: Any, **kwargs: Any) -> "_Noop":
            return self

        def inc(self, *args: Any, **kwargs: Any) -> None:
            pass

        def observe(self, *args: Any, **kwargs: Any) -> None:
            pass

    Counter = _Noop  # type: ignore[assignment]
    Histogram = _Noop  # type: ignore[assignment]


LLM_REQUESTS = Counter(
    "llm_requests_total",
    "Total LLM adapter requests",
    ["provider", "model", "outcome"],
)

LLM_LATENCY = Histogram(
    "llm_request_latency_ms",
    "LLM adapter request latency in milliseconds",
    ["provider", "model"],
    buckets=[50, 100, 250, 500, 1000, 2000, 5000, 10000],
)

LLM_TOKENS_IN = Counter(
    "llm_request_tokens_in_total",
    "Total input tokens processed by provider adapter",
    ["provider", "model"],
)

LLM_TOKENS_OUT = Counter(
    "llm_request_tokens_out_total",
    "Total output tokens processed by provider adapter",
    ["provider", "model"],
)

# Policy metrics
POLICY_VIOLATIONS = Counter(
    "policy_violations_total",
    "Policy violations by type",
    ["type"],
)

POLICY_REDACTIONS = Counter(
    "policy_redactions_total",
    "PII redaction counts by type",
    ["pii_type"],
)

# HTTP metrics
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests",
    ["route", "method", "status"],
)

HTTP_LATENCY = Histogram(
    "http_request_latency_ms",
    "HTTP request latency (ms)",
    ["route", "method"],
    buckets=[50, 100, 250, 500, 1000, 2000, 5000, 10000],
)

# Router metrics
ROUTER_REQUESTS = Counter(
    "llm_router_requests_total",
    "Router decisions",
    ["route"],
)

ROUTER_DECISION_LATENCY = Histogram(
    "llm_router_decision_latency_ms",
    "Router decision latency (ms)",
    ["route"],
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000],
)

ROUTER_FALLBACKS = Counter(
    "llm_router_fallbacks_total",
    "Router fallbacks",
    ["route", "reason"],
)

# Intent classification metrics
ROUTER_INTENT_TOTAL = Counter(
    "llm_router_intent_total",
    "Intent classifications by label",
    ["label"],
)
