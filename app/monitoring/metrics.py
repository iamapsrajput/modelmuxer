# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Prometheus metrics collection for ModelMuxer.

This module provides comprehensive metrics collection for monitoring
system performance, usage patterns, and health indicators.
"""

import time
from collections import defaultdict
from typing import Any

import structlog

# Optional prometheus imports
try:
    from prometheus_client import CollectorRegistry, Gauge, Histogram, Info
    from prometheus_client import Counter as PrometheusCounter

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Create dummy classes that do nothing
    class DummyCounter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def inc(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, *args: Any, **kwargs: Any) -> "DummyCounter":
            return self

    class DummyGauge:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def set(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, *args: Any, **kwargs: Any) -> "DummyGauge":
            return self

    class DummyHistogram:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def observe(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, *args: Any, **kwargs: Any) -> "DummyHistogram":
            return self

    class DummyInfo:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def info(self, *args: Any, **kwargs: Any) -> None:
            pass

    class DummyCollectorRegistry:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    # Assign dummy classes when Prometheus is not available
    # Using type ignore for compatibility with prometheus_client types
    PrometheusCounter = DummyCounter  # type: ignore[misc,assignment]
    Gauge = DummyGauge  # type: ignore[misc,assignment]
    Histogram = DummyHistogram  # type: ignore[misc,assignment]
    Info = DummyInfo  # type: ignore[misc,assignment]
    CollectorRegistry = DummyCollectorRegistry  # type: ignore[misc,assignment]


logger = structlog.get_logger(__name__)


class MetricsCollector:
    """
    Comprehensive metrics collector for ModelMuxer.

    Collects and exposes metrics for Prometheus monitoring including
    request metrics, provider performance, routing decisions, and system health.
    """

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()

        if not PROMETHEUS_AVAILABLE:
            logger.warning(
                "prometheus_not_available",
                message="Metrics collection disabled - prometheus-client not installed",
            )

        # Request metrics
        self.request_total = PrometheusCounter(
            "modelmuxer_requests_total",
            "Total number of requests",
            ["method", "endpoint", "status_code", "user_id"],
            registry=self.registry,
        )

        self.request_duration = Histogram(
            "modelmuxer_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry,
        )

        # Provider metrics
        self.provider_requests = PrometheusCounter(
            "modelmuxer_provider_requests_total",
            "Total requests per provider",
            ["provider", "model", "status"],
            registry=self.registry,
        )

        self.provider_duration = Histogram(
            "modelmuxer_provider_duration_seconds",
            "Provider response duration",
            ["provider", "model"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            registry=self.registry,
        )

        self.provider_tokens = PrometheusCounter(
            "modelmuxer_provider_tokens_total",
            "Total tokens processed by provider",
            ["provider", "model", "type"],  # token type: input/output
            registry=self.registry,
        )

        self.provider_cost = PrometheusCounter(
            "modelmuxer_provider_cost_total",
            "Total cost by provider",
            ["provider", "model"],
            registry=self.registry,
        )

        # Routing metrics
        self.routing_decisions = PrometheusCounter(
            "modelmuxer_routing_decisions_total",
            "Routing decisions by strategy",
            ["strategy", "selected_provider", "selected_model"],
            registry=self.registry,
        )

        self.routing_confidence = Histogram(
            "modelmuxer_routing_confidence",
            "Routing decision confidence scores",
            ["strategy"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry,
        )

        # Classification metrics
        self.classification_results = PrometheusCounter(
            "modelmuxer_classification_total",
            "Classification results by category",
            ["category", "method"],
            registry=self.registry,
        )

        self.classification_confidence = Histogram(
            "modelmuxer_classification_confidence",
            "Classification confidence scores",
            ["category"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry,
        )

        # Rate limiting metrics
        self.rate_limit_hits = PrometheusCounter(
            "modelmuxer_rate_limit_hits_total",
            "Rate limit violations",
            ["user_id", "limit_type"],
            registry=self.registry,
        )

        # Cache metrics
        self.cache_operations = PrometheusCounter(
            "modelmuxer_cache_operations_total",
            "Cache operations",
            ["operation", "result"],  # operation: get/set/delete, result: hit/miss/success/error
            registry=self.registry,
        )

        # System metrics
        self.active_connections = Gauge(
            "modelmuxer_active_connections", "Number of active connections", registry=self.registry
        )

        self.memory_usage = Gauge(
            "modelmuxer_memory_usage_bytes",
            "Memory usage in bytes",
            ["type"],  # memory type: rss/vms/shared
            registry=self.registry,
        )

        # Error metrics
        self.errors_total = PrometheusCounter(
            "modelmuxer_errors_total",
            "Total errors by type",
            ["error_type", "endpoint", "provider"],
            registry=self.registry,
        )

        # Health metrics
        self.provider_health = Gauge(
            "modelmuxer_provider_health",
            "Provider health status (1=healthy, 0=unhealthy)",
            ["provider"],
            registry=self.registry,
        )

        # System info
        self.system_info = Info(
            "modelmuxer_system_info", "System information", registry=self.registry
        )

        # Enhanced Production Metrics (Part 3)
        # =====================================

        # Cascade routing metrics
        self.cascade_steps_total = Histogram(
            "modelmuxer_cascade_steps_total",
            "Number of cascade steps per request",
            ["cascade_type", "final_provider"],
            buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            registry=self.registry,
        )

        self.cost_per_request = Histogram(
            "modelmuxer_cost_per_request",
            "Cost per request in USD",
            ["provider", "model", "routing_strategy"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            registry=self.registry,
        )

        self.quality_score_distribution = Histogram(
            "modelmuxer_quality_score_distribution",
            "Distribution of response quality scores",
            ["provider", "model"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry,
        )

        self.confidence_score_distribution = Histogram(
            "modelmuxer_confidence_score_distribution",
            "Distribution of confidence scores",
            ["provider", "model"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry,
        )

        # Budget and cost tracking
        self.budget_utilization_ratio = Gauge(
            "modelmuxer_budget_utilization_ratio",
            "Budget utilization percentage",
            ["user_id", "budget_type", "provider"],
            registry=self.registry,
        )

        self.active_users = Gauge(
            "modelmuxer_active_users",
            "Number of active users in last 24h",
            registry=self.registry,
        )

        # Cache performance
        self.cache_hit_ratio = Gauge(
            "modelmuxer_cache_hit_ratio",
            "Cache hit percentage",
            ["cache_type"],
            registry=self.registry,
        )

        # Multi-tenancy metrics
        self.organization_requests = PrometheusCounter(
            "modelmuxer_organization_requests_total",
            "Total requests per organization",
            ["org_id", "plan_type"],
            registry=self.registry,
        )

        self.organization_cost = PrometheusCounter(
            "modelmuxer_organization_cost_total",
            "Total cost per organization",
            ["org_id", "plan_type"],
            registry=self.registry,
        )

        # Security metrics
        self.auth_attempts = PrometheusCounter(
            "modelmuxer_auth_attempts_total",
            "Authentication attempts",
            ["method", "result", "user_type"],
            registry=self.registry,
        )

        self.pii_detections = PrometheusCounter(
            "modelmuxer_pii_detections_total",
            "PII detection events",
            ["pii_type", "action_taken"],
            registry=self.registry,
        )

        # Internal tracking
        self.start_time = time.time()
        self.request_counts: defaultdict[str, int] = defaultdict(int)
        self.error_counts: defaultdict[str, int] = defaultdict(int)
        self.active_users_set: set[tuple[str, float]] = set()
        self.last_active_users_update = time.time()

        logger.info("metrics_collector_initialized")

    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        user_id: str | None = None,
    ) -> None:
        """Record request metrics."""
        labels = {
            "method": method,
            "endpoint": endpoint,
            "status_code": str(status_code),
            "user_id": user_id or "anonymous",
        }

        self.request_total.labels(**labels).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)

        # Update internal tracking
        self.request_counts[f"{method}:{endpoint}"] += 1

    def record_provider_request(
        self,
        provider: str,
        model: str,
        status: str,
        duration: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Record provider-specific metrics."""
        self.provider_requests.labels(provider=provider, model=model, status=status).inc()

        self.provider_duration.labels(provider=provider, model=model).observe(duration)

        if input_tokens > 0:
            self.provider_tokens.labels(provider=provider, model=model, type="input").inc(
                input_tokens
            )

        if output_tokens > 0:
            self.provider_tokens.labels(provider=provider, model=model, type="output").inc(
                output_tokens
            )

        if cost > 0:
            self.provider_cost.labels(provider=provider, model=model).inc(cost)

    def record_routing_decision(
        self, strategy: str, selected_provider: str, selected_model: str, confidence: float
    ) -> None:
        """Record routing decision metrics."""
        self.routing_decisions.labels(
            strategy=strategy, selected_provider=selected_provider, selected_model=selected_model
        ).inc()

        self.routing_confidence.labels(strategy=strategy).observe(confidence)

    def record_classification(self, category: str, method: str, confidence: float) -> None:
        """Record classification metrics."""
        self.classification_results.labels(category=category, method=method).inc()

        self.classification_confidence.labels(category=category).observe(confidence)

    def record_rate_limit_hit(self, user_id: str, limit_type: str) -> None:
        """Record rate limit violation."""
        self.rate_limit_hits.labels(user_id=user_id, limit_type=limit_type).inc()

    def record_cache_operation(self, operation: str, result: str) -> None:
        """Record cache operation metrics."""
        self.cache_operations.labels(operation=operation, result=result).inc()

    def record_error(self, error_type: str, endpoint: str, provider: str | None = None) -> None:
        """Record error metrics."""
        self.errors_total.labels(
            error_type=error_type, endpoint=endpoint, provider=provider or "unknown"
        ).inc()

        # Update internal tracking
        self.error_counts[error_type] += 1

    def update_provider_health(self, provider: str, is_healthy: bool) -> None:
        """Update provider health status."""
        self.provider_health.labels(provider=provider).set(1 if is_healthy else 0)

    def update_active_connections(self, count: int) -> None:
        """Update active connections count."""
        self.active_connections.set(count)

    def update_memory_usage(self, rss: int, vms: int, shared: int = 0) -> None:
        """Update memory usage metrics."""
        self.memory_usage.labels(type="rss").set(rss)
        self.memory_usage.labels(type="vms").set(vms)
        if shared > 0:
            self.memory_usage.labels(type="shared").set(shared)

    def set_system_info(self, info: dict[str, str]) -> None:
        """Set system information."""
        self.system_info.info(info)

    # Enhanced Production Metrics Methods (Part 3)
    # =============================================

    def record_cascade_request(
        self,
        cascade_type: str,
        final_provider: str,
        steps_count: int,
        total_cost: float,
        quality_score: float | None = None,
        confidence_score: float | None = None,
        routing_strategy: str = "cascade",
    ) -> None:
        """Record cascade routing metrics."""
        self.cascade_steps_total.labels(
            cascade_type=cascade_type, final_provider=final_provider
        ).observe(steps_count)

        self.cost_per_request.labels(
            provider=final_provider, model="cascade", routing_strategy=routing_strategy
        ).observe(total_cost)

        if quality_score is not None:
            self.quality_score_distribution.labels(
                provider=final_provider, model="cascade"
            ).observe(quality_score)

        if confidence_score is not None:
            self.confidence_score_distribution.labels(
                provider=final_provider, model="cascade"
            ).observe(confidence_score)

    def record_single_request_cost(
        self, provider: str, model: str, cost: float, routing_strategy: str = "single"
    ) -> None:
        """Record cost for single model requests."""
        self.cost_per_request.labels(
            provider=provider, model=model, routing_strategy=routing_strategy
        ).observe(cost)

    def update_budget_utilization(
        self, user_id: str, budget_type: str, provider: str, utilization_percent: float
    ) -> None:
        """Update budget utilization metrics."""
        self.budget_utilization_ratio.labels(
            user_id=user_id, budget_type=budget_type, provider=provider
        ).set(utilization_percent)

    def record_user_activity(self, user_id: str) -> None:
        """Record user activity for active users tracking."""
        current_time = time.time()
        self.active_users_set.add((user_id, current_time))

        # Clean up old entries every 5 minutes
        if current_time - self.last_active_users_update > 300:
            cutoff_time = current_time - 86400  # 24 hours ago
            self.active_users_set = {
                (uid, timestamp)
                for uid, timestamp in self.active_users_set
                if timestamp > cutoff_time
            }

            # Update active users count
            unique_users = len({uid for uid, _ in self.active_users_set})
            self.active_users.set(unique_users)
            self.last_active_users_update = current_time

    def update_cache_hit_ratio(self, cache_type: str, hit_ratio: float) -> None:
        """Update cache hit ratio metrics."""
        self.cache_hit_ratio.labels(cache_type=cache_type).set(hit_ratio)

    def record_organization_activity(self, org_id: str, plan_type: str, cost: float = 0.0) -> None:
        """Record organization-level metrics."""
        self.organization_requests.labels(org_id=org_id, plan_type=plan_type).inc()

        if cost > 0:
            self.organization_cost.labels(org_id=org_id, plan_type=plan_type).inc(cost)

    def record_auth_attempt(self, method: str, result: str, user_type: str = "user") -> None:
        """Record authentication attempts."""
        self.auth_attempts.labels(method=method, result=result, user_type=user_type).inc()

    def record_pii_detection(self, pii_type: str, action_taken: str) -> None:
        """Record PII detection events."""
        self.pii_detections.labels(pii_type=pii_type, action_taken=action_taken).inc()

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics."""
        current_time = time.time()
        uptime = current_time - self.start_time

        return {
            "uptime_seconds": uptime,
            "total_requests": sum(self.request_counts.values()),
            "total_errors": sum(self.error_counts.values()),
            "requests_per_endpoint": dict(self.request_counts),
            "errors_by_type": dict(self.error_counts),
            "error_rate": sum(self.error_counts.values())
            / max(sum(self.request_counts.values()), 1),
        }

    def reset_counters(self) -> None:
        """Reset internal counters (not Prometheus metrics)."""
        self.request_counts.clear()
        self.error_counts.clear()
        logger.info("internal_counters_reset")


class HealthChecker:
    """
    Health check system for ModelMuxer components.

    Monitors the health of providers, cache, database, and other
    critical system components.
    """

    def __init__(self, metrics_collector: MetricsCollector | None = None):
        self.metrics_collector = metrics_collector
        self.health_checks: dict[str, dict[str, Any]] = {}
        self.last_check_time = 0
        self.check_interval = 30  # seconds

        logger.info("health_checker_initialized")

    async def check_provider_health(self, provider_name: str, provider: Any) -> bool:
        """Check health of a specific provider."""
        try:
            is_healthy_result = await provider.health_check()
            is_healthy = bool(is_healthy_result)

            self.health_checks[f"provider_{provider_name}"] = {
                "status": "healthy" if is_healthy else "unhealthy",
                "last_check": time.time(),
                "error": None,
            }

            if self.metrics_collector:
                self.metrics_collector.update_provider_health(provider_name, is_healthy)

            return is_healthy

        except Exception as e:
            self.health_checks[f"provider_{provider_name}"] = {
                "status": "unhealthy",
                "last_check": time.time(),
                "error": str(e),
            }

            if self.metrics_collector:
                self.metrics_collector.update_provider_health(provider_name, False)

            logger.error("provider_health_check_failed", provider=provider_name, error=str(e))
            return False

    async def check_cache_health(self, cache: Any) -> bool:
        """Check cache system health."""
        try:
            # Try a simple cache operation
            test_key = "health_check_test"
            test_value = "test_value"

            await cache.set(test_key, test_value, ttl=10)
            retrieved_value = await cache.get(test_key)
            await cache.delete(test_key)

            is_healthy = bool(retrieved_value == test_value)

            self.health_checks["cache"] = {
                "status": "healthy" if is_healthy else "unhealthy",
                "last_check": time.time(),
                "error": None,
            }

            return is_healthy

        except Exception as e:
            self.health_checks["cache"] = {
                "status": "unhealthy",
                "last_check": time.time(),
                "error": str(e),
            }

            logger.error("cache_health_check_failed", error=str(e))
            return False

    def check_system_resources(self) -> dict[str, Any]:
        """Check system resource health."""
        try:
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()

            # Disk usage
            disk = psutil.disk_usage("/")

            # System load
            load_avg = psutil.getloadavg() if hasattr(psutil, "getloadavg") else (0, 0, 0)

            resource_status = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available,
                "disk_percent": disk.percent,
                "disk_free": disk.free,
                "load_avg_1m": load_avg[0],
                "load_avg_5m": load_avg[1],
                "load_avg_15m": load_avg[2],
            }

            # Determine overall health
            is_healthy = cpu_percent < 90 and memory.percent < 90 and disk.percent < 90

            self.health_checks["system_resources"] = {
                "status": "healthy" if is_healthy else "unhealthy",
                "last_check": time.time(),
                "details": resource_status,
                "error": None,
            }

            # Update metrics if available
            if self.metrics_collector:
                rss_value = getattr(memory, "rss", 0)
                vms_value = getattr(memory, "vms", 0)
                self.metrics_collector.update_memory_usage(rss_value, vms_value)

            return resource_status

        except ImportError:
            logger.warning("psutil_not_available_for_system_health_check")
            return {}
        except Exception as e:
            logger.error("system_health_check_failed", error=str(e))
            return {}

    def get_overall_health(self) -> dict[str, Any]:
        """Get overall system health status."""
        current_time = time.time()

        # Count healthy vs unhealthy components
        healthy_count = 0
        unhealthy_count = 0
        total_count = len(self.health_checks)

        for status in self.health_checks.values():
            if status["status"] == "healthy":
                healthy_count += 1
            else:
                unhealthy_count += 1

        # Determine overall status
        if unhealthy_count == 0:
            overall_status = "healthy"
        elif healthy_count > unhealthy_count:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        return {
            "status": overall_status,
            "timestamp": current_time,
            "components": self.health_checks.copy(),
            "summary": {
                "total_components": total_count,
                "healthy_components": healthy_count,
                "unhealthy_components": unhealthy_count,
            },
        }
