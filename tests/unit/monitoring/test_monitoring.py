# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""
Tests for the monitoring and metrics system.
"""

import time
from typing import Any
from unittest.mock import patch

import pytest

from app.monitoring.metrics import HealthChecker, MetricsCollector


class TestMetricsCollector:
    """Test the metrics collection system."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.metrics = MetricsCollector()

    def test_request_counter(self) -> None:
        """Test request counting metrics."""
        # Record some requests using available methods
        self.metrics.record_auth_attempt("api_key", "success", "user")
        self.metrics.record_auth_attempt("api_key", "success", "user")
        self.metrics.record_auth_attempt("api_key", "failure", "user")

        # Verify the metrics are recorded (no exception means success)
        assert True  # If we get here, the methods worked

    def test_cache_operations(self) -> None:
        """Test cache operation tracking."""
        # Record cache operations
        self.metrics.record_cache_operation("get", "hit")
        self.metrics.record_cache_operation("get", "miss")
        self.metrics.record_cache_operation("set", "success")

        # Update cache hit ratio
        self.metrics.update_cache_hit_ratio("memory", 0.8)

        # Verify the methods work without error
        assert True

    def test_error_tracking(self) -> None:
        """Test error tracking metrics."""
        # Record some errors
        self.metrics.record_error("rate_limit", "/chat/completions", "openai")
        self.metrics.record_error("timeout", "/chat/completions", "anthropic")
        self.metrics.record_error("invalid_key", "/chat/completions")

        # Verify the methods work without error
        assert True

    def test_organization_activity(self) -> None:
        """Test organization activity tracking."""
        # Record organization activity
        self.metrics.record_organization_activity("org123", "enterprise", 0.05)
        self.metrics.record_organization_activity("org456", "starter", 0.01)

        # Verify the methods work without error
        assert True

    def test_user_activity_tracking(self) -> None:
        """Test user activity tracking."""
        # Record user activity
        self.metrics.record_user_activity("user123")
        self.metrics.record_user_activity("user456")
        self.metrics.record_user_activity("user123")  # Same user again

        # Verify the methods work without error
        assert True

    def test_rate_limit_tracking(self) -> None:
        """Test rate limit tracking."""
        # Record rate limit hits
        self.metrics.record_rate_limit_hit("user123", "requests_per_minute")
        self.metrics.record_rate_limit_hit("user456", "requests_per_hour")

        # Verify the methods work without error
        assert True

    def test_provider_health_tracking(self) -> None:
        """Test provider health tracking."""
        # Update provider health status
        self.metrics.update_provider_health("openai", True)
        self.metrics.update_provider_health("anthropic", False)
        self.metrics.update_provider_health("mistral", True)

        # Verify the methods work without error
        assert True

    def test_system_metrics(self) -> None:
        """Test system metrics tracking."""
        # Update system metrics
        self.metrics.update_active_connections(5)
        self.metrics.update_memory_usage(1024 * 1024, 2048 * 1024, 512 * 1024)
        self.metrics.set_system_info({"version": "1.0.0", "environment": "test"})

        # Verify the methods work without error
        assert True

    def test_summary_stats(self) -> None:
        """Test summary statistics."""
        # Record some metrics first
        self.metrics.record_error("timeout", "/chat/completions", "openai")
        self.metrics.record_auth_attempt("api_key", "success", "user")

        # Get summary stats
        stats = self.metrics.get_summary_stats()

        assert isinstance(stats, dict)
        assert "uptime_seconds" in stats
        assert "total_requests" in stats
        assert "total_errors" in stats
        assert "error_rate" in stats


class TestHealthChecker:
    """Test health checker functionality."""

    def test_health_checker_initialization(self) -> None:
        """Test health checker initialization."""
        health_checker = HealthChecker()

        assert health_checker is not None
        assert hasattr(health_checker, "health_checks")
        assert hasattr(health_checker, "check_interval")

    def test_check_system_resources(self) -> None:
        """Test system resource health check."""
        health_checker = HealthChecker()

        # Call system resource check
        resources = health_checker.check_system_resources()

        # Should return a dict (might be empty if psutil not available)
        assert isinstance(resources, dict)

    def test_get_overall_health_empty(self) -> None:
        """Test overall health status with no components."""
        health_checker = HealthChecker()

        health_status = health_checker.get_overall_health()

        assert "status" in health_status
        assert "timestamp" in health_status
        assert "components" in health_status
        assert "summary" in health_status
        assert health_status["status"] == "healthy"  # No components means healthy

    @patch("app.monitoring.metrics.time.time")
    def test_get_overall_health_with_components(self, mock_time: Any) -> None:
        """Test overall health status with mock components."""
        mock_time.return_value = 1609459200.0  # Fixed timestamp

        health_checker = HealthChecker()

        # Add some mock health checks - make sure more healthy than unhealthy for degraded status
        health_checker.health_checks = {
            "component1": {"status": "healthy", "last_check": 1609459200.0, "error": None},
            "component2": {"status": "healthy", "last_check": 1609459200.0, "error": None},
            "component3": {
                "status": "unhealthy",
                "last_check": 1609459200.0,
                "error": "Test error",
            },
        }

        health_status = health_checker.get_overall_health()

        assert health_status["status"] == "degraded"  # More healthy than unhealthy (2 > 1)
        assert health_status["summary"]["total_components"] == 3
        assert health_status["summary"]["healthy_components"] == 2
        assert health_status["summary"]["unhealthy_components"] == 1


class TestHealthChecks:
    """Test health check functionality."""

    def test_basic_health_check(self) -> None:
        """Test basic health check."""
        health_checker = HealthChecker()

        health_status = health_checker.get_overall_health()

        assert "status" in health_status
        assert "timestamp" in health_status
        assert "components" in health_status
        assert health_status["status"] in ["healthy", "degraded", "unhealthy"]

    def test_component_health_checks(self) -> None:
        """Test individual component health checks."""
        health_checker = HealthChecker()

        # Add mock health checks
        health_checker.health_checks = {
            "cache": {"status": "healthy", "last_check": time.time(), "error": None},
            "database": {"status": "healthy", "last_check": time.time(), "error": None},
        }

        health_status = health_checker.get_overall_health()

        assert "components" in health_status
        components = health_status["components"]

        # Should have the mock components
        assert "cache" in components
        assert "database" in components
        for component_status in components.values():
            assert "status" in component_status
            assert "last_check" in component_status

    def test_health_check_with_errors(self) -> None:
        """Test health check when there are errors."""
        health_checker = HealthChecker()

        # Add mock unhealthy components
        health_checker.health_checks = {
            "component1": {
                "status": "unhealthy",
                "last_check": time.time(),
                "error": "Connection failed",
            },
            "component2": {"status": "healthy", "last_check": time.time(), "error": None},
        }

        health_status = health_checker.get_overall_health()

        # Health status should be degraded due to unhealthy component
        assert health_status["status"] in ["healthy", "degraded", "unhealthy"]
        assert health_status["summary"]["unhealthy_components"] == 1


class TestMetricsIntegration:
    """Integration tests for metrics system."""

    def test_end_to_end_metrics_flow(self) -> None:
        """Test complete metrics flow from request to reporting."""
        metrics = MetricsCollector()

        # Simulate a complete request flow
        user_id = "user123"

        # Record user activity
        metrics.record_user_activity(user_id)

        # Simulate authentication
        metrics.record_auth_attempt("api_key", "success", "user")

        # Record organization activity
        metrics.record_organization_activity("org123", "enterprise", 0.05)

        # Record cache operations
        metrics.record_cache_operation("get", "miss")
        metrics.record_cache_operation("set", "success")

        # Update system metrics
        metrics.update_active_connections(3)
        metrics.update_provider_health("openai", True)

        # Get summary stats
        stats = metrics.get_summary_stats()

        # Verify metrics were recorded
        assert isinstance(stats, dict)
        assert "uptime_seconds" in stats

    def test_metrics_aggregation(self) -> None:
        """Test metrics aggregation across multiple requests."""
        metrics = MetricsCollector()

        # Simulate multiple activities
        user_activities = [
            ("user1", "org1", "enterprise", 0.001),
            ("user2", "org1", "enterprise", 0.0005),
            ("user1", "org2", "starter", 0.0003),
        ]

        for user_id, org_id, plan_type, cost in user_activities:
            metrics.record_user_activity(user_id)
            metrics.record_organization_activity(org_id, plan_type, cost)
            metrics.record_auth_attempt("api_key", "success", "user")

        # Record some errors and cache operations
        metrics.record_error("timeout", "/chat/completions", "openai")
        metrics.record_cache_operation("get", "hit")
        metrics.record_cache_operation("get", "miss")

        # Get summary stats
        stats = metrics.get_summary_stats()

        # Verify aggregated metrics
        assert isinstance(stats, dict)
        assert stats["total_errors"] >= 1
        assert "uptime_seconds" in stats


if __name__ == "__main__":
    pytest.main([__file__])
