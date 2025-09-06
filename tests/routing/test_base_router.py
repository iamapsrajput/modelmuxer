# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

from unittest.mock import AsyncMock, Mock

import pytest

from app.models import ChatMessage
from app.routing.base_router import BaseRouter


class MockRouter(BaseRouter):
    """Mock router for testing BaseRouter functionality."""

    async def analyze_prompt(self, messages):
        return {"intent": "test", "confidence": 0.8}

    async def _route_request(self, messages, analysis, user_id, constraints):
        return ("openai", "gpt-4", "Test selection", 0.9)


class TestBaseRouter:
    """Test suite for BaseRouter."""

    @pytest.fixture
    def router(self):
        """Create BaseRouter instance for testing."""
        return MockRouter("test-router")

    def test_init(self, router):
        """Test router initialization."""
        assert router.name == "test-router"
        assert router.config == {}
        assert router.metrics == {
            "total_requests": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "average_response_time": 0.0,
        }

    def test_init_with_config(self):
        """Test router initialization with config."""
        config = {"key": "value"}
        router = MockRouter("test-router", config)
        assert router.config == config

    def test_get_metrics(self, router):
        """Test getting metrics."""
        metrics = router.get_metrics()
        expected_keys = {
            "router_name",
            "metrics",
            "success_rate",
        }
        assert set(metrics.keys()) == expected_keys
        assert isinstance(metrics["router_name"], str)
        assert isinstance(metrics["success_rate"], float)
        assert isinstance(metrics["metrics"], dict)
        inner_metrics = metrics["metrics"]
        expected_inner_keys = {
            "total_requests",
            "successful_routes",
            "failed_routes",
            "average_response_time",
        }
        assert set(inner_metrics.keys()) == expected_inner_keys

    def test_reset_metrics(self, router):
        """Test resetting metrics."""
        # Update some metrics
        router.metrics["requests_total"] = 10
        router.metrics["success_total"] = 5
        router.metrics["total_response_time"] = 100.0

        router.reset_metrics()

        assert router.metrics["total_requests"] == 0
        assert router.metrics["successful_routes"] == 0
        assert router.metrics["failed_routes"] == 0
        assert router.metrics["average_response_time"] == 0.0

    def test_update_metrics_success(self, router):
        """Test updating metrics for successful request."""
        router.metrics["total_requests"] = 1  # Simulate increment from select_provider_and_model
        router._update_metrics(1.5, True)

        assert router.metrics["total_requests"] == 1
        assert router.metrics["successful_routes"] == 1
        assert router.metrics["failed_routes"] == 0
        assert router.metrics["average_response_time"] == 1.5

    def test_update_metrics_error(self, router):
        """Test updating metrics for failed request."""
        router.metrics["total_requests"] = 1
        router._update_metrics(2.0, False)

        assert router.metrics["total_requests"] == 1
        assert router.metrics["successful_routes"] == 0
        assert router.metrics["failed_routes"] == 1
        assert router.metrics["average_response_time"] == 2.0

    def test_update_metrics_multiple(self, router):
        """Test updating metrics for multiple requests."""
        router.metrics["total_requests"] = 1
        router._update_metrics(1.0, True)
        router.metrics["total_requests"] = 2
        router._update_metrics(2.0, False)
        router.metrics["total_requests"] = 3
        router._update_metrics(3.0, True)

        assert router.metrics["total_requests"] == 3
        assert router.metrics["successful_routes"] == 2
        assert router.metrics["failed_routes"] == 1
        assert router.metrics["average_response_time"] == 2.0

    def test_apply_constraints_no_constraints(self, router):
        """Test applying constraints with no constraints."""
        analysis = {"intent": "test"}
        result = router._apply_constraints(analysis, {})

        assert result == analysis

    def test_apply_constraints_with_budget(self, router):
        """Test applying constraints with budget."""
        analysis = {"intent": "test"}
        constraints = {"max_cost": 1.0}
        result = router._apply_constraints(analysis, constraints)

        assert result == {"intent": "test", "max_cost": 1.0}

    def test_apply_constraints_with_region(self, router):
        """Test applying constraints with region."""
        analysis = {"intent": "test"}
        constraints = {"region": "us"}
        result = router._apply_constraints(analysis, constraints)

        assert result == analysis  # No change since region not handled

    def test_apply_constraints_multiple_constraints(self, router):
        """Test applying constraints with multiple constraints."""
        analysis = {"intent": "test"}
        constraints = {"max_cost": 1.0, "region": "us", "tenant_id": "test"}
        result = router._apply_constraints(analysis, constraints)

        assert result == {"intent": "test", "max_cost": 1.0}

    @pytest.mark.asyncio
    async def test_select_provider_and_model_abstract(self, router):
        """Test select_provider_and_model with mock implementation."""
        messages = [ChatMessage(role="user", content="Hello")]

        provider, model, reasoning, confidence = await router.select_provider_and_model(messages)

        assert provider == "openai"
        assert model == "gpt-4"
        assert reasoning == "Test selection"
        assert confidence == 0.9

    @pytest.mark.asyncio
    async def test_select_provider_and_model_with_constraints(self, router):
        """Test select_provider_and_model with constraints."""
        messages = [ChatMessage(role="user", content="Hello")]
        constraints = {"budget_constraint": 1.0, "region": "us"}

        provider, model, reasoning, confidence = await router.select_provider_and_model(
            messages, constraints=constraints
        )

        assert provider == "openai"
        assert model == "gpt-4"
        assert reasoning == "Test selection"
        assert confidence == 0.9
