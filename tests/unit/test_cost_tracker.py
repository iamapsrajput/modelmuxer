# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Comprehensive tests for cost_tracker.py to improve coverage.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.cost_tracker import (
    AdvancedCostTracker,
    CostTracker,
    create_advanced_cost_tracker,
    cost_tracker,
)
from app.models import ChatMessage


class TestCostTracker:
    """Test basic CostTracker functionality."""

    def test_init(self):
        """Test CostTracker initialization."""
        tracker = CostTracker()
        assert tracker.pricing is not None
        assert tracker._tokenizers is not None

    def test_estimate_request_cost_with_messages(self):
        """Test cost estimation with chat messages."""
        tracker = CostTracker()
        messages = [
            ChatMessage(role="user", content="Hello, how are you?", name=None),
            ChatMessage(role="assistant", content="I'm doing well, thank you!", name=None),
        ]

        estimate = tracker.estimate_request_cost(
            messages=messages, provider="openai", model="gpt-3.5-turbo", max_tokens=100
        )

        assert estimate["estimated_cost"] >= 0
        assert estimate["input_tokens"] > 0
        assert estimate["estimated_output_tokens"] > 0

    def test_estimate_request_cost_unknown_model(self):
        """Test cost estimation with unknown model."""
        tracker = CostTracker()
        messages = [ChatMessage(role="user", content="Test prompt", name=None)]

        estimate = tracker.estimate_request_cost(
            messages=messages, provider="unknown", model="unknown-model", max_tokens=100
        )

        # Should return 0 for unknown models
        assert estimate["estimated_cost"] == 0.0

    def test_calculate_cost(self):
        """Test cost calculation."""
        tracker = CostTracker()

        cost = tracker.calculate_cost(
            provider="openai", model="gpt-3.5-turbo", input_tokens=100, output_tokens=50
        )

        assert cost >= 0
        assert isinstance(cost, float)

    def test_calculate_cost_unknown_model(self):
        """Test cost calculation with unknown model."""
        tracker = CostTracker()

        cost = tracker.calculate_cost(
            provider="unknown", model="unknown-model", input_tokens=100, output_tokens=50
        )

        # Should return 0 for unknown models
        assert cost == 0.0

    def test_count_tokens_openai(self):
        """Test token counting for OpenAI models."""
        tracker = CostTracker()
        messages = [ChatMessage(role="user", content="Hello world", name=None)]

        with patch("tiktoken.encoding_for_model") as mock_encoding:
            mock_encoder = Mock()
            mock_encoder.encode.return_value = [1, 2, 3, 4, 5]
            mock_encoding.return_value = mock_encoder

            count = tracker.count_tokens(messages, "openai", "gpt-3.5-turbo")
            assert count > 0

    def test_count_tokens_anthropic(self):
        """Test token counting for Anthropic models."""
        tracker = CostTracker()
        messages = [ChatMessage(role="user", content="Hello world", name=None)]

        with patch("tiktoken.encoding_for_model") as mock_encoding:
            mock_encoder = Mock()
            mock_encoder.encode.return_value = [1, 2, 3, 4, 5]
            mock_encoding.return_value = mock_encoder

            count = tracker.count_tokens(messages, "anthropic", "claude-3-sonnet")
            assert count > 0

    def test_count_tokens_other_providers(self):
        """Test token counting for other providers."""
        tracker = CostTracker()
        messages = [ChatMessage(role="user", content="Hello world from test", name=None)]

        with patch("tiktoken.get_encoding") as mock_encoding:
            mock_encoder = Mock()
            mock_encoder.encode.return_value = [1, 2, 3, 4, 5]
            mock_encoding.return_value = mock_encoder

            count = tracker.count_tokens(messages, "mistral", "mistral-large")
            assert count > 0

    def test_get_cheapest_model_for_task(self):
        """Test getting cheapest model for task."""
        tracker = CostTracker()

        result = tracker.get_cheapest_model_for_task("simple")
        assert "provider" in result
        assert "model" in result

        result = tracker.get_cheapest_model_for_task("code")
        assert "provider" in result
        assert "model" in result

        result = tracker.get_cheapest_model_for_task("complex")
        assert "provider" in result
        assert "model" in result

    def test_compare_model_costs(self):
        """Test comparing costs across models."""
        tracker = CostTracker()
        messages = [ChatMessage(role="user", content="Test message", name=None)]

        models = [
            {"provider": "openai", "model": "gpt-3.5-turbo"},
            {"provider": "openai", "model": "gpt-4o"},
        ]

        comparisons = tracker.compare_model_costs(messages, models, max_tokens=100)
        assert len(comparisons) == 2
        assert comparisons[0]["estimated_cost"] <= comparisons[1]["estimated_cost"]

    def test_get_model_info(self):
        """Test getting model information."""
        tracker = CostTracker()

        info = tracker.get_model_info("openai", "gpt-3.5-turbo")
        assert "provider" in info
        assert "model" in info
        assert "input_price_per_million" in info
        assert "output_price_per_million" in info

        # Test unknown model
        info = tracker.get_model_info("unknown", "unknown-model")
        assert info == {}

    def test_estimate_output_tokens(self):
        """Test output token estimation."""
        tracker = CostTracker()

        tokens = tracker.estimate_output_tokens(max_tokens=500)
        assert tokens == 500

        tokens = tracker.estimate_output_tokens(max_tokens=2000)
        assert tokens == 1000  # Capped at 1000

        tokens = tracker.estimate_output_tokens(max_tokens=None)
        assert tokens > 0


class TestAdvancedCostTracker:
    """Test AdvancedCostTracker functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def tracker(self, temp_db):
        """Create an AdvancedCostTracker instance."""
        tracker = AdvancedCostTracker(db_path=temp_db)
        return tracker

    def test_initialize(self, temp_db):
        """Test AdvancedCostTracker initialization."""
        tracker = AdvancedCostTracker(db_path=temp_db)

        assert tracker.db_path == temp_db
        assert hasattr(tracker, "redis_client")

    def test_initialize_with_redis(self, temp_db):
        """Test AdvancedCostTracker initialization with Redis."""
        with patch("redis.from_url") as mock_redis:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client

            tracker = AdvancedCostTracker(db_path=temp_db, redis_url="redis://localhost:6379")

            assert tracker.redis_client is not None

    async def test_log_request_with_cascade(self, tracker):
        """Test logging a cascade request."""
        cascade_metadata = {
            "type": "quality",
            "steps": [
                {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "cost": 0.001,
                    "success": True,
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                }
            ],
            "final_quality_score": 0.85,
        }

        await tracker.log_request_with_cascade(
            user_id="test-user",
            session_id="test-session",
            cascade_metadata=cascade_metadata,
            success=True,
        )

        # Should not raise any exceptions
        assert True

    async def test_log_simple_request(self, tracker):
        """Test logging a simple request."""
        await tracker.log_simple_request(
            user_id="test-user",
            session_id="test-session",
            provider="openai",
            model="gpt-3.5-turbo",
            cost=0.001,
            success=True,
            prompt_tokens=10,
            completion_tokens=8,
        )

        # Should not raise any exceptions
        assert True

    async def test_set_budget(self, tracker):
        """Test setting budget limits."""
        await tracker.set_budget(
            user_id="test-user",
            budget_type="monthly",
            budget_limit=100.0,
            provider=None,
            model=None,
            alert_thresholds=[50, 80, 95],
        )

        # Verify budget was set
        status = await tracker.get_budget_status("test-user")
        assert len(status) > 0
        assert status[0]["budget_type"] == "monthly"
        assert status[0]["budget_limit"] == 100.0

    async def test_set_budget_with_provider(self, tracker):
        """Test setting budget with specific provider."""
        await tracker.set_budget(
            user_id="test-user",
            budget_type="daily",
            budget_limit=10.0,
            provider="openai",
            model=None,
            alert_thresholds=[50, 80],
        )

        status = await tracker.get_budget_status("test-user", "daily")
        assert len(status) > 0
        assert status[0]["provider"] == "openai"

    async def test_get_budget_status(self, tracker):
        """Test getting budget status."""
        # Set a budget
        await tracker.set_budget(user_id="test-user", budget_type="monthly", budget_limit=100.0)

        # Log some usage
        await tracker.log_simple_request(
            user_id="test-user",
            session_id="test-session",
            provider="openai",
            model="gpt-3.5-turbo",
            cost=25.0,
            success=True,
        )

        status = await tracker.get_budget_status("test-user")
        assert len(status) > 0
        assert status[0]["current_usage"] >= 25.0
        assert status[0]["usage_percentage"] >= 25.0

    async def test_update_usage_cache(self, tracker):
        """Test updating usage cache."""
        # This is a private method but we can test it doesn't crash
        await tracker._update_usage_cache(
            user_id="test-user", cost=0.001, provider="openai", model="gpt-3.5-turbo"
        )
        assert True

    async def test_check_budget_alerts(self, tracker):
        """Test checking budget alerts."""
        # This is a private method but we can test it doesn't crash
        await tracker._check_budget_alerts(
            user_id="test-user", cost=0.001, provider="openai", model="gpt-3.5-turbo"
        )
        assert True

    async def test_get_current_usage(self, tracker):
        """Test getting current usage."""
        # Log some usage first
        await tracker.log_simple_request(
            user_id="test-user",
            session_id="test-session",
            provider="openai",
            model="gpt-3.5-turbo",
            cost=1.0,
            success=True,
        )

        usage = await tracker._get_current_usage(
            user_id="test-user", budget_type="daily", provider=None, model=None
        )

        assert usage >= 1.0

    def test_get_budget_period_dates(self, tracker):
        """Test getting budget period dates."""
        start, end = tracker._get_budget_period_dates("daily")
        assert start is not None
        assert end is not None

        start, end = tracker._get_budget_period_dates("weekly")
        assert start is not None
        assert end is not None

        start, end = tracker._get_budget_period_dates("monthly")
        assert start is not None
        assert end is not None

        start, end = tracker._get_budget_period_dates("yearly")
        assert start is not None
        assert end is not None

    def test_enhanced_features_available(self):
        """Test that enhanced features flag is set correctly."""
        from app.cost_tracker import ENHANCED_FEATURES_AVAILABLE

        # Should be True or False, not None
        assert ENHANCED_FEATURES_AVAILABLE is not None


class TestCreateAdvancedCostTracker:
    """Test create_advanced_cost_tracker function."""

    def test_create_advanced_cost_tracker(self):
        """Test creating an advanced cost tracker."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            tracker = create_advanced_cost_tracker(
                db_path=db_path, redis_url="redis://localhost:6379/0"
            )

            assert isinstance(tracker, AdvancedCostTracker)
            assert tracker.db_path == db_path
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestGlobalCostTracker:
    """Test the global cost_tracker instance."""

    def test_global_cost_tracker_exists(self):
        """Test that the global cost_tracker is initialized."""
        assert cost_tracker is not None
        assert isinstance(cost_tracker, CostTracker)

    def test_global_cost_tracker_functionality(self):
        """Test that the global cost_tracker works."""
        messages = [ChatMessage(role="user", content="Test prompt", name=None)]

        estimate = cost_tracker.estimate_request_cost(
            messages=messages, provider="openai", model="gpt-3.5-turbo", max_tokens=100
        )

        # Should return a cost estimate
        assert isinstance(estimate, dict)
        assert "estimated_cost" in estimate
