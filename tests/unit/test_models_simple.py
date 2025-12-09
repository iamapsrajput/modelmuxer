# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""Simple tests for models.py to improve coverage."""

import pytest
from unittest.mock import patch
from datetime import datetime

from app.models import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
    Choice,
    RouterMetadata,
    UserStats,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ChatResponse,
    BudgetPeriodEnum,
    BudgetRequest,
    CascadeConfig,
    EnhancedChatCompletionRequest,
    RoutingMetadata,
    EnhancedChatCompletionResponse,
    BudgetAlert,
    BudgetStatus,
    BudgetResponse,
)


class TestModels:
    """Test model classes."""

    def test_chat_message_basic(self):
        """Test ChatMessage creation."""
        msg = ChatMessage(role="user", content="Hello", name="TestUser")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name == "TestUser"

    def test_chat_completion_request(self):
        """Test ChatCompletionRequest."""
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Test", name="User")],
            model="gpt-3.5-turbo",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            n=1,
            stream=False,
            stop=["END"],
            presence_penalty=0.0,
            frequency_penalty=0.0,
            logit_bias={},
            user="test-user",
            region="us-east",
        )
        assert request.model == "gpt-3.5-turbo"
        assert len(request.messages) == 1

    def test_usage_model(self):
        """Test Usage model."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.total_tokens == 30

    def test_choice_model(self):
        """Test Choice model."""
        choice = Choice(
            index=0,
            message=ChatMessage(role="assistant", content="Response", name="Bot"),
            finish_reason="stop",
        )
        assert choice.index == 0

    def test_router_metadata(self):
        """Test RouterMetadata model."""
        metadata = RouterMetadata(
            selected_provider="openai",
            selected_model="gpt-3.5-turbo",
            routing_reason="heuristic match",
            estimated_cost=0.002,
            response_time_ms=150.5,
            intent_label="general",
            intent_confidence=0.95,
            intent_signals={"complexity": 0.3},
            estimated_cost_usd=0.002,
            estimated_eta_ms=200,
            estimated_tokens_in=50,
            estimated_tokens_out=100,
            direct_providers_only=False,
        )
        assert metadata.selected_provider == "openai"
        assert metadata.response_time_ms == 150.5

    def test_chat_response(self):
        """Test ChatResponse model."""
        response = ChatResponse(
            id="chat-123",
            object="chat.completion",
            created=1234567890,
            model="gpt-3.5-turbo",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
        assert response.id == "chat-123"

    def test_chat_completion_response(self):
        """Test ChatCompletionResponse model."""
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="gpt-3.5-turbo",
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hello", name="Bot"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
            router_metadata=RouterMetadata(
                selected_provider="openai",
                selected_model="gpt-3.5-turbo",
                routing_reason="default",
                estimated_cost=0.001,
                response_time_ms=100,
            ),
        )
        assert response.id == "chatcmpl-123"

    def test_health_response(self):
        """Test HealthResponse model."""
        health = HealthResponse(status="healthy", version="1.0.0", timestamp=datetime.now())
        assert health.status == "healthy"

    def test_user_stats(self):
        """Test UserStats model."""
        stats = UserStats(
            user_id="user-123",
            total_requests=100,
            total_cost=1.5,
            daily_cost=0.5,
            monthly_cost=1.5,
            daily_budget=10.0,
            monthly_budget=100.0,
            favorite_model="gpt-3.5-turbo",
        )
        assert stats.total_requests == 100

    def test_metrics_response(self):
        """Test MetricsResponse model."""
        metrics = MetricsResponse(
            total_requests=1000,
            total_cost=50.0,
            active_users=10,
            provider_usage={"openai": 800, "anthropic": 200},
            model_usage={"gpt-3.5-turbo": 800, "claude-2": 200},
            average_response_time=200.5,
        )
        assert metrics.total_requests == 1000

    def test_error_response(self):
        """Test ErrorResponse model."""
        error = ErrorResponse(
            error={
                "message": "Invalid API key",
                "type": "authentication_error",
                "code": "invalid_api_key",
            }
        )
        assert error.error["type"] == "authentication_error"

    def test_error_response_create(self):
        """Test ErrorResponse.create method."""
        error = ErrorResponse.create(
            message="Test error",
            error_type="test_error",
            code="TEST001",
            details={"field": "value"},
        )
        assert error.error["message"] == "Test error"
        assert error.error["code"] == "TEST001"

    def test_budget_period_enum(self):
        """Test BudgetPeriodEnum."""
        assert BudgetPeriodEnum.daily == "daily"
        assert BudgetPeriodEnum.weekly == "weekly"
        assert BudgetPeriodEnum.monthly == "monthly"
        assert BudgetPeriodEnum.yearly == "yearly"

    def test_budget_request(self):
        """Test BudgetRequest model."""
        request = BudgetRequest(
            budget_type=BudgetPeriodEnum.monthly,
            budget_limit=100.0,
            provider="openai",
            model="gpt-4",
            alert_thresholds=[50.0, 80.0, 95.0],
        )
        assert request.budget_limit == 100.0

    def test_budget_request_validation(self):
        """Test BudgetRequest threshold validation."""
        with pytest.raises(ValueError):
            BudgetRequest(
                budget_type=BudgetPeriodEnum.daily,
                budget_limit=10.0,
                alert_thresholds=[50.0, 150.0],  # Invalid: > 100
            )

    def test_cascade_config(self):
        """Test CascadeConfig model."""
        config = CascadeConfig(
            cascade_type="balanced", max_budget=0.5, quality_threshold=0.8, confidence_threshold=0.7
        )
        assert config.cascade_type == "balanced"

    def test_enhanced_chat_completion_request(self):
        """Test EnhancedChatCompletionRequest."""
        request = EnhancedChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Test", name="User")],
            session_id="session-123",
            cascade_config=CascadeConfig(),
            enable_analytics=True,
            routing_preference="cost",
            # Required fields from parent
            model="gpt-3.5-turbo",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            n=1,
            stream=False,
            stop=None,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            logit_bias=None,
            user="test-user",
            region=None,
        )
        assert request.session_id == "session-123"

    def test_routing_metadata(self):
        """Test RoutingMetadata model."""
        metadata = RoutingMetadata(
            strategy_used="cascade",
            total_cost=0.005,
            cascade_steps=2,
            quality_score=0.9,
            confidence_score=0.85,
            provider_chain=["groq", "openai"],
            escalation_reasons=["quality threshold"],
            response_time_ms=250.5,
        )
        assert metadata.strategy_used == "cascade"

    def test_enhanced_chat_completion_response(self):
        """Test EnhancedChatCompletionResponse."""
        response = EnhancedChatCompletionResponse(
            id="enhanced-123",
            object="chat.completion",
            created=1234567890,
            model="gpt-4",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            routing_metadata=RoutingMetadata(
                strategy_used="cascade", total_cost=0.01, response_time_ms=300
            ),
        )
        assert response.id == "enhanced-123"

    def test_budget_alert(self):
        """Test BudgetAlert model."""
        alert = BudgetAlert(
            type="warning", message="Budget 80% consumed", threshold=80.0, current_usage=82.5
        )
        assert alert.type == "warning"

    def test_budget_status(self):
        """Test BudgetStatus model."""
        status = BudgetStatus(
            budget_type=BudgetPeriodEnum.monthly,
            budget_limit=100.0,
            current_usage=50.0,
            usage_percentage=50.0,
            remaining_budget=50.0,
            provider="openai",
            model="gpt-4",
            alerts=[
                BudgetAlert(type="warning", message="50% used", threshold=50.0, current_usage=50.0)
            ],
            period_start="2024-01-01",
            period_end="2024-01-31",
        )
        assert status.budget_limit == 100.0

    def test_budget_response(self):
        """Test BudgetResponse model."""
        response = BudgetResponse(
            message="Budget status retrieved",
            budgets=[
                BudgetStatus(
                    budget_type=BudgetPeriodEnum.daily,
                    budget_limit=10.0,
                    current_usage=5.0,
                    usage_percentage=50.0,
                    remaining_budget=5.0,
                    period_start="2024-01-01",
                    period_end="2024-01-01",
                )
            ],
            total_budgets=1,
        )
        assert response.total_budgets == 1


class TestSettingsImports:
    """Test that settings can be imported and used."""

    def test_settings_import(self):
        """Test that settings can be imported."""
        from app.settings import Settings, settings

        assert settings is not None
        assert isinstance(settings, Settings)

    def test_settings_api_keys(self):
        """Test settings API keys configuration."""
        from app.settings import settings

        # Test get_allowed_api_keys method
        keys = settings.get_allowed_api_keys()
        assert isinstance(keys, list)
        # Should return default test keys when not configured
        assert "test-api-key" in keys

    def test_get_provider_pricing(self):
        """Test get_provider_pricing helper function."""
        from app.settings import get_provider_pricing

        pricing = get_provider_pricing()
        assert isinstance(pricing, dict)
        assert "openai" in pricing
        assert "gpt-3.5-turbo" in pricing["openai"]
        assert "input" in pricing["openai"]["gpt-3.5-turbo"]
        assert "output" in pricing["openai"]["gpt-3.5-turbo"]
