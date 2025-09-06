#
# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""
Tests for core exception classes.
"""

import pytest

from app.core.exceptions import (AuthenticationError, BudgetExceededError,
                                 CacheError, ClassificationError,
                                 ConfigurationError, ModelMuxerError,
                                 ModelNotFoundError, NoProvidersAvailableError,
                                 ProviderError, QuotaExceededError,
                                 RateLimitError, RouterConfigurationError,
                                 RoutingError)
from app.core.exceptions import TimeoutError as MuxerTimeoutError
from app.core.exceptions import ValidationError


class TestModelMuxerError:
    """Test the base ModelMuxerError class."""

    def test_basic_exception(self) -> None:
        """Test basic exception creation."""
        error = ModelMuxerError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.error_code is None
        assert error.details == {}

    def test_exception_with_code_and_details(self) -> None:
        """Test exception with error code and details."""
        details = {"key": "value"}
        error = ModelMuxerError("Test error", error_code="test_code", details=details)
        assert error.error_code == "test_code"
        assert error.details == details

    def test_to_dict(self) -> None:
        """Test converting exception to dictionary."""
        error = ModelMuxerError("Test error", error_code="test_code", details={"key": "value"})
        result = error.to_dict()

        expected = {
            "error": {
                "message": "Test error",
                "type": "ModelMuxerError",
                "code": "test_code",
                "details": {"key": "value"},
            }
        }
        assert result == expected


class TestProviderError:
    """Test ProviderError exception."""

    def test_basic_provider_error(self) -> None:
        """Test basic provider error."""
        error = ProviderError("Provider failed", provider="openai")
        assert error.provider == "openai"
        assert error.status_code is None
        assert error.details["provider"] == "openai"

    def test_provider_error_with_status(self) -> None:
        """Test provider error with status code."""
        error = ProviderError("Provider failed", provider="openai", status_code=500)
        assert error.status_code == 500
        assert error.details["status_code"] == 500


class TestRoutingError:
    """Test RoutingError exception."""

    def test_routing_error(self) -> None:
        """Test routing error with strategy."""
        error = RoutingError("Routing failed", routing_strategy="heuristic")
        assert error.routing_strategy == "heuristic"
        assert error.details["routing_strategy"] == "heuristic"


class TestRouterConfigurationError:
    """Test RouterConfigurationError exception."""

    def test_router_config_error(self) -> None:
        """Test router configuration error."""
        error = RouterConfigurationError("Configuration invalid")
        assert error.error_code == "router_configuration_error"


class TestAuthenticationError:
    """Test AuthenticationError exception."""

    def test_auth_error(self) -> None:
        """Test authentication error."""
        error = AuthenticationError()
        assert str(error) == "Authentication failed"
        assert error.error_code == "authentication_error"

    def test_auth_error_custom_message(self) -> None:
        """Test authentication error with custom message."""
        error = AuthenticationError("Custom auth failure")
        assert str(error) == "Custom auth failure"


class TestRateLimitError:
    """Test RateLimitError exception."""

    def test_rate_limit_error(self) -> None:
        """Test rate limit error."""
        error = RateLimitError(retry_after=60)
        assert error.retry_after == 60
        assert error.details["retry_after"] == 60
        assert error.error_code == "rate_limit_exceeded"


class TestBudgetExceededError:
    """Test BudgetExceededError exception."""

    def test_budget_error(self) -> None:
        """Test budget exceeded error."""
        estimates = [("model1", 0.5), ("model2", None)]
        error = BudgetExceededError(
            "Budget exceeded", limit=1.0, estimates=estimates, reason="cost_too_high"
        )
        assert error.limit == 1.0
        assert error.estimates == estimates
        assert error.reason == "cost_too_high"
        assert error.details["limit"] == 1.0
        assert error.details["estimates"] == estimates
        assert error.details["reason"] == "cost_too_high"


class TestConfigurationError:
    """Test ConfigurationError exception."""

    def test_config_error(self) -> None:
        """Test configuration error."""
        error = ConfigurationError("Invalid config", config_key="api_key")
        assert error.config_key == "api_key"
        assert error.details["config_key"] == "api_key"
        assert error.error_code == "configuration_error"


class TestCacheError:
    """Test CacheError exception."""

    def test_cache_error(self) -> None:
        """Test cache error."""
        error = CacheError("Cache failed", cache_type="redis")
        assert error.cache_type == "redis"
        assert error.details["cache_type"] == "redis"
        assert error.error_code == "cache_error"


class TestClassificationError:
    """Test ClassificationError exception."""

    def test_classification_error(self) -> None:
        """Test classification error."""
        error = ClassificationError("Classification failed", classifier_type="intent")
        assert error.classifier_type == "intent"
        assert error.details["classifier_type"] == "intent"
        assert error.error_code == "classification_error"


class TestValidationError:
    """Test ValidationError exception."""

    def test_validation_error(self) -> None:
        """Test validation error."""
        error = ValidationError("Invalid input", field="messages")
        assert error.field == "messages"
        assert error.details["field"] == "messages"
        assert error.error_code == "validation_error"


class TestTimeoutError:
    """Test TimeoutError exception."""

    def test_timeout_error(self) -> None:
        """Test timeout error."""
        error = MuxerTimeoutError(timeout_duration=30.0)
        assert error.timeout_duration == 30.0
        assert error.details["timeout_duration"] == 30.0
        assert error.error_code == "timeout_error"


class TestModelNotFoundError:
    """Test ModelNotFoundError exception."""

    def test_model_not_found_error(self) -> None:
        """Test model not found error."""
        error = ModelNotFoundError("Model not found", model="gpt-5")
        assert error.model == "gpt-5"
        assert error.details["model"] == "gpt-5"
        assert error.error_code == "model_not_found"


class TestQuotaExceededError:
    """Test QuotaExceededError exception."""

    def test_quota_exceeded_error(self) -> None:
        """Test quota exceeded error."""
        error = QuotaExceededError(provider="openai")
        assert error.provider == "openai"
        assert error.error_code == "quota_exceeded"


class TestNoProvidersAvailableError:
    """Test NoProvidersAvailableError exception."""

    def test_no_providers_error(self) -> None:
        """Test no providers available error."""
        error = NoProvidersAvailableError()
        assert str(error) == "No LLM providers available"
        assert error.error_code == "no_providers_available"

    def test_no_providers_error_custom_message(self) -> None:
        """Test no providers error with custom message."""
        error = NoProvidersAvailableError("Custom no providers message")
        assert str(error) == "Custom no providers message"
