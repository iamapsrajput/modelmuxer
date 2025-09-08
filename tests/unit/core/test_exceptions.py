# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""Unit tests for core exceptions."""

import pytest
from app.core.exceptions import (
    ModelMuxerError,
    ProviderError,
    RoutingError,
    RouterConfigurationError,
    AuthenticationError,
    RateLimitError,
    BudgetExceededError,
    ConfigurationError,
    CacheError,
    ClassificationError,
    ValidationError,
    TimeoutError as MuxerTimeoutError,
    ModelNotFoundError,
    QuotaExceededError,
    NoProvidersAvailableError,
)


class TestModelMuxerError:
    """Test base ModelMuxer error."""

    def test_base_error_message(self):
        """Test base error with message."""
        error = ModelMuxerError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.error_code is None
        assert error.details == {}

    def test_base_error_with_code(self):
        """Test base error with error code."""
        error = ModelMuxerError("Test error", error_code="TEST_ERROR")
        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"

    def test_base_error_with_details(self):
        """Test base error with details."""
        details = {"key": "value", "number": 42}
        error = ModelMuxerError("Test error", details=details)
        assert error.details == details

    def test_base_error_to_dict(self):
        """Test converting error to dictionary."""
        error = ModelMuxerError("Test error", error_code="TEST_ERROR", details={"key": "value"})

        result = error.to_dict()
        assert result["error"]["message"] == "Test error"
        assert result["error"]["type"] == "ModelMuxerError"
        assert result["error"]["code"] == "TEST_ERROR"
        assert result["error"]["details"] == {"key": "value"}


class TestProviderError:
    """Test provider-related errors."""

    def test_provider_error(self):
        """Test provider error creation."""
        error = ProviderError("Provider unavailable", provider="openai", status_code=500)
        assert str(error) == "Provider unavailable"
        assert error.provider == "openai"
        assert error.status_code == 500
        assert error.details["provider"] == "openai"
        assert error.details["status_code"] == 500

    def test_provider_error_without_details(self):
        """Test provider error without provider details."""
        error = ProviderError("General provider error")
        assert str(error) == "General provider error"
        assert error.provider is None
        assert error.status_code is None

    def test_provider_error_with_custom_details(self):
        """Test provider error with custom details."""
        error = ProviderError("API error", provider="anthropic", details={"retry_after": 60})
        assert error.details["provider"] == "anthropic"
        assert error.details["retry_after"] == 60


class TestRoutingError:
    """Test routing errors."""

    def test_routing_error(self):
        """Test routing error creation."""
        error = RoutingError("Routing failed", routing_strategy="heuristic")
        assert str(error) == "Routing failed"
        assert error.routing_strategy == "heuristic"
        assert error.details["routing_strategy"] == "heuristic"

    def test_routing_error_with_details(self):
        """Test routing error with additional details."""
        error = RoutingError(
            "No suitable model", routing_strategy="cascade", details={"attempts": 3}
        )
        assert error.details["routing_strategy"] == "cascade"
        assert error.details["attempts"] == 3


class TestRouterConfigurationError:
    """Test router configuration errors."""

    def test_router_configuration_error(self):
        """Test router configuration error creation."""
        error = RouterConfigurationError("Invalid router config")
        assert str(error) == "Invalid router config"
        assert error.error_code == "router_configuration_error"

    def test_router_configuration_error_with_details(self):
        """Test router configuration error with details."""
        error = RouterConfigurationError(
            "Missing required field", details={"field": "provider_preferences"}
        )
        assert error.details["field"] == "provider_preferences"


class TestAuthenticationError:
    """Test authentication errors."""

    def test_authentication_error_default(self):
        """Test authentication error with default message."""
        error = AuthenticationError()
        assert str(error) == "Authentication failed"
        assert error.error_code == "authentication_error"

    def test_authentication_error_custom(self):
        """Test authentication error with custom message."""
        error = AuthenticationError("Invalid API key")
        assert str(error) == "Invalid API key"

    def test_authentication_error_with_details(self):
        """Test authentication error with details."""
        error = AuthenticationError(
            "Token expired", details={"token_type": "bearer", "expired_at": "2024-01-01"}
        )
        assert error.details["token_type"] == "bearer"


class TestRateLimitError:
    """Test rate limit errors."""

    def test_rate_limit_error_default(self):
        """Test rate limit error with default message."""
        error = RateLimitError()
        assert str(error) == "Rate limit exceeded"
        assert error.error_code == "rate_limit_exceeded"

    def test_rate_limit_error_with_retry(self):
        """Test rate limit error with retry_after."""
        error = RateLimitError("Too many requests", retry_after=60)
        assert str(error) == "Too many requests"
        assert error.retry_after == 60
        assert error.details["retry_after"] == 60

    def test_rate_limit_error_with_details(self):
        """Test rate limit error with additional details."""
        error = RateLimitError(retry_after=30, details={"limit": 100, "window": "1h"})
        assert error.details["retry_after"] == 30
        assert error.details["limit"] == 100
        assert error.details["window"] == "1h"


class TestBudgetExceededError:
    """Test budget exceeded errors."""

    def test_budget_exceeded_default(self):
        """Test budget exceeded error with default message."""
        error = BudgetExceededError()
        assert str(error) == "Budget limit exceeded"
        assert error.error_code == "budget_exceeded"
        assert error.estimates == []

    def test_budget_exceeded_with_limit(self):
        """Test budget exceeded error with limit."""
        error = BudgetExceededError(limit=100.0, reason="Monthly limit reached")
        assert error.limit == 100.0
        assert error.reason == "Monthly limit reached"
        assert error.details["limit"] == 100.0
        assert error.details["reason"] == "Monthly limit reached"

    def test_budget_exceeded_with_estimates(self):
        """Test budget exceeded error with cost estimates."""
        estimates = [("openai/gpt-4", 0.05), ("anthropic/claude-3", 0.03)]
        error = BudgetExceededError("Request would exceed budget", limit=0.10, estimates=estimates)
        assert error.estimates == estimates
        assert error.details["estimates"] == estimates

    def test_budget_exceeded_with_none_estimates(self):
        """Test budget exceeded with None in estimates (unpriced models)."""
        estimates = [("custom/model", None), ("openai/gpt-4", 0.05)]
        error = BudgetExceededError(estimates=estimates)
        assert error.estimates == estimates
        assert estimates[0][1] is None  # Unpriced model


class TestConfigurationError:
    """Test configuration errors."""

    def test_configuration_error(self):
        """Test configuration error creation."""
        error = ConfigurationError("Invalid configuration")
        assert str(error) == "Invalid configuration"
        assert error.error_code == "configuration_error"

    def test_configuration_error_with_key(self):
        """Test configuration error with config key."""
        error = ConfigurationError("Missing value", config_key="API_KEY")
        assert error.config_key == "API_KEY"
        assert error.details["config_key"] == "API_KEY"

    def test_configuration_error_with_details(self):
        """Test configuration error with additional details."""
        error = ConfigurationError(
            "Invalid format",
            config_key="CORS_ORIGINS",
            details={"expected": "list", "got": "string"},
        )
        assert error.details["config_key"] == "CORS_ORIGINS"
        assert error.details["expected"] == "list"


class TestCacheError:
    """Test cache errors."""

    def test_cache_error(self):
        """Test cache error creation."""
        error = CacheError("Cache operation failed")
        assert str(error) == "Cache operation failed"
        assert error.error_code == "cache_error"

    def test_cache_error_with_type(self):
        """Test cache error with cache type."""
        error = CacheError("Redis connection failed", cache_type="redis")
        assert error.cache_type == "redis"
        assert error.details["cache_type"] == "redis"

    def test_cache_error_with_details(self):
        """Test cache error with additional details."""
        error = CacheError(
            "Key not found", cache_type="memory", details={"key": "prompt_123", "ttl": 3600}
        )
        assert error.details["cache_type"] == "memory"
        assert error.details["key"] == "prompt_123"


class TestClassificationError:
    """Test classification errors."""

    def test_classification_error(self):
        """Test classification error creation."""
        error = ClassificationError("Classification failed")
        assert str(error) == "Classification failed"
        assert error.error_code == "classification_error"

    def test_classification_error_with_type(self):
        """Test classification error with classifier type."""
        error = ClassificationError("Model not loaded", classifier_type="intent")
        assert error.classifier_type == "intent"
        assert error.details["classifier_type"] == "intent"


class TestValidationError:
    """Test validation errors."""

    def test_validation_error(self):
        """Test validation error creation."""
        error = ValidationError("Invalid input")
        assert str(error) == "Invalid input"
        assert error.error_code == "validation_error"

    def test_validation_error_with_field(self):
        """Test validation error with field."""
        error = ValidationError("Value out of range", field="temperature")
        assert error.field == "temperature"
        assert error.details["field"] == "temperature"

    def test_validation_error_with_details(self):
        """Test validation error with additional details."""
        error = ValidationError(
            "Invalid value", field="max_tokens", details={"min": 1, "max": 4096, "got": 5000}
        )
        assert error.details["field"] == "max_tokens"
        assert error.details["max"] == 4096


class TestMuxerTimeoutError:
    """Test timeout errors."""

    def test_timeout_error_default(self):
        """Test timeout error with default message."""
        error = MuxerTimeoutError()
        assert str(error) == "Request timeout"
        assert error.error_code == "timeout_error"

    def test_timeout_error_with_duration(self):
        """Test timeout error with timeout duration."""
        error = MuxerTimeoutError("Provider timeout", timeout_duration=30.0)
        assert str(error) == "Provider timeout"
        assert error.timeout_duration == 30.0
        assert error.details["timeout_duration"] == 30.0


class TestModelNotFoundError:
    """Test model not found errors."""

    def test_model_not_found_error(self):
        """Test model not found error creation."""
        error = ModelNotFoundError("Model not available", model="gpt-5")
        assert str(error) == "Model not available"
        assert error.model == "gpt-5"
        assert error.details["model"] == "gpt-5"
        assert error.error_code == "model_not_found"

    def test_model_not_found_inherits_provider_error(self):
        """Test model not found inherits from ProviderError."""
        error = ModelNotFoundError("Not found", model="test", provider="openai")
        assert isinstance(error, ProviderError)
        assert error.provider == "openai"


class TestQuotaExceededError:
    """Test quota exceeded errors."""

    def test_quota_exceeded_default(self):
        """Test quota exceeded error with default message."""
        error = QuotaExceededError()
        assert str(error) == "Provider quota exceeded"
        assert error.error_code == "quota_exceeded"

    def test_quota_exceeded_custom(self):
        """Test quota exceeded error with custom message."""
        error = QuotaExceededError("Daily quota exhausted", provider="openai")
        assert str(error) == "Daily quota exhausted"
        assert error.provider == "openai"

    def test_quota_exceeded_inherits_provider_error(self):
        """Test quota exceeded inherits from ProviderError."""
        error = QuotaExceededError()
        assert isinstance(error, ProviderError)


class TestNoProvidersAvailableError:
    """Test no providers available error."""

    def test_no_providers_default(self):
        """Test no providers available with default message."""
        error = NoProvidersAvailableError()
        assert str(error) == "No LLM providers available"
        assert error.error_code == "no_providers_available"

    def test_no_providers_custom(self):
        """Test no providers available with custom message."""
        error = NoProvidersAvailableError(
            "All providers are down", details={"attempted": ["openai", "anthropic"]}
        )
        assert str(error) == "All providers are down"
        assert error.details["attempted"] == ["openai", "anthropic"]


class TestExceptionInheritance:
    """Test exception inheritance and relationships."""

    def test_all_exceptions_inherit_from_base(self):
        """Test all exceptions inherit from ModelMuxerError."""
        exceptions = [
            ProviderError("test"),
            RoutingError("test"),
            RouterConfigurationError("test"),
            AuthenticationError("test"),
            RateLimitError("test"),
            BudgetExceededError("test"),
            ConfigurationError("test"),
            CacheError("test"),
            ClassificationError("test"),
            ValidationError("test"),
            MuxerTimeoutError("test"),
            ModelNotFoundError("test"),
            QuotaExceededError("test"),
            NoProvidersAvailableError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, ModelMuxerError)
            assert isinstance(exc, Exception)

    def test_provider_error_subclasses(self):
        """Test provider error inheritance."""
        model_error = ModelNotFoundError("test")
        quota_error = QuotaExceededError("test")

        assert isinstance(model_error, ProviderError)
        assert isinstance(model_error, ModelMuxerError)
        assert isinstance(quota_error, ProviderError)
        assert isinstance(quota_error, ModelMuxerError)
