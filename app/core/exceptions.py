# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Custom exceptions for ModelMuxer.

This module defines all custom exceptions used throughout the application
to provide clear error handling and debugging information.
"""

from typing import Any


class ModelMuxerError(Exception):
    """Base exception for all ModelMuxer errors."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize ModelMuxer base exception."""
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary format."""
        return {
            "error": {
                "message": self.message,
                "type": self.__class__.__name__,
                "code": self.error_code,
                "details": self.details,
            }
        }


class ProviderError(ModelMuxerError):
    """Exception for provider-related errors."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        status_code: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize provider error with provider context."""
        self.provider = provider
        self.status_code = status_code
        details = kwargs.get("details", {})
        details.update({"provider": provider, "status_code": status_code})
        super().__init__(message, kwargs.get("error_code"), details)


class RoutingError(ModelMuxerError):
    """Exception for routing-related errors."""

    def __init__(self, message: str, routing_strategy: str | None = None, **kwargs: Any) -> None:
        """Initialize routing error with strategy context."""
        self.routing_strategy = routing_strategy
        details = kwargs.get("details", {})
        details["routing_strategy"] = routing_strategy
        super().__init__(message, kwargs.get("error_code"), details)


class RouterConfigurationError(ModelMuxerError):
    """Exception for router configuration errors, especially at startup."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        """Initialize router configuration error."""
        super().__init__(message, error_code="router_configuration_error", **kwargs)


class AuthenticationError(ModelMuxerError):
    """Exception for authentication failures."""

    def __init__(self, message: str = "Authentication failed", **kwargs: Any) -> None:
        """Initialize authentication error."""
        super().__init__(message, error_code="authentication_error", **kwargs)


class RateLimitError(ModelMuxerError):
    """Exception for rate limit exceeded."""

    def __init__(
        self, message: str = "Rate limit exceeded", retry_after: int | None = None, **kwargs: Any
    ) -> None:
        """Initialize rate limit error with retry timing."""
        self.retry_after = retry_after
        details = kwargs.get("details", {})
        details["retry_after"] = retry_after
        super().__init__(message, error_code="rate_limit_exceeded", details=details)


class BudgetExceededError(ModelMuxerError):
    """Exception for budget limit exceeded.

    estimates may include None for unpriced/unknown model costs.
    """

    def __init__(
        self,
        message: str = "Budget limit exceeded",
        limit: float | None = None,
        estimates: list[tuple[str, float | None]] | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize budget exceeded error with cost details."""
        self.limit = limit
        self.estimates = estimates or []
        self.reason = reason
        details = kwargs.get("details", {})
        details.update({"limit": limit, "estimates": self.estimates, "reason": reason})
        super().__init__(message, error_code="budget_exceeded", details=details)


class ConfigurationError(ModelMuxerError):
    """Exception for configuration-related errors."""

    def __init__(self, message: str, config_key: str | None = None, **kwargs: Any) -> None:
        """Initialize configuration error with key context."""
        self.config_key = config_key
        details = kwargs.get("details", {})
        details["config_key"] = config_key
        super().__init__(message, error_code="configuration_error", details=details)


class CacheError(ModelMuxerError):
    """Exception for cache-related errors."""

    def __init__(self, message: str, cache_type: str | None = None, **kwargs: Any) -> None:
        """Initialize cache error with cache type context."""
        self.cache_type = cache_type
        details = kwargs.get("details", {})
        details["cache_type"] = cache_type
        super().__init__(message, error_code="cache_error", details=details)


class ClassificationError(ModelMuxerError):
    """Exception for classification-related errors."""

    def __init__(self, message: str, classifier_type: str | None = None, **kwargs: Any) -> None:
        """Initialize classification error with classifier context."""
        self.classifier_type = classifier_type
        details = kwargs.get("details", {})
        details["classifier_type"] = classifier_type
        super().__init__(message, error_code="classification_error", details=details)


class ValidationError(ModelMuxerError):
    """Exception for input validation errors."""

    def __init__(self, message: str, field: str | None = None, **kwargs: Any) -> None:
        """Initialize validation error with field context."""
        self.field = field
        details = kwargs.get("details", {})
        details["field"] = field
        super().__init__(message, error_code="validation_error", details=details)


class TimeoutError(ModelMuxerError):
    """Exception for timeout errors."""

    def __init__(
        self, message: str = "Request timeout", timeout_duration: float | None = None, **kwargs: Any
    ) -> None:
        """Initialize timeout error with duration context."""
        self.timeout_duration = timeout_duration
        details = kwargs.get("details", {})
        details["timeout_duration"] = timeout_duration
        super().__init__(message, error_code="timeout_error", details=details)


class ModelNotFoundError(ProviderError):
    """Exception for model not found errors."""

    def __init__(self, message: str, model: str | None = None, **kwargs: Any) -> None:
        """Initialize model not found error with model context."""
        self.model = model
        details = kwargs.get("details", {})
        details["model"] = model
        super().__init__(message, error_code="model_not_found", details=details, **kwargs)


class QuotaExceededError(ProviderError):
    """Exception for quota exceeded errors."""

    def __init__(self, message: str = "Provider quota exceeded", **kwargs: Any) -> None:
        """Initialize quota exceeded error."""
        super().__init__(message, error_code="quota_exceeded", **kwargs)


class NoProvidersAvailableError(ModelMuxerError):
    """Exception for when no providers are available."""

    def __init__(self, message: str = "No LLM providers available", **kwargs: Any) -> None:
        """Initialize no providers available error."""
        super().__init__(message, error_code="no_providers_available", **kwargs)
