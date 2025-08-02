# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Custom exceptions for ModelMuxer.

This module defines all custom exceptions used throughout the application
to provide clear error handling and debugging information.
"""

from typing import Optional, Dict, Any


class ModelMuxerError(Exception):
    """Base exception for all ModelMuxer errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format."""
        return {
            "error": {
                "message": self.message,
                "type": self.__class__.__name__,
                "code": self.error_code,
                "details": self.details
            }
        }


class ProviderError(ModelMuxerError):
    """Exception for provider-related errors."""
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        self.provider = provider
        self.status_code = status_code
        details = kwargs.get("details", {})
        details.update({
            "provider": provider,
            "status_code": status_code
        })
        super().__init__(message, kwargs.get("error_code"), details)


class RoutingError(ModelMuxerError):
    """Exception for routing-related errors."""
    
    def __init__(
        self,
        message: str,
        routing_strategy: Optional[str] = None,
        **kwargs
    ):
        self.routing_strategy = routing_strategy
        details = kwargs.get("details", {})
        details["routing_strategy"] = routing_strategy
        super().__init__(message, kwargs.get("error_code"), details)


class AuthenticationError(ModelMuxerError):
    """Exception for authentication failures."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, error_code="authentication_error", **kwargs)


class RateLimitError(ModelMuxerError):
    """Exception for rate limit exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs
    ):
        self.retry_after = retry_after
        details = kwargs.get("details", {})
        details["retry_after"] = retry_after
        super().__init__(message, error_code="rate_limit_exceeded", details=details)


class BudgetExceededError(ModelMuxerError):
    """Exception for budget limit exceeded."""
    
    def __init__(
        self,
        message: str = "Budget limit exceeded",
        current_usage: Optional[float] = None,
        budget_limit: Optional[float] = None,
        **kwargs
    ):
        self.current_usage = current_usage
        self.budget_limit = budget_limit
        details = kwargs.get("details", {})
        details.update({
            "current_usage": current_usage,
            "budget_limit": budget_limit
        })
        super().__init__(message, error_code="budget_exceeded", details=details)


class ConfigurationError(ModelMuxerError):
    """Exception for configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        self.config_key = config_key
        details = kwargs.get("details", {})
        details["config_key"] = config_key
        super().__init__(message, error_code="configuration_error", details=details)


class CacheError(ModelMuxerError):
    """Exception for cache-related errors."""
    
    def __init__(self, message: str, cache_type: Optional[str] = None, **kwargs):
        self.cache_type = cache_type
        details = kwargs.get("details", {})
        details["cache_type"] = cache_type
        super().__init__(message, error_code="cache_error", details=details)


class ClassificationError(ModelMuxerError):
    """Exception for classification-related errors."""
    
    def __init__(self, message: str, classifier_type: Optional[str] = None, **kwargs):
        self.classifier_type = classifier_type
        details = kwargs.get("details", {})
        details["classifier_type"] = classifier_type
        super().__init__(message, error_code="classification_error", details=details)


class ValidationError(ModelMuxerError):
    """Exception for input validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        self.field = field
        details = kwargs.get("details", {})
        details["field"] = field
        super().__init__(message, error_code="validation_error", details=details)


class TimeoutError(ModelMuxerError):
    """Exception for timeout errors."""
    
    def __init__(
        self,
        message: str = "Request timeout",
        timeout_duration: Optional[float] = None,
        **kwargs
    ):
        self.timeout_duration = timeout_duration
        details = kwargs.get("details", {})
        details["timeout_duration"] = timeout_duration
        super().__init__(message, error_code="timeout_error", details=details)


class ModelNotFoundError(ProviderError):
    """Exception for model not found errors."""
    
    def __init__(self, message: str, model: Optional[str] = None, **kwargs):
        self.model = model
        details = kwargs.get("details", {})
        details["model"] = model
        super().__init__(message, error_code="model_not_found", details=details, **kwargs)


class QuotaExceededError(ProviderError):
    """Exception for quota exceeded errors."""
    
    def __init__(self, message: str = "Provider quota exceeded", **kwargs):
        super().__init__(message, error_code="quota_exceeded", **kwargs)
