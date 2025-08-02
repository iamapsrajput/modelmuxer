# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Core utilities and interfaces for ModelMuxer.

This module contains shared utilities, abstract interfaces, and core functionality
that is used across the entire application.
"""

from .interfaces import RouterInterface, ProviderInterface, CacheInterface
from .exceptions import (
    ModelMuxerError,
    ProviderError,
    RoutingError,
    AuthenticationError,
    RateLimitError,
    BudgetExceededError,
    ConfigurationError
)
from .utils import (
    hash_prompt,
    estimate_tokens,
    format_cost,
    sanitize_input,
    generate_request_id
)

__all__ = [
    # Interfaces
    "RouterInterface",
    "ProviderInterface", 
    "CacheInterface",
    # Exceptions
    "ModelMuxerError",
    "ProviderError",
    "RoutingError",
    "AuthenticationError",
    "RateLimitError",
    "BudgetExceededError",
    "ConfigurationError",
    # Utilities
    "hash_prompt",
    "estimate_tokens",
    "format_cost",
    "sanitize_input",
    "generate_request_id"
]
