# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Core utilities and interfaces for ModelMuxer.

This module contains shared utilities, abstract interfaces, and core functionality
that is used across the entire application.
"""

from .exceptions import (
    AuthenticationError,
    BudgetExceededError,
    ConfigurationError,
    ModelMuxerError,
    ProviderError,
    RateLimitError,
    RoutingError,
)
from .interfaces import CacheInterface, ProviderInterface, RouterInterface
from .utils import (
    estimate_tokens,
    format_cost,
    generate_request_id,
    hash_prompt,
    sanitize_input,
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
    "generate_request_id",
]
