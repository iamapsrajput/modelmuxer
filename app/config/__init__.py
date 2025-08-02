# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Configuration management for ModelMuxer.

This package provides comprehensive configuration management including
enhanced configuration with support for all advanced features.
"""

from .enhanced_config import (
    ModelMuxerConfig,
    ProviderConfig,
    RoutingConfig,
    CacheConfig,
    AuthConfig,
    RateLimitConfig,
    MonitoringConfig,
    LoggingConfig,
    ClassificationConfig,
    enhanced_config,
    load_enhanced_config,
)

# Alias for backward compatibility
settings = enhanced_config

__all__ = [
    "ModelMuxerConfig",
    "ProviderConfig",
    "RoutingConfig",
    "CacheConfig",
    "AuthConfig",
    "RateLimitConfig",
    "MonitoringConfig",
    "LoggingConfig",
    "ClassificationConfig",
    "enhanced_config",
    "load_enhanced_config",
    "settings",
]
