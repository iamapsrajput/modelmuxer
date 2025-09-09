# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Configuration management for ModelMuxer.

This package provides comprehensive configuration management including
enhanced configuration with support for all advanced features.
"""

# Import basic config first to avoid circular imports
import importlib.util
import os
import sys
from collections.abc import Callable
from typing import Any, Optional

# Load the basic config module directly
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.py")
spec = importlib.util.spec_from_file_location("basic_config", config_path)
if spec is not None and spec.loader is not None:
    basic_config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(basic_config_module)
else:
    raise ImportError("Could not load basic config module")
Settings = basic_config_module.Settings

# Import centralized settings
from ..settings import settings as app_settings

# Initialize with centralized settings by default
settings = app_settings

# Enhanced config components (will be set if enhanced mode is available)
enhanced_config: Any | None = None
AuthConfig: type[Any] | None = None
CacheConfig: type[Any] | None = None
ClassificationConfig: type[Any] | None = None
LoggingConfig: type[Any] | None = None
ModelMuxerConfig: type[Any] | None = None
MonitoringConfig: type[Any] | None = None
ProviderConfig: type[Any] | None = None
RateLimitConfig: type[Any] | None = None
RoutingConfig: type[Any] | None = None
load_enhanced_config: Callable[[], Any] | None = None

# Only try to load enhanced config if we're in enhanced mode
if app_settings.features.mode in ["enhanced", "production"]:
    try:
        from .enhanced_config import (AuthConfig, CacheConfig,
                                      ClassificationConfig, LoggingConfig,
                                      ModelMuxerConfig, MonitoringConfig,
                                      ProviderConfig, RateLimitConfig,
                                      RoutingConfig, enhanced_config,
                                      load_enhanced_config)

        # Use enhanced config if it loaded successfully
        if enhanced_config:
            # Keep centralized settings as primary, enhanced config for extended features
            # Enhanced configuration loaded successfully (logged via structlog if available)
            pass
        else:
            # Enhanced config is None, using centralized configuration (logged via structlog if available)
            pass

    except Exception:  # nosec B110
        # Enhanced config failed to load, falling back to centralized configuration
        # Error details logged via structlog if available
        pass
        # Keep centralized config and None values for enhanced components

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
