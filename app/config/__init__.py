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
from typing import Any, Callable, Optional, Type

# Load the basic config module directly
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.py")
spec = importlib.util.spec_from_file_location("basic_config", config_path)
if spec is not None and spec.loader is not None:
    basic_config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(basic_config_module)
else:
    raise ImportError("Could not load basic config module")
Settings = basic_config_module.Settings

# Initialize with basic config by default
settings = Settings()

# Enhanced config components (will be set if enhanced mode is available)
enhanced_config: Optional[Any] = None
AuthConfig: Optional[Type[Any]] = None
CacheConfig: Optional[Type[Any]] = None
ClassificationConfig: Optional[Type[Any]] = None
LoggingConfig: Optional[Type[Any]] = None
ModelMuxerConfig: Optional[Type[Any]] = None
MonitoringConfig: Optional[Type[Any]] = None
ProviderConfig: Optional[Type[Any]] = None
RateLimitConfig: Optional[Type[Any]] = None
RoutingConfig: Optional[Type[Any]] = None
load_enhanced_config: Optional[Callable[[], Any]] = None

# Only try to load enhanced config if we're in enhanced mode
if os.getenv("MODELMUXER_MODE", "basic").lower() in ["enhanced", "production"]:
    try:
        from .enhanced_config import (  # type: ignore[attr-defined]
            AuthConfig,  # type: ignore[misc]
            CacheConfig,  # type: ignore[misc]
            ClassificationConfig,  # type: ignore[misc]
            LoggingConfig,  # type: ignore[misc]
            ModelMuxerConfig,  # type: ignore[misc]
            MonitoringConfig,  # type: ignore[misc]
            ProviderConfig,  # type: ignore[misc]
            RateLimitConfig,  # type: ignore[misc]
            RoutingConfig,  # type: ignore[misc]
            enhanced_config,  # type: ignore[misc]
            load_enhanced_config,  # type: ignore[misc]
        )

        # Use enhanced config if it loaded successfully
        if enhanced_config:
            settings = enhanced_config
            print("✅ Enhanced configuration loaded")
        else:
            print("⚠️ Enhanced config is None, using basic configuration")

    except Exception as e:
        print(f"Enhanced config failed to load: {e}")
        print("Falling back to basic configuration...")
        # Keep basic config and None values for enhanced components

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
