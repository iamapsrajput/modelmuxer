# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Metrics collector module for ModelMuxer.

This module provides a clean interface to the MetricsCollector class
and related functionality for monitoring and observability.
"""

# Import the MetricsCollector from the main metrics module
from .metrics import MetricsCollector

# Re-export for clean imports
__all__ = ["MetricsCollector"]
