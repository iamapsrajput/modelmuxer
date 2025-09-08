# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Advanced routing strategies for ModelMuxer.

This module contains various routing implementations including heuristic,
semantic, cascade, and hybrid routing strategies.
"""

from .base_router import BaseRouter
from .cascade_router import CascadeRouter
from .heuristic_router import EnhancedHeuristicRouter
from .hybrid_router import HybridRouter

# Import SemanticRouter only if numpy is available
try:
    from .semantic_router import SemanticRouter
except ImportError:
    # SemanticRouter requires numpy which might not be available in test environment
    SemanticRouter = None

__all__ = [
    "BaseRouter",
    "EnhancedHeuristicRouter",
    "CascadeRouter",
    "HybridRouter",
]

if SemanticRouter is not None:
    __all__.append("SemanticRouter")
