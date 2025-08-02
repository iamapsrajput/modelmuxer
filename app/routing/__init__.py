# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Advanced routing strategies for ModelMuxer.

This module contains various routing implementations including heuristic,
semantic, cascade, and hybrid routing strategies.
"""

from .base_router import BaseRouter
from .heuristic_router import HeuristicRouter
from .semantic_router import SemanticRouter
from .cascade_router import CascadeRouter
from .hybrid_router import HybridRouter

__all__ = [
    "BaseRouter",
    "HeuristicRouter",
    "SemanticRouter", 
    "CascadeRouter",
    "HybridRouter"
]
