# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
ML-based prompt classification for ModelMuxer.

This module contains machine learning models and utilities for classifying
prompts and determining optimal routing strategies.
"""

from .embeddings import EmbeddingManager
from .prompt_classifier import PromptClassifier

__all__ = ["PromptClassifier", "EmbeddingManager"]
