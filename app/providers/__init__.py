# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
LLM Provider implementations for ModelMuxer.

This package contains the unified LLMProviderAdapter interface and concrete
adapter implementations for multiple LLM providers including OpenAI,
Anthropic, Mistral, Google, Cohere, Groq, and Together AI.
"""

from .anthropic import AnthropicAdapter
from .base import (
    AuthenticationError,
    LLMProviderAdapter,
    ModelNotFoundError,
    ProviderError,
    ProviderResponse,
    RateLimitError,
)
from .cohere import CohereAdapter
from .google import GoogleAdapter
from .groq import GroqAdapter
from .mistral import MistralAdapter
from .openai import OpenAIAdapter
from .together import TogetherAdapter

__all__ = [
    # Base classes and exceptions
    "LLMProviderAdapter",
    "ProviderResponse",
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "ModelNotFoundError",
    # Provider adapters
    "OpenAIAdapter",
    "AnthropicAdapter",
    "MistralAdapter",
    "GroqAdapter",
    "GoogleAdapter",
    "CohereAdapter",
    "TogetherAdapter",
]
