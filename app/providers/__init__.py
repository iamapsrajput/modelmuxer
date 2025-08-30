# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
LLM Provider implementations for ModelMuxer.

This package contains the abstract base provider interface and concrete
implementations for multiple LLM providers including OpenAI, Anthropic,
Mistral, Google, Cohere, Groq, and Together AI. Both legacy LLMProvider
classes and new LLMProviderAdapter classes are available.
"""

from .anthropic_provider import AnthropicProvider
from .base import AuthenticationError, LLMProvider, ProviderError, RateLimitError
from .cohere_provider import CohereProvider
from .google_provider import GoogleProvider
from .groq_provider import GroqProvider
from .mistral_provider import MistralProvider
from .openai_provider import OpenAIProvider
from .together_provider import TogetherProvider

# New adapter classes (LLMProviderAdapter pattern)
from .openai import OpenAIAdapter
from .anthropic import AnthropicAdapter
from .mistral import MistralAdapter
from .groq import GroqAdapter
from .google import GoogleAdapter
from .cohere import CohereAdapter
from .together import TogetherAdapter

__all__ = [
    # Base classes and exceptions
    "LLMProvider",
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    # Legacy provider implementations
    "OpenAIProvider",
    "AnthropicProvider",
    "MistralProvider",
    "GoogleProvider",
    "CohereProvider",
    "GroqProvider",
    "TogetherProvider",
    # New adapter classes
    "OpenAIAdapter",
    "AnthropicAdapter",
    "MistralAdapter",
    "GroqAdapter",
    "GoogleAdapter",
    "CohereAdapter",
    "TogetherAdapter",
]
