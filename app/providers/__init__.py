# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
LLM Provider implementations for ModelMuxer.

This package contains the abstract base provider interface and concrete
implementations for multiple LLM providers including OpenAI, Anthropic,
Mistral, Google, Cohere, Groq, and Together AI. Both legacy LLMProvider
classes and new LLMProviderAdapter classes are available.
"""

from .anthropic import AnthropicAdapter
from .anthropic_provider import AnthropicProvider
from .base import AuthenticationError, LLMProvider, ProviderError, RateLimitError
from .cohere import CohereAdapter
from .cohere_provider import CohereProvider
from .google import GoogleAdapter
from .google_provider import GoogleProvider
from .groq import GroqAdapter
from .groq_provider import GroqProvider
from .mistral import MistralAdapter
from .mistral_provider import MistralProvider

# New adapter classes (LLMProviderAdapter pattern)
from .openai import OpenAIAdapter
from .openai_provider import OpenAIProvider
from .together import TogetherAdapter
from .together_provider import TogetherProvider

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
