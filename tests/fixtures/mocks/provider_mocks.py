# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Centralized provider mock configurations.

This module provides standardized mock providers, registries, and responses
to ensure consistent testing across all test files.
"""

import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock

from app.providers.base import ProviderResponse

# Standard mock provider registry with multiple providers
MOCK_PROVIDER_REGISTRY = {
    "openai": Mock(),
    "anthropic": Mock(),
    "mistral": Mock(),
    "groq": Mock(),
}

# Empty provider registry for testing no-provider scenarios
EMPTY_PROVIDER_REGISTRY = {}

# Single provider registry for specific tests
SINGLE_PROVIDER_REGISTRY = {
    "openai": Mock(),
}


def create_mock_provider_response(
    output_text: str = "Test response",
    tokens_in: int = 10,
    tokens_out: int = 5,
    latency_ms: int = 100,
    model: str = "gpt-3.5-turbo",
    provider: str = "openai",
) -> ProviderResponse:
    """Create a standardized mock provider response."""
    return ProviderResponse(
        output_text=output_text,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
        raw={
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": tokens_in,
                "completion_tokens": tokens_out,
                "total_tokens": tokens_in + tokens_out,
            },
        },
    )


def create_mock_provider_registry(providers: List[str] = None) -> Mock:
    """Create a mock provider registry with specified providers."""
    if providers is None:
        providers = ["openai", "anthropic", "mistral", "groq"]

    registry = {}
    for provider_name in providers:
        mock_provider = Mock()
        mock_provider.chat_completion = AsyncMock(
            return_value=create_mock_provider_response(provider=provider_name)
        )
        mock_provider.stream_chat_completion = AsyncMock()
        mock_provider.get_supported_models = Mock(return_value=[f"{provider_name}-model"])
        registry[provider_name] = mock_provider

    return registry


def create_mock_provider_adapter(
    provider_name: str = "openai",
    success: bool = True,
    error_type: str = None,
) -> Mock:
    """Create a mock provider adapter."""
    adapter = Mock()
    adapter.provider_name = provider_name

    if success:
        adapter.invoke = AsyncMock(
            return_value=create_mock_provider_response(provider=provider_name)
        )
        adapter.stream_invoke = AsyncMock()
    else:
        if error_type == "timeout":
            adapter.invoke = AsyncMock(side_effect=TimeoutError("Request timeout"))
        elif error_type == "rate_limit":
            adapter.invoke = AsyncMock(side_effect=Exception("Rate limit exceeded"))
        else:
            adapter.invoke = AsyncMock(side_effect=Exception("Provider error"))

    return adapter


# Common mock configurations
MOCK_OPENAI_REGISTRY = create_mock_provider_registry(["openai"])
MOCK_ANTHROPIC_REGISTRY = create_mock_provider_registry(["anthropic"])
MOCK_MULTI_PROVIDER_REGISTRY = create_mock_provider_registry()
