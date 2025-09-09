# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""Realistic mock providers for integration testing."""

import asyncio
import json
import random
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock


class RealisticMockProvider:
    """A more realistic mock provider that simulates API behavior."""

    def __init__(self, name: str, latency_ms: int = 100, error_rate: float = 0.0):
        self.name = name
        self.latency_ms = latency_ms
        self.error_rate = error_rate
        self.request_count = 0
        self.error_count = 0

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Simulate a realistic chat completion request."""
        self.request_count += 1

        # Simulate network latency
        await asyncio.sleep(self.latency_ms / 1000.0)

        # Simulate random errors based on error rate
        if random.random() < self.error_rate:
            self.error_count += 1
            raise Exception(f"Simulated API error from {self.name}")

        # Calculate realistic token counts
        prompt_tokens = sum(len(m["content"].split()) * 1.3 for m in messages)
        completion_tokens = min(max_tokens or 100, random.randint(10, 150))

        # Generate response content based on input
        user_message = next((m["content"] for m in messages if m["role"] == "user"), "")

        # Simulate different response qualities based on model
        if "gpt-4" in model:
            response_content = f"This is a high-quality response to: {user_message[:50]}..."
        elif "gpt-3.5" in model:
            response_content = f"This is a standard response to: {user_message[:30]}..."
        else:
            response_content = f"Response to: {user_message[:20]}..."

        # Return OpenAI-compatible response
        return {
            "id": f"chatcmpl-{self.name}-{self.request_count}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": completion_tokens,
                "total_tokens": int(prompt_tokens) + completion_tokens,
            },
            "system_fingerprint": f"fp_{self.name}",
        }

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Calculate realistic costs based on model."""
        # Simplified pricing per 1M tokens
        pricing = {
            "gpt-4": (30.0, 60.0),
            "gpt-4o": (5.0, 15.0),
            "gpt-4o-mini": (0.15, 0.6),
            "gpt-3.5-turbo": (0.5, 1.5),
            "claude-3-opus": (15.0, 75.0),
            "claude-3-sonnet": (3.0, 15.0),
            "claude-3-haiku": (0.25, 1.25),
            "llama-3.1-8b": (0.05, 0.05),
            "mixtral-8x7b": (0.24, 0.24),
        }

        # Find matching price or use default
        for model_key, (input_price, output_price) in pricing.items():
            if model_key in model.lower():
                return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000

        # Default pricing
        return (prompt_tokens * 0.01 + completion_tokens * 0.03) / 1000

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics."""
        return {
            "name": self.name,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.request_count),
            "average_latency_ms": self.latency_ms,
        }


class RealisticMockRouter:
    """A more realistic mock router with intelligent routing logic."""

    def __init__(self):
        self.routing_history = []
        self.providers = {
            "openai": ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"],
            "anthropic": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"],
            "groq": ["llama-3.1-8b", "mixtral-8x7b"],
            "mistral": ["mistral-small", "mistral-large"],
        }

    async def select_model(
        self,
        messages: List[Dict[str, str]],
        user_preferences: Optional[Dict] = None,
        constraints: Optional[Dict] = None,
    ) -> tuple:
        """Select model based on realistic routing logic."""
        # Analyze prompt complexity
        total_chars = sum(len(m["content"]) for m in messages)
        is_complex = total_chars > 500 or any(
            keyword in str(messages).lower()
            for keyword in ["analyze", "explain", "compare", "code", "implement"]
        )

        # Apply constraints
        max_cost = (constraints or {}).get("max_cost", 0.1)
        preferred_providers = (constraints or {}).get("preferred_providers", [])

        # Select provider and model based on complexity and constraints
        if is_complex and max_cost > 0.05:
            provider = "openai"
            model = "gpt-4o-mini"
            reason = "Complex query requires advanced model"
        elif preferred_providers:
            provider = preferred_providers[0]
            model = self.providers.get(provider, ["gpt-3.5-turbo"])[0]
            reason = f"Using preferred provider: {provider}"
        elif max_cost < 0.001:
            provider = "groq"
            model = "llama-3.1-8b"
            reason = "Budget constraint - using most cost-effective option"
        else:
            provider = "openai"
            model = "gpt-3.5-turbo"
            reason = "Standard routing for general query"

        # Calculate realistic metrics
        metadata = {
            "usd": 0.002 if "groq" in provider else 0.01,
            "eta_ms": 500 if is_complex else 200,
            "model_key": f"{provider}:{model}",
            "tokens_in": len(str(messages)) // 4,
            "tokens_out": 100,
            "confidence": 0.85,
        }

        # Record routing decision
        self.routing_history.append(
            {
                "timestamp": time.time(),
                "provider": provider,
                "model": model,
                "reason": reason,
                "complexity": "high" if is_complex else "low",
            }
        )

        return provider, model, reason, {}, metadata

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        if not self.routing_history:
            return {"total_routes": 0}

        provider_counts = {}
        for route in self.routing_history:
            provider = route["provider"]
            provider_counts[provider] = provider_counts.get(provider, 0) + 1

        return {
            "total_routes": len(self.routing_history),
            "provider_distribution": provider_counts,
            "average_confidence": 0.85,
            "last_route": self.routing_history[-1] if self.routing_history else None,
        }


def create_realistic_provider_registry(
    providers: Optional[List[str]] = None, latency_ms: int = 100, error_rate: float = 0.0
) -> Dict[str, RealisticMockProvider]:
    """Create a registry of realistic mock providers."""
    if providers is None:
        providers = ["openai", "anthropic", "groq", "mistral", "cohere", "google", "together"]

    registry = {}
    for provider_name in providers:
        # Vary latency by provider
        provider_latency = latency_ms
        if provider_name == "groq":
            provider_latency = 50  # Groq is typically faster
        elif provider_name == "anthropic":
            provider_latency = 150  # Claude can be slower

        registry[provider_name] = RealisticMockProvider(
            name=provider_name, latency_ms=provider_latency, error_rate=error_rate
        )

    return registry


def create_realistic_mock_response(
    model: str, messages: List[Dict[str, str]], provider: str = "openai", stream: bool = False
) -> Dict[str, Any]:
    """Create a realistic mock response for testing."""
    # Get user message
    user_message = next((m["content"] for m in messages if m["role"] == "user"), "Hello")

    # Generate appropriate response based on model tier
    if any(tier in model for tier in ["gpt-4", "claude-3-opus", "claude-3-sonnet"]):
        response_quality = "comprehensive and detailed"
    elif any(tier in model for tier in ["gpt-3.5", "claude-3-haiku", "mistral-large"]):
        response_quality = "clear and concise"
    else:
        response_quality = "basic"

    response_content = (
        f"This is a {response_quality} response to your query about: {user_message[:100]}"
    )

    if stream:
        # Return a generator for streaming responses
        async def stream_generator():
            words = response_content.split()
            for i, word in enumerate(words):
                chunk = {
                    "id": f"chatcmpl-stream-{i}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": word + " "},
                            "finish_reason": None if i < len(words) - 1 else "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.01)  # Simulate streaming delay
            yield "data: [DONE]\n\n"

        return stream_generator()

    # Regular response
    return {
        "id": f"chatcmpl-{provider}-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": sum(len(m["content"].split()) for m in messages) * 2,
            "completion_tokens": len(response_content.split()) * 2,
            "total_tokens": (
                sum(len(m["content"].split()) for m in messages) + len(response_content.split())
            )
            * 2,
        },
        "system_fingerprint": f"fp_{provider}_2024",
    }


__all__ = [
    "RealisticMockProvider",
    "RealisticMockRouter",
    "create_realistic_provider_registry",
    "create_realistic_mock_response",
]
