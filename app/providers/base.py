# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Abstract base class for LLM providers.

This module contains the unified Provider Adapter interface (invoke) with
shared resilience helpers and a common ProviderResponse dataclass.
"""

import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

from ..models import ChatCompletionResponse, ChatMessage, Choice, RouterMetadata, Usage


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(self, message: str, status_code: int | None = None, provider: str | None = None):
        self.message = message
        self.status_code = status_code
        self.provider = provider
        super().__init__(message)


class RateLimitError(ProviderError):
    """Exception for rate limit errors."""

    pass


class AuthenticationError(ProviderError):
    """Exception for authentication errors."""

    pass


class ModelNotFoundError(ProviderError):
    """Exception for model not found errors."""

    pass


import asyncio
import random
import time

# ===================== New Adapter Interface =====================
from dataclasses import dataclass

from app.settings import settings

# ===================== Shared Utilities =====================


def estimate_tokens(text: str) -> int:
    """Simple token estimation (roughly 4 characters per token)."""
    return max(len(text) // 4, 1)


def normalize_finish_reason(provider: str, finish_reason: str | None) -> str:
    """Normalize provider-specific finish reasons to standard format.

    Args:
        provider: Provider name (e.g., 'google', 'cohere', 'together')
        finish_reason: Provider-specific finish reason

    Returns:
        Normalized finish reason: 'stop', 'length', 'content_filter', or 'stop' as default
    """
    if not finish_reason:
        return "stop"

    # Provider-specific mappings
    mappings = {
        "google": {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
        },
        "cohere": {
            "COMPLETE": "stop",
            "MAX_TOKENS": "length",
            "ERROR": "content_filter",
            "ERROR_TOXIC": "content_filter",
        },
        "together": {
            "stop": "stop",
            "length": "length",
            "content_filter": "content_filter",
        },
        "openai": {
            "stop": "stop",
            "length": "length",
            "content_filter": "content_filter",
        },
        "anthropic": {
            "end_turn": "stop",
            "max_tokens": "length",
            "content_filter": "content_filter",
        },
    }

    provider_mapping = mappings.get(provider.lower(), {})
    return provider_mapping.get(finish_reason, "stop")


def _is_retryable_error(provider: str, status_code: int | None, payload: dict | None) -> bool:
    """Determine if a provider-level error is retryable based on status code and payload.

    Args:
        provider: Provider name (e.g., 'google', 'cohere', 'together')
        status_code: HTTP status code if available
        payload: Response payload containing error information

    Returns:
        True if the error is retryable, False otherwise
    """
    # HTTP status code based retry logic
    if status_code:
        if status_code == 429 or status_code >= 500:
            return True
        if status_code in (400, 401, 403, 404):
            return False

    # Provider-specific payload error analysis
    if not payload or "error" not in payload:
        return False

    error_data = payload["error"]
    error_code = error_data.get("code", "")
    error_msg = error_data.get("message", "").lower()

    # Provider-specific retryable error patterns
    provider_patterns = {
        "google": {
            "codes": {"quota_exceeded", "resource_exhausted", "unavailable", "internal"},
            "keywords": {"quota", "rate limit", "unavailable", "internal", "temporary"},
        },
        "cohere": {
            "codes": {"rate_limit", "overloaded", "server_error", "internal_error"},
            "keywords": {"rate limit", "overloaded", "server", "internal", "temporary"},
        },
        "together": {
            "codes": {"rate_limit", "overloaded", "server_error", "internal_error"},
            "keywords": {"rate limit", "overloaded", "server", "internal", "temporary", "retry"},
        },
        "openai": {
            "codes": {"rate_limit_exceeded", "server_error", "internal_error"},
            "keywords": {"rate limit", "server", "internal", "temporary"},
        },
        "anthropic": {
            "codes": {"rate_limit", "server_error", "internal_error"},
            "keywords": {"rate limit", "server", "internal", "temporary"},
        },
    }

    patterns = provider_patterns.get(provider.lower(), {"codes": set(), "keywords": set()})

    # Check error code
    if error_code in patterns["codes"]:
        return True

    # Check error message keywords
    if any(keyword in error_msg for keyword in patterns["keywords"]):
        return True

    return False


# Standard User-Agent for all provider adapters
USER_AGENT = "ModelMuxer/1.0.0"


def messages_to_prompt_text(messages: list[ChatMessage]) -> str:
    """Flatten structured messages to a single prompt for legacy adapters."""
    parts = [msg.content for msg in messages if msg.content]
    return "\n".join(parts) if parts else ""


def messages_to_openai_format(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert ChatMessage list to OpenAI chat message dicts."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        entry: dict[str, Any] = {"role": msg.role}
        if msg.role == "assistant":
            if msg.tool_calls:
                entry["tool_calls"] = msg.tool_calls
            entry["content"] = msg.content
        elif msg.role == "tool":
            entry["content"] = msg.content or ""
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            if msg.name:
                entry["name"] = msg.name
        else:
            if msg.content is not None:
                entry["content"] = msg.content
        if msg.name and msg.role != "tool":
            entry["name"] = msg.name
        result.append(entry)
    return result


def openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI tool definitions to Anthropic Messages API tool schemas."""
    anthropic_tools: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fn = tool.get("function", {})
        anthropic_tools.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return anthropic_tools


def build_stream_chunk(
    *,
    chunk_id: str,
    model: str,
    content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
    created: int | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-compatible chat.completion.chunk payload."""
    delta: dict[str, str] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created or int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def format_sse_data(payload: dict[str, Any] | str) -> str:
    """Format a payload as an SSE data line."""
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload)}\n\n"


async def with_retries(
    coro_factory, *, max_attempts: int, base_s: float, retry_on: tuple[type[Exception], ...]
):
    """Reusable retry and backoff logic for provider adapters."""
    attempt = 0
    last_err = None
    while attempt < max_attempts:
        attempt += 1
        try:
            return await coro_factory(attempt)
        except retry_on as e:
            last_err = e
            if attempt >= max_attempts:
                break
            delay = base_s * (2 ** (attempt - 1)) + random.uniform(0, base_s)
            await asyncio.sleep(delay)
    raise last_err


@dataclass
class ProviderResponse:
    """Standard adapter response used across providers.

    - output_text: final assistant text
    - tokens_in/tokens_out: token counts from provider
    - latency_ms: total latency in milliseconds
    - raw: raw provider payload (optional)
    - error: error string if any
    - tool_calls: OpenAI-format tool calls when the model requests tools
    - finish_reason: normalized finish reason from the provider
    """

    output_text: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    raw: Any | None = None
    error: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None


class SimpleCircuitBreaker:
    """Minimal circuit breaker for adapters."""

    def __init__(self, fail_threshold: int | None = None, cooldown_sec: int | None = None) -> None:
        self.failures = 0
        self.open_until: float = 0.0
        self.fail_threshold = fail_threshold or settings.providers.circuit_fail_threshold
        self.cooldown_sec = cooldown_sec or settings.providers.circuit_cooldown_sec

    def is_open(self) -> bool:
        return time.time() < self.open_until

    def on_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.fail_threshold:
            self.open_until = time.time() + self.cooldown_sec

    def on_success(self) -> None:
        self.failures = 0
        self.open_until = 0.0


class LLMProviderAdapter(ABC):
    """Abstract provider adapter with a unified invoke() method and lifecycle management."""

    @abstractmethod
    async def invoke(
        self, model: str, messages: list[ChatMessage], **kwargs: Any
    ) -> ProviderResponse:
        """Invoke the provider with structured messages and return standardized response."""
        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        """Close the adapter's HTTP client and clean up resources."""
        raise NotImplementedError

    @abstractmethod
    def get_supported_models(self) -> list[str]:
        """Get the list of models supported by this provider."""
        raise NotImplementedError

    def supports_streaming(self) -> bool:
        """Return True when the adapter implements native streaming."""
        return type(self).invoke_stream is not LLMProviderAdapter.invoke_stream

    async def invoke_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream OpenAI-compatible chunk dicts. Override for provider-native streaming."""
        if False:  # pragma: no cover - makes this a generator for type checkers
            yield {}
        response = await self.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            **kwargs,
        )
        chunk_id = response.id
        yield build_stream_chunk(
            chunk_id=chunk_id,
            model=model,
            role="assistant",
            content="",
            created=response.created,
        )
        content = response.choices[0].message.content or ""
        if content:
            yield build_stream_chunk(
                chunk_id=chunk_id,
                model=model,
                content=content,
                created=response.created,
            )
        yield build_stream_chunk(
            chunk_id=chunk_id,
            model=model,
            finish_reason=response.choices[0].finish_reason or "stop",
            created=response.created,
        )

    async def chat_completion(
        self,
        messages: list[ChatMessage],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        """
        Chat completion shim that calls invoke() internally.

        This provides compatibility with the legacy chat_completion interface
        while using the unified invoke() method internally.
        """
        # Call the unified invoke method
        provider_response = await self.invoke(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Convert ProviderResponse to ChatCompletionResponse
        if provider_response.error:
            raise ProviderError(provider_response.error)

        assistant_message = ChatMessage(
            role="assistant",
            content=provider_response.output_text or None,
            tool_calls=provider_response.tool_calls,
        )
        finish_reason = provider_response.finish_reason or (
            "tool_calls" if provider_response.tool_calls else "stop"
        )

        return ChatCompletionResponse(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=assistant_message,
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=provider_response.tokens_in,
                completion_tokens=provider_response.tokens_out,
                total_tokens=provider_response.tokens_in + provider_response.tokens_out,
            ),
            router_metadata=RouterMetadata(
                selected_provider=self.__class__.__name__.replace("Adapter", "").lower(),
                selected_model=model,
                routing_reason="adapter_invoke",
                estimated_cost=0.0,
                response_time_ms=float(provider_response.latency_ms),
                request_id=str(uuid.uuid4()),
            ),
        )

    async def stream_chat_completion(
        self,
        messages: list[ChatMessage],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Yield OpenAI-compatible streaming chunk dicts for the chat route."""
        async for chunk in self.invoke_stream(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        ):
            yield chunk
