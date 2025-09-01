# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Abstract base class for LLM providers.

This module contains legacy provider interfaces (chat_completion, streaming)
and the new unified Provider Adapter interface (invoke) with shared
resilience helpers and a common ProviderResponse dataclass.
"""

import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, Optional

import httpx

from ..models import ChatCompletionResponse, ChatMessage, Choice, RouterMetadata, Usage
from ..security.config import SecurityConfig


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


class LLMProvider(ABC):
    """Legacy abstract base class for providers using chat_completion interface."""

    def __init__(self, api_key: str, base_url: str, provider_name: str):
        self.api_key = api_key
        self.base_url = base_url
        self.provider_name = provider_name
        # Use secure HTTP client configuration
        self.client = SecurityConfig.get_secure_httpx_client()

    async def __aenter__(self) -> "LLMProvider":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.client.aclose()

    @abstractmethod
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
        Generate a chat completion.

        Args:
            messages: List of chat messages
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Returns:
            ChatCompletionResponse with standardized format

        Raises:
            ProviderError: For various provider-specific errors
        """
        raise NotImplementedError

    @abstractmethod
    async def stream_chat_completion(
        self,
        messages: list[ChatMessage],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream a chat completion.

        Args:
            messages: List of chat messages
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters

        Yields:
            Streaming response chunks in OpenAI format
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """
        Calculate the cost for a request.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name used

        Returns:
            Cost in USD
        """
        raise NotImplementedError

    @abstractmethod
    def get_supported_models(self) -> list[str]:
        """
        Get list of supported models for this provider.

        Returns:
            List of model names
        """
        raise NotImplementedError

    def _create_headers(self) -> dict[str, str]:
        """Create headers for API requests."""
        return {
            "Content-Type": "application/json",
            "User-Agent": f"ModelMuxer/1.0.0 ({self.provider_name})",
        }

    def _handle_http_error(self, response: httpx.Response) -> None:
        """Handle HTTP errors and convert to appropriate exceptions."""
        if response.status_code == 401:
            raise AuthenticationError(
                f"Authentication failed for {self.provider_name}",
                status_code=401,
                provider=self.provider_name,
            )
        elif response.status_code == 404:
            raise ModelNotFoundError(
                f"Model not found on {self.provider_name}",
                status_code=404,
                provider=self.provider_name,
            )
        elif response.status_code == 429:
            raise RateLimitError(
                f"Rate limit exceeded for {self.provider_name}",
                status_code=429,
                provider=self.provider_name,
            )
        elif response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "Unknown error")
            except (ValueError, KeyError, TypeError):
                error_message = f"HTTP {response.status_code} error"

            raise ProviderError(
                f"{self.provider_name} error: {error_message}",
                status_code=response.status_code,
                provider=self.provider_name,
            )

    def _create_standard_response(
        self,
        content: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        routing_reason: str,
        response_time_ms: float,
        finish_reason: str = "stop",
    ) -> ChatCompletionResponse:
        """Create a standardized response format."""
        cost = self.calculate_cost(input_tokens, output_tokens, model)

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content, name=None),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            router_metadata=RouterMetadata(
                selected_provider=self.provider_name,
                selected_model=model,
                routing_reason=routing_reason,
                estimated_cost=cost,
                response_time_ms=response_time_ms,
            ),
        )

    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible."""
        try:
            # Simple health check - attempt to make a minimal request
            test_messages = [ChatMessage(role="user", content="Hi", name=None)]
            models = self.get_supported_models()
            if not models:
                return False

            # Use the first available model for health check
            await self.chat_completion(
                messages=test_messages, model=models[0], max_tokens=1, temperature=0.0
            )
            return True
        except Exception:
            return False


# ===================== New Adapter Interface =====================
from dataclasses import dataclass
import asyncio
import random
import time

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
    """

    output_text: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    raw: Optional[Any] = None
    error: Optional[str] = None


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
    async def invoke(self, model: str, prompt: str, **kwargs: Any) -> ProviderResponse:
        """Invoke the provider synchronously and return standardized response."""
        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        """Close the adapter's HTTP client and clean up resources."""
        raise NotImplementedError

    @abstractmethod
    def get_supported_models(self) -> list[str]:
        """Get the list of models supported by this provider."""
        raise NotImplementedError

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
        # Convert messages to prompt text
        prompt_text = " ".join([msg.content for msg in messages if msg.content])

        # Call the unified invoke method
        provider_response = await self.invoke(
            model=model,
            prompt=prompt_text,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Convert ProviderResponse to ChatCompletionResponse
        if provider_response.error:
            raise ProviderError(provider_response.error)

        return ChatCompletionResponse(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=provider_response.output_text),
                    finish_reason="stop",
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
    ) -> AsyncGenerator[str, None]:
        """
        Streaming chat completion shim that calls invoke() internally.

        Note: This is a simplified implementation that calls invoke() and yields
        the result. Real streaming would require adapter-specific implementation.
        """
        # For now, call the regular chat completion and yield chunks
        response = await self.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,  # Force non-streaming for the underlying call
            **kwargs,
        )

        # Simulate streaming by yielding the response in chunks
        import json

        response_dict = response.dict()
        yield f"data: {json.dumps(response_dict)}\n\n"
        yield "data: [DONE]\n\n"
