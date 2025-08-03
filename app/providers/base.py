# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Abstract base class for LLM providers.
"""

import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

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
    """Abstract base class for LLM providers."""

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
