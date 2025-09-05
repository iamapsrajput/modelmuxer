# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Together AI provider implementation.

This module provides integration with Together AI's API for the ModelMuxer
LLM routing system. Together AI provides access to various open-source models.
"""

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import structlog

from ..core.utils import estimate_tokens
from ..models import ChatCompletionResponse, ChatMessage
from .base import (AuthenticationError, LLMProvider, ProviderError,
                   RateLimitError)

logger = structlog.get_logger(__name__)


class TogetherProvider(LLMProvider):
    """Together AI API provider implementation."""

    def __init__(self, api_key: str | None = None):
        if not api_key:
            raise AuthenticationError("Together AI API key is required")

        super().__init__(
            api_key=api_key, base_url="https://api.together.xyz/v1", provider_name="together"
        )

        # Pricing per million tokens (as of 2024)
        self.pricing = {
            "meta-llama/Llama-3-70b-chat-hf": {"input": 0.9, "output": 0.9},
            "meta-llama/Llama-3-8b-chat-hf": {"input": 0.2, "output": 0.2},
            "mistralai/Mixtral-8x7B-Instruct-v0.1": {"input": 0.6, "output": 0.6},
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {"input": 0.6, "output": 0.6},
            "togethercomputer/RedPajama-INCITE-7B-Chat": {"input": 0.2, "output": 0.2},
            "microsoft/DialoGPT-medium": {"input": 0.1, "output": 0.1},
        }

        self.supported_models = list(self.pricing.keys())

        # Rate limits (requests per minute)
        self.rate_limits = {
            "meta-llama/Llama-3-70b-chat-hf": 60,
            "meta-llama/Llama-3-8b-chat-hf": 200,
            "mistralai/Mixtral-8x7B-Instruct-v0.1": 100,
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": 100,
            "togethercomputer/RedPajama-INCITE-7B-Chat": 200,
            "microsoft/DialoGPT-medium": 200,
        }

    def _create_headers(self) -> dict[str, str]:
        """Create headers for Together AI API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ModelMuxer/1.0.0 (Together AI)",
        }

    def get_supported_models(self) -> list[str]:
        """Get list of supported Together AI models."""
        return self.supported_models

    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost for Together AI request."""
        if model not in self.pricing:
            model = "meta-llama/Llama-3-8b-chat-hf"  # Default fallback

        model_pricing = self.pricing[model]
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

        return input_cost + output_cost

    def get_rate_limits(self) -> dict[str, Any]:
        """Get rate limit information."""
        return {
            "requests_per_minute": self.rate_limits,
            "tokens_per_minute": {
                "meta-llama/Llama-3-70b-chat-hf": 10000,
                "meta-llama/Llama-3-8b-chat-hf": 50000,
                "mistralai/Mixtral-8x7B-Instruct-v0.1": 20000,
                "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": 20000,
                "togethercomputer/RedPajama-INCITE-7B-Chat": 50000,
                "microsoft/DialoGPT-medium": 50000,
            },
        }

    def _prepare_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        """Convert ChatMessage objects to Together AI format (OpenAI-compatible)."""
        return [
            {"role": msg.role, "content": msg.content, **({"name": msg.name} if msg.name else {})}
            for msg in messages
        ]

    async def chat_completion(
        self,
        messages: list[ChatMessage],
        model: str = "meta-llama/Llama-3-8b-chat-hf",
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        """Generate a chat completion using Together AI API."""
        start_time = time.time()

        # Prepare request payload (OpenAI-compatible)
        payload = {"model": model, "messages": self._prepare_messages(messages), "stream": stream}

        # Add optional parameters
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ["messages", "model", "stream"] and value is not None:
                payload[key] = value

        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._create_headers(),
                timeout=60.0,
            )

            self._handle_http_error(response)
            response_data = response.json()

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Extract usage information
            usage = response_data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            # Fallback token estimation if not provided
            if input_tokens == 0:
                input_tokens = sum(estimate_tokens(msg.content, model) for msg in messages)
            if output_tokens == 0:
                output_tokens = estimate_tokens(
                    response_data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                    model,
                )

            # Extract content
            choices = response_data.get("choices", [])
            if not choices:
                raise ProviderError(
                    "No choices returned from Together AI", provider=self.provider_name
                )

            content = choices[0].get("message", {}).get("content", "")
            finish_reason = choices[0].get("finish_reason", "stop")

            # Create standardized response
            return self._create_standard_response(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                routing_reason="Selected Together AI for open-source models",
                response_time_ms=response_time_ms,
                finish_reason=finish_reason,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(
                    "Together AI API rate limit exceeded", provider=self.provider_name
                ) from e
            elif e.response.status_code == 401:
                raise AuthenticationError(
                    "Together AI API authentication failed", provider=self.provider_name
                ) from e
            else:
                error_detail = ""
                try:
                    error_data = e.response.json()
                    error_detail = error_data.get("error", {}).get("message", "")
                except (ValueError, KeyError, TypeError, AttributeError):
                    pass

                raise ProviderError(
                    f"Together AI API error: {e.response.status_code} {error_detail}",
                    provider=self.provider_name,
                    status_code=e.response.status_code,
                ) from e
        except httpx.RequestError as e:
            raise ProviderError(
                f"Together AI request failed: {str(e)}", provider=self.provider_name
            ) from e
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(
                f"Together AI unexpected error: {str(e)}", provider=self.provider_name
            ) from e

    async def stream_chat_completion(
        self,
        messages: list[ChatMessage],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a chat completion using Together AI API."""
        # Prepare request payload
        payload = {"model": model, "messages": self._prepare_messages(messages), "stream": True}

        # Add optional parameters
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ["messages", "model", "stream"] and value is not None:
                payload[key] = value

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._create_headers(),
                timeout=120.0,
            ) as response:
                self._handle_http_error(response)

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix

                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            yield chunk
                        except json.JSONDecodeError:
                            continue

        except httpx.RequestError as e:
            raise ProviderError(
                f"Together AI streaming request failed: {str(e)}", provider=self.provider_name
            ) from e
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(
                f"Together AI streaming unexpected error: {str(e)}", provider=self.provider_name
            ) from e

    async def health_check(self) -> bool:
        """Check if Together AI API is accessible."""
        try:
            test_messages = [ChatMessage(role="user", content="Hi", name=None)]
            await self.chat_completion(
                messages=test_messages, model="meta-llama/Llama-3-8b-chat-hf", max_tokens=1
            )
            return True
        except Exception as e:
            logger.warning("together_health_check_failed", error=str(e))
            return False

    def get_model_info(self, model: str) -> dict[str, Any]:
        """Get detailed information about a Together AI model."""
        model_info = {
            "meta-llama/Llama-3-70b-chat-hf": {
                "description": "Meta's Llama 3 70B chat model",
                "context_length": 8192,
                "strengths": ["reasoning", "code", "general tasks"],
                "provider": "Meta",
            },
            "meta-llama/Llama-3-8b-chat-hf": {
                "description": "Meta's Llama 3 8B chat model",
                "context_length": 8192,
                "strengths": ["speed", "efficiency", "general tasks"],
                "provider": "Meta",
            },
            "mistralai/Mixtral-8x7B-Instruct-v0.1": {
                "description": "Mistral's Mixtral 8x7B instruction model",
                "context_length": 32768,
                "strengths": ["multilingual", "code", "reasoning"],
                "provider": "Mistral AI",
            },
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {
                "description": "Nous Research's Hermes 2 based on Mixtral",
                "context_length": 32768,
                "strengths": ["instruction following", "reasoning"],
                "provider": "Nous Research",
            },
        }

        return model_info.get(
            model,
            {
                "description": "Open-source model via Together AI",
                "context_length": 4096,
                "strengths": ["open-source", "customizable"],
                "provider": "Various",
            },
        )
