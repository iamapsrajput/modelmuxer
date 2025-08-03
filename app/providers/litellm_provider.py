# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
LiteLLM proxy provider implementation.

This module provides integration with LiteLLM proxy servers for the ModelMuxer
LLM routing system. LiteLLM allows organizations to proxy multiple LLM providers
through a single endpoint.
"""

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import structlog

from ..core.utils import estimate_tokens
from ..models import ChatCompletionResponse, ChatMessage
from .base import AuthenticationError, LLMProvider, ProviderError, RateLimitError

logger = structlog.get_logger(__name__)


class LiteLLMProvider(LLMProvider):
    """LiteLLM proxy provider implementation."""

    def __init__(
        self, base_url: str, api_key: str | None = None, custom_models: dict[str, dict] | None = None
    ):
        if not base_url:
            raise ValueError("LiteLLM base URL is required")

        # Use a clearly marked test placeholder if no API key provided
        # This is NOT a real secret - it's an obvious test placeholder
        test_placeholder = "TEST-PLACEHOLDER-NOT-A-REAL-SECRET-FOR-LITELLM-TESTING-ONLY"
        super().__init__(
            api_key=api_key or test_placeholder,
            base_url=base_url.rstrip("/"),
            provider_name="litellm",
        )

        # Custom model configurations
        self.custom_models = custom_models or {}

        # Default pricing (organizations should configure their own)
        self.default_pricing = {"input": 1.0, "output": 2.0}  # Per million tokens

        # Extract model names and pricing
        self.supported_models = list(self.custom_models.keys())
        self.pricing = {}

        for model_name, model_config in self.custom_models.items():
            self.pricing[model_name] = model_config.get("pricing", self.default_pricing)

        # Default rate limits (can be overridden per model)
        self.default_rate_limits = {"requests_per_minute": 100, "tokens_per_minute": 100000}

    def _create_headers(self) -> dict[str, str]:
        """Create headers for LiteLLM proxy requests."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ModelMuxer/1.0.0 (LiteLLM Proxy)",
        }

        # Only add auth header if we have a real API key (not test placeholder)
        if (
            self.api_key
            and not self.api_key.startswith("TEST-PLACEHOLDER")
            and self.api_key != "dummy-key"
        ):
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def get_supported_models(self) -> list[str]:
        """Get list of supported models from LiteLLM proxy."""
        return self.supported_models

    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost based on configured pricing."""
        model_pricing = self.pricing.get(model, self.default_pricing)

        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

        return input_cost + output_cost

    def get_rate_limits(self) -> dict[str, Any]:
        """Get rate limit information."""
        rate_limits = {}

        for model_name, model_config in self.custom_models.items():
            rate_limits[model_name] = model_config.get("rate_limits", self.default_rate_limits)

        return {
            "requests_per_minute": rate_limits,
            "note": "Rate limits are configured per model in LiteLLM proxy",
        }

    def _prepare_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        """Convert ChatMessage objects to OpenAI format (LiteLLM is OpenAI-compatible)."""
        return [
            {"role": msg.role, "content": msg.content, **({"name": msg.name} if msg.name else {})}
            for msg in messages
        ]

    async def chat_completion(
        self,
        messages: list[ChatMessage],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
        **kwargs,
    ) -> ChatCompletionResponse:
        """Generate a chat completion using LiteLLM proxy."""
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
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self._create_headers(),
                timeout=120.0,  # Longer timeout for proxy
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
                    "No choices returned from LiteLLM proxy", provider=self.provider_name
                )

            content = choices[0].get("message", {}).get("content", "")
            finish_reason = choices[0].get("finish_reason", "stop")

            # Create standardized response
            return self._create_standard_response(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                routing_reason="Selected LiteLLM proxy for request",
                response_time_ms=response_time_ms,
                finish_reason=finish_reason,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(
                    "LiteLLM proxy rate limit exceeded", provider=self.provider_name
                ) from e
            elif e.response.status_code == 401:
                raise AuthenticationError(
                    "LiteLLM proxy authentication failed", provider=self.provider_name
                ) from e
            else:
                error_detail = ""
                try:
                    error_data = e.response.json()
                    error_detail = error_data.get("error", {}).get("message", "")
                except (ValueError, KeyError, TypeError, AttributeError):
                    pass

                raise ProviderError(
                    f"LiteLLM proxy error: {e.response.status_code} {error_detail}",
                    provider=self.provider_name,
                    status_code=e.response.status_code,
                ) from e
        except httpx.RequestError as e:
            raise ProviderError(
                f"LiteLLM proxy request failed: {str(e)}", provider=self.provider_name
            ) from e
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(
                f"LiteLLM proxy unexpected error: {str(e)}", provider=self.provider_name
            ) from e

    async def stream_chat_completion(
        self,
        messages: list[ChatMessage],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a chat completion using LiteLLM proxy."""
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
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self._create_headers(),
                timeout=180.0,  # Longer timeout for proxy streaming
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
                f"LiteLLM proxy streaming request failed: {str(e)}", provider=self.provider_name
            ) from e
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(
                f"LiteLLM proxy streaming unexpected error: {str(e)}", provider=self.provider_name
            ) from e

    async def health_check(self) -> bool:
        """Check if LiteLLM proxy is accessible."""
        try:
            # Try to get model list first
            response = await self.client.get(
                f"{self.base_url}/v1/models", headers=self._create_headers(), timeout=30.0
            )

            if response.status_code == 200:
                return True

            # Fallback: try a simple chat completion
            if self.supported_models:
                test_messages = [ChatMessage(role="user", content="Hi")]
                await self.chat_completion(
                    messages=test_messages, model=self.supported_models[0], max_tokens=1
                )
                return True

            return False

        except Exception as e:
            logger.warning("litellm_health_check_failed", error=str(e))
            return False

    async def get_available_models(self) -> list[dict[str, Any]]:
        """Get available models from LiteLLM proxy."""
        try:
            response = await self.client.get(
                f"{self.base_url}/v1/models", headers=self._create_headers(), timeout=30.0
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])

            return []

        except Exception as e:
            logger.warning("failed_to_get_litellm_models", error=str(e))
            return []

    def add_custom_model(
        self,
        model_name: str,
        pricing: dict[str, float],
        rate_limits: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a custom model configuration."""
        self.custom_models[model_name] = {
            "pricing": pricing,
            "rate_limits": rate_limits or self.default_rate_limits,
            "metadata": metadata or {},
        }

        self.supported_models.append(model_name)
        self.pricing[model_name] = pricing

        logger.info(
            "custom_model_added", model=model_name, pricing=pricing, rate_limits=rate_limits
        )

    def get_model_info(self, model: str) -> dict[str, Any]:
        """Get detailed information about a model."""
        if model in self.custom_models:
            model_config = self.custom_models[model]
            return {
                "model": model,
                "provider": "litellm",
                "pricing": model_config.get("pricing", self.default_pricing),
                "rate_limits": model_config.get("rate_limits", self.default_rate_limits),
                "metadata": model_config.get("metadata", {}),
                "proxy_url": self.base_url,
            }

        return {
            "model": model,
            "provider": "litellm",
            "pricing": self.default_pricing,
            "rate_limits": self.default_rate_limits,
            "metadata": {},
            "proxy_url": self.base_url,
        }
