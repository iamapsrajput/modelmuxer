# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Mistral provider implementation.
"""

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from ..models import ChatCompletionResponse, ChatMessage
from .base import LLMProvider, ProviderError


class MistralProvider(LLMProvider):
    """Mistral API provider implementation."""

    def __init__(self, api_key: str = None):
        if not api_key:
            raise ValueError("Mistral API key is required")

        super().__init__(api_key=api_key, base_url="https://api.mistral.ai/v1", provider_name="mistral")

        # Pricing per million tokens (updated as of 2024)
        self.pricing = {
            "mistral-small-latest": {"input": 0.2, "output": 0.6},
            "mistral-medium-latest": {"input": 2.7, "output": 8.1},
            "mistral-large-latest": {"input": 8.0, "output": 24.0},
        }

        self.supported_models = list(self.pricing.keys())

    def _create_headers(self) -> dict[str, str]:
        """Create headers for Mistral API requests."""
        headers = super()._create_headers()
        headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_supported_models(self) -> list[str]:
        """Get list of supported Mistral models."""
        return self.supported_models

    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost for Mistral request."""
        if model not in self.pricing:
            return 0.0

        model_pricing = self.pricing[model]
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

        return input_cost + output_cost

    def _prepare_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        """Convert ChatMessage objects to Mistral format (same as OpenAI)."""
        return [
            {"role": msg.role, "content": msg.content, **({"name": msg.name} if msg.name else {})} for msg in messages
        ]

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation for Mistral (similar to OpenAI)."""
        # Mistral uses roughly 4 characters per token
        return len(text) // 4

    async def chat_completion(
        self,
        messages: list[ChatMessage],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
        **kwargs,
    ) -> ChatCompletionResponse:
        """Generate a chat completion using Mistral API."""
        start_time = time.time()

        # Prepare request payload
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
                f"{self.base_url}/chat/completions", headers=self._create_headers(), json=payload
            )

            self._handle_http_error(response)
            response_data = response.json()

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Extract usage information
            usage = response_data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            # If usage not provided, estimate
            if input_tokens == 0:
                input_text = " ".join([msg.content for msg in messages])
                input_tokens = self._estimate_tokens(input_text)

            # Extract content
            choices = response_data.get("choices", [])
            if not choices:
                raise ProviderError("No choices returned from Mistral", provider=self.provider_name)

            content = choices[0].get("message", {}).get("content", "")
            finish_reason = choices[0].get("finish_reason", "stop")

            if output_tokens == 0:
                output_tokens = self._estimate_tokens(content)

            # Create standardized response
            return self._create_standard_response(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                routing_reason="Selected Mistral for request",
                response_time_ms=response_time_ms,
                finish_reason=finish_reason,
            )

        except httpx.RequestError as e:
            raise ProviderError(f"Mistral request failed: {str(e)}", provider=self.provider_name) from e
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(f"Mistral unexpected error: {str(e)}", provider=self.provider_name) from e

    async def stream_chat_completion(
        self,
        messages: list[ChatMessage],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a chat completion using Mistral API."""
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
                headers=self._create_headers(),
                json=payload,
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
            raise ProviderError(f"Mistral streaming request failed: {str(e)}", provider=self.provider_name) from e
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(f"Mistral streaming unexpected error: {str(e)}", provider=self.provider_name) from e
