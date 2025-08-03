# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Cohere provider implementation.

This module provides integration with Cohere's API for the ModelMuxer
LLM routing system.
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


class CohereProvider(LLMProvider):
    """Cohere API provider implementation."""

    def __init__(self, api_key: str | None = None):
        if not api_key:
            raise AuthenticationError("Cohere API key is required")

        super().__init__(
            api_key=api_key, base_url="https://api.cohere.ai/v1", provider_name="cohere"
        )

        # Pricing per million tokens (as of 2024)
        self.pricing = {
            "command-r-plus": {"input": 3.0, "output": 15.0},
            "command-r": {"input": 0.5, "output": 1.5},
            "command": {"input": 1.0, "output": 2.0},
            "command-light": {"input": 0.3, "output": 0.6},
        }

        self.supported_models = list(self.pricing.keys())

        # Rate limits (requests per minute)
        self.rate_limits = {
            "command-r-plus": 100,
            "command-r": 1000,
            "command": 1000,
            "command-light": 1000,
        }

    def _create_headers(self) -> dict[str, str]:
        """Create headers for Cohere API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ModelMuxer/1.0.0 (Cohere)",
        }

    def get_supported_models(self) -> list[str]:
        """Get list of supported Cohere models."""
        return self.supported_models

    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost for Cohere request."""
        if model not in self.pricing:
            model = "command-r"  # Default fallback

        model_pricing = self.pricing[model]
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

        return input_cost + output_cost

    def get_rate_limits(self) -> dict[str, Any]:
        """Get rate limit information."""
        return {
            "requests_per_minute": self.rate_limits,
            "tokens_per_minute": {
                "command-r-plus": 40000,
                "command-r": 100000,
                "command": 100000,
                "command-light": 100000,
            },
        }

    def _convert_messages_to_cohere_format(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """Convert OpenAI format messages to Cohere format."""
        # Cohere uses a different format - it expects a message and chat_history
        if not messages:
            return {"message": "", "chat_history": []}

        # The last message is the current message
        current_message = messages[-1].content

        # Previous messages become chat history
        chat_history = []
        for msg in messages[:-1]:
            if msg.role == "user":
                chat_history.append({"role": "USER", "message": msg.content})
            elif msg.role == "assistant":
                chat_history.append({"role": "CHATBOT", "message": msg.content})
            # Skip system messages for now (Cohere handles them differently)

        return {"message": current_message, "chat_history": chat_history}

    async def chat_completion(
        self,
        messages: list[ChatMessage],
        model: str = "command-r",
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        """Generate a chat completion using Cohere API."""
        start_time = time.time()

        # Convert messages to Cohere format
        cohere_messages = self._convert_messages_to_cohere_format(messages)

        # Prepare request payload
        payload = {
            "model": model,
            "message": cohere_messages["message"],
            "chat_history": cohere_messages["chat_history"],
            "stream": stream,
        }

        # Add optional parameters
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        # Add system message if present
        system_messages = [msg for msg in messages if msg.role == "system"]
        if system_messages:
            payload["preamble"] = system_messages[0].content

        try:
            response = await self.client.post(
                f"{self.base_url}/chat", json=payload, headers=self._create_headers(), timeout=60.0
            )

            self._handle_http_error(response)
            response_data = response.json()

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Extract content from Cohere response
            content = response_data.get("text", "")
            finish_reason = "stop"

            # Check for finish reason
            if "finish_reason" in response_data:
                cohere_finish_reason = response_data["finish_reason"]
                finish_reason_mapping = {
                    "COMPLETE": "stop",
                    "MAX_TOKENS": "length",
                    "ERROR": "stop",
                    "ERROR_TOXIC": "content_filter",
                }
                finish_reason = finish_reason_mapping.get(cohere_finish_reason, "stop")

            # Get token usage
            input_tokens = 0
            output_tokens = 0

            if "meta" in response_data and "tokens" in response_data["meta"]:
                tokens_info = response_data["meta"]["tokens"]
                input_tokens = tokens_info.get("input_tokens", 0)
                output_tokens = tokens_info.get("output_tokens", 0)

            # Fallback token estimation if not provided
            if input_tokens == 0:
                input_tokens = sum(estimate_tokens(msg.content, model) for msg in messages)
            if output_tokens == 0:
                output_tokens = estimate_tokens(content, model)

            # Create standardized response
            return self._create_standard_response(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                routing_reason="Selected Cohere for request",
                response_time_ms=response_time_ms,
                finish_reason=finish_reason,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(
                    "Cohere API rate limit exceeded", provider=self.provider_name
                ) from e
            elif e.response.status_code == 401:
                raise AuthenticationError(
                    "Cohere API authentication failed", provider=self.provider_name
                ) from e
            else:
                error_detail = ""
                try:
                    error_data = e.response.json()
                    error_detail = error_data.get("message", "")
                except (ValueError, KeyError, TypeError, AttributeError):
                    pass

                raise ProviderError(
                    f"Cohere API error: {e.response.status_code} {error_detail}",
                    provider=self.provider_name,
                    status_code=e.response.status_code,
                ) from e
        except httpx.RequestError as e:
            raise ProviderError(
                f"Cohere request failed: {str(e)}", provider=self.provider_name
            ) from e
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(
                f"Cohere unexpected error: {str(e)}", provider=self.provider_name
            ) from e

    async def stream_chat_completion(
        self,
        messages: list[ChatMessage],
        model: str = "command-r",
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a chat completion using Cohere API."""
        # Convert messages to Cohere format
        cohere_messages = self._convert_messages_to_cohere_format(messages)

        # Prepare request payload
        payload = {
            "model": model,
            "message": cohere_messages["message"],
            "chat_history": cohere_messages["chat_history"],
            "stream": True,
        }

        # Add optional parameters
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        # Add system message if present
        system_messages = [msg for msg in messages if msg.role == "system"]
        if system_messages:
            payload["preamble"] = system_messages[0].content

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat",
                json=payload,
                headers=self._create_headers(),
                timeout=120.0,
            ) as response:
                self._handle_http_error(response)

                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            # Cohere returns JSON objects separated by newlines
                            chunk_data = json.loads(line)

                            # Convert to OpenAI streaming format
                            if "text" in chunk_data:
                                text = chunk_data["text"]

                                openai_chunk = {
                                    "id": f"chatcmpl-cohere-{int(time.time())}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": text},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield openai_chunk

                            # Check for finish reason
                            if "finish_reason" in chunk_data:
                                final_chunk = {
                                    "id": f"chatcmpl-cohere-{int(time.time())}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                }
                                yield final_chunk
                                break

                        except json.JSONDecodeError:
                            continue

        except httpx.RequestError as e:
            raise ProviderError(
                f"Cohere streaming request failed: {str(e)}", provider=self.provider_name
            ) from e
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(
                f"Cohere streaming unexpected error: {str(e)}", provider=self.provider_name
            ) from e

    async def health_check(self) -> bool:
        """Check if Cohere API is accessible."""
        try:
            test_messages = [ChatMessage(role="user", content="Hi", name=None)]
            await self.chat_completion(messages=test_messages, model="command-light", max_tokens=1)
            return True
        except Exception as e:
            logger.warning("cohere_health_check_failed", error=str(e))
            return False
