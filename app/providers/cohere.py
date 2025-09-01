# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Optional

import httpx

from app.providers.base import (
    LLMProviderAdapter,
    ProviderResponse,
    SimpleCircuitBreaker,
    USER_AGENT,
    with_retries,
    normalize_finish_reason,
    _is_retryable_error,
)
from app.settings import settings
from app.telemetry.metrics import PROVIDER_LATENCY, PROVIDER_REQUESTS, TOKENS_TOTAL
from app.telemetry.tracing import start_span_async
from app.models import ChatMessage


class CohereAdapter(LLMProviderAdapter):
    def __init__(self, api_key: str, base_url: str = "https://api.cohere.ai/v1") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.circuit = SimpleCircuitBreaker(
            fail_threshold=settings.providers.circuit_fail_threshold,
            cooldown_sec=settings.providers.circuit_cooldown_sec,
        )
        self._client = httpx.AsyncClient(timeout=settings.providers.timeout_ms / 1000)

    def _convert_prompt_to_cohere_format(self, prompt: str) -> dict[str, Any]:
        """Convert simple prompt to Cohere format."""
        # For simple prompt, it's just the message with no chat history
        return {"message": prompt, "chat_history": []}

    def _convert_messages_to_cohere_format(
        self, messages: list[dict[str, Any] | ChatMessage]
    ) -> dict[str, Any]:
        """Convert OpenAI format messages to Cohere format.

        Args:
            messages: List of message dictionaries or ChatMessage objects

        Returns:
            Cohere format payload
        """
        chat_history = []
        system_message = None

        for msg in messages:
            # Support both dict and ChatMessage objects
            if hasattr(msg, "role") and hasattr(msg, "content"):
                # ChatMessage object
                role = msg.role.lower()
                content = msg.content
            else:
                # Dict object
                role = msg.get("role", "").lower()
                content = msg.get("content", "")

            if role == "system":
                system_message = content
            elif role == "user":
                # Add each user message to chat history
                chat_history.append({"role": "USER", "message": content})
            elif role == "assistant":
                # Add each assistant message to chat history
                chat_history.append({"role": "CHATBOT", "message": content})

        # Extract the last user message as the current message
        message = ""
        for i in range(len(chat_history) - 1, -1, -1):
            if chat_history[i]["role"] == "USER":
                message = chat_history.pop(i)["message"]
                break

        return {"message": message, "chat_history": chat_history, "system_message": system_message}

    async def invoke(self, model: str, prompt: str, **kwargs: Any) -> ProviderResponse:
        # Early validation for messages and system parameters
        messages = kwargs.get("messages")
        if messages is not None:
            if not isinstance(messages, list):
                raise ValueError("messages must be a list of dicts or ChatMessage objects")
            for i, msg in enumerate(messages):
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    # ChatMessage object - already validated by Pydantic
                    continue
                if isinstance(msg, dict):
                    if "role" not in msg or "content" not in msg:
                        raise ValueError(f"message {i} must have 'role' and 'content' keys")
                else:
                    raise TypeError(f"message {i} must be a dict or ChatMessage object")

        system = kwargs.get("system")
        if system is not None and not isinstance(system, str):
            raise ValueError("system parameter must be a string")

        start = time.perf_counter()
        provider = "cohere"

        if self.circuit.is_open():
            PROVIDER_REQUESTS.labels(provider=provider, model=model, outcome="circuit_open").inc()
            return ProviderResponse(
                output_text="",
                tokens_in=0,
                tokens_out=0,
                latency_ms=0,
                raw=None,
                error="circuit_open",
            )

        async with start_span_async("provider.invoke", provider=provider, model=model):
            try:

                async def make_request(attempt: int):
                    async with start_span_async("cohere.request", attempt=attempt):
                        # Check if messages are provided for chat history
                        messages = kwargs.get("messages")
                        if messages:
                            # Convert messages to Cohere format
                            cohere_format = self._convert_messages_to_cohere_format(messages)
                            payload = {
                                "model": model,
                                "message": cohere_format["message"],
                                "chat_history": cohere_format["chat_history"],
                                "stream": False,
                            }
                        else:
                            # Convert simple prompt to Cohere format
                            cohere_format = self._convert_prompt_to_cohere_format(prompt)
                            payload = {
                                "model": model,
                                "message": cohere_format["message"],
                                "chat_history": cohere_format["chat_history"],
                                "stream": False,
                            }

                        # Add optional parameters with normalized sampling parameters
                        payload["temperature"] = kwargs.get(
                            "temperature", settings.router.temperature_default
                        )
                        payload["max_tokens"] = kwargs.get(
                            "max_tokens", settings.router.max_tokens_default
                        )

                        # Cohere-specific sampling parameters
                        if "top_p" in kwargs:
                            payload["p"] = kwargs["top_p"]
                        if "top_k" in kwargs:
                            payload["k"] = kwargs["top_k"]

                        # Add stop sequences if provided
                        if "stop" in kwargs:
                            stop_list = kwargs["stop"]
                            if isinstance(stop_list, str):
                                stop_list = [stop_list]
                            payload["stop_sequences"] = stop_list

                        # Add preamble/system message - compute single preamble with precedence
                        preamble = None
                        if kwargs.get("system") and isinstance(kwargs["system"], str):
                            preamble = kwargs["system"]
                        elif kwargs.get("preamble") and isinstance(kwargs["preamble"], str):
                            preamble = kwargs["preamble"]
                        elif messages and cohere_format.get("system_message"):
                            # Fall back to system message from converted messages if no kwargs provided
                            preamble = cohere_format["system_message"]

                        if preamble:
                            payload["preamble"] = preamble

                        # Ensure message is non-empty to prevent API validation errors
                        if not payload.get("message"):
                            # try to promote last USER from chat_history
                            for i in range(len(payload.get("chat_history", [])) - 1, -1, -1):
                                if payload["chat_history"][i]["role"] == "USER":
                                    payload["message"] = payload["chat_history"].pop(i)["message"]
                                    break
                        if not payload.get("message") and isinstance(prompt, str) and prompt:
                            payload["message"] = prompt
                        if not payload.get("message"):
                            raise ValueError("Cohere payload requires non-empty 'message'")

                        headers = {
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "User-Agent": f"{USER_AGENT} (Cohere)",
                        }

                        resp = await self._client.post(
                            f"{self.base_url}/chat", json=payload, headers=headers
                        )

                        # Handle HTTP status errors for retry logic
                        try:
                            resp.raise_for_status()
                        except httpx.HTTPStatusError as e:
                            code = e.response.status_code
                            if code == 429 or code >= 500:
                                # signal retry
                                raise httpx.RequestError(
                                    f"retryable status: {code}", request=resp.request
                                ) from e
                            # non-retryable -> propagate
                            raise  # noqa: B904

                        data = resp.json()

                        # Check for error in response payload
                        if data.get("error"):
                            error_msg = data["error"].get("message", "Cohere AI returned an error")

                            # Use centralized retryable error detection
                            is_retryable = _is_retryable_error("cohere", resp.status_code, data)

                            if is_retryable:
                                raise httpx.RequestError(
                                    f"Cohere retryable error: {error_msg}", request=resp.request
                                )
                            else:
                                self.circuit.on_failure()
                                latency_ms = int((time.perf_counter() - start) * 1000)
                                PROVIDER_REQUESTS.labels(
                                    provider=provider, model=model, outcome="error"
                                ).inc()
                                PROVIDER_LATENCY.labels(provider=provider, model=model).observe(
                                    latency_ms
                                )
                                # Return ProviderResponse with error to let callers decide on fallbacks - adapter contract
                                return ProviderResponse(
                                    output_text="",
                                    tokens_in=0,
                                    tokens_out=0,
                                    latency_ms=latency_ms,
                                    raw=data,
                                    error=error_msg,
                                )

                        # Extract content from Cohere response
                        content = data.get("text", "")

                        # Get token usage from response if available
                        in_tokens = 0
                        out_tokens = 0

                        if "meta" in data and "tokens" in data["meta"]:
                            tokens_info = data["meta"]["tokens"]
                            in_tokens = int(tokens_info.get("input_tokens", 0))
                            out_tokens = int(tokens_info.get("output_tokens", 0))

                        # Map Cohere finish reasons to standard format
                        cohere_finish_reason = data.get("finish_reason") or data.get(
                            "meta", {}
                        ).get("finish_reason")
                        finish_reason = normalize_finish_reason("cohere", cohere_finish_reason)

                        # Add parsed finish reason to response data
                        data["_parsed_finish_reason"] = finish_reason

                        latency_ms = int((time.perf_counter() - start) * 1000)

                        # Metrics
                        PROVIDER_REQUESTS.labels(
                            provider=provider, model=model, outcome="success"
                        ).inc()
                        PROVIDER_LATENCY.labels(provider=provider, model=model).observe(latency_ms)
                        TOKENS_TOTAL.labels(provider=provider, model=model, type="input").inc(
                            in_tokens
                        )
                        TOKENS_TOTAL.labels(provider=provider, model=model, type="output").inc(
                            out_tokens
                        )

                        self.circuit.on_success()
                        return ProviderResponse(
                            output_text=content,
                            tokens_in=in_tokens,
                            tokens_out=out_tokens,
                            latency_ms=latency_ms,
                            raw=data,
                        )

                # Use centralized retry logic
                result = await with_retries(
                    make_request,
                    max_attempts=settings.providers.retry_max_attempts,
                    base_s=settings.providers.retry_base_ms / 1000,
                    retry_on=(httpx.TimeoutException, httpx.RequestError),
                )
                return result

            except httpx.HTTPStatusError as e:
                # Non-retryable HTTP errors
                error_msg = f"Non-retryable error: {e.response.status_code} - {str(e)}"
                self.circuit.on_failure()
                latency_ms = int((time.perf_counter() - start) * 1000)
                PROVIDER_REQUESTS.labels(provider=provider, model=model, outcome="error").inc()
                PROVIDER_LATENCY.labels(provider=provider, model=model).observe(latency_ms)
                # Return ProviderResponse with error to let callers decide on fallbacks - adapter contract
                return ProviderResponse(
                    output_text="",
                    tokens_in=0,
                    tokens_out=0,
                    latency_ms=latency_ms,
                    raw=None,
                    error=error_msg,
                )
            except Exception as e:  # pragma: no cover - safety net
                # Any other exceptions
                error_msg = str(e)
                self.circuit.on_failure()
                latency_ms = int((time.perf_counter() - start) * 1000)
                PROVIDER_REQUESTS.labels(provider=provider, model=model, outcome="error").inc()
                PROVIDER_LATENCY.labels(provider=provider, model=model).observe(latency_ms)
                # Return ProviderResponse with error to let callers decide on fallbacks - adapter contract
                return ProviderResponse(
                    output_text="",
                    tokens_in=0,
                    tokens_out=0,
                    latency_ms=latency_ms,
                    raw=None,
                    error=error_msg or "request_failed",
                )

    async def aclose(self) -> None:
        """Close the HTTP client to prevent connection leaks."""
        await self._client.aclose()

    def get_supported_models(self) -> list[str]:
        """Get the list of models supported by Cohere."""
        return ["command-r-plus", "command-r", "command", "command-light", "command-nightly"]
