# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Optional

import httpx

from app.models import ChatMessage
from app.providers.base import (USER_AGENT, LLMProviderAdapter,
                                ProviderResponse, SimpleCircuitBreaker,
                                _is_retryable_error, normalize_finish_reason,
                                with_retries)
from app.settings import settings
from app.telemetry.metrics import (PROVIDER_LATENCY, PROVIDER_REQUESTS,
                                   TOKENS_TOTAL)
from app.telemetry.tracing import start_span_async


class TogetherAdapter(LLMProviderAdapter):
    def __init__(self, api_key: str, base_url: str = "https://api.together.xyz/v1") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.circuit = SimpleCircuitBreaker(
            fail_threshold=settings.providers.circuit_fail_threshold,
            cooldown_sec=settings.providers.circuit_cooldown_sec,
        )
        self._client = httpx.AsyncClient(timeout=settings.providers.timeout_ms / 1000)

    def _normalize_messages(self, messages: list[dict[str, Any] | ChatMessage]) -> list[dict]:
        """Normalize messages to OpenAI format, handling both dict and ChatMessage objects.

        Args:
            messages: List of message dictionaries or ChatMessage objects

        Returns:
            List of normalized message dictionaries
        """
        norm = []
        for m in messages:
            if hasattr(m, "role") and hasattr(m, "content"):
                # Handle ChatMessage objects
                message_dict = {"role": m.role, "content": m.content}
                if hasattr(m, "name") and m.name is not None:
                    message_dict["name"] = m.name
                norm.append(message_dict)
            elif isinstance(m, dict):
                # Handle dict objects
                message_dict = {"role": m.get("role", "user"), "content": m.get("content", "")}
                if "name" in m and m["name"] is not None:
                    message_dict["name"] = m["name"]
                norm.append(message_dict)
        return norm

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
        provider = "together"

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
                    async with start_span_async("together.request", attempt=attempt):
                        # Check if messages are provided, otherwise use single user prompt
                        messages = kwargs.get("messages")
                        if messages and isinstance(messages, list) and len(messages) > 0:
                            payload_messages = self._normalize_messages(messages)
                        else:
                            payload_messages = [{"role": "user", "content": prompt}]

                        # Add system message if provided
                        if kwargs.get("system"):
                            payload_messages = [
                                {"role": "system", "content": kwargs["system"]}
                            ] + payload_messages

                        # Validate that we have non-empty content
                        if not payload_messages or all(
                            not m.get("content") for m in payload_messages
                        ):
                            if isinstance(prompt, str) and prompt:
                                payload_messages = [{"role": "user", "content": prompt}]
                            else:
                                raise ValueError(
                                    "Together payload requires a non-empty prompt or messages"
                                )

                        # Together AI uses OpenAI-compatible format
                        payload = {
                            "model": model,
                            "messages": payload_messages,
                            "stream": False,
                        }

                        # Add optional parameters with normalized sampling parameters
                        payload["temperature"] = kwargs.get(
                            "temperature", settings.router.temperature_default
                        )
                        payload["max_tokens"] = kwargs.get(
                            "max_tokens", settings.router.max_tokens_default
                        )

                        # OpenAI-compatible sampling parameters (Together AI supports these)
                        if "top_p" in kwargs:
                            payload["top_p"] = kwargs["top_p"]
                        if "frequency_penalty" in kwargs:
                            payload["frequency_penalty"] = kwargs["frequency_penalty"]
                        if "presence_penalty" in kwargs:
                            payload["presence_penalty"] = kwargs["presence_penalty"]
                        if "stop" in kwargs:
                            payload["stop"] = kwargs["stop"]

                        headers = {
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "User-Agent": f"{USER_AGENT} (Together AI)",
                        }

                        resp = await self._client.post(
                            f"{self.base_url}/chat/completions", json=payload, headers=headers
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
                            error_msg = data["error"].get(
                                "message", "Together AI returned an error"
                            )

                            # Use centralized retryable error detection
                            is_retryable = _is_retryable_error("together", resp.status_code, data)

                            if is_retryable:
                                raise httpx.RequestError(
                                    f"Together retryable error: {error_msg}", request=resp.request
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

                        # Extract content from Together AI response (OpenAI-compatible)
                        choices = data.get("choices", [])
                        if not choices:
                            raise httpx.RequestError(
                                "Together: no choices in response", request=resp.request
                            )

                        content = choices[0].get("message", {}).get("content", "")

                        # Map finish reason for consistency
                        finish_reason = choices[0].get("finish_reason")
                        data["_parsed_finish_reason"] = normalize_finish_reason(
                            "openai", finish_reason
                        )

                        # Extract token usage from standard usage object
                        usage = data.get("usage", {})
                        in_tokens = int(usage.get("prompt_tokens", 0))
                        out_tokens = int(usage.get("completion_tokens", 0))

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
        """Get the list of models supported by Together AI."""
        return [
            "meta-llama/Llama-3-70b-chat-hf",
            "meta-llama/Llama-3-8b-chat-hf",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            "togethercomputer/RedPajama-INCITE-7B-Chat",
            "teknium/OpenHermes-2.5-Mistral-7B",
            "Qwen/Qwen1.5-72B-Chat",
        ]
