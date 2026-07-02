# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
from __future__ import annotations

import json
import random
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Optional

import httpx

from app.models import ChatMessage
from app.providers.base import (
    LLMProviderAdapter,
    ProviderError,
    ProviderResponse,
    SimpleCircuitBreaker,
    build_stream_chunk,
    messages_to_openai_format,
    with_retries,
)
from app.settings import settings
from app.telemetry.metrics import PROVIDER_LATENCY, PROVIDER_REQUESTS, TOKENS_TOTAL
from app.telemetry.tracing import start_span_async


class OpenAIAdapter(LLMProviderAdapter):
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.circuit = SimpleCircuitBreaker(
            fail_threshold=settings.providers.circuit_fail_threshold,
            cooldown_sec=settings.providers.circuit_cooldown_sec,
        )
        self._client = httpx.AsyncClient(timeout=settings.providers.timeout_ms / 1000)

    async def invoke(
        self, model: str, prompt: str, **kwargs: Any
    ) -> ProviderResponse:  # noqa: D401
        start = time.perf_counter()
        provider = "openai"

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
                    async with start_span_async("openai.request", attempt=attempt):
                        use_responses = model in {"o1", "o1-mini"}

                        if use_responses:
                            # Use OpenAI Responses API for o1 models
                            payload = {
                                "model": model,
                                "input": prompt,
                                "temperature": kwargs.get(
                                    "temperature", settings.router.temperature_default
                                ),
                                "max_output_tokens": kwargs.get(
                                    "max_tokens", settings.router.max_tokens_default
                                ),
                            }
                            headers = {
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json",
                            }
                            resp = await self._client.post(
                                f"{self.base_url}/responses", json=payload, headers=headers
                            )
                        else:
                            # Use standard chat completions API
                            payload = {
                                "model": model,
                                "messages": [{"role": "user", "content": prompt}],
                                "temperature": kwargs.get(
                                    "temperature", settings.router.temperature_default
                                ),
                                "max_tokens": kwargs.get(
                                    "max_tokens", settings.router.max_tokens_default
                                ),
                            }
                            headers = {
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json",
                            }
                            resp = await self._client.post(
                                f"{self.base_url}/chat/completions", json=payload, headers=headers
                            )

                        # Handle HTTP status errors for retry logic
                        try:
                            resp.raise_for_status()
                        except httpx.HTTPStatusError as e:
                            # Retry 429/5xx errors by re-raising as RequestError
                            if e.response.status_code == 429 or e.response.status_code >= 500:
                                raise httpx.RequestError(
                                    f"Retryable HTTP error: {e.response.status_code}",
                                    request=e.request,
                                ) from e
                            # Non-retryable errors (401, 403, 404) are handled by outer exception handler
                            raise  # noqa: B904

                        data = resp.json()

                        if use_responses:
                            # Parse Responses API format
                            text = ""
                            out = data.get("output", [])
                            for item in out:
                                for part in item.get("content", []):
                                    if part.get("type") == "output_text":
                                        text += part.get("text", "")
                            usage = data.get("usage", {})
                            in_tokens = int(usage.get("input_tokens", 0))
                            out_tokens = int(usage.get("output_tokens", 0))
                        else:
                            # Parse standard chat completions format
                            text = data["choices"][0]["message"]["content"]
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
                            output_text=text,
                            tokens_in=in_tokens,
                            tokens_out=out_tokens,
                            latency_ms=latency_ms,
                            raw=data,
                        )

                # Use centralized retry logic
                result = await with_retries(
                    make_request,
                    max_attempts=settings.adapters.retry_max_attempts,
                    base_s=settings.adapters.retry_base_ms / 1000,
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
                return ProviderResponse(
                    output_text="",
                    tokens_in=0,
                    tokens_out=0,
                    latency_ms=latency_ms,
                    raw=None,
                    error=error_msg or "request_failed",
                )

    async def invoke_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream OpenAI chat completion chunks via provider-native SSE."""
        provider = "openai"
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if self.circuit.is_open():
            PROVIDER_REQUESTS.labels(provider=provider, model=model, outcome="circuit_open").inc()
            raise ProviderError("circuit_open", provider=provider)

        if model in {"o1", "o1-mini"}:
            async for chunk in super().invoke_stream(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            ):
                yield chunk
            return

        payload = {
            "model": model,
            "messages": messages_to_openai_format(messages),
            "temperature": (
                temperature if temperature is not None else settings.router.temperature_default
            ),
            "max_tokens": (
                max_tokens if max_tokens is not None else settings.router.max_tokens_default
            ),
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        start = time.perf_counter()
        saw_content = False
        finish_reason: str | None = None

        async with start_span_async("provider.invoke_stream", provider=provider, model=model):
            try:
                async with self._client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        for choice in event.get("choices", []):
                            delta = choice.get("delta") or {}
                            if delta.get("role") and not saw_content:
                                yield build_stream_chunk(
                                    chunk_id=event.get("id", chunk_id),
                                    model=event.get("model", model),
                                    role=delta["role"],
                                    created=event.get("created", created),
                                )
                            if delta.get("content"):
                                saw_content = True
                                yield build_stream_chunk(
                                    chunk_id=event.get("id", chunk_id),
                                    model=event.get("model", model),
                                    content=delta["content"],
                                    created=event.get("created", created),
                                )
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]

                latency_ms = int((time.perf_counter() - start) * 1000)
                PROVIDER_REQUESTS.labels(provider=provider, model=model, outcome="success").inc()
                PROVIDER_LATENCY.labels(provider=provider, model=model).observe(latency_ms)
                self.circuit.on_success()
                yield build_stream_chunk(
                    chunk_id=chunk_id,
                    model=model,
                    finish_reason=finish_reason or "stop",
                    created=created,
                )
            except httpx.HTTPStatusError as e:
                self.circuit.on_failure()
                PROVIDER_REQUESTS.labels(provider=provider, model=model, outcome="error").inc()
                raise ProviderError(
                    f"Non-retryable error: {e.response.status_code} - {e}",
                    status_code=e.response.status_code,
                    provider=provider,
                ) from e
            except Exception as e:
                self.circuit.on_failure()
                PROVIDER_REQUESTS.labels(provider=provider, model=model, outcome="error").inc()
                raise ProviderError(str(e), provider=provider) from e

    async def aclose(self) -> None:
        """Close the HTTP client to prevent connection leaks."""
        await self._client.aclose()

    def get_supported_models(self) -> list[str]:
        """Get the list of models supported by OpenAI."""
        return ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o1-mini"]
