# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.models import ChatMessage
from app.providers.base import (
    USER_AGENT,
    LLMProviderAdapter,
    ProviderError,
    ProviderResponse,
    SimpleCircuitBreaker,
    build_stream_chunk,
    messages_to_openai_format,
    normalize_finish_reason,
    with_retries,
)
from app.settings import settings
from app.telemetry.metrics import PROVIDER_LATENCY, PROVIDER_REQUESTS, TOKENS_TOTAL
from app.telemetry.tracing import start_span_async


class OllamaAdapter(LLMProviderAdapter):
    """Adapter for local Ollama instances via the OpenAI-compatible API."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.circuit = SimpleCircuitBreaker(
            fail_threshold=settings.providers.circuit_fail_threshold,
            cooldown_sec=settings.providers.circuit_cooldown_sec,
        )
        self._client = httpx.AsyncClient(timeout=settings.providers.timeout_ms / 1000)
        self._cached_models: list[str] = []

    def _headers(self, *, stream: bool = False) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"{USER_AGENT} (Ollama)",
        }
        if stream:
            headers["Accept"] = "text/event-stream"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _chat_completions_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    async def invoke(
        self, model: str, messages: list[ChatMessage], **kwargs: Any
    ) -> ProviderResponse:
        start = time.perf_counter()
        provider = "ollama"

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
                    async with start_span_async("ollama.request", attempt=attempt):
                        payload: dict[str, Any] = {
                            "model": model,
                            "messages": messages_to_openai_format(messages),
                            "temperature": kwargs.get(
                                "temperature", settings.router.temperature_default
                            ),
                            "max_tokens": kwargs.get(
                                "max_tokens", settings.router.max_tokens_default
                            ),
                            "stream": False,
                        }
                        if kwargs.get("tools"):
                            payload["tools"] = kwargs["tools"]
                        if kwargs.get("tool_choice") is not None:
                            payload["tool_choice"] = kwargs["tool_choice"]

                        resp = await self._client.post(
                            self._chat_completions_url(),
                            json=payload,
                            headers=self._headers(),
                        )

                        try:
                            resp.raise_for_status()
                        except httpx.HTTPStatusError as e:
                            if e.response.status_code == 429 or e.response.status_code >= 500:
                                raise httpx.RequestError(
                                    f"Retryable HTTP error: {e.response.status_code}",
                                    request=e.request,
                                ) from e
                            raise  # noqa: B904

                        data = resp.json()
                        choice = data["choices"][0]
                        message = choice.get("message", {})
                        text = message.get("content") or ""
                        tool_calls = message.get("tool_calls")
                        finish_reason = choice.get("finish_reason") or (
                            "tool_calls" if tool_calls else "stop"
                        )
                        usage = data.get("usage", {})
                        in_tokens = int(usage.get("prompt_tokens", 0))
                        out_tokens = int(usage.get("completion_tokens", 0))
                        latency_ms = int((time.perf_counter() - start) * 1000)

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
                            tool_calls=tool_calls,
                            finish_reason=normalize_finish_reason("openai", finish_reason),
                        )

                return await with_retries(
                    make_request,
                    max_attempts=settings.adapters.retry_max_attempts,
                    base_s=settings.adapters.retry_base_ms / 1000,
                    retry_on=(httpx.TimeoutException, httpx.RequestError),
                )

            except httpx.HTTPStatusError as e:
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
                    error=f"Non-retryable error: {e.response.status_code} - {e}",
                )
            except Exception as e:
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
                    error=str(e) or "request_failed",
                )

    async def invoke_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream Ollama chat completion chunks via OpenAI-compatible SSE."""
        provider = "ollama"
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if self.circuit.is_open():
            PROVIDER_REQUESTS.labels(provider=provider, model=model, outcome="circuit_open").inc()
            raise ProviderError("circuit_open", provider=provider)

        payload: dict[str, Any] = {
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
        if kwargs.get("tools"):
            payload["tools"] = kwargs["tools"]
        if kwargs.get("tool_choice") is not None:
            payload["tool_choice"] = kwargs["tool_choice"]

        start = time.perf_counter()
        saw_content = False
        finish_reason: str | None = None

        async with start_span_async("provider.invoke_stream", provider=provider, model=model):
            try:
                async with self._client.stream(
                    "POST",
                    self._chat_completions_url(),
                    json=payload,
                    headers=self._headers(stream=True),
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
        await self._client.aclose()

    def get_supported_models(self) -> list[str]:
        """Return models reported by Ollama /api/tags (cached on success)."""
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                models = [item["name"] for item in data.get("models", []) if item.get("name")]
                if models:
                    self._cached_models = models
                return models or self._cached_models
        except Exception:
            return self._cached_models
