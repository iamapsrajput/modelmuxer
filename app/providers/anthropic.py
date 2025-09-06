# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Optional

import httpx

from app.providers.base import (
    USER_AGENT,
    LLMProviderAdapter,
    ProviderResponse,
    SimpleCircuitBreaker,
    normalize_finish_reason,
    with_retries,
)
from app.settings import settings
from app.telemetry.metrics import PROVIDER_LATENCY, PROVIDER_REQUESTS, TOKENS_TOTAL
from app.telemetry.tracing import start_span_async


class AnthropicAdapter(LLMProviderAdapter):
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.circuit = SimpleCircuitBreaker(
            fail_threshold=settings.providers.circuit_fail_threshold,
            cooldown_sec=settings.providers.circuit_cooldown_sec,
        )
        self._client = httpx.AsyncClient(timeout=settings.providers.timeout_ms / 1000)

    async def invoke(self, model: str, prompt: str, **kwargs: Any) -> ProviderResponse:  # noqa: D401
        start = time.perf_counter()
        provider = "anthropic"

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
                    async with start_span_async("anthropic.request", attempt=attempt):
                        # Build messages with optional system prompt
                        messages = []
                        if kwargs.get("system"):
                            messages.append({
                                "role": "system",
                                "content": [{"type": "text", "text": kwargs["system"]}],
                            })
                        messages.append({
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        })

                        payload = {
                            "model": model,
                            "max_tokens": kwargs.get(
                                "max_tokens", settings.router.max_tokens_default
                            ),
                            "messages": messages,
                            "temperature": kwargs.get(
                                "temperature", settings.router.temperature_default
                            ),
                        }
                        headers = {
                            "x-api-key": self.api_key,
                            "anthropic-version": "2023-06-01",
                            "Content-Type": "application/json",
                            "User-Agent": f"{USER_AGENT} (Anthropic)",
                        }
                        resp = await self._client.post(
                            f"{self.base_url}/v1/messages", json=payload, headers=headers
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
                            raise

                        data = resp.json()
                        # Anthropic response parsing
                        text = ""
                        parts = data.get("content", [])
                        text = "".join(
                            p.get("text", "")
                            for p in parts
                            if isinstance(p, dict) and p.get("type") == "text"
                        )

                        # Extract and normalize finish reason
                        finish_reason = "stop"  # default
                        if data.get("stop_reason"):
                            finish_reason = normalize_finish_reason(
                                "anthropic", data["stop_reason"]
                            )
                        data["_parsed_finish_reason"] = finish_reason

                        usage = data.get("usage", {})
                        in_tokens = int(usage.get("input_tokens", 0))
                        out_tokens = int(usage.get("output_tokens", 0))
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
                    max_attempts=settings.providers.retry_max_attempts,
                    base_s=settings.providers.retry_base_ms / 1000,
                    retry_on=(httpx.TimeoutException, httpx.RequestError),
                )
                return result

            except Exception as e:
                self.circuit.on_failure()
                PROVIDER_REQUESTS.labels(provider=provider, model=model, outcome="error").inc()
                return ProviderResponse(
                    output_text="",
                    tokens_in=0,
                    tokens_out=0,
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    raw=None,
                    error=str(e),
                )

    async def aclose(self) -> None:
        """Close the HTTP client to prevent connection leaks."""
        await self._client.aclose()

    def get_supported_models(self) -> list[str]:
        """Get the list of models supported by Anthropic."""
        return [
            "claude-3-5-sonnet-latest",
            "claude-3-5-sonnet-20241022",
            "claude-3-sonnet",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-haiku-20241022",
        ]
