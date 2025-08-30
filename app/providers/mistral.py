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
    _is_retryable_error,
    normalize_finish_reason,
)
from app.settings import settings
from app.telemetry.metrics import PROVIDER_LATENCY, PROVIDER_REQUESTS, TOKENS_TOTAL
from app.telemetry.tracing import start_span_async


class MistralAdapter(LLMProviderAdapter):
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
        provider = "mistral"

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
                    async with start_span_async("mistral.request", attempt=attempt):
                        payload = {
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": kwargs.get("temperature", settings.router.temperature_default),
                            "max_tokens": kwargs.get("max_tokens", settings.router.max_tokens_default),
                        }
                        headers = {
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "User-Agent": f"{USER_AGENT} (Mistral)",
                        }
                        resp = await self._client.post(
                            f"{self.base_url}/v1/chat/completions", json=payload, headers=headers
                        )

                        # Handle HTTP status errors for retry logic
                        try:
                            resp.raise_for_status()
                        except httpx.HTTPStatusError as e:
                            # Retry 429/5xx errors by re-raising as RequestError
                            if e.response.status_code in {429} or e.response.status_code >= 500:
                                raise httpx.RequestError(
                                    f"Retryable HTTP error: {e.response.status_code}",
                                    request=e.request,
                                )
                            # Non-retryable errors (401, 403, 404) are handled by outer exception handler
                            raise

                        data = resp.json()

                        # Check for error in response payload
                        if data.get("error"):
                            err = data["error"].get("message", "Provider returned an error")
                            if _is_retryable_error("mistral", resp.status_code, data):
                                raise httpx.RequestError(f"Retryable provider error: {err}", request=resp.request)
                            self.circuit.on_failure()
                            latency_ms = int((time.perf_counter() - start) * 1000)
                            PROVIDER_REQUESTS.labels(provider=provider, model=model, outcome="error").inc()
                            PROVIDER_LATENCY.labels(provider=provider, model=model).observe(latency_ms)
                            return ProviderResponse(
                                output_text="",
                                tokens_in=0,
                                tokens_out=0,
                                latency_ms=latency_ms,
                                raw=data,
                                error=err,
                            )

                        text = data["choices"][0]["message"]["content"]

                        # Extract and normalize finish reason
                        finish_reason = data["choices"][0].get("finish_reason")
                        data["_parsed_finish_reason"] = normalize_finish_reason("openai", finish_reason)

                        usage = data.get("usage", {})
                        in_tokens = int(usage.get("prompt_tokens", 0))
                        out_tokens = int(usage.get("completion_tokens", 0))
                        latency_ms = int((time.perf_counter() - start) * 1000)

                        # Metrics
                        PROVIDER_REQUESTS.labels(provider=provider, model=model, outcome="success").inc()
                        PROVIDER_LATENCY.labels(provider=provider, model=model).observe(latency_ms)
                        TOKENS_TOTAL.labels(provider=provider, model=model, type="input").inc(in_tokens)
                        TOKENS_TOTAL.labels(provider=provider, model=model, type="output").inc(out_tokens)

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
        """Get the list of models supported by Mistral."""
        return ["mistral-large-latest", "mistral-small-latest", "mistral-medium-latest", "codestral-latest"]
