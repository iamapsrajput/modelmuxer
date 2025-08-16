from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from app.settings import settings
from app.telemetry.metrics import LLM_LATENCY, LLM_REQUESTS, LLM_TOKENS_IN, LLM_TOKENS_OUT
from app.telemetry.tracing import start_span_async


@dataclass
class ProviderResponse:
    output_text: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    raw: Optional[Any] = None
    error: Optional[str] = None


class CircuitState:
    def __init__(self) -> None:
        self.failures = 0
        self.open_until: float = 0.0

    def is_open(self) -> bool:
        return time.time() < self.open_until

    def on_failure(self) -> None:
        self.failures += 1
        if self.failures >= settings.providers.circuit_fail_threshold:
            self.open_until = time.time() + settings.providers.circuit_cooldown_sec

    def on_success(self) -> None:
        self.failures = 0
        self.open_until = 0.0


class AnthropicAdapter:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.circuit = CircuitState()
        self._client = httpx.AsyncClient(timeout=settings.providers.timeout_ms / 1000)

    async def invoke(
        self, model: str, prompt: str, **kwargs: Any
    ) -> ProviderResponse:  # noqa: D401
        start = time.perf_counter()
        provider = "anthropic"

        if self.circuit.is_open():
            LLM_REQUESTS.labels(provider=provider, model=model, outcome="circuit_open").inc()
            return ProviderResponse(
                output_text="",
                tokens_in=0,
                tokens_out=0,
                latency_ms=0,
                raw=None,
                error="circuit_open",
            )

        attempt = 0
        error_msg: Optional[str] = None
        async with start_span_async("provider.invoke", provider=provider, model=model):
            while attempt < settings.providers.retry_max_attempts:
                attempt += 1
                try:
                    async with start_span_async("anthropic.request", attempt=attempt):
                        payload = {
                            "model": model,
                            "max_tokens": kwargs.get(
                                "max_tokens", settings.router.max_tokens_default
                            ),
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": kwargs.get(
                                "temperature", settings.router.temperature_default
                            ),
                        }
                        headers = {
                            "x-api-key": self.api_key,
                            "Content-Type": "application/json",
                        }
                        resp = await self._client.post(
                            f"{self.base_url}/v1/messages", json=payload, headers=headers
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        # Anthropic response parsing
                        text = ""
                        if data.get("content"):
                            parts = data["content"]
                            if isinstance(parts, list) and parts and "text" in parts[0]:
                                text = parts[0]["text"]
                        usage = data.get("usage", {})
                        in_tokens = int(usage.get("input_tokens", 0))
                        out_tokens = int(usage.get("output_tokens", 0))
                        latency_ms = int((time.perf_counter() - start) * 1000)

                        LLM_REQUESTS.labels(provider=provider, model=model, outcome="success").inc()
                        LLM_LATENCY.labels(provider=provider, model=model).observe(latency_ms)
                        LLM_TOKENS_IN.labels(provider=provider, model=model).inc(in_tokens)
                        LLM_TOKENS_OUT.labels(provider=provider, model=model).inc(out_tokens)

                        self.circuit.on_success()
                        return ProviderResponse(
                            output_text=text,
                            tokens_in=in_tokens,
                            tokens_out=out_tokens,
                            latency_ms=latency_ms,
                            raw=data,
                        )
                except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as e:
                    error_msg = str(e)
                    base = settings.providers.retry_base_ms / 1000
                    delay = base * (2 ** (attempt - 1)) + random.uniform(0, base)
                    if attempt >= settings.providers.retry_max_attempts:
                        break
                    await asyncio.sleep(delay)

            self.circuit.on_failure()
            latency_ms = int((time.perf_counter() - start) * 1000)
            LLM_REQUESTS.labels(provider=provider, model=model, outcome="error").inc()
            LLM_LATENCY.labels(provider=provider, model=model).observe(latency_ms)
            return ProviderResponse(
                output_text="",
                tokens_in=0,
                tokens_out=0,
                latency_ms=latency_ms,
                raw=None,
                error=error_msg or "request_failed",
            )
