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
    normalize_finish_reason,
    openai_tools_to_anthropic,
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

    def _parse_tool_arguments(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                return parsed if isinstance(parsed, dict) else {"raw": arguments}
            except json.JSONDecodeError:
                return {"raw": arguments}
        return {}

    def _anthropic_messages_from_chat(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Split system prompt and convert chat messages to Anthropic format."""
        system_prompt: str | None = None
        anthropic_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content or ""
                continue
            if msg.role == "tool":
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id or "",
                                "content": msg.content or "",
                            }
                        ],
                    }
                )
                continue
            if msg.role == "assistant" and msg.tool_calls:
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tool_call in msg.tool_calls:
                    fn = tool_call.get("function", {})
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": self._parse_tool_arguments(fn.get("arguments", "{}")),
                        }
                    )
                anthropic_messages.append({"role": "assistant", "content": content_blocks})
                continue
            role = "assistant" if msg.role == "assistant" else "user"
            anthropic_messages.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": msg.content or ""}],
                }
            )
        if not anthropic_messages:
            anthropic_messages = [
                {"role": "user", "content": [{"type": "text", "text": ""}]},
            ]
        return system_prompt, anthropic_messages

    def _parse_anthropic_response(
        self, data: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]] | None, str]:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for part in data.get("content", []):
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": part.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": part.get("name", ""),
                            "arguments": json.dumps(part.get("input", {})),
                        },
                    }
                )
        text = "".join(text_parts)
        stop_reason = data.get("stop_reason")
        if stop_reason == "tool_use" and tool_calls:
            finish_reason = "tool_calls"
        elif stop_reason:
            finish_reason = normalize_finish_reason("anthropic", stop_reason)
        else:
            finish_reason = "tool_calls" if tool_calls else "stop"
        return text, tool_calls or None, finish_reason

    def _build_payload(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int | None,
        temperature: float | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        system_prompt, anthropic_messages = self._anthropic_messages_from_chat(messages)
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": (
                max_tokens if max_tokens is not None else settings.router.max_tokens_default
            ),
            "messages": anthropic_messages,
            "temperature": (
                temperature if temperature is not None else settings.router.temperature_default
            ),
        }
        if system_prompt:
            payload["system"] = system_prompt
        if kwargs.get("tools"):
            payload["tools"] = openai_tools_to_anthropic(kwargs["tools"])
        tool_choice = kwargs.get("tool_choice")
        if tool_choice == "required":
            payload["tool_choice"] = {"type": "any"}
        elif tool_choice == "none":
            payload["tool_choice"] = {"type": "none"}
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            fn = tool_choice.get("function", {})
            if fn.get("name"):
                payload["tool_choice"] = {"type": "tool", "name": fn["name"]}
        return payload

    async def invoke(
        self, model: str, messages: list[ChatMessage], **kwargs: Any
    ) -> ProviderResponse:  # noqa: D401
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
                        payload = self._build_payload(
                            model,
                            messages,
                            kwargs.get("max_tokens"),
                            kwargs.get("temperature"),
                            **kwargs,
                        )
                        headers = {
                            "x-api-key": self.api_key,
                            "anthropic-version": "2023-06-01",
                            "Content-Type": "application/json",
                            "User-Agent": f"{USER_AGENT} (Anthropic)",
                        }
                        resp = await self._client.post(
                            f"{self.base_url}/v1/messages", json=payload, headers=headers
                        )

                        try:
                            resp.raise_for_status()
                        except httpx.HTTPStatusError as e:
                            if e.response.status_code == 429 or e.response.status_code >= 500:
                                raise httpx.RequestError(
                                    f"Retryable HTTP error: {e.response.status_code}",
                                    request=e.request,
                                ) from e
                            raise

                        data = resp.json()
                        text, tool_calls, finish_reason = self._parse_anthropic_response(data)
                        data["_parsed_finish_reason"] = finish_reason

                        usage = data.get("usage", {})
                        in_tokens = int(usage.get("input_tokens", 0))
                        out_tokens = int(usage.get("output_tokens", 0))
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
                            finish_reason=finish_reason,
                        )

                result = await with_retries(
                    make_request,
                    max_attempts=settings.adapters.retry_max_attempts,
                    base_s=settings.adapters.retry_base_ms / 1000,
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

    async def invoke_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream Anthropic Messages API events as OpenAI-compatible chunks."""
        provider = "anthropic"
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if self.circuit.is_open():
            PROVIDER_REQUESTS.labels(provider=provider, model=model, outcome="circuit_open").inc()
            raise ProviderError("circuit_open", provider=provider)

        payload = self._build_payload(model, messages, max_tokens, temperature, **kwargs)
        payload["stream"] = True

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": f"{USER_AGENT} (Anthropic)",
        }

        start = time.perf_counter()
        sent_role = False
        finish_reason: str | None = None

        async with start_span_async("provider.invoke_stream", provider=provider, model=model):
            try:
                async with self._client.stream(
                    "POST",
                    f"{self.base_url}/v1/messages",
                    json=payload,
                    headers=headers,
                ) as resp:
                    resp.raise_for_status()
                    event_name: str | None = None
                    async for raw_line in resp.aiter_lines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        if line.startswith("event:"):
                            event_name = line.split(":", 1)[1].strip()
                            continue
                        if not line.startswith("data:"):
                            continue
                        data_str = line.split(":", 1)[1].strip()
                        if not data_str:
                            continue
                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get("type") or event_name
                        if event_type == "message_start":
                            msg = event.get("message", {})
                            chunk_id = msg.get("id", chunk_id)
                            if not sent_role:
                                sent_role = True
                                yield build_stream_chunk(
                                    chunk_id=chunk_id,
                                    model=msg.get("model", model),
                                    role="assistant",
                                    content="",
                                    created=created,
                                )
                        elif event_type == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta" and delta.get("text"):
                                yield build_stream_chunk(
                                    chunk_id=chunk_id,
                                    model=model,
                                    content=delta["text"],
                                    created=created,
                                )
                        elif event_type == "message_delta":
                            stop_reason = event.get("delta", {}).get("stop_reason")
                            if stop_reason:
                                finish_reason = normalize_finish_reason("anthropic", stop_reason)

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
        """Get the list of models supported by Anthropic."""
        return [
            "claude-3-5-sonnet-latest",
            "claude-3-5-sonnet-20241022",
            "claude-3-sonnet",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-haiku-20241022",
        ]
