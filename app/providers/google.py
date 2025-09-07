# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Optional

import httpx

from app.models import ChatMessage
from app.providers.base import (
    USER_AGENT,
    LLMProviderAdapter,
    ProviderResponse,
    SimpleCircuitBreaker,
    _is_retryable_error,
    normalize_finish_reason,
    with_retries,
)
from app.settings import settings
from app.telemetry.metrics import PROVIDER_LATENCY, PROVIDER_REQUESTS, TOKENS_TOTAL
from app.telemetry.tracing import start_span_async


class GoogleAdapter(LLMProviderAdapter):
    def __init__(
        self, api_key: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.circuit = SimpleCircuitBreaker(
            fail_threshold=settings.providers.circuit_fail_threshold,
            cooldown_sec=settings.providers.circuit_cooldown_sec,
        )
        self._client = httpx.AsyncClient(timeout=settings.providers.timeout_ms / 1000)

    def _convert_prompt_to_google_format(self, prompt: str) -> dict[str, Any]:
        """Convert simple prompt to Google Gemini format."""
        return {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

    def _convert_messages_to_google_format(
        self, messages: list[dict[str, Any] | ChatMessage], prompt: str = ""
    ) -> dict[str, Any]:
        """Convert OpenAI format messages to Google Gemini format.

        Args:
            messages: List of message dictionaries or ChatMessage objects
            prompt: Fallback prompt if no messages provided

        Returns:
            Google Gemini format payload
        """
        contents = []
        system_instruction = None

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
                # Google uses systemInstruction separately
                system_instruction = content
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({
                    "role": "model",  # Google uses "model" instead of "assistant"
                    "parts": [{"text": content}],
                })

        result = {"contents": contents}
        if not contents:
            # Fallback to prompt if provided
            if isinstance(prompt, str) and prompt:
                result["contents"] = [{"role": "user", "parts": [{"text": prompt}]}]
            else:
                raise ValueError(
                    "Google payload requires at least one user message or non-empty prompt"
                )
        if system_instruction:
            result["systemInstruction"] = {
                "role": "system",
                "parts": [{"text": system_instruction}],
            }

        return result

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
        provider = "google"

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
                    async with start_span_async("google.request", attempt=attempt):
                        # Check if messages are provided for chat history
                        messages = kwargs.get("messages")
                        if messages:
                            # Convert messages to Google format
                            google_payload = self._convert_messages_to_google_format(
                                messages, prompt
                            )
                        else:
                            # Validate prompt is non-empty
                            if not isinstance(prompt, str) or not prompt.strip():
                                raise ValueError(
                                    "Google payload requires a non-empty prompt or messages"
                                )
                            # Convert simple prompt to Google format
                            google_payload = self._convert_prompt_to_google_format(prompt)

                        # Add system instruction if provided and not already set
                        if kwargs.get("system") and "systemInstruction" not in google_payload:
                            google_payload["systemInstruction"] = {
                                "role": "system",
                                "parts": [{"text": kwargs["system"]}],
                            }

                        # Add generation config with normalized sampling parameters
                        generation_config = {}

                        # Core parameters
                        if "temperature" in kwargs:
                            generation_config["temperature"] = kwargs["temperature"]
                        else:
                            generation_config["temperature"] = settings.router.temperature_default

                        if "max_tokens" in kwargs:
                            generation_config["maxOutputTokens"] = kwargs["max_tokens"]
                        else:
                            generation_config["maxOutputTokens"] = (
                                settings.router.max_tokens_default
                            )

                        # Google-specific sampling parameters
                        if "top_p" in kwargs:
                            generation_config["topP"] = kwargs["top_p"]
                        if "top_k" in kwargs:
                            generation_config["topK"] = kwargs["top_k"]
                        if "candidate_count" in kwargs:
                            generation_config["candidateCount"] = kwargs["candidate_count"]

                        google_payload["generationConfig"] = generation_config

                        # Add safety settings only if explicitly requested or enabled in settings
                        if kwargs.get("safety_settings") is not None:
                            google_payload["safetySettings"] = kwargs["safety_settings"]
                        elif settings.google.default_safety:
                            google_payload["safetySettings"] = [
                                {
                                    "category": "HARM_CATEGORY_HARASSMENT",
                                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                                },
                                {
                                    "category": "HARM_CATEGORY_HATE_SPEECH",
                                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                                },
                                {
                                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                                },
                                {
                                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                                },
                            ]

                        # Add stop sequences if provided
                        if "stop" in kwargs:
                            stop_list = kwargs["stop"]
                            if isinstance(stop_list, str):
                                stop_list = [stop_list]
                            generation_config["stopSequences"] = stop_list

                        headers = {
                            "Content-Type": "application/json",
                            "User-Agent": f"{USER_AGENT} (Google Gemini)",
                        }

                        url = f"{self.base_url}/models/{model}:generateContent"
                        params = {"key": self.api_key}

                        resp = await self._client.post(
                            url, json=google_payload, headers=headers, params=params
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
                            error_msg = data["error"].get("message", "Google AI returned an error")

                            # Use centralized retryable error detection
                            is_retryable = _is_retryable_error("google", resp.status_code, data)

                            if is_retryable:
                                raise httpx.RequestError(
                                    f"Google retryable error: {error_msg}", request=resp.request
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

                        # Extract content from Google response
                        content = ""
                        if "candidates" in data and data["candidates"]:
                            candidate = data["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                parts = candidate["content"]["parts"]
                                # Handle mixed tool parts - only extract text parts
                                content = "".join(
                                    p["text"]
                                    for p in parts
                                    if isinstance(p, dict) and p.get("text")
                                )
                                # Store non-text parts for downstream handling
                                non_text_parts = [
                                    p for p in parts if isinstance(p, dict) and not p.get("text")
                                ]
                                if non_text_parts:
                                    data["_non_text_parts"] = non_text_parts

                        # Extract and map finish reason
                        finish_reason = "stop"  # default
                        if "candidates" in data and data["candidates"]:
                            candidate = data["candidates"][0]
                            google_finish_reason = candidate.get("finishReason")
                            finish_reason = normalize_finish_reason("google", google_finish_reason)

                        # Store parsed finish reason in response data
                        data["_parsed_finish_reason"] = finish_reason

                        # Extract token usage from response if available
                        in_tokens = 0
                        out_tokens = 0

                        if "usageMetadata" in data:
                            usage_metadata = data["usageMetadata"]
                            in_tokens = int(usage_metadata.get("promptTokenCount", 0))
                            out_tokens = int(usage_metadata.get("candidatesTokenCount", 0))
                        elif (
                            "candidates" in data
                            and data["candidates"]
                            and "usageMetadata" in data["candidates"][0]
                        ):
                            # Check alternate location for usage metadata
                            usage_metadata = data["candidates"][0]["usageMetadata"]
                            in_tokens = int(usage_metadata.get("promptTokenCount", 0))
                            out_tokens = int(usage_metadata.get("candidatesTokenCount", 0))

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
        """Get the list of models supported by Google Gemini."""
        return [
            "gemini-1.5-pro",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.0-pro",
            "gemini-pro",
        ]
