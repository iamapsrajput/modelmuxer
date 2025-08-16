from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.providers.openai import OpenAIAdapter, ProviderResponse


@pytest.mark.asyncio
async def test_provider_response_success(monkeypatch):
    adapter = OpenAIAdapter(api_key="test", base_url="http://example.com")

    async def fake_post(url: str, json: Any = None, headers: Any = None):  # type: ignore[no-redef]
        class R:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {
                    "choices": [{"message": {"content": "hello"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                }

        return R()

    monkeypatch.setattr(adapter._client, "post", fake_post)

    resp = await adapter.invoke(model="gpt-3.5-turbo", prompt="hi")
    assert isinstance(resp, ProviderResponse)
    assert resp.output_text == "hello"
    assert resp.tokens_in == 10
    assert resp.tokens_out == 20
    assert resp.error is None


@pytest.mark.asyncio
async def test_provider_retry_and_circuit(monkeypatch):
    adapter = OpenAIAdapter(api_key="test", base_url="http://example.com")
    calls = {"count": 0}

    class BoomError(Exception):
        pass

    async def boom(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        calls["count"] += 1
        raise BoomError("fail")

    monkeypatch.setattr(adapter._client, "post", boom)

    resp = await adapter.invoke(model="gpt-3.5-turbo", prompt="hi")
    assert resp.error is not None
    assert calls["count"] >= 1

    # Simulate circuit open
    adapter.circuit.open_until = 9999999999
    resp2 = await adapter.invoke(model="gpt-3.5-turbo", prompt="hi")
    assert resp2.error == "circuit_open"
