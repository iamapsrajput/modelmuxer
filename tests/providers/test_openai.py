from __future__ import annotations

import pytest

from app.providers.openai import OpenAIAdapter


@pytest.mark.asyncio
async def test_openai_adapter_success(monkeypatch):
    adapter = OpenAIAdapter(api_key="k", base_url="http://example.com")

    async def fake_post(url, json=None, headers=None):  # type: ignore[no-redef]
        class R:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                }

        return R()

    monkeypatch.setattr(adapter._client, "post", fake_post)
    r = await adapter.invoke(model="gpt-3.5-turbo", prompt="hi")
    assert r.output_text == "ok"
    assert r.tokens_in == 1 and r.tokens_out == 2


@pytest.mark.asyncio
async def test_openai_adapter_circuit_open():
    adapter = OpenAIAdapter(api_key="k", base_url="http://example.com")
    adapter.circuit.open_until = 9999999999
    r = await adapter.invoke(model="gpt-3.5-turbo", prompt="hi")
    assert r.error == "circuit_open"
