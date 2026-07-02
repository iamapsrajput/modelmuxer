# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""Tests for /v1/models polish and SSE streaming (Increment 1)."""

import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import ChatMessage
from app.providers.anthropic import AnthropicAdapter
from app.providers.base import build_stream_chunk, format_sse_data
from app.providers.openai import OpenAIAdapter


class MockAdapterForStream:
    """Minimal adapter exposing get_supported_models for list_models tests."""

    def __init__(self, models: list[str]):
        self._models = models

    def get_supported_models(self) -> list[str]:
        return self._models


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_patch():
    with patch("app.auth.auth.authenticate_request", return_value={"user_id": "test-user"}):
        yield


class TestListModels:
    def test_lists_only_registered_provider_models(self, client, auth_patch):
        registry = {
            "openai": MockAdapterForStream(["gpt-4o-mini"]),
            "together": MockAdapterForStream(["meta-llama/Llama-3-8b-chat-hf"]),
        }
        price_table = {
            "openai:gpt-4o-mini": Mock(input_per_1k_usd=0.001, output_per_1k_usd=0.002),
            "together:meta-llama/Llama-3-8b-chat-hf": Mock(
                input_per_1k_usd=0.0002, output_per_1k_usd=0.0002
            ),
            "ghost:unregistered-model": Mock(input_per_1k_usd=1.0, output_per_1k_usd=1.0),
        }

        with patch("app.providers.registry.get_provider_registry", return_value=registry):
            with patch("app.main.load_price_table", return_value=price_table):
                response = client.get("/v1/models", headers={"Authorization": "Bearer test-key"})

        assert response.status_code == 200
        data = response.json()
        ids = {m["id"] for m in data["data"]}
        assert "openai/gpt-4o-mini" in ids
        assert "together/meta-llama/Llama-3-8b-chat-hf" in ids
        assert "ghost/unregistered-model" not in ids

        together = next(m for m in data["data"] if m["provider"] == "together")
        assert together["model"] == "meta-llama/Llama-3-8b-chat-hf"
        assert together["owned_by"] == "together"
        assert together["object"] == "model"
        assert "created" in together

    def test_includes_router_aliases(self, client, auth_patch):
        with patch("app.providers.registry.get_provider_registry", return_value={}):
            with patch("app.main.load_price_table", return_value={}):
                response = client.get("/v1/models", headers={"Authorization": "Bearer test-key"})

        ids = {m["id"] for m in response.json()["data"]}
        assert "router" in ids
        assert "auto" in ids


class TestSSEHelpers:
    def test_format_sse_data_json(self):
        payload = {"choices": [{"delta": {"content": "hi"}}]}
        assert format_sse_data(payload) == f"data: {json.dumps(payload)}\n\n"

    def test_format_sse_done_sentinel(self):
        assert format_sse_data("[DONE]") == "data: [DONE]\n\n"

    def test_build_stream_chunk_shape(self):
        chunk = build_stream_chunk(
            chunk_id="chatcmpl-test",
            model="gpt-4o-mini",
            content="Hello",
            finish_reason=None,
        )
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["choices"][0]["delta"]["content"] == "Hello"


class TestOpenAIStreamingAdapter:
    @pytest.mark.asyncio
    async def test_invoke_stream_yields_openai_chunks(self):
        adapter = OpenAIAdapter(api_key="sk-test", base_url="https://api.openai.com/v1")

        sse_lines = [
            'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}',
            "data: [DONE]",
        ]

        async def mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()
        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__.return_value = mock_response
        mock_stream_ctx.__aexit__.return_value = None

        adapter._client.stream = Mock(return_value=mock_stream_ctx)

        messages = [ChatMessage(role="user", content="Hi")]
        chunks = [chunk async for chunk in adapter.invoke_stream("gpt-4o-mini", messages)]

        contents = [
            c["choices"][0]["delta"].get("content")
            for c in chunks
            if c["choices"][0]["delta"].get("content")
        ]
        assert contents == ["Hel", "lo"]
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert adapter.supports_streaming() is True


class TestAnthropicStreamingAdapter:
    @pytest.mark.asyncio
    async def test_invoke_stream_maps_anthropic_events(self):
        adapter = AnthropicAdapter(api_key="sk-ant-test", base_url="https://api.anthropic.com")

        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"msg_123","model":"claude-3-5-haiku-20241022"}}',
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}',
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}',
            "event: message_stop",
            'data: {"type":"message_stop"}',
        ]

        async def mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()
        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__.return_value = mock_response
        mock_stream_ctx.__aexit__.return_value = None

        adapter._client.stream = Mock(return_value=mock_stream_ctx)

        messages = [ChatMessage(role="user", content="Hello")]
        chunks = [
            chunk async for chunk in adapter.invoke_stream("claude-3-5-haiku-20241022", messages)
        ]

        contents = [
            c["choices"][0]["delta"].get("content")
            for c in chunks
            if c["choices"][0]["delta"].get("content")
        ]
        assert contents == ["Hi"]
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


class TestChatStreamingEndpoint:
    @pytest.mark.asyncio
    async def test_streaming_sse_framing_and_done(self, client, auth_patch):
        async def mock_stream(**kwargs):
            yield build_stream_chunk(
                chunk_id="chatcmpl-test",
                model="gpt-4",
                role="assistant",
                content="",
            )
            yield build_stream_chunk(
                chunk_id="chatcmpl-test",
                model="gpt-4",
                content="Hello",
            )
            yield build_stream_chunk(
                chunk_id="chatcmpl-test",
                model="gpt-4",
                finish_reason="stop",
            )

        mock_provider = Mock()
        mock_provider.stream_chat_completion = mock_stream

        mock_router = AsyncMock()
        mock_router.select_model.return_value = (
            "openai",
            "gpt-4",
            "test",
            {"label": "general", "confidence": 0.5, "signals": {}},
            {"usd": 0.01, "eta_ms": 100, "tokens_in": 10, "tokens_out": 20},
        )

        with patch(
            "app.providers.registry.get_provider_registry", return_value={"openai": mock_provider}
        ):
            with patch("app.main.router", mock_router):
                with patch("app.main.HeuristicRouter", return_value=mock_router):
                    response = client.post(
                        "/v1/chat/completions",
                        json={
                            "messages": [{"role": "user", "content": "Hello"}],
                            "model": "gpt-4",
                            "stream": True,
                        },
                        headers={"Authorization": "Bearer test-key"},
                    )

        assert response.status_code == 200
        body = response.text
        assert body.endswith("data: [DONE]\n\n")
        assert "chat.completion.chunk" in body
        assert '"content": "Hello"' in body or '"content":"Hello"' in body
