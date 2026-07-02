# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""Tests for message-native adapters and tool-calling passthrough (Increment 2)."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import ChatMessage
from app.providers.anthropic import AnthropicAdapter
from app.providers.base import (
    ProviderResponse,
    messages_to_openai_format,
    openai_tools_to_anthropic,
)
from app.providers.openai import OpenAIAdapter


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_patch():
    with patch("app.auth.auth.authenticate_request", return_value={"user_id": "test-user"}):
        yield


class TestMessageFormatHelpers:
    def test_messages_to_openai_format_preserves_roles_and_tools(self):
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="What's the weather?"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                    }
                ],
            ),
            ChatMessage(role="tool", content='{"temp": 72}', tool_call_id="call_1"),
        ]
        payload = messages_to_openai_format(messages)
        assert payload[0] == {"role": "system", "content": "You are helpful."}
        assert payload[1]["role"] == "user"
        assert payload[2]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert payload[3]["role"] == "tool"
        assert payload[3]["tool_call_id"] == "call_1"

    def test_openai_tools_to_anthropic(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        converted = openai_tools_to_anthropic(tools)
        assert converted[0]["name"] == "get_weather"
        assert converted[0]["input_schema"]["properties"]["city"]["type"] == "string"


class TestOpenAIToolCallingAdapter:
    @pytest.mark.asyncio
    async def test_invoke_passthrough_tools_and_returns_tool_calls(self):
        adapter = OpenAIAdapter(api_key="sk-test", base_url="https://api.openai.com/v1")
        captured: dict = {}

        async def fake_post(url, json=None, headers=None):  # noqa: A002
            captured["payload"] = json

            class R:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_abc",
                                            "type": "function",
                                            "function": {
                                                "name": "get_weather",
                                                "arguments": '{"city":"Boston"}',
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                        "usage": {"prompt_tokens": 12, "completion_tokens": 8},
                    }

            return R()

        adapter._client.post = fake_post
        messages = [
            ChatMessage(role="system", content="Use tools when needed."),
            ChatMessage(role="user", content="Weather in Boston?"),
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = await adapter.invoke(
            "gpt-4o-mini",
            messages,
            tools=tools,
            tool_choice="auto",
        )

        assert captured["payload"]["messages"][0]["role"] == "system"
        assert captured["payload"]["tools"] == tools
        assert captured["payload"]["tool_choice"] == "auto"
        assert result.tool_calls is not None
        assert result.tool_calls[0]["function"]["name"] == "get_weather"
        assert result.finish_reason == "tool_calls"


class TestAnthropicToolCallingAdapter:
    @pytest.mark.asyncio
    async def test_invoke_maps_tool_messages_and_response(self):
        adapter = AnthropicAdapter(api_key="sk-ant-test", base_url="https://api.anthropic.com")
        captured: dict = {}

        async def fake_post(url, json=None, headers=None):  # noqa: A002
            captured["payload"] = json

            class R:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_01",
                                "name": "get_weather",
                                "input": {"city": "Boston"},
                            }
                        ],
                        "stop_reason": "tool_use",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    }

            return R()

        adapter._client.post = fake_post
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Weather?"),
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        result = await adapter.invoke("claude-3-5-haiku-20241022", messages, tools=tools)

        assert captured["payload"]["system"] == "You are helpful."
        assert captured["payload"]["tools"][0]["name"] == "get_weather"
        assert result.tool_calls is not None
        assert result.tool_calls[0]["function"]["name"] == "get_weather"
        assert result.finish_reason == "tool_calls"


class TestChatCompletionsToolRoundTrip:
    @pytest.mark.asyncio
    async def test_openai_tool_round_trip_via_endpoint(self, client, auth_patch):
        mock_router = AsyncMock()
        mock_router.select_model.return_value = (
            "openai",
            "gpt-4o-mini",
            "test",
            {"label": "general", "confidence": 0.5, "signals": {}},
            {"usd": 0.01, "eta_ms": 100, "tokens_in": 10, "tokens_out": 20},
        )
        mock_router.invoke_via_adapter = AsyncMock(
            return_value=ProviderResponse(
                output_text="",
                tokens_in=10,
                tokens_out=5,
                latency_ms=50,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                    }
                ],
                finish_reason="tool_calls",
            )
        )

        with patch("app.providers.registry.get_provider_registry", return_value={"openai": Mock()}):
            with patch("app.main.router", mock_router):
                with patch("app.main.HeuristicRouter", return_value=mock_router):
                    response = client.post(
                        "/v1/chat/completions",
                        json={
                            "messages": [
                                {"role": "system", "content": "Use tools."},
                                {"role": "user", "content": "Weather in NYC?"},
                            ],
                            "model": "gpt-4o-mini",
                            "tools": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "parameters": {"type": "object", "properties": {}},
                                    },
                                }
                            ],
                            "tool_choice": "auto",
                        },
                        headers={"Authorization": "Bearer test-key"},
                    )

        assert response.status_code == 200
        data = response.json()
        message = data["choices"][0]["message"]
        assert message["tool_calls"][0]["function"]["name"] == "get_weather"
        assert data["choices"][0]["finish_reason"] == "tool_calls"

        invoke_kwargs = mock_router.invoke_via_adapter.await_args.kwargs
        assert invoke_kwargs["tools"] is not None
        passed_messages = mock_router.invoke_via_adapter.await_args.kwargs["messages"]
        assert passed_messages[0].role == "system"
        assert passed_messages[0].content == "Use tools."
