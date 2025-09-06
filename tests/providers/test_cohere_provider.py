# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.models import ChatMessage
from app.providers.base import AuthenticationError, ProviderError, RateLimitError
from app.providers.cohere_provider import CohereProvider


class TestCohereProvider:
    """Test suite for CohereProvider."""

    @pytest.fixture
    def provider(self):
        """Create CohereProvider instance for testing."""
        return CohereProvider(api_key="test-key")

    @pytest.fixture
    def mock_client(self, provider, monkeypatch):
        """Mock httpx AsyncClient."""
        mock_client = AsyncMock()
        provider.client = mock_client
        return mock_client

    def test_init_valid_key(self):
        """Test provider initialization with valid API key."""
        provider = CohereProvider(api_key="valid-key")
        assert provider.api_key == "valid-key"
        assert provider.provider_name == "cohere"
        assert provider.base_url == "https://api.cohere.ai/v1"
        assert "command-r-plus" in provider.supported_models
        assert "command-r" in provider.supported_models
        assert "command" in provider.supported_models
        assert "command-light" in provider.supported_models

    def test_init_missing_key(self):
        """Test provider initialization fails without API key."""
        with pytest.raises(AuthenticationError, match="Cohere API key is required"):
            CohereProvider(api_key=None)

    def test_get_supported_models(self, provider):
        """Test getting supported models."""
        models = provider.get_supported_models()
        assert isinstance(models, list)
        assert len(models) == 4
        assert "command-r-plus" in models
        assert "command-r" in models
        assert "command" in models
        assert "command-light" in models

    def test_calculate_cost_supported_model(self, provider):
        """Test cost calculation for supported model."""
        cost = provider.calculate_cost(1000, 500, "command-r")
        expected = (1000 / 1_000_000) * 0.5 + (500 / 1_000_000) * 1.5
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_calculate_cost_unsupported_model_fallback(self, provider):
        """Test cost calculation for unsupported model falls back to default."""
        cost = provider.calculate_cost(1000, 500, "unknown-model")
        # Should use command-r pricing as fallback
        expected = (1000 / 1_000_000) * 0.5 + (500 / 1_000_000) * 1.5
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_calculate_cost_edge_cases(self, provider):
        """Test cost calculation with zero tokens."""
        cost = provider.calculate_cost(0, 0, "command-r")
        assert cost == 0.0

    def test_get_rate_limits(self, provider):
        """Test getting rate limit information."""
        limits = provider.get_rate_limits()
        assert "requests_per_minute" in limits
        assert "tokens_per_minute" in limits
        assert limits["requests_per_minute"]["command-r"] == 1000
        assert limits["tokens_per_minute"]["command-r"] == 100000

    def test_create_headers(self, provider):
        """Test header creation."""
        headers = provider._create_headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"
        assert "User-Agent" in headers
        assert "ModelMuxer" in headers["User-Agent"]

    def test_convert_messages_to_cohere_format_empty(self, provider):
        """Test message conversion with empty messages."""
        result = provider._convert_messages_to_cohere_format([])
        assert result == {"message": "", "chat_history": []}

    def test_convert_messages_to_cohere_format_single_user(self, provider):
        """Test message conversion with single user message."""
        messages = [ChatMessage(role="user", content="Hello", name=None)]
        result = provider._convert_messages_to_cohere_format(messages)

        assert result["message"] == "Hello"
        assert result["chat_history"] == []

    def test_convert_messages_to_cohere_format_with_history(self, provider):
        """Test message conversion with chat history."""
        messages = [
            ChatMessage(role="user", content="Hello", name=None),
            ChatMessage(role="assistant", content="Hi there", name=None),
            ChatMessage(role="user", content="How are you?", name=None),
        ]
        result = provider._convert_messages_to_cohere_format(messages)

        assert result["message"] == "How are you?"
        assert len(result["chat_history"]) == 2
        assert result["chat_history"][0] == {"role": "USER", "message": "Hello"}
        assert result["chat_history"][1] == {"role": "CHATBOT", "message": "Hi there"}

    def test_convert_messages_to_cohere_format_skip_system_in_history(self, provider):
        """Test that system messages are skipped in chat history."""
        messages = [
            ChatMessage(role="system", content="You are helpful", name=None),
            ChatMessage(role="user", content="Hello", name=None),
            ChatMessage(role="assistant", content="Hi there", name=None),
            ChatMessage(role="user", content="How are you?", name=None),
        ]
        result = provider._convert_messages_to_cohere_format(messages)

        assert result["message"] == "How are you?"
        assert len(result["chat_history"]) == 2
        # System message should not appear in chat history
        assert not any(msg.get("role") == "SYSTEM" for msg in result["chat_history"])

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, provider, mock_client):
        """Test successful chat completion."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "Hello world",
            "finish_reason": "COMPLETE",
            "meta": {
                "tokens": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                }
            },
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        result = await provider.chat_completion(messages=messages, model="command-r", max_tokens=100)

        assert result.choices[0].message.content == "Hello world"
        assert result.model == "command-r"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.choices[0].finish_reason == "stop"

        # Verify request was made correctly
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/chat" in call_args[0][0]
        request_data = call_args[1]["json"]
        assert request_data["model"] == "command-r"
        assert request_data["message"] == "Hello"
        assert request_data["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_chat_completion_with_system_message(self, provider, mock_client):
        """Test chat completion with system message."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "Hello world",
            "meta": {"tokens": {"input_tokens": 10, "output_tokens": 5}},
        }
        mock_client.post.return_value = mock_response

        messages = [
            ChatMessage(role="system", content="You are helpful", name=None),
            ChatMessage(role="user", content="Hello", name=None),
        ]
        await provider.chat_completion(messages=messages, model="command-r")

        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["preamble"] == "You are helpful"

    @pytest.mark.asyncio
    async def test_chat_completion_with_temperature(self, provider, mock_client):
        """Test chat completion with temperature parameter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "Response",
            "meta": {"tokens": {"input_tokens": 10, "output_tokens": 5}},
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        await provider.chat_completion(messages=messages, model="command-r", temperature=0.7)

        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_chat_completion_no_meta_tokens(self, provider, mock_client):
        """Test chat completion when meta tokens not provided."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "Short response",
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello world", name=None)]
        result = await provider.chat_completion(messages=messages, model="command-r")

        # Should estimate tokens
        assert result.usage.prompt_tokens == 2  # "Hello world" ~ 2 tokens
        assert result.usage.completion_tokens == 2  # "Short response" ~ 2 tokens

    @pytest.mark.asyncio
    async def test_chat_completion_finish_reason_mapping(self, provider, mock_client):
        """Test finish reason mapping from Cohere to OpenAI format."""
        test_cases = [
            ("MAX_TOKENS", "length"),
            ("ERROR_TOXIC", "content_filter"),
            ("COMPLETE", "stop"),
            ("ERROR", "stop"),
            ("UNKNOWN", "stop"),
        ]

        for cohere_reason, expected_reason in test_cases:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "text": "Response",
                "finish_reason": cohere_reason,
                "meta": {"tokens": {"input_tokens": 10, "output_tokens": 5}},
            }
            mock_client.post.return_value = mock_response

            messages = [ChatMessage(role="user", content="Hello", name=None)]
            result = await provider.chat_completion(messages=messages, model="command-r")

            assert result.choices[0].finish_reason == expected_reason

    @pytest.mark.asyncio
    async def test_chat_completion_http_error_rate_limit(self, provider, mock_client):
        """Test chat completion with rate limit error."""
        import httpx

        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"message": "Rate limit exceeded"}

        http_error = httpx.HTTPStatusError(
            "Rate limit exceeded",
            request=Mock(),
            response=mock_response,
        )
        mock_client.post.side_effect = http_error

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        with pytest.raises(RateLimitError, match="Cohere API rate limit exceeded"):
            await provider.chat_completion(messages=messages, model="command-r")

    @pytest.mark.asyncio
    async def test_chat_completion_http_error_auth(self, provider, mock_client):
        """Test chat completion with authentication error."""
        import httpx

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"message": "Invalid API key"}

        http_error = httpx.HTTPStatusError(
            "Invalid API key",
            request=Mock(),
            response=mock_response,
        )
        mock_client.post.side_effect = http_error

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        with pytest.raises(AuthenticationError, match="Cohere API authentication failed"):
            await provider.chat_completion(messages=messages, model="command-r")

    @pytest.mark.asyncio
    async def test_chat_completion_http_error_other(self, provider, mock_client):
        """Test chat completion with other HTTP error."""
        import httpx

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"message": "Internal server error"}

        http_error = httpx.HTTPStatusError(
            "Internal server error",
            request=Mock(),
            response=mock_response,
        )
        mock_client.post.side_effect = http_error

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        with pytest.raises(ProviderError, match="Cohere API error: 500 Internal server error"):
            await provider.chat_completion(messages=messages, model="command-r")

    @pytest.mark.asyncio
    async def test_chat_completion_http_error_no_json(self, provider, mock_client):
        """Test chat completion with HTTP error that has no JSON response."""
        import httpx

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError("Not JSON")

        http_error = httpx.HTTPStatusError(
            "Internal server error",
            request=Mock(),
            response=mock_response,
        )
        mock_client.post.side_effect = http_error

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        with pytest.raises(ProviderError, match="Cohere API error: 500 "):
            await provider.chat_completion(messages=messages, model="command-r")

    @pytest.mark.asyncio
    async def test_chat_completion_request_error(self, provider, mock_client):
        """Test chat completion with request error."""
        import httpx

        mock_client.post.side_effect = httpx.RequestError("Network error")

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        with pytest.raises(ProviderError, match="Cohere request failed"):
            await provider.chat_completion(messages=messages, model="command-r")

    @pytest.mark.skip(reason="Complex mocking issue with httpx stream context manager")
    @pytest.mark.asyncio
    async def test_stream_chat_completion_success(self, provider, mock_client):
        """Test successful streaming chat completion."""
        # TODO: Fix complex mocking for httpx.AsyncClient.stream
        pass

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider, mock_client):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "OK",
            "meta": {"tokens": {"input_tokens": 1, "output_tokens": 1}},
        }
        mock_client.post.return_value = mock_response

        result = await provider.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, provider, mock_client):
        """Test health check failure."""
        import httpx

        mock_client.post.side_effect = httpx.RequestError("Connection failed")

        result = await provider.health_check()
        assert result is False

    def test_calculate_cost_comprehensive_edge_cases(self, provider):
        """Test cost calculation with various edge cases."""
        # Test with zero tokens
        cost = provider.calculate_cost(0, 0, "command-r")
        assert cost == 0.0

        # Test with very large token counts
        cost = provider.calculate_cost(1000000, 1000000, "command-r")
        expected = (1000000 / 1_000_000) * 0.5 + (1000000 / 1_000_000) * 1.5
        assert cost == expected

        # Test with negative tokens (should still work)
        cost = provider.calculate_cost(-100, -50, "command-r")
        expected = (-100 / 1_000_000) * 0.5 + (-50 / 1_000_000) * 1.5
        assert cost == expected

    def test_calculate_cost_different_models(self, provider):
        """Test cost calculation for different models."""
        # Test Command R Plus
        cost = provider.calculate_cost(1000, 500, "command-r-plus")
        expected = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert cost == expected

        # Test Command Light
        cost = provider.calculate_cost(1000, 500, "command-light")
        expected = (1000 / 1_000_000) * 0.3 + (500 / 1_000_000) * 0.6
        assert cost == expected

    def test_convert_messages_edge_cases(self, provider):
        """Test message conversion with edge cases."""
        # Only system message
        messages = [ChatMessage(role="system", content="You are helpful", name=None)]
        result = provider._convert_messages_to_cohere_format(messages)
        assert result["message"] == "You are helpful"
        assert result["chat_history"] == []

        # Multiple system messages (should use last as current message)
        messages = [
            ChatMessage(role="system", content="First system", name=None),
            ChatMessage(role="system", content="Last system", name=None),
        ]
        result = provider._convert_messages_to_cohere_format(messages)
        assert result["message"] == "Last system"
        assert result["chat_history"] == []

        # Mixed roles with names (names should be ignored)
        messages = [
            ChatMessage(role="user", content="Hello", name="user1"),
            ChatMessage(role="assistant", content="Hi", name="assistant1"),
            ChatMessage(role="user", content="How are you?", name="user2"),
        ]
        result = provider._convert_messages_to_cohere_format(messages)
        assert result["message"] == "How are you?"
        assert len(result["chat_history"]) == 2
        assert result["chat_history"][0]["role"] == "USER"
        assert result["chat_history"][1]["role"] == "CHATBOT"
