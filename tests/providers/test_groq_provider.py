# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.models import ChatMessage
from app.providers.base import AuthenticationError, ProviderError, RateLimitError
from app.providers.groq_provider import GroqProvider


class TestGroqProvider:
    """Test suite for GroqProvider."""

    @pytest.fixture
    def provider(self):
        """Create GroqProvider instance for testing."""
        return GroqProvider(api_key="test-key")

    @pytest.fixture
    def mock_client(self, provider, monkeypatch):
        """Mock httpx AsyncClient."""
        mock_client = AsyncMock()
        provider.client = mock_client
        return mock_client

    def test_init_valid_key(self):
        """Test provider initialization with valid API key."""
        provider = GroqProvider(api_key="valid-key")
        assert provider.api_key == "valid-key"
        assert provider.provider_name == "groq"
        assert provider.base_url == "https://api.groq.com/openai/v1"
        assert "llama-3.1-70b-versatile" in provider.supported_models
        assert "llama-3.1-8b-instant" in provider.supported_models
        assert "mixtral-8x7b-32768" in provider.supported_models
        assert "gemma-7b-it" in provider.supported_models
        assert "gemma2-9b-it" in provider.supported_models

    def test_init_missing_key(self):
        """Test provider initialization fails without API key."""
        with pytest.raises(AuthenticationError, match="Groq API key is required"):
            GroqProvider(api_key=None)

    def test_get_supported_models(self, provider):
        """Test getting supported models."""
        models = provider.get_supported_models()
        assert isinstance(models, list)
        assert len(models) == 5
        assert "llama-3.1-70b-versatile" in models
        assert "llama-3.1-8b-instant" in models
        assert "mixtral-8x7b-32768" in models
        assert "gemma-7b-it" in models
        assert "gemma2-9b-it" in models

    def test_calculate_cost_supported_model(self, provider):
        """Test cost calculation for supported model."""
        cost = provider.calculate_cost(1000, 500, "llama-3.1-8b-instant")
        expected = (1000 / 1_000_000) * 0.05 + (500 / 1_000_000) * 0.08
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_calculate_cost_unsupported_model_fallback(self, provider):
        """Test cost calculation for unsupported model falls back to default."""
        cost = provider.calculate_cost(1000, 500, "unknown-model")
        # Should use llama-3.1-8b-instant pricing as fallback
        expected = (1000 / 1_000_000) * 0.05 + (500 / 1_000_000) * 0.08
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_calculate_cost_edge_cases(self, provider):
        """Test cost calculation with zero tokens."""
        cost = provider.calculate_cost(0, 0, "llama-3.1-8b-instant")
        assert cost == 0.0

    def test_get_rate_limits(self, provider):
        """Test getting rate limit information."""
        limits = provider.get_rate_limits()
        assert "requests_per_minute" in limits
        assert "tokens_per_minute" in limits
        assert limits["requests_per_minute"]["llama-3.1-8b-instant"] == 30
        assert limits["tokens_per_minute"]["llama-3.1-8b-instant"] == 30000

    def test_get_model_info_supported_model(self, provider):
        """Test getting model info for supported model."""
        info = provider.get_model_info("llama-3.1-70b-versatile")
        assert info["description"] == "Meta's Llama 3.1 70B model, versatile for many tasks"
        assert info["context_length"] == 131072
        assert "reasoning" in info["strengths"]
        assert info["speed"] == "medium"

    def test_get_model_info_unsupported_model(self, provider):
        """Test getting model info for unsupported model."""
        info = provider.get_model_info("unknown-model")
        assert info["description"] == "Unknown model"
        assert info["context_length"] == 4096
        assert info["strengths"] == []
        assert info["speed"] == "unknown"

    def test_create_headers(self, provider):
        """Test header creation."""
        headers = provider._create_headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"
        assert "User-Agent" in headers
        assert "ModelMuxer" in headers["User-Agent"]

    def test_prepare_messages_basic(self, provider):
        """Test message preparation with basic messages."""
        messages = [
            ChatMessage(role="user", content="Hello", name=None),
            ChatMessage(role="assistant", content="Hi there", name=None),
        ]
        result = provider._prepare_messages(messages)

        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi there"}

    def test_prepare_messages_with_names(self, provider):
        """Test message preparation with message names."""
        messages = [
            ChatMessage(role="user", content="Hello", name="user1"),
            ChatMessage(role="assistant", content="Hi there", name="assistant1"),
        ]
        result = provider._prepare_messages(messages)

        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello", "name": "user1"}
        assert result[1] == {"role": "assistant", "content": "Hi there", "name": "assistant1"}

    def test_prepare_messages_without_names(self, provider):
        """Test message preparation without message names."""
        messages = [
            ChatMessage(role="user", content="Hello", name=None),
            ChatMessage(role="assistant", content="Hi there", name=None),
        ]
        result = provider._prepare_messages(messages)

        assert len(result) == 2
        assert "name" not in result[0]
        assert "name" not in result[1]

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, provider, mock_client):
        """Test successful chat completion."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "Hello world"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        result = await provider.chat_completion(messages=messages, model="llama-3.1-8b-instant", max_tokens=100)

        assert result.choices[0].message.content == "Hello world"
        assert result.model == "llama-3.1-8b-instant"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.choices[0].finish_reason == "stop"

        # Verify request was made correctly
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/chat/completions" in call_args[0][0]
        request_data = call_args[1]["json"]
        assert request_data["model"] == "llama-3.1-8b-instant"
        assert request_data["messages"][0]["content"] == "Hello"
        assert request_data["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_chat_completion_with_temperature(self, provider, mock_client):
        """Test chat completion with temperature parameter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        await provider.chat_completion(messages=messages, model="llama-3.1-8b-instant", temperature=0.7)

        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_chat_completion_no_choices(self, provider, mock_client):
        """Test chat completion when no choices returned."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        with pytest.raises(ProviderError, match="No choices returned from Groq"):
            await provider.chat_completion(messages=messages, model="llama-3.1-8b-instant")

    @pytest.mark.asyncio
    async def test_chat_completion_no_message_content(self, provider, mock_client):
        """Test chat completion when message content is missing."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0},
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        result = await provider.chat_completion(messages=messages, model="llama-3.1-8b-instant")

        assert result.choices[0].message.content == ""
        assert result.usage.completion_tokens == 0

    @pytest.mark.asyncio
    async def test_chat_completion_with_kwargs(self, provider, mock_client):
        """Test chat completion with additional kwargs."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        await provider.chat_completion(messages=messages, model="llama-3.1-8b-instant", top_p=0.9, presence_penalty=0.1)

        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["top_p"] == 0.9
        assert request_data["presence_penalty"] == 0.1

    @pytest.mark.asyncio
    async def test_chat_completion_http_error_rate_limit(self, provider, mock_client):
        """Test chat completion with rate limit error."""
        import httpx

        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}

        http_error = httpx.HTTPStatusError(
            "Rate limit exceeded",
            request=Mock(),
            response=mock_response,
        )
        mock_client.post.side_effect = http_error

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        with pytest.raises(RateLimitError, match="Groq API rate limit exceeded"):
            await provider.chat_completion(messages=messages, model="llama-3.1-8b-instant")

    @pytest.mark.asyncio
    async def test_chat_completion_http_error_auth(self, provider, mock_client):
        """Test chat completion with authentication error."""
        import httpx

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}

        http_error = httpx.HTTPStatusError(
            "Invalid API key",
            request=Mock(),
            response=mock_response,
        )
        mock_client.post.side_effect = http_error

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        with pytest.raises(AuthenticationError, match="Groq API authentication failed"):
            await provider.chat_completion(messages=messages, model="llama-3.1-8b-instant")

    @pytest.mark.asyncio
    async def test_chat_completion_http_error_other(self, provider, mock_client):
        """Test chat completion with other HTTP error."""
        import httpx

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": {"message": "Internal server error"}}

        http_error = httpx.HTTPStatusError(
            "Internal server error",
            request=Mock(),
            response=mock_response,
        )
        mock_client.post.side_effect = http_error

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        with pytest.raises(ProviderError, match="Groq API error: 500 Internal server error"):
            await provider.chat_completion(messages=messages, model="llama-3.1-8b-instant")

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
        with pytest.raises(ProviderError, match="Groq API error: 500 "):
            await provider.chat_completion(messages=messages, model="llama-3.1-8b-instant")

    @pytest.mark.asyncio
    async def test_chat_completion_request_error(self, provider, mock_client):
        """Test chat completion with request error."""
        import httpx

        mock_client.post.side_effect = httpx.RequestError("Network error")

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        with pytest.raises(ProviderError, match="Groq request failed"):
            await provider.chat_completion(messages=messages, model="llama-3.1-8b-instant")

    @pytest.mark.skip(reason="Complex mocking issue with httpx stream context manager")
    @pytest.mark.asyncio
    async def test_stream_chat_completion_success(self, provider, mock_client):
        """Test successful streaming chat completion."""
        # TODO: Fix complex mocking for httpx.AsyncClient.stream
        pass

    @pytest.mark.skip(reason="Complex mocking issue with httpx stream context manager")
    @pytest.mark.asyncio
    async def test_stream_chat_completion_request_error(self, provider, mock_client):
        """Test streaming chat completion with request error."""
        # TODO: Fix complex mocking for httpx.AsyncClient.stream
        pass

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider, mock_client):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
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
        cost = provider.calculate_cost(0, 0, "llama-3.1-8b-instant")
        assert cost == 0.0

        # Test with very large token counts
        cost = provider.calculate_cost(1000000, 1000000, "llama-3.1-8b-instant")
        expected = (1000000 / 1_000_000) * 0.05 + (1000000 / 1_000_000) * 0.08
        assert cost == expected

        # Test with negative tokens (should still work)
        cost = provider.calculate_cost(-100, -50, "llama-3.1-8b-instant")
        expected = (-100 / 1_000_000) * 0.05 + (-50 / 1_000_000) * 0.08
        assert cost == expected

    def test_calculate_cost_different_models(self, provider):
        """Test cost calculation for different models."""
        # Test Llama 3.1 70B
        cost = provider.calculate_cost(1000, 500, "llama-3.1-70b-versatile")
        expected = (1000 / 1_000_000) * 0.59 + (500 / 1_000_000) * 0.79
        assert cost == expected

        # Test Mixtral
        cost = provider.calculate_cost(1000, 500, "mixtral-8x7b-32768")
        expected = (1000 / 1_000_000) * 0.24 + (500 / 1_000_000) * 0.24
        assert cost == expected

        # Test Gemma models
        cost = provider.calculate_cost(1000, 500, "gemma-7b-it")
        expected = (1000 / 1_000_000) * 0.07 + (500 / 1_000_000) * 0.07
        assert cost == expected

    def test_prepare_messages_edge_cases(self, provider):
        """Test message preparation with edge cases."""
        # Empty messages
        result = provider._prepare_messages([])
        assert result == []

        # Single message
        messages = [ChatMessage(role="user", content="Hello", name=None)]
        result = provider._prepare_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

        # Messages with empty content
        messages = [ChatMessage(role="user", content="", name="user1")]
        result = provider._prepare_messages(messages)
        assert result[0]["content"] == ""
        assert result[0]["name"] == "user1"

    def test_get_model_info_comprehensive(self, provider):
        """Test getting model info for all supported models."""
        models_to_test = [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma-7b-it",
            "gemma2-9b-it",
        ]

        for model in models_to_test:
            info = provider.get_model_info(model)
            assert "description" in info
            assert "context_length" in info
            assert "strengths" in info
            assert "speed" in info
            assert isinstance(info["context_length"], int)
            assert isinstance(info["strengths"], list)
