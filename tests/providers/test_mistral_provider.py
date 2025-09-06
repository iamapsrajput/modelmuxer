# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.models import ChatMessage
from app.providers.base import ProviderError
from app.providers.mistral_provider import MistralProvider


class TestMistralProvider:
    """Test suite for MistralProvider."""

    @pytest.fixture
    def provider(self):
        """Create MistralProvider instance for testing."""
        return MistralProvider(api_key="test-key")

    @pytest.fixture
    def mock_client(self, provider, monkeypatch):
        """Mock httpx AsyncClient."""
        mock_client = AsyncMock()
        provider.client = mock_client
        return mock_client

    def test_init_valid_key(self):
        """Test provider initialization with valid API key."""
        provider = MistralProvider(api_key="valid-key")
        assert provider.api_key == "valid-key"
        assert provider.provider_name == "mistral"
        assert provider.base_url == "https://api.mistral.ai/v1"
        assert "mistral-small-latest" in provider.supported_models
        assert "mistral-medium-latest" in provider.supported_models
        assert "mistral-large-latest" in provider.supported_models

    def test_init_missing_key(self):
        """Test provider initialization fails without API key."""
        with pytest.raises(ValueError, match="Mistral API key is required"):
            MistralProvider(api_key=None)

    def test_get_supported_models(self, provider):
        """Test getting supported models."""
        models = provider.get_supported_models()
        assert isinstance(models, list)
        assert len(models) == 3
        assert "mistral-small-latest" in models
        assert "mistral-medium-latest" in models
        assert "mistral-large-latest" in models

    def test_calculate_cost_supported_model(self, provider):
        """Test cost calculation for supported model."""
        cost = provider.calculate_cost(1000, 500, "mistral-small-latest")
        expected = (1000 / 1_000_000) * 0.2 + (500 / 1_000_000) * 0.6
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_calculate_cost_unsupported_model(self, provider):
        """Test cost calculation for unsupported model returns zero."""
        cost = provider.calculate_cost(1000, 500, "unknown-model")
        assert cost == 0.0

    def test_calculate_cost_edge_cases(self, provider):
        """Test cost calculation with zero tokens."""
        cost = provider.calculate_cost(0, 0, "mistral-small-latest")
        assert cost == 0.0

    def test_create_headers(self, provider):
        """Test header creation."""
        headers = provider._create_headers()
        assert headers["Authorization"] == "Bearer test-key"
        assert "Content-Type" in headers
        assert "User-Agent" in headers

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

    def test_estimate_tokens(self, provider):
        """Test token estimation."""
        text = "Hello world this is a test"
        tokens = provider._estimate_tokens(text)
        # Should be roughly len(text) // 4
        expected = len(text) // 4
        assert tokens == expected

    def test_estimate_tokens_empty(self, provider):
        """Test token estimation with empty text."""
        tokens = provider._estimate_tokens("")
        assert tokens == 0

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
        result = await provider.chat_completion(messages=messages, model="mistral-small-latest", max_tokens=100)

        assert result.choices[0].message.content == "Hello world"
        assert result.model == "mistral-small-latest"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.choices[0].finish_reason == "stop"

        # Verify request was made correctly
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/chat/completions" in call_args[0][0]
        request_data = call_args[1]["json"]
        assert request_data["model"] == "mistral-small-latest"
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
        await provider.chat_completion(messages=messages, model="mistral-small-latest", temperature=0.7)

        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_chat_completion_no_usage_info(self, provider, mock_client):
        """Test chat completion when usage info not provided."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Short response"}}],
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello world", name=None)]
        result = await provider.chat_completion(messages=messages, model="mistral-small-latest")

        # Should estimate tokens
        assert result.usage.prompt_tokens == 2  # "Hello world" ~ 2 tokens
        assert result.usage.completion_tokens == 3  # "Short response" ~ 3 tokens

    @pytest.mark.asyncio
    async def test_chat_completion_no_choices(self, provider, mock_client):
        """Test chat completion when no choices returned."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        with pytest.raises(ProviderError, match="No choices returned from Mistral"):
            await provider.chat_completion(messages=messages, model="mistral-small-latest")

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
        result = await provider.chat_completion(messages=messages, model="mistral-small-latest")

        assert result.choices[0].message.content == ""
        assert result.usage.completion_tokens == 0

    @pytest.mark.asyncio
    async def test_chat_completion_request_error(self, provider, mock_client):
        """Test chat completion with request error."""
        import httpx

        mock_client.post.side_effect = httpx.RequestError("Network error")

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        with pytest.raises(ProviderError, match="Mistral request failed"):
            await provider.chat_completion(messages=messages, model="mistral-small-latest")

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
        await provider.chat_completion(
            messages=messages,
            model="mistral-small-latest",
            top_p=0.9,
            presence_penalty=0.1
        )

        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["top_p"] == 0.9
        assert request_data["presence_penalty"] == 0.1

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

    def test_calculate_cost_comprehensive_edge_cases(self, provider):
        """Test cost calculation with various edge cases."""
        # Test with zero tokens
        cost = provider.calculate_cost(0, 0, "mistral-small-latest")
        assert cost == 0.0

        # Test with very large token counts
        cost = provider.calculate_cost(1000000, 1000000, "mistral-small-latest")
        expected = (1000000 / 1_000_000) * 0.2 + (1000000 / 1_000_000) * 0.6
        assert cost == expected

        # Test with negative tokens (should still work)
        cost = provider.calculate_cost(-100, -50, "mistral-small-latest")
        expected = (-100 / 1_000_000) * 0.2 + (-50 / 1_000_000) * 0.6
        assert cost == expected

    def test_calculate_cost_different_models(self, provider):
        """Test cost calculation for different models."""
        # Test Mistral Large
        cost = provider.calculate_cost(1000, 500, "mistral-large-latest")
        expected = (1000 / 1_000_000) * 8.0 + (500 / 1_000_000) * 24.0
        assert cost == expected

        # Test Mistral Medium
        cost = provider.calculate_cost(1000, 500, "mistral-medium-latest")
        expected = (1000 / 1_000_000) * 2.7 + (500 / 1_000_000) * 8.1
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

    def test_estimate_tokens_various_lengths(self, provider):
        """Test token estimation with various text lengths."""
        test_cases = [
            ("", 0),
            ("a", 0),
            ("hello", 1),
            ("hello world", 2),
            ("This is a longer sentence with more words.", 10),
        ]

        for text, expected in test_cases:
            tokens = provider._estimate_tokens(text)
            assert tokens == expected