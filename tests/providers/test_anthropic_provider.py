# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.models import ChatMessage
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import ProviderError


class TestAnthropicProvider:
    """Test suite for AnthropicProvider."""

    @pytest.fixture
    def provider(self):
        """Create AnthropicProvider instance for testing."""
        return AnthropicProvider(api_key="test-key")

    @pytest.fixture
    def mock_client(self, provider, monkeypatch):
        """Mock httpx AsyncClient."""
        mock_client = AsyncMock()
        provider.client = mock_client
        # Mock settings for max_tokens_default
        from app.settings import settings

        monkeypatch.setattr(settings.router, "max_tokens_default", 1000)
        return mock_client

    def test_init_valid_key(self):
        """Test provider initialization with valid API key."""
        provider = AnthropicProvider(api_key="valid-key")
        assert provider.api_key == "valid-key"
        assert provider.provider_name == "anthropic"
        assert provider.base_url == "https://api.anthropic.com/v1"
        assert "claude-3-sonnet-20240229" in provider.supported_models

    def test_init_missing_key(self):
        """Test provider initialization fails without API key."""
        with pytest.raises(ValueError, match="Anthropic API key is required"):
            AnthropicProvider(api_key=None)

    def test_get_supported_models(self, provider):
        """Test getting supported models."""
        models = provider.get_supported_models()
        assert isinstance(models, list)
        assert len(models) == 3
        assert "claude-3-sonnet-20240229" in models
        assert "claude-3-haiku-20240307" in models
        assert "claude-3-opus-20240229" in models

    def test_calculate_cost_supported_model(self, provider):
        """Test cost calculation for supported model."""
        cost = provider.calculate_cost(1000, 500, "claude-3-sonnet-20240229")
        expected = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_calculate_cost_unsupported_model(self, provider):
        """Test cost calculation for unsupported model returns 0."""
        cost = provider.calculate_cost(1000, 500, "unknown-model")
        assert cost == 0.0

    def test_calculate_cost_edge_cases(self, provider):
        """Test cost calculation with zero tokens."""
        cost = provider.calculate_cost(0, 0, "claude-3-haiku-20240307")
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, provider, mock_client):
        """Test successful chat completion."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello world"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        result = await provider.chat_completion(
            messages=messages, model="claude-3-haiku-20240307", max_tokens=100
        )

        assert result.choices[0].message.content == "Hello world"
        assert result.model == "claude-3-haiku-20240307"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.choices[0].finish_reason == "stop"

        # Verify request was made correctly
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://api.anthropic.com/v1/messages"
        request_data = call_args[1]["json"]
        assert request_data["model"] == "claude-3-haiku-20240307"
        assert request_data["messages"] == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_chat_completion_with_system_message(self, provider, mock_client):
        """Test chat completion with system message."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Response"}],
            "usage": {"input_tokens": 15, "output_tokens": 8},
            "stop_reason": "end_turn",
        }
        mock_client.post.return_value = mock_response

        messages = [
            ChatMessage(role="system", content="You are helpful", name=None),
            ChatMessage(role="user", content="Hello", name=None),
        ]
        result = await provider.chat_completion(
            messages=messages, model="claude-3-haiku-20240307", max_tokens=100
        )

        # Verify system message was extracted
        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["system"] == "You are helpful"
        assert request_data["messages"] == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_chat_completion_with_temperature(self, provider, mock_client):
        """Test chat completion with temperature parameter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Response"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        await provider.chat_completion(
            messages=messages, model="claude-3-haiku-20240307", temperature=0.7, max_tokens=100
        )

        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_chat_completion_no_content_blocks(self, provider, mock_client):
        """Test chat completion when no content blocks returned."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [],
            "usage": {"input_tokens": 10, "output_tokens": 0},
            "stop_reason": "end_turn",
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        with pytest.raises(ProviderError, match="No content returned from Anthropic"):
            await provider.chat_completion(
                messages=messages, model="claude-3-haiku-20240307", max_tokens=100
            )

    @pytest.mark.asyncio
    async def test_chat_completion_http_error(self, provider, mock_client):
        """Test chat completion with HTTP error."""
        import httpx

        mock_client.post.side_effect = httpx.RequestError("Network error")

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        with pytest.raises(ProviderError, match="Anthropic request failed"):
            await provider.chat_completion(
                messages=messages, model="claude-3-haiku-20240307", max_tokens=100
            )

    @pytest.mark.asyncio
    async def test_chat_completion_token_estimation(self, provider, mock_client):
        """Test token estimation when usage not provided."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Short response"}],
            "usage": {"input_tokens": 0, "output_tokens": 0},  # No usage provided
            "stop_reason": "end_turn",
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello world", name=None)]
        result = await provider.chat_completion(
            messages=messages, model="claude-3-haiku-20240307", max_tokens=100
        )

        # Should estimate tokens
        assert result.usage.prompt_tokens == 2  # "Hello world" ~ 2 tokens (len/4)
        assert result.usage.completion_tokens == 3  # "Short response" ~ 3 tokens (len/4)

    @pytest.mark.skip(reason="Complex mocking issue with httpx stream context manager")
    @pytest.mark.asyncio
    async def test_stream_chat_completion_success(self, provider, mock_client):
        """Test successful streaming chat completion."""
        # TODO: Fix complex mocking for httpx.AsyncClient.stream
        pass

    @pytest.mark.asyncio
    async def test_stream_chat_completion_error(self, provider, mock_client):
        """Test streaming chat completion with error."""
        import httpx
        from unittest.mock import AsyncMock

        # Create a mock context manager that raises RequestError on enter
        mock_cm = AsyncMock()
        mock_cm.__aenter__.side_effect = httpx.RequestError("Stream error")
        mock_cm.__aexit__.return_value = None

        mock_client.stream.return_value = mock_cm

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        with pytest.raises(ProviderError, match="Anthropic streaming unexpected error"):
            async for _ in provider.stream_chat_completion(
                messages=messages, model="claude-3-haiku-20240307", max_tokens=100
            ):
                pass

    def test_prepare_messages_no_system(self, provider):
        """Test message preparation without system message."""
        messages = [
            ChatMessage(role="user", content="Hello", name=None),
            ChatMessage(role="assistant", content="Hi there", name=None),
        ]
        system, conv_messages = provider._prepare_messages(messages)
        assert system == ""
        assert conv_messages == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

    def test_prepare_messages_with_system(self, provider):
        """Test message preparation with system message."""
        messages = [
            ChatMessage(role="system", content="You are helpful", name=None),
            ChatMessage(role="user", content="Hello", name=None),
        ]
        system, conv_messages = provider._prepare_messages(messages)
        assert system == "You are helpful"
        assert conv_messages == [{"role": "user", "content": "Hello"}]

    def test_estimate_tokens(self, provider):
        """Test token estimation."""
        tokens = provider._estimate_tokens("Hello world")
        assert tokens == 2  # 12 characters / 4 = 3, but let's check actual
        # Actually len("Hello world") = 11, 11//4 = 2

    def test_create_headers(self, provider):
        """Test header creation."""
        headers = provider._create_headers()
        assert headers["x-api-key"] == "test-key"
        assert headers["anthropic-version"] == "2023-06-01"
        assert "User-Agent" in headers  # From base class

    @pytest.mark.asyncio
    async def test_chat_completion_with_max_tokens_parameter(self, provider, mock_client):
        """Test chat completion with max_tokens parameter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Response"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "max_tokens",
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        result = await provider.chat_completion(
            messages=messages, model="claude-3-haiku-20240307", max_tokens=50
        )

        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["max_tokens"] == 50
        assert result.choices[0].finish_reason == "length"

    @pytest.mark.asyncio
    async def test_chat_completion_with_stop_sequences(
        self, provider, mock_client, mock_settings_max_tokens
    ):
        """Test chat completion with stop sequences."""

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 10, "output_tokens": 2},
            "stop_reason": "stop_sequence",
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        result = await provider.chat_completion(
            messages=messages, model="claude-3-haiku-20240307", stop=["world"]
        )

        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["stop"] == ["world"]

    @pytest.mark.asyncio
    async def test_chat_completion_with_top_p(
        self, provider, mock_client, mock_settings_max_tokens
    ):
        """Test chat completion with top_p parameter."""

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Response"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        await provider.chat_completion(
            messages=messages, model="claude-3-haiku-20240307", top_p=0.9
        )

        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_chat_completion_with_top_k(
        self, provider, mock_client, mock_settings_max_tokens
    ):
        """Test chat completion with top_k parameter."""

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Response"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        await provider.chat_completion(messages=messages, model="claude-3-haiku-20240307", top_k=40)

        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["top_k"] == 40

    @pytest.mark.asyncio
    async def test_chat_completion_with_multiple_content_blocks(self, provider, mock_client):
        """Test chat completion with multiple content blocks."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world!"},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        result = await provider.chat_completion(
            messages=messages, model="claude-3-haiku-20240307", max_tokens=100
        )

        assert result.choices[0].message.content == "Hello world!"

    @pytest.mark.asyncio
    async def test_chat_completion_with_non_text_content_blocks(self, provider, mock_client):
        """Test chat completion with non-text content blocks (should be ignored)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image", "data": "ignored"},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 2},
            "stop_reason": "end_turn",
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        result = await provider.chat_completion(
            messages=messages, model="claude-3-haiku-20240307", max_tokens=100
        )

        assert result.choices[0].message.content == "Hello"

    @pytest.mark.asyncio
    async def test_chat_completion_with_empty_content_blocks(self, provider, mock_client):
        """Test chat completion with empty content blocks."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [],
            "usage": {"input_tokens": 10, "output_tokens": 0},
            "stop_reason": "end_turn",
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        with pytest.raises(ProviderError, match="No content returned from Anthropic"):
            await provider.chat_completion(
                messages=messages, model="claude-3-haiku-20240307", max_tokens=100
            )

    @pytest.mark.asyncio
    async def test_chat_completion_with_invalid_json_response(self, provider, mock_client):
        """Test chat completion with invalid JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        with pytest.raises(ProviderError, match="Anthropic unexpected error"):
            await provider.chat_completion(
                messages=messages, model="claude-3-haiku-20240307", max_tokens=100
            )

    @pytest.mark.asyncio
    async def test_chat_completion_with_missing_usage(self, provider, mock_client):
        """Test chat completion with missing usage information."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello world"}],
            "stop_reason": "end_turn",
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        result = await provider.chat_completion(
            messages=messages, model="claude-3-haiku-20240307", max_tokens=100
        )

        # Should estimate tokens when usage is missing
        assert result.usage.prompt_tokens == 1  # "Hello" ~ 1 token (5//4=1)
        assert result.usage.completion_tokens == 2  # "Hello world" ~ 2 tokens (11//4=2)

    @pytest.mark.asyncio
    async def test_chat_completion_with_partial_usage(self, provider, mock_client):
        """Test chat completion with partial usage information."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello world"}],
            "usage": {"input_tokens": 10},  # Missing output_tokens
            "stop_reason": "end_turn",
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        result = await provider.chat_completion(
            messages=messages, model="claude-3-haiku-20240307", max_tokens=100
        )

        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 2  # Estimated (11//4=2)

    @pytest.mark.asyncio
    async def test_chat_completion_with_zero_usage(self, provider, mock_client):
        """Test chat completion with zero usage values."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello world"}],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "stop_reason": "end_turn",
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        result = await provider.chat_completion(
            messages=messages, model="claude-3-haiku-20240307", max_tokens=100
        )

        # Should estimate tokens when usage is zero
        assert result.usage.prompt_tokens == 1  # "Hello" ~ 1 token
        assert result.usage.completion_tokens == 2  # "Hello world" ~ 2 tokens

    @pytest.mark.asyncio
    async def test_chat_completion_with_long_content(self, provider, mock_client):
        """Test chat completion with long content for token estimation."""
        long_content = (
            "This is a very long response that should be properly estimated for tokens. " * 10
        )
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": long_content}],
            "usage": {"input_tokens": 50, "output_tokens": 0},
            "stop_reason": "end_turn",
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        result = await provider.chat_completion(
            messages=messages, model="claude-3-haiku-20240307", max_tokens=100
        )

        assert result.usage.prompt_tokens == 50
        assert result.usage.completion_tokens == len(long_content) // 4  # Estimated

    def test_calculate_cost_comprehensive_edge_cases(self, provider):
        """Test cost calculation with various edge cases."""
        # Test with zero tokens
        cost = provider.calculate_cost(0, 0, "claude-3-haiku-20240307")
        assert cost == 0.0

        # Test with very large token counts
        cost = provider.calculate_cost(1000000, 1000000, "claude-3-haiku-20240307")
        expected = (1000000 / 1_000_000) * 0.25 + (1000000 / 1_000_000) * 1.25
        assert cost == expected

        # Test with negative tokens (should still work)
        cost = provider.calculate_cost(-100, -50, "claude-3-haiku-20240307")
        expected = (-100 / 1_000_000) * 0.25 + (-50 / 1_000_000) * 1.25
        assert cost == expected

    def test_calculate_cost_different_models(self, provider):
        """Test cost calculation for different models."""
        # Test Sonnet
        cost = provider.calculate_cost(1000, 500, "claude-3-sonnet-20240229")
        expected = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert cost == expected

        # Test Opus
        cost = provider.calculate_cost(1000, 500, "claude-3-opus-20240229")
        expected = (1000 / 1_000_000) * 15.0 + (500 / 1_000_000) * 75.0
        assert cost == expected

    def test_estimate_tokens_edge_cases(self, provider):
        """Test token estimation with edge cases."""
        # Empty string
        tokens = provider._estimate_tokens("")
        assert tokens == 0

        # Very short string
        tokens = provider._estimate_tokens("Hi")
        assert tokens == 0  # 2 // 4 = 0

        # Longer string
        tokens = provider._estimate_tokens("This is a test message")
        assert tokens == 5  # 22 // 4 = 5

        # String with special characters
        tokens = provider._estimate_tokens("Hello, world! @#$%^&*()")
        assert tokens == 5  # 23 // 4 = 5

    def test_prepare_messages_edge_cases(self, provider):
        """Test message preparation with edge cases."""
        # Empty messages
        system, conv = provider._prepare_messages([])
        assert system == ""
        assert conv == []

        # Only system message
        messages = [ChatMessage(role="system", content="You are helpful", name=None)]
        system, conv = provider._prepare_messages(messages)
        assert system == "You are helpful"
        assert conv == []

        # Only user message
        messages = [ChatMessage(role="user", content="Hello", name=None)]
        system, conv = provider._prepare_messages(messages)
        assert system == ""
        assert conv == [{"role": "user", "content": "Hello"}]

        # Multiple system messages (should use last one)
        messages = [
            ChatMessage(role="system", content="First system", name=None),
            ChatMessage(role="system", content="Last system", name=None),
            ChatMessage(role="user", content="Hello", name=None),
        ]
        system, conv = provider._prepare_messages(messages)
        assert system == "Last system"
        assert conv == [{"role": "user", "content": "Hello"}]

        # Mixed roles
        messages = [
            ChatMessage(role="system", content="System prompt", name=None),
            ChatMessage(role="user", content="Hello", name=None),
            ChatMessage(role="assistant", content="Hi there", name=None),
            ChatMessage(role="user", content="How are you?", name=None),
        ]
        system, conv = provider._prepare_messages(messages)
        assert system == "System prompt"
        assert conv == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]

    def test_prepare_messages_with_names(self, provider):
        """Test message preparation with message names (should be ignored for Anthropic)."""
        messages = [
            ChatMessage(role="user", content="Hello", name="user1"),
            ChatMessage(role="assistant", content="Hi", name="assistant1"),
        ]
        system, conv = provider._prepare_messages(messages)
        assert system == ""
        assert conv == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

    def test_prepare_messages_empty_content(self, provider):
        """Test message preparation with empty content."""
        messages = [
            ChatMessage(role="system", content="", name=None),
            ChatMessage(role="user", content="Hello", name=None),
        ]
        system, conv = provider._prepare_messages(messages)
        assert system == ""
        assert conv == [{"role": "user", "content": "Hello"}]
