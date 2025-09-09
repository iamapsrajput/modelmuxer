# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.models import ChatMessage
from app.providers.base import (AuthenticationError, ProviderError,
                                RateLimitError)
from app.providers.google_provider import GoogleProvider


class TestGoogleProvider:
    """Test suite for GoogleProvider."""

    @pytest.fixture
    def provider(self):
        """Create GoogleProvider instance for testing."""
        return GoogleProvider(api_key="test-key")

    @pytest.fixture
    def mock_client(self, provider, monkeypatch):
        """Mock httpx AsyncClient."""
        mock_client = AsyncMock()
        provider.client = mock_client
        return mock_client

    def test_init_valid_key(self):
        """Test provider initialization with valid API key."""
        provider = GoogleProvider(api_key="valid-key")
        assert provider.api_key == "valid-key"
        assert provider.provider_name == "google"
        assert provider.base_url == "https://generativelanguage.googleapis.com/v1beta"
        assert "gemini-1.5-pro" in provider.supported_models
        assert "gemini-1.5-flash" in provider.supported_models
        assert "gemini-1.0-pro" in provider.supported_models

    def test_init_missing_key(self):
        """Test provider initialization fails without API key."""
        with pytest.raises(AuthenticationError, match="Google API key is required"):
            GoogleProvider(api_key=None)

    def test_get_supported_models(self, provider):
        """Test getting supported models."""
        models = provider.get_supported_models()
        assert isinstance(models, list)
        assert len(models) == 3
        assert "gemini-1.5-pro" in models
        assert "gemini-1.5-flash" in models
        assert "gemini-1.0-pro" in models

    def test_calculate_cost_supported_model(self, provider):
        """Test cost calculation for supported model."""
        cost = provider.calculate_cost(1000, 500, "gemini-1.5-flash")
        expected = (1000 / 1_000_000) * 0.075 + (500 / 1_000_000) * 0.3
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_calculate_cost_unsupported_model_fallback(self, provider):
        """Test cost calculation for unsupported model falls back to default."""
        cost = provider.calculate_cost(1000, 500, "unknown-model")
        # Should use gemini-1.5-flash pricing as fallback
        expected = (1000 / 1_000_000) * 0.075 + (500 / 1_000_000) * 0.3
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_calculate_cost_edge_cases(self, provider):
        """Test cost calculation with zero tokens."""
        cost = provider.calculate_cost(0, 0, "gemini-1.5-flash")
        assert cost == 0.0

    def test_get_rate_limits(self, provider):
        """Test getting rate limit information."""
        limits = provider.get_rate_limits()
        assert "requests_per_minute" in limits
        assert "tokens_per_minute" in limits
        assert limits["requests_per_minute"]["gemini-1.5-flash"] == 1000
        assert limits["tokens_per_minute"]["gemini-1.5-flash"] == 1000000

    def test_create_headers(self, provider):
        """Test header creation."""
        headers = provider._create_headers()
        assert headers["Content-Type"] == "application/json"
        assert "User-Agent" in headers
        assert "ModelMuxer" in headers["User-Agent"]

    def test_convert_messages_to_google_format_user_only(self, provider):
        """Test message conversion with user message only."""
        messages = [ChatMessage(role="user", content="Hello", name=None)]
        result = provider._convert_messages_to_google_format(messages)

        assert "contents" in result
        assert len(result["contents"]) == 1
        assert result["contents"][0]["role"] == "user"
        assert result["contents"][0]["parts"][0]["text"] == "Hello"
        assert "systemInstruction" not in result

    def test_convert_messages_to_google_format_with_system(self, provider):
        """Test message conversion with system message."""
        messages = [
            ChatMessage(role="system", content="You are helpful", name=None),
            ChatMessage(role="user", content="Hello", name=None),
        ]
        result = provider._convert_messages_to_google_format(messages)

        assert "contents" in result
        assert len(result["contents"]) == 1
        assert result["contents"][0]["role"] == "user"
        assert result["systemInstruction"]["parts"][0]["text"] == "You are helpful"

    def test_convert_messages_to_google_format_with_assistant(self, provider):
        """Test message conversion with assistant message."""
        messages = [
            ChatMessage(role="user", content="Hello", name=None),
            ChatMessage(role="assistant", content="Hi there", name=None),
        ]
        result = provider._convert_messages_to_google_format(messages)

        assert len(result["contents"]) == 2
        assert result["contents"][0]["role"] == "user"
        assert result["contents"][1]["role"] == "model"
        assert result["contents"][1]["parts"][0]["text"] == "Hi there"

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, provider, mock_client):
        """Test successful chat completion."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello world"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        result = await provider.chat_completion(
            messages=messages, model="gemini-1.5-flash", max_tokens=100
        )

        assert result.choices[0].message.content == "Hello world"
        assert result.model == "gemini-1.5-flash"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.choices[0].finish_reason == "stop"

        # Verify request was made correctly
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "models/gemini-1.5-flash:generateContent" in call_args[0][0]
        request_data = call_args[1]["json"]
        assert "contents" in request_data
        assert "generationConfig" in request_data
        assert request_data["generationConfig"]["maxOutputTokens"] == 100

    @pytest.mark.asyncio
    async def test_chat_completion_with_temperature(self, provider, mock_client):
        """Test chat completion with temperature parameter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Response"}]}}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        await provider.chat_completion(messages=messages, model="gemini-1.5-flash", temperature=0.7)

        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["generationConfig"]["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_chat_completion_no_candidates(self, provider, mock_client):
        """Test chat completion when no candidates returned."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"candidates": []}
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        result = await provider.chat_completion(
            messages=messages, model="gemini-1.5-flash", max_tokens=100
        )

        assert result.choices[0].message.content == ""
        assert result.usage.prompt_tokens == 1  # Estimated
        assert result.usage.completion_tokens == 0

    @pytest.mark.asyncio
    async def test_chat_completion_no_content_parts(self, provider, mock_client):
        """Test chat completion when no content parts returned."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": []}}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 0},
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        result = await provider.chat_completion(
            messages=messages, model="gemini-1.5-flash", max_tokens=100
        )

        assert result.choices[0].message.content == ""
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 0

    @pytest.mark.asyncio
    async def test_chat_completion_finish_reason_mapping(self, provider, mock_client):
        """Test finish reason mapping from Google to OpenAI format."""
        test_cases = [
            ("MAX_TOKENS", "length"),
            ("SAFETY", "content_filter"),
            ("RECITATION", "content_filter"),
            ("OTHER", "stop"),
            ("UNKNOWN", "stop"),
        ]

        for google_reason, expected_reason in test_cases:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Response"}]},
                        "finishReason": google_reason,
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            }
            mock_client.post.return_value = mock_response

            messages = [ChatMessage(role="user", content="Hello", name=None)]
            result = await provider.chat_completion(
                messages=messages, model="gemini-1.5-flash", max_tokens=100
            )

            assert result.choices[0].finish_reason == expected_reason

    @pytest.mark.asyncio
    async def test_chat_completion_token_estimation_fallback(self, provider, mock_client):
        """Test token estimation when usage metadata not provided."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Short response"}]}}],
        }
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Hello world", name=None)]
        result = await provider.chat_completion(
            messages=messages, model="gemini-1.5-flash", max_tokens=100
        )

        # Should estimate tokens
        assert result.usage.prompt_tokens == 2  # "Hello world" ~ 2 tokens
        assert result.usage.completion_tokens == 2  # "Short response" ~ 2 tokens

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
        with pytest.raises(RateLimitError, match="Google API rate limit exceeded"):
            await provider.chat_completion(
                messages=messages, model="gemini-1.5-flash", max_tokens=100
            )

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
        with pytest.raises(AuthenticationError, match="Google API authentication failed"):
            await provider.chat_completion(
                messages=messages, model="gemini-1.5-flash", max_tokens=100
            )

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
        with pytest.raises(ProviderError, match="Google API error: 500"):
            await provider.chat_completion(
                messages=messages, model="gemini-1.5-flash", max_tokens=100
            )

    @pytest.mark.asyncio
    async def test_chat_completion_request_error(self, provider, mock_client):
        """Test chat completion with request error."""
        import httpx

        mock_client.post.side_effect = httpx.RequestError("Network error")

        messages = [ChatMessage(role="user", content="Hello", name=None)]
        with pytest.raises(ProviderError, match="Google request failed"):
            await provider.chat_completion(
                messages=messages, model="gemini-1.5-flash", max_tokens=100
            )

    @pytest.mark.skip(reason="Complex mocking issue with httpx stream context manager")
    @pytest.mark.asyncio
    async def test_stream_chat_completion_success(self, provider, mock_client):
        """Test successful streaming chat completion."""
        # TODO: Fix complex mocking for httpx.AsyncClient.stream
        pass

    @pytest.mark.skip(reason="Complex mocking issue with httpx stream context manager")
    @pytest.mark.asyncio
    async def test_stream_chat_completion_error(self, provider, mock_client):
        """Test streaming chat completion with error."""
        # TODO: Fix complex mocking for httpx.AsyncClient.stream
        pass

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider, mock_client):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "OK"}]}}],
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
        cost = provider.calculate_cost(0, 0, "gemini-1.5-flash")
        assert cost == 0.0

        # Test with very large token counts
        cost = provider.calculate_cost(1000000, 1000000, "gemini-1.5-flash")
        expected = (1000000 / 1_000_000) * 0.075 + (1000000 / 1_000_000) * 0.3
        assert cost == expected

        # Test with negative tokens (should still work)
        cost = provider.calculate_cost(-100, -50, "gemini-1.5-flash")
        expected = (-100 / 1_000_000) * 0.075 + (-50 / 1_000_000) * 0.3
        assert cost == expected

    def test_calculate_cost_different_models(self, provider):
        """Test cost calculation for different models."""
        # Test Gemini 1.5 Pro
        cost = provider.calculate_cost(1000, 500, "gemini-1.5-pro")
        expected = (1000 / 1_000_000) * 3.5 + (500 / 1_000_000) * 10.5
        assert cost == expected

        # Test Gemini 1.0 Pro
        cost = provider.calculate_cost(1000, 500, "gemini-1.0-pro")
        expected = (1000 / 1_000_000) * 0.5 + (500 / 1_000_000) * 1.5
        assert cost == expected

    def test_convert_messages_edge_cases(self, provider):
        """Test message conversion with edge cases."""
        # Empty messages
        result = provider._convert_messages_to_google_format([])
        assert result == {"contents": []}

        # Only system message
        messages = [ChatMessage(role="system", content="You are helpful", name=None)]
        result = provider._convert_messages_to_google_format(messages)
        assert result["contents"] == []
        assert result["systemInstruction"]["parts"][0]["text"] == "You are helpful"

        # Multiple system messages (should use last one)
        messages = [
            ChatMessage(role="system", content="First system", name=None),
            ChatMessage(role="system", content="Last system", name=None),
            ChatMessage(role="user", content="Hello", name=None),
        ]
        result = provider._convert_messages_to_google_format(messages)
        assert result["systemInstruction"]["parts"][0]["text"] == "Last system"
        assert len(result["contents"]) == 1

    def test_convert_messages_with_names(self, provider):
        """Test message conversion with message names (should be ignored for Google)."""
        messages = [
            ChatMessage(role="user", content="Hello", name="user1"),
            ChatMessage(role="assistant", content="Hi", name="assistant1"),
        ]
        result = provider._convert_messages_to_google_format(messages)
        assert result["contents"][0]["role"] == "user"
        assert result["contents"][1]["role"] == "model"
        # Names should not appear in Google format
        assert "name" not in result["contents"][0]
        assert "name" not in result["contents"][1]

    def test_convert_messages_empty_content(self, provider):
        """Test message conversion with empty content."""
        messages = [
            ChatMessage(role="system", content="", name=None),
            ChatMessage(role="user", content="Hello", name=None),
        ]
        result = provider._convert_messages_to_google_format(messages)
        # Empty system instruction is not included
        assert "systemInstruction" not in result
        assert result["contents"][0]["parts"][0]["text"] == "Hello"
