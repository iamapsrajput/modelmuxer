# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Comprehensive tests for LiteLLM provider implementation.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.models import ChatMessage
from app.providers.base import AuthenticationError, RateLimitError
from app.providers.litellm_provider import LiteLLMProvider


class TestLiteLLMProvider:
    """Test suite for LiteLLM provider."""

    @pytest.fixture
    def provider(self):
        """Create a LiteLLM provider instance for testing."""
        custom_models = {
            "gpt-3.5-turbo": {
                "pricing": {"input": 0.0015, "output": 0.002},
                "rate_limits": {"requests_per_minute": 60, "tokens_per_minute": 60000},
                "metadata": {"context_window": 4096},
            },
            "claude-3-haiku": {
                "pricing": {"input": 0.00025, "output": 0.00125},
                "rate_limits": {"requests_per_minute": 100, "tokens_per_minute": 100000},
                "metadata": {"context_window": 200000},
            },
        }

        return LiteLLMProvider(
            base_url="http://localhost:4000", api_key="test-api-key", custom_models=custom_models
        )

    @pytest.fixture
    def provider_no_key(self):
        """Create a LiteLLM provider without API key."""
        return LiteLLMProvider(base_url="http://localhost:4000", api_key=None)

    @pytest.fixture
    def sample_messages(self):
        """Sample chat messages for testing."""
        return [
            ChatMessage(role="system", content="You are a helpful assistant.", name=None),
            ChatMessage(role="user", content="Hello, how are you?", name=None),
        ]

    def test_provider_initialization(self, provider):
        """Test provider initialization."""
        assert provider.base_url == "http://localhost:4000"
        assert provider.api_key == "test-api-key"
        assert provider.provider_name == "litellm"
        assert len(provider.supported_models) == 2
        assert "gpt-3.5-turbo" in provider.supported_models
        assert "claude-3-haiku" in provider.supported_models

    def test_provider_initialization_no_key(self, provider_no_key):
        """Test provider initialization without API key."""
        assert provider_no_key.base_url == "http://localhost:4000"
        assert provider_no_key.api_key.startswith("TEST-PLACEHOLDER")
        assert provider_no_key.provider_name == "litellm"

    def test_provider_initialization_invalid_url(self):
        """Test provider initialization with invalid URL."""
        with pytest.raises(ValueError, match="LiteLLM base URL is required"):
            LiteLLMProvider(base_url="", api_key="test-key")

    def test_create_headers_with_key(self, provider):
        """Test header creation with API key."""
        headers = provider._create_headers()
        assert headers["Content-Type"] == "application/json"
        assert headers["User-Agent"] == "ModelMuxer/1.0.0 (LiteLLM Proxy)"
        assert headers["Authorization"] == "Bearer test-api-key"

    def test_create_headers_without_key(self, provider_no_key):
        """Test header creation without API key."""
        headers = provider_no_key._create_headers()
        assert headers["Content-Type"] == "application/json"
        assert headers["User-Agent"] == "ModelMuxer/1.0.0 (LiteLLM Proxy)"
        assert "Authorization" not in headers

    def test_get_supported_models(self, provider):
        """Test getting supported models."""
        models = provider.get_supported_models()
        assert len(models) == 2
        assert "gpt-3.5-turbo" in models
        assert "claude-3-haiku" in models

    def test_calculate_cost(self, provider):
        """Test cost calculation."""
        # Test with configured model
        cost = provider.calculate_cost(1000, 500, "gpt-3.5-turbo")
        expected_cost = (1000 / 1_000_000) * 0.0015 + (500 / 1_000_000) * 0.002
        assert cost == expected_cost

        # Test with unconfigured model (uses default pricing)
        cost = provider.calculate_cost(1000, 500, "unknown-model")
        expected_cost = (1000 / 1_000_000) * 1.0 + (500 / 1_000_000) * 2.0
        assert cost == expected_cost

    def test_get_rate_limits(self, provider):
        """Test getting rate limits."""
        rate_limits = provider.get_rate_limits()
        assert "requests_per_minute" in rate_limits
        assert "note" in rate_limits
        assert "gpt-3.5-turbo" in rate_limits["requests_per_minute"]
        assert rate_limits["requests_per_minute"]["gpt-3.5-turbo"]["requests_per_minute"] == 60

    def test_prepare_messages(self, provider, sample_messages):
        """Test message preparation."""
        prepared = provider._prepare_messages(sample_messages)
        assert len(prepared) == 2
        assert prepared[0]["role"] == "system"
        assert prepared[0]["content"] == "You are a helpful assistant."
        assert prepared[1]["role"] == "user"
        assert prepared[1]["content"] == "Hello, how are you?"

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, provider, sample_messages):
        """Test successful chat completion."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "Hello! I'm doing well, thank you."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }

        with patch.object(provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            response = await provider.chat_completion(
                messages=sample_messages, model="gpt-3.5-turbo", max_tokens=100, temperature=0.7
            )

            assert response.choices[0].message.content == "Hello! I'm doing well, thank you."
            assert response.model == "gpt-3.5-turbo"
            assert response.usage.prompt_tokens == 20
            assert response.usage.completion_tokens == 10
            assert response.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_completion_rate_limit_error(self, provider, sample_messages):
        """Test rate limit error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch.object(provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "Rate limit exceeded", request=MagicMock(), response=mock_response
            )

            with pytest.raises(RateLimitError):
                await provider.chat_completion(messages=sample_messages, model="gpt-3.5-turbo")

    @pytest.mark.asyncio
    async def test_chat_completion_auth_error(self, provider, sample_messages):
        """Test authentication error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch.object(provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "Unauthorized", request=MagicMock(), response=mock_response
            )

            with pytest.raises(AuthenticationError):
                await provider.chat_completion(messages=sample_messages, model="gpt-3.5-turbo")

    @pytest.mark.asyncio
    async def test_stream_chat_completion(self, provider, sample_messages):
        """Test streaming chat completion."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        # Mock streaming response
        stream_data = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" there!"}}]}',
            "data: [DONE]",
        ]

        async def mock_aiter_lines():
            for line in stream_data:
                yield line

        mock_response.aiter_lines = mock_aiter_lines

        with patch.object(provider.client, "stream") as mock_stream:
            mock_stream.return_value.__aenter__.return_value = mock_response

            chunks = []
            async for chunk in provider.stream_chat_completion(
                messages=sample_messages, model="gpt-3.5-turbo"
            ):
                chunks.append(chunk)

            assert len(chunks) == 2  # Excluding [DONE]
            assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
            assert chunks[1]["choices"][0]["delta"]["content"] == " there!"

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            result = await provider.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, provider):
        """Test health check failure."""
        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.RequestError("Connection failed")

            result = await provider.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_get_available_models(self, provider):
        """Test getting available models from proxy."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-3.5-turbo", "object": "model"},
                {"id": "claude-3-haiku", "object": "model"},
            ]
        }

        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            models = await provider.get_available_models()
            assert len(models) == 2
            assert models[0]["id"] == "gpt-3.5-turbo"
            assert models[1]["id"] == "claude-3-haiku"

    def test_add_custom_model(self, provider):
        """Test adding custom model configuration."""
        initial_count = len(provider.supported_models)

        provider.add_custom_model(
            model_name="custom-model",
            pricing={"input": 0.001, "output": 0.002},
            rate_limits={"requests_per_minute": 50, "tokens_per_minute": 50000},
            metadata={"provider": "custom"},
        )

        assert len(provider.supported_models) == initial_count + 1
        assert "custom-model" in provider.supported_models
        assert provider.pricing["custom-model"]["input"] == 0.001

    def test_get_model_info(self, provider):
        """Test getting model information."""
        # Test configured model
        info = provider.get_model_info("gpt-3.5-turbo")
        assert info["model"] == "gpt-3.5-turbo"
        assert info["provider"] == "litellm"
        assert info["pricing"]["input"] == 0.0015
        assert info["proxy_url"] == "http://localhost:4000"

        # Test unconfigured model
        info = provider.get_model_info("unknown-model")
        assert info["model"] == "unknown-model"
        assert info["provider"] == "litellm"
        assert info["pricing"] == provider.default_pricing
