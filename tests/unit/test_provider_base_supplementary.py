# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""
Supplementary tests for app/providers/base.py to increase coverage.
"""

from unittest.mock import AsyncMock, Mock, patch
import pytest
from typing import Any, Dict, List, Optional

from app.providers.base import (
    LLMProvider,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
    ProviderResponse,
    SimpleCircuitBreaker,
    estimate_tokens,
    normalize_finish_reason,
    _is_retryable_error,
)
from app.models import ChatMessage


class TestProviderBase:
    """Test provider base classes and utilities."""

    def test_estimate_tokens(self):
        """Test token estimation function."""
        # Test normal text
        assert estimate_tokens("Hello world") == 2  # 11 chars / 4 = 2.75 -> 2
        assert (
            estimate_tokens("This is a longer sentence with more words") == 10
        )  # 42 chars / 4 = 10.5 -> 10

        # Test edge cases
        assert estimate_tokens("") == 1  # Empty string returns 1
        assert estimate_tokens("a") == 1  # Single char returns 1
        assert estimate_tokens("abc") == 1  # Less than 4 chars returns 1

    def test_normalize_finish_reason(self):
        """Test finish reason normalization."""
        # Test Google provider
        assert normalize_finish_reason("google", "STOP") == "stop"
        assert normalize_finish_reason("google", "MAX_TOKENS") == "length"
        assert normalize_finish_reason("google", "SAFETY") == "content_filter"
        assert normalize_finish_reason("google", "RECITATION") == "content_filter"
        assert normalize_finish_reason("google", "UNKNOWN") == "stop"  # Default

        # Test Cohere provider
        assert normalize_finish_reason("cohere", "COMPLETE") == "stop"
        assert normalize_finish_reason("cohere", "MAX_TOKENS") == "length"
        assert normalize_finish_reason("cohere", "ERROR_TOXIC") == "content_filter"

        # Test OpenAI provider
        assert normalize_finish_reason("openai", "stop") == "stop"
        assert normalize_finish_reason("openai", "length") == "length"

        # Test Anthropic provider
        assert normalize_finish_reason("anthropic", "end_turn") == "stop"
        assert normalize_finish_reason("anthropic", "max_tokens") == "length"

        # Test unknown provider
        assert normalize_finish_reason("unknown", "anything") == "stop"

        # Test None finish reason
        assert normalize_finish_reason("google", None) == "stop"

    def test_is_retryable_error(self):
        """Test retryable error detection."""
        # Test HTTP status codes
        assert _is_retryable_error("google", 429, None) is True  # Rate limit
        assert _is_retryable_error("google", 500, None) is True  # Server error
        assert _is_retryable_error("google", 503, None) is True  # Service unavailable
        assert _is_retryable_error("google", 400, None) is False  # Bad request
        assert _is_retryable_error("google", 401, None) is False  # Unauthorized
        assert _is_retryable_error("google", 404, None) is False  # Not found

        # Test Google-specific error codes
        payload = {"error": {"code": "quota_exceeded"}}
        assert _is_retryable_error("google", None, payload) is True

        payload = {"error": {"code": "resource_exhausted"}}
        assert _is_retryable_error("google", None, payload) is True

        # Test error message keywords
        payload = {"error": {"message": "Rate limit exceeded"}}
        assert _is_retryable_error("openai", None, payload) is True

        payload = {"error": {"message": "Internal server error"}}
        assert _is_retryable_error("anthropic", None, payload) is True

        # Test non-retryable errors
        payload = {"error": {"message": "Invalid API key"}}
        assert _is_retryable_error("openai", None, payload) is False

        # Test empty payload
        assert _is_retryable_error("google", None, {}) is False
        assert _is_retryable_error("google", None, None) is False

    def test_provider_response(self):
        """Test ProviderResponse dataclass."""
        response = ProviderResponse(
            output_text="Hello world", tokens_in=10, tokens_out=20, latency_ms=150
        )

        assert response.output_text == "Hello world"
        assert response.tokens_in == 10
        assert response.tokens_out == 20
        assert response.latency_ms == 150
        assert response.raw is None
        assert response.error is None

        # Test with optional fields
        response = ProviderResponse(
            output_text="Error occurred",
            tokens_in=0,
            tokens_out=0,
            latency_ms=50,
            raw={"error": "test"},
            error="Test error",
        )

        assert response.error == "Test error"
        assert response.raw == {"error": "test"}

    def test_simple_circuit_breaker(self):
        """Test SimpleCircuitBreaker functionality."""
        # Create circuit breaker with low threshold for testing
        breaker = SimpleCircuitBreaker(fail_threshold=2, cooldown_sec=1)

        # Initially closed
        assert breaker.is_open() is False
        assert breaker.failures == 0

        # First failure
        breaker.on_failure()
        assert breaker.failures == 1
        assert breaker.is_open() is False

        # Second failure - should open
        breaker.on_failure()
        assert breaker.failures == 2
        assert breaker.is_open() is True

        # Success should reset
        import time

        time.sleep(1.1)  # Wait for cooldown
        breaker.on_success()
        assert breaker.failures == 0
        assert breaker.is_open() is False

    def test_provider_error_initialization(self):
        """Test ProviderError exception."""
        error = ProviderError("Test error message", status_code=500, provider="test-provider")

        assert str(error) == "Test error message"
        assert error.status_code == 500
        assert error.provider == "test-provider"

        # Test without optional parameters
        error = ProviderError("Simple error")
        assert str(error) == "Simple error"
        assert error.status_code is None
        assert error.provider is None

    def test_rate_limit_error(self):
        """Test RateLimitError exception."""
        error = RateLimitError("Rate limit exceeded", status_code=429, provider="test-provider")

        assert str(error) == "Rate limit exceeded"
        assert error.status_code == 429
        assert error.provider == "test-provider"

    def test_authentication_error(self):
        """Test AuthenticationError exception."""
        error = AuthenticationError("Invalid API key", status_code=401, provider="test-provider")

        assert str(error) == "Invalid API key"
        assert error.status_code == 401
        assert error.provider == "test-provider"

    def test_model_not_found_error(self):
        """Test ModelNotFoundError exception."""
        error = ModelNotFoundError(
            "Model gpt-5 not found", status_code=404, provider="test-provider"
        )

        assert str(error) == "Model gpt-5 not found"
        assert error.status_code == 404
        assert error.provider == "test-provider"

    def test_llm_provider_abstract_class(self):
        """Test that LLMProvider is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            # Should raise TypeError because abstract methods are not implemented
            LLMProvider("test-key", "https://api.test.com", "test-provider")

    def test_exception_inheritance(self):
        """Test that custom exceptions inherit from ProviderError."""
        rate_limit_error = RateLimitError("Rate limited", provider="test")
        auth_error = AuthenticationError("Auth failed", provider="test")
        model_error = ModelNotFoundError("Model not found", provider="test")

        assert isinstance(rate_limit_error, ProviderError)
        assert isinstance(auth_error, ProviderError)
        assert isinstance(model_error, ProviderError)

        # All should also be Exception instances
        assert isinstance(rate_limit_error, Exception)
        assert isinstance(auth_error, Exception)
        assert isinstance(model_error, Exception)

    def test_is_retryable_error_with_cohere(self):
        """Test Cohere-specific retryable error patterns."""
        # Test Cohere error codes
        payload = {"error": {"code": "rate_limit"}}
        assert _is_retryable_error("cohere", None, payload) is True

        payload = {"error": {"code": "overloaded"}}
        assert _is_retryable_error("cohere", None, payload) is True

        # Test Cohere error messages
        payload = {"error": {"message": "Server is overloaded, please retry"}}
        assert _is_retryable_error("cohere", None, payload) is True

    def test_is_retryable_error_with_together(self):
        """Test Together-specific retryable error patterns."""
        # Test Together error codes
        payload = {"error": {"code": "server_error"}}
        assert _is_retryable_error("together", None, payload) is True

        # Test Together error messages
        payload = {"error": {"message": "Please retry your request"}}
        assert _is_retryable_error("together", None, payload) is True

    def test_normalize_finish_reason_together(self):
        """Test Together provider finish reason normalization."""
        assert normalize_finish_reason("together", "stop") == "stop"
        assert normalize_finish_reason("together", "length") == "length"
        assert normalize_finish_reason("together", "content_filter") == "content_filter"
        assert normalize_finish_reason("together", "unknown") == "stop"


class MockProvider(LLMProvider):
    """Mock provider for testing concrete implementation."""

    def __init__(self, api_key: str, base_url: str, provider_name: str):
        super().__init__(api_key, base_url, provider_name)

    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
        **kwargs,
    ):
        """Mock chat completion."""
        return {
            "id": "mock-id",
            "choices": [{"message": {"role": "assistant", "content": "Mock response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

    async def stream_chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ):
        """Mock streaming chat completion."""
        yield {"choices": [{"delta": {"content": "Mock"}}]}
        yield {"choices": [{"delta": {"content": " response"}}]}

    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Mock cost calculation."""
        # Simple mock calculation
        return (input_tokens * 0.001) + (output_tokens * 0.002)

    def get_supported_models(self) -> List[str]:
        """Get list of supported models."""
        return ["mock-model-1", "mock-model-2"]

    def validate_model(self, model: str) -> bool:
        """Validate if model is supported."""
        return model in self.get_supported_models()


class TestMockProvider:
    """Test the mock provider implementation."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider instance."""
        return MockProvider("mock-key", "https://api.mock.com", "mock-provider")

    @pytest.mark.asyncio
    async def test_mock_chat_completion(self, mock_provider):
        """Test mock provider chat completion."""
        messages = [ChatMessage(role="user", content="Hello")]
        response = await mock_provider.chat_completion(messages, "mock-model-1")

        assert response["id"] == "mock-id"
        assert response["choices"][0]["message"]["content"] == "Mock response"
        assert response["usage"]["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_mock_stream_chat_completion(self, mock_provider):
        """Test mock provider streaming."""
        messages = [ChatMessage(role="user", content="Hello")]

        chunks = []
        async for chunk in mock_provider.stream_chat_completion(messages, "mock-model-1"):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0]["choices"][0]["delta"]["content"] == "Mock"
        assert chunks[1]["choices"][0]["delta"]["content"] == " response"

    def test_mock_calculate_cost(self, mock_provider):
        """Test mock provider cost calculation."""
        cost = mock_provider.calculate_cost(100, 200, "mock-model-1")

        # 100 * 0.001 + 200 * 0.002 = 0.1 + 0.4 = 0.5
        assert cost == 0.5

    def test_mock_get_supported_models(self, mock_provider):
        """Test getting supported models."""
        models = mock_provider.get_supported_models()

        assert len(models) == 2
        assert "mock-model-1" in models
        assert "mock-model-2" in models

    def test_mock_validate_model(self, mock_provider):
        """Test model validation."""
        assert mock_provider.validate_model("mock-model-1") is True
        assert mock_provider.validate_model("mock-model-2") is True
        assert mock_provider.validate_model("unsupported-model") is False
