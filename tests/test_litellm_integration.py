# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Integration tests for LiteLLM provider with ModelMuxer routing system.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.models import ChatMessage
from app.providers.base import AuthenticationError, ProviderError, RateLimitError
from app.providers.litellm_provider import LiteLLMProvider
from app.router import HeuristicRouter


class TestLiteLLMIntegration:
    """Integration tests for LiteLLM provider with routing system."""

    @pytest.fixture
    def litellm_provider(self):
        """Create a LiteLLM provider for integration testing."""
        custom_models = {
            "gpt-3.5-turbo": {
                "pricing": {"input": 0.0015, "output": 0.002},
                "rate_limits": {"requests_per_minute": 60, "tokens_per_minute": 60000},
                "metadata": {"context_window": 4096, "task_types": ["general", "simple"]},
            },
            "claude-3-haiku": {
                "pricing": {"input": 0.00025, "output": 0.00125},
                "rate_limits": {"requests_per_minute": 100, "tokens_per_minute": 100000},
                "metadata": {"context_window": 200000, "task_types": ["complex", "code"]},
            },
            "gpt-4": {
                "pricing": {"input": 0.03, "output": 0.06},
                "rate_limits": {"requests_per_minute": 20, "tokens_per_minute": 20000},
                "metadata": {"context_window": 8192, "task_types": ["complex", "code"]},
            },
        }

        return LiteLLMProvider(
            base_url="http://localhost:4000",
            api_key="test-integration-key",
            custom_models=custom_models,
        )

    @pytest.fixture
    def router(self):
        """Create a heuristic router for testing."""
        return HeuristicRouter()

    @pytest.fixture
    def providers_dict(self, litellm_provider):
        """Create providers dictionary for routing tests."""
        return {"litellm": litellm_provider}

    def test_litellm_provider_in_routing_system(self, router, providers_dict):
        """Test that LiteLLM provider integrates with routing system."""
        # Test simple query routing
        simple_messages = [ChatMessage(role="user", content="What is 2+2?", name=None)]

        # Mock the routing logic to prefer LiteLLM for simple tasks
        with patch.object(router, "select_model") as mock_select:
            mock_select.return_value = (
                "litellm",
                "gpt-3.5-turbo",
                "Simple query, using cost-effective model",
            )

            provider, model, reason = router.select_model(simple_messages)
            assert provider == "litellm"
            assert model == "gpt-3.5-turbo"
            assert "cost-effective" in reason.lower()

    def test_litellm_code_task_routing(self, router, providers_dict):
        """Test LiteLLM provider selection for code tasks."""
        code_messages = [
            ChatMessage(
                role="user",
                content="Write a Python function to implement binary search algorithm",
                name=None,
            )
        ]

        with patch.object(router, "select_model") as mock_select:
            mock_select.return_value = (
                "litellm",
                "claude-3-haiku",
                "Code task, using capable model",
            )

            provider, model, reason = router.select_model(code_messages)
            assert provider == "litellm"
            assert model == "claude-3-haiku"

    @pytest.mark.asyncio
    async def test_litellm_provider_fallback_mechanism(self, litellm_provider):
        """Test LiteLLM provider fallback when primary model fails."""
        messages = [ChatMessage(role="user", content="Hello", name=None)]

        # Mock first call to fail, second to succeed
        mock_responses = [
            httpx.HTTPStatusError(
                "Rate limit", request=MagicMock(), response=MagicMock(status_code=429)
            ),
            MagicMock(
                status_code=200,
                json=lambda: {
                    "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                },
            ),
        ]

        with patch.object(litellm_provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = mock_responses

            # First call should raise RateLimitError
            with pytest.raises((RateLimitError, ProviderError)):
                await litellm_provider.chat_completion(messages, "gpt-3.5-turbo")

            # Reset mock for second call
            mock_post.side_effect = None
            mock_post.return_value = mock_responses[1]

            # Second call should succeed
            response = await litellm_provider.chat_completion(messages, "claude-3-haiku")
            assert response.choices[0].message.content == "Hello!"

    @pytest.mark.asyncio
    async def test_litellm_cost_optimization(self, litellm_provider):
        """Test cost calculation and optimization with LiteLLM."""
        # Test cost calculation for different models
        gpt35_cost = litellm_provider.calculate_cost(1000, 500, "gpt-3.5-turbo")
        claude_cost = litellm_provider.calculate_cost(1000, 500, "claude-3-haiku")
        gpt4_cost = litellm_provider.calculate_cost(1000, 500, "gpt-4")

        # Claude should be cheapest, GPT-4 most expensive
        assert claude_cost < gpt35_cost < gpt4_cost

        # Verify actual cost calculations
        expected_gpt35 = (1000 / 1_000_000) * 0.0015 + (500 / 1_000_000) * 0.002
        expected_claude = (1000 / 1_000_000) * 0.00025 + (500 / 1_000_000) * 0.00125
        expected_gpt4 = (1000 / 1_000_000) * 0.03 + (500 / 1_000_000) * 0.06

        assert abs(gpt35_cost - expected_gpt35) < 0.0001
        assert abs(claude_cost - expected_claude) < 0.0001
        assert abs(gpt4_cost - expected_gpt4) < 0.0001

    @pytest.mark.asyncio
    async def test_litellm_rate_limiting_integration(self, litellm_provider):
        """Test rate limiting integration with LiteLLM provider."""
        rate_limits = litellm_provider.get_rate_limits()

        assert "requests_per_minute" in rate_limits
        assert "gpt-3.5-turbo" in rate_limits["requests_per_minute"]
        assert rate_limits["requests_per_minute"]["gpt-3.5-turbo"]["requests_per_minute"] == 60
        assert rate_limits["requests_per_minute"]["claude-3-haiku"]["requests_per_minute"] == 100

    @pytest.mark.asyncio
    async def test_litellm_streaming_integration(self, litellm_provider):
        """Test streaming integration with LiteLLM provider."""
        messages = [ChatMessage(role="user", content="Count to 5", name=None)]

        # Mock streaming response
        stream_chunks = [
            'data: {"choices":[{"delta":{"content":"1"}}]}',
            'data: {"choices":[{"delta":{"content":", 2"}}]}',
            'data: {"choices":[{"delta":{"content":", 3"}}]}',
            'data: {"choices":[{"delta":{"content":", 4"}}]}',
            'data: {"choices":[{"delta":{"content":", 5"}}]}',
            "data: [DONE]",
        ]

        async def mock_aiter_lines():
            for chunk in stream_chunks:
                yield chunk

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        with patch.object(litellm_provider.client, "stream") as mock_stream:
            mock_stream.return_value.__aenter__.return_value = mock_response

            collected_content = []
            async for chunk in litellm_provider.stream_chat_completion(
                messages=messages, model="gpt-3.5-turbo"
            ):
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        collected_content.append(delta["content"])

            full_content = "".join(collected_content)
            assert "1, 2, 3, 4, 5" == full_content

    def test_litellm_model_configuration_validation(self, litellm_provider):
        """Test model configuration validation."""
        # Test that all configured models have required fields
        for model_name in litellm_provider.supported_models:
            model_info = litellm_provider.get_model_info(model_name)

            assert "model" in model_info
            assert "provider" in model_info
            assert "pricing" in model_info
            assert "rate_limits" in model_info
            assert model_info["provider"] == "litellm"

            # Verify pricing structure
            pricing = model_info["pricing"]
            assert "input" in pricing
            assert "output" in pricing
            assert isinstance(pricing["input"], int | float)
            assert isinstance(pricing["output"], int | float)

    @pytest.mark.asyncio
    async def test_litellm_error_handling_integration(self, litellm_provider):
        """Test comprehensive error handling in integration scenarios."""
        messages = [ChatMessage(role="user", content="Test message", name=None)]

        # Test various HTTP errors
        error_scenarios = [
            (400, "Bad Request"),
            (401, "Unauthorized"),
            (403, "Forbidden"),
            (404, "Not Found"),
            (429, "Rate Limited"),
            (500, "Internal Server Error"),
            (502, "Bad Gateway"),
            (503, "Service Unavailable"),
        ]

        for status_code, error_msg in error_scenarios:
            mock_response = MagicMock()
            mock_response.status_code = status_code

            with patch.object(litellm_provider.client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.side_effect = httpx.HTTPStatusError(
                    error_msg, request=MagicMock(), response=mock_response
                )

                with pytest.raises((RateLimitError, AuthenticationError, ProviderError)):
                    await litellm_provider.chat_completion(messages, "gpt-3.5-turbo")

    def test_litellm_custom_model_management(self, litellm_provider):
        """Test dynamic model management capabilities."""
        initial_model_count = len(litellm_provider.supported_models)

        # Add a new custom model
        litellm_provider.add_custom_model(
            model_name="custom-llama-7b",
            pricing={"input": 0.0001, "output": 0.0002},
            rate_limits={"requests_per_minute": 200, "tokens_per_minute": 200000},
            metadata={"provider": "ollama", "size": "7B", "task_types": ["general"]},
        )

        # Verify model was added
        assert len(litellm_provider.supported_models) == initial_model_count + 1
        assert "custom-llama-7b" in litellm_provider.supported_models

        # Verify model configuration
        model_info = litellm_provider.get_model_info("custom-llama-7b")
        assert model_info["pricing"]["input"] == 0.0001
        assert model_info["rate_limits"]["requests_per_minute"] == 200
        assert model_info["metadata"]["provider"] == "ollama"

    @pytest.mark.asyncio
    async def test_litellm_health_monitoring_integration(self, litellm_provider):
        """Test health monitoring integration."""
        # Test successful health check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "gpt-3.5-turbo"}]}

        with patch.object(litellm_provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            health_status = await litellm_provider.health_check()
            assert health_status is True

        # Test failed health check
        with patch.object(litellm_provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.RequestError("Connection failed")

            health_status = await litellm_provider.health_check()
            assert health_status is False

    def test_litellm_environment_configuration(self):
        """Test LiteLLM provider configuration from environment variables."""
        test_env = {"LITELLM_BASE_URL": "http://test-proxy:4000", "LITELLM_API_KEY": "test-env-key"}

        with patch.dict(os.environ, test_env):
            provider = LiteLLMProvider(
                base_url=os.getenv("LITELLM_BASE_URL"), api_key=os.getenv("LITELLM_API_KEY")
            )

            assert provider.base_url == "http://test-proxy:4000"
            assert provider.api_key == "test-env-key"
            assert provider.provider_name == "litellm"
