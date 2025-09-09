# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""Example tests using realistic provider mocks."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from tests.fixtures.realistic_mocks import (
    RealisticMockProvider,
    RealisticMockRouter,
    create_realistic_mock_response,
    create_realistic_provider_registry,
)


class TestRealisticProviderMocks:
    """Test suite demonstrating realistic provider mocking."""

    @pytest.mark.asyncio
    async def test_realistic_provider_behavior(self):
        """Test realistic provider with latency and error simulation."""
        # Create provider with 5% error rate and 200ms latency
        provider = RealisticMockProvider("openai", latency_ms=200, error_rate=0.05)

        # Test multiple requests
        success_count = 0
        error_count = 0

        for i in range(20):
            try:
                response = await provider.chat_completion(
                    messages=[{"role": "user", "content": f"Test message {i}"}],
                    model="gpt-4o-mini",
                    max_tokens=50,
                )
                assert "choices" in response
                assert response["model"] == "gpt-4o-mini"
                success_count += 1
            except Exception as e:
                assert "Simulated API error" in str(e)
                error_count += 1

        # Verify error rate is approximately as configured
        actual_error_rate = error_count / 20
        assert 0.0 <= actual_error_rate <= 0.15  # Allow some variance

        # Check statistics
        stats = provider.get_stats()
        assert stats["request_count"] == 20
        assert stats["error_count"] == error_count

    @pytest.mark.asyncio
    async def test_realistic_routing_decisions(self):
        """Test realistic routing based on query complexity."""
        router = RealisticMockRouter()

        # Test simple query
        simple_result = await router.select_model(
            messages=[{"role": "user", "content": "Hello"}], constraints={"max_cost": 0.1}
        )
        provider, model, reason, _, metadata = simple_result
        assert provider == "openai"
        assert model == "gpt-3.5-turbo"
        assert "general query" in reason.lower()

        # Test complex query
        complex_result = await router.select_model(
            messages=[
                {
                    "role": "user",
                    "content": "Please analyze the following code and explain the algorithmic complexity of each function, providing detailed examples and optimization suggestions: "
                    + "x" * 500,
                }
            ],
            constraints={"max_cost": 0.1},
        )
        provider, model, reason, _, metadata = complex_result
        assert provider == "openai"
        assert model == "gpt-4o-mini"
        assert "complex" in reason.lower()

        # Test budget-constrained query
        budget_result = await router.select_model(
            messages=[{"role": "user", "content": "Translate this to French"}],
            constraints={"max_cost": 0.0005},
        )
        provider, model, reason, _, metadata = budget_result
        assert provider == "groq"
        assert model == "llama-3.1-8b"
        assert "budget" in reason.lower()

        # Check routing statistics
        stats = router.get_routing_stats()
        assert stats["total_routes"] == 3
        assert "openai" in stats["provider_distribution"]
        assert "groq" in stats["provider_distribution"]

    @pytest.mark.asyncio
    async def test_provider_registry_creation(self):
        """Test creation of multiple realistic providers."""
        registry = create_realistic_provider_registry(
            providers=["openai", "anthropic", "groq"], latency_ms=100, error_rate=0.01
        )

        assert len(registry) == 3
        assert "openai" in registry
        assert "anthropic" in registry
        assert "groq" in registry

        # Test different latencies
        assert registry["groq"].latency_ms == 50  # Groq should be faster
        assert registry["anthropic"].latency_ms == 150  # Claude should be slower

        # Test a request with each provider
        for name, provider in registry.items():
            response = await provider.chat_completion(
                messages=[{"role": "user", "content": f"Test {name}"}],
                model="test-model",
                max_tokens=20,
            )
            assert response["system_fingerprint"] == f"fp_{name}"

    @pytest.mark.asyncio
    async def test_realistic_response_generation(self):
        """Test generation of realistic responses."""
        messages = [{"role": "user", "content": "Explain quantum computing"}]

        # Test high-tier model response
        premium_response = create_realistic_mock_response(
            model="gpt-4", messages=messages, provider="openai"
        )
        assert "comprehensive and detailed" in premium_response["choices"][0]["message"]["content"]

        # Test mid-tier model response
        standard_response = create_realistic_mock_response(
            model="gpt-3.5-turbo", messages=messages, provider="openai"
        )
        assert "clear and concise" in standard_response["choices"][0]["message"]["content"]

        # Test budget model response
        budget_response = create_realistic_mock_response(
            model="llama-3.1-8b", messages=messages, provider="groq"
        )
        assert "basic" in budget_response["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_streaming_response_mock(self):
        """Test realistic streaming response generation."""
        messages = [{"role": "user", "content": "Tell me a story"}]

        stream = create_realistic_mock_response(
            model="gpt-4", messages=messages, provider="openai", stream=True
        )

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
            if chunk.startswith("data: [DONE]"):
                break

        # Verify we got multiple chunks
        assert len(chunks) > 5  # Should have several word chunks
        assert chunks[-1] == "data: [DONE]\n\n"

        # Verify chunk format
        import json

        first_chunk = chunks[0]
        assert first_chunk.startswith("data: ")
        chunk_data = json.loads(first_chunk[6:])  # Skip "data: "
        assert chunk_data["object"] == "chat.completion.chunk"
        assert "delta" in chunk_data["choices"][0]

    @pytest.mark.asyncio
    async def test_cost_calculation_accuracy(self):
        """Test accurate cost calculation for different models."""
        provider = RealisticMockProvider("openai")

        # Test GPT-4 pricing
        cost_gpt4 = provider.calculate_cost(1000, 500, "gpt-4")
        assert 0.04 < cost_gpt4 <= 0.06  # ~$0.045 for 1K in + 0.5K out

        # Test GPT-3.5 pricing
        cost_gpt35 = provider.calculate_cost(1000, 500, "gpt-3.5-turbo")
        assert 0.001 < cost_gpt35 < 0.002  # Much cheaper

        # Test Groq pricing
        cost_groq = provider.calculate_cost(1000, 500, "llama-3.1-8b")
        assert cost_groq < 0.0001  # Very cheap

    def test_integration_with_existing_tests(self):
        """Example of how to integrate realistic mocks with existing tests."""
        from fastapi.testclient import TestClient

        # Create realistic mocks
        mock_providers = create_realistic_provider_registry()
        mock_router = RealisticMockRouter()

        # Patch the actual implementations
        with patch("app.providers.registry.get_provider_registry", return_value=mock_providers):
            with patch("app.router.HeuristicRouter", return_value=mock_router):
                # Now your tests will use realistic mocks
                # that simulate latency, errors, and routing logic
                pass


@pytest.mark.asyncio
async def test_concurrent_provider_requests():
    """Test concurrent requests to multiple providers."""
    registry = create_realistic_provider_registry(
        providers=["openai", "anthropic", "groq", "mistral"], latency_ms=100
    )

    # Create tasks for concurrent requests
    tasks = []
    for provider_name, provider in registry.items():
        task = provider.chat_completion(
            messages=[{"role": "user", "content": f"Test {provider_name}"}],
            model=f"{provider_name}-model",
            max_tokens=50,
        )
        tasks.append(task)

    # Run all requests concurrently
    start_time = asyncio.get_event_loop().time()
    responses = await asyncio.gather(*tasks)
    end_time = asyncio.get_event_loop().time()

    # Verify all responses
    assert len(responses) == 4
    for response in responses:
        assert "choices" in response
        assert response["usage"]["total_tokens"] > 0

    # Verify concurrent execution (should take ~150ms for anthropic, not 400ms+)
    elapsed_ms = (end_time - start_time) * 1000
    assert elapsed_ms < 300  # Should run in parallel


if __name__ == "__main__":
    # Run a simple demonstration
    asyncio.run(test_concurrent_provider_requests())
    print("Realistic mock tests completed successfully!")
