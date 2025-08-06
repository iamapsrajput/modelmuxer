# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Performance benchmarks for LiteLLM provider.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models import ChatMessage
from app.providers.litellm_provider import LiteLLMProvider


class TestLiteLLMBenchmarks:
    """Performance benchmarks for LiteLLM provider."""

    @pytest.fixture
    def provider(self):
        """Create a LiteLLM provider for benchmarking."""
        custom_models = {
            "gpt-3.5-turbo": {
                "pricing": {"input": 0.0015, "output": 0.002},
                "rate_limits": {"requests_per_minute": 60, "tokens_per_minute": 60000},
            },
            "claude-3-haiku": {
                "pricing": {"input": 0.00025, "output": 0.00125},
                "rate_limits": {"requests_per_minute": 100, "tokens_per_minute": 100000},
            },
        }

        return LiteLLMProvider(
            base_url="http://localhost:4000", api_key="benchmark-key", custom_models=custom_models
        )

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for benchmarking."""
        return [
            ChatMessage(role="system", content="You are a helpful assistant.", name=None),
            ChatMessage(
                role="user", content="Explain quantum computing in simple terms.", name=None
            ),
        ]

    def test_provider_initialization_performance(self, benchmark):
        """Benchmark provider initialization time."""

        def create_provider():
            return LiteLLMProvider(
                base_url="http://localhost:4000",
                api_key="test-key",
                custom_models={
                    f"model-{i}": {
                        "pricing": {"input": 0.001, "output": 0.002},
                        "rate_limits": {"requests_per_minute": 100, "tokens_per_minute": 100000},
                    }
                    for i in range(10)  # 10 models
                },
            )

        result = benchmark(create_provider)
        assert result is not None
        assert len(result.supported_models) == 10

    def test_cost_calculation_performance(self, benchmark, provider):
        """Benchmark cost calculation performance."""

        def calculate_costs():
            total_cost = 0
            for model in provider.supported_models:
                for input_tokens in [100, 1000, 10000]:
                    for output_tokens in [50, 500, 5000]:
                        total_cost += provider.calculate_cost(input_tokens, output_tokens, model)
            return total_cost

        result = benchmark(calculate_costs)
        assert result > 0

    def test_message_preparation_performance(self, benchmark, provider):
        """Benchmark message preparation performance."""
        # Create a large set of messages
        large_message_set = [
            ChatMessage(role="user", content=f"Message {i} with some content", name=f"user_{i}")
            for i in range(100)
        ]

        def prepare_messages():
            return provider._prepare_messages(large_message_set)

        result = benchmark(prepare_messages)
        assert len(result) == 100

    def test_header_creation_performance(self, benchmark, provider):
        """Benchmark header creation performance."""

        def create_headers():
            headers_list = []
            for _ in range(1000):
                headers_list.append(provider._create_headers())
            return headers_list

        result = benchmark(create_headers)
        assert len(result) == 1000

    @pytest.mark.asyncio
    async def test_concurrent_chat_completions_performance(self, provider, sample_messages):
        """Benchmark concurrent chat completion performance."""
        # Mock successful responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        with patch.object(provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            # Test concurrent requests
            start_time = time.time()

            tasks = [
                provider.chat_completion(
                    messages=sample_messages, model="gpt-3.5-turbo", max_tokens=100
                )
                for _ in range(10)  # 10 concurrent requests
            ]

            responses = await asyncio.gather(*tasks)

            end_time = time.time()
            duration = end_time - start_time

            # All requests should complete
            assert len(responses) == 10
            assert all(r.choices[0].message.content == "Test response" for r in responses)

            # Should complete in reasonable time (less than 5 seconds for mocked requests)
            assert duration < 5.0

            # Log performance metrics
            print(f"Concurrent requests completed in {duration:.2f} seconds")
            print(f"Average time per request: {duration / 10:.3f} seconds")

    @pytest.mark.asyncio
    async def test_streaming_performance(self, provider, sample_messages):
        """Benchmark streaming performance."""
        # Create a large streaming response
        stream_chunks = [
            f'data: {{"choices":[{{"delta":{{"content":"Chunk {i} "}}}}]}}' for i in range(100)
        ] + ["data: [DONE]"]

        async def mock_aiter_lines():
            for chunk in stream_chunks:
                yield chunk
                # Small delay to simulate network latency
                await asyncio.sleep(0.001)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        with patch.object(provider.client, "stream") as mock_stream:
            mock_stream.return_value.__aenter__.return_value = mock_response

            start_time = time.time()

            chunk_count = 0
            async for _ in provider.stream_chat_completion(
                messages=sample_messages, model="gpt-3.5-turbo"
            ):
                chunk_count += 1

            end_time = time.time()
            duration = end_time - start_time

            assert chunk_count == 100  # Should receive all chunks

            # Log streaming performance
            print(f"Streamed {chunk_count} chunks in {duration:.2f} seconds")
            print(f"Average time per chunk: {duration / chunk_count:.4f} seconds")

    def test_model_info_retrieval_performance(self, benchmark, provider):
        """Benchmark model info retrieval performance."""

        def get_all_model_info():
            model_infos = []
            for model in provider.supported_models:
                model_infos.append(provider.get_model_info(model))
            # Also test unknown models
            for i in range(10):
                model_infos.append(provider.get_model_info(f"unknown-model-{i}"))
            return model_infos

        result = benchmark(get_all_model_info)
        assert len(result) >= len(provider.supported_models) + 10

    def test_rate_limit_calculation_performance(self, benchmark, provider):
        """Benchmark rate limit calculation performance."""

        def get_rate_limits_multiple():
            rate_limits_list = []
            for _ in range(100):
                rate_limits_list.append(provider.get_rate_limits())
            return rate_limits_list

        result = benchmark(get_rate_limits_multiple)
        assert len(result) == 100

    @pytest.mark.asyncio
    async def test_health_check_performance(self, provider):
        """Benchmark health check performance."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(provider.client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            # Test multiple health checks
            start_time = time.time()

            health_results = []
            for _ in range(10):
                result = await provider.health_check()
                health_results.append(result)

            end_time = time.time()
            duration = end_time - start_time

            assert all(health_results)  # All should be True

            # Health checks should be fast
            assert duration < 2.0

            print(f"10 health checks completed in {duration:.2f} seconds")
            print(f"Average health check time: {duration / 10:.3f} seconds")

    def test_custom_model_addition_performance(self, benchmark, provider):
        """Benchmark custom model addition performance."""

        def add_multiple_models():
            for i in range(50):
                provider.add_custom_model(
                    model_name=f"benchmark-model-{i}",
                    pricing={"input": 0.001 + i * 0.0001, "output": 0.002 + i * 0.0001},
                    rate_limits={
                        "requests_per_minute": 100 + i,
                        "tokens_per_minute": 100000 + i * 1000,
                    },
                    metadata={"benchmark": True, "index": i},
                )

        initial_count = len(provider.supported_models)
        benchmark(add_multiple_models)
        final_count = len(provider.supported_models)

        # Note: benchmark may run multiple times, so we check that models were added
        assert final_count >= initial_count + 50

    @pytest.mark.asyncio
    async def test_error_handling_performance(self, provider, sample_messages):
        """Benchmark error handling performance."""
        # Test how quickly errors are processed
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch.object(provider.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = Exception("Test error")

            start_time = time.time()

            error_count = 0
            for _ in range(10):
                try:
                    await provider.chat_completion(messages=sample_messages, model="gpt-3.5-turbo")
                except Exception:
                    error_count += 1

            end_time = time.time()
            duration = end_time - start_time

            assert error_count == 10  # All should raise errors

            # Error handling should be fast
            assert duration < 1.0

            print(f"10 error scenarios handled in {duration:.2f} seconds")
            print(f"Average error handling time: {duration / 10:.3f} seconds")

    def test_memory_usage_benchmark(self, provider):
        """Benchmark memory usage of provider operations."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform memory-intensive operations
        large_messages = [
            ChatMessage(role="user", content="x" * 10000, name=None)  # 10KB content
            for _ in range(100)
        ]

        # Prepare messages multiple times
        for _ in range(100):
            provider._prepare_messages(large_messages)

        # Calculate costs for many scenarios
        for _ in range(1000):
            for model in provider.supported_models:
                provider.calculate_cost(10000, 5000, model)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Memory usage increased by {memory_increase:.2f} MB")

        # Memory increase should be reasonable (less than 50MB for this test)
        assert memory_increase < 50
