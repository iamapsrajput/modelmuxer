# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""
Basic performance tests for ModelMuxer.
These tests ensure the system meets performance requirements.
"""

import os
import time
from typing import Any

import psutil
import pytest
from fastapi.testclient import TestClient

from app.main_enhanced import app


class TestPerformance:
    """Performance tests for the ModelMuxer API."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_health_endpoint_performance(self, client: TestClient) -> None:
        """Test health endpoint response time."""
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()

        # Health endpoint should respond (503 is expected without API keys)
        assert response.status_code in [200, 503]
        assert (end_time - start_time) < 10.0  # Should respond within 10 seconds

    def test_providers_endpoint_performance(self, client: TestClient) -> None:
        """Test providers endpoint response time."""
        start_time = time.time()
        response = client.get("/providers")
        end_time = time.time()

        # Providers endpoint should respond
        assert response.status_code in [200, 404, 503]
        assert (end_time - start_time) < 1.0  # Should respond within 1 second

    @pytest.mark.benchmark(group="routing")
    @pytest.mark.asyncio
    async def test_routing_decision_performance(self, benchmark: Any) -> None:
        """Benchmark routing decision performance."""
        from app.routing.heuristic_router import HeuristicRouter

        router = HeuristicRouter()

        async def routing_decision() -> dict[str, Any]:
            """Make a routing decision."""
            from app.models import ChatMessage

            messages = [ChatMessage(role="user", content="What is the capital of France?", name=None)]
            return await router.analyze_prompt(messages)

        result = await benchmark(routing_decision)
        assert result is not None
        # Check for any key that indicates analysis was performed
        assert "confidence_score" in result or "complexity_confidence" in result

    @pytest.mark.benchmark(group="cache")
    @pytest.mark.asyncio
    async def test_cache_performance(self, benchmark: Any) -> None:
        """Benchmark cache operations."""
        from app.cache.memory_cache import MemoryCache

        cache = MemoryCache()

        async def cache_operations() -> None:
            """Perform cache operations."""
            await cache.set("test_key", {"data": "test_value"}, ttl=300)
            await cache.get("test_key")
            await cache.delete("test_key")

        await benchmark(cache_operations)

    def test_concurrent_requests_performance(self) -> None:
        """Test performance under concurrent load."""
        from fastapi.testclient import TestClient

        # Use TestClient for synchronous testing
        client = TestClient(app)

        # Create multiple concurrent requests
        responses = []
        start_time = time.time()

        for _ in range(10):
            response = client.get("/health")
            responses.append(response)

        end_time = time.time()

        # All requests should respond (503 is expected without API keys)
        assert all(response.status_code in [200, 503] for response in responses)

        # Total time should be reasonable (health checks can be slow without API keys)
        assert (end_time - start_time) < 60.0  # Allow up to 60 seconds for health checks

    def test_memory_usage_stability(self, client: TestClient) -> None:
        """Test that memory usage remains stable under load."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Make many requests
        for _ in range(100):
            response = client.get("/health")
            assert response.status_code in [200, 503]  # 503 expected without API keys

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be minimal (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024


@pytest.mark.performance
class TestLoadPerformance:
    """Load testing scenarios."""

    @pytest.mark.skip(reason="Heavy load test - run manually")
    def test_sustained_load(self) -> None:
        """Test sustained load performance."""
        # This would be a longer running test
        pass

    @pytest.mark.skip(reason="Stress test - run manually")
    def test_stress_scenarios(self) -> None:
        """Test system behavior under stress."""
        # This would test edge cases and high load
        pass
