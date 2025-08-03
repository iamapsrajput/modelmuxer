# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""
Basic performance tests for ModelMuxer.
These tests ensure the system meets performance requirements.
"""

import asyncio
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

        assert response.status_code == 200
        assert (end_time - start_time) < 0.1  # Should respond within 100ms

    def test_providers_endpoint_performance(self, client: TestClient) -> None:
        """Test providers endpoint response time."""
        start_time = time.time()
        response = client.get("/providers")
        end_time = time.time()

        assert response.status_code == 200
        assert (end_time - start_time) < 0.5  # Should respond within 500ms

    @pytest.mark.benchmark(group="routing")
    def test_routing_decision_performance(self, benchmark: Any) -> None:
        """Benchmark routing decision performance."""
        from app.routing.heuristic_router import HeuristicRouter

        router = HeuristicRouter()

        def routing_decision() -> dict[str, Any]:
            """Make a routing decision."""
            return router.analyze_prompt([{"role": "user", "content": "What is the capital of France?"}])

        result = benchmark(routing_decision)
        assert result is not None
        assert "analysis_method" in result

    @pytest.mark.benchmark(group="cache")
    def test_cache_performance(self, benchmark: Any) -> None:
        """Benchmark cache operations."""
        from app.cache.memory_cache import MemoryCache

        cache = MemoryCache()

        def cache_operations() -> None:
            """Perform cache operations."""
            cache.set("test_key", {"data": "test_value"}, ttl=300)
            cache.get("test_key")
            cache.delete("test_key")

        benchmark(cache_operations)

    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self) -> None:
        """Test performance under concurrent load."""
        from httpx import AsyncClient

        async with AsyncClient(app=app, base_url="http://test") as client:
            # Create multiple concurrent requests
            tasks = []
            for _ in range(10):
                task = client.get("/health")
                tasks.append(task)

            start_time = time.time()
            responses = await asyncio.gather(*tasks)
            end_time = time.time()

            # All requests should succeed
            assert all(response.status_code == 200 for response in responses)

            # Total time should be reasonable (not 10x single request time)
            assert (end_time - start_time) < 2.0

    def test_memory_usage_stability(self, client: TestClient) -> None:
        """Test that memory usage remains stable under load."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Make many requests
        for _ in range(100):
            response = client.get("/health")
            assert response.status_code == 200

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
