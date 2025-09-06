# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""
Simple performance tests for CI/CD pipeline.
These tests ensure basic performance without heavy dependencies.
"""

import time

import pytest


@pytest.mark.performance
class TestBasicPerformance:
    """Basic performance tests that run in CI."""

    def test_import_performance(self) -> None:
        """Test that imports happen quickly."""
        start_time = time.time()

        # Test critical imports
        from app.config.enhanced_config import enhanced_config  # noqa: F401
        from app.models import ChatMessage  # noqa: F401

        end_time = time.time()
        import_time = end_time - start_time

        # Imports should be fast
        assert import_time < 2.0, f"Imports took too long: {import_time:.2f}s"

    def test_config_loading_performance(self) -> None:
        """Test configuration loading performance."""
        start_time = time.time()

        try:
            from app.config.enhanced_config import enhanced_config

            # Access some config values to ensure they're loaded
            _ = enhanced_config.debug
            _ = enhanced_config.host
        except Exception:
            # Fallback if config doesn't load
            pass

        end_time = time.time()
        config_time = end_time - start_time

        # Config loading should be reasonably fast
        assert config_time < 2.0, f"Config loading took too long: {config_time:.2f}s"

    def test_model_validation_performance(self) -> None:
        """Test pydantic model validation performance."""
        from app.models import ChatMessage

        start_time = time.time()

        # Create many message objects to test validation performance
        for i in range(100):
            msg = ChatMessage(role="user", content=f"Test message {i}", name=None)
            assert msg.role == "user"

        end_time = time.time()
        validation_time = end_time - start_time

        # 100 model validations should be fast
        assert validation_time < 1.0, f"Model validation took too long: {validation_time:.2f}s"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_async_operations_performance(self) -> None:
        """Test basic async operations performance."""
        import asyncio

        async def simple_async_task() -> str:
            await asyncio.sleep(0.001)  # 1ms sleep
            return "done"

        start_time = time.time()

        # Run 10 concurrent simple tasks
        tasks = [simple_async_task() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        async_time = end_time - start_time

        assert len(results) == 10
        assert all(result == "done" for result in results)
        # Should not take much longer than the sleep time
        assert async_time < 0.1, f"Async operations took too long: {async_time:.2f}s"

    @pytest.mark.performance
    def test_benchmark_import_performance(self, benchmark) -> None:
        """Benchmark test for import performance using pytest-benchmark."""

        def import_modules():
            from app.config.enhanced_config import \
                enhanced_config  # noqa: F401
            from app.models import ChatMessage  # noqa: F401

            return True

        # Use benchmark fixture to measure performance
        result = benchmark(import_modules)
        assert result is True

    @pytest.mark.performance
    def test_benchmark_model_validation(self, benchmark) -> None:
        """Benchmark test for model validation using pytest-benchmark."""
        from app.models import ChatMessage

        def create_message():
            return ChatMessage(role="user", content="Test message", name=None)

        # Use benchmark fixture to measure performance
        result = benchmark(create_message)
        assert result.role == "user"


# Mark these as performance tests
pytestmark = pytest.mark.performance
