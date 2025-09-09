# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Pytest helper utilities for working with ModelMuxer's testing infrastructure.

This module provides utilities to properly test specific scenarios
while working with the pytest short-circuit logic in main.py.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, Mock, patch

from app.core.exceptions import BudgetExceededError
from app.providers.base import ProviderError


@contextmanager
def mock_for_budget_exceeded_test():
    """
    Context manager that sets up proper mocking for budget exceeded tests.

    This works by:
    1. Providing an empty provider registry (bypasses pytest short-circuits)
    2. Mocking the HeuristicRouter to raise BudgetExceededError
    3. Ensuring the error handling path is tested
    """
    empty_registry = {}

    # Create a mock router that raises BudgetExceededError (async method)
    mock_router = Mock()
    mock_router.select_model = AsyncMock(
        side_effect=BudgetExceededError(
            "Budget exceeded", limit=100.0, estimates=[("openai:gpt-3.5-turbo", 0.05)]
        )
    )

    # Mock the HeuristicRouter constructor to return our mock
    with patch("app.providers.registry.get_provider_registry", return_value=empty_registry):
        with patch("app.router.HeuristicRouter", return_value=mock_router):
            with patch("app.main.HeuristicRouter", return_value=mock_router):
                yield mock_router


@contextmanager
def mock_for_provider_error_test():
    """
    Context manager that sets up proper mocking for provider error tests.

    Simplified approach: Use empty registry to force router path, then directly
    mock the provider access when it's retrieved by dictionary access.
    """
    # Use empty registry to force router path (same as budget exceeded test)
    empty_registry = {}

    # Create mock provider that throws error
    mock_provider = Mock()
    mock_provider.chat_completion = AsyncMock(side_effect=ProviderError("API Error"))

    # Create router that returns selection (async method) - needs 5 values
    mock_router = Mock()
    mock_router.select_model = AsyncMock(
        return_value=(
            "openai",  # provider_name
            "gpt-3.5-turbo",  # model_name
            "selected",  # routing_reason
            {"label": "simple", "confidence": 0.9},  # intent_metadata
            {"usd": 0.01, "eta_ms": 500, "tokens_in": 10, "tokens_out": 20},  # estimate_metadata
        )
    )
    # Router also needs invoke_via_adapter method that will fail with ProviderError
    mock_router.invoke_via_adapter = AsyncMock(side_effect=ProviderError("API Error"))

    # Mock the HeuristicRouter constructor (proven approach)
    with patch("app.providers.registry.get_provider_registry", return_value=empty_registry):
        with patch("app.router.HeuristicRouter", return_value=mock_router):
            with patch("app.main.HeuristicRouter", return_value=mock_router):
                # After router selection, when provider_registry[provider_name] is accessed,
                # we need that specific access to return the failing provider
                # Mock this specific access pattern by patching the registry module's get_provider_registry
                # to return a registry that has the provider when accessed after router selection
                registry_with_provider = {"openai": mock_provider}

                # Create a context where the second call to get_provider_registry returns the provider
                call_count = {"value": 0}

                def counting_get_provider_registry():
                    call_count["value"] += 1
                    if call_count["value"] == 1:
                        return empty_registry  # First call during initial check
                    else:
                        return registry_with_provider  # Subsequent calls return provider

                with patch(
                    "app.providers.registry.get_provider_registry",
                    side_effect=counting_get_provider_registry,
                ):
                    # Mock database logging to prevent OperationalError
                    with patch("app.main.db.log_request", new=AsyncMock()) as mock_log_request:
                        yield mock_router, mock_provider


@contextmanager
def mock_for_database_logging_test():
    """
    Context manager for database logging tests that need to verify log_request calls.

    Unlike other test helpers, this does NOT mock database calls since we need
    to verify that database logging actually happens.
    """
    from tests.fixtures.mocks.provider_mocks import \
        create_mock_provider_response

    # Create successful mock provider
    mock_provider = Mock()
    mock_response = create_mock_provider_response()
    mock_provider.chat_completion = AsyncMock(return_value=mock_response)
    mock_registry = {"openai": mock_provider}

    # Create router that returns selection
    mock_router = Mock()
    mock_router.select_model = AsyncMock(
        return_value=(
            "openai",
            "gpt-3.5-turbo",
            "selected",
            {"label": "simple", "confidence": 0.9},
            {"usd": 0.01, "eta_ms": 500, "tokens_in": 10, "tokens_out": 20},
        )
    )

    with patch("app.providers.registry.get_provider_registry", return_value=mock_registry):
        with patch("app.router.HeuristicRouter", return_value=mock_router):
            with patch("app.main.HeuristicRouter", return_value=mock_router):
                # NOTE: We do NOT mock db.log_request here since these tests verify it's called
                yield mock_router, mock_provider


@contextmanager
def mock_for_successful_test():
    """
    Context manager that sets up mocking for successful completion tests.

    This creates a working provider and router setup.
    """
    from tests.fixtures.mocks.provider_mocks import \
        create_mock_provider_response

    # Create successful mock provider
    mock_provider = Mock()
    mock_response = create_mock_provider_response()
    mock_provider.chat_completion = AsyncMock(return_value=mock_response)
    mock_registry = {"openai": mock_provider}

    # Create router that returns selection
    mock_router = Mock()
    mock_router.select_model.return_value = (
        "openai",
        "gpt-3.5-turbo",
        "selected",
        {"usd": 0.01, "eta_ms": 500, "tokens_in": 10, "tokens_out": 20},
    )

    with patch("app.providers.registry.get_provider_registry", return_value=mock_registry):
        with patch("app.main.router", mock_router):
            yield mock_router, mock_provider


@contextmanager
def mock_for_provider_timeout_test():
    """
    Context manager for testing provider timeout scenarios.
    Sets up mocking to simulate asyncio.TimeoutError from providers.
    """
    import asyncio

    # Create provider that raises TimeoutError
    mock_provider = Mock()
    mock_provider.chat_completion = AsyncMock(side_effect=TimeoutError("Provider timeout"))

    mock_registry = {"openai": mock_provider}

    # Create router that returns selection
    mock_router = Mock()
    mock_router.select_model = AsyncMock(
        return_value=(
            "openai",
            "gpt-3.5-turbo",
            "selected",
            {"label": "simple", "confidence": 0.9},
            {"usd": 0.01, "eta_ms": 500, "tokens_in": 10, "tokens_out": 20},
        )
    )

    with patch("app.providers.registry.get_provider_registry", return_value=mock_registry):
        with patch("app.router.HeuristicRouter", return_value=mock_router):
            with patch("app.main.HeuristicRouter", return_value=mock_router):
                # Prevent database errors during timeout handling
                with patch("app.main.db.log_request", new=AsyncMock()):
                    yield mock_router, mock_provider
