# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Pytest helper utilities for working with ModelMuxer's testing infrastructure.

These helpers mock the HTTP request path at its two seams:
- the provider registry (``app.providers.registry.get_provider_registry``)
- the router (``HeuristicRouter`` constructor, since the chat handler
  re-instantiates the router per request outside production mode)
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, Mock, patch

from app.core.exceptions import BudgetExceededError
from app.providers.base import ProviderError, ProviderResponse


def make_provider_response(
    output_text: str = "Test response",
    tokens_in: int = 10,
    tokens_out: int = 20,
    latency_ms: int = 150,
    error: str | None = None,
) -> ProviderResponse:
    """Build a deterministic ProviderResponse for adapter mocking."""
    return ProviderResponse(
        output_text=output_text,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
        raw={},
        error=error,
    )


def _make_mock_registry() -> dict:
    """Registry with a single mock adapter exposing an async invoke()."""
    mock_adapter = Mock()
    mock_adapter.invoke = AsyncMock(return_value=make_provider_response())
    mock_adapter.get_supported_models = Mock(return_value=["gpt-3.5-turbo"])
    return {"openai": mock_adapter}


def _make_selecting_router() -> Mock:
    """Router mock that selects openai:gpt-3.5-turbo and succeeds on invoke."""
    mock_router = Mock()
    mock_router.select_model = AsyncMock(
        return_value=(
            "openai",  # provider_name
            "gpt-3.5-turbo",  # model_name
            "selected",  # routing_reason
            {"label": "simple", "confidence": 0.9, "signals": {}},  # intent_metadata
            {"usd": 0.01, "eta_ms": 500, "tokens_in": 10, "tokens_out": 20},  # estimate_metadata
        )
    )
    mock_router.invoke_via_adapter = AsyncMock(return_value=make_provider_response())
    mock_router.record_latency = Mock()
    return mock_router


@contextmanager
def _patched_request_path(mock_router: Mock, registry: dict):
    """Patch the registry and router constructor used by the chat handler."""
    with patch("app.providers.registry.get_provider_registry", return_value=registry):
        with patch("app.router.HeuristicRouter", return_value=mock_router):
            with patch("app.main.HeuristicRouter", return_value=mock_router):
                with patch("app.main.router", mock_router):
                    yield


@contextmanager
def mock_for_budget_exceeded_test():
    """
    Context manager that sets up proper mocking for budget exceeded tests.

    The registry contains a provider (so the request passes the availability
    check) and the router raises BudgetExceededError during selection.
    """
    mock_router = Mock()
    mock_router.select_model = AsyncMock(
        side_effect=BudgetExceededError(
            "Budget exceeded", limit=100.0, estimates=[("openai:gpt-3.5-turbo", 0.05)]
        )
    )

    with _patched_request_path(mock_router, _make_mock_registry()):
        yield mock_router


@contextmanager
def mock_for_provider_error_test():
    """
    Context manager that sets up proper mocking for provider error tests.

    The router selects a provider but the adapter invocation fails with
    ProviderError, which the handler maps to a 502 response.
    """
    registry = _make_mock_registry()
    mock_provider = registry["openai"]
    mock_provider.invoke = AsyncMock(side_effect=ProviderError("API Error"))

    mock_router = _make_selecting_router()
    mock_router.invoke_via_adapter = AsyncMock(side_effect=ProviderError("API Error"))

    with _patched_request_path(mock_router, registry):
        # Mock database logging to prevent OperationalError
        with patch("app.main.db.log_request", new=AsyncMock()):
            yield mock_router, mock_provider


@contextmanager
def mock_for_database_logging_test():
    """
    Context manager for database logging tests that need to verify log_request calls.

    Unlike other test helpers, this does NOT mock database calls since we need
    to verify that database logging actually happens. The advanced cost tracker
    is disabled so the handler falls back to db.log_request.
    """
    registry = _make_mock_registry()
    mock_provider = registry["openai"]
    mock_router = _make_selecting_router()

    with _patched_request_path(mock_router, registry):
        with patch("app.main.model_muxer.advanced_cost_tracker", None):
            yield mock_router, mock_provider


@contextmanager
def mock_for_successful_test():
    """
    Context manager that sets up mocking for successful completion tests.

    This creates a working provider and router setup.
    """
    registry = _make_mock_registry()
    mock_provider = registry["openai"]
    mock_router = _make_selecting_router()

    with _patched_request_path(mock_router, registry):
        with patch("app.main.db.log_request", new=AsyncMock()):
            yield mock_router, mock_provider


@contextmanager
def mock_for_provider_timeout_test():
    """
    Context manager for testing provider timeout scenarios.
    Sets up mocking to simulate asyncio.TimeoutError from providers.
    """
    registry = _make_mock_registry()
    mock_provider = registry["openai"]
    mock_provider.invoke = AsyncMock(side_effect=TimeoutError("Provider timeout"))

    mock_router = _make_selecting_router()
    mock_router.invoke_via_adapter = AsyncMock(side_effect=TimeoutError("Provider timeout"))

    with _patched_request_path(mock_router, registry):
        # Prevent database errors during timeout handling
        with patch("app.main.db.log_request", new=AsyncMock()):
            yield mock_router, mock_provider
