# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Centralized router mock configurations.

This module provides standardized router mocks and routing responses
to ensure consistent testing behavior.
"""

from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List, Tuple
from app.core.exceptions import BudgetExceededError, ProviderError


def create_mock_router(
    selected_provider: str = "openai",
    selected_model: str = "gpt-3.5-turbo",
    routing_reason: str = "test",
    cost_estimate: float = 0.01,
) -> Mock:
    """Create a standardized mock router."""
    router = Mock()

    # Mock select_model method
    router.select_model.return_value = (
        selected_provider,
        selected_model,
        routing_reason,
        {
            "usd": cost_estimate,
            "eta_ms": 100,
            "tokens_in": 10,
            "tokens_out": 5,
        },
    )

    # Mock other router methods
    router.record_latency = Mock()
    router.get_preferences = Mock(return_value=[(selected_provider, selected_model)])
    router.estimate_cost = Mock(return_value=cost_estimate)

    return router


def create_budget_exceeded_router(
    limit: float = 0.08,
    estimates: List[Tuple[str, float]] = None,
    reason: str = "budget_exceeded",
) -> Mock:
    """Create a router mock that raises BudgetExceededError."""
    if estimates is None:
        estimates = [("openai:gpt-3.5-turbo", 0.1)]

    router = Mock()
    router.select_model.side_effect = BudgetExceededError(
        message="Budget exceeded",
        limit=limit,
        estimates=estimates,
        reason=reason,
    )

    return router


def create_provider_error_router(
    provider: str = "openai",
    error_message: str = "Provider unavailable",
    status_code: int = 502,
) -> Mock:
    """Create a router mock that raises ProviderError."""
    router = Mock()
    router.select_model.side_effect = ProviderError(
        message=error_message,
        provider=provider,
        status_code=status_code,
    )

    return router


def create_no_providers_router() -> Mock:
    """Create a router mock that raises NoProvidersAvailableError."""
    from app.core.exceptions import NoProvidersAvailableError

    router = Mock()
    router.select_model.side_effect = NoProvidersAvailableError("No LLM providers available")

    return router


# Standard router configurations
MOCK_SUCCESSFUL_ROUTER = create_mock_router()
MOCK_BUDGET_EXCEEDED_ROUTER = create_budget_exceeded_router()
MOCK_PROVIDER_ERROR_ROUTER = create_provider_error_router()
MOCK_NO_PROVIDERS_ROUTER = create_no_providers_router()
