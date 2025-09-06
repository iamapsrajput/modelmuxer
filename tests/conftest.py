# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Comprehensive test fixtures for ModelMuxer test suite.

This module provides fixtures for:
- Feature flag management
- Price table configuration
- Mock provider registry
- Router instances
- Message fixtures for different task types
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from app.models import ChatMessage
from app.providers.base import ProviderResponse
from app.settings import settings


class MockProviderAdapter:
    """Mock provider adapter for testing."""

    def __init__(
        self, provider_name: str, success_rate: float = 1.0, error_type: str | None = None
    ):
        self.provider_name = provider_name
        self.success_rate = success_rate
        self.error_type = error_type
        self.circuit_open = False
        self.request_count = 0

    async def invoke(self, model: str, prompt: str, **kwargs) -> ProviderResponse:
        """Mock invoke method that returns deterministic responses."""
        self.request_count += 1

        if self.circuit_open:
            return ProviderResponse(
                output_text="",
                tokens_in=0,
                tokens_out=0,
                latency_ms=0,
                raw={},
                error="circuit_open",
            )

        # Guard against division by zero
        if self.success_rate <= 0:
            if self.error_type:
                return ProviderResponse(
                    output_text="",
                    tokens_in=len(prompt.split()),
                    tokens_out=0,
                    latency_ms=100,
                    raw={"error": self.error_type},
                    error=self.error_type,
                )
        else:
            # Simulate success/failure based on success_rate
            interval = max(1, int(1 / self.success_rate))
            if self.request_count % interval == 0 and self.error_type:
                return ProviderResponse(
                    output_text="",
                    tokens_in=len(prompt.split()),
                    tokens_out=0,
                    latency_ms=100,
                    raw={"error": self.error_type},
                    error=self.error_type,
                )

        # Successful response
        response_text = f"Mock response from {self.provider_name} for model {model}"
        return ProviderResponse(
            output_text=response_text,
            tokens_in=len(prompt.split()),
            tokens_out=len(response_text.split()),
            latency_ms=150 + (hash(model) % 100),  # Deterministic but varied latency
            raw={"provider": self.provider_name, "model": model, "response": response_text},
            error=None,
        )


@pytest.fixture
def direct_providers_only_mode(monkeypatch):
    """Enable direct providers only mode for testing (now always enabled)."""
    original_provider_adapters_enabled = settings.features.provider_adapters_enabled
    original_test_mode = settings.features.test_mode

    monkeypatch.setattr(settings.features, "provider_adapters_enabled", True)
    monkeypatch.setattr(settings.features, "test_mode", True)

    yield

    # Cleanup - restore original values
    monkeypatch.setattr(
        settings.features, "provider_adapters_enabled", original_provider_adapters_enabled
    )
    monkeypatch.setattr(settings.features, "test_mode", original_test_mode)


@pytest.fixture
def test_mode_enabled(monkeypatch):
    """Enable test mode to ensure fresh provider registry builds."""
    monkeypatch.setattr(settings.features, "test_mode", True)

    yield

    # Cleanup
    monkeypatch.setattr(settings.features, "test_mode", False)


@pytest.fixture
def deterministic_price_table(tmp_path):
    """Create a temporary price table with deterministic pricing for testing."""
    price_data = {
        "openai:gpt-4o": {"input_per_1k_usd": 0.005, "output_per_1k_usd": 0.015},
        "openai:gpt-4o-mini": {"input_per_1k_usd": 0.00015, "output_per_1k_usd": 0.0006},
        "openai:gpt-3.5-turbo": {"input_per_1k_usd": 0.0005, "output_per_1k_usd": 0.0015},
        "anthropic:claude-3-5-sonnet-20241022": {
            "input_per_1k_usd": 0.003,
            "output_per_1k_usd": 0.015,
        },
        "anthropic:claude-3-haiku-20240307": {
            "input_per_1k_usd": 0.00025,
            "output_per_1k_usd": 0.00125,
        },
        "anthropic:claude-3-opus-20240229": {"input_per_1k_usd": 0.015, "output_per_1k_usd": 0.075},
        "mistral:mistral-large-latest": {"input_per_1k_usd": 0.007, "output_per_1k_usd": 0.024},
        "mistral:mistral-medium-latest": {"input_per_1k_usd": 0.0027, "output_per_1k_usd": 0.0081},
        "mistral:mistral-small-latest": {"input_per_1k_usd": 0.002, "output_per_1k_usd": 0.006},
        "groq:llama3-70b-8192": {"input_per_1k_usd": 0.0001, "output_per_1k_usd": 0.0008},
        "groq:llama3-8b-8192": {"input_per_1k_usd": 0.00005, "output_per_1k_usd": 0.0004},
        "groq:mixtral-8x7b-32768": {"input_per_1k_usd": 0.00024, "output_per_1k_usd": 0.00144},
        "google:gemini-1.5-pro": {"input_per_1k_usd": 0.0035, "output_per_1k_usd": 0.0105},
        "google:gemini-1.5-flash": {"input_per_1k_usd": 0.000075, "output_per_1k_usd": 0.0003},
        "google:gemini-pro": {"input_per_1k_usd": 0.0005, "output_per_1k_usd": 0.0015},
        "cohere:command-r-plus": {"input_per_1k_usd": 0.003, "output_per_1k_usd": 0.015},
        "cohere:command-r": {"input_per_1k_usd": 0.0005, "output_per_1k_usd": 0.0025},
        "cohere:command": {"input_per_1k_usd": 0.00015, "output_per_1k_usd": 0.0006},
        "together:meta-llama/Llama-3.1-8B-Instruct": {
            "input_per_1k_usd": 0.0002,
            "output_per_1k_usd": 0.0002,
        },
        "together:meta-llama/Llama-3.1-70B-Instruct": {
            "input_per_1k_usd": 0.0009,
            "output_per_1k_usd": 0.0009,
        },
        "together:microsoft/DialoGPT-medium": {
            "input_per_1k_usd": 0.0001,
            "output_per_1k_usd": 0.0001,
        },
    }

    price_file = tmp_path / "test_price_table.json"
    with open(price_file, "w") as f:
        json.dump(price_data, f)

    return str(price_file)


@pytest.fixture
def mock_provider_registry():
    """Create a mock provider registry with fake adapter instances."""
    registry = {
        "openai": MockProviderAdapter("openai", success_rate=1.0),
        "anthropic": MockProviderAdapter("anthropic", success_rate=1.0),
        "mistral": MockProviderAdapter("mistral", success_rate=1.0),
        "groq": MockProviderAdapter("groq", success_rate=1.0),
        "google": MockProviderAdapter("google", success_rate=1.0),
        "cohere": MockProviderAdapter("cohere", success_rate=1.0),
        "together": MockProviderAdapter("together", success_rate=1.0),
    }

    return registry


@pytest.fixture
def mock_provider_registry_with_failures():
    """Create a mock provider registry with some failing providers."""
    registry = {
        "openai": MockProviderAdapter("openai", success_rate=0.5, error_type="rate_limit"),
        "anthropic": MockProviderAdapter("anthropic", success_rate=1.0),
        "mistral": MockProviderAdapter("mistral", success_rate=0.3, error_type="timeout"),
        "groq": MockProviderAdapter("groq", success_rate=1.0),
        "google": MockProviderAdapter("google", success_rate=1.0),
        "cohere": MockProviderAdapter("cohere", success_rate=1.0),
        "together": MockProviderAdapter("together", success_rate=1.0),
    }

    return registry


@pytest.fixture
def mock_provider_registry_circuit_open():
    """Create a mock provider registry with circuit breakers open."""
    registry = {
        "openai": MockProviderAdapter("openai", success_rate=1.0),
        "anthropic": MockProviderAdapter("anthropic", success_rate=1.0),
        "mistral": MockProviderAdapter("mistral", success_rate=1.0),
        "groq": MockProviderAdapter("groq", success_rate=1.0),
        "google": MockProviderAdapter("google", success_rate=1.0),
        "cohere": MockProviderAdapter("cohere", success_rate=1.0),
        "together": MockProviderAdapter("together", success_rate=1.0),
    }

    # Open circuit breakers for some providers
    registry["openai"].circuit_open = True
    registry["anthropic"].circuit_open = True

    return registry


@pytest.fixture
def direct_router(deterministic_price_table, mock_provider_registry, monkeypatch):
    """Create a HeuristicRouter instance with mocked dependencies."""
    from app.core.costing import load_price_table
    from app.router import HeuristicRouter

    # Set budget threshold to allow most models to pass
    monkeypatch.setattr(settings.router_thresholds, "max_estimated_usd_per_request", 10.0)

    # Enable provider adapters and test mode before creating router
    monkeypatch.setattr(settings.features, "provider_adapters_enabled", True)
    monkeypatch.setattr(settings.features, "test_mode", True)

    # Create router with mocked provider registry
    def mock_provider_registry_fn():
        return mock_provider_registry

    # Create a custom router that uses the test price table
    router = HeuristicRouter(provider_registry_fn=mock_provider_registry_fn)

    # Override the price table with the test data
    router.price_table = load_price_table(deterministic_price_table)
    router.estimator.prices = router.price_table

    # Override the model preferences after router initialization to ensure determinism in tests
    router.model_preferences = {
        "code": [
            ("anthropic", "claude-3-5-sonnet-20241022"),
            ("openai", "gpt-4o"),
            ("openai", "gpt-4o-mini"),
        ],
        "complex": [
            ("anthropic", "claude-3-5-sonnet-20241022"),
            ("openai", "gpt-4o"),
            ("openai", "gpt-4o-mini"),
        ],
        "simple": [
            ("openai", "gpt-3.5-turbo"),
            ("openai", "gpt-4o-mini"),
            ("anthropic", "claude-3-haiku-20240307"),
        ],
        "general": [
            ("openai", "gpt-3.5-turbo"),
            ("openai", "gpt-4o-mini"),
            ("anthropic", "claude-3-haiku-20240307"),
        ],
    }

    return router


@pytest.fixture
def budget_constrained_router(deterministic_price_table, mock_provider_registry, monkeypatch):
    """Create a router with very low budget thresholds for testing budget gates."""
    from app.core.costing import load_price_table
    from app.router import HeuristicRouter

    # Set budget threshold to a very low value for testing budget constraints
    monkeypatch.setattr(settings.router_thresholds, "max_estimated_usd_per_request", 0.02)

    # Enable provider adapters and test mode before creating router
    monkeypatch.setattr(settings.features, "provider_adapters_enabled", True)
    monkeypatch.setattr(settings.features, "test_mode", True)

    # Create router with mocked provider registry - fix the closure issue
    def mock_provider_registry_fn():
        return dict(mock_provider_registry)  # Ensure we return a dict, not the fixture

    # Create a custom router that uses the test price table
    router = HeuristicRouter(provider_registry_fn=mock_provider_registry_fn)

    # Override the price table with the test data
    router.price_table = load_price_table(deterministic_price_table)
    router.estimator.prices = router.price_table

    # Override the model preferences after router initialization to ensure determinism in tests
    router.model_preferences = {
        "code": [
            ("anthropic", "claude-3-5-sonnet-20241022"),
            ("openai", "gpt-4o"),
            ("openai", "gpt-4o-mini"),
        ],
        "complex": [
            ("anthropic", "claude-3-5-sonnet-20241022"),
            ("openai", "gpt-4o"),
            ("openai", "gpt-4o-mini"),
        ],
        "simple": [
            ("openai", "gpt-3.5-turbo"),
            ("openai", "gpt-4o-mini"),
            ("anthropic", "claude-3-haiku-20240307"),
        ],
        "general": [
            ("openai", "gpt-3.5-turbo"),
            ("openai", "gpt-4o-mini"),
            ("anthropic", "claude-3-haiku-20240307"),
        ],
    }

    return router


@pytest.fixture
def code_messages():
    """Sample messages for code-related tasks."""
    return [
        ChatMessage(role="user", content="Write a Python function to sort a list of integers", name=None),
        ChatMessage(role="assistant", content="Here's a Python function to sort integers:", name=None),
        ChatMessage(role="user", content="Can you also add error handling for edge cases?", name=None),
    ]


@pytest.fixture
def complex_messages():
    """Sample messages for complex reasoning tasks."""
    return [
        ChatMessage(
            role="user", content="Explain the implications of quantum computing on cryptography", name=None
        ),
        ChatMessage(
            role="assistant",
            content="Quantum computing poses significant challenges to current cryptographic systems...", name=None
        ),
        ChatMessage(role="user", content="What are the potential solutions and their trade-offs?", name=None),
    ]


@pytest.fixture
def simple_messages():
    """Sample messages for simple tasks."""
    return [
        ChatMessage(role="user", content="What is the capital of France?", name=None),
        ChatMessage(role="assistant", content="The capital of France is Paris.", name=None),
    ]


@pytest.fixture
def general_messages():
    """Sample messages for general conversation."""
    return [
        ChatMessage(role="user", content="Hello, how are you today?", name=None),
        ChatMessage(role="assistant", content="I'm doing well, thank you for asking!", name=None),
        ChatMessage(role="user", content="Can you help me with a question?", name=None),
    ]


@pytest.fixture
def expensive_model_messages():
    """Sample messages that would trigger expensive model selection."""
    return [
        ChatMessage(
            role="user",
            content="I need a comprehensive analysis of the entire codebase with detailed architectural recommendations, performance optimization suggestions, security audit findings, and migration strategies for a large-scale enterprise application with microservices architecture, distributed systems, and real-time data processing requirements.", name=None
        ),
    ]


@pytest.fixture
def cheap_model_messages():
    """Sample messages that would trigger cheap model selection."""
    return [
        ChatMessage(role="user", content="Hi", name=None),
    ]


@pytest.fixture
def mock_estimator():
    """Create a mock cost estimator for testing."""
    mock_estimator = Mock()
    mock_estimator.estimate.return_value = {
        "cost_usd": 0.01,
        "tokens_in": 100,
        "tokens_out": 50,
        "model": "test-model",
        "provider": "test-provider",
    }
    return mock_estimator


@pytest.fixture
def mock_latency_priors():
    """Create mock latency priors for testing."""
    mock_priors = Mock()
    mock_priors.get_prior.return_value = 1000.0  # 1 second
    mock_priors.update_prior = Mock()
    return mock_priors


@pytest.fixture
def mock_telemetry():
    """Mock telemetry components for testing."""
    with (
        patch("app.router.ROUTER_REQUESTS") as mock_requests,
        patch("app.router.ROUTER_DECISION_LATENCY") as mock_latency,
        patch("app.router.ROUTER_FALLBACKS") as mock_fallbacks,
        patch("app.router.LLM_ROUTER_COST_ESTIMATE_USD_SUM") as mock_cost_sum,
        patch("app.router.LLM_ROUTER_BUDGET_EXCEEDED_TOTAL") as mock_budget_exceeded,
    ):
        # Set up labeled mocks for each metric
        mock_requests_labeled = Mock(inc=Mock())
        mock_requests.labels.return_value = mock_requests_labeled

        mock_latency_labeled = Mock(observe=Mock())
        mock_latency.labels.return_value = mock_latency_labeled

        mock_fallbacks_labeled = Mock(inc=Mock())
        mock_fallbacks.labels.return_value = mock_fallbacks_labeled

        mock_cost_sum_labeled = Mock(inc=Mock())
        mock_cost_sum.labels.return_value = mock_cost_sum_labeled

        mock_budget_exceeded_labeled = Mock(inc=Mock())
        mock_budget_exceeded.labels.return_value = mock_budget_exceeded_labeled

        yield {
            "requests": mock_requests,
            "latency": mock_latency,
            "fallbacks": mock_fallbacks,
            "cost_sum": mock_cost_sum,
            "budget_exceeded": mock_budget_exceeded,
            "requests_labeled": mock_requests_labeled,
            "latency_labeled": mock_latency_labeled,
            "fallbacks_labeled": mock_fallbacks_labeled,
            "cost_sum_labeled": mock_cost_sum_labeled,
            "budget_exceeded_labeled": mock_budget_exceeded_labeled,
        }


@pytest.fixture
def content():
    """Provide default content text for streaming tests."""
    return "Hello from test suite to validate streaming response behavior."


@pytest.fixture
def deterministic_env(monkeypatch):
    """Provide a deterministic environment for comprehensive tests."""
    monkeypatch.setenv("PROVIDER_ADAPTERS_ENABLED", "1")
    # Price table
    from app.core.costing import Price

    with patch(
        "app.router.load_price_table",
        return_value={
            "openai:gpt-4o": Price(input_per_1k_usd=0.0005, output_per_1k_usd=0.0015),
            "openai:gpt-4o-mini": Price(input_per_1k_usd=0.00015, output_per_1k_usd=0.0006),
            "anthropic:claude-3-haiku-20240307": Price(
                input_per_1k_usd=0.00025, output_per_1k_usd=0.00125
            ),
        },
    ):
        yield


@pytest.fixture
def mock_settings_max_tokens():
    """Mock settings.max_tokens_default for provider tests."""
    with patch("app.providers.anthropic_provider.settings") as mock_settings:
        mock_settings.max_tokens_default = 1000
        yield


@pytest.fixture
def mock_provider_registry_patch(mock_provider_registry):
    """Patch the global provider registry function to return mock providers."""
    with patch("app.providers.registry.get_provider_registry", return_value=mock_provider_registry):
        yield
