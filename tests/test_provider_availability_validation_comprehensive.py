# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
from unittest.mock import Mock, patch

import pytest

from app.core.costing import load_price_table
from app.core.exceptions import BudgetExceededError, NoProvidersAvailableError
from app.models import ChatMessage
from app.router import HeuristicRouter
from app.settings import settings


@pytest.mark.asyncio
@pytest.mark.provider_availability_comprehensive
async def test_router_queries_registry_and_respects_availability(
    mock_provider_registry, monkeypatch, deterministic_price_table
):
    def reg():
        return mock_provider_registry

    router = HeuristicRouter(provider_registry_fn=reg)
    # Normalize price table and preferences to deterministic entries
    router.price_table = load_price_table(deterministic_price_table)
    router.estimator.prices = router.price_table
    router.model_preferences = {
        "code": [("openai", "gpt-4o"), ("openai", "gpt-4o-mini")],
        "complex": [("openai", "gpt-4o"), ("openai", "gpt-4o-mini")],
        "simple": [("openai", "gpt-3.5-turbo"), ("openai", "gpt-4o-mini")],
        "general": [("openai", "gpt-3.5-turbo"), ("openai", "gpt-4o-mini")],
    }
    monkeypatch.setattr(settings.router_thresholds, "max_estimated_usd_per_request", 10.0)

    async def fake_classify_intent(_):
        return {"label": "simple", "confidence": 0.8, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    messages = [ChatMessage(role="user", content="What is 2+2?", name=None)]
    provider, model, _, _, _ = await router.select_model(messages, budget_constraint=1.0)
    assert provider in reg().keys()


@pytest.mark.asyncio
@pytest.mark.provider_availability_comprehensive
async def test_empty_registry_raises_no_providers(monkeypatch, deterministic_price_table):
    from app.core.costing import load_price_table

    def empty_reg():
        return {}

    router = HeuristicRouter(provider_registry_fn=empty_reg)
    router.price_table = load_price_table(deterministic_price_table)
    router.estimator.prices = router.price_table
    router.model_preferences = {}  # Clear preferences to force empty case
    monkeypatch.setattr(settings.router_thresholds, "max_estimated_usd_per_request", 10.0)

    async def fake_classify_intent(_):
        return {"label": "general", "confidence": 0.5, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    with pytest.raises(NoProvidersAvailableError):
        await router.select_model(
            [ChatMessage(role="user", content="hi", name=None)], budget_constraint=1.0
        )


@pytest.mark.asyncio
@pytest.mark.provider_availability_comprehensive
async def test_fallback_when_preferred_missing(monkeypatch, deterministic_price_table):
    from app.core.costing import load_price_table

    # Registry has only openai; enforce a task where first preference is anthropic
    def only_openai():
        class A: ...

        return {"openai": A()}

    router = HeuristicRouter(provider_registry_fn=only_openai)
    # Ensure priced preference but unavailable provider
    router.price_table = load_price_table(deterministic_price_table)
    router.estimator.prices = router.price_table
    router.model_preferences["code"] = [("anthropic", "claude-3-haiku-20240307")]

    async def fake_classify_intent(_):
        return {"label": "code", "confidence": 0.9, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    messages = [ChatMessage(role="user", content="class X: pass  # review", name=None)]
    with pytest.raises(BudgetExceededError) as ei:
        await router.select_model(
            messages, budget_constraint=0.00001
        )  # Very low budget to force error
    assert getattr(ei.value, "reason", None) is None  # Budget exceeded for preferred model


@pytest.mark.asyncio
@pytest.mark.provider_availability_comprehensive
async def test_fallback_metrics_recorded(monkeypatch, deterministic_price_table):
    from app.core.costing import load_price_table

    def only_openai():
        return {"openai": object()}

    router = HeuristicRouter(provider_registry_fn=only_openai)
    router.price_table = load_price_table(deterministic_price_table)
    router.estimator.prices = router.price_table

    async def fake_classify_intent(_):
        return {"label": "code", "confidence": 0.9, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    with patch("app.router.ROUTER_FALLBACKS") as mock_fallbacks:
        labeled = Mock()
        labeled.inc = Mock()
        mock_fallbacks.labels.return_value = labeled

        # Ensure preferred providers are not available and budget is sufficient
        router.model_preferences["code"] = [("anthropic", "claude-3-haiku-20240307")]
        with pytest.raises(BudgetExceededError) as ei:
            await router.select_model(
                [ChatMessage(role="user", content="class X: pass", name=None)],
                budget_constraint=0.00001,
            )

        assert labeled.inc.called
        assert getattr(ei.value, "reason", None) is None  # Budget exceeded for preferred model


@pytest.mark.asyncio
@pytest.mark.provider_availability_comprehensive
async def test_circuit_open_registry_skips_open_circuits(
    mock_provider_registry_circuit_open, monkeypatch, deterministic_price_table
):
    def reg():
        return mock_provider_registry_circuit_open

    router = HeuristicRouter(provider_registry_fn=reg)
    router.price_table = load_price_table(deterministic_price_table)
    router.estimator.prices = router.price_table

    async def fake_classify_intent(_):
        return {"label": "general", "confidence": 0.6, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)
    router.model_preferences["general"] = [
        ("anthropic", "claude-3-haiku-20240307"),
        ("openai", "gpt-4o-mini"),
    ]

    reg()["anthropic"].circuit_open = True
    reg()["openai"].circuit_open = False

    provider, model, _, _, _ = await router.select_model(
        [ChatMessage(role="user", content="Hello", name=None)], budget_constraint=1.0
    )
    assert provider == "openai"


@pytest.mark.asyncio
@pytest.mark.provider_availability_comprehensive
async def test_provider_registry_refresh_is_reflected(monkeypatch, deterministic_price_table):
    from types import SimpleNamespace

    current_registry = {"openai": SimpleNamespace(circuit_open=False)}

    def get_current_registry():
        return current_registry

    router = HeuristicRouter(provider_registry_fn=get_current_registry)
    router.price_table = load_price_table(deterministic_price_table)
    router.estimator.prices = router.price_table
    router.model_preferences = {
        "general": [("openai", "gpt-4o-mini"), ("anthropic", "claude-3-haiku-20240307")]
    }
    monkeypatch.setattr(settings.router_thresholds, "max_estimated_usd_per_request", 10.0)

    messages = [ChatMessage(role="user", content="Hello", name=None)]

    provider, _, _, _, _ = await router.select_model(messages)
    assert provider == "openai"

    current_registry.clear()
    current_registry["anthropic"] = SimpleNamespace(circuit_open=False)

    provider, _, _, _, _ = await router.select_model(messages)
    assert provider == "anthropic"


@pytest.mark.asyncio
@pytest.mark.provider_availability_comprehensive
async def test_no_affordable_available_fallback(monkeypatch, deterministic_price_table):
    from app.core.costing import load_price_table
    from types import SimpleNamespace

    # Registry with providers present but circuit open
    def registry_with_open_circuits():
        return {
            "openai": SimpleNamespace(circuit_open=True),
            "anthropic": SimpleNamespace(circuit_open=True),
        }

    router = HeuristicRouter(provider_registry_fn=registry_with_open_circuits)
    router.price_table = load_price_table(deterministic_price_table)
    router.estimator.prices = router.price_table
    router.model_preferences = {
        "general": [("openai", "gpt-4o-mini"), ("anthropic", "claude-3-haiku-20240307")]
    }
    monkeypatch.setattr(settings.router_thresholds, "max_estimated_usd_per_request", 10.0)

    async def fake_classify_intent(_):
        return {"label": "general", "confidence": 0.6, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    with patch("app.router.ROUTER_FALLBACKS") as mock_fallbacks:
        labeled = Mock()
        labeled.inc = Mock()
        mock_fallbacks.labels.return_value = labeled

        with pytest.raises(BudgetExceededError) as ei:
            await router.select_model(
                [ChatMessage(role="user", content="Hello", name=None)], budget_constraint=0.00001
            )

        assert labeled.inc.called
        assert getattr(ei.value, "reason", None) == "no_affordable_available"


@pytest.mark.asyncio
@pytest.mark.provider_availability_comprehensive
async def test_production_mode_missing_providers_raises(monkeypatch):
    from app.core.exceptions import RouterConfigurationError
    from app.settings import settings

    monkeypatch.setattr(settings.features, "mode", "production")

    with pytest.raises(RouterConfigurationError) as ei:
        HeuristicRouter(provider_registry_fn=dict)

    assert "not available" in str(ei.value)
