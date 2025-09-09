# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
import json
from pathlib import Path

import pytest

from app.core.costing import (Estimator, LatencyPriors, Price, estimate_tokens,
                              load_price_table)
from app.models import ChatMessage
from app.router import HeuristicRouter
from app.settings import settings


@pytest.mark.cost_estimation_comprehensive
def test_price_table_integration_with_router_preferences(
    deterministic_price_table, mock_provider_registry, monkeypatch
):
    # Build router with a subset of preferences guaranteed to exist in deterministic price table
    def registry():
        return mock_provider_registry

    router = HeuristicRouter(provider_registry_fn=registry)
    prices = load_price_table(deterministic_price_table)

    subset_prefs = {
        "code": [("openai", "gpt-4o"), ("openai", "gpt-4o-mini")],
        "complex": [("anthropic", "claude-3-5-sonnet-20241022"), ("openai", "gpt-4o")],
        "simple": [("openai", "gpt-3.5-turbo"), ("openai", "gpt-4o-mini")],
        "general": [("openai", "gpt-3.5-turbo"), ("anthropic", "claude-3-haiku-20240307")],
    }
    router.model_preferences = subset_prefs

    # Verify all preferences exist in price table
    for prefs in subset_prefs.values():
        for provider, model in prefs:
            key = f"{provider}:{model}"
            assert key in prices


@pytest.mark.cost_estimation_comprehensive
def test_estimate_tokens_various_lengths():
    # Short simple
    messages = [ChatMessage(role="user", content="Hi", name=None)]
    tokens_in, tokens_out = estimate_tokens(messages, defaults=settings, floor=1)
    assert tokens_in >= 1 and tokens_out > 0

    # Long content
    long_text = "a" * 1000
    messages = [ChatMessage(role="user", content=long_text, name=None)]
    tokens_in, _ = estimate_tokens(messages, defaults=settings, floor=1)
    assert tokens_in >= 250  # 1000 chars ~ 250 tokens


@pytest.mark.cost_estimation_comprehensive
def test_estimator_returns_estimate_fields(deterministic_price_table):
    prices = load_price_table(deterministic_price_table)
    estimator = Estimator(prices, LatencyPriors(), settings)
    est = estimator.estimate("openai:gpt-4o", tokens_in=100, tokens_out=50)
    assert est.model_key == "openai:gpt-4o"
    assert isinstance(est.eta_ms, int)
    assert est.tokens_in == 100 and est.tokens_out == 50
    assert est.usd is not None and est.usd > 0


@pytest.mark.cost_estimation_comprehensive
def test_cost_calculation_accuracy_known_pricing(deterministic_price_table):
    prices = load_price_table(deterministic_price_table)
    estimator = Estimator(prices, LatencyPriors(), settings)

    # openai:gpt-4o has input=0.005, output=0.015 per 1k
    est = estimator.estimate("openai:gpt-4o", tokens_in=2000, tokens_out=1000)
    expected = (2000 / 1000) * 0.005 + (1000 / 1000) * 0.015
    assert abs(est.usd - expected) < 1e-9


@pytest.mark.cost_estimation_comprehensive
def test_error_handling_missing_price_entries(tmp_path):
    # Create empty/corrupted price file
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{ not-json }")

    prices = load_price_table(str(bad_file))
    assert prices == {}

    estimator = Estimator(prices, LatencyPriors(), settings)
    est = estimator.estimate("unknown:unknown", tokens_in=100, tokens_out=100)
    assert est.usd is None  # Unknown models return None cost


@pytest.mark.asyncio
@pytest.mark.cost_estimation_comprehensive
@pytest.mark.xfail(reason="depends on production pricing completeness")
async def test_all_default_router_preferences_have_pricing():
    from app.core.costing import load_price_table
    from app.router import HeuristicRouter

    # Load production price table
    prices = load_price_table("scripts/data/prices.json")

    # Create router with default preferences
    router = HeuristicRouter()

    # Verify all preferences exist in price table
    for task_type, prefs in router.model_preferences.items():
        for provider, model in prefs:
            key = f"{provider}:{model}"
            assert key in prices, f"Model {key} for {task_type} not found in price table"


@pytest.mark.asyncio
@pytest.mark.cost_estimation_comprehensive
async def test_max_tokens_override_affects_estimate(
    monkeypatch, deterministic_price_table, mock_provider_registry
):
    def reg():
        return mock_provider_registry

    router = HeuristicRouter(provider_registry_fn=reg)

    async def fake_classify_intent(_):
        return {"label": "simple", "confidence": 0.9, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    # ensure model exists in pricing
    router.model_preferences["simple"] = [("openai", "gpt-4o-mini")]

    messages = [ChatMessage(role="user", content="Short question", name=None)]
    provider, model, _, _, estimate_meta = await router.select_model(
        messages, max_tokens=10, budget_constraint=1.0
    )

    assert estimate_meta["tokens_out"] == 10
    assert estimate_meta["usd"] is not None and estimate_meta["usd"] > 0


@pytest.mark.cost_estimation_comprehensive
def test_latency_priors_influence_eta(deterministic_price_table):
    from app.core.costing import Estimator, LatencyPriors
    from app.settings import settings

    prices = load_price_table(deterministic_price_table)
    latency_priors = LatencyPriors()
    estimator = Estimator(prices, latency_priors, settings)
    model_key = "test:model"

    default_eta = estimator.estimate(model_key).eta_ms

    latency_priors.update(model_key, 2000)
    latency_priors.update(model_key, 2200)
    latency_priors.update(model_key, 2100)
    latency_priors.update(model_key, 5000)

    new_eta = estimator.estimate(model_key).eta_ms
    assert new_eta > default_eta

    for _ in range(20):
        latency_priors.update(model_key, 100)

    final_eta = estimator.estimate(model_key).eta_ms
    assert final_eta < new_eta


@pytest.mark.asyncio
@pytest.mark.cost_estimation_comprehensive
async def test_edge_case_very_long_input(direct_router, monkeypatch):
    async def fake_classify_intent(_):
        return {"label": "simple", "confidence": 0.9, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    # Very long input (>10k chars)
    long_content = "a" * 15000
    messages = [ChatMessage(role="user", content=long_content, name=None)]

    provider, model, _, _, estimate_meta = await direct_router.select_model(
        messages, budget_constraint=1.0
    )
    assert (provider, model) in direct_router.model_preferences["general"]
    assert estimate_meta["tokens_in"] > 2500  # Rough check for high token count
    assert estimate_meta["usd"] is not None
