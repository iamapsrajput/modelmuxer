# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
import asyncio
from unittest.mock import Mock, patch

import pytest

from app.core.costing import load_price_table
from app.models import ChatMessage
from app.router import HeuristicRouter


@pytest.mark.asyncio
@pytest.mark.integration_comprehensive
async def test_end_to_end_flow_code_then_budget_then_provider(
    monkeypatch, deterministic_price_table, mock_provider_registry
):
    def reg():
        return mock_provider_registry

    router = HeuristicRouter(provider_registry_fn=reg)
    router.price_table = load_price_table(deterministic_price_table)
    router.estimator.prices = router.price_table

    async def fake_classify_intent(_):
        return {"label": "code", "confidence": 0.9, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    messages = [
        ChatMessage(
            role="user", content="""```python\nclass A: pass\n``` Please refactor and optimize."""
        ),
    ]

    provider, model, reasoning, intent_meta, estimate_meta = await router.select_model(
        messages, budget_constraint=1.0
    )

    assert provider in reg().keys()
    assert intent_meta["label"] == "code"
    assert estimate_meta["usd"] is not None
    assert "Task type:" in reasoning


@pytest.mark.asyncio
@pytest.mark.integration_comprehensive
async def test_multi_turn_conversation_with_fallbacks(monkeypatch, deterministic_price_table):
    from app.settings import settings

    # Only OpenAI available; ensure general conversation still routes
    def only_openai():
        from types import SimpleNamespace

        return {"openai": SimpleNamespace(circuit_open=False)}

    monkeypatch.setattr(settings.router_thresholds, "max_estimated_usd_per_request", 10.0)
    router = HeuristicRouter(provider_registry_fn=only_openai)
    router.price_table = load_price_table(deterministic_price_table)
    router.estimator.prices = router.price_table

    async def fake_classify_intent(_):
        return {"label": "general", "confidence": 0.6, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    messages = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi!"),
        ChatMessage(role="user", content="Can you summarize our chat?"),
    ]

    provider, model, _, _, _ = await router.select_model(messages, budget_constraint=1.0)
    assert provider == "openai"


@pytest.mark.asyncio
@pytest.mark.integration_comprehensive
async def test_concurrent_select_model_isolation(
    monkeypatch, mock_provider_registry, deterministic_price_table
):
    router = HeuristicRouter(provider_registry_fn=lambda: mock_provider_registry)
    router.price_table = load_price_table(deterministic_price_table)
    router.estimator.prices = router.price_table

    async def fake_classify_intent(_):
        return {"label": "simple", "confidence": 0.9, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)
    msgs = [[ChatMessage(role="user", content=f"Hi {i}")] for i in range(10)]
    results = await asyncio.gather(*[router.select_model(m, budget_constraint=1.0) for m in msgs])
    assert len(results) == 10
