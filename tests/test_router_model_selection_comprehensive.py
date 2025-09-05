from typing import Any
from unittest.mock import Mock, patch

import pytest

from app.core.costing import load_price_table
from app.core.exceptions import BudgetExceededError, NoProvidersAvailableError
from app.models import ChatMessage
from app.router import HeuristicRouter
from app.settings import settings


@pytest.mark.asyncio
@pytest.mark.router_comprehensive
async def test_code_detection_and_selection_code_block(direct_router, monkeypatch):
    async def fake_classify_intent(_):
        return {"label": "code", "confidence": 0.9, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)
    monkeypatch.setattr(settings.router, "code_detection_threshold", 0.2)

    messages = [
        ChatMessage(role="user", content="""```python\ndef foo(x):\n    return x * 2\n```"""),
        ChatMessage(role="user", content="Please review the function and suggest improvements."),
    ]

    provider, model, reasoning, intent_meta, estimate_meta = await direct_router.select_model(
        messages
    )

    assert intent_meta["label"] == "code"
    assert provider in {p for p, _ in direct_router.model_preferences["code"]}
    assert (provider, model) in direct_router.model_preferences["code"]
    assert estimate_meta["usd"] is not None and estimate_meta["usd"] >= 0
    assert "Code detected" in reasoning


@pytest.mark.asyncio
@pytest.mark.router_comprehensive
async def test_complex_detection_and_selection_keywords(direct_router, monkeypatch):
    async def fake_classify_intent(_):
        return {"label": "analysis", "confidence": 0.8, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    messages = [
        ChatMessage(
            role="user", content="Analyze the trade-offs and architecture patterns for this system."
        ),
    ]

    provider, model, _, intent_meta, estimate_meta = await direct_router.select_model(messages)

    assert intent_meta["label"] == "analysis"
    assert (provider, model) in direct_router.model_preferences["complex"]
    assert estimate_meta["tokens_in"] > 0 and estimate_meta["tokens_out"] > 0


@pytest.mark.asyncio
@pytest.mark.router_comprehensive
async def test_simple_query_selection(direct_router, monkeypatch):
    async def fake_classify_intent(_):
        return {"label": "simple", "confidence": 0.7, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    messages = [
        ChatMessage(role="user", content="What is the capital of France?"),
    ]

    with patch("app.telemetry.metrics.ROUTER_INTENT_TOTAL") as mock_intent:
        labeled = Mock()
        labeled.inc = Mock()
        mock_intent.labels.return_value = labeled

        provider, model, _, _, _ = await direct_router.select_model(messages)

        assert (provider, model) in direct_router.model_preferences["simple"]
        assert labeled.inc.called


@pytest.mark.asyncio
@pytest.mark.router_comprehensive
async def test_general_fallback_selection(direct_router, monkeypatch):
    async def fake_classify_intent(_):
        return {"label": "general", "confidence": 0.6, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    messages = [
        ChatMessage(role="user", content="Hello there!"),
    ]

    provider, model, _, _, _ = await direct_router.select_model(messages)

    assert (provider, model) in direct_router.model_preferences["general"]


@pytest.mark.asyncio
@pytest.mark.router_comprehensive
async def test_provider_registry_subset_availability(monkeypatch, deterministic_price_table):
    # Only OpenAI available
    def only_openai_registry():
        from types import SimpleNamespace

        return {"openai": SimpleNamespace(circuit_open=False)}

    router = HeuristicRouter(provider_registry_fn=only_openai_registry)
    # Normalize preferences and pricing to deterministic table
    router.price_table = load_price_table(deterministic_price_table)
    router.estimator.prices = router.price_table
    router.model_preferences = {
        "code": [("openai", "gpt-4o"), ("openai", "gpt-4o-mini")],
        "complex": [("openai", "gpt-4o"), ("openai", "gpt-4o-mini")],
        "simple": [("openai", "gpt-3.5-turbo"), ("openai", "gpt-4o-mini")],
        "general": [("openai", "gpt-3.5-turbo"), ("openai", "gpt-4o-mini")],
    }
    monkeypatch.setattr(settings.router_thresholds, "max_estimated_usd_per_request", 10.0)

    messages = [ChatMessage(role="user", content="```js\nfunction x(){}\n``` review this")]

    provider, model, _, _, _ = await router.select_model(messages, budget_constraint=1.0)

    assert provider == "openai"


@pytest.mark.asyncio
@pytest.mark.router_comprehensive
async def test_message_analysis_metadata_population(direct_router, monkeypatch):
    async def fake_classify_intent(_):
        return {"label": "mix", "confidence": 0.5, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    messages = [
        ChatMessage(
            role="user", content="Explain and then optimize this code: `def f(x): return x+1`"
        ),
        ChatMessage(role="user", content="Be comprehensive but concise."),
    ]

    provider, model, reasoning, intent_meta, estimate_meta = await direct_router.select_model(
        messages
    )

    assert intent_meta["label"] == "mix"
    assert isinstance(estimate_meta["eta_ms"], int)
    assert "Task type:" in reasoning


@pytest.mark.asyncio
@pytest.mark.router_comprehensive
async def test_edge_cases_empty_and_non_english(monkeypatch, deterministic_price_table):
    def empty_registry():
        return {"openai": object(), "anthropic": object()}

    router = HeuristicRouter(provider_registry_fn=empty_registry)
    router.price_table = load_price_table(deterministic_price_table)
    router.estimator.prices = router.price_table

    async def fake_classify_intent(_):
        return {"label": "unknown", "confidence": 0.0, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    # Empty message content
    messages_empty = [ChatMessage(role="user", content=" ")]
    provider, model, _, intent_meta, estimate_meta = await router.select_model(
        messages_empty, budget_constraint=1.0
    )
    assert (provider, model) in router.model_preferences["general"]
    assert estimate_meta["tokens_in"] >= settings.pricing.min_tokens_in_floor
    assert intent_meta["label"] == "unknown"

    # Non-English
    messages_non_english = [ChatMessage(role="user", content="¿Puedes explicar el algoritmo?")]
    provider2, model2, _, _, estimate_meta2 = await router.select_model(
        messages_non_english, budget_constraint=1.0
    )
    assert (provider2, model2) in router.model_preferences["general"]
    assert estimate_meta2["usd"] is None or estimate_meta2["usd"] >= 0


@pytest.mark.asyncio
@pytest.mark.router_comprehensive
async def test_production_mode_missing_pricing_raises(monkeypatch):
    from app.core.exceptions import RouterConfigurationError
    from app.settings import settings

    monkeypatch.setattr(settings.features, "mode", "production")

    with patch("app.router.load_price_table", return_value={}):
        with pytest.raises(RouterConfigurationError) as ei:
            HeuristicRouter()

        assert "Price table is empty in production mode" in str(ei.value)


@pytest.mark.asyncio
@pytest.mark.router_comprehensive
async def test_edge_case_very_long_input(monkeypatch, deterministic_price_table):
    def empty_registry():
        from types import SimpleNamespace

        return {"openai": SimpleNamespace(circuit_open=False)}

    router = HeuristicRouter(provider_registry_fn=empty_registry)
    router.price_table = load_price_table(deterministic_price_table)
    router.estimator.prices = router.price_table

    async def fake_classify_intent(_):
        return {"label": "complex", "confidence": 0.8, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    # Very long input (>10k chars)
    long_content = "a" * 15000
    messages = [ChatMessage(role="user", content=long_content)]

    provider, model, _, _, estimate_meta = await router.select_model(
        messages, budget_constraint=1.0
    )
    assert (provider, model) in router.model_preferences["general"]
    assert estimate_meta["tokens_in"] > 2500  # Rough check for high token count
    assert estimate_meta["usd"] is not None


@pytest.mark.asyncio
@pytest.mark.router_comprehensive
async def test_edge_case_special_characters(monkeypatch, deterministic_price_table):
    def empty_registry():
        from types import SimpleNamespace

        return {"openai": SimpleNamespace(circuit_open=False)}

    router = HeuristicRouter(provider_registry_fn=empty_registry)
    router.price_table = load_price_table(deterministic_price_table)
    router.estimator.prices = router.price_table

    async def fake_classify_intent(_):
        return {"label": "code", "confidence": 0.9, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    # Special characters heavy content
    special_content = (
        "🔥🚀💻📊🌟⭐⭐⭐[ ] { } ( ) ; : , . ! ? @ # $ % ^ & * + = - _ / \\ | < > \" ' ` ~ \n\t\r"
    )
    messages = [
        ChatMessage(
            role="user", content=f"Debug this code: ```js\nconsole.log('{special_content}');\n```"
        )
    ]

    provider, model, _, _, estimate_meta = await router.select_model(
        messages, budget_constraint=1.0
    )
    assert (provider, model) in router.model_preferences["code"]
    assert estimate_meta["tokens_in"] > 0
    assert estimate_meta["usd"] is not None
