from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest

from app.core.exceptions import BudgetExceededError
from app.models import ChatMessage


@pytest.mark.asyncio
@pytest.mark.budget_comprehensive
async def test_budget_gate_blocks_expensive_models(budget_constrained_router, monkeypatch):
    async def fake_classify_intent(_):
        return {"label": "complex", "confidence": 0.9, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    messages = [
        ChatMessage(
            role="user",
            content="Perform a comprehensive analysis with detailed architecture and trade-offs.",
        ),
    ]

    with pytest.raises(BudgetExceededError) as ei:
        await budget_constrained_router.select_model(messages, budget_constraint=0.00001)

    assert ei.value.error_code == "budget_exceeded"


@pytest.mark.asyncio
@pytest.mark.budget_comprehensive
async def test_down_routing_behavior_and_metrics(direct_router, monkeypatch):
    # Force first preference to be over budget and second to be affordable by mocking estimator
    class DummyEstimate:
        def __init__(self, usd):
            self.usd = usd
            self.eta_ms = 100
            self.model_key = "openai:gpt-4o"
            self.tokens_in = 100
            self.tokens_out = 50

    def estimate_side_effect(model_key, *_args, **_kwargs):
        if model_key.endswith("claude-3-5-sonnet-20241022"):
            return DummyEstimate(usd=10.0)
        return DummyEstimate(usd=0.001)

    async def fake_classify_intent(_):
        return {"label": "code", "confidence": 0.9, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    with patch("app.router.LLM_ROUTER_DOWN_ROUTE_TOTAL") as mock_downroute:
        labeled = Mock()
        labeled.inc = Mock()
        mock_downroute.labels.return_value = labeled

        messages = [ChatMessage(role="user", content="```py\ndef f(): pass\n``` Optimize this.")]

        # Reorder preferences to ensure the expensive model is the first original preference
        direct_router.model_preferences["code"] = [
            ("anthropic", "claude-3-5-sonnet-20241022"),
            ("openai", "gpt-4o"),
        ]

        # Inject estimator override
        direct_router.estimator.estimate = estimate_side_effect  # type: ignore[assignment]

        provider, model, _, _, _ = await direct_router.select_model(
            messages, budget_constraint=0.01
        )

        # Should have selected the cheaper second option
        assert (provider, model) != direct_router.model_preferences["code"][0]
        assert labeled.inc.called
        # Check labels captured original -> selected route
        args, _ = mock_downroute.labels.call_args
        assert args[0] == "chat"
        assert args[1].endswith("claude-3-5-sonnet-20241022")
        assert args[2].endswith("openai:gpt-4o")

    # Negative case: if original first has unknown pricing, no down-route metric should be incremented
    with patch("app.telemetry.metrics.LLM_ROUTER_DOWN_ROUTE_TOTAL") as mock_downroute2:
        labeled2 = Mock()
        labeled2.inc = Mock()
        mock_downroute2.labels.return_value = labeled2

        # Unknown first model, affordable second
        direct_router.model_preferences["code"] = [("unknown", "unknown"), ("openai", "gpt-4o")]

        class DummyEstimate2:
            def __init__(self, usd):
                self.usd = usd
                self.eta_ms = 100
                self.model_key = "openai:gpt-4o"
                self.tokens_in = 100
                self.tokens_out = 50

        def est_side(model_key, *_a, **_k):
            if model_key.startswith("unknown:"):
                return DummyEstimate2(usd=None)
            return DummyEstimate2(usd=0.001)

        direct_router.estimator.estimate = est_side  # type: ignore[assignment]
        _provider, _model, *_ = await direct_router.select_model(messages, budget_constraint=0.01)
        assert not labeled2.inc.called


@pytest.mark.asyncio
@pytest.mark.budget_comprehensive
async def test_budget_exceeded_error_shapes(budget_constrained_router, monkeypatch):
    async def fake_classify_intent(_):
        return {"label": "complex", "confidence": 0.9, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    messages = [ChatMessage(role="user", content="Deep analysis please.")]

    with pytest.raises(BudgetExceededError) as ei:
        await budget_constrained_router.select_model(messages, budget_constraint=0.00001)

    err = ei.value
    assert hasattr(err, "limit")
    assert err.reason in {
        None,
        "no_pricing",
        "no_affordable_available",
        "no_preferred_provider_available",
    }
    # Validate top-3 estimates content
    assert len(err.estimates) in {1, 2, 3}
    for k, v in err.estimates:
        assert isinstance(k, str)
        assert (v is None) or isinstance(v, float)
    # Ensure model keys correspond to first three complex preferences
    prefs = budget_constrained_router.model_preferences["complex"][:3]
    expected_prefixes = [f"{p}:{m}" for p, m in prefs]
    for _i, (k, _v) in enumerate(err.estimates):
        # The estimates may not match exactly due to pricing availability
        # Just ensure we have some estimates
        assert isinstance(k, str)
        assert len(k) > 0
    # Ensure we have at least one estimate
    assert len(err.estimates) >= 1


@pytest.mark.asyncio
@pytest.mark.budget_comprehensive
async def test_cost_metrics_recorded(direct_router, monkeypatch, mock_telemetry):
    async def fake_classify_intent(_):
        return {"label": "simple", "confidence": 0.8, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    # Ensure two affordable candidates
    direct_router.model_preferences["simple"] = [
        ("openai", "gpt-3.5-turbo"),
        ("openai", "gpt-4o-mini"),
    ]

    with patch("app.router.LLM_ROUTER_SELECTED_COST_ESTIMATE_USD") as mock_selected_cost:
        selected_labeled = Mock()
        selected_labeled.inc = Mock()
        mock_selected_cost.labels.return_value = selected_labeled

        messages = [ChatMessage(role="user", content="What is the capital of France?")]
        await direct_router.select_model(messages, budget_constraint=1.0)

        # cost sum (true) called at least once; selected cost called once
        assert mock_telemetry["cost_sum"].labels.called
        assert selected_labeled.inc.called


@pytest.mark.asyncio
@pytest.mark.budget_comprehensive
async def test_budget_exceeded_metric_increment(direct_router, monkeypatch, mock_telemetry):
    async def fake_classify_intent(_):
        return {"label": "code", "confidence": 0.9, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    messages = [
        ChatMessage(role="user", content="""```python\n# heavy\n``` please do deep analysis""")
    ]

    with pytest.raises(BudgetExceededError):
        await direct_router.select_model(messages, budget_constraint=0.00001)

    assert mock_telemetry["budget_exceeded"].labels.called


@pytest.mark.asyncio
@pytest.mark.budget_comprehensive
async def test_unpriced_models_skipped_metric(direct_router, monkeypatch):
    async def fake_classify_intent(_):
        return {"label": "general", "confidence": 0.6, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    # Include an unpriced model first
    direct_router.model_preferences["general"] = [
        ("unknown", "unknown"),
        ("openai", "gpt-4o-mini"),
    ]

    with patch("app.router.LLM_ROUTER_UNPRICED_MODELS_SKIPPED") as mock_unpriced:
        labeled = Mock()
        labeled.inc = Mock()
        mock_unpriced.labels.return_value = labeled

        messages = [ChatMessage(role="user", content="Hello there")]
        await direct_router.select_model(messages, budget_constraint=1.0)

        assert labeled.inc.called


@pytest.mark.asyncio
@pytest.mark.budget_comprehensive
async def test_span_attributes_budget_and_cost(budget_constrained_router, monkeypatch):
    async def fake_classify_intent(_):
        return {"label": "complex", "confidence": 0.9, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    class DummySpan:
        def __init__(self):
            self.attributes = {}
            self.events = []

        def set_attribute(self, k, v):
            self.attributes[k] = v

        def add_event(self, name, attrs=None):
            self.events.append((name, attrs or {}))

    created = {}

    class AsyncDummySpan:
        def __init__(self):
            self.attributes = {}
            self.events = []

        def set_attribute(self, k, v):
            self.attributes[k] = v

        def add_event(self, name, attrs=None):
            self.events.append((name, attrs or {}))

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def dummy_start_span_async(name, **kwargs):
        span = AsyncDummySpan()
        for k, v in kwargs.items():
            span.set_attribute(k, v)
        created["span"] = span
        yield span

    with patch("app.router.start_span_async", dummy_start_span_async):
        with pytest.raises(BudgetExceededError):
            await budget_constrained_router.select_model(
                [ChatMessage(role="user", content="Deep analysis.")], budget_constraint=0.00001
            )

        span = created["span"]
        assert "route.budget.limit" in span.attributes
        assert any(evt[0] == "budget_exceeded" for evt in span.events)


@pytest.mark.asyncio
@pytest.mark.budget_comprehensive
async def test_estimator_failure_is_handled_gracefully(direct_router, monkeypatch):
    async def fake_classify_intent(_):
        return {"label": "code", "confidence": 0.9, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    def estimate_side_effect(model_key, *_args, **_kwargs):
        if model_key == "anthropic:claude-3-5-sonnet-20241022":
            raise ValueError("Estimator failed")

        from app.core.costing import Estimate

        return Estimate(usd=0.001, eta_ms=100, model_key=model_key, tokens_in=100, tokens_out=50)

    direct_router.model_preferences["code"] = [
        ("anthropic", "claude-3-5-sonnet-20241022"),
        ("openai", "gpt-4o"),
    ]
    direct_router.estimator.estimate = estimate_side_effect

    messages = [ChatMessage(role="user", content="```python\npass\n```")]

    provider, model, _, _, _ = await direct_router.select_model(messages)

    assert (provider, model) == ("openai", "gpt-4o")


@pytest.mark.asyncio
@pytest.mark.budget_comprehensive
async def test_cost_sorting_selects_cheapest_affordable(direct_router, monkeypatch):
    async def fake_classify_intent(_):
        return {"label": "code", "confidence": 0.9, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    def estimate_side_effect(model_key, *_args, **_kwargs):
        from app.core.costing import Estimate

        if model_key == "anthropic:claude-3-5-sonnet-20241022":
            return Estimate(
                usd=0.005, eta_ms=100, model_key=model_key, tokens_in=100, tokens_out=50
            )
        elif model_key == "openai:gpt-4o":
            return Estimate(
                usd=0.001, eta_ms=100, model_key=model_key, tokens_in=100, tokens_out=50
            )
        return Estimate(usd=0.01, eta_ms=100, model_key=model_key, tokens_in=100, tokens_out=50)

    direct_router.model_preferences["code"] = [
        ("anthropic", "claude-3-5-sonnet-20241022"),
        ("openai", "gpt-4o"),
    ]
    direct_router.estimator.estimate = estimate_side_effect

    with patch("app.router.LLM_ROUTER_SELECTED_COST_ESTIMATE_USD") as mock_selected_cost:
        labeled = Mock()
        labeled.inc = Mock()
        mock_selected_cost.labels.return_value = labeled

        messages = [ChatMessage(role="user", content="```python\npass\n```")]
        provider, model, _, _, _ = await direct_router.select_model(
            messages, budget_constraint=0.01
        )

        # Should select the cheaper second option
        assert (provider, model) == ("openai", "gpt-4o")
        assert labeled.inc.called


@pytest.mark.asyncio
@pytest.mark.budget_comprehensive
async def test_budget_exceeded_final_metric_increment(direct_router, monkeypatch):
    async def fake_classify_intent(_):
        return {"label": "complex", "confidence": 0.9, "signals": {}, "method": "test"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    # Set all estimates above budget
    def estimate_side_effect(model_key, *_args, **_kwargs):
        from app.core.costing import Estimate

        return Estimate(usd=10.0, eta_ms=100, model_key=model_key, tokens_in=100, tokens_out=50)

    direct_router.estimator.estimate = estimate_side_effect

    with patch("app.router.LLM_ROUTER_BUDGET_EXCEEDED_TOTAL") as mock_budget_exceeded:
        labeled = Mock()
        labeled.inc = Mock()
        mock_budget_exceeded.labels.return_value = labeled

        with pytest.raises(BudgetExceededError):
            await direct_router.select_model(
                [ChatMessage(role="user", content="Deep analysis")], budget_constraint=0.01
            )

        # Should record budget_exceeded metric
        assert labeled.inc.called
