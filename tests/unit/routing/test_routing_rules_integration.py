#!/usr/bin/env python3
# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
from __future__ import annotations

from unittest.mock import Mock

import pytest

from app.core.routing_rules import load_routing_rules_from_data
from app.models import ChatMessage
from app.router import HeuristicRouter
from app.settings import settings


@pytest.mark.asyncio
async def test_router_applies_declarative_rule_by_intent(direct_router, monkeypatch):
    monkeypatch.setattr(settings.router, "prefer_local", False)

    direct_router.routing_rules = load_routing_rules_from_data(
        {
            "rules": [
                {
                    "name": "code_cheap",
                    "intent": "code_gen",
                    "prefer": [
                        "openai:gpt-4o-mini",
                        "anthropic:claude-3-haiku-20240307",
                    ],
                }
            ]
        }
    )

    async def fake_classify_intent(_):
        return {
            "label": "code_gen",
            "confidence": 0.9,
            "signals": {},
            "method": "cheap_llm",
        }

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    messages = [ChatMessage(role="user", content="Implement quicksort in Python")]
    provider, model, _, intent_meta, _ = await direct_router.select_model(messages)

    assert intent_meta["routing_rule"] == "code_cheap"
    assert intent_meta["method"] == "cheap_llm"
    assert (provider, model) == ("openai", "gpt-4o-mini")


@pytest.mark.asyncio
async def test_explicit_requested_model_not_overridden_by_rules(direct_router, monkeypatch):
    monkeypatch.setattr(settings.router, "prefer_local", False)

    direct_router.routing_rules = load_routing_rules_from_data(
        {
            "rules": [
                {
                    "name": "code_cheap",
                    "intent": "code_gen",
                    "prefer": ["openai:gpt-4o-mini"],
                }
            ]
        }
    )

    async def fake_classify_intent(_):
        return {"label": "code_gen", "confidence": 0.9, "signals": {}, "method": "cheap_llm"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    messages = [ChatMessage(role="user", content="hello")]
    provider, model, reason, intent_meta, _ = await direct_router.select_model(
        messages,
        requested_model="anthropic/claude-3-haiku-20240307",
    )

    assert provider == "anthropic"
    assert model == "claude-3-haiku-20240307"
    assert "Explicit model requested" in reason
    assert intent_meta.get("routing_rule") is None


@pytest.mark.asyncio
async def test_prefer_local_still_wins_over_rules(deterministic_price_table, monkeypatch):
    monkeypatch.setattr(settings.pricing, "price_table_path", deterministic_price_table)
    monkeypatch.setattr(settings.router, "prefer_local", True)
    monkeypatch.setattr(settings.router, "local_default_model", "llama3.2")

    ollama_adapter = Mock()
    ollama_adapter.get_supported_models.return_value = ["llama3.2"]
    ollama_adapter.circuit_open = False

    registry = {
        "ollama": ollama_adapter,
        "openai": Mock(circuit_open=False),
    }

    router = HeuristicRouter(provider_registry_fn=lambda: registry)
    router.routing_rules = load_routing_rules_from_data(
        {
            "rules": [
                {
                    "name": "code_cheap",
                    "intent": "code_gen",
                    "prefer": ["openai:gpt-4o-mini"],
                }
            ]
        }
    )

    async def fake_classify_intent(_):
        return {"label": "code_gen", "confidence": 0.9, "signals": {}, "method": "heuristic"}

    monkeypatch.setattr("app.router.classify_intent", fake_classify_intent)

    provider, model, reason, _, _ = await router.select_model(
        [ChatMessage(role="user", content="Write code")]
    )

    assert provider == "ollama"
    assert model == "llama3.2"
    assert "PREFER_LOCAL" in reason
