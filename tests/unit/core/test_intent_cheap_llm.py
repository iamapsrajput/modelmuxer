#!/usr/bin/env python3
# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.core.intent import _parse_llm_intent_response, classify_intent
from app.models import ChatMessage
from app.providers.base import ProviderResponse
from app.settings import settings


def test_parse_llm_intent_response_accepts_json_and_markdown():
    label, conf = _parse_llm_intent_response('{"label":"code_gen","confidence":0.91}')
    assert label == "code_gen"
    assert conf == 0.91

    label, conf = _parse_llm_intent_response(
        '```json\n{"label":"deep_reason","confidence":0.82}\n```'
    )
    assert label == "deep_reason"
    assert conf == 0.82


@pytest.mark.asyncio
async def test_cheap_llm_intent_used_when_configured(monkeypatch):
    monkeypatch.setattr(settings.features, "test_mode", False)
    monkeypatch.setattr(settings.router, "intent_classifier_enabled", True)
    monkeypatch.setattr(settings.router, "intent_model", "openai:gpt-4o-mini")
    monkeypatch.setattr(settings.router, "intent_low_confidence", 0.4)

    mock_adapter = Mock()
    mock_adapter.invoke = AsyncMock(
        return_value=ProviderResponse(
            output_text='{"label":"code_gen","confidence":0.95}',
            tokens_in=10,
            tokens_out=8,
            latency_ms=50,
        )
    )

    with patch(
        "app.providers.registry.get_provider_registry",
        return_value={"openai": mock_adapter},
    ):
        result = await classify_intent([ChatMessage(role="user", content="hello there")])

    assert result["label"] == "code_gen"
    assert result["method"] == "cheap_llm"
    assert result["confidence"] == 0.95


@pytest.mark.asyncio
async def test_cheap_llm_low_confidence_falls_back_to_heuristic(monkeypatch):
    monkeypatch.setattr(settings.features, "test_mode", False)
    monkeypatch.setattr(settings.router, "intent_classifier_enabled", True)
    monkeypatch.setattr(settings.router, "intent_model", "openai:gpt-4o-mini")
    monkeypatch.setattr(settings.router, "intent_low_confidence", 0.9)

    mock_adapter = Mock()
    mock_adapter.invoke = AsyncMock(
        return_value=ProviderResponse(
            output_text='{"label":"code_gen","confidence":0.5}',
            tokens_in=10,
            tokens_out=8,
            latency_ms=50,
        )
    )

    with patch(
        "app.providers.registry.get_provider_registry",
        return_value={"openai": mock_adapter},
    ):
        result = await classify_intent(
            [ChatMessage(role="user", content="Write a Python function to add numbers")]
        )

    assert result["method"] == "heuristic"
    assert result["label"] == "code_gen"


@pytest.mark.asyncio
async def test_heuristic_only_when_intent_model_unset(monkeypatch):
    monkeypatch.setattr(settings.features, "test_mode", False)
    monkeypatch.setattr(settings.router, "intent_classifier_enabled", True)
    monkeypatch.setattr(settings.router, "intent_model", None)

    result = await classify_intent(
        [ChatMessage(role="user", content="Write a Python function to add numbers")]
    )
    assert result["method"] == "heuristic"
    assert result["label"] == "code_gen"


@pytest.mark.asyncio
async def test_disabled_classifier_unchanged():
    settings.router.intent_classifier_enabled = False
    result = await classify_intent([ChatMessage(role="user", content="hello")])
    assert result["method"] == "disabled"
    assert result["label"] == "unknown"
