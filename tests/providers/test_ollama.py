# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from app.models import ChatMessage
from app.providers.ollama import OllamaAdapter
from app.providers.registry import build_registry
from app.router import HeuristicRouter
from app.settings import settings


@pytest.mark.asyncio
async def test_ollama_adapter_invoke_success(monkeypatch):
    adapter = OllamaAdapter(base_url="http://ollama.test")

    async def fake_post(url, json=None, headers=None):  # type: ignore[no-redef]
        class R:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {"content": "local reply"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 5},
                }

        return R()

    monkeypatch.setattr(adapter._client, "post", fake_post)
    result = await adapter.invoke(
        model="llama3.2",
        messages=[ChatMessage(role="user", content="hello")],
    )
    assert result.output_text == "local reply"
    assert result.tokens_in == 3
    assert result.tokens_out == 5
    assert result.error is None


@pytest.mark.asyncio
async def test_ollama_adapter_passes_tools(monkeypatch):
    adapter = OllamaAdapter(base_url="http://ollama.test")
    captured: dict = {}

    async def fake_post(url, json=None, headers=None):  # type: ignore[no-redef]
        captured["payload"] = json

        class R:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }

        return R()

    monkeypatch.setattr(adapter._client, "post", fake_post)
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    await adapter.invoke(
        model="llama3.2",
        messages=[ChatMessage(role="user", content="find docs")],
        tools=tools,
        tool_choice="auto",
    )
    assert captured["payload"]["tools"] == tools
    assert captured["payload"]["tool_choice"] == "auto"


def test_ollama_adapter_lists_models_from_tags(monkeypatch):
    adapter = OllamaAdapter(base_url="http://ollama.test")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            class FakeResponse:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"models": [{"name": "llama3.2:latest"}, {"name": "mistral:7b"}]}

            return FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)
    models = adapter.get_supported_models()
    assert models == ["llama3.2:latest", "mistral:7b"]


def test_registry_includes_ollama_when_base_url_set(monkeypatch):
    monkeypatch.setattr(settings.features, "provider_adapters_enabled", True)
    monkeypatch.setattr(settings.endpoints, "ollama_base_url", "http://localhost:11434")
    registry = build_registry()
    assert "ollama" in registry


def test_registry_omits_ollama_without_base_url(monkeypatch):
    monkeypatch.setattr(settings.features, "provider_adapters_enabled", True)
    monkeypatch.setattr(settings.endpoints, "ollama_base_url", None)
    registry = build_registry()
    assert "ollama" not in registry


@pytest.mark.asyncio
async def test_router_routes_explicit_ollama_model(deterministic_price_table, monkeypatch):
    monkeypatch.setattr(settings.pricing, "price_table_path", deterministic_price_table)
    ollama_adapter = Mock()
    ollama_adapter.get_supported_models.return_value = ["llama3.2"]
    ollama_adapter.circuit_open = False

    registry = {"ollama": ollama_adapter}

    router = HeuristicRouter(provider_registry_fn=lambda: registry)
    provider, model, reason, _, estimate = await router.select_model(
        messages=[ChatMessage(role="user", content="hi")],
        requested_model="ollama/llama3.2",
    )
    assert provider == "ollama"
    assert model == "llama3.2"
    assert "Explicit local model" in reason
    assert estimate["usd"] == 0.0


@pytest.mark.asyncio
async def test_router_prefer_local_selects_ollama(deterministic_price_table, monkeypatch):
    monkeypatch.setattr(settings.pricing, "price_table_path", deterministic_price_table)
    monkeypatch.setattr(settings.router, "prefer_local", True)
    ollama_adapter = Mock()
    ollama_adapter.get_supported_models.return_value = ["llama3.2:latest"]
    ollama_adapter.circuit_open = False

    registry = {
        "ollama": ollama_adapter,
        "openai": Mock(circuit_open=False),
    }

    router = HeuristicRouter(provider_registry_fn=lambda: registry)
    provider, model, reason, _, estimate = await router.select_model(
        messages=[ChatMessage(role="user", content="hello")],
    )
    assert provider == "ollama"
    assert model == "llama3.2:latest"
    assert "PREFER_LOCAL" in reason
    assert estimate["usd"] == 0.0


def test_v1_models_lists_ollama_models():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    registry = {
        "ollama": Mock(get_supported_models=lambda: ["llama3.2", "mistral:7b"]),
    }
    with patch("app.auth.auth.authenticate_request", return_value={"user_id": "test-user"}):
        with patch("app.providers.registry.get_provider_registry", return_value=registry):
            with patch("app.main.load_price_table", return_value={}):
                response = client.get("/v1/models", headers={"Authorization": "Bearer test-key"})

    assert response.status_code == 200
    ids = {m["id"] for m in response.json()["data"]}
    assert "ollama/llama3.2" in ids
    assert "ollama/mistral:7b" in ids

    ollama_model = next(m for m in response.json()["data"] if m["id"] == "ollama/llama3.2")
    assert ollama_model["owned_by"] == "ollama"
    assert ollama_model["pricing"]["input"] == 0.0
    assert ollama_model["pricing"]["output"] == 0.0
