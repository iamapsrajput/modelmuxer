# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""Provider and model listing endpoints."""

from typing import Any

from fastapi import APIRouter, Depends

import app.main as app_main
from app.main import get_authenticated_user

router = APIRouter()

# Stable created timestamp for OpenAI-compatible model objects
_MODEL_CREATED_TS = 1686935000

# Virtual routing aliases exposed to OpenWebUI
_ROUTING_ALIASES = (
    {"id": "router", "model": "router", "owned_by": "modelmuxer"},
    {"id": "auto", "model": "auto", "owned_by": "modelmuxer"},
)


@router.get("/providers")
@router.get("/v1/providers")
async def get_providers(user_info: dict[str, Any] = Depends(get_authenticated_user)):
    """Get available providers and their models."""
    provider_info = {}

    provider_registry = app_main.providers_registry.get_provider_registry()
    for name, provider in provider_registry.items():
        provider_info[name] = {
            "name": name,
            "models": provider.get_supported_models(),
            "status": "available",
        }

    return {"providers": provider_info}


def _default_pricing() -> dict[str, Any]:
    return {"input": 0.0, "output": 0.0, "unit": "per_1k_tokens"}


def _pricing_from_price(price: Any) -> dict[str, Any]:
    return {
        "input": price.input_per_1k_usd,
        "output": price.output_per_1k_usd,
        "unit": "per_1k_tokens",
    }


@router.get("/v1/models")
async def list_models(user_info: dict[str, Any] = Depends(get_authenticated_user)):
    """List reachable models from registered provider adapters (OpenAI-compatible shape)."""
    models: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    provider_registry = app_main.providers_registry.get_provider_registry()
    price_table = app_main.load_price_table(app_main.app_settings.pricing.price_table_path)

    for provider_name, adapter in provider_registry.items():
        get_models = getattr(adapter, "get_supported_models", None)
        if not callable(get_models):
            continue

        for model_name in get_models():
            model_id = f"{provider_name}/{model_name}"
            if model_id in seen_ids:
                continue
            seen_ids.add(model_id)

            price_key = f"{provider_name}:{model_name}"
            price = price_table.get(price_key)
            pricing = _pricing_from_price(price) if price else _default_pricing()

            models.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": _MODEL_CREATED_TS,
                    "owned_by": provider_name,
                    "provider": provider_name,
                    "model": model_name,
                    "pricing": pricing,
                }
            )

    for alias in _ROUTING_ALIASES:
        if alias["id"] in seen_ids:
            continue
        seen_ids.add(alias["id"])
        models.append(
            {
                "id": alias["id"],
                "object": "model",
                "created": _MODEL_CREATED_TS,
                "owned_by": alias["owned_by"],
                "provider": "modelmuxer",
                "model": alias["model"],
                "pricing": _default_pricing(),
            }
        )

    return {"object": "list", "data": models}
