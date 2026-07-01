# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""Provider and model listing endpoints."""

from typing import Any

from fastapi import APIRouter, Depends

import app.main as app_main
from app.main import get_authenticated_user

router = APIRouter()


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


@router.get("/v1/models")
async def list_models(user_info: dict[str, Any] = Depends(get_authenticated_user)):
    """List all available models across providers."""

    models = []

    # Load pricing data directly from price table
    price_table = app_main.load_price_table(app_main.app_settings.pricing.price_table_path)

    for model_key, price in price_table.items():
        # Parse model key as "provider:model"
        if ":" in model_key:
            provider, model = model_key.split(":", 1)

            # Skip models with separator characters to prevent proxy-style model names
            if ":" in model or "/" in model:
                continue

            models.append(
                {
                    "id": f"{provider}/{model}",
                    "object": "model",
                    "provider": provider,
                    "model": model,
                    "pricing": {
                        "input": price.input_per_1k_usd,
                        "output": price.output_per_1k_usd,
                        "unit": "per_1k_tokens",
                    },
                }
            )

    return {"object": "list", "data": models}
