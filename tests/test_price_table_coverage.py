# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""Test price table coverage for router preferences."""

import pytest

from app.providers.registry import get_provider_registry
from app.router import HeuristicRouter


def test_router_preferences_in_price_table():
    """Test that all models in router preferences are included in the price table."""
    # Create router to get model preferences
    router = HeuristicRouter(provider_registry_fn=get_provider_registry)

    # Get all model keys from preferences
    candidate_keys = set()
    for plist in router.model_preferences.values():
        candidate_keys.update(f"{provider}:{model}" for provider, model in plist)

    # Check against price table
    price_table_keys = set(router.price_table.keys())
    missing_keys = candidate_keys - price_table_keys

    # Assert that all router preference models are in the price table
    assert (
        not missing_keys
    ), f"Router preference models missing from price table: {sorted(missing_keys)}"

    # Also verify that we have a reasonable number of models in preferences
    assert len(candidate_keys) > 0, "Router preferences should contain at least one model"
    assert len(price_table_keys) > 0, "Price table should contain at least one model"


def test_price_table_json_valid():
    """Test that the price table JSON is valid."""
    import json

    from app.settings import settings

    # Load and parse the price table
    price_table_path = settings.pricing.price_table_path
    with open(price_table_path) as f:
        price_data = json.load(f)

    # Verify it's a dictionary
    assert isinstance(price_data, dict), "Price table should be a dictionary"

    # Verify it has entries
    assert len(price_data) > 0, "Price table should contain at least one entry"

    # Verify each entry has the required structure (skip comment fields)
    for model_key, pricing in price_data.items():
        # Skip comment fields that start with underscore
        if model_key.startswith("_"):
            continue

        assert isinstance(model_key, str), f"Model key should be string: {model_key}"
        assert isinstance(pricing, dict), f"Pricing should be dict for {model_key}"
        assert "input_per_1k_usd" in pricing, f"Missing input_per_1k_usd for {model_key}"
        assert "output_per_1k_usd" in pricing, f"Missing output_per_1k_usd for {model_key}"
        assert isinstance(
            pricing["input_per_1k_usd"], int | float
        ), f"input_per_1k_usd should be numeric for {model_key}"
        assert isinstance(
            pricing["output_per_1k_usd"], int | float
        ), f"output_per_1k_usd should be numeric for {model_key}"
