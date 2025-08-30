# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""Test direct router preferences and model selection."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.router import HeuristicRouter
from app.providers.registry import get_provider_registry


def test_router_preferences_structure():
    """Test that router preferences have the expected structure."""
    router = HeuristicRouter(provider_registry_fn=get_provider_registry)

    # Check that preferences exist
    assert hasattr(router, "model_preferences")
    assert isinstance(router.model_preferences, dict)

    # Check expected task types
    expected_task_types = ["code", "complex", "simple", "general"]
    for task_type in expected_task_types:
        assert task_type in router.model_preferences, f"Missing task type: {task_type}"
        assert isinstance(router.model_preferences[task_type], list), f"Task type {task_type} should be a list"
        assert len(router.model_preferences[task_type]) > 0, f"Task type {task_type} should have at least one model"

    # Check that each preference is a tuple of (provider, model)
    for task_type, preferences in router.model_preferences.items():
        for preference in preferences:
            assert isinstance(preference, tuple), f"Preference in {task_type} should be a tuple"
            assert len(preference) == 2, f"Preference in {task_type} should have 2 elements (provider, model)"
            assert isinstance(preference[0], str), f"Provider should be string in {task_type}"
            assert isinstance(preference[1], str), f"Model should be string in {task_type}"


def test_model_name_formatting_no_separators():
    """Test that model names in router preferences don't contain separators."""
    from app.router import HeuristicRouter
    from app.providers.registry import get_provider_registry

    router = HeuristicRouter(provider_registry_fn=get_provider_registry)
    for _, preferences in router.model_preferences.items():
        for provider, model in preferences:
            assert ":" not in model and "/" not in model, f"Model '{model}' should not contain ':' or '/'"
