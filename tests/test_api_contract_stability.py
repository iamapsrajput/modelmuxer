# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""Test API contract stability for /providers and /v1/models endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_providers_endpoint_contract():
    """Test that /providers endpoint maintains stable contract."""
    client = TestClient(app)

    # Test both /providers and /v1/providers endpoints
    for endpoint in ["/providers", "/v1/providers"]:
        response = client.get(endpoint, headers={"Authorization": "Bearer sk-test-claude-dev"})

        # Should not return 401 (authentication error) or 500 (server error)
        assert response.status_code in [
            200,
            404,
            503,
        ], f"Unexpected status code {response.status_code} for {endpoint}"

        if response.status_code == 200:
            data = response.json()

            # Verify response structure
            assert "providers" in data, f"Response missing 'providers' key for {endpoint}"
            assert isinstance(data["providers"], dict), f"'providers' should be dict for {endpoint}"

            # Verify each provider has required fields
            for provider_name, provider_info in data["providers"].items():
                assert "name" in provider_info, f"Provider missing 'name' field for {endpoint}"
                assert "models" in provider_info, f"Provider missing 'models' field for {endpoint}"
                assert "status" in provider_info, f"Provider missing 'status' field for {endpoint}"

                assert (
                    provider_info["name"] == provider_name
                ), f"Provider name mismatch for {endpoint}"
                assert isinstance(
                    provider_info["models"], list
                ), f"Models should be list for {endpoint}"
                assert provider_info["status"] in [
                    "available",
                    "unavailable",
                ], f"Invalid status for {endpoint}"


def test_models_endpoint_contract():
    """Test that /v1/models endpoint maintains stable contract."""
    client = TestClient(app)

    response = client.get("/v1/models", headers={"Authorization": "Bearer sk-test-claude-dev"})

    # Should not return 401 (authentication error) or 500 (server error)
    assert response.status_code in [200, 404, 503], f"Unexpected status code {response.status_code}"

    if response.status_code == 200:
        data = response.json()

        # Verify response structure
        assert "object" in data, "Response missing 'object' key"
        assert "data" in data, "Response missing 'data' key"
        assert data["object"] == "list", "'object' should be 'list'"
        assert isinstance(data["data"], list), "'data' should be list"

        # Verify each model has required fields
        for model in data["data"]:
            assert "id" in model, "Model missing 'id' field"
            assert "object" in model, "Model missing 'object' field"
            assert "provider" in model, "Model missing 'provider' field"
            assert "model" in model, "Model missing 'model' field"
            assert "pricing" in model, "Model missing 'pricing' field"

            assert model["object"] == "model", "'object' should be 'model'"
            assert isinstance(model["pricing"], dict), "'pricing' should be dict"

            # Verify pricing structure
            pricing = model["pricing"]
            assert "input" in pricing, "Pricing missing 'input' field"
            assert "output" in pricing, "Pricing missing 'output' field"
            assert "unit" in pricing, "Pricing missing 'unit' field"
            assert pricing["unit"] == "per_million_tokens", "'unit' should be 'per_million_tokens'"

            # Verify model name format (no proxy-style separators)
            assert (
                ":" not in model["model"] and "/" not in model["model"]
            ), f"Model name should not contain ':' or '/': {model['model']}"
