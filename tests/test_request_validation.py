"""Test request validation for invalid model input formatting."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_invalid_model_format_rejected():
    """Test that invalid model formats are rejected with HTTP 400."""
    client = TestClient(app)

    # Test with proxy-style model names
    invalid_models = [
        "proxy:foo",
        "a/b",
        "proxy:gpt-4",
        "vendor/gpt-4",
        "openai:gpt-4",
        "anthropic/claude-3",
    ]

    for invalid_model in invalid_models:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": invalid_model,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            headers={"Authorization": "Bearer sk-test-claude-dev"},
        )

        assert response.status_code == 400, f"Expected 400 for model: {invalid_model}"
        error_data = response.json()
        assert error_data["error"]["type"] == "invalid_request"
        assert error_data["error"]["code"] == "invalid_model_format"
        assert "Invalid model name format" in error_data["error"]["message"]
        assert error_data["error"]["details"]["provided_model"] == invalid_model


def test_valid_model_formats_accepted():
    """Test that valid model formats are accepted."""
    client = TestClient(app)

    # Test with valid model names
    valid_models = [
        "gpt-4",
        "gpt-3.5-turbo",
        "claude-3-5-sonnet-latest",
        "claude-3-haiku-20240307",
        "mistral-small",
        "gemini-pro",
    ]

    for valid_model in valid_models:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": valid_model,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            headers={"Authorization": "Bearer sk-test-claude-dev"},
        )

        # Should not return 400 for invalid model format
        # (may return other errors like 401, 503, etc. which is expected)
        assert (
            response.status_code != 400 or "invalid_model_format" not in response.text
        ), f"Valid model {valid_model} was incorrectly rejected for format reasons"
