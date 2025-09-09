# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""Mock fixtures for testing."""

from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock

__all__ = [
    "MockProvider",
    "MockRouter",
    "MockAuthenticator",
    "MockDatabase",
    "create_mock_response",
    "create_mock_user",
]


class MockProvider(Mock):
    """Mock provider for testing."""

    def __init__(self, name: str = "mock_provider", **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.chat_completion = AsyncMock(return_value=create_mock_response())
        self.calculate_cost = Mock(return_value=0.001)


class MockRouter(Mock):
    """Mock router for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.route = AsyncMock(return_value=("mock_provider", "mock_model", "Mock reasoning", 0.9))


class MockAuthenticator(Mock):
    """Mock authenticator for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.authenticate = AsyncMock(return_value=create_mock_user())


class MockDatabase(Mock):
    """Mock database for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_request = AsyncMock(return_value=123)
        self.get_request = AsyncMock(return_value=None)


def create_mock_response(**kwargs) -> dict:
    """Create a mock OpenAI-style response."""
    default_response = {
        "id": "mock-response-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Mock response content"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
    }

    # Update with any provided kwargs
    for key, value in kwargs.items():
        if key in default_response:
            if isinstance(default_response[key], dict):
                default_response[key].update(value)
            else:
                default_response[key] = value

    return default_response


def create_mock_user(user_id: str = "test_user", **kwargs) -> dict:
    """Create a mock user for testing."""
    default_user = {
        "user_id": user_id,
        "email": f"{user_id}@example.com",
        "role": "user",
        "api_key": f"test_key_{user_id}",
        "rate_limit": 100,
        "budget": 10.0,
        "active": True,
    }
    default_user.update(kwargs)
    return default_user
