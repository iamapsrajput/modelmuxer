# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import pytest


@contextmanager
def temp_env(env: dict[str, str]) -> Iterator[None]:
    old_env = {k: os.environ.get(k) for k in env}
    try:
        os.environ.update(env)
        yield
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def reload_settings_module():
    # Force reload to apply new env
    import importlib

    import app.settings as settings_module

    importlib.reload(settings_module)
    return settings_module.settings


def test_defaults_load():
    with temp_env({}):
        settings = reload_settings_module()
        assert settings.db.database_url.startswith("sqlite:///")
        assert settings.server.port == 8000
        assert isinstance(settings.observability.cors_origins, list)


def test_env_overrides():
    with temp_env(
        {
            "DATABASE_URL": "sqlite:///./test.db",
            "PORT": "9000",
            "OPENAI_API_KEY": "sk-abc",
        }
    ):
        settings = reload_settings_module()
        assert settings.db.database_url.endswith("test.db")
        assert settings.server.port == 9000
        assert settings.api.openai_api_key == "sk-abc"


def test_validation_errors():
    with temp_env({"PORT": "0"}):
        with pytest.raises(ValueError):
            reload_settings_module()

    with temp_env({"REDIS_DB": "-1"}):
        with pytest.raises(ValueError):
            reload_settings_module()


def test_settings_structure():
    """Test that the settings object has the expected structure."""
    from app.settings import settings

    # Test that all expected groups exist
    assert hasattr(settings, "api")
    assert hasattr(settings, "db")
    assert hasattr(settings, "redis")
    assert hasattr(settings, "observability")
    assert hasattr(settings, "features")
    assert hasattr(settings, "server")
    assert hasattr(settings, "router")
    assert hasattr(settings, "pricing")

    # Test that key fields exist
    assert hasattr(settings.api, "openai_api_key")
    assert hasattr(settings.db, "database_url")
    assert hasattr(settings.server, "port")
    assert hasattr(settings.features, "mode")
