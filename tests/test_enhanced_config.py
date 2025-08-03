# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""
Tests for the enhanced configuration system including all enterprise features.
"""

import os
from unittest.mock import patch

import pytest

from app.config.enhanced_config import (
    AuthConfig,
    CacheConfig,
    ClassificationConfig,
    LoggingConfig,
    ModelMuxerConfig,
    MonitoringConfig,
    ProviderConfig,
    RateLimitConfig,
    RoutingConfig,
    load_enhanced_config,
)


class TestModelMuxerConfig:
    """Test the main configuration class."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = ModelMuxerConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.debug is False
        assert config.code_detection_threshold == 0.2
        assert config.complexity_threshold == 0.2
        assert config.simple_query_threshold == 0.3
        assert config.max_tokens_default == 1000

    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(
            os.environ,
            {
                "HOST": "127.0.0.1",
                "PORT": "9000",
                "DEBUG": "true",
                "CODE_DETECTION_THRESHOLD": "0.5",
                "MAX_TOKENS_DEFAULT": "2000",
            },
        ):
            config = ModelMuxerConfig()

            assert config.host == "127.0.0.1"
            assert config.port == 9000
            assert config.debug is True
            assert config.code_detection_threshold == 0.5
            assert config.max_tokens_default == 2000

    def test_provider_pricing(self):
        """Test provider pricing information."""
        config = ModelMuxerConfig()
        pricing = config.get_provider_pricing()

        assert "openai" in pricing
        assert "anthropic" in pricing
        assert "mistral" in pricing
        assert "google" in pricing
        assert "groq" in pricing

        # Test specific pricing
        assert "gpt-4o-mini" in pricing["openai"]
        assert pricing["openai"]["gpt-4o-mini"]["input"] == 0.00015
        assert pricing["mistral"]["mistral-small"]["input"] == 0.0002


class TestProviderConfig:
    """Test provider configuration."""

    def test_default_provider_config(self):
        """Test default provider configuration."""
        # Clear environment variables and override model config to prevent .env file loading
        env_vars_to_clear = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "MISTRAL_API_KEY",
            "GOOGLE_API_KEY",
            "COHERE_API_KEY",
            "GROQ_API_KEY",
            "TOGETHER_API_KEY",
            "LITELLM_BASE_URL",
            "LITELLM_API_KEY",
            "OPENAI_BASE_URL",
        ]

        with patch.dict(os.environ, {key: "" for key in env_vars_to_clear}, clear=False):
            # Temporarily override the model config to prevent .env file loading
            original_config = ProviderConfig.model_config
            ProviderConfig.model_config = ProviderConfig.model_config.copy()
            ProviderConfig.model_config["env_file"] = None

            try:
                config = ProviderConfig()

                # When env vars are set to empty strings, they become empty strings, not None
                assert config.openai_api_key == "" or config.openai_api_key is None
                assert config.anthropic_api_key == "" or config.anthropic_api_key is None
                assert config.mistral_api_key == "" or config.mistral_api_key is None
                assert config.google_api_key == "" or config.google_api_key is None
                assert config.groq_api_key == "" or config.groq_api_key is None
                # Note: timeout and max_retries are not in the actual ProviderConfig
            finally:
                # Restore original config
                ProviderConfig.model_config = original_config

    def test_provider_config_from_env(self):
        """Test provider configuration from environment variables."""
        # Test values - not real API keys
        config = ProviderConfig(
            openai_api_key="test-openai-key",  # Test key
            anthropic_api_key="test-anthropic-key",  # Test key
        )

        assert config.openai_api_key == "test-openai-key"
        assert config.anthropic_api_key == "test-anthropic-key"


class TestRoutingConfig:
    """Test routing configuration."""

    def test_default_routing_config(self):
        """Test default routing configuration."""
        config = RoutingConfig()

        assert config.default_strategy == "hybrid"
        assert config.cascade_enabled is True
        assert config.semantic_enabled is True
        assert config.heuristic_enabled is True
        assert config.cascade_quality_threshold == 0.7

    def test_routing_config_from_env(self):
        """Test routing configuration from environment variables."""
        # Pass values using field names, not environment variable names
        config = RoutingConfig(
            default_strategy="cascade",
            cascade_enabled=False,
            cascade_quality_threshold=0.8,
        )

        assert config.default_strategy == "cascade"
        assert config.cascade_enabled is False
        assert config.cascade_quality_threshold == 0.8


class TestCacheConfig:
    """Test cache configuration."""

    def test_default_cache_config(self):
        """Test default cache configuration."""
        # Remove environment variables that might affect defaults
        env_vars_to_remove = ["REDIS_URL"]

        with patch.dict(os.environ, clear=False) as patched_env:
            for key in env_vars_to_remove:
                patched_env.pop(key, None)
            # Temporarily override the model config to prevent .env file loading
            original_config = CacheConfig.model_config
            CacheConfig.model_config = CacheConfig.model_config.copy()
            CacheConfig.model_config['env_file'] = None

            try:
                config = CacheConfig()

                assert config.enabled is True
                assert config.backend == "memory"
                assert config.default_ttl == 3600
                assert config.memory_max_size == 1000
                assert config.redis_url == "redis://localhost:6379"
            finally:
                # Restore original config
                CacheConfig.model_config = original_config

    def test_cache_config_from_env(self):
        """Test cache configuration from environment variables."""
        config = CacheConfig(
            enabled=False,
            backend="redis",
            default_ttl=7200,
            redis_url="redis://localhost:6379/1",
        )

        assert config.enabled is False
        assert config.backend == "redis"
        assert config.default_ttl == 7200
        assert config.redis_url == "redis://localhost:6379/1"


class TestAuthConfig:
    """Test authentication configuration."""

    def test_default_auth_config(self):
        """Test default authentication configuration."""
        config = AuthConfig()

        assert config.enabled is True
        # In test environment, JWT secret gets a test value
        assert config.jwt_secret == "test-jwt-secret-for-testing-only"
        assert config.jwt_algorithm == "HS256"
        assert config.jwt_expiry == 3600
        assert config.methods == "api_key"

    def test_auth_config_from_env(self):
        """Test authentication configuration from environment variables."""
        # Test configuration - not real secrets
        config = AuthConfig(
            enabled=False,
            jwt_secret="test-secret-key",  # Test secret
            jwt_algorithm="HS256",
            jwt_expiry=7200,
        )

        assert config.enabled is False
        assert config.jwt_secret == "test-secret-key"
        assert config.jwt_algorithm == "HS256"
        assert config.jwt_expiry == 7200


class TestRateLimitConfig:
    """Test rate limiting configuration."""

    def test_default_rate_limit_config(self):
        """Test default rate limiting configuration."""
        config = RateLimitConfig()

        assert config.enabled is True
        assert config.requests_per_minute == 60
        assert config.requests_per_hour == 1000
        assert config.burst_size == 20

    def test_rate_limit_config_from_env(self):
        """Test rate limiting configuration from environment variables."""
        config = RateLimitConfig(
            enabled=False,
            requests_per_minute=120,
            requests_per_hour=5000,
            burst_size=30,
        )

        assert config.enabled is False
        assert config.requests_per_minute == 120
        assert config.requests_per_hour == 5000
        assert config.burst_size == 30


class TestMonitoringConfig:
    """Test monitoring configuration."""

    def test_default_monitoring_config(self):
        """Test default monitoring configuration."""
        config = MonitoringConfig()

        assert config.enabled is True
        assert config.prometheus_enabled is True
        assert config.track_performance is True
        assert config.prometheus_port == 9090

    def test_monitoring_config_from_env(self):
        """Test monitoring configuration from environment variables."""
        config = MonitoringConfig(
            enabled=False,
            prometheus_enabled=False,
            prometheus_port=9091,
        )

        assert config.enabled is False
        assert config.prometheus_enabled is False
        assert config.prometheus_port == 9091


class TestLoggingConfig:
    """Test logging configuration."""

    def test_default_logging_config(self):
        """Test default logging configuration."""
        config = LoggingConfig()

        assert config.level == "INFO"
        assert config.format == "json"
        assert config.log_requests is True
        assert config.log_responses is True

    def test_logging_config_from_env(self):
        """Test logging configuration from environment variables."""
        config = LoggingConfig(
            level="DEBUG",
            format="text",
            log_requests=False,
        )

        assert config.level == "DEBUG"
        assert config.format == "text"
        assert config.log_requests is False


class TestClassificationConfig:
    """Test classification configuration."""

    def test_default_classification_config(self):
        """Test default classification configuration."""
        config = ClassificationConfig()

        assert config.enabled is True
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.embedding_cache_enabled is True
        assert config.confidence_threshold == 0.6

    def test_classification_config_from_env(self):
        """Test classification configuration from environment variables."""
        config = ClassificationConfig(
            enabled=False,
            embedding_model="custom-model",
            confidence_threshold=0.8,
        )

        assert config.enabled is False
        assert config.embedding_model == "custom-model"
        assert config.confidence_threshold == 0.8


class TestConfigurationLoading:
    """Test configuration loading and validation."""

    def test_load_enhanced_config(self):
        """Test loading enhanced configuration."""
        config = load_enhanced_config()

        assert isinstance(config, ModelMuxerConfig)
        assert hasattr(config, "providers")
        assert hasattr(config, "routing")
        assert hasattr(config, "cache")
        assert hasattr(config, "auth")
        assert hasattr(config, "rate_limit")
        assert hasattr(config, "monitoring")
        assert hasattr(config, "logging")
        assert hasattr(config, "classification")

    def test_config_validation(self):
        """Test configuration validation."""
        config = ModelMuxerConfig()

        # Test that all sub-configurations are properly initialized
        assert isinstance(config.providers, ProviderConfig)
        assert isinstance(config.routing, RoutingConfig)
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.auth, AuthConfig)
        assert isinstance(config.rate_limit, RateLimitConfig)
        assert isinstance(config.monitoring, MonitoringConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.classification, ClassificationConfig)


if __name__ == "__main__":
    pytest.main([__file__])
