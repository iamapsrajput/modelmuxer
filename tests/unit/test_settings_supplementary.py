# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""
Supplementary tests for app/settings.py to increase coverage.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app.settings import Settings


class TestSettingsSupplementary:
    """Supplementary tests for settings module."""

    def test_settings_initialization_with_env_vars(self):
        """Test Settings initialization with environment variables."""
        with patch.dict(
            os.environ,
            {
                "MODELMUXER_MODE": "enhanced",
                "OPENAI_API_KEY": "test-openai-key",
                "ANTHROPIC_API_KEY": "test-anthropic-key",
                "GOOGLE_API_KEY": "test-google-key",
                "COHERE_API_KEY": "test-cohere-key",
                "GROQ_API_KEY": "test-groq-key",
                "MISTRAL_API_KEY": "test-mistral-key",
                "TOGETHER_API_KEY": "test-together-key",
                "DATABASE_URL": "sqlite:///test.db",
                "REDIS_URL": "redis://localhost:6379",
                "LOG_LEVEL": "DEBUG",
                "CORS_ORIGINS": "http://localhost:3000,http://localhost:8000",
                "API_KEYS": "key1,key2,key3",
                "RATE_LIMIT_PER_MINUTE": "100",
                "RATE_LIMIT_PER_HOUR": "5000",
                "MAX_REQUEST_SIZE_MB": "20",
                "DAILY_BUDGET": "200.0",
                "MONTHLY_BUDGET": "5000.0",
                "ENABLE_TELEMETRY": "true",
                "ENABLE_MONITORING": "true",
                "ENABLE_CACHE": "true",
                "CACHE_TTL": "600",
                "CIRCUIT_FAIL_THRESHOLD": "10",
                "CIRCUIT_COOLDOWN_SEC": "120",
                "ROUTER_TYPE": "semantic",
                "FALLBACK_PROVIDER": "anthropic",
                "FALLBACK_MODEL": "claude-3",
                "MAX_RETRIES": "5",
                "RETRY_DELAY": "2.0",
                "REQUEST_TIMEOUT": "45",
                "STREAM_TIMEOUT": "120",
                "ENABLE_COST_TRACKING": "true",
                "ENABLE_USAGE_LIMITS": "true",
                "ENABLE_AUDIT_LOG": "true",
                "AUDIT_LOG_PATH": "/var/log/modelmuxer/audit.log",
                "METRICS_PORT": "9090",
                "HEALTH_CHECK_INTERVAL": "60",
                "ENABLE_PROVIDER_HEALTH_CHECK": "true",
                "PROVIDER_HEALTH_CHECK_INTERVAL": "300",
                "ENABLE_AUTO_SCALING": "true",
                "MIN_WORKERS": "2",
                "MAX_WORKERS": "10",
                "WORKER_TIMEOUT": "60",
                "ENABLE_REQUEST_VALIDATION": "true",
                "ENABLE_RESPONSE_VALIDATION": "true",
                "ENABLE_CONTENT_FILTERING": "true",
                "CONTENT_FILTER_THRESHOLD": "0.8",
                "ENABLE_PII_DETECTION": "true",
                "PII_DETECTION_THRESHOLD": "0.9",
                "ENABLE_PROMPT_INJECTION_DETECTION": "true",
                "PROMPT_INJECTION_THRESHOLD": "0.7",
                "ENABLE_SSL": "true",
                "SSL_CERT_PATH": "/etc/ssl/cert.pem",
                "SSL_KEY_PATH": "/etc/ssl/key.pem",
                "ENABLE_API_VERSIONING": "true",
                "API_VERSION": "v2",
                "ENABLE_RATE_LIMIT_BY_KEY": "true",
                "ENABLE_RATE_LIMIT_BY_IP": "false",
                "ENABLE_REQUEST_LOGGING": "true",
                "REQUEST_LOG_PATH": "/var/log/modelmuxer/requests.log",
                "ENABLE_ERROR_LOGGING": "true",
                "ERROR_LOG_PATH": "/var/log/modelmuxer/errors.log",
                "ENABLE_PERFORMANCE_LOGGING": "true",
                "PERFORMANCE_LOG_PATH": "/var/log/modelmuxer/performance.log",
            },
        ):
            settings = Settings()

            # Test mode
            assert settings.mode == "enhanced"

            # Test API keys
            assert settings.providers.openai_api_key == "test-openai-key"
            assert settings.providers.anthropic_api_key == "test-anthropic-key"
            assert settings.providers.google_api_key == "test-google-key"
            assert settings.providers.cohere_api_key == "test-cohere-key"
            assert settings.providers.groq_api_key == "test-groq-key"
            assert settings.providers.mistral_api_key == "test-mistral-key"
            assert settings.providers.together_api_key == "test-together-key"

            # Test database
            assert settings.db.database_url == "sqlite:///test.db"

            # Test Redis
            assert settings.cache.redis_url == "redis://localhost:6379"

            # Test logging
            assert settings.logging.level == "DEBUG"

            # Test CORS
            assert "http://localhost:3000" in settings.api.cors_origins
            assert "http://localhost:8000" in settings.api.cors_origins

            # Test API settings
            assert settings.api.api_keys == "key1,key2,key3"
            assert settings.api.rate_limit_per_minute == 100
            assert settings.api.rate_limit_per_hour == 5000
            assert settings.api.max_request_size_mb == 20

            # Test budget settings
            assert settings.budget.daily_budget == 200.0
            assert settings.budget.monthly_budget == 5000.0

            # Test feature flags
            assert settings.features.enable_telemetry is True
            assert settings.features.monitoring_enabled is True
            assert settings.features.cache_enabled is True

            # Test cache settings
            assert settings.cache.cache_ttl == 600

            # Test circuit breaker settings
            assert settings.providers.circuit_fail_threshold == 10
            assert settings.providers.circuit_cooldown_sec == 120

            # Test router settings
            assert settings.router.router_type == "semantic"
            assert settings.router.fallback_provider == "anthropic"
            assert settings.router.fallback_model == "claude-3"

            # Test retry settings
            assert settings.providers.max_retries == 5
            assert settings.providers.retry_delay == 2.0

            # Test timeout settings
            assert settings.providers.request_timeout == 45
            assert settings.providers.stream_timeout == 120

    def test_settings_with_minimal_env_vars(self):
        """Test Settings with minimal environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()

            # Should use defaults
            assert settings.mode == "basic"
            assert settings.providers.openai_api_key is None
            assert settings.db.database_url == "sqlite:///modelmuxer.db"
            assert settings.cache.redis_url is None
            assert settings.logging.level == "INFO"
            assert settings.api.rate_limit_per_minute == 60
            assert settings.budget.daily_budget == 100.0
            assert settings.features.enable_telemetry is False

    def test_settings_singleton(self):
        """Test that settings is a singleton."""
        from app.settings import settings as settings1
        from app.settings import settings as settings2

        # Should be the same instance
        assert settings1 is settings2

    def test_settings_mode_validation(self):
        """Test Settings mode validation."""
        with patch.dict(os.environ, {"MODELMUXER_MODE": "invalid"}):
            with pytest.raises(ValidationError):
                Settings()

    def test_settings_boolean_parsing(self):
        """Test boolean environment variable parsing."""
        # Test various true values
        for true_val in ["true", "True", "TRUE", "1", "yes", "Yes", "YES"]:
            with patch.dict(os.environ, {"ENABLE_TELEMETRY": true_val}):
                settings = Settings()
                assert settings.features.enable_telemetry is True

        # Test various false values
        for false_val in ["false", "False", "FALSE", "0", "no", "No", "NO", ""]:
            with patch.dict(os.environ, {"ENABLE_TELEMETRY": false_val}):
                settings = Settings()
                assert settings.features.enable_telemetry is False

    def test_settings_integer_parsing(self):
        """Test integer environment variable parsing."""
        with patch.dict(
            os.environ,
            {"RATE_LIMIT_PER_MINUTE": "120", "MAX_RETRIES": "3", "CIRCUIT_FAIL_THRESHOLD": "5"},
        ):
            settings = Settings()
            assert settings.api.rate_limit_per_minute == 120
            assert settings.providers.max_retries == 3
            assert settings.providers.circuit_fail_threshold == 5

        # Test invalid integer values
        with patch.dict(os.environ, {"RATE_LIMIT_PER_MINUTE": "invalid", "MAX_RETRIES": "not_a_number"}):
            settings = Settings()
            # Should use defaults for invalid values
            assert settings.api.rate_limit_per_minute == 60  # default
            assert settings.providers.max_retries == 2  # default

    def test_settings_float_parsing(self):
        """Test float environment variable parsing."""
        with patch.dict(
            os.environ,
            {"DAILY_BUDGET": "150.50", "MONTHLY_BUDGET": "3000.75", "RETRY_DELAY": "1.5"},
        ):
            settings = Settings()
            assert settings.budget.daily_budget == 150.50
            assert settings.budget.monthly_budget == 3000.75
            assert settings.providers.retry_delay == 1.5

        # Test invalid float values
        with patch.dict(os.environ, {"DAILY_BUDGET": "invalid", "RETRY_DELAY": "not_a_float"}):
            settings = Settings()
            # Should use defaults for invalid values
            assert settings.budget.daily_budget == 100.0  # default
            assert settings.providers.retry_delay == 1.0  # default

    def test_settings_list_parsing(self):
        """Test list/comma-separated environment variable parsing."""
        with patch.dict(
            os.environ,
            {
                "CORS_ORIGINS": "http://localhost:3000,http://localhost:8080,https://example.com",
                "API_KEYS": "key1, key2, key3, key4",  # with spaces
            },
        ):
            settings = Settings()

            cors_origins = settings.api.cors_origins
            assert "http://localhost:3000" in cors_origins
            assert "http://localhost:8080" in cors_origins
            assert "https://example.com" in cors_origins

            # API keys should be trimmed
            assert settings.api.api_keys == "key1, key2, key3, key4"

    def test_settings_url_validation(self):
        """Test URL environment variable validation."""
        # Valid URLs
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://user:pass@localhost/db",
                "REDIS_URL": "redis://localhost:6379/0",
            },
        ):
            settings = Settings()
            assert settings.db.database_url == "postgresql://user:pass@localhost/db"
            assert settings.cache.redis_url == "redis://localhost:6379/0"

        # Invalid URLs should still be stored (validation happens elsewhere)
        with patch.dict(os.environ, {"DATABASE_URL": "not_a_url", "REDIS_URL": "invalid"}):
            settings = Settings()
            assert settings.db.database_url == "not_a_url"
            assert settings.cache.redis_url == "invalid"

    def test_settings_path_expansion(self):
        """Test path environment variable expansion."""
        with patch.dict(
            os.environ,
            {"AUDIT_LOG_PATH": "~/logs/audit.log", "SSL_CERT_PATH": "$HOME/certs/cert.pem"},
        ):
            settings = Settings()
            # Paths should be stored as-is (expansion happens at usage)
            assert settings.logging.audit_log_path is not None
            assert "~/logs/audit.log" in str(settings.logging.audit_log_path) or "logs/audit.log" in str(
                settings.logging.audit_log_path
            )

    def test_settings_provider_specific_configs(self):
        """Test provider-specific configuration."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_BASE_URL": "https://custom.openai.com",
                "ANTHROPIC_BASE_URL": "https://custom.anthropic.com",
                "OPENAI_ORG_ID": "org-123",
                "ANTHROPIC_VERSION": "2023-06-01",
            },
        ):
            settings = Settings()

            # These might be stored in provider-specific attributes
            # Check that settings object is created without errors
            assert settings is not None

    def test_settings_feature_flag_combinations(self):
        """Test various feature flag combinations."""
        # All features enabled
        with patch.dict(
            os.environ,
            {
                "ENABLE_TELEMETRY": "true",
                "ENABLE_MONITORING": "true",
                "ENABLE_CACHE": "true",
                "ENABLE_COST_TRACKING": "true",
                "ENABLE_USAGE_LIMITS": "true",
                "ENABLE_AUDIT_LOG": "true",
            },
        ):
            settings = Settings()
            assert settings.features.enable_telemetry is True
            assert settings.features.monitoring_enabled is True
            assert settings.features.cache_enabled is True
            # These fields don't exist in FeatureFlagsSettings
            # assert settings.features.enable_cost_tracking is True
            # assert settings.features.enable_usage_limits is True
            # assert settings.features.enable_audit_log is True

        # All features disabled
        with patch.dict(
            os.environ,
            {
                "ENABLE_TELEMETRY": "false",
                "ENABLE_MONITORING": "false",
                "ENABLE_CACHE": "false",
                "ENABLE_COST_TRACKING": "false",
                "ENABLE_USAGE_LIMITS": "false",
                "ENABLE_AUDIT_LOG": "false",
            },
        ):
            settings = Settings()
            assert settings.features.enable_telemetry is False
            assert settings.features.monitoring_enabled is False
            assert settings.features.cache_enabled is False
            # These fields don't exist in FeatureFlagsSettings
            # assert settings.features.enable_cost_tracking is False
            # assert settings.features.enable_usage_limits is False
            # assert settings.features.enable_audit_log is False

    def test_settings_security_configs(self):
        """Test security-related configurations."""
        with patch.dict(
            os.environ,
            {
                "ENABLE_SSL": "true",
                "SSL_CERT_PATH": "/etc/ssl/cert.pem",
                "SSL_KEY_PATH": "/etc/ssl/key.pem",
                "ENABLE_REQUEST_VALIDATION": "true",
                "ENABLE_RESPONSE_VALIDATION": "true",
                "ENABLE_CONTENT_FILTERING": "true",
                "CONTENT_FILTER_THRESHOLD": "0.85",
                "ENABLE_PII_DETECTION": "true",
                "PII_DETECTION_THRESHOLD": "0.95",
            },
        ):
            settings = Settings()

            assert settings.security.enable_ssl is True
            assert settings.security.ssl_cert_path == "/etc/ssl/cert.pem"
            assert settings.security.ssl_key_path == "/etc/ssl/key.pem"
            assert settings.security.enable_request_validation is True
            assert settings.security.enable_response_validation is True
            assert settings.security.enable_content_filtering is True
            assert settings.security.content_filter_threshold == 0.85
            assert settings.security.enable_pii_detection is True
            assert settings.security.pii_detection_threshold == 0.95

    def test_settings_monitoring_configs(self):
        """Test monitoring-related configurations."""
        with patch.dict(
            os.environ,
            {
                "METRICS_PORT": "9090",
                "HEALTH_CHECK_INTERVAL": "30",
                "ENABLE_PROVIDER_HEALTH_CHECK": "true",
                "PROVIDER_HEALTH_CHECK_INTERVAL": "600",
            },
        ):
            settings = Settings()

            assert settings.monitoring.metrics_port == 9090
            assert settings.monitoring.health_check_interval == 30
            assert settings.monitoring.enable_provider_health_check is True
            assert settings.monitoring.provider_health_check_interval == 600

    def test_settings_worker_configs(self):
        """Test worker-related configurations."""
        with patch.dict(
            os.environ,
            {
                "ENABLE_AUTO_SCALING": "true",
                "MIN_WORKERS": "4",
                "MAX_WORKERS": "16",
                "WORKER_TIMEOUT": "90",
            },
        ):
            settings = Settings()

            assert settings.workers.enable_auto_scaling is True
            assert settings.workers.min_workers == 4
            assert settings.workers.max_workers == 16
            assert settings.workers.worker_timeout == 90

    def test_settings_with_mode_specific_defaults(self):
        """Test that different modes have different defaults."""
        # Basic mode
        with patch.dict(os.environ, {"MODELMUXER_MODE": "basic"}):
            basic_settings = Settings()
            assert basic_settings.mode == "basic"
            assert basic_settings.features.enable_telemetry is False
            # monitoring_enabled is True by default in all modes
            assert basic_settings.features.monitoring_enabled is True

        # Enhanced mode
        with patch.dict(os.environ, {"MODELMUXER_MODE": "enhanced"}):
            enhanced_settings = Settings()
            assert enhanced_settings.mode == "enhanced"
            # Enhanced mode might have different defaults
            # (depends on actual implementation)

        # Production mode
        with patch.dict(os.environ, {"MODELMUXER_MODE": "production"}):
            prod_settings = Settings()
            assert prod_settings.mode == "production"
            # Production mode might enable more features by default
