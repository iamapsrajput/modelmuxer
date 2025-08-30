# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Enhanced configuration management for ModelMuxer.

This module provides comprehensive configuration management for all
ModelMuxer features including routing, caching, authentication, and monitoring.
"""

import structlog
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Import centralized settings to avoid duplication
from ..settings import settings as app_settings

logger = structlog.get_logger(__name__)


class ProviderConfig(BaseSettings):
    """Configuration for LLM providers."""

    # OpenAI
    openai_api_key: str | None = Field(default=None, env="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, env="OPENAI_BASE_URL")

    # Anthropic
    anthropic_api_key: str | None = Field(default=None, env="ANTHROPIC_API_KEY")

    # Mistral
    mistral_api_key: str | None = Field(default=None, env="MISTRAL_API_KEY")

    # Google
    google_api_key: str | None = Field(default=None, env="GOOGLE_API_KEY")

    # Cohere
    cohere_api_key: str | None = Field(default=None, env="COHERE_API_KEY")

    # Groq
    groq_api_key: str | None = Field(default=None, env="GROQ_API_KEY")

    # Together AI
    together_api_key: str | None = Field(default=None, env="TOGETHER_API_KEY")


    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    def validate_at_least_one_provider(self) -> bool:
        """Validate that at least one provider API key is configured."""
        import sys
        import os

        # Skip validation in test environments
        if "pytest" in sys.modules or "TESTING" in os.environ:
            return True

        providers = [
            self.openai_api_key or app_settings.api.openai_api_key,
            self.anthropic_api_key or app_settings.api.anthropic_api_key,
            self.mistral_api_key or app_settings.api.mistral_api_key,
            self.google_api_key,
            self.cohere_api_key,
            self.groq_api_key,
            self.together_api_key,
        ]


        configured_providers = [
            p for p in providers if p and not p.startswith("your-") and not p.endswith("-here")
        ]

        if not configured_providers:
            raise ValueError(
                "At least one LLM provider API key must be configured. "
                "Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, MISTRAL_API_KEY, "
                "GOOGLE_API_KEY, COHERE_API_KEY, GROQ_API_KEY, TOGETHER_API_KEY, "
            )

        return True


class RoutingConfig(BaseSettings):
    """Configuration for routing strategies."""

    default_strategy: str = Field(default="hybrid", env="DEFAULT_ROUTING_STRATEGY")

    # Heuristic router settings
    heuristic_enabled: bool = Field(default=True, env="HEURISTIC_ROUTING_ENABLED")

    # Semantic router settings
    semantic_enabled: bool = Field(default=True, env="SEMANTIC_ROUTING_ENABLED")
    semantic_model: str = Field(default="all-MiniLM-L6-v2", env="SEMANTIC_MODEL")
    semantic_threshold: float = Field(default=0.6, env="SEMANTIC_THRESHOLD")

    # Cascade router settings
    cascade_enabled: bool = Field(default=True, env="CASCADE_ROUTING_ENABLED")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class CacheConfig(BaseSettings):
    """Configuration for caching features."""

    enabled: bool = Field(default=True, env="CACHE_ENABLED")
    backend: str = Field(default="memory", env="CACHE_BACKEND")
    ttl: int = Field(default=3600, env="CACHE_TTL")
    max_size: int = Field(default=1000, env="CACHE_MAX_SIZE")

    # Redis settings (if using Redis backend)
    redis_url: str | None = Field(default=None, env="REDIS_URL")
    redis_db: int = Field(default=0, env="REDIS_DB")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class AuthConfig(BaseSettings):
    """Configuration for authentication and authorization."""

    enabled: bool = Field(default=True, env="AUTH_ENABLED")
    api_keys: str = Field(default="", env="API_KEYS")
    jwt_secret: str | None = Field(default=None, env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration: int = Field(default=3600, env="JWT_EXPIRATION")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    def get_api_keys_list(self) -> list[str]:
        """Get list of allowed API keys."""
        if not self.api_keys:
            return app_settings.api.api_keys
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]


class RateLimitConfig(BaseSettings):
    """Configuration for rate limiting."""

    enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    requests_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    requests_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    burst_limit: int = Field(default=10, env="RATE_LIMIT_BURST")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class MonitoringConfig(BaseSettings):
    """Configuration for monitoring and observability."""

    enabled: bool = Field(default=True, env="MONITORING_ENABLED")
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    health_check_enabled: bool = Field(default=True, env="HEALTH_CHECK_ENABLED")
    metrics_interval: int = Field(default=60, env="METRICS_INTERVAL")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class LoggingConfig(BaseSettings):
    """Configuration for logging."""

    level: str = Field(default="info", env="LOG_LEVEL")
    format: str = Field(default="json", env="LOG_FORMAT")
    structured: bool = Field(default=True, env="LOG_STRUCTURED")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class ClassificationConfig(BaseSettings):
    """Configuration for ML-based classification."""

    enabled: bool = Field(default=True, env="CLASSIFICATION_ENABLED")
    model_name: str = Field(default="all-MiniLM-L6-v2", env="CLASSIFICATION_MODEL")
    confidence_threshold: float = Field(default=0.6, env="CLASSIFICATION_CONFIDENCE_THRESHOLD")
    max_history_size: int = Field(default=1000, env="CLASSIFICATION_HISTORY_SIZE")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class ModelMuxerConfig:
    """Main configuration class combining all sub-configurations."""

    def __init__(self):
        # Server settings from centralized settings
        self.host = app_settings.server.host
        self.port = app_settings.server.port
        self.debug = app_settings.server.debug

        # Initialize sub-configurations
        self.providers = ProviderConfig()
        self.routing = RoutingConfig()
        self.cache = CacheConfig()
        self.auth = AuthConfig()
        self.rate_limit = RateLimitConfig()
        self.monitoring = MonitoringConfig()
        self.logging = LoggingConfig()
        self.classification = ClassificationConfig()

        # Legacy compatibility attributes from centralized settings
        self.code_detection_threshold = app_settings.router.code_detection_threshold
        self.complexity_threshold = app_settings.router.complexity_threshold
        self.simple_query_threshold = 0.3  # Default value for enhanced mode
        self.simple_query_max_length = app_settings.router.simple_query_max_length
        self.max_tokens_default = app_settings.router.max_tokens_default

    def get_allowed_api_keys(self) -> list[str]:
        """Get list of allowed API keys from auth configuration."""
        # Use centralized settings for API keys
        return app_settings.api.api_keys

    def get_provider_pricing(self) -> dict[str, dict[str, float]]:
        """Get provider pricing information for cost calculation."""
        # Use centralized pricing settings
        from ..settings import get_provider_pricing

        return get_provider_pricing()


def load_enhanced_config() -> ModelMuxerConfig:
    """Load and validate enhanced configuration."""
    try:
        config = ModelMuxerConfig()

        # Log configuration summary (without sensitive data)
        logger.info(
            "enhanced_configuration_loaded",
            host=config.host,
            port=config.port,
            debug=config.debug,
            routing_strategy=config.routing.default_strategy,
            cache_enabled=config.cache.enabled,
            cache_backend=config.cache.backend,
            auth_enabled=config.auth.enabled,
            rate_limit_enabled=config.rate_limit.enabled,
            monitoring_enabled=config.monitoring.enabled,
            classification_enabled=config.classification.enabled,
            log_level=config.logging.level,
        )

        return config

    except Exception as e:
        logger.error("enhanced_configuration_load_failed", error=str(e))
        raise


# Global enhanced configuration instance
enhanced_config = load_enhanced_config()
