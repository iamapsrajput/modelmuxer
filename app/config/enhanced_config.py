# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Enhanced configuration management for ModelMuxer.

This module provides comprehensive configuration management for all
ModelMuxer features including routing, caching, authentication, and monitoring.
"""

import structlog
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Import centralized settings to avoid duplication
from ..settings import settings as app_settings

logger = structlog.get_logger(__name__)


class ProviderConfig(BaseSettings):
    """Configuration for LLM providers."""

    # OpenAI
    openai_api_key: str | None = Field(default=None)
    openai_base_url: str | None = Field(default=None)

    # Anthropic
    anthropic_api_key: str | None = Field(default=None)

    # Mistral
    mistral_api_key: str | None = Field(default=None)

    # Google
    google_api_key: str | None = Field(default=None)

    # Cohere
    cohere_api_key: str | None = Field(default=None)

    # Groq
    groq_api_key: str | None = Field(default=None)

    # Together AI
    together_api_key: str | None = Field(default=None)

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    def validate_at_least_one_provider(self) -> bool:
        """Validate that at least one provider API key is configured."""
        import os
        import sys

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

    default_strategy: str = Field(default="hybrid")

    # Heuristic router settings
    heuristic_enabled: bool = Field(default=True)

    # Semantic router settings
    semantic_enabled: bool = Field(default=True)
    semantic_model: str = Field(default="all-MiniLM-L6-v2")
    semantic_threshold: float = Field(default=0.6)

    # Cascade router settings
    cascade_enabled: bool = Field(default=True)
    cascade_quality_threshold: float = Field(default=0.7)

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")


class CacheConfig(BaseSettings):
    """Configuration for caching features."""

    enabled: bool = Field(default=True)
    backend: str = Field(default="memory")
    ttl: int = Field(default=3600)
    max_size: int = Field(default=1000)
    default_ttl: int = Field(default=3600)
    memory_max_size: int = Field(default=1000)

    # Redis settings (if using Redis backend)
    redis_url: str | None = Field(default="redis://localhost:6379")
    redis_db: int = Field(default=0)

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")


class AuthConfig(BaseSettings):
    """Configuration for authentication and authorization."""

    enabled: bool = Field(default=True)
    api_keys: str = Field(default="")
    jwt_secret: str | None = Field(default=None)
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration: int = Field(default=3600)
    jwt_expiry: int = Field(default=3600)
    methods: str = Field(default="api_key")

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    def get_api_keys_list(self) -> list[str]:
        """Get list of allowed API keys."""
        if not self.api_keys:
            return app_settings.api.api_keys
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]


class RateLimitConfig(BaseSettings):
    """Configuration for rate limiting."""

    enabled: bool = Field(default=True)
    requests_per_minute: int = Field(default=60)
    requests_per_hour: int = Field(default=1000)
    burst_limit: int = Field(default=10)
    burst_size: int = Field(default=20)

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")


class MonitoringConfig(BaseSettings):
    """Configuration for monitoring and observability."""

    enabled: bool = Field(default=True)
    prometheus_enabled: bool = Field(default=True)
    health_check_enabled: bool = Field(default=True)
    metrics_interval: int = Field(default=60)
    track_performance: bool = Field(default=True)
    prometheus_port: int = Field(default=9090)

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")


class LoggingConfig(BaseSettings):
    """Configuration for logging."""

    level: str = Field(default="INFO")
    format: str = Field(default="json")
    structured: bool = Field(default=True)
    log_requests: bool = Field(default=True)
    log_responses: bool = Field(default=True)

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")


class ClassificationConfig(BaseSettings):
    """Configuration for ML-based classification."""

    enabled: bool = Field(default=True)
    model_name: str = Field(default="all-MiniLM-L6-v2")
    confidence_threshold: float = Field(default=0.6)
    max_history_size: int = Field(default=1000)
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_cache_enabled: bool = Field(default=True)

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")


class ModelMuxerConfig:
    """Main configuration class combining all sub-configurations."""

    def __init__(self):
        # Server settings - allow environment override for testing
        import os

        self.host = os.getenv("HOST", app_settings.server.host)
        self.port = int(os.getenv("PORT", str(app_settings.server.port)))
        self.debug = os.getenv("DEBUG", str(app_settings.server.debug)).lower() in (
            "true",
            "1",
            "yes",
        )

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
        self.code_detection_threshold = float(
            os.getenv("CODE_DETECTION_THRESHOLD", str(app_settings.router.code_detection_threshold))
        )
        self.complexity_threshold = app_settings.router.complexity_threshold
        self.simple_query_threshold = 0.3  # Default value for enhanced mode
        self.simple_query_max_length = app_settings.router.simple_query_max_length
        self.max_tokens_default = int(
            os.getenv("MAX_TOKENS_DEFAULT", str(app_settings.router.max_tokens_default))
        )

    def get_allowed_api_keys(self) -> list[str]:
        """Get list of allowed API keys from auth configuration."""
        # Use centralized settings for API keys
        return app_settings.api.api_keys

    def get_provider_pricing(self) -> dict[str, dict[str, dict[str, float]]]:
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
