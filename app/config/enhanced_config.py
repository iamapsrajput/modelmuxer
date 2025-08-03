# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Enhanced configuration management for ModelMuxer.

This module provides comprehensive configuration management for all
ModelMuxer features including routing, caching, authentication, and monitoring.
"""

import os

import structlog
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    # LiteLLM Proxy
    litellm_base_url: str | None = Field(default=None, env="LITELLM_BASE_URL")
    litellm_api_key: str | None = Field(default=None, env="LITELLM_API_KEY")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    def validate_at_least_one_provider(self) -> bool:
        """Validate that at least one provider API key is configured."""
        import os

        # Skip validation in test environments
        if os.getenv("TESTING") == "true" or "pytest" in os.getenv("_", ""):
            return True

        providers = [
            self.openai_api_key,
            self.anthropic_api_key,
            self.mistral_api_key,
            self.google_api_key,
            self.cohere_api_key,
            self.groq_api_key,
            self.together_api_key,
        ]

        configured_providers = [p for p in providers if p and not p.startswith("your-") and not p.endswith("-here")]

        if not configured_providers:
            raise ValueError(
                "At least one LLM provider API key must be configured. "
                "Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, MISTRAL_API_KEY, "
                "GOOGLE_API_KEY, COHERE_API_KEY, GROQ_API_KEY, TOGETHER_API_KEY"
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
    cascade_max_levels: int = Field(default=3, env="CASCADE_MAX_LEVELS")
    cascade_quality_threshold: float = Field(default=0.7, env="CASCADE_QUALITY_THRESHOLD")

    # Hybrid router settings
    hybrid_strategy_weights: str = Field(
        default="heuristic:0.4,semantic:0.4,cascade:0.2", env="HYBRID_STRATEGY_WEIGHTS"
    )
    hybrid_consensus_threshold: float = Field(default=0.6, env="HYBRID_CONSENSUS_THRESHOLD")

    @field_validator("hybrid_strategy_weights", mode="before")
    def parse_strategy_weights(cls, v) -> None:
        if isinstance(v, dict):
            # Convert dict back to string format for storage
            return ",".join([f"{k}:{v}" for k, v in v.items()])
        elif isinstance(v, str):
            return v
        return "heuristic:0.4,semantic:0.4,cascade:0.2"

    def get_strategy_weights_dict(self) -> dict[str, float]:
        """Parse strategy weights string into dictionary."""
        weights = {}
        for pair in self.hybrid_strategy_weights.split(","):
            if ":" in pair:
                strategy, weight = pair.split(":", 1)
                weights[strategy.strip()] = float(weight.strip())
        return weights

    @field_validator("default_strategy", mode="before")
    def validate_strategy(cls, v) -> None:
        valid_strategies = ["heuristic", "semantic", "cascade", "hybrid"]
        if v not in valid_strategies:
            raise ValueError(f"Invalid routing strategy. Must be one of: {valid_strategies}")
        return v

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class CacheConfig(BaseSettings):
    """Configuration for caching."""

    enabled: bool = Field(default=True, env="CACHE_ENABLED")
    backend: str = Field(default="memory", env="CACHE_BACKEND")  # memory, redis
    default_ttl: int = Field(default=3600, env="CACHE_DEFAULT_TTL")

    # Memory cache settings
    memory_max_size: int = Field(default=1000, env="MEMORY_CACHE_MAX_SIZE")
    memory_max_memory_mb: int | None = Field(default=None, env="MEMORY_CACHE_MAX_MEMORY_MB")

    # Redis cache settings
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_key_prefix: str = Field(default="modelmuxer:", env="REDIS_KEY_PREFIX")
    redis_compression: bool = Field(default=True, env="REDIS_COMPRESSION")

    @field_validator("backend", mode="before")
    def validate_backend(cls, v) -> None:
        valid_backends = ["memory", "redis"]
        if v not in valid_backends:
            raise ValueError(f"Invalid cache backend. Must be one of: {valid_backends}")
        return v

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class AuthConfig(BaseSettings):
    """Configuration for authentication."""

    enabled: bool = Field(default=True, env="AUTH_ENABLED")
    methods: str = Field(default="api_key", env="AUTH_METHODS")

    # API key authentication
    api_keys: str = Field(default="", env="API_KEYS")

    # JWT authentication
    jwt_secret: str = Field(default="", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiry: int = Field(default=3600, env="JWT_EXPIRY")

    # Security settings
    require_https: bool = Field(default=False, env="REQUIRE_HTTPS")
    allowed_origins: str = Field(default="*", env="ALLOWED_ORIGINS")

    @field_validator("api_keys", mode="before")
    def parse_api_keys(cls, v) -> None:
        if isinstance(v, list):
            return ",".join(v)
        return v if isinstance(v, str) else ""

    @field_validator("methods", mode="before")
    def parse_auth_methods(cls, v) -> None:
        if isinstance(v, list):
            return ",".join(v)
        return v if isinstance(v, str) else "api_key"

    @field_validator("allowed_origins", mode="before")
    def parse_allowed_origins(cls, v) -> None:
        if isinstance(v, list):
            return ",".join(v)
        return v if isinstance(v, str) else "*"

    @field_validator("jwt_secret")
    def validate_jwt_secret(cls, v) -> None:
        """Validate JWT secret is set and secure."""
        import os

        # Allow empty JWT secret in test environments
        if os.getenv("TESTING") == "true" or "pytest" in os.getenv("_", ""):
            return v or "test-jwt-secret-for-testing-only"

        if not v:
            raise ValueError(
                "JWT_SECRET_KEY environment variable is required. "
                "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )
        if len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters long for security")
        return v

    def get_api_keys_list(self) -> list[str]:
        """Parse API keys string into list."""
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]

    def get_methods_list(self) -> list[str]:
        """Parse auth methods string into list."""
        return [method.strip() for method in self.methods.split(",") if method.strip()]

    def get_allowed_origins_list(self) -> list[str]:
        """Parse allowed origins string into list."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class RateLimitConfig(BaseSettings):
    """Configuration for rate limiting."""

    enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    algorithm: str = Field(default="sliding_window", env="RATE_LIMIT_ALGORITHM")

    # Default limits
    requests_per_second: int = Field(default=10, env="RATE_LIMIT_REQUESTS_PER_SECOND")
    requests_per_minute: int = Field(default=60, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    requests_per_hour: int = Field(default=1000, env="RATE_LIMIT_REQUESTS_PER_HOUR")
    burst_size: int = Field(default=20, env="RATE_LIMIT_BURST_SIZE")

    # Global limits
    global_enabled: bool = Field(default=True, env="GLOBAL_RATE_LIMIT_ENABLED")
    global_requests_per_second: int = Field(default=1000, env="GLOBAL_RATE_LIMIT_RPS")
    global_requests_per_minute: int = Field(default=10000, env="GLOBAL_RATE_LIMIT_RPM")

    # Adaptive throttling
    adaptive_enabled: bool = Field(default=False, env="ADAPTIVE_THROTTLING_ENABLED")
    system_load_threshold: float = Field(default=0.8, env="SYSTEM_LOAD_THRESHOLD")

    @field_validator("algorithm", mode="before")
    def validate_algorithm(cls, v) -> None:
        valid_algorithms = ["token_bucket", "sliding_window", "fixed_window"]
        if v not in valid_algorithms:
            raise ValueError(f"Invalid rate limit algorithm. Must be one of: {valid_algorithms}")
        return v

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class MonitoringConfig(BaseSettings):
    """Configuration for monitoring and observability."""

    enabled: bool = Field(default=True, env="MONITORING_ENABLED")

    # Prometheus metrics
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")

    # Health checks
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")

    # Performance tracking
    track_performance: bool = Field(default=True, env="TRACK_PERFORMANCE")
    slow_request_threshold: float = Field(default=5.0, env="SLOW_REQUEST_THRESHOLD")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class LoggingConfig(BaseSettings):
    """Configuration for logging."""

    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="json", env="LOG_FORMAT")

    # Request/response logging
    log_requests: bool = Field(default=True, env="LOG_REQUESTS")
    log_responses: bool = Field(default=True, env="LOG_RESPONSES")
    log_request_body: bool = Field(default=False, env="LOG_REQUEST_BODY")
    log_response_body: bool = Field(default=False, env="LOG_RESPONSE_BODY")
    log_headers: bool = Field(default=False, env="LOG_HEADERS")

    # Security and privacy
    sanitize_sensitive_data: bool = Field(default=True, env="SANITIZE_SENSITIVE_DATA")

    # Audit logging
    audit_enabled: bool = Field(default=True, env="AUDIT_LOGGING_ENABLED")

    @field_validator("level", mode="before")
    def validate_log_level(cls, v) -> None:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class ClassificationConfig(BaseSettings):
    """Configuration for ML-based classification."""

    enabled: bool = Field(default=True, env="CLASSIFICATION_ENABLED")

    # Embedding settings
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_cache_enabled: bool = Field(default=True, env="EMBEDDING_CACHE_ENABLED")
    embedding_cache_dir: str | None = Field(default=None, env="EMBEDDING_CACHE_DIR")

    # Classification settings
    confidence_threshold: float = Field(default=0.6, env="CLASSIFICATION_CONFIDENCE_THRESHOLD")
    max_history_size: int = Field(default=1000, env="CLASSIFICATION_HISTORY_SIZE")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class ModelMuxerConfig:
    """Main configuration class combining all sub-configurations."""

    def __init__(self):
        # Server settings from environment
        # Note: 0.0.0.0 binding is intentional for container deployment
        self.host = os.getenv("HOST", "0.0.0.0")  # nosec B104
        self.port = int(os.getenv("PORT", "8000"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"

        # Initialize sub-configurations
        self.providers = ProviderConfig()
        self.routing = RoutingConfig()
        self.cache = CacheConfig()
        self.auth = AuthConfig()
        self.rate_limit = RateLimitConfig()
        self.monitoring = MonitoringConfig()
        self.logging = LoggingConfig()
        self.classification = ClassificationConfig()

        # Legacy compatibility attributes
        self.code_detection_threshold = float(os.getenv("CODE_DETECTION_THRESHOLD", "0.2"))
        self.complexity_threshold = float(os.getenv("COMPLEXITY_THRESHOLD", "0.2"))
        self.simple_query_threshold = float(os.getenv("SIMPLE_QUERY_THRESHOLD", "0.3"))
        self.simple_query_max_length = int(os.getenv("SIMPLE_QUERY_MAX_LENGTH", "100"))
        self.max_tokens_default = int(os.getenv("MAX_TOKENS_DEFAULT", "1000"))

    def get_allowed_api_keys(self) -> list[str]:
        """Get list of allowed API keys from auth configuration."""
        return self.auth.get_api_keys_list()

    def get_provider_pricing(self) -> dict[str, dict[str, float]]:
        """Get provider pricing information for cost calculation."""
        return {
            "openai": {
                "gpt-4o": {"input": 0.005, "output": 0.015},
                "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
                "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            },
            "anthropic": {
                "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
                "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
                "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            },
            "mistral": {
                "mistral-small": {"input": 0.0002, "output": 0.0006},
                "mistral-medium": {"input": 0.0027, "output": 0.0081},
                "mistral-large": {"input": 0.008, "output": 0.024},
            },
            "google": {
                "gemini-pro": {"input": 0.0005, "output": 0.0015},
                "gemini-pro-vision": {"input": 0.0005, "output": 0.0015},
            },
            "groq": {
                "llama2-70b-4096": {"input": 0.0007, "output": 0.0008},
                "mixtral-8x7b-32768": {"input": 0.0002, "output": 0.0002},
            },
        }


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
