"""
Centralized application settings for ModelMuxer.

This module provides a typed configuration system using Pydantic BaseSettings.
It organizes settings into logical groups and loads values from the environment
and an optional .env file at the repository root.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, AnyUrl, Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIKeysSettings(BaseSettings):
    """API keys and provider configuration.

    Fields are loaded from environment variables and support common aliases
    to remain backward compatible (e.g., OPENAI_API_KEY).
    """

    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for accessing OpenAI models.",
        validation_alias=AliasChoices("OPENAI_API_KEY", "API_OPENAI_API_KEY"),
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key for Claude models.",
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "API_ANTHROPIC_API_KEY"),
    )
    mistral_api_key: str | None = Field(
        default=None,
        description="Mistral API key for Mistral models.",
        validation_alias=AliasChoices("MISTRAL_API_KEY", "API_MISTRAL_API_KEY"),
    )
    groq_api_key: str | None = Field(
        default=None,
        description="Groq API key for Groq models.",
        validation_alias=AliasChoices("GROQ_API_KEY", "API_GROQ_API_KEY"),
    )
    litellm_base_url: HttpUrl | None = Field(
        default=None,
        description="Base URL for LiteLLM proxy when used as a provider.",
        validation_alias=AliasChoices("LITELLM_BASE_URL", "API_LITELLM_BASE_URL"),
    )
    litellm_api_key: str | None = Field(
        default=None,
        description="API key for LiteLLM proxy if required by the proxy.",
        validation_alias=AliasChoices("LITELLM_API_KEY", "API_LITELLM_API_KEY"),
    )
    api_keys: list[str] = Field(
        default_factory=list,
        description="Comma-separated list of allowed API keys for the built-in API key auth.",
        validation_alias=AliasChoices("API_KEYS", "AUTH_API_KEYS"),
    )

    @field_validator("api_keys", mode="before")
    @classmethod
    def _parse_api_keys(cls, value):  # type: ignore[override]
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [key.strip() for key in value.split(",") if key.strip()]
        return []


class DatabaseSettings(BaseSettings):
    """Database configuration settings.

    Supports SQLite (by default) and other SQLAlchemy-compatible URLs.
    """

    database_url: str = Field(
        default="sqlite:///./modelmuxer.db",
        description="SQLAlchemy-style database URL. Defaults to local SQLite.",
        validation_alias=AliasChoices("DATABASE_URL", "DB_URL"),
    )

    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, value: str) -> str:  # type: ignore[override]
        if not isinstance(value, str) or "://" not in value:
            raise ValueError("DATABASE_URL must be a valid URL (e.g., sqlite:///file.db)")
        return value


class RedisSettings(BaseSettings):
    """Redis cache configuration."""

    url: AnyUrl | None = Field(
        default=None,
        description="Redis connection URL (e.g., redis://localhost:6379 or rediss://).",
        validation_alias=AliasChoices("REDIS_URL", "CACHE_REDIS_URL"),
    )
    db: int = Field(
        default=0,
        description="Redis logical database index (non-negative integer).",
        validation_alias=AliasChoices("REDIS_DB", "CACHE_REDIS_DB"),
    )
    tls_enabled: bool = Field(
        default=False,
        description="Enable TLS for Redis connections (use with rediss:// URLs).",
        validation_alias=AliasChoices("REDIS_TLS", "CACHE_REDIS_TLS"),
    )

    @field_validator("db")
    @classmethod
    def _validate_db(cls, value: int) -> int:  # type: ignore[override]
        if value < 0:
            raise ValueError("REDIS_DB must be >= 0")
        return value


class ObservabilitySettings(BaseSettings):
    """Observability and telemetry settings."""

    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:8080",
            "https://modelmuxer.com",
        ],
        description="List of allowed CORS origins (comma-separated).",
        validation_alias=AliasChoices("CORS_ORIGINS", "OBS_CORS_ORIGINS"),
    )
    log_level: str = Field(
        default="info",
        description="Log level for application logs.",
        validation_alias=AliasChoices("LOG_LEVEL", "OBS_LOG_LEVEL"),
    )
    otel_exporter_otlp_endpoint: AnyUrl | None = Field(
        default=None,
        description="OpenTelemetry OTLP endpoint for exporting traces/metrics.",
        validation_alias=AliasChoices("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_ENDPOINT"),
    )
    prometheus_enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics endpoint.",
        validation_alias=AliasChoices("PROMETHEUS_ENABLED", "OBS_PROMETHEUS_ENABLED"),
    )
    sentry_dsn: AnyUrl | None = Field(
        default=None,
        description="Sentry DSN for error reporting.",
        validation_alias=AliasChoices("SENTRY_DSN", "OBS_SENTRY_DSN"),
    )
    enable_tracing: bool = Field(
        default=True,
        description="Enable OpenTelemetry tracing.",
        validation_alias=AliasChoices("ENABLE_TRACING", "OBS_ENABLE_TRACING"),
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics exposition.",
        validation_alias=AliasChoices("ENABLE_METRICS", "OBS_ENABLE_METRICS"),
    )
    sampling_ratio: float = Field(
        default=1.0,
        description="Trace sampling ratio between 0.0 and 1.0.",
        validation_alias=AliasChoices("OTEL_SAMPLING_RATIO", "OBS_SAMPLING_RATIO"),
    )
    prom_metrics_path: str = Field(
        default="/metrics/prometheus",
        description="Path where Prometheus metrics are exposed.",
        validation_alias=AliasChoices("PROM_METRICS_PATH", "OBS_PROM_METRICS_PATH"),
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value):  # type: ignore[override]
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return []

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value):  # type: ignore[override]
        if isinstance(value, str):
            return value.lower()
        return value

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        env_parse_none_str=None,
    )


class FeatureFlagsSettings(BaseSettings):
    """Feature flag toggles and deployment mode."""

    mode: Literal["basic", "enhanced", "production"] = Field(
        default="basic",
        description="Deployment mode: basic, enhanced, or production.",
        validation_alias=AliasChoices("MODELMUXER_MODE", "APP_MODE"),
    )
    auth_enabled: bool = Field(
        default=True,
        description="Enable authentication middleware and API key checks.",
        validation_alias=AliasChoices("AUTH_ENABLED", "FEATURE_AUTH_ENABLED"),
    )
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable simple in-memory rate limiting (use Redis in production).",
        validation_alias=AliasChoices("RATE_LIMIT_ENABLED", "FEATURE_RATE_LIMIT_ENABLED"),
    )
    monitoring_enabled: bool = Field(
        default=True,
        description="Enable health checks and metrics collection.",
        validation_alias=AliasChoices("MONITORING_ENABLED", "FEATURE_MONITORING_ENABLED"),
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable caching features when available.",
        validation_alias=AliasChoices("CACHE_ENABLED", "FEATURE_CACHE_ENABLED"),
    )
    enable_semantic_routing: bool = Field(
        default=True,
        description="Enable semantic routing components when available.",
        validation_alias=AliasChoices("ENABLE_SEMANTIC_ROUTING", "FEATURE_ENABLE_SEMANTIC_ROUTING"),
    )
    enable_cascade_routing: bool = Field(
        default=True,
        description="Enable cascade routing strategy.",
        validation_alias=AliasChoices("ENABLE_CASCADE_ROUTING", "FEATURE_ENABLE_CASCADE_ROUTING"),
    )
    enable_litellm: bool = Field(
        default=True,
        description="Enable LiteLLM provider integration if configured.",
        validation_alias=AliasChoices("ENABLE_LITELLM", "FEATURE_ENABLE_LITELLM"),
    )
    provider_adapters_enabled: bool = Field(
        default=False,
        description="Enable the new provider adapters invocation path (feature-gated).",
        validation_alias=AliasChoices("PROVIDER_ADAPTERS_ENABLED"),
    )
    redact_pii: bool = Field(
        default=True,
        description="Enable PII redaction in policy enforcement.",
        validation_alias=AliasChoices("FEATURES_REDACT_PII", "REDACT_PII"),
    )
    enable_pii_ner: bool = Field(
        default=False,
        description="Enable optional NER-based PII detection (stub).",
        validation_alias=AliasChoices("FEATURES_ENABLE_PII_NER"),
    )
    test_mode: bool = Field(
        default=False,
        description="Enable test mode to disable external calls and use stubs.",
        validation_alias=AliasChoices("TEST_MODE"),
    )


class PolicySettings(BaseSettings):
    """Policy configuration for allow/deny and jailbreak detection."""

    model_allow: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Per-tenant allow list for models.",
        validation_alias=AliasChoices("POLICY_MODEL_ALLOW"),
    )
    model_deny: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Per-tenant deny list for models.",
        validation_alias=AliasChoices("POLICY_MODEL_DENY"),
    )
    region_allow: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Per-tenant allow list for regions.",
        validation_alias=AliasChoices("POLICY_REGION_ALLOW"),
    )
    region_deny: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Per-tenant deny list for regions.",
        validation_alias=AliasChoices("POLICY_REGION_DENY"),
    )
    enable_jailbreak_detection: bool = Field(
        default=True,
        description="Enable jailbreak detection patterns.",
        validation_alias=AliasChoices("POLICY_ENABLE_JAILBREAK_DETECTION"),
    )
    jailbreak_patterns_path: str = Field(
        default="app/policy/patterns/jailbreak.txt",
        description="Path to jailbreak patterns file.",
        validation_alias=AliasChoices("POLICY_JAILBREAK_PATTERNS_PATH"),
    )
    extra_pii_regex: list[str] = Field(
        default_factory=list,
        description="Additional regex patterns for PII detection.",
        validation_alias=AliasChoices("POLICY_EXTRA_PII_REGEX"),
    )


class ServerSettings(BaseSettings):
    """Server runtime parameters."""

    host: str = Field(
        default="0.0.0.0",
        description="Server host to bind to (use 0.0.0.0 in containers).",
        validation_alias=AliasChoices("HOST", "SERVER_HOST"),
    )
    port: int = Field(
        default=8000,
        description="Server port to listen on (must be > 0).",
        validation_alias=AliasChoices("PORT", "SERVER_PORT"),
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode with verbose logging.",
        validation_alias=AliasChoices("DEBUG", "SERVER_DEBUG"),
    )

    @field_validator("port")
    @classmethod
    def _validate_port(cls, value: int) -> int:  # type: ignore[override]
        if value <= 0:
            raise ValueError("PORT must be > 0")
        return value


class RouterSettings(BaseSettings):
    """Router behavior and model selection settings."""

    default_model: str = Field(
        default="gpt-3.5-turbo",
        description="Default model to use when a specific model is not provided.",
        validation_alias=AliasChoices("DEFAULT_MODEL", "ROUTER_DEFAULT_MODEL"),
    )
    max_tokens_default: int = Field(
        default=1000,
        description="Default maximum output tokens when not specified in a request.",
        validation_alias=AliasChoices("MAX_TOKENS_DEFAULT", "ROUTER_MAX_TOKENS_DEFAULT"),
    )
    temperature_default: float = Field(
        default=0.7,
        description="Default sampling temperature to use when not provided.",
        validation_alias=AliasChoices("TEMPERATURE_DEFAULT", "ROUTER_TEMPERATURE_DEFAULT"),
    )
    code_detection_threshold: float = Field(
        default=0.2,
        description="Threshold for detecting code-like content in prompts.",
        validation_alias=AliasChoices("CODE_DETECTION_THRESHOLD", "ROUTER_CODE_THRESHOLD"),
    )
    complexity_threshold: float = Field(
        default=0.2,
        description="Threshold for detecting complex reasoning prompts.",
        validation_alias=AliasChoices("COMPLEXITY_THRESHOLD", "ROUTER_COMPLEXITY_THRESHOLD"),
    )
    simple_query_max_length: int = Field(
        default=100,
        description="Maximum length for simple queries used in routing heuristics.",
        validation_alias=AliasChoices("SIMPLE_QUERY_MAX_LENGTH", "ROUTER_SIMPLE_QUERY_MAX_LENGTH"),
    )

    # Intent classifier controls
    intent_classifier_enabled: bool = Field(
        default=True,
        description="Enable Routing Mind intent classifier (cheap LLM + heuristics).",
        validation_alias=AliasChoices("ROUTER_INTENT_CLASSIFIER_ENABLED"),
    )
    intent_low_confidence: float = Field(
        default=0.4,
        description="Threshold below which results are treated as low-confidence. "
        "Chosen as 0.4 (vs suggested 0.55) for MVP to be more permissive "
        "and allow more requests to be classified rather than defaulting to unknown.",
        validation_alias=AliasChoices("INTENT_LOW_CONFIDENCE"),
    )
    intent_min_conf_for_direct: float = Field(
        default=0.7,
        description="Min confidence to allow direct routing decisions from intent. "
        "Chosen as 0.7 (vs suggested 0.72) for MVP to balance accuracy "
        "with practical routing decisions. Can be tuned based on production performance.",
        validation_alias=AliasChoices("INTENT_MIN_CONF_FOR_DIRECT"),
    )

    @field_validator("max_tokens_default", "simple_query_max_length")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:  # type: ignore[override]
        if value <= 0:
            raise ValueError("Value must be > 0")
        return value


class ProviderPricingSettings(BaseSettings):
    """Per-provider pricing configuration (per million tokens)."""

    openai_gpt4o_input_price: float = Field(
        default=0.005,
        description="OpenAI GPT-4o input price per million tokens.",
        validation_alias=AliasChoices("OPENAI_GPT4O_INPUT_PRICE"),
    )
    openai_gpt4o_output_price: float = Field(
        default=0.015,
        description="OpenAI GPT-4o output price per million tokens.",
        validation_alias=AliasChoices("OPENAI_GPT4O_OUTPUT_PRICE"),
    )
    openai_gpt4o_mini_input_price: float = Field(
        default=0.00015,
        description="OpenAI GPT-4o-mini input price per million tokens.",
        validation_alias=AliasChoices("OPENAI_GPT4O_MINI_INPUT_PRICE"),
    )
    openai_gpt4o_mini_output_price: float = Field(
        default=0.0006,
        description="OpenAI GPT-4o-mini output price per million tokens.",
        validation_alias=AliasChoices("OPENAI_GPT4O_MINI_OUTPUT_PRICE"),
    )
    openai_gpt35_input_price: float = Field(
        default=0.0005,
        description="OpenAI GPT-3.5-turbo input price per million tokens.",
        validation_alias=AliasChoices("OPENAI_GPT35_INPUT_PRICE"),
    )
    openai_gpt35_output_price: float = Field(
        default=0.0015,
        description="OpenAI GPT-3.5-turbo output price per million tokens.",
        validation_alias=AliasChoices("OPENAI_GPT35_OUTPUT_PRICE"),
    )
    anthropic_sonnet_input_price: float = Field(
        default=0.003,
        description="Anthropic Claude-3 Sonnet input price per million tokens.",
        validation_alias=AliasChoices("ANTHROPIC_SONNET_INPUT_PRICE"),
    )
    anthropic_sonnet_output_price: float = Field(
        default=0.015,
        description="Anthropic Claude-3 Sonnet output price per million tokens.",
        validation_alias=AliasChoices("ANTHROPIC_SONNET_OUTPUT_PRICE"),
    )
    anthropic_haiku_input_price: float = Field(
        default=0.00025,
        description="Anthropic Claude-3 Haiku input price per million tokens.",
        validation_alias=AliasChoices("ANTHROPIC_HAIKU_INPUT_PRICE"),
    )
    anthropic_haiku_output_price: float = Field(
        default=0.00125,
        description="Anthropic Claude-3 Haiku output price per million tokens.",
        validation_alias=AliasChoices("ANTHROPIC_HAIKU_OUTPUT_PRICE"),
    )
    mistral_small_input_price: float = Field(
        default=0.0002,
        description="Mistral-small input price per million tokens.",
        validation_alias=AliasChoices("MISTRAL_SMALL_INPUT_PRICE"),
    )
    mistral_small_output_price: float = Field(
        default=0.0006,
        description="Mistral-small output price per million tokens.",
        validation_alias=AliasChoices("MISTRAL_SMALL_OUTPUT_PRICE"),
    )


class ProviderAdapterSettings(BaseSettings):
    """Resilience settings for provider adapters (timeouts, retries, circuit breaker)."""

    retry_max_attempts: int = Field(
        default=3,
        description="Maximum number of retry attempts for provider calls.",
        validation_alias=AliasChoices("PROVIDER_RETRY_MAX_ATTEMPTS"),
    )
    retry_base_ms: int = Field(
        default=100,
        description="Base delay in milliseconds for exponential backoff retries.",
        validation_alias=AliasChoices("PROVIDER_RETRY_BASE_MS"),
    )
    timeout_ms: int = Field(
        default=10000,
        description="Per-request timeout in milliseconds for provider calls.",
        validation_alias=AliasChoices("PROVIDER_TIMEOUT_MS"),
    )
    circuit_fail_threshold: int = Field(
        default=5,
        description="Consecutive failures to open the circuit.",
        validation_alias=AliasChoices("PROVIDER_CIRCUIT_FAIL_THRESHOLD"),
    )
    circuit_cooldown_sec: int = Field(
        default=30,
        description="Cool-down seconds before half-open state.",
        validation_alias=AliasChoices("PROVIDER_CIRCUIT_COOLDOWN_SEC"),
    )

    @field_validator(
        "retry_max_attempts",
        "retry_base_ms",
        "timeout_ms",
        "circuit_fail_threshold",
        "circuit_cooldown_sec",
    )
    @classmethod
    def _validate_positive(cls, value: int) -> int:  # type: ignore[override]
        if value <= 0:
            raise ValueError("Value must be > 0")
        return value


class ProviderEndpointsSettings(BaseSettings):
    """Base URLs and per-provider API keys if needed (dup keys are allowed for clarity)."""

    openai_base_url: AnyUrl | None = Field(
        default=None,
        description="OpenAI base URL (optional).",
        validation_alias=AliasChoices("OPENAI_BASE_URL"),
    )
    anthropic_base_url: AnyUrl | None = Field(
        default=None,
        description="Anthropic base URL (optional).",
        validation_alias=AliasChoices("ANTHROPIC_BASE_URL"),
    )
    mistral_base_url: AnyUrl | None = Field(
        default=None,
        description="Mistral base URL (optional).",
        validation_alias=AliasChoices("MISTRAL_BASE_URL"),
    )
    groq_base_url: AnyUrl | None = Field(
        default=None,
        description="Groq base URL (optional).",
        validation_alias=AliasChoices("GROQ_BASE_URL"),
    )


class Settings(BaseSettings):
    """Root settings object combining all configuration groups."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    api: APIKeysSettings = APIKeysSettings()
    db: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    features: FeatureFlagsSettings = FeatureFlagsSettings()
    server: ServerSettings = ServerSettings()
    router: RouterSettings = RouterSettings()
    pricing: ProviderPricingSettings = ProviderPricingSettings()
    providers: ProviderAdapterSettings = ProviderAdapterSettings()
    endpoints: ProviderEndpointsSettings = ProviderEndpointsSettings()
    policy: PolicySettings = PolicySettings()


# Singleton settings instance used across the application
settings = Settings()


def get_provider_pricing() -> dict[str, dict[str, dict[str, float]]]:
    """Helper to return pricing map in the legacy structure used by cost tracker and router."""
    p = settings.pricing
    return {
        "openai": {
            "gpt-4o": {"input": p.openai_gpt4o_input_price, "output": p.openai_gpt4o_output_price},
            "gpt-4o-mini": {
                "input": p.openai_gpt4o_mini_input_price,
                "output": p.openai_gpt4o_mini_output_price,
            },
            "gpt-3.5-turbo": {
                "input": p.openai_gpt35_input_price,
                "output": p.openai_gpt35_output_price,
            },
        },
        "anthropic": {
            "claude-3-sonnet-20240229": {
                "input": p.anthropic_sonnet_input_price,
                "output": p.anthropic_sonnet_output_price,
            },
            "claude-3-haiku-20240307": {
                "input": p.anthropic_haiku_input_price,
                "output": p.anthropic_haiku_output_price,
            },
        },
        "mistral": {
            "mistral-small-latest": {
                "input": p.mistral_small_input_price,
                "output": p.mistral_small_output_price,
            }
        },
    }
