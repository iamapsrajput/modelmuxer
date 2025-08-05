# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Configuration management using Pydantic Settings.
"""

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # LLM Provider API Keys
    openai_api_key: str = Field(..., description="OpenAI API key")
    anthropic_api_key: str = Field(..., description="Anthropic API key")
    mistral_api_key: str = Field(..., description="Mistral API key")

    # Router Configuration
    default_model: str = Field("gpt-3.5-turbo", description="Default model to use")
    max_tokens_default: int = Field(1000, description="Default max tokens")
    temperature_default: float = Field(0.7, description="Default temperature")

    # Database
    database_url: str = Field("sqlite:///router_data.db", description="Database URL")

    # Security
    api_key_header: str = Field("Authorization", description="API key header name")
    allowed_api_keys: str = Field(default="", env="API_KEYS", description="Comma-separated allowed API keys")

    # Server Configuration
    # Note: 0.0.0.0 binding is intentional for container deployment
    host: str = Field("0.0.0.0", description="Server host")  # nosec B104
    port: int = Field(8000, description="Server port")
    debug: bool = Field(False, description="Debug mode")

    # Cost Limits (USD)
    default_daily_budget: float = Field(10.0, description="Default daily budget per user")
    default_monthly_budget: float = Field(100.0, description="Default monthly budget per user")

    # Routing Configuration
    code_detection_threshold: float = Field(0.2, description="Threshold for code detection")
    complexity_threshold: float = Field(0.2, description="Threshold for complexity detection")
    simple_query_max_length: int = Field(100, description="Max length for simple queries")

    # Provider Pricing (per million tokens)
    openai_gpt4o_input_price: float = Field(0.005, description="GPT-4o input price per million tokens")
    openai_gpt4o_output_price: float = Field(0.015, description="GPT-4o output price per million tokens")
    openai_gpt4o_mini_input_price: float = Field(0.00015, description="GPT-4o-mini input price per million tokens")
    openai_gpt4o_mini_output_price: float = Field(0.0006, description="GPT-4o-mini output price per million tokens")
    openai_gpt35_input_price: float = Field(0.0005, description="GPT-3.5-turbo input price per million tokens")
    openai_gpt35_output_price: float = Field(0.0015, description="GPT-3.5-turbo output price per million tokens")

    anthropic_sonnet_input_price: float = Field(0.003, description="Claude-3-sonnet input price per million tokens")
    anthropic_sonnet_output_price: float = Field(0.015, description="Claude-3-sonnet output price per million tokens")
    anthropic_haiku_input_price: float = Field(0.00025, description="Claude-3-haiku input price per million tokens")
    anthropic_haiku_output_price: float = Field(0.00125, description="Claude-3-haiku output price per million tokens")

    mistral_small_input_price: float = Field(0.0002, description="Mistral-small input price per million tokens")
    mistral_small_output_price: float = Field(0.0006, description="Mistral-small output price per million tokens")

    @validator("allowed_api_keys")
    def parse_api_keys(cls, v) -> None:
        """Parse comma-separated API keys into a list."""
        if isinstance(v, str):
            return [key.strip() for key in v.split(",") if key.strip()]
        return v

    def get_allowed_api_keys(self) -> list[str]:
        """Get list of allowed API keys."""
        if isinstance(self.allowed_api_keys, str):
            return [key.strip() for key in self.allowed_api_keys.split(",") if key.strip()]
        return self.allowed_api_keys

    def get_provider_pricing(self) -> dict:
        """Get pricing information for all providers."""
        return {
            "openai": {
                "gpt-4o": {
                    "input": self.openai_gpt4o_input_price,
                    "output": self.openai_gpt4o_output_price,
                },
                "gpt-4o-mini": {
                    "input": self.openai_gpt4o_mini_input_price,
                    "output": self.openai_gpt4o_mini_output_price,
                },
                "gpt-3.5-turbo": {
                    "input": self.openai_gpt35_input_price,
                    "output": self.openai_gpt35_output_price,
                },
            },
            "anthropic": {
                "claude-3-sonnet-20240229": {
                    "input": self.anthropic_sonnet_input_price,
                    "output": self.anthropic_sonnet_output_price,
                },
                "claude-3-haiku-20240307": {
                    "input": self.anthropic_haiku_input_price,
                    "output": self.anthropic_haiku_output_price,
                },
            },
            "mistral": {
                "mistral-small-latest": {
                    "input": self.mistral_small_input_price,
                    "output": self.mistral_small_output_price,
                }
            },
        }

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


# Global settings instance
settings = Settings()
