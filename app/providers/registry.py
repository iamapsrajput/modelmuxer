from __future__ import annotations

from typing import Dict

from app.settings import settings

# Placeholders for adapter classes that will be implemented
try:
    from .openai import OpenAIAdapter
except Exception:  # pragma: no cover
    OpenAIAdapter = None  # type: ignore[assignment]

try:
    from .anthropic import AnthropicAdapter
except Exception:  # pragma: no cover
    AnthropicAdapter = None  # type: ignore[assignment]

try:
    from .mistral import MistralAdapter
except Exception:  # pragma: no cover
    MistralAdapter = None  # type: ignore[assignment]

try:
    from .groq import GroqAdapter
except Exception:  # pragma: no cover
    GroqAdapter = None  # type: ignore[assignment]


def build_registry() -> Dict[str, object]:
    registry: Dict[str, object] = {}

    if settings.api.openai_api_key and OpenAIAdapter is not None:
        registry["openai"] = OpenAIAdapter(
            api_key=settings.api.openai_api_key,
            base_url=(settings.endpoints.openai_base_url or "https://api.openai.com/v1"),
        )

    if settings.api.anthropic_api_key and AnthropicAdapter is not None:
        registry["anthropic"] = AnthropicAdapter(
            api_key=settings.api.anthropic_api_key,
            base_url=(settings.endpoints.anthropic_base_url or "https://api.anthropic.com"),
        )

    if settings.api.mistral_api_key and MistralAdapter is not None:
        registry["mistral"] = MistralAdapter(
            api_key=settings.api.mistral_api_key,
            base_url=(settings.endpoints.mistral_base_url or "https://api.mistral.ai"),
        )

    if settings.api.litellm_api_key and settings.api.litellm_base_url:
        # Optional: add a LiteLLM proxy adapter when available
        pass

    if settings.api.openai_api_key and GroqAdapter is not None:
        # Groq requires GROQ_API_KEY if present; if missing, skip
        registry["groq"] = GroqAdapter(
            api_key=getattr(settings.api, "groq_api_key", None) or "",
            base_url=(settings.endpoints.groq_base_url or "https://api.groq.com"),
        )

    return registry


PROVIDERS = build_registry()
