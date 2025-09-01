from __future__ import annotations

import logging
from typing import Dict

from app.settings import settings

logger = logging.getLogger(__name__)

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

try:
    from .google import GoogleAdapter
except Exception:  # pragma: no cover
    GoogleAdapter = None  # type: ignore[assignment]

try:
    from .cohere import CohereAdapter
except Exception:  # pragma: no cover
    CohereAdapter = None  # type: ignore[assignment]

try:
    from .together import TogetherAdapter
except Exception:  # pragma: no cover
    TogetherAdapter = None  # type: ignore[assignment]


def build_registry() -> Dict[str, object]:
    # Check if provider adapters are enabled via feature flag
    if not settings.features.provider_adapters_enabled:
        logger.info("Provider adapters are disabled via feature flag")
        return {}

    registry: Dict[str, object] = {}

    if settings.api.openai_api_key and OpenAIAdapter is not None:
        base_url = (
            str(settings.endpoints.openai_base_url)
            if settings.endpoints.openai_base_url
            else "https://api.openai.com/v1"
        )
        registry["openai"] = OpenAIAdapter(
            api_key=settings.api.openai_api_key,
            base_url=base_url,
        )

    if settings.api.anthropic_api_key and AnthropicAdapter is not None:
        base_url = (
            str(settings.endpoints.anthropic_base_url)
            if settings.endpoints.anthropic_base_url
            else "https://api.anthropic.com"
        )
        registry["anthropic"] = AnthropicAdapter(
            api_key=settings.api.anthropic_api_key,
            base_url=base_url,
        )

    if settings.api.mistral_api_key and MistralAdapter is not None:
        base_url = (
            str(settings.endpoints.mistral_base_url)
            if settings.endpoints.mistral_base_url
            else "https://api.mistral.ai"
        )
        registry["mistral"] = MistralAdapter(
            api_key=settings.api.mistral_api_key,
            base_url=base_url,
        )

    if settings.api.groq_api_key and GroqAdapter is not None:
        base_url = (
            str(settings.endpoints.groq_base_url)
            if settings.endpoints.groq_base_url
            else "https://api.groq.com"
        )
        registry["groq"] = GroqAdapter(
            api_key=settings.api.groq_api_key,
            base_url=base_url,
        )

    if settings.api.google_api_key and GoogleAdapter is not None:
        base_url = (
            str(settings.endpoints.google_base_url)
            if settings.endpoints.google_base_url
            else "https://generativelanguage.googleapis.com/v1beta"
        )
        registry["google"] = GoogleAdapter(
            api_key=settings.api.google_api_key,
            base_url=base_url,
        )

    if settings.api.cohere_api_key and CohereAdapter is not None:
        base_url = (
            str(settings.endpoints.cohere_base_url)
            if settings.endpoints.cohere_base_url
            else "https://api.cohere.ai/v1"
        )
        registry["cohere"] = CohereAdapter(
            api_key=settings.api.cohere_api_key,
            base_url=base_url,
        )

    if settings.api.together_api_key and TogetherAdapter is not None:
        base_url = (
            str(settings.endpoints.together_base_url)
            if settings.endpoints.together_base_url
            else "https://api.together.xyz/v1"
        )
        registry["together"] = TogetherAdapter(
            api_key=settings.api.together_api_key,
            base_url=base_url,
        )

    return registry


PROVIDERS = build_registry()


async def refresh_provider_registry() -> None:
    """
    Refresh the provider registry by rebuilding it from current settings.

    This is useful for test mode or when settings change dynamically.
    Uses the LLMProviderAdapter protocol to ensure proper resource cleanup.
    """
    from .base import LLMProviderAdapter

    global PROVIDERS
    # Close existing adapters using protocol
    for provider_name, adapter in PROVIDERS.items():
        if isinstance(adapter, LLMProviderAdapter):
            try:
                await adapter.aclose()
            except Exception as e:
                logger.warning("Failed to close %s during refresh: %s", provider_name, e)
    PROVIDERS = build_registry()


def get_provider_registry() -> Dict[str, object]:
    """
    Get the current provider registry.

    This function provides a central access point for the provider registry
    that can be easily mocked in tests and injected as a dependency.

    In test mode, returns a fresh snapshot to ensure settings changes are reflected.

    Returns:
        Dictionary of provider name to provider instance mappings
    """
    # In test mode, return a fresh snapshot to ensure settings changes are reflected
    if settings.features.test_mode:
        return build_registry()
    return PROVIDERS


async def cleanup_provider_registry() -> None:
    """
    Clean up all provider adapters by closing their HTTP clients.

    This should be called during application shutdown to prevent connection leaks.
    Uses the LLMProviderAdapter protocol to ensure proper resource cleanup.
    """
    from .base import LLMProviderAdapter

    for provider_name, adapter in PROVIDERS.items():
        if isinstance(adapter, LLMProviderAdapter):
            try:
                await adapter.aclose()
                logger.debug("Successfully closed %s adapter", provider_name)
            except Exception as e:
                # Log but don't fail shutdown
                logger.warning("Failed to close %s adapter: %s", provider_name, e)
        else:
            logger.warning(
                "Provider %s does not implement LLMProviderAdapter protocol", provider_name
            )
