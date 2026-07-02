# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Routing Mind — lightweight intent classifier.

Provides {label, confidence, signals} for routing. Uses heuristics by default,
optionally a cheap LLM when available. Deterministic in test mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from app.models import ChatMessage
from app.settings import settings

from .features import extract_features

logger = logging.getLogger(__name__)

INTENT_LABELS = {
    "chat_lite",
    "deep_reason",
    "code_gen",
    "json_extract",
    "translation",
    "vision",
    "safety_risk",
}

INTENT_CLASSIFIER_SYSTEM_PROMPT = """You classify user prompts for an LLM router.
Respond with JSON only, no markdown, using this schema:
{"label":"<one of chat_lite, deep_reason, code_gen, json_extract, translation, vision, safety_risk>","confidence":0.0}

Use safety_risk for harmful or policy-violating requests. Use code_gen for programming tasks.
Use deep_reason for analysis, explanation, or multi-step reasoning. Use chat_lite for general chat."""


def _heuristic_classify(text: str) -> tuple[str, float, dict[str, Any]]:
    f = extract_features(text)

    # Priority ordering for mutually exclusive labels
    if f["safety_hint_count"] > 0:
        return "safety_risk", 0.95, f
    if f["signals"]["vision"]:
        return "vision", 0.85, f

    # Check for code BEFORE translation to avoid false positives
    # Many code examples contain " to " or " in " which shouldn't be treated as translation
    if f["signals"]["code"] or f["has_inline_code"] or f["has_code_fence"]:
        base = 0.8 + min(0.15, 0.05 * (f["code_fence_count"] + f["programming_keywords_count"]))
        return "code_gen", min(base, 0.95), f

    # Only classify as translation if there are strong translation signals
    # and no conflicting code signals
    if f["signals"]["translation"] and f["translation_hint_count"] >= 2:
        return "translation", 0.9, f

    if f["signals"]["json"]:
        # Favor json_extract when explicit JSON/tool signals present
        base = 0.8 + min(0.15, 0.05 * f["json_hint_count"])
        return "json_extract", min(base, 0.95), f

    # Deep reasoning if text is long or uses analytical language
    text_l = text.lower()
    deep_terms = [
        "analyze",
        "analysis",
        "explain",
        "reason",
        "step-by-step",
        "compare",
        "evaluate",
        "trade-off",
        "why",
        "complex",
        "detailed",
        "thorough",
        "comprehensive",
    ]
    deep_hits = sum(1 for t in deep_terms if t in text_l)
    if f["token_length_est"] > 300 or deep_hits >= 1:
        conf = 0.75 + min(0.2, 0.1 * deep_hits)
        return "deep_reason", min(conf, 0.9), f

    # Default small-talk / general chat
    return "chat_lite", 0.7, f


def _parse_intent_model(value: str) -> tuple[str, str] | None:
    normalized = value.strip()
    if ":" in normalized:
        provider, model = normalized.split(":", 1)
    elif "/" in normalized:
        provider, model = normalized.split("/", 1)
    else:
        return None
    if not provider or not model:
        return None
    return provider, model


def _parse_llm_intent_response(text: str) -> tuple[str, float] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    label = payload.get("label")
    confidence = payload.get("confidence")
    if not isinstance(label, str):
        return None
    label = label.strip().lower()
    if label not in INTENT_LABELS:
        return None
    try:
        conf_value = float(confidence)
    except (TypeError, ValueError):
        return None
    return label, max(0.0, min(1.0, conf_value))


async def _cheap_llm_classify(text: str) -> tuple[str, float, dict[str, Any]] | None:
    """Call the configured cheap LLM for intent classification."""
    intent_model = settings.router.intent_model
    if not intent_model:
        return None

    parsed = _parse_intent_model(intent_model)
    if not parsed:
        logger.warning("Invalid ROUTER_INTENT_MODEL format: %s", intent_model)
        return None
    provider, model = parsed

    try:
        from app.providers.registry import get_provider_registry
    except ImportError:
        return None

    adapter = get_provider_registry().get(provider)
    if adapter is None:
        logger.debug("Intent model provider unavailable: %s", provider)
        return None

    truncated = text[:4000]
    messages = [
        ChatMessage(role="system", content=INTENT_CLASSIFIER_SYSTEM_PROMPT),
        ChatMessage(role="user", content=truncated),
    ]
    timeout_s = settings.router.intent_timeout_ms / 1000.0

    try:
        response = await asyncio.wait_for(
            adapter.invoke(
                model=model,
                messages=messages,
                max_tokens=settings.router.intent_max_tokens,
                temperature=0.0,
            ),
            timeout=timeout_s,
        )
    except Exception as exc:
        logger.debug("Cheap LLM intent classification failed: %s", exc)
        return None

    if response.error or not response.output_text:
        logger.debug("Cheap LLM intent classification returned empty/error response")
        return None

    parsed_response = _parse_llm_intent_response(response.output_text)
    if parsed_response is None:
        return None

    label, confidence = parsed_response
    if confidence < settings.router.intent_low_confidence:
        logger.debug(
            "Cheap LLM intent confidence %.3f below threshold %.3f",
            confidence,
            settings.router.intent_low_confidence,
        )
        return None

    feats = extract_features(text)
    return label, confidence, feats


async def classify_intent(messages: list[ChatMessage]) -> dict[str, Any]:
    """Classify intent for a list of chat messages.

    Returns a dict with: label, confidence, signals, method.
    Respects settings.router.intent_classifier_enabled and settings.features.test_mode.
    """
    if not settings.router.intent_classifier_enabled:
        return {"label": "unknown", "confidence": 0.0, "signals": {}, "method": "disabled"}

    # Join all message contents
    text = "\n".join([m.content for m in messages if m.content])

    # In test mode, force deterministic heuristics only
    if settings.features.test_mode:
        label, conf, feats = _heuristic_classify(text)
        return {
            "label": label,
            "confidence": round(float(conf), 3),
            "signals": feats,
            "method": "heuristic",
        }

    try:
        llm_result = await _cheap_llm_classify(text)
        if llm_result is not None:
            label, conf, feats = llm_result
            return {
                "label": label,
                "confidence": round(float(conf), 3),
                "signals": feats,
                "method": "cheap_llm",
            }
        label, conf, feats = _heuristic_classify(text)
        return {
            "label": label,
            "confidence": round(float(conf), 3),
            "signals": feats,
            "method": "heuristic",
        }
    except Exception:
        label, conf, feats = _heuristic_classify(text)
        return {
            "label": label,
            "confidence": round(float(conf), 3),
            "signals": feats,
            "method": "heuristic_fallback",
        }


__all__ = ["classify_intent", "INTENT_LABELS"]
