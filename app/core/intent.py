# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Routing Mind — lightweight intent classifier.

Provides {label, confidence, signals} for routing. Uses heuristics by default,
optionally a cheap LLM (stubbed) when available. Deterministic in test mode.
"""

from __future__ import annotations

from typing import Any

from app.models import ChatMessage
from app.settings import settings
from .features import extract_features


INTENT_LABELS = {
    "chat_lite",
    "deep_reason",
    "code_gen",
    "json_extract",
    "translation",
    "vision",
    "safety_risk",
}


def _heuristic_classify(text: str) -> tuple[str, float, dict[str, Any]]:
    f = extract_features(text)

    # Priority ordering for mutually exclusive labels
    if f["safety_hint_count"] > 0:
        return "safety_risk", 0.95, f
    if f["signals"]["vision"]:
        return "vision", 0.85, f
    if f["signals"]["translation"]:
        return "translation", 0.9, f
    if f["signals"]["json"]:
        # Favor json_extract when explicit JSON/tool signals present
        base = 0.8 + min(0.15, 0.05 * f["json_hint_count"])
        return "json_extract", min(base, 0.95), f
    if f["signals"]["code"] or f["has_inline_code"] or f["has_code_fence"]:
        base = 0.8 + min(0.15, 0.05 * (f["code_fence_count"] + f["programming_keywords_count"]))
        return "code_gen", min(base, 0.95), f

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
        return {"label": label, "confidence": round(float(conf), 3), "signals": feats, "method": "heuristic"}

    # Cheap LLM adapter path (optional, stubbed for now)
    # Keep deterministic fallback even when LLM not available
    try:
        # Placeholder: integrate with adapters (e.g., openai/anthropic/mistral) if desired
        # For Phase 1, rely on heuristics to avoid network and preserve determinism.
        label, conf, feats = _heuristic_classify(text)
        return {"label": label, "confidence": round(float(conf), 3), "signals": feats, "method": "heuristic"}
    except Exception:
        label, conf, feats = _heuristic_classify(text)
        return {"label": label, "confidence": round(float(conf), 3), "signals": feats, "method": "heuristic_fallback"}


__all__ = ["classify_intent", "INTENT_LABELS"]
