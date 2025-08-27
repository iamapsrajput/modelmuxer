# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Lightweight feature extraction for intent classification.

Extracts lexical and structural signals from user prompts for routing-time
intent classification. Designed to be deterministic and fast.
"""

from __future__ import annotations

import re
from typing import Any, Dict


CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
JSON_OBJECT_RE = re.compile(r"\{\s*\"[\w-]+\"\s*:\s*[\s\S]*?\}")
SQL_SELECT_RE = re.compile(r"\bSELECT\b[\s\S]*?\bFROM\b", re.IGNORECASE)


PROGRAMMING_KEYWORDS = [
    "def ",
    "class ",
    "function ",
    "import ",
    "from ",
    "return ",
    "lambda",
    "async ",
    "await ",
    "write a",
    "create a",
    "implement",
    "algorithm",
    "code",
    "program",
    "script",
    "api",
    "sql",
    "query",
    "database",
]

JSON_TOOL_HINTS = [
    "JSON:",
    "json:",
    "tool_call",
    "function_call",
    "parameters",
    "schema",
    "extract",
    "parse",
    "data:",
    "structured",
    "format",
]

TRANSLATION_HINTS = [
    "translate",
    "translation",
    "into english",
    "to english",
    "from english",
    "spanish",
    "german",
    "french",
    "hindi",
    "japanese",
    "mandarin",
]

VISION_HINTS = [
    "image",
    "photo",
    "picture",
    "screenshot",
    "ocr",
    "vision",
]

SAFETY_KEYWORDS = [
    "bomb",
    "explosive",
    "make a weapon",
    "harm",
    "kill",
    "suicide",
    "self-harm",
    "credit card",
    "ssn",
    "exploit",
    "sql injection",
]


def _word_count(text: str) -> int:
    return len(text.split())


def extract_features(text: str) -> Dict[str, Any]:
    """Extract routing-time features from text.

    Returns a dict with booleans, counts, and lightweight scores.
    """
    text_l = text.lower()

    code_fences = CODE_FENCE_RE.findall(text)
    inline_code = INLINE_CODE_RE.findall(text)
    has_json_like = bool(JSON_OBJECT_RE.search(text)) or any(h in text for h in JSON_TOOL_HINTS)
    has_sql = bool(SQL_SELECT_RE.search(text))

    prog_hits = sum(1 for k in PROGRAMMING_KEYWORDS if k in text_l)
    json_hits = sum(1 for k in JSON_TOOL_HINTS if k in text_l)
    translation_hits = sum(1 for k in TRANSLATION_HINTS if k in text_l)
    vision_hits = sum(1 for k in VISION_HINTS if k in text_l)
    safety_hits = sum(1 for k in SAFETY_KEYWORDS if k in text_l)

    token_len_est = max(_word_count(text) * 1.3, len(text) / 4)

    return {
        "token_length_est": float(token_len_est),
        "has_code_fence": bool(code_fences),
        "code_fence_count": len(code_fences),
        "has_inline_code": bool(inline_code),
        "inline_code_count": len(inline_code),
        "has_programming_keywords": prog_hits > 0,
        "programming_keywords_count": prog_hits,
        "has_json": has_json_like,
        "json_hint_count": json_hits,
        "has_sql": has_sql,
        "translation_hint_count": translation_hits,
        "vision_hint_count": vision_hits,
        "safety_hint_count": safety_hits,
        "signals": {
            "json": has_json_like or json_hits > 0,
            "code": bool(code_fences) or prog_hits > 0,
            "translation": translation_hits > 0,
            "vision": vision_hits > 0,
            "safety": safety_hits > 0,
        },
    }


__all__ = ["extract_features"]
