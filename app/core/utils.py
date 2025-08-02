# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Shared utilities for ModelMuxer.

This module contains utility functions that are used across the application
for common operations like hashing, token estimation, and input sanitization.
"""

import hashlib
import re
import time
import uuid
from decimal import ROUND_HALF_UP, Decimal

import tiktoken

from ..models import ChatMessage


def generate_request_id() -> str:
    """Generate a unique request ID."""
    timestamp = int(time.time() * 1000)
    random_part = uuid.uuid4().hex[:8]
    return f"req_{timestamp}_{random_part}"


def hash_prompt(messages: list[ChatMessage], algorithm: str = "sha256") -> str:
    """
    Create a hash of the prompt for caching and analytics.

    Args:
        messages: List of chat messages
        algorithm: Hashing algorithm to use

    Returns:
        Hexadecimal hash string
    """
    # Combine all message content
    content = ""
    for msg in messages:
        content += f"{msg.role}:{msg.content}\n"

    # Create hash - only SHA256 for security
    if algorithm == "sha256":
        return hashlib.sha256(content.encode()).hexdigest()
    else:
        # MD5 is deprecated due to security vulnerabilities
        raise ValueError(
            f"Unsupported hashing algorithm: {algorithm}. Only 'sha256' is supported for security reasons."
        )


def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Estimate token count for a given text.

    Args:
        text: Input text
        model: Model name for tokenizer selection

    Returns:
        Estimated token count
    """
    try:
        # Try to get the exact tokenizer for the model
        if "gpt-4" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            # Fallback to cl100k_base for most modern models
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))
    except Exception:
        # Fallback estimation: roughly 4 characters per token
        return len(text) // 4


def format_cost(cost: float, currency: str = "USD") -> str:
    """
    Format cost for display.

    Args:
        cost: Cost value
        currency: Currency code

    Returns:
        Formatted cost string
    """
    if cost < 0.000001:
        return f"<$0.000001 {currency}"
    elif cost < 0.01:
        return f"${cost:.6f} {currency}"
    else:
        return f"${cost:.4f} {currency}"


def sanitize_input(text: str, max_length: int = 100000) -> str:
    """
    Sanitize user input to prevent injection attacks.

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized text

    Raises:
        ValueError: If input is too long
    """
    if len(text) > max_length:
        raise ValueError(f"Input too long. Maximum length is {max_length} characters.")

    # Remove null bytes and control characters (except newlines and tabs)
    sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # Normalize whitespace
    sanitized = re.sub(r"\r\n", "\n", sanitized)
    sanitized = re.sub(r"\r", "\n", sanitized)

    # Remove excessive consecutive newlines
    sanitized = re.sub(r"\n{4,}", "\n\n\n", sanitized)

    return sanitized.strip()


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using Jaccard similarity.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1
    """
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)

    if not union:
        return 0.0

    return len(intersection) / len(union)


def extract_code_blocks(text: str) -> list[dict[str, str]]:
    """
    Extract code blocks from text.

    Args:
        text: Input text

    Returns:
        List of code blocks with language and content
    """
    code_blocks = []

    # Pattern for fenced code blocks
    pattern = r"```(\w+)?\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)

    for language, code in matches:
        code_blocks.append({"language": language or "unknown", "content": code.strip()})

    return code_blocks


def detect_programming_language(text: str) -> str | None:
    """
    Detect programming language from text content.

    Args:
        text: Input text

    Returns:
        Detected language or None
    """
    # Language patterns
    patterns = {
        "python": [
            r"\bdef\s+\w+\s*\(",
            r"\bimport\s+\w+",
            r"\bfrom\s+\w+\s+import",
            r"print\s*\(",
            r'if\s+__name__\s*==\s*["\']__main__["\']',
        ],
        "javascript": [
            r"\bfunction\s+\w+\s*\(",
            r"\bconst\s+\w+\s*=",
            r"\blet\s+\w+\s*=",
            r"console\.log\s*\(",
            r"=>",
        ],
        "java": [
            r"\bpublic\s+class\s+\w+",
            r"\bpublic\s+static\s+void\s+main",
            r"System\.out\.println",
            r"\bprivate\s+\w+\s+\w+",
            r"\bpublic\s+\w+\s+\w+\s*\(",
        ],
        "sql": [
            r"\bSELECT\s+.*\s+FROM\b",
            r"\bINSERT\s+INTO\b",
            r"\bUPDATE\s+.*\s+SET\b",
            r"\bDELETE\s+FROM\b",
            r"\bCREATE\s+TABLE\b",
        ],
    }

    for language, lang_patterns in patterns.items():
        matches = sum(1 for pattern in lang_patterns if re.search(pattern, text, re.IGNORECASE))
        if matches >= 2:  # Require at least 2 pattern matches
            return language

    return None


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add when truncating

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def parse_model_name(full_model_name: str) -> dict[str, str]:
    """
    Parse a full model name into provider and model components.

    Args:
        full_model_name: Full model name (e.g., "openai/gpt-4o")

    Returns:
        Dictionary with provider and model keys
    """
    if "/" in full_model_name:
        provider, model = full_model_name.split("/", 1)
        return {"provider": provider, "model": model}
    else:
        return {"provider": "unknown", "model": full_model_name}


def round_cost(cost: float, decimal_places: int = 6) -> float:
    """
    Round cost to specified decimal places.

    Args:
        cost: Cost value
        decimal_places: Number of decimal places

    Returns:
        Rounded cost
    """
    decimal_cost = Decimal(str(cost))
    rounded = decimal_cost.quantize(Decimal("0." + "0" * decimal_places), rounding=ROUND_HALF_UP)
    return float(rounded)
