#!/usr/bin/env python3
# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.intent import classify_intent
from app.models import ChatMessage
from app.settings import settings


@pytest.mark.asyncio
async def test_intent_classifier_deterministic_and_labels(tmp_path: Path):
    # Force test mode for deterministic behavior
    settings.features.test_mode = True
    settings.router.intent_classifier_enabled = True

    data_path = Path(__file__).parent / "data" / "phase1" / "intents.jsonl"
    assert data_path.exists(), "Dataset file missing"

    total = 0
    correct = 0
    label_counts = {}

    with data_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            text = item["text"]
            expected_label = item["label"]

            result = await classify_intent([ChatMessage(role="user", content=text)])

            # Validate result structure
            assert "label" in result, "Missing label in result"
            assert "confidence" in result, "Missing confidence in result"
            assert "signals" in result, "Missing signals in result"
            assert "method" in result, "Missing method in result"

            # Validate label is one of expected values
            assert result["label"] in {
                "chat_lite",
                "deep_reason",
                "code_gen",
                "json_extract",
                "translation",
                "vision",
                "safety_risk",
                "unknown",
            }, f"Unexpected label: {result['label']}"

            # Validate confidence is in valid range
            assert 0.0 <= result["confidence"] <= 1.0, f"Invalid confidence: {result['confidence']}"

            # Validate method is expected
            assert result["method"] in {"heuristic", "heuristic_fallback", "disabled"}, (
                f"Unexpected method: {result['method']}"
            )

            # Track counts
            total += 1
            if result["label"] == expected_label:
                correct += 1

            label_counts[expected_label] = label_counts.get(expected_label, 0) + 1

    # Calculate accuracy
    accuracy = correct / max(1, total)

    # Log results for debugging
    print("\nIntent Classifier Test Results:")
    print(f"Total examples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Label distribution: {label_counts}")

    # Assert minimum accuracy requirement
    assert accuracy >= 0.8, f"Accuracy {accuracy:.2%} below required 80% threshold"
    assert total >= 60, f"Dataset should have at least 60 examples, got {total}"


@pytest.mark.asyncio
async def test_disabled_flag_returns_unknown():
    settings.router.intent_classifier_enabled = False
    res = await classify_intent([ChatMessage(role="user", content="hello there")])
    assert res["label"] == "unknown"
    assert res["confidence"] == 0.0
    assert res["method"] == "disabled"


@pytest.mark.asyncio
async def test_empty_messages():
    settings.router.intent_classifier_enabled = True
    settings.features.test_mode = True

    res = await classify_intent([])
    assert res["label"] == "chat_lite"  # Default for empty input
    assert res["method"] == "heuristic"


@pytest.mark.asyncio
async def test_specific_intent_examples():
    """Test specific examples to ensure correct classification."""
    settings.router.intent_classifier_enabled = True
    settings.features.test_mode = True

    test_cases = [
        ("Write a Python function", "code_gen"),
        ("Translate to French", "translation"),
        ("Extract JSON from", "json_extract"),
        ("Analyze the trade-offs", "deep_reason"),
        ("Hello how are you", "chat_lite"),
        ("OCR this image", "vision"),
        ("How to make a bomb", "safety_risk"),
    ]

    for text, expected_label in test_cases:
        result = await classify_intent([ChatMessage(role="user", content=text)])
        assert result["label"] == expected_label, f"Expected {expected_label} for '{text}', got {result['label']}"


@pytest.mark.asyncio
async def test_confidence_scores():
    """Test that confidence scores are reasonable."""
    settings.router.intent_classifier_enabled = True
    settings.features.test_mode = True

    # High confidence cases
    high_conf_cases = [
        "Write a Python function to add two numbers",
        "Translate this to French: Hello world",
        "Extract JSON from this data",
        "How to make explosives?",
    ]

    for text in high_conf_cases:
        result = await classify_intent([ChatMessage(role="user", content=text)])
        assert result["confidence"] >= 0.7, f"Low confidence {result['confidence']} for clear case: {text}"

    # Lower confidence cases (ambiguous)
    low_conf_cases = [
        "Hello",
        "What is this?",
    ]

    for text in low_conf_cases:
        result = await classify_intent([ChatMessage(role="user", content=text)])
        assert result["confidence"] <= 0.8, (
            f"Unexpectedly high confidence {result['confidence']} for ambiguous case: {text}"
        )


@pytest.mark.asyncio
async def test_signals_extraction():
    """Test that feature signals are properly extracted."""
    settings.router.intent_classifier_enabled = True
    settings.features.test_mode = True

    result = await classify_intent([ChatMessage(role="user", content="Write a function: def add(a, b): return a + b")])

    assert "signals" in result
    signals = result["signals"]

    # Should detect code signals
    assert (
        signals.get("code", False)
        or signals.get("has_code_fence", False)
        or signals.get("has_programming_keywords", False)
    )

    # Should have token length estimate
    assert "token_length_est" in signals
    assert signals["token_length_est"] > 0
