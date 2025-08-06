# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Heuristic-based routing implementation.

This module contains the enhanced heuristic router that uses rule-based
logic to determine the optimal provider and model for requests.
"""

import re
from typing import Any

import structlog

from ..core.utils import detect_programming_language, extract_code_blocks
from ..models import ChatMessage
from .base_router import BaseRouter

logger = structlog.get_logger(__name__)


class HeuristicRouter(BaseRouter):
    """
    Enhanced heuristic router with improved pattern matching and rules.

    This router uses a comprehensive set of heuristics to analyze prompts
    and route them to the most appropriate provider and model.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__("heuristic", config)

        # Enhanced pattern definitions
        self.code_patterns = [
            r"```[\s\S]*?```",  # Code blocks
            r"`[^`\n]+`",  # Inline code
            r"\bdef\s+\w+\s*\(",  # Python functions
            r"\bclass\s+\w+\s*[:\(]",  # Class definitions
            r"\bfunction\s+\w+\s*\(",  # JavaScript functions
            r"\bpublic\s+\w+\s+\w+\s*\(",  # Java/C# methods
            r"\bimport\s+\w+",  # Import statements
            r"\bfrom\s+\w+\s+import",  # Python imports
            r"#include\s*<\w+>",  # C/C++ includes
            r"\$\w+\s*=",  # Variable assignments
            r"SELECT\s+.*\s+FROM",  # SQL queries
            r"CREATE\s+TABLE",  # SQL DDL
            r"<\w+[^>]*>.*</\w+>",  # HTML/XML tags
            r'{\s*"[\w":\s,\[\]{}]+\s*}',  # JSON objects
        ]

        # Programming keywords with weights
        self.programming_keywords = {
            "function": 0.8,
            "class": 0.9,
            "import": 0.7,
            "export": 0.6,
            "const": 0.5,
            "let": 0.5,
            "var": 0.5,
            "def": 0.9,
            "return": 0.6,
            "if": 0.3,
            "else": 0.3,
            "elif": 0.4,
            "for": 0.4,
            "while": 0.4,
            "try": 0.5,
            "except": 0.6,
            "public": 0.7,
            "private": 0.7,
            "protected": 0.7,
            "static": 0.6,
            "void": 0.6,
            "int": 0.5,
            "string": 0.5,
            "array": 0.6,
            "list": 0.6,
            "dict": 0.6,
            "object": 0.5,
            "null": 0.4,
            "undefined": 0.5,
            "true": 0.3,
            "false": 0.3,
            "async": 0.7,
            "await": 0.7,
            "promise": 0.7,
            "callback": 0.6,
            "lambda": 0.8,
            "yield": 0.7,
        }

        # Complexity indicators with weights
        self.complexity_keywords = {
            "analyze": 0.9,
            "analysis": 0.9,
            "explain": 0.7,
            "explanation": 0.7,
            "debug": 0.8,
            "debugging": 0.8,
            "reasoning": 0.9,
            "reason": 0.7,
            "complex": 0.8,
            "complicated": 0.8,
            "algorithm": 0.9,
            "algorithms": 0.9,
            "optimize": 0.8,
            "optimization": 0.8,
            "architecture": 0.9,
            "design pattern": 1.0,
            "patterns": 0.7,
            "performance": 0.8,
            "scalability": 0.9,
            "trade-off": 0.8,
            "tradeoffs": 0.8,
            "comparison": 0.7,
            "evaluate": 0.8,
            "assessment": 0.8,
            "review": 0.6,
            "critique": 0.8,
            "detailed": 0.6,
            "comprehensive": 0.8,
            "step-by-step": 0.7,
            "methodology": 0.8,
            "approach": 0.6,
            "strategy": 0.7,
            "framework": 0.8,
            "implementation": 0.7,
            "solution": 0.6,
            "problem-solving": 0.8,
            "troubleshoot": 0.8,
        }

        # Simple query indicators
        self.simple_indicators = {
            "what is": 0.9,
            "who is": 0.9,
            "when is": 0.9,
            "where is": 0.9,
            "how much": 0.8,
            "how many": 0.8,
            "define": 0.8,
            "definition": 0.8,
            "meaning": 0.7,
            "translate": 0.9,
            "translation": 0.9,
            "calculate": 0.7,
            "convert": 0.8,
            "list": 0.6,
            "name": 0.6,
            "tell me": 0.7,
            "show me": 0.7,
        }

        # Creative writing indicators
        self.creative_indicators = {
            "write a story": 1.0,
            "create a poem": 1.0,
            "write a poem": 1.0,
            "creative writing": 1.0,
            "storytelling": 0.9,
            "narrative": 0.8,
            "fiction": 0.8,
            "character": 0.7,
            "plot": 0.8,
            "dialogue": 0.8,
            "marketing copy": 0.9,
            "advertisement": 0.8,
            "social media": 0.7,
            "blog post": 0.7,
            "article": 0.6,
            "essay": 0.6,
        }

        # Model preferences by task type and constraints
        self.model_preferences = {
            "code_generation": [
                ("openai", "gpt-4o", 0.95),
                ("anthropic", "claude-3-5-sonnet", 0.90),
                ("litellm", "gpt-4", 0.85),  # LiteLLM proxy for GPT-4
                ("openai", "gpt-3.5-turbo", 0.75),
                ("litellm", "gpt-3.5-turbo", 0.70),  # LiteLLM proxy fallback
            ],
            "code_review": [
                ("openai", "gpt-4o", 0.95),
                ("anthropic", "claude-3-5-sonnet", 0.90),
                ("litellm", "gpt-4", 0.85),  # LiteLLM proxy for GPT-4
                ("openai", "gpt-3.5-turbo", 0.70),
                ("litellm", "gpt-3.5-turbo", 0.65),  # LiteLLM proxy fallback
            ],
            "complex_analysis": [
                ("anthropic", "claude-3-5-sonnet", 0.95),
                ("openai", "gpt-4o", 0.90),
                ("litellm", "claude-3-sonnet", 0.88),  # LiteLLM proxy for Claude
                ("anthropic", "claude-3-opus", 0.85),
                ("litellm", "gpt-4", 0.80),  # LiteLLM proxy for GPT-4
                ("openai", "gpt-3.5-turbo", 0.65),
                ("litellm", "claude-3-haiku", 0.60),  # LiteLLM proxy for Haiku
            ],
            "simple_qa": [
                ("google", "gemini-1.5-flash", 0.90),
                ("mistral", "mistral-small", 0.85),
                ("litellm", "claude-3-haiku", 0.82),  # LiteLLM proxy for Haiku
                ("anthropic", "claude-3-haiku", 0.80),
                ("litellm", "gpt-3.5-turbo", 0.78),  # LiteLLM proxy for GPT-3.5
                ("openai", "gpt-3.5-turbo", 0.75),
            ],
            "creative_writing": [
                ("openai", "gpt-4o", 0.95),
                ("anthropic", "claude-3-5-sonnet", 0.85),
                ("litellm", "claude-3-sonnet", 0.82),  # LiteLLM proxy for Claude
                ("litellm", "gpt-4", 0.78),  # LiteLLM proxy for GPT-4
                ("openai", "gpt-3.5-turbo", 0.80),
            ],
            "general": [
                ("openai", "gpt-3.5-turbo", 0.85),
                ("litellm", "gpt-3.5-turbo", 0.82),  # LiteLLM proxy for GPT-3.5
                ("anthropic", "claude-3-haiku", 0.80),
                ("litellm", "claude-3-haiku", 0.78),  # LiteLLM proxy for Haiku
                ("google", "gemini-1.5-flash", 0.75),
                ("mistral", "mistral-small", 0.70),
            ],
        }

    async def analyze_prompt(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """Enhanced prompt analysis with weighted scoring."""
        # Combine all message content
        full_text = " ".join([msg.content for msg in messages if msg.content])
        full_text_lower = full_text.lower()

        analysis: dict[str, Any] = {
            "total_length": len(full_text),
            "message_count": len(messages),
            "word_count": len(full_text.split()),
            "has_code": False,
            "code_confidence": 0.0,
            "has_complexity": False,
            "complexity_confidence": 0.0,
            "is_simple": False,
            "simple_confidence": 0.0,
            "is_creative": False,
            "creative_confidence": 0.0,
            "detected_language": None,
            "code_blocks": [],
            "task_type": "general",
            "confidence_score": 0.0,
        }

        # Code detection with weighted scoring
        code_score = 0.0
        code_matches = 0

        for pattern in self.code_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            code_matches += len(matches)
            if matches:
                code_score += 0.3 * len(matches)

        # Programming keyword detection
        for keyword, weight in self.programming_keywords.items():
            if keyword.lower() in full_text_lower:
                code_score += weight * 0.1

        # Detect programming language
        detected_lang = detect_programming_language(full_text)
        if detected_lang:
            analysis["detected_language"] = detected_lang
            code_score += 0.5

        # Extract code blocks
        analysis["code_blocks"] = extract_code_blocks(full_text)
        if analysis["code_blocks"]:
            code_score += 0.4 * len(analysis["code_blocks"])

        analysis["code_confidence"] = min(1.0, code_score)
        analysis["has_code"] = analysis["code_confidence"] > 0.3

        # Complexity detection with weighted scoring
        complexity_score = 0.0
        for keyword, weight in self.complexity_keywords.items():
            if keyword.lower() in full_text_lower:
                complexity_score += weight * 0.1

        analysis["complexity_confidence"] = min(1.0, complexity_score)
        analysis["has_complexity"] = analysis["complexity_confidence"] > 0.5

        # Simple query detection
        simple_score = 0.0
        for indicator, weight in self.simple_indicators.items():
            if indicator.lower() in full_text_lower:
                simple_score += weight * 0.2

        analysis["simple_confidence"] = min(1.0, simple_score)
        analysis["is_simple"] = (
            analysis["simple_confidence"] > 0.3
            and analysis["total_length"] < 200
            and analysis["message_count"] <= 2
        )

        # Creative writing detection
        creative_score = 0.0
        for indicator, weight in self.creative_indicators.items():
            if indicator.lower() in full_text_lower:
                creative_score += weight * 0.2

        analysis["creative_confidence"] = min(1.0, creative_score)
        analysis["is_creative"] = analysis["creative_confidence"] > 0.4

        # Determine task type and confidence
        task_scores = {
            "code_generation": analysis["code_confidence"],
            "complex_analysis": analysis["complexity_confidence"],
            "simple_qa": analysis["simple_confidence"],
            "creative_writing": analysis["creative_confidence"],
        }

        # Special case for code review
        if analysis["has_code"] and analysis["has_complexity"]:
            task_scores["code_review"] = (
                analysis["code_confidence"] + analysis["complexity_confidence"]
            ) / 2

        # Find the highest scoring task type
        best_task = max(task_scores.items(), key=lambda x: x[1])
        if best_task[1] > 0.4:
            analysis["task_type"] = best_task[0]
            analysis["confidence_score"] = best_task[1]
        else:
            analysis["task_type"] = "general"
            analysis["confidence_score"] = 0.5

        return analysis

    async def _route_request(
        self,
        messages: list[ChatMessage],
        analysis: dict[str, Any],
        user_id: str | None,
        constraints: dict[str, Any] | None,
    ) -> tuple[str, str, str, float]:
        """Route request based on heuristic analysis."""
        task_type = analysis["task_type"]
        confidence = analysis["confidence_score"]

        # Get model preferences for this task type
        preferences = self.model_preferences.get(task_type, self.model_preferences["general"])

        # Apply constraints if provided
        if constraints:
            preferences = self._filter_by_constraints(preferences, constraints)

        # Select the best available model
        for provider, model, model_confidence in preferences:
            # Check if provider is available (this would be checked against actual provider status)
            if self._is_provider_available(provider):
                # Calculate combined confidence
                combined_confidence = (confidence + model_confidence) / 2

                # Generate reasoning
                reasoning = self._generate_reasoning(analysis, provider, model, task_type)

                return provider, model, reasoning, combined_confidence

        # Fallback to default
        return "openai", "gpt-3.5-turbo", "Fallback to default model", 0.5

    def _filter_by_constraints(
        self, preferences: list[tuple[str, str, float]], constraints: dict[str, Any]
    ) -> list[tuple[str, str, float]]:
        """Filter model preferences based on constraints."""
        filtered = []

        for provider, model, confidence in preferences:
            # Budget constraints
            if "max_cost" in constraints:
                estimated_cost = self._estimate_model_cost(provider, model)
                if estimated_cost > constraints["max_cost"]:
                    continue

            # Provider preferences
            if "preferred_providers" in constraints:
                if provider not in constraints["preferred_providers"]:
                    continue

            # Excluded models
            if "excluded_models" in constraints:
                if f"{provider}/{model}" in constraints["excluded_models"]:
                    continue

            filtered.append((provider, model, confidence))

        return filtered or preferences  # Return original if all filtered out

    def _is_provider_available(self, provider: str) -> bool:
        """Check if a provider is available (placeholder for actual implementation)."""
        # In the actual implementation, this would check provider health status
        return True

    def _estimate_model_cost(self, provider: str, model: str) -> float:
        """Estimate cost for a model (placeholder for actual implementation)."""
        # This would use the actual cost tracker
        cost_estimates = {
            ("openai", "gpt-4o"): 0.01,
            ("openai", "gpt-3.5-turbo"): 0.002,
            ("anthropic", "claude-3-5-sonnet"): 0.008,
            ("anthropic", "claude-3-haiku"): 0.001,
            ("google", "gemini-1.5-flash"): 0.0005,
            ("mistral", "mistral-small"): 0.0003,
        }
        return cost_estimates.get((provider, model), 0.005)

    def _generate_reasoning(
        self, analysis: dict[str, Any], provider: str, model: str, task_type: str
    ) -> str:
        """Generate human-readable reasoning for the routing decision."""
        reasons = []

        if analysis["has_code"]:
            reasons.append(f"Code detected (confidence: {analysis['code_confidence']:.2f})")
            if analysis["detected_language"]:
                reasons.append(f"Language: {analysis['detected_language']}")

        if analysis["has_complexity"]:
            reasons.append(
                f"Complex analysis required (confidence: {analysis['complexity_confidence']:.2f})"
            )

        if analysis["is_simple"]:
            reasons.append(f"Simple query (length: {analysis['total_length']} chars)")

        if analysis["is_creative"]:
            reasons.append(
                f"Creative writing task (confidence: {analysis['creative_confidence']:.2f})"
            )

        if analysis["total_length"] > 2000:
            reasons.append("Long prompt requires capable model")

        if analysis["message_count"] > 5:
            reasons.append("Multi-turn conversation")

        task_reason = f"Task type: {task_type}"
        if reasons:
            task_reason += f" ({', '.join(reasons)})"

        model_reason = f"Selected {provider}/{model} for optimal {task_type} performance"

        return f"{task_reason}. {model_reason}"
