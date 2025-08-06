# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Heuristic routing logic for selecting the best LLM provider and model.
"""

import re
from typing import Any

from .config import settings
from .models import ChatMessage


class HeuristicRouter:
    """Intelligent router that selects the best LLM based on prompt characteristics."""

    def __init__(self):
        # Code detection patterns
        self.code_patterns = [
            r"```[\s\S]*?```",  # Code blocks
            r"`[^`\n]+`",  # Inline code
            r"\bdef\s+\w+\s*\(",  # Python function definitions
            r"\bclass\s+\w+\s*[:\(]",  # Class definitions
            r"\bfunction\s+\w+\s*\(",  # JavaScript functions
            r"\bpublic\s+\w+\s+\w+\s*\(",  # Java/C# methods
            r"\bimport\s+\w+",  # Import statements
            r"\bfrom\s+\w+\s+import",  # Python imports
            r"#include\s*<\w+>",  # C/C++ includes
            r"\$\w+\s*=",  # Variable assignments (PHP, shell)
            r"SELECT\s+.*\s+FROM",  # SQL queries
            r"CREATE\s+TABLE",  # SQL DDL
            r"<\w+[^>]*>.*</\w+>",  # HTML/XML tags
            r'{\s*"[\w":\s,\[\]{}]+\s*}',  # JSON objects
        ]

        # Programming language keywords
        self.programming_keywords = [
            "function",
            "class",
            "import",
            "export",
            "const",
            "let",
            "var",
            "def",
            "return",
            "if",
            "else",
            "elif",
            "for",
            "while",
            "try",
            "except",
            "public",
            "private",
            "protected",
            "static",
            "void",
            "int",
            "string",
            "array",
            "list",
            "dict",
            "object",
            "null",
            "undefined",
            "true",
            "false",
            "async",
            "await",
            "promise",
            "callback",
            "lambda",
            "yield",
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "CREATE",
            "DROP",
            "ALTER",
            "html",
            "css",
            "javascript",
            "python",
            "java",
            "cpp",
            "sql",
            "json",
        ]

        # Pre-compile regex patterns for efficiency
        self._compiled_keyword_patterns = [
            re.compile(r"\b" + re.escape(keyword.lower()) + r"\b", re.IGNORECASE)
            for keyword in self.programming_keywords
        ]

        # Complexity indicators
        self.complexity_keywords = [
            "analyze",
            "analysis",
            "explain",
            "explanation",
            "debug",
            "debugging",
            "reasoning",
            "reason",
            "complex",
            "complicated",
            "algorithm",
            "algorithms",
            "optimize",
            "optimization",
            "architecture",
            "design pattern",
            "patterns",
            "performance",
            "scalability",
            "trade-off",
            "tradeoffs",
            "comparison",
            "evaluate",
            "assessment",
            "review",
            "critique",
            "detailed",
            "comprehensive",
            "step-by-step",
            "methodology",
            "approach",
            "strategy",
            "framework",
            "implementation",
            "solution",
            "problem-solving",
            "troubleshoot",
        ]

        # Simple query indicators
        self.simple_indicators = [
            "what is",
            "who is",
            "when is",
            "where is",
            "how much",
            "how many",
            "define",
            "definition",
            "meaning",
            "translate",
            "translation",
            "calculate",
            "convert",
            "list",
            "name",
            "tell me",
            "show me",
        ]

        # Model preferences by task type (including LiteLLM proxy models)
        self.model_preferences = {
            "code": [
                ("openai", "gpt-4o"),
                ("anthropic", "claude-3-sonnet-20240229"),
                ("anthropic", "claude-3-5-sonnet-20241022"),
                ("litellm", "gpt-4"),  # LiteLLM proxy for GPT-4
                ("openai", "gpt-3.5-turbo"),
                ("litellm", "gpt-3.5-turbo"),  # LiteLLM proxy fallback
                ("groq", "mixtral-8x7b-32768"),
            ],
            "complex": [
                ("openai", "gpt-4o"),
                ("anthropic", "claude-3-sonnet-20240229"),
                ("anthropic", "claude-3-5-sonnet-20241022"),
                ("litellm", "claude-3-sonnet"),  # LiteLLM proxy for Claude
                ("anthropic", "claude-3-haiku-20240307"),
                ("litellm", "gpt-4"),  # LiteLLM proxy for GPT-4
                ("openai", "gpt-3.5-turbo"),
                ("litellm", "claude-3-haiku"),  # LiteLLM proxy for Haiku
                ("groq", "mixtral-8x7b-32768"),
            ],
            "simple": [
                ("groq", "mixtral-8x7b-32768"),
                ("openai", "gpt-4o-mini"),
                ("litellm", "claude-3-haiku"),  # LiteLLM proxy for Haiku
                ("mistral", "mistral-small"),
                ("mistral", "mistral-small-latest"),
                ("anthropic", "claude-3-haiku-20240307"),
                ("litellm", "gpt-3.5-turbo"),  # LiteLLM proxy for GPT-3.5
                ("openai", "gpt-3.5-turbo"),
            ],
            "general": [
                ("groq", "mixtral-8x7b-32768"),
                ("openai", "gpt-4o-mini"),
                ("litellm", "gpt-3.5-turbo"),  # LiteLLM proxy for GPT-3.5
                ("mistral", "mistral-small"),
                ("mistral", "mistral-small-latest"),
                ("openai", "gpt-3.5-turbo"),
                ("litellm", "claude-3-haiku"),  # LiteLLM proxy for Haiku
                ("anthropic", "claude-3-haiku-20240307"),
            ],
        }

        # Pre-compile regex patterns for efficiency after all attributes are defined
        self._compiled_code_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in self.code_patterns
        ]

        self._compiled_complexity_patterns = [
            re.compile(r"\b" + re.escape(keyword.lower()) + r"\b", re.IGNORECASE)
            for keyword in self.complexity_keywords
        ]

    def analyze_prompt(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """Analyze prompt characteristics to determine routing strategy."""
        # Combine all message content for analysis
        full_text = " ".join([msg.content for msg in messages if msg.content])
        full_text_lower = full_text.lower()

        analysis = {
            "total_length": len(full_text),
            "message_count": len(messages),
            "has_code": False,
            "code_confidence": 0.0,
            "has_complexity": False,
            "complexity_confidence": 0.0,
            "is_simple": False,
            "simple_confidence": 0.0,
            "detected_languages": [],
            "task_type": "general",
        }

        # Code detection
        code_matches = 0
        for pattern in self._compiled_code_patterns:
            matches = pattern.findall(full_text)
            code_matches += len(matches)

        # Programming keyword detection (using word boundaries to avoid false positives)
        programming_matches = 0
        for pattern in self._compiled_keyword_patterns:
            if pattern.search(full_text_lower):
                programming_matches += 1

        # Calculate code confidence
        analysis["code_confidence"] = min(1.0, (code_matches * 0.3 + programming_matches * 0.1))
        analysis["has_code"] = analysis["code_confidence"] >= settings.code_detection_threshold

        # Complexity detection (using word boundaries to avoid false positives)
        # Complexity analysis
        complexity_matches = 0
        for pattern in self._compiled_complexity_patterns:
            if pattern.search(full_text_lower):
                complexity_matches += 1

        # Calculate complexity confidence
        analysis["complexity_confidence"] = min(1.0, complexity_matches * 0.1)
        analysis["has_complexity"] = (
            analysis["complexity_confidence"] >= settings.complexity_threshold
        )

        # Simple query detection
        simple_matches = 0
        for indicator in self.simple_indicators:
            if indicator.lower() in full_text_lower:
                simple_matches += 1

        # Calculate simple confidence
        analysis["simple_confidence"] = min(1.0, simple_matches * 0.2)
        analysis["is_simple"] = (
            analysis["simple_confidence"] >= 0.2
            and analysis["total_length"] < settings.simple_query_max_length
            and analysis["message_count"] <= 2
        )

        # Determine task type
        if analysis["has_code"]:
            analysis["task_type"] = "code"
        elif analysis["has_complexity"]:
            analysis["task_type"] = "complex"
        elif analysis["is_simple"]:
            analysis["task_type"] = "simple"
        else:
            analysis["task_type"] = "general"

        return analysis

    def select_model(
        self,
        messages: list[ChatMessage],
        user_id: str | None = None,
        budget_constraint: float | None = None,
    ) -> tuple[str, str, str]:
        """
        Select the best provider and model for the given messages.

        Returns:
            Tuple of (provider, model, reasoning)
        """
        analysis = self.analyze_prompt(messages)
        task_type = analysis["task_type"]

        # Get model preferences for this task type
        preferences = self.model_preferences.get(task_type, self.model_preferences["general"])

        # If budget constraint is specified, filter by cost
        if budget_constraint:
            # Get pricing information
            pricing = settings.get_provider_pricing()

            # Calculate estimated costs for each model and filter
            affordable_preferences = []
            for provider, model in preferences:
                if provider in pricing and model in pricing[provider]:
                    input_cost = pricing[provider][model]["input"]
                    output_cost = pricing[provider][model]["output"]
                    # Estimate cost for ~100 input + 50 output tokens (typical short query)
                    estimated_cost = (100 / 1_000_000) * input_cost + (50 / 1_000_000) * output_cost
                    if estimated_cost <= budget_constraint:
                        affordable_preferences.append((provider, model, estimated_cost))

            # Sort by cost and take the cheapest options
            if affordable_preferences:
                affordable_preferences.sort(key=lambda x: x[2])  # Sort by cost
                preferences = [(p[0], p[1]) for p in affordable_preferences]
            else:
                # If no models are affordable, fall back to cheapest available
                if budget_constraint < 0.001:  # Very low budget
                    preferences = [
                        ("groq", "mixtral-8x7b-32768"),
                        ("openai", "gpt-4o-mini"),
                        ("mistral", "mistral-small"),
                        ("mistral", "mistral-small-latest"),
                    ]

        # Select the first available model from preferences
        for provider, model in preferences:
            # Check if the provider and model are available in current configuration
            pricing = settings.get_provider_pricing()
            if provider in pricing and model in pricing[provider]:
                return provider, model, self._generate_reasoning(analysis, provider, model)

        # Fallback
        return "openai", "gpt-3.5-turbo", "Fallback to default model"

    def _generate_reasoning(self, analysis: dict, provider: str, model: str) -> str:
        """Generate human-readable reasoning for the routing decision."""
        reasons = []

        if analysis["has_code"]:
            reasons.append(f"Code detected (confidence: {analysis['code_confidence']:.2f})")

        if analysis["has_complexity"]:
            reasons.append(
                f"Complex analysis required (confidence: {analysis['complexity_confidence']:.2f})"
            )

        if analysis["is_simple"]:
            reasons.append(f"Simple query detected (length: {analysis['total_length']} chars)")

        if analysis["total_length"] > 1000:
            reasons.append("Long prompt requires capable model")

        if analysis["message_count"] > 5:
            reasons.append("Multi-turn conversation")

        task_reason = f"Task type: {analysis['task_type']}"
        if reasons:
            task_reason += f" ({', '.join(reasons)})"

        model_reason = (
            f"Selected {provider}/{model} for optimal {analysis['task_type']} performance"
        )

        return f"{task_reason}. {model_reason}"

    def get_routing_stats(self) -> dict[str, Any]:
        """Get statistics about routing decisions (for monitoring)."""
        # In a real implementation, this would query the database
        # for routing statistics
        return {
            "total_requests": 0,
            "routing_by_task_type": {"code": 0, "complex": 0, "simple": 0, "general": 0},
            "routing_by_provider": {"openai": 0, "anthropic": 0, "mistral": 0},
        }


# Global router instance
router = HeuristicRouter()
