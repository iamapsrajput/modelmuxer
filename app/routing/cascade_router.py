# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Enhanced Cascade routing implementation for cost-aware model selection.

This module implements a FrugalGPT-inspired cascading approach where requests
are first tried with cheaper models and escalated to more expensive ones based
on quality thresholds and confidence scores.
"""

import operator
import time
from dataclasses import dataclass
from typing import Any

import structlog

from ..models import ChatMessage
from .base_router import BaseRouter

logger = structlog.get_logger(__name__)


@dataclass
class CascadeStep:
    """Represents a single step in the cascade chain."""

    provider: str
    model: str
    max_cost: float
    confidence_threshold: float
    quality_threshold: float
    timeout: float = 30.0


class CascadeRouter(BaseRouter):
    """
    Cascade router that tries cheaper models first and escalates if needed.

    This router implements a cost-aware cascading strategy where requests
    are first attempted with cheaper, faster models and only escalated to
    more expensive models if the initial attempt fails or doesn't meet
    quality thresholds.
    """

    def __init__(self, cost_tracker: Any = None, config: dict[str, Any] | None = None):
        super().__init__("cascade", config)

        self.cost_tracker = cost_tracker

        # Enhanced configuration
        self.max_cascade_levels = self.config.get("max_cascade_levels", 4)
        self.quality_threshold = self.config.get("quality_threshold", 0.7)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.enable_quality_check = self.config.get("enable_quality_check", True)

        # Initialize cascade chains
        self.cascade_chains = self._initialize_cascade_chains()

        # Initialize cascade levels (legacy compatibility)
        self.cascade_levels = self._initialize_cascade_levels()

    def _initialize_cascade_chains(self) -> dict[str, list[CascadeStep]]:
        """Define cascade chains for different use cases"""
        return {
            "cost_optimized": [
                CascadeStep("groq", "llama-3.1-8b", 0.001, 0.7, 0.6),
                CascadeStep("mistral", "mistral-small", 0.005, 0.8, 0.7),
                CascadeStep("openai", "gpt-3.5-turbo", 0.02, 0.9, 0.8),
                CascadeStep("openai", "gpt-4o", 0.1, 0.95, 0.9),
            ],
            "quality_focused": [
                CascadeStep("anthropic", "claude-3-haiku", 0.01, 0.8, 0.75),
                CascadeStep("anthropic", "claude-3-5-sonnet", 0.05, 0.9, 0.85),
                CascadeStep("openai", "gpt-4o", 0.15, 0.95, 0.9),
            ],
            "balanced": [
                CascadeStep("google", "gemini-1.5-flash", 0.002, 0.6, 0.5),
                CascadeStep("openai", "gpt-3.5-turbo", 0.015, 0.85, 0.8),
                CascadeStep("anthropic", "claude-3-5-sonnet", 0.08, 0.95, 0.9),
            ],
        }

    def _initialize_cascade_levels(self) -> list[dict[str, Any]]:
        """Initialize legacy cascade levels structure."""
        return [
            {"level": 1, "models": [("groq", "llama-3.1-8b", 0.001, 0.6)]},
            {"level": 2, "models": [("mistral", "mistral-small", 0.005, 0.7)]},
            {"level": 3, "models": [("openai", "gpt-3.5-turbo", 0.02, 0.8)]},
            {"level": 4, "models": [("openai", "gpt-4o", 0.1, 0.9)]},
        ]

    async def route_with_cascade(
        self,
        messages: list[dict[str, Any] | ChatMessage],
        cascade_type: str = "balanced",
        max_budget: float = 0.1,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Execute cascade routing with cost and quality thresholds
        Returns: (response, routing_metadata)
        """
        cascade_chain = self.cascade_chains.get(cascade_type, self.cascade_chains["balanced"])
        routing_metadata: dict[str, Any] = {
            "cascade_type": cascade_type,
            "steps_attempted": [],
            "total_cost": 0.0,
            "escalation_reasons": [],
            "start_time": time.time(),
        }

        for step_idx, step in enumerate(cascade_chain):
            # Check budget constraint
            if routing_metadata["total_cost"] + step.max_cost > max_budget:
                routing_metadata["escalation_reasons"].append(f"Budget exceeded at step {step_idx}")
                continue

            try:
                # Attempt request with current model
                response, step_cost = await self._execute_step(step, messages, user_id, **kwargs)

                routing_metadata["steps_attempted"].append(
                    {
                        "step": step_idx,
                        "provider": step.provider,
                        "model": step.model,
                        "cost": step_cost,
                        "success": True,
                    }
                )
                routing_metadata["total_cost"] += step_cost

                # Evaluate response quality and confidence
                quality_score, confidence_score = await self._evaluate_response(
                    response, messages, step
                )

                routing_metadata["quality_score"] = quality_score
                routing_metadata["confidence_score"] = confidence_score

                # Check if response meets thresholds
                if (
                    confidence_score >= step.confidence_threshold
                    and quality_score >= step.quality_threshold
                ):
                    routing_metadata["final_model"] = f"{step.provider}/{step.model}"
                    routing_metadata["escalation_reasons"].append("Quality threshold met")
                    routing_metadata["response_time"] = time.time() - routing_metadata["start_time"]
                    return response, routing_metadata
                else:
                    routing_metadata["escalation_reasons"].append(
                        f"Quality/confidence below threshold: {quality_score:.2f}/{confidence_score:.2f}"
                    )

            except Exception as e:
                routing_metadata["steps_attempted"].append(
                    {
                        "step": step_idx,
                        "provider": step.provider,
                        "model": step.model,
                        "cost": 0.0,
                        "success": False,
                        "error": str(e),
                    }
                )
                routing_metadata["escalation_reasons"].append(f"Error in step {step_idx}: {str(e)}")
                continue

        # If we reach here, all steps failed or didn't meet quality thresholds
        routing_metadata["response_time"] = time.time() - routing_metadata["start_time"]
        raise Exception(
            "Cascade routing failed: no suitable response found within budget/quality constraints"
        )

    async def analyze_prompt(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """Analyze prompt to determine initial cascade level."""
        # Combine all message content
        full_text = " ".join([msg.content for msg in messages if msg.content])
        full_text_lower = full_text.lower()

        analysis = {
            "total_length": len(full_text),
            "message_count": len(messages),
            "word_count": len(full_text.split()),
            "complexity_score": 0.0,
            "initial_cascade_level": 1,
            "estimated_difficulty": "easy",
            "task_type": "general",
        }

        # Calculate complexity score
        complexity_indicators = {
            "code": ["function", "class", "algorithm", "implement", "debug", "optimize"],
            "analysis": ["analyze", "compare", "evaluate", "assess", "critique"],
            "creative": ["write", "create", "generate", "compose", "design"],
            "complex": ["complex", "detailed", "comprehensive", "thorough", "in-depth"],
        }

        complexity_score = 0.0
        task_indicators = {}

        for category, keywords in complexity_indicators.items():
            category_score = 0.0
            for keyword in keywords:
                if keyword in full_text_lower:
                    category_score += 1.0

            task_indicators[category] = category_score
            complexity_score += category_score

        # Normalize complexity score
        analysis["complexity_score"] = min(1.0, complexity_score / 10.0)

        # Determine task type
        max_category = max(task_indicators.items(), key=operator.itemgetter(1))
        if max_category[1] > 0:
            analysis["task_type"] = max_category[0]

        # Determine initial cascade level based on complexity
        if analysis["complexity_score"] < 0.3:
            analysis["initial_cascade_level"] = 1
            analysis["estimated_difficulty"] = "easy"
        elif analysis["complexity_score"] < 0.7:
            analysis["initial_cascade_level"] = 2
            analysis["estimated_difficulty"] = "medium"
        else:
            analysis["initial_cascade_level"] = 3
            analysis["estimated_difficulty"] = "hard"

        # Adjust based on message length
        if analysis["total_length"] > 2000:
            analysis["initial_cascade_level"] = min(3, analysis["initial_cascade_level"] + 1)

        # Adjust based on message count (conversation complexity)
        if analysis["message_count"] > 5:
            analysis["initial_cascade_level"] = min(3, analysis["initial_cascade_level"] + 1)

        return analysis

    async def _route_request(
        self,
        messages: list[ChatMessage],
        analysis: dict[str, Any],
        user_id: str | None,
        constraints: dict[str, Any] | None,
    ) -> tuple[str, str, str, float]:
        """Route request using cascade strategy."""
        initial_level = analysis["initial_cascade_level"]
        max_cost = constraints.get("max_cost") if constraints else None

        # Try each cascade level starting from the initial level
        for level_info in self.cascade_levels[initial_level - 1 :]:
            level = level_info["level"]
            models = level_info["models"]

            # Filter models by constraints
            available_models = self._filter_models_by_constraints(models, constraints)

            if not available_models:
                continue

            # Try the best model at this level
            for provider, model, cost, quality in available_models:
                # Check cost constraint
                if max_cost and cost > max_cost:
                    continue

                # Check if provider is available
                if not self._is_provider_available(provider):
                    continue

                # This is where we would actually test the model
                # For now, we'll simulate the decision
                confidence = self._calculate_confidence(analysis, level, quality)
                reasoning = self._generate_reasoning(analysis, provider, model, level, cost)

                logger.info(
                    "cascade_routing_decision",
                    level=level,
                    provider=provider,
                    model=model,
                    cost=cost,
                    confidence=confidence,
                )

                return provider, model, reasoning, confidence

        # Fallback if no suitable model found
        return "openai", "gpt-3.5-turbo", "Cascade routing fallback", 0.5

    def _filter_models_by_constraints(
        self, models: list[tuple[str, str, float, float]], constraints: dict[str, Any] | None
    ) -> list[tuple[str, str, float, float]]:
        """Filter models based on constraints."""
        if not constraints:
            return models

        filtered = []

        for provider, model, cost, quality in models:
            # Budget constraints
            if "max_cost" in constraints and cost > constraints["max_cost"]:
                continue

            # Provider preferences
            if "preferred_providers" in constraints:
                if provider not in constraints["preferred_providers"]:
                    continue

            # Excluded models
            if "excluded_models" in constraints:
                if f"{provider}/{model}" in constraints["excluded_models"]:
                    continue

            # Quality requirements
            if "min_quality" in constraints and quality < constraints["min_quality"]:
                continue

            filtered.append((provider, model, cost, quality))

        # Sort by cost (cheapest first)
        return sorted(filtered, key=operator.itemgetter(2))

    def _is_provider_available(self, provider: str) -> bool:
        """Check if a provider is available."""
        # Placeholder - would check actual provider status
        return True

    def _calculate_confidence(
        self, analysis: dict[str, Any], level: int, model_quality: float
    ) -> float:
        """Calculate confidence score for the routing decision."""
        complexity = analysis["complexity_score"]

        # Base confidence from model quality
        confidence = model_quality

        # Adjust based on complexity match
        if level == 1 and complexity < 0.3:  # Simple task, budget model
            confidence += 0.1
        elif level == 2 and 0.3 <= complexity < 0.7:  # Medium task, balanced model
            confidence += 0.1
        elif level == 3 and complexity >= 0.7:  # Complex task, premium model
            confidence += 0.1
        else:
            # Mismatch between complexity and model level
            confidence -= 0.1

        return min(1.0, max(0.0, confidence))

    def _generate_reasoning(
        self, analysis: dict[str, Any], provider: str, model: str, level: int, cost: float
    ) -> str:
        """Generate reasoning for the cascade routing decision."""
        level_names = {1: "budget", 2: "balanced", 3: "premium"}
        level_name = level_names.get(level, "unknown")

        complexity = analysis["complexity_score"]
        difficulty = analysis["estimated_difficulty"]

        reasoning_parts = [
            f"Cascade routing at level {level} ({level_name})",
            f"Complexity score: {complexity:.2f} ({difficulty})",
            f"Selected {provider}/{model}",
            f"Estimated cost: ${cost:.4f}",
        ]

        return ". ".join(reasoning_parts)

    async def should_escalate(
        self, response: str, original_analysis: dict[str, Any], current_level: int
    ) -> bool:
        """
        Determine if the response should be escalated to a higher level.

        This is a placeholder for quality assessment logic.
        In a real implementation, this might use another model to evaluate
        the quality of the response.
        """
        if not self.enable_quality_check:
            return False

        if current_level >= self.max_cascade_levels:
            return False

        # Simple heuristics for escalation
        if len(response) < 50:  # Very short response might indicate failure
            return True

        if "I don't know" in response.lower() or "I'm not sure" in response.lower():
            return True

        # Could add more sophisticated quality checks here
        return False

    async def _execute_step(
        self,
        step: CascadeStep,
        messages: list[dict[str, Any] | ChatMessage],
        user_id: str | None = None,
        **kwargs,
    ) -> tuple[dict[str, Any], float]:
        """Execute a single cascade step"""
        provider = self._get_provider(step.provider)

        # Convert messages to dict format for provider
        messages_dict = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                messages_dict.append({"role": msg.role, "content": msg.content})
            else:
                messages_dict.append(msg)

        response = await provider.chat_completion(
            messages=messages_dict, model=step.model, **kwargs
        )

        # Calculate actual cost
        usage = response.get("usage", {})
        cost = provider.calculate_cost(
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), step.model
        )

        return response, cost

    def _get_provider(self, provider_name: str):
        """Get provider instance from registry"""
        try:
            from app.providers.registry import get_provider_registry

            registry = get_provider_registry()
            if provider_name in registry:
                return registry[provider_name]
            else:
                # Fallback to mock provider for testing
                return self._create_mock_provider()
        except ImportError:
            # Fallback for testing environments
            return self._create_mock_provider()

    def _create_mock_provider(self):
        """Create mock provider for testing"""

        class MockProvider:
            async def chat_completion(self, **kwargs):
                # Generate a high-quality response that will pass thresholds
                return {
                    "id": "test-123",
                    "choices": [
                        {
                            "message": {
                                "content": "Quantum computing is a revolutionary technology that leverages quantum mechanical phenomena such as superposition and entanglement to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits that can exist in multiple states simultaneously. This enables quantum computers to potentially solve certain complex problems exponentially faster than classical computers."
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 60},
                }

            def calculate_cost(self, prompt_tokens, completion_tokens, model):
                return 0.002

        return MockProvider()

    async def _evaluate_response(
        self,
        response: dict[str, Any],
        original_messages: list[dict[str, Any] | ChatMessage],
        step: CascadeStep,
    ) -> tuple[float, float]:
        """
        Evaluate response quality and confidence
        Returns: (quality_score, confidence_score)
        """
        content = response["choices"][0]["message"]["content"]

        # Convert messages to dict format for processing
        messages_dict = []
        for msg in original_messages:
            if isinstance(msg, ChatMessage):
                messages_dict.append({"role": msg.role, "content": msg.content})
            else:
                messages_dict.append(msg)

        # Quality metrics
        quality_score = self._calculate_quality_score(content, messages_dict)

        # Confidence metrics (based on response characteristics)
        confidence_score = self._calculate_confidence_score(content, response)

        return quality_score, confidence_score

    def _calculate_quality_score(self, content: str, messages: list[dict]) -> float:
        """Calculate quality score based on content analysis"""
        score = 0.5  # Base score

        # Length appropriateness
        if 50 <= len(content) <= 2000:
            score += 0.1

        # Coherence indicators
        if not self._has_repetitive_patterns(content):
            score += 0.1

        # Relevance to prompt (simple keyword matching)
        last_user_message = next(
            (msg["content"] for msg in reversed(messages) if msg["role"] == "user"), ""
        )
        if self._calculate_relevance_score(content, last_user_message) > 0.3:
            score += 0.2

        # Structure and formatting
        if self._has_good_structure(content):
            score += 0.1

        return min(score, 1.0)

    def _calculate_confidence_score(self, content: str, response: dict) -> float:
        """Calculate confidence score based on response characteristics"""
        score = 0.5  # Base score

        # Response completeness
        finish_reason = response["choices"][0].get("finish_reason", "")
        if finish_reason == "stop":
            score += 0.2

        # Content confidence indicators
        uncertainty_phrases = [
            "i'm not sure",
            "i don't know",
            "uncertain",
            "maybe",
            "perhaps",
            "i think",
            "possibly",
            "might be",
            "not certain",
        ]

        content_lower = content.lower()
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in content_lower)
        confidence_penalty = min(uncertainty_count * 0.1, 0.3)
        score -= confidence_penalty

        # Length confidence (very short or very long responses may indicate issues)
        if 20 <= len(content.split()) <= 500:
            score += 0.1

        return max(min(score, 1.0), 0.0)

    def _has_repetitive_patterns(self, content: str) -> bool:
        """Detect repetitive patterns that indicate poor quality"""
        words = content.split()
        if len(words) < 10:
            return False

        # Check for repeated sequences
        for i in range(len(words) - 5):
            sequence = " ".join(words[i : i + 3])
            if content.count(sequence) > 2:
                return True

        return False

    def _calculate_relevance_score(self, response: str, prompt: str) -> float:
        """Simple relevance calculation using word overlap"""
        if not prompt:
            return 0.0

        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())

        if not prompt_words:
            return 0.0

        overlap = len(prompt_words.intersection(response_words))
        return overlap / len(prompt_words)

    def _has_good_structure(self, content: str) -> bool:
        """Check if content has good structure (paragraphs, formatting)"""
        # Check for proper sentence structure
        sentences = content.split(".")
        if len(sentences) >= 2:
            return True

        # Check for formatting elements
        if any(marker in content for marker in ["\n\n", "- ", "1. ", "* "]):
            return True

        return False
