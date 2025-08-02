# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Hybrid routing implementation combining multiple strategies.

This module implements a hybrid router that combines heuristic, semantic,
and cascade routing strategies to make optimal routing decisions.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import structlog

from ..core.exceptions import RoutingError
from ..models import ChatMessage
from .base_router import BaseRouter
from .cascade_router import CascadeRouter
from .heuristic_router import HeuristicRouter
from .semantic_router import SemanticRouter

logger = structlog.get_logger(__name__)


class HybridRouter(BaseRouter):
    """
    Hybrid router combining multiple routing strategies.

    This router uses multiple routing strategies and combines their
    recommendations to make the optimal routing decision.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("hybrid", config)

        # Configuration
        self.strategy_weights = self.config.get(
            "strategy_weights", {"heuristic": 0.4, "semantic": 0.4, "cascade": 0.2}
        )

        self.enable_consensus = self.config.get("enable_consensus", True)
        self.consensus_threshold = self.config.get("consensus_threshold", 0.6)
        self.fallback_strategy = self.config.get("fallback_strategy", "heuristic")

        # Initialize sub-routers
        try:
            self.heuristic_router = HeuristicRouter(self.config.get("heuristic", {}))
            logger.info("heuristic_router_initialized")
        except Exception as e:
            logger.warning("heuristic_router_init_failed", error=str(e))
            self.heuristic_router = None

        try:
            self.semantic_router = SemanticRouter(self.config.get("semantic", {}))
            logger.info("semantic_router_initialized")
        except Exception as e:
            logger.warning("semantic_router_init_failed", error=str(e))
            self.semantic_router = None

        try:
            self.cascade_router = CascadeRouter(self.config.get("cascade", {}))
            logger.info("cascade_router_initialized")
        except Exception as e:
            logger.warning("cascade_router_init_failed", error=str(e))
            self.cascade_router = None

        # Validate that at least one router is available
        available_routers = sum(
            [
                1
                for router in [self.heuristic_router, self.semantic_router, self.cascade_router]
                if router is not None
            ]
        )

        if available_routers == 0:
            raise RoutingError("No sub-routers available for hybrid routing")

        logger.info("hybrid_router_initialized", available_routers=available_routers)

    async def analyze_prompt(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """Analyze prompt using all available strategies."""
        analyses = {}

        # Run analysis with all available routers
        if self.heuristic_router:
            try:
                analyses["heuristic"] = await self.heuristic_router.analyze_prompt(messages)
            except Exception as e:
                logger.warning("heuristic_analysis_failed", error=str(e))

        if self.semantic_router:
            try:
                analyses["semantic"] = await self.semantic_router.analyze_prompt(messages)
            except Exception as e:
                logger.warning("semantic_analysis_failed", error=str(e))

        if self.cascade_router:
            try:
                analyses["cascade"] = await self.cascade_router.analyze_prompt(messages)
            except Exception as e:
                logger.warning("cascade_analysis_failed", error=str(e))

        # Combine analyses
        combined_analysis = self._combine_analyses(analyses, messages)

        logger.debug(
            "hybrid_analysis_complete",
            strategies_used=list(analyses.keys()),
            combined_task_type=combined_analysis.get("task_type"),
            combined_confidence=combined_analysis.get("confidence_score"),
        )

        return combined_analysis

    def _combine_analyses(
        self, analyses: Dict[str, Dict[str, Any]], messages: List[ChatMessage]
    ) -> Dict[str, Any]:
        """Combine analyses from multiple strategies."""
        if not analyses:
            # Fallback analysis
            full_text = " ".join([msg.content for msg in messages if msg.content])
            return {
                "total_length": len(full_text),
                "message_count": len(messages),
                "task_type": "general",
                "confidence_score": 0.5,
                "strategy_analyses": {},
                "consensus": False,
            }

        # Extract common fields
        full_text = " ".join([msg.content for msg in messages if msg.content])
        combined = {
            "total_length": len(full_text),
            "message_count": len(messages),
            "word_count": len(full_text.split()),
            "strategy_analyses": analyses,
            "consensus": False,
            "consensus_score": 0.0,
        }

        # Collect task types and confidence scores
        task_types = []
        confidence_scores = []

        for strategy, analysis in analyses.items():
            task_type = analysis.get("task_type", "general")
            confidence = analysis.get("confidence_score", 0.0)
            weight = self.strategy_weights.get(strategy, 1.0)

            task_types.append((task_type, confidence * weight))
            confidence_scores.append(confidence * weight)

        # Determine consensus task type
        task_type_votes = {}
        for task_type, weighted_confidence in task_types:
            if task_type not in task_type_votes:
                task_type_votes[task_type] = []
            task_type_votes[task_type].append(weighted_confidence)

        # Calculate weighted averages for each task type
        task_type_scores = {}
        for task_type, confidences in task_type_votes.items():
            task_type_scores[task_type] = sum(confidences) / len(confidences)

        # Select the task type with highest weighted confidence
        best_task_type = max(task_type_scores.items(), key=lambda x: x[1])
        combined["task_type"] = best_task_type[0]
        combined["confidence_score"] = best_task_type[1]

        # Check for consensus
        if self.enable_consensus:
            # Count how many strategies agree on the task type
            agreement_count = sum(
                1 for task_type, _ in task_types if task_type == best_task_type[0]
            )
            consensus_ratio = agreement_count / len(task_types)

            combined["consensus"] = consensus_ratio >= self.consensus_threshold
            combined["consensus_score"] = consensus_ratio

        # Add strategy-specific insights
        combined["has_code"] = any(
            analysis.get("has_code", False) for analysis in analyses.values()
        )
        combined["has_complexity"] = any(
            analysis.get("has_complexity", False) for analysis in analyses.values()
        )
        combined["is_simple"] = any(
            analysis.get("is_simple", False) for analysis in analyses.values()
        )

        return combined

    async def _route_request(
        self,
        messages: List[ChatMessage],
        analysis: Dict[str, Any],
        user_id: Optional[str],
        constraints: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, str, float]:
        """Route request using hybrid strategy."""
        strategy_recommendations = []

        # Get recommendations from all available strategies
        if self.heuristic_router:
            try:
                heuristic_rec = await self.heuristic_router._route_request(
                    messages,
                    analysis.get("strategy_analyses", {}).get("heuristic", analysis),
                    user_id,
                    constraints,
                )
                strategy_recommendations.append(("heuristic", heuristic_rec))
            except Exception as e:
                logger.warning("heuristic_routing_failed", error=str(e))

        if self.semantic_router:
            try:
                semantic_rec = await self.semantic_router._route_request(
                    messages,
                    analysis.get("strategy_analyses", {}).get("semantic", analysis),
                    user_id,
                    constraints,
                )
                strategy_recommendations.append(("semantic", semantic_rec))
            except Exception as e:
                logger.warning("semantic_routing_failed", error=str(e))

        if self.cascade_router:
            try:
                cascade_rec = await self.cascade_router._route_request(
                    messages,
                    analysis.get("strategy_analyses", {}).get("cascade", analysis),
                    user_id,
                    constraints,
                )
                strategy_recommendations.append(("cascade", cascade_rec))
            except Exception as e:
                logger.warning("cascade_routing_failed", error=str(e))

        if not strategy_recommendations:
            return (
                "openai",
                "gpt-3.5-turbo",
                "Hybrid routing fallback - no strategies available",
                0.5,
            )

        # Combine recommendations
        final_recommendation = self._combine_recommendations(
            strategy_recommendations, analysis, constraints
        )

        return final_recommendation

    def _combine_recommendations(
        self,
        recommendations: List[Tuple[str, Tuple[str, str, str, float]]],
        analysis: Dict[str, Any],
        constraints: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, str, float]:
        """Combine recommendations from multiple strategies."""
        if len(recommendations) == 1:
            # Only one recommendation available
            strategy, (provider, model, reasoning, confidence) = recommendations[0]
            combined_reasoning = f"Hybrid routing using {strategy} strategy: {reasoning}"
            return provider, model, combined_reasoning, confidence

        # Score each recommendation
        scored_recommendations = []

        for strategy, (provider, model, reasoning, confidence) in recommendations:
            weight = self.strategy_weights.get(strategy, 1.0)

            # Calculate weighted score
            score = confidence * weight

            # Bonus for consensus
            if analysis.get("consensus", False):
                # Check if this recommendation aligns with consensus
                consensus_task_type = analysis.get("task_type", "general")
                if self._model_matches_task_type(provider, model, consensus_task_type):
                    score += 0.1

            # Apply constraint penalties
            if constraints:
                if "max_cost" in constraints:
                    estimated_cost = self._estimate_cost(provider, model)
                    if estimated_cost > constraints["max_cost"]:
                        score -= 0.2

            scored_recommendations.append((score, strategy, provider, model, reasoning, confidence))

        # Select the highest scoring recommendation
        scored_recommendations.sort(key=lambda x: x[0], reverse=True)
        best_score, best_strategy, provider, model, reasoning, confidence = scored_recommendations[
            0
        ]

        # Generate combined reasoning
        strategy_names = [rec[0] for rec in recommendations]  # rec[0] is the strategy name
        combined_reasoning = f"Hybrid routing (strategies: {', '.join(strategy_names)}) selected {best_strategy}: {reasoning}"

        # Calculate combined confidence
        combined_confidence = min(1.0, best_score)

        logger.info(
            "hybrid_routing_decision",
            selected_strategy=best_strategy,
            provider=provider,
            model=model,
            score=best_score,
            confidence=combined_confidence,
            consensus=analysis.get("consensus", False),
        )

        return provider, model, combined_reasoning, combined_confidence

    def _model_matches_task_type(self, provider: str, model: str, task_type: str) -> bool:
        """Check if a model is appropriate for a task type."""
        # Simple heuristics for model-task matching
        high_quality_models = ["gpt-4o", "claude-3-5-sonnet", "claude-3-opus"]
        budget_models = ["mistral-small", "gemini-1.5-flash", "claude-3-haiku"]

        if task_type in ["code_generation", "complex_analysis", "creative_writing"]:
            return model in high_quality_models
        elif task_type in ["simple_qa", "basic_tasks"]:
            return model in budget_models
        else:
            return True  # General tasks can use any model

    def _estimate_cost(self, provider: str, model: str) -> float:
        """Estimate cost for a model."""
        cost_estimates = {
            ("openai", "gpt-4o"): 0.01,
            ("openai", "gpt-3.5-turbo"): 0.002,
            ("anthropic", "claude-3-5-sonnet"): 0.008,
            ("anthropic", "claude-3-haiku"): 0.001,
            ("anthropic", "claude-3-opus"): 0.015,
            ("google", "gemini-1.5-flash"): 0.0005,
            ("google", "gemini-1.5-pro"): 0.003,
            ("mistral", "mistral-small"): 0.0003,
        }
        return cost_estimates.get((provider, model), 0.005)
