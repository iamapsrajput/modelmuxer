# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Semantic routing implementation using sentence transformers.

This module provides ML-based routing using semantic similarity to classify
prompts and route them to the most appropriate provider and model.
"""

from typing import Any

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from ..core.exceptions import ConfigurationError, RoutingError
from ..models import ChatMessage
from .base_router import BaseRouter

logger = structlog.get_logger(__name__)


class SemanticRouter(BaseRouter):
    """
    Semantic router using sentence transformers for prompt classification.

    This router uses machine learning to understand the semantic meaning
    of prompts and route them based on similarity to known patterns.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__("semantic", config)

        # Configuration
        self.model_name = self.config.get("model_name", "all-MiniLM-L6-v2")
        self.similarity_threshold = self.config.get("similarity_threshold", 0.6)
        self.cache_embeddings = self.config.get("cache_embeddings", True)

        # Initialize the sentence transformer
        try:
            self.encoder = SentenceTransformer(self.model_name)
            logger.info("semantic_router_initialized", model=self.model_name)
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize sentence transformer: {e}") from e

        # Route definitions and embeddings
        self.route_embeddings = {}
        self.route_examples = {}
        self.route_model_mapping = {}

        # Load or create route definitions
        self._initialize_routes()

    def _initialize_routes(self) -> None:
        """Initialize route definitions and generate embeddings."""
        # Define route categories with examples
        route_definitions = {
            "code_generation": {
                "examples": [
                    "Write a Python function to sort a list",
                    "Create a React component for a button",
                    "Implement a binary search algorithm in Java",
                    "Write a SQL query to find duplicate records",
                    "Create a REST API endpoint in Node.js",
                    "Implement a hash table in C++",
                    "Write a function to validate email addresses",
                    "Create a database migration script",
                    "Implement authentication middleware",
                    "Write unit tests for a calculator function",
                ],
                "models": [
                    ("openai", "gpt-4o", 0.95),
                    ("anthropic", "claude-3-5-sonnet", 0.90),
                    ("openai", "gpt-3.5-turbo", 0.75),
                ],
            },
            "code_review": {
                "examples": [
                    "Review this Python code for bugs",
                    "Analyze this JavaScript function for performance issues",
                    "Check this SQL query for security vulnerabilities",
                    "Optimize this algorithm for better performance",
                    "Refactor this code to follow best practices",
                    "Find potential memory leaks in this C++ code",
                    "Suggest improvements for this API design",
                    "Review this code for maintainability",
                    "Check for race conditions in this concurrent code",
                    "Analyze code complexity and suggest simplifications",
                ],
                "models": [
                    ("openai", "gpt-4o", 0.95),
                    ("anthropic", "claude-3-5-sonnet", 0.90),
                    ("openai", "gpt-3.5-turbo", 0.70),
                ],
            },
            "simple_qa": {
                "examples": [
                    "What is the capital of France?",
                    "How do you say hello in Spanish?",
                    "What is 2 + 2?",
                    "Define machine learning",
                    "What time is it in Tokyo?",
                    "Convert 100 fahrenheit to celsius",
                    "What is the population of New York?",
                    "Who invented the telephone?",
                    "What is the chemical formula for water?",
                    "How many days are in a leap year?",
                ],
                "models": [
                    ("google", "gemini-1.5-flash", 0.90),
                    ("mistral", "mistral-small", 0.85),
                    ("anthropic", "claude-3-haiku", 0.80),
                    ("openai", "gpt-3.5-turbo", 0.75),
                ],
            },
            "complex_analysis": {
                "examples": [
                    "Analyze the pros and cons of microservices architecture",
                    "Explain the economic impact of AI on job markets",
                    "Compare different machine learning algorithms for text classification",
                    "Design a scalable system for real-time data processing",
                    "Evaluate the security implications of cloud migration",
                    "Analyze market trends in renewable energy",
                    "Compare database technologies for high-traffic applications",
                    "Evaluate the effectiveness of different marketing strategies",
                    "Analyze the environmental impact of cryptocurrency mining",
                    "Compare programming paradigms for large-scale systems",
                ],
                "models": [
                    ("anthropic", "claude-3-5-sonnet", 0.95),
                    ("openai", "gpt-4o", 0.90),
                    ("anthropic", "claude-3-opus", 0.85),
                ],
            },
            "creative_writing": {
                "examples": [
                    "Write a short story about space exploration",
                    "Create a poem about nature",
                    "Draft a marketing email for a new product",
                    "Write a compelling product description",
                    "Generate creative social media captions",
                    "Create a fictional character backstory",
                    "Write dialogue for a dramatic scene",
                    "Compose a song about friendship",
                    "Create a brand slogan for a tech startup",
                    "Write a persuasive essay about climate change",
                ],
                "models": [
                    ("openai", "gpt-4o", 0.95),
                    ("anthropic", "claude-3-5-sonnet", 0.85),
                    ("openai", "gpt-3.5-turbo", 0.80),
                ],
            },
            "data_analysis": {
                "examples": [
                    "Analyze this dataset for trends",
                    "Create a statistical summary of this data",
                    "Identify outliers in this dataset",
                    "Perform correlation analysis on these variables",
                    "Generate insights from customer behavior data",
                    "Create visualizations for sales data",
                    "Analyze A/B test results",
                    "Identify patterns in user engagement metrics",
                    "Perform time series analysis on stock prices",
                    "Analyze survey responses for key insights",
                ],
                "models": [
                    ("openai", "gpt-4o", 0.90),
                    ("anthropic", "claude-3-5-sonnet", 0.85),
                    ("openai", "gpt-3.5-turbo", 0.75),
                ],
            },
        }

        # Generate embeddings for each route category
        for route_name, route_data in route_definitions.items():
            examples = route_data["examples"]
            models = route_data["models"]

            try:
                # Generate embeddings for all examples
                embeddings = self.encoder.encode(examples)

                # Use mean embedding as route representative
                mean_embedding = np.mean(embeddings, axis=0)

                self.route_embeddings[route_name] = mean_embedding
                self.route_examples[route_name] = examples
                self.route_model_mapping[route_name] = models

                logger.debug(
                    "route_embeddings_generated",
                    route=route_name,
                    examples_count=len(examples),
                    embedding_dim=mean_embedding.shape[0],
                )

            except Exception as e:
                logger.error("failed_to_generate_embeddings", route=route_name, error=str(e))
                raise RoutingError(f"Failed to generate embeddings for route {route_name}: {e}") from e

    async def analyze_prompt(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """Analyze prompt using semantic similarity."""
        # Combine all message content
        full_text = " ".join([msg.content for msg in messages if msg.content])

        try:
            # Generate embedding for the prompt
            prompt_embedding = self.encoder.encode([full_text])[0]

            # Calculate similarities to all routes
            similarities = {}
            for route_name, route_embedding in self.route_embeddings.items():
                similarity = self._calculate_cosine_similarity(prompt_embedding, route_embedding)
                similarities[route_name] = similarity

            # Find the best matching route
            best_route = max(similarities.items(), key=lambda x: x[1])
            route_name, similarity_score = best_route

            analysis = {
                "total_length": len(full_text),
                "message_count": len(messages),
                "word_count": len(full_text.split()),
                "semantic_route": route_name,
                "similarity_score": similarity_score,
                "all_similarities": similarities,
                "confidence_score": similarity_score,
                "task_type": (route_name if similarity_score > self.similarity_threshold else "general"),
            }

            logger.debug(
                "semantic_analysis_complete",
                route=route_name,
                similarity=similarity_score,
                threshold=self.similarity_threshold,
                task_type=analysis["task_type"],
            )

            return analysis

        except Exception as e:
            logger.error("semantic_analysis_failed", error=str(e))
            # Fallback to basic analysis
            return {
                "total_length": len(full_text),
                "message_count": len(messages),
                "word_count": len(full_text.split()),
                "semantic_route": "general",
                "similarity_score": 0.0,
                "confidence_score": 0.0,
                "task_type": "general",
                "error": str(e),
            }

    def _calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def _route_request(
        self,
        messages: list[ChatMessage],
        analysis: dict[str, Any],
        user_id: str | None,
        constraints: dict[str, Any] | None,
    ) -> tuple[str, str, str, float]:
        """Route request based on semantic analysis."""
        task_type = analysis["task_type"]
        confidence = analysis["confidence_score"]

        # Get model preferences for the detected route
        if task_type in self.route_model_mapping:
            preferences = self.route_model_mapping[task_type]
        else:
            # Fallback to general routing
            preferences = [
                ("openai", "gpt-3.5-turbo", 0.75),
                ("anthropic", "claude-3-haiku", 0.70),
                ("google", "gemini-1.5-flash", 0.65),
            ]

        # Apply constraints if provided
        if constraints:
            preferences = self._filter_by_constraints(preferences, constraints)

        # Select the best available model
        for provider, model, model_confidence in preferences:
            if self._is_provider_available(provider):
                # Calculate combined confidence
                combined_confidence = min(1.0, (confidence + model_confidence) / 2)

                # Generate reasoning
                reasoning = self._generate_reasoning(analysis, provider, model, task_type)

                return provider, model, reasoning, combined_confidence

        # Fallback
        return "openai", "gpt-3.5-turbo", "Semantic routing fallback", 0.5

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

        return filtered or preferences

    def _is_provider_available(self, provider: str) -> bool:
        """Check if a provider is available."""
        # Placeholder - would check actual provider status
        return True

    def _estimate_model_cost(self, provider: str, model: str) -> float:
        """Estimate cost for a model."""
        cost_estimates = {
            ("openai", "gpt-4o"): 0.01,
            ("openai", "gpt-3.5-turbo"): 0.002,
            ("anthropic", "claude-3-5-sonnet"): 0.008,
            ("anthropic", "claude-3-haiku"): 0.001,
            ("anthropic", "claude-3-opus"): 0.015,
            ("google", "gemini-1.5-flash"): 0.0005,
            ("mistral", "mistral-small"): 0.0003,
        }
        return cost_estimates.get((provider, model), 0.005)

    def _generate_reasoning(self, analysis: dict[str, Any], provider: str, model: str, task_type: str) -> str:
        """Generate reasoning for the routing decision."""
        similarity = analysis.get("similarity_score", 0.0)
        semantic_route = analysis.get("semantic_route", "unknown")

        reasoning_parts = [
            f"Semantic analysis classified as '{semantic_route}'",
            f"Similarity score: {similarity:.3f}",
            f"Selected {provider}/{model} for {task_type} tasks",
        ]

        if similarity < self.similarity_threshold:
            reasoning_parts.append(f"Low confidence (< {self.similarity_threshold}), using general routing")

        return ". ".join(reasoning_parts)

    def add_training_example(self, text: str, route: str) -> bool:
        """Add a new training example to improve routing accuracy."""
        try:
            if route not in self.route_examples:
                logger.warning("unknown_route_for_training", route=route)
                return False

            # Add to examples
            self.route_examples[route].append(text)

            # Regenerate embeddings for this route
            examples = self.route_examples[route]
            embeddings = self.encoder.encode(examples)
            mean_embedding = np.mean(embeddings, axis=0)
            self.route_embeddings[route] = mean_embedding

            logger.info(
                "training_example_added",
                route=route,
                text_length=len(text),
                total_examples=len(examples),
            )

            return True

        except Exception as e:
            logger.error("failed_to_add_training_example", error=str(e))
            return False
