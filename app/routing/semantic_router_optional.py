# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Optional semantic routing implementation that gracefully handles missing dependencies.

This module provides an alternative to the full semantic router that can work
without sentence-transformers for environments with security constraints.
"""

from typing import Any

import structlog

from ..core.exceptions import ConfigurationError, RoutingError
from ..models import ChatMessage
from .base_router import BaseRouter

logger = structlog.get_logger(__name__)

# Try to import sentence-transformers, fallback to None if not available
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    np = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class OptionalSemanticRouter(BaseRouter):
    """
    Semantic router with optional sentence-transformers dependency.

    This router can work in two modes:
    1. Full mode: Uses sentence-transformers for ML-based semantic routing
    2. Fallback mode: Uses simple keyword matching when sentence-transformers is unavailable
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__("semantic", config)

        # Configuration
        self.model_name = self.config.get("model_name", "all-MiniLM-L6-v2")
        self.similarity_threshold = self.config.get("similarity_threshold", 0.6)
        self.cache_embeddings = self.config.get("cache_embeddings", True)
        self.use_fallback = self.config.get("use_fallback", True)

        # Initialize the encoder if available
        self.encoder = None
        self.use_ml_mode = False

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer(self.model_name)
                self.use_ml_mode = True
                logger.info("semantic_router_initialized_ml_mode", model=self.model_name)
            except Exception as e:
                if not self.use_fallback:
                    raise ConfigurationError(
                        f"Failed to initialize sentence transformer: {e}"
                    ) from e
                logger.warning("semantic_router_ml_init_failed_using_fallback", error=str(e))

        if not self.use_ml_mode:
            if not self.use_fallback:
                raise ConfigurationError(
                    "Sentence transformers not available and fallback disabled"
                )
            logger.info("semantic_router_initialized_fallback_mode")

        # Route definitions
        self.route_embeddings = {}
        self.route_examples = {}
        self.route_model_mapping = {}
        self.route_keywords = {}

        # Load or create route definitions
        self._initialize_routes()

    def _initialize_routes(self) -> None:
        """Initialize route definitions with either ML embeddings or keyword patterns."""
        # Define route categories with examples and keywords
        route_definitions = {
            "code_generation": {
                "examples": [
                    "Write a Python function to sort a list",
                    "Create a React component for a button",
                    "Implement a binary search algorithm in Java",
                    "Write a SQL query to find duplicate records",
                    "Create a REST API endpoint in Node.js",
                ],
                "keywords": [
                    "write",
                    "create",
                    "implement",
                    "build",
                    "code",
                    "function",
                    "class",
                    "algorithm",
                    "api",
                    "endpoint",
                    "component",
                    "query",
                    "script",
                    "program",
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
                ],
                "keywords": [
                    "review",
                    "analyze",
                    "check",
                    "optimize",
                    "refactor",
                    "improve",
                    "bug",
                    "performance",
                    "security",
                    "vulnerability",
                    "best practices",
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
                ],
                "keywords": [
                    "what",
                    "how",
                    "when",
                    "where",
                    "why",
                    "who",
                    "define",
                    "explain",
                    "tell me",
                    "capital",
                    "population",
                    "time",
                    "convert",
                    "calculate",
                ],
                "models": [
                    ("google", "gemini-1.5-flash", 0.90),
                    ("mistral", "mistral-small", 0.85),
                    ("anthropic", "claude-3-haiku", 0.80),
                ],
            },
            "complex_analysis": {
                "examples": [
                    "Analyze the pros and cons of microservices architecture",
                    "Explain the economic impact of AI on job markets",
                    "Compare different machine learning algorithms",
                    "Design a scalable system for real-time data processing",
                    "Evaluate the security implications of cloud migration",
                ],
                "keywords": [
                    "analyze",
                    "compare",
                    "evaluate",
                    "design",
                    "architecture",
                    "system",
                    "pros and cons",
                    "impact",
                    "implications",
                    "trade-offs",
                    "strategy",
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
                ],
                "keywords": [
                    "write",
                    "create",
                    "draft",
                    "generate",
                    "story",
                    "poem",
                    "email",
                    "description",
                    "caption",
                    "creative",
                    "marketing",
                    "content",
                ],
                "models": [
                    ("openai", "gpt-4o", 0.95),
                    ("anthropic", "claude-3-5-sonnet", 0.85),
                    ("openai", "gpt-3.5-turbo", 0.80),
                ],
            },
        }

        if self.use_ml_mode:
            self._initialize_ml_routes(route_definitions)
        else:
            self._initialize_keyword_routes(route_definitions)

    def _initialize_ml_routes(self, route_definitions: dict) -> None:
        """Initialize routes using ML embeddings."""
        for route_name, route_data in route_definitions.items():
            examples = route_data["examples"]
            models = route_data["models"]

            try:
                # Generate embeddings for all examples
                embeddings = self.encoder.encode(examples)
                mean_embedding = np.mean(embeddings, axis=0)

                self.route_embeddings[route_name] = mean_embedding
                self.route_examples[route_name] = examples
                self.route_model_mapping[route_name] = models

                logger.debug(
                    "ml_route_embeddings_generated",
                    route=route_name,
                    examples_count=len(examples),
                    embedding_dim=mean_embedding.shape[0],
                )

            except Exception as e:
                logger.error("failed_to_generate_ml_embeddings", route=route_name, error=str(e))
                raise RoutingError(
                    f"Failed to generate embeddings for route {route_name}: {e}"
                ) from e

    def _initialize_keyword_routes(self, route_definitions: dict) -> None:
        """Initialize routes using keyword patterns."""
        for route_name, route_data in route_definitions.items():
            keywords = route_data["keywords"]
            models = route_data["models"]

            self.route_keywords[route_name] = [kw.lower() for kw in keywords]
            self.route_model_mapping[route_name] = models

            logger.debug(
                "keyword_route_patterns_loaded",
                route=route_name,
                keywords_count=len(keywords),
            )

    async def analyze_prompt(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """
        Analyze the prompt and determine routing based on semantic similarity or keywords.

        Args:
            messages: List of chat messages to analyze

        Returns:
            Analysis results with routing recommendation
        """
        try:
            # Extract text content
            combined_text = self._extract_text_content(messages)

            if self.use_ml_mode:
                return await self._analyze_with_ml(combined_text)
            else:
                return await self._analyze_with_keywords(combined_text)

        except Exception as e:
            logger.error("semantic_analysis_failed", error=str(e))
            raise RoutingError(f"Semantic analysis failed: {e}") from e

    async def _analyze_with_ml(self, text: str) -> dict[str, Any]:
        """Analyze using ML embeddings."""
        # Generate embedding for the input text
        text_embedding = self.encoder.encode([text])[0]

        # Calculate similarities with all route embeddings
        similarities = {}
        for route_name, route_embedding in self.route_embeddings.items():
            # Calculate cosine similarity
            similarity = np.dot(text_embedding, route_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(route_embedding)
            )
            similarities[route_name] = float(similarity)

        # Find the best match
        best_route = max(similarities, key=similarities.get)
        best_similarity = similarities[best_route]

        logger.info(
            "ml_semantic_analysis_complete",
            best_route=best_route,
            similarity=best_similarity,
            all_similarities=similarities,
        )

        # Check if similarity meets threshold
        if best_similarity >= self.similarity_threshold:
            recommended_models = self.route_model_mapping[best_route]
        else:
            # Fall back to default models
            recommended_models = [("openai", "gpt-3.5-turbo", 0.8)]
            best_route = "fallback"

        return {
            "route_category": best_route,
            "confidence": best_similarity,
            "recommended_models": recommended_models,
            "analysis_method": "ml_embeddings",
            "all_similarities": similarities,
        }

    async def _analyze_with_keywords(self, text: str) -> dict[str, Any]:
        """Analyze using keyword matching."""
        text_lower = text.lower()

        # Calculate keyword matches for each route
        route_scores = {}
        for route_name, keywords in self.route_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            score = matches / len(keywords) if keywords else 0
            route_scores[route_name] = score

        # Find the best match
        best_route = max(route_scores, key=route_scores.get)
        best_score = route_scores[best_route]

        logger.info(
            "keyword_semantic_analysis_complete",
            best_route=best_route,
            score=best_score,
            all_scores=route_scores,
        )

        # Check if score meets threshold (adjusted for keyword matching)
        keyword_threshold = self.similarity_threshold * 0.5  # Lower threshold for keywords
        if best_score >= keyword_threshold:
            recommended_models = self.route_model_mapping[best_route]
        else:
            # Fall back to default models
            recommended_models = [("openai", "gpt-3.5-turbo", 0.8)]
            best_route = "fallback"

        return {
            "route_category": best_route,
            "confidence": best_score,
            "recommended_models": recommended_models,
            "analysis_method": "keyword_matching",
            "all_scores": route_scores,
        }

    def _extract_text_content(self, messages: list[ChatMessage]) -> str:
        """Extract text content from messages."""
        contents = []
        for message in messages:
            if hasattr(message, "content") and message.content:
                if isinstance(message.content, str):
                    contents.append(message.content)
                elif isinstance(message.content, list):
                    for item in message.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            contents.append(item.get("text", ""))
                        elif isinstance(item, str):
                            contents.append(item)

        return " ".join(contents).strip()

    async def route_request(self, messages: list[ChatMessage]) -> tuple[str, str]:
        """
        Route the request to the appropriate provider and model.

        Args:
            messages: List of chat messages

        Returns:
            Tuple of (provider, model) for the request
        """
        analysis = await self.analyze_prompt(messages)
        recommended_models = analysis["recommended_models"]

        if recommended_models:
            # Return the highest confidence model
            provider, model, confidence = recommended_models[0]
            logger.info(
                "semantic_routing_decision",
                provider=provider,
                model=model,
                confidence=confidence,
                route_category=analysis["route_category"],
                method=analysis["analysis_method"],
            )
            return provider, model

        # Fallback to default
        logger.warning("semantic_routing_fallback")
        return "openai", "gpt-3.5-turbo"
