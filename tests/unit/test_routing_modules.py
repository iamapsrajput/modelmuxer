# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Comprehensive tests for routing modules to improve coverage.
"""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.models import ChatMessage
from app.routing.heuristic_router import EnhancedHeuristicRouter
from app.routing.hybrid_router import HybridRouter
from app.routing.semantic_router import SemanticRouter
from app.routing.base_router import BaseRouter


class TestBaseRouter:
    """Test BaseRouter functionality."""

    def test_base_router_abstract(self):
        """Test that BaseRouter is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseRouter()

    def test_base_router_interface(self):
        """Test BaseRouter interface definition."""

        # Create a concrete implementation for testing
        class ConcreteRouter(BaseRouter):
            async def select_provider_and_model(self, messages, **kwargs):
                return "provider", "model", "reason", {}, {}

        router = ConcreteRouter()
        assert hasattr(router, "select_provider_and_model")


class TestEnhancedHeuristicRouter:
    """Test EnhancedHeuristicRouter functionality."""

    @pytest.fixture
    def router_config(self):
        """Create router configuration."""
        return {
            "enabled": True,
            "complexity_threshold": 0.7,
            "code_detection_keywords": ["function", "class", "def", "import"],
            "math_detection_keywords": ["calculate", "equation", "solve"],
            "creative_detection_keywords": ["story", "poem", "creative"],
            "model_preferences": {
                "code": [("openai", "gpt-4")],
                "math": [("openai", "gpt-4")],
                "creative": [("anthropic", "claude-3")],
                "general": [("openai", "gpt-3.5-turbo")],
            },
            "fallback_model": ("openai", "gpt-3.5-turbo"),
            "use_intent_classification": True,
        }

    @pytest.fixture
    def router(self, router_config):
        """Create EnhancedHeuristicRouter instance."""
        with patch("app.routing.heuristic_router.load_price_table", return_value={}):
            return EnhancedHeuristicRouter(router_config)

    async def test_select_provider_and_model_code(self, router):
        """Test model selection for code-related tasks."""
        messages = [
            ChatMessage(role="user", content="Write a Python function to sort a list", name=None)
        ]

        with patch.object(router, "_classify_task", return_value="code"):
            with patch.object(router, "_get_available_models", return_value=[("openai", "gpt-4")]):
                provider, model, reason, intent, estimate = await router.select_provider_and_model(
                    messages
                )

                assert provider == "openai"
                assert model == "gpt-4"
                assert "code" in reason.lower()

    async def test_select_provider_and_model_math(self, router):
        """Test model selection for math-related tasks."""
        messages = [ChatMessage(role="user", content="Calculate the derivative of x^2", name=None)]

        with patch.object(router, "_classify_task", return_value="math"):
            with patch.object(router, "_get_available_models", return_value=[("openai", "gpt-4")]):
                provider, model, reason, intent, estimate = await router.select_provider_and_model(
                    messages
                )

                assert provider == "openai"
                assert model == "gpt-4"

    async def test_select_provider_and_model_creative(self, router):
        """Test model selection for creative tasks."""
        messages = [
            ChatMessage(role="user", content="Write a short story about a robot", name=None)
        ]

        with patch.object(router, "_classify_task", return_value="creative"):
            with patch.object(
                router, "_get_available_models", return_value=[("anthropic", "claude-3")]
            ):
                provider, model, reason, intent, estimate = await router.select_provider_and_model(
                    messages
                )

                assert provider == "anthropic"
                assert model == "claude-3"

    async def test_select_provider_and_model_fallback(self, router):
        """Test fallback model selection."""
        messages = [ChatMessage(role="user", content="Hello", name=None)]

        with patch.object(router, "_classify_task", return_value="general"):
            with patch.object(router, "_get_available_models", return_value=[]):
                with patch.object(
                    router, "_get_fallback_model", return_value=("openai", "gpt-3.5-turbo")
                ):
                    provider, model, reason, intent, estimate = (
                        await router.select_provider_and_model(messages)
                    )

                    assert provider == "openai"
                    assert model == "gpt-3.5-turbo"
                    assert "fallback" in reason.lower()

    def test_classify_task_code(self, router):
        """Test task classification for code."""
        messages = [
            ChatMessage(role="user", content="def calculate_sum(a, b): return a + b", name=None)
        ]

        task_type = router._classify_task(messages)
        assert task_type == "code"

    def test_classify_task_math(self, router):
        """Test task classification for math."""
        messages = [ChatMessage(role="user", content="Solve the equation 2x + 5 = 15", name=None)]

        task_type = router._classify_task(messages)
        assert task_type == "math"

    def test_classify_task_creative(self, router):
        """Test task classification for creative content."""
        messages = [
            ChatMessage(role="user", content="Write a creative story about space", name=None)
        ]

        task_type = router._classify_task(messages)
        assert task_type == "creative"

    def test_classify_task_general(self, router):
        """Test task classification for general content."""
        messages = [ChatMessage(role="user", content="What is the weather today?", name=None)]

        task_type = router._classify_task(messages)
        assert task_type == "general"

    def test_calculate_complexity(self, router):
        """Test complexity calculation."""
        messages = [ChatMessage(role="user", content="Simple question", name=None)]

        complexity = router._calculate_complexity(messages)
        assert 0 <= complexity <= 1

    def test_calculate_complexity_high(self, router):
        """Test high complexity calculation."""
        long_content = " ".join(["complex"] * 100)
        messages = [ChatMessage(role="user", content=long_content, name=None)]

        complexity = router._calculate_complexity(messages)
        assert complexity > 0.5

    def test_get_available_models(self, router):
        """Test getting available models."""
        with patch.object(
            router, "provider_registry", return_value={"openai": Mock(), "anthropic": Mock()}
        ):
            models = router._get_available_models("code")
            assert isinstance(models, list)

    def test_get_fallback_model(self, router):
        """Test getting fallback model."""
        fallback = router._get_fallback_model()
        assert fallback == router.config["fallback_model"]

    async def test_estimate_cost(self, router):
        """Test cost estimation."""
        messages = [ChatMessage(role="user", content="Test message", name=None)]

        with patch.object(router, "cost_estimator") as mock_estimator:
            mock_estimator.estimate.return_value = {"usd": 0.01, "tokens_in": 10, "tokens_out": 20}

            estimate = await router._estimate_cost("openai", "gpt-4", messages)
            assert estimate["usd"] == 0.01


class TestHybridRouter:
    """Test HybridRouter functionality."""

    @pytest.fixture
    def heuristic_router(self):
        """Create mock heuristic router."""
        router = Mock()
        router.select_provider_and_model = AsyncMock(
            return_value=("openai", "gpt-4", "heuristic reason", {"label": "code"}, {"usd": 0.01})
        )
        return router

    @pytest.fixture
    def semantic_router(self):
        """Create mock semantic router."""
        router = Mock()
        router.select_provider_and_model = AsyncMock(
            return_value=(
                "anthropic",
                "claude-3",
                "semantic reason",
                {"label": "creative"},
                {"usd": 0.02},
            )
        )
        return router

    @pytest.fixture
    def router_config(self):
        """Create hybrid router configuration."""
        return {
            "enabled": True,
            "heuristic_weight": 0.6,
            "semantic_weight": 0.4,
            "confidence_threshold": 0.7,
            "use_voting": True,
            "fallback_to_heuristic": True,
        }

    @pytest.fixture
    def router(self, heuristic_router, semantic_router, router_config):
        """Create HybridRouter instance."""
        return HybridRouter(
            heuristic_router=heuristic_router, semantic_router=semantic_router, config=router_config
        )

    async def test_select_provider_and_model_heuristic_wins(self, router):
        """Test model selection when heuristic router wins."""
        messages = [ChatMessage(role="user", content="Write code", name=None)]

        provider, model, reason, intent, estimate = await router.select_provider_and_model(messages)

        # With default weights (0.6 heuristic, 0.4 semantic), heuristic should win
        assert provider == "openai"
        assert model == "gpt-4"
        assert "hybrid" in reason.lower()

    async def test_select_provider_and_model_semantic_wins(self, router, semantic_router):
        """Test model selection when semantic router wins."""
        # Adjust weights to favor semantic
        router.config["heuristic_weight"] = 0.3
        router.config["semantic_weight"] = 0.7

        messages = [ChatMessage(role="user", content="Write a story", name=None)]

        provider, model, reason, intent, estimate = await router.select_provider_and_model(messages)

        assert provider == "anthropic"
        assert model == "claude-3"

    async def test_select_provider_and_model_voting(
        self, router, heuristic_router, semantic_router
    ):
        """Test voting mechanism."""
        # Both routers return the same model
        semantic_router.select_provider_and_model.return_value = (
            "openai",
            "gpt-4",
            "semantic reason",
            {"label": "code"},
            {"usd": 0.01},
        )

        messages = [ChatMessage(role="user", content="Test", name=None)]

        provider, model, reason, intent, estimate = await router.select_provider_and_model(messages)

        assert provider == "openai"
        assert model == "gpt-4"
        assert "consensus" in reason.lower() or "agreement" in reason.lower()

    async def test_select_provider_and_model_fallback(self, router, semantic_router):
        """Test fallback to heuristic when semantic fails."""
        semantic_router.select_provider_and_model.side_effect = Exception("Semantic router failed")

        messages = [ChatMessage(role="user", content="Test", name=None)]

        provider, model, reason, intent, estimate = await router.select_provider_and_model(messages)

        assert provider == "openai"
        assert model == "gpt-4"
        assert "fallback" in reason.lower()

    def test_calculate_weighted_score(self, router):
        """Test weighted score calculation."""
        heuristic_result = ("openai", "gpt-4", "reason", {"confidence": 0.8}, {})
        semantic_result = ("anthropic", "claude-3", "reason", {"confidence": 0.9}, {})

        scores = router._calculate_weighted_scores(heuristic_result, semantic_result)

        assert "openai:gpt-4" in scores
        assert "anthropic:claude-3" in scores
        assert scores["openai:gpt-4"] == 0.6 * 0.8  # heuristic_weight * confidence
        assert scores["anthropic:claude-3"] == 0.4 * 0.9  # semantic_weight * confidence

    def test_merge_intents(self, router):
        """Test intent merging."""
        heuristic_intent = {"label": "code", "confidence": 0.8, "method": "heuristic"}
        semantic_intent = {"label": "creative", "confidence": 0.9, "method": "semantic"}

        merged = router._merge_intents(heuristic_intent, semantic_intent)

        assert "heuristic" in merged
        assert "semantic" in merged
        assert merged["heuristic"]["label"] == "code"
        assert merged["semantic"]["label"] == "creative"

    def test_merge_estimates(self, router):
        """Test estimate merging."""
        heuristic_estimate = {"usd": 0.01, "tokens_in": 10, "tokens_out": 20}
        semantic_estimate = {"usd": 0.02, "tokens_in": 15, "tokens_out": 25}

        merged = router._merge_estimates(heuristic_estimate, semantic_estimate)

        # Should average the estimates
        assert merged["usd"] == 0.015
        assert merged["tokens_in"] == 12
        assert merged["tokens_out"] == 22


class TestSemanticRouter:
    """Test SemanticRouter functionality."""

    @pytest.fixture
    def embedding_manager(self):
        """Create mock embedding manager."""
        manager = Mock()
        manager.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        manager.compute_similarity = Mock(return_value=0.85)
        return manager

    @pytest.fixture
    def router_config(self):
        """Create semantic router configuration."""
        return {
            "enabled": True,
            "similarity_threshold": 0.8,
            "use_cache": True,
            "cache_ttl": 3600,
            "model_embeddings": {
                "openai:gpt-4": [0.1, 0.2, 0.3],
                "anthropic:claude-3": [0.4, 0.5, 0.6],
            },
            "task_embeddings": {"code": [0.1, 0.2, 0.3], "creative": [0.4, 0.5, 0.6]},
        }

    @pytest.fixture
    def router(self, embedding_manager, router_config):
        """Create SemanticRouter instance."""
        with patch("app.routing.semantic_router.load_price_table", return_value={}):
            return SemanticRouter(embedding_manager=embedding_manager, config=router_config)

    async def test_select_provider_and_model_similarity_match(self, router, embedding_manager):
        """Test model selection based on similarity."""
        messages = [ChatMessage(role="user", content="Write Python code", name=None)]

        with patch.object(router, "_find_best_match", return_value=("openai", "gpt-4", 0.9)):
            with patch.object(router, "_estimate_cost", return_value={"usd": 0.01}):
                provider, model, reason, intent, estimate = await router.select_provider_and_model(
                    messages
                )

                assert provider == "openai"
                assert model == "gpt-4"
                assert "semantic" in reason.lower()

    async def test_select_provider_and_model_no_match(self, router, embedding_manager):
        """Test model selection when no good match is found."""
        messages = [ChatMessage(role="user", content="Random content", name=None)]

        with patch.object(router, "_find_best_match", return_value=(None, None, 0.5)):
            with patch.object(
                router, "_get_fallback_model", return_value=("openai", "gpt-3.5-turbo")
            ):
                provider, model, reason, intent, estimate = await router.select_provider_and_model(
                    messages
                )

                assert provider == "openai"
                assert model == "gpt-3.5-turbo"
                assert "fallback" in reason.lower()

    async def test_get_message_embedding(self, router, embedding_manager):
        """Test getting message embedding."""
        messages = [ChatMessage(role="user", content="Test message", name=None)]

        embedding = await router._get_message_embedding(messages)
        assert embedding == [0.1, 0.2, 0.3]
        embedding_manager.get_embedding.assert_called_once()

    def test_find_best_match(self, router, embedding_manager):
        """Test finding best matching model."""
        query_embedding = [0.1, 0.2, 0.3]

        provider, model, similarity = router._find_best_match(query_embedding)

        assert provider is not None
        assert model is not None
        assert 0 <= similarity <= 1

    def test_compute_similarity(self, router):
        """Test similarity computation."""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [1.0, 0.0, 0.0]

        similarity = router._compute_similarity(embedding1, embedding2)
        assert similarity == 1.0  # Identical vectors

    def test_compute_similarity_orthogonal(self, router):
        """Test similarity computation for orthogonal vectors."""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]

        similarity = router._compute_similarity(embedding1, embedding2)
        assert similarity == 0.0  # Orthogonal vectors

    async def test_cache_lookup(self, router):
        """Test cache lookup for embeddings."""
        messages = [ChatMessage(role="user", content="Cached message", name=None)]

        # Mock cache hit
        with patch.object(router, "_get_from_cache", return_value=[0.7, 0.8, 0.9]):
            embedding = await router._get_message_embedding(messages)
            assert embedding == [0.7, 0.8, 0.9]

    async def test_cache_store(self, router, embedding_manager):
        """Test storing embeddings in cache."""
        messages = [ChatMessage(role="user", content="New message", name=None)]

        with patch.object(router, "_get_from_cache", return_value=None):
            with patch.object(router, "_store_in_cache") as mock_store:
                embedding = await router._get_message_embedding(messages)
                mock_store.assert_called_once()

    def test_normalize_embedding(self, router):
        """Test embedding normalization."""
        embedding = [3.0, 4.0, 0.0]

        normalized = router._normalize_embedding(embedding)

        # Should be unit vector
        import math

        length = math.sqrt(sum(x**2 for x in normalized))
        assert abs(length - 1.0) < 0.001

    def test_get_fallback_model(self, router):
        """Test getting fallback model."""
        with patch.object(router, "provider_registry", return_value={"openai": Mock()}):
            provider, model = router._get_fallback_model()
            assert provider is not None
            assert model is not None
