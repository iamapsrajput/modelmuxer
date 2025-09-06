# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Test that HeuristicRouter doesn't have variable scope issues on early exceptions.

This test verifies that the router handles exceptions gracefully without
UnboundLocalError when preferences variable might not be initialized.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from app.core.exceptions import BudgetExceededError, NoProvidersAvailableError
from app.models import ChatMessage
from app.router import HeuristicRouter


class TestHeuristicRouterScope:
    """Test HeuristicRouter variable scope handling."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a minimal router with mocked dependencies
        self.router = HeuristicRouter()

        # Mock the provider registry to return empty dict
        self.router.provider_registry_fn = Mock(return_value={})

        # Mock settings to avoid configuration issues
        self.router.settings = Mock()
        self.router.settings.router_thresholds.max_estimated_usd_per_request = 0.50
        self.router.settings.pricing.min_tokens_in_floor = 50
        self.router.settings.server.debug = False
        self.router.settings.features.mode = "test"

    async def test_early_exception_does_not_cause_unbound_local_error(self):
        """Test that early exceptions don't cause UnboundLocalError."""
        messages = [ChatMessage(role="user", content="Hello", name=None)]

        # Mock analyze_prompt to raise an exception early
        with patch.object(self.router, "analyze_prompt", side_effect=Exception("Early failure")):
            with pytest.raises(Exception) as exc_info:
                await self.router.select_model(messages=messages)

            # Should raise the original exception, not UnboundLocalError
            assert "Early failure" in str(exc_info.value)
            assert "UnboundLocalError" not in str(exc_info.value)

    async def test_intent_classification_exception_does_not_cause_unbound_local_error(self):
        """Test that intent classification exceptions don't cause UnboundLocalError."""
        messages = [ChatMessage(role="user", content="Hello", name=None)]

        # Mock classify_intent to raise an exception
        with patch(
            "app.router.classify_intent", side_effect=Exception("Intent classification failed")
        ):
            # Mock analyze_prompt to work normally
            with patch.object(
                self.router,
                "analyze_prompt",
                return_value={
                    "total_length": 5,
                    "message_count": 1,
                    "has_code": False,
                    "code_confidence": 0.0,
                    "has_complexity": False,
                    "complexity_confidence": 0.0,
                    "is_simple": False,
                    "simple_confidence": 0.0,
                    "detected_languages": [],
                    "task_type": "general",
                },
            ):
                # Mock estimate_tokens to work normally
                with patch("app.router.estimate_tokens", return_value=(50, 100)):
                    # Mock estimator to work normally
                    with patch.object(self.router, "estimator") as mock_estimator:
                        mock_estimate = Mock()
                        mock_estimate.usd = 0.10
                        mock_estimate.eta_ms = 1000
                        mock_estimate.tokens_in = 50
                        mock_estimate.tokens_out = 100
                        mock_estimate.model_key = "test:model"
                        mock_estimator.estimate.return_value = mock_estimate

                        # Should not raise UnboundLocalError
                        result = await self.router.select_model(messages=messages)
                        assert result[0] == "openai"  # First in general preferences
                        assert result[1] == "gpt-3.5-turbo"

    async def test_span_context_exception_does_not_cause_unbound_local_error(self):
        """Test that span context exceptions don't cause UnboundLocalError."""
        messages = [ChatMessage(role="user", content="Hello", name=None)]

        # Mock classify_intent to return proper dict to avoid Mock conversion errors
        with patch("app.router.classify_intent", return_value={"label": "general", "confidence": 0.5, "signals": {}, "method": "test"}):
            # Mock start_span_async to raise an exception (router uses start_span_async, not start_span)
            with patch("app.router.start_span_async", side_effect=Exception("Span context failed")):
                with pytest.raises(Exception) as exc_info:
                    await self.router.select_model(messages=messages)

                # Should raise the original exception, not UnboundLocalError
                assert "Span context failed" in str(exc_info.value)
                assert "UnboundLocalError" not in str(exc_info.value)

    async def test_estimate_tokens_exception_does_not_cause_unbound_local_error(self):
        """Test that estimate_tokens exceptions don't cause UnboundLocalError."""
        messages = [ChatMessage(role="user", content="Hello", name=None)]

        # Mock estimate_tokens to raise an exception
        with patch("app.router.estimate_tokens", side_effect=Exception("Token estimation failed")):
            # Mock analyze_prompt to work normally
            with patch.object(
                self.router,
                "analyze_prompt",
                return_value={
                    "total_length": 5,
                    "message_count": 1,
                    "has_code": False,
                    "code_confidence": 0.0,
                    "has_complexity": False,
                    "complexity_confidence": 0.0,
                    "is_simple": False,
                    "simple_confidence": 0.0,
                    "detected_languages": [],
                    "task_type": "general",
                },
            ):
                with pytest.raises(Exception) as exc_info:
                    await self.router.select_model(messages=messages)

                # Should raise the original exception, not UnboundLocalError
                assert "Token estimation failed" in str(exc_info.value)
                assert "UnboundLocalError" not in str(exc_info.value)

    async def test_empty_preferences_handled_gracefully(self):
        """Test that empty preferences are handled gracefully."""
        messages = [ChatMessage(role="user", content="Hello", name=None)]

        # Mock model_preferences to return empty list
        self.router.model_preferences = {"general": []}

        # Mock analyze_prompt to work normally
        with patch.object(
            self.router,
            "analyze_prompt",
            return_value={
                "task_type": "general",
                "code_confidence": 0.0,
                "complexity_confidence": 0.0,
                "simple_confidence": 0.0,
            },
        ):
            # Mock estimate_tokens to work normally
            with patch("app.router.estimate_tokens", return_value=(50, 100)):
                # Should raise NoProvidersAvailableError when no providers are available
                with pytest.raises(NoProvidersAvailableError) as exc_info:
                    await self.router.select_model(messages=messages)

                assert "No LLM providers available" in str(exc_info.value)

    async def test_preferences_initialization_prevents_unbound_local_error(self):
        """Test that preferences initialization prevents UnboundLocalError in fallback logic."""
        messages = [ChatMessage(role="user", content="Hello", name=None)]

        # Mock analyze_prompt to work normally
        with patch.object(
            self.router,
            "analyze_prompt",
            return_value={
                "task_type": "general",
                "code_confidence": 0.0,
                "complexity_confidence": 0.0,
                "simple_confidence": 0.0,
            },
        ):
            # Mock estimate_tokens to work normally
            with patch("app.router.estimate_tokens", return_value=(50, 100)):
                # Mock estimator to raise exception during estimation
                with patch.object(self.router, "estimator") as mock_estimator:
                    mock_estimator.estimate.side_effect = Exception("Estimator failed")

                    # Should not raise UnboundLocalError, should handle gracefully
                    with pytest.raises(BudgetExceededError):
                        await self.router.select_model(messages=messages)
