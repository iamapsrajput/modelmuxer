# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""
Tests for the cost tracking and budget management system.
"""


import pytest

from app.cost_tracker import CostTracker
from app.models import ChatMessage


class TestCostTracker:
    """Test the basic cost tracking functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.cost_tracker = CostTracker()

    def test_calculate_cost_openai(self) -> None:
        """Test cost calculation for OpenAI models."""
        # Test GPT-4o-mini (cheapest)
        cost = self.cost_tracker.calculate_cost("openai", "gpt-4o-mini", 1000, 500)
        expected = (1000 / 1_000_000) * 0.00015 + (500 / 1_000_000) * 0.0006
        assert abs(cost - expected) < 0.0001

        # Test GPT-4o
        cost = self.cost_tracker.calculate_cost("openai", "gpt-4o", 1000, 500)
        expected = (1000 / 1_000_000) * 0.005 + (500 / 1_000_000) * 0.015
        assert abs(cost - expected) < 0.0001

    def test_calculate_cost_anthropic(self) -> None:
        """Test cost calculation for Anthropic models."""
        cost = self.cost_tracker.calculate_cost("anthropic", "claude-3-haiku-20240307", 1000, 500)
        expected = (1000 / 1_000_000) * 0.00025 + (500 / 1_000_000) * 0.00125
        assert abs(cost - expected) < 0.0001

    def test_calculate_cost_mistral(self) -> None:
        """Test cost calculation for Mistral models."""
        cost = self.cost_tracker.calculate_cost("mistral", "mistral-small", 1000, 500)
        expected = (1000 / 1_000_000) * 0.0002 + (500 / 1_000_000) * 0.0006
        assert abs(cost - expected) < 0.0001

    def test_calculate_cost_unknown_provider(self) -> None:
        """Test cost calculation for unknown provider."""
        cost = self.cost_tracker.calculate_cost("unknown", "model", 1000, 500)
        assert cost == 0.0

    def test_calculate_cost_unknown_model(self) -> None:
        """Test cost calculation for unknown model."""
        cost = self.cost_tracker.calculate_cost("openai", "unknown-model", 1000, 500)
        assert cost == 0.0

    def test_estimate_request_cost(self) -> None:
        """Test request cost estimation."""
        messages = [ChatMessage(role="user", content="What is Python?", name=None)]
        provider = "openai"
        model = "gpt-4o-mini"

        estimate = self.cost_tracker.estimate_request_cost(messages, provider, model)

        assert "input_tokens" in estimate
        assert "estimated_output_tokens" in estimate
        assert "estimated_cost" in estimate
        assert "provider" in estimate
        assert "model" in estimate
        assert estimate["provider"] == provider
        assert estimate["model"] == model
        assert estimate["estimated_cost"] > 0

    def test_get_cheapest_model(self) -> None:
        """Test getting the cheapest model for a task."""
        cheapest = self.cost_tracker.get_cheapest_model_for_task("simple")

        assert "provider" in cheapest
        assert "model" in cheapest
        # Should return a valid provider/model combination
        assert cheapest["provider"] in ["openai", "anthropic", "mistral", "google", "groq"]

    def test_compare_model_costs(self) -> None:
        """Test comparing costs across multiple models."""
        messages = [ChatMessage(role="user", content="What is Python?", name=None)]
        models = [
            {"provider": "openai", "model": "gpt-4o-mini"},
            {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
            {"provider": "mistral", "model": "mistral-small"},
        ]

        comparisons = self.cost_tracker.compare_model_costs(messages, models)

        assert len(comparisons) == 3
        # Should be sorted by cost (cheapest first)
        for i in range(len(comparisons) - 1):
            assert comparisons[i]["estimated_cost"] <= comparisons[i + 1]["estimated_cost"]

    def test_count_tokens(self) -> None:
        """Test token counting functionality."""
        messages = [
            ChatMessage(role="user", content="What is Python?", name=None),
            ChatMessage(role="assistant", content="Python is a programming language.", name=None),
        ]

        token_count = self.cost_tracker.count_tokens(messages, "openai", "gpt-3.5-turbo")

        assert isinstance(token_count, int)
        assert token_count > 0
        # Should be reasonable for the given messages
        assert token_count < 100

        # Note: AdvancedCostTracker tests are commented out as they require async implementation
        # and database setup. These should be implemented as integration tests.

        # class TestAdvancedCostTracker:
        #     """Test the advanced cost tracking with budget management."""
        #
        #     def setup_method(self) -> None:
        #         """Set up test fixtures."""
        #         self.advanced_tracker = AdvancedCostTracker()
        #
        #     async def test_set_budget(self) -> None:
        #         """Test setting user budget."""
        #         user_id = "test_user"
        #         budget_type = BudgetPeriod.DAILY
        #         budget_limit = 10.0
        #
        #         await self.advanced_tracker.set_budget(user_id, budget_type, budget_limit)
        #         status = await self.advanced_tracker.get_budget_status(user_id)
        #
        #         assert status is not None

    # Advanced tracker tests are commented out as they require async implementation
    # and proper database setup. These should be implemented as integration tests.
    pass


@pytest.mark.integration
class TestCostTrackingIntegration:
    """Integration tests for cost tracking with routing."""

    def test_cost_aware_routing(self) -> None:
        """Test that routing considers cost constraints."""
        from app.router import HeuristicRouter

        router = HeuristicRouter()
        messages = [ChatMessage(role="user", content="Simple question", name=None)]

        # Test with very low budget
        provider, model, reason = router.select_model(messages, budget_constraint=0.0001)

        # Should select a cheap model
        assert provider in ["mistral", "openai"]
        if provider == "openai":
            assert "mini" in model.lower()
        elif provider == "mistral":
            assert "small" in model.lower()

    def test_budget_enforcement_in_routing(self) -> None:
        """Test that budget constraints are enforced in routing."""
        # This test requires the enhanced cost tracker to be properly integrated
        # For now, we just test that the router respects budget constraints
        from app.router import HeuristicRouter

        router = HeuristicRouter()
        messages = [ChatMessage(role="user", content="Expensive request", name=None)]

        # Test with extremely low budget - should select cheapest model
        provider, model, reason = router.select_model(messages, budget_constraint=0.00001)

        # Should select the cheapest available model
        assert provider in ["mistral", "openai"]
        if provider == "openai":
            assert "mini" in model.lower()
        elif provider == "mistral":
            assert "small" in model.lower()


if __name__ == "__main__":
    pytest.main([__file__])
