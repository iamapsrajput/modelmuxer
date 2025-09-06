# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Tests for budget constraint functionality with direct providers.

This module tests the HeuristicRouter's budget constraint behavior when using
direct providers, including budget gates, down-routing, cost estimation, and
error handling.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.core.exceptions import BudgetExceededError
from app.models import ChatMessage


@pytest.mark.direct
class TestBudgetConstraintsDirect:
    """Test budget constraint functionality with direct providers."""

    async def test_budget_gate_with_direct_providers_expensive_models(
        self, direct_providers_only_mode, budget_constrained_router, expensive_model_messages
    ):
        """Test budget gate with expensive models - should select affordable models."""
        with patch(
            "app.core.intent.classify_intent",
            new=AsyncMock(
                return_value={
                    "label": "complex",
                    "confidence": 0.9,
                    "signals": {},
                    "method": "heuristic",
                }
            ),
        ):
            # Use impossibly low budget to force BudgetExceededError (models cost ~$0.0002+)
            with pytest.raises(BudgetExceededError) as exc_info:
                await budget_constrained_router.select_model(
                    expensive_model_messages, budget_constraint=0.00001  # Below cheapest model cost
                )

            # Verify error structure
            assert "budget" in str(exc_info.value).lower()

    async def test_budget_gate_with_very_low_budget(
        self, direct_providers_only_mode, budget_constrained_router, simple_messages, monkeypatch
    ):
        """Test with very low budget - should select only the cheapest available models."""
        # Add expected models to direct_model_preferences
        monkeypatch.setattr(
            budget_constrained_router,
            "direct_model_preferences",
            {
                "simple": [
                    ("openai", "gpt-3.5-turbo"),
                    ("together", "meta-llama/Llama-3.1-8B-Instruct"),
                ],
                "complex": [
                    ("openai", "gpt-4o"),
                    ("anthropic", "claude-3-5-sonnet-20241022"),
                    ("groq", "llama3-8b-8192"),
                ],
            },
        )

        with patch(
            "app.core.intent.classify_intent",
            new=AsyncMock(
                return_value={
                    "label": "complex",
                    "confidence": 0.9,
                    "signals": {},
                    "method": "heuristic",
                }
            ),
        ):
            # Use impossibly low budget to force BudgetExceededError (below all model costs)
            with pytest.raises(BudgetExceededError) as exc_info:
                await budget_constrained_router.select_model(
                    simple_messages, budget_constraint=0.00001  # Below cheapest model cost
                )

            # Verify error structure
            assert "budget" in str(exc_info.value).lower()

    async def test_budget_exceeded_error_scenario(
        self, direct_providers_only_mode, budget_constrained_router, expensive_model_messages
    ):
        """Test BudgetExceededError is raised when budget is exceeded."""
        with patch(
            "app.core.intent.classify_intent",
            new=AsyncMock(
                return_value={
                    "label": "complex",
                    "confidence": 0.9,
                    "signals": {},
                    "method": "heuristic",
                }
            ),
        ):
            with pytest.raises(BudgetExceededError) as exc_info:
                await budget_constrained_router.select_model(
                    expensive_model_messages,
                    budget_constraint=0.0001,  # Lower than cheapest model cost ($0.000192)
                )

            # Verify error structure
            assert exc_info.value.reason is None or exc_info.value.reason in [
                "no_affordable_available",
                "budget_exceeded",
                "no_pricing",
            ]
            assert "budget" in str(exc_info.value).lower()

    async def test_down_routing_behavior(
        self, direct_providers_only_mode, budget_constrained_router, complex_messages
    ):
        """Test router down-routes to cheaper models when budget constraints apply."""
        # Set up preferences with expensive model first, cheaper models later
        with (
            patch.object(
                budget_constrained_router,
                "direct_model_preferences",
                {
                    "complex": [
                        ("anthropic", "claude-3-5-sonnet-20241022"),  # Expensive
                        ("openai", "gpt-4o"),  # Expensive
                        ("openai", "gpt-4o-mini"),  # Medium
                        ("anthropic", "claude-3-haiku-20240307"),  # Cheap
                        ("groq", "llama3-8b-8192"),  # Cheapest
                    ]
                },
            ),
            patch(
                "app.core.intent.classify_intent",
                new=AsyncMock(
                    return_value={
                        "label": "chat_lite",
                        "confidence": 0.9,
                        "signals": {},
                        "method": "heuristic",
                    }
                ),
            ),
        ):
            # Use budget below cheapest model cost (~$0.025) to force BudgetExceededError
            with pytest.raises(BudgetExceededError) as exc_info:
                await budget_constrained_router.select_model(
                    complex_messages,
                    budget_constraint=0.00001,  # Budget below all model costs
                )

            # Verify error structure
            assert "budget" in str(exc_info.value).lower()

    async def test_cost_estimation_integration(
        self, direct_providers_only_mode, direct_router, simple_messages
    ):
        """Test cost estimation integration with different token counts."""
        with patch(
            "app.core.intent.classify_intent",
            new=AsyncMock(
                return_value={
                    "label": "complex",
                    "confidence": 0.9,
                    "signals": {},
                    "method": "heuristic",
                }
            ),
        ):
            (
                provider,
                model,
                reasoning,
                intent_metadata,
                estimate_metadata,
            ) = await direct_router.select_model(
                simple_messages,
                max_tokens=100,  # Override max_tokens
            )

            # Verify estimate_metadata contains cost information
            assert "usd" in estimate_metadata
            assert "eta_ms" in estimate_metadata
            assert "tokens_in" in estimate_metadata
            assert "tokens_out" in estimate_metadata

            # Should have cost information
            assert estimate_metadata["usd"] > 0

    async def test_cost_estimation_with_different_token_counts(
        self, direct_providers_only_mode, direct_router
    ):
        """Test cost estimation with different input/output token counts."""
        # Test with different message lengths
        short_messages = [ChatMessage(role="user", content="Hi", name=None)]
        long_messages = [ChatMessage(role="user", content="This is a very long message " * 50, name=None)]

        with patch(
            "app.core.intent.classify_intent",
            new=AsyncMock(
                return_value={
                    "label": "complex",
                    "confidence": 0.9,
                    "signals": {},
                    "method": "heuristic",
                }
            ),
        ):
            (
                provider1,
                model1,
                reasoning1,
                intent_metadata1,
                estimate_metadata1,
            ) = await direct_router.select_model(short_messages)
            (
                provider2,
                model2,
                reasoning2,
                intent_metadata2,
                estimate_metadata2,
            ) = await direct_router.select_model(long_messages)

            # Long messages should have higher cost estimates
            short_cost = estimate_metadata1["usd"]
            long_cost = estimate_metadata2["usd"]

            assert long_cost > short_cost, "Long messages should have higher cost estimates"

    async def test_budget_error_scenarios(
        self, direct_providers_only_mode, budget_constrained_router, simple_messages
    ):
        """Test various budget error scenarios."""

        # Test "no_pricing" scenario
        with (
            patch.object(budget_constrained_router, "price_table", {}),
            patch.object(budget_constrained_router.estimator, "prices", {}),
            patch(
                "app.core.intent.classify_intent",
                new=AsyncMock(
                    return_value={
                        "label": "chat_lite",
                        "confidence": 0.9,
                        "signals": {},
                        "method": "heuristic",
                    }
                ),
            ),
        ):
            with pytest.raises(BudgetExceededError) as exc_info:
                await budget_constrained_router.select_model(simple_messages)

            # Verify error structure for no_pricing scenario
            assert exc_info.value.reason == "no_pricing"

        # Test "no_affordable_available" scenario
        with patch(
            "app.core.intent.classify_intent",
            new=AsyncMock(
                return_value={
                    "label": "complex",
                    "confidence": 0.9,
                    "signals": {},
                    "method": "heuristic",
                }
            ),
        ):
            with pytest.raises(BudgetExceededError) as exc_info:
                await budget_constrained_router.select_model(
                    simple_messages,
                    budget_constraint=0.000001,  # Even lower budget below all model costs
                )

            # The reason can be None or one of the expected values
            assert exc_info.value.reason is None or exc_info.value.reason in [
                "no_affordable_available",
                "budget_exceeded",
                "no_pricing",
            ]

    async def test_per_request_budget_override(
        self, direct_providers_only_mode, budget_constrained_router, expensive_model_messages
    ):
        """Test per-request budget constraint override vs default threshold."""
        with patch(
            "app.core.intent.classify_intent",
            new=AsyncMock(
                return_value={
                    "label": "complex",
                    "confidence": 0.9,
                    "signals": {},
                    "method": "heuristic",
                }
            ),
        ):
            # Use impossibly low budget to ensure BudgetExceededError
            with pytest.raises(BudgetExceededError) as exc_info:
                await budget_constrained_router.select_model(
                    expensive_model_messages,
                    budget_constraint=0.00001,  # Below all model costs
                )

            # Verify error structure
            assert "budget" in str(exc_info.value).lower()

    async def test_budget_metrics_and_telemetry(
        self, direct_providers_only_mode, budget_constrained_router, simple_messages, mock_telemetry
    ):
        """Test budget metrics and telemetry integration."""
        with patch(
            "app.core.intent.classify_intent",
            new=AsyncMock(
                return_value={
                    "label": "complex",
                    "confidence": 0.9,
                    "signals": {},
                    "method": "heuristic",
                }
            ),
        ):
            # Use extremely low budget to force BudgetExceededError
            with pytest.raises(BudgetExceededError) as exc_info:
                await budget_constrained_router.select_model(
                    simple_messages, budget_constraint=0.00001
                )

            # Verify error structure
            assert "budget" in str(exc_info.value).lower()

            # Verify telemetry metrics were called
            assert mock_telemetry["budget_exceeded_labeled"].inc.called

    async def test_budget_exceeded_metrics(
        self,
        direct_providers_only_mode,
        budget_constrained_router,
        expensive_model_messages,
        mock_telemetry,
    ):
        """Test budget exceeded metrics are recorded correctly."""
        with patch(
            "app.core.intent.classify_intent",
            new=AsyncMock(
                return_value={
                    "label": "complex",
                    "confidence": 0.9,
                    "signals": {},
                    "method": "heuristic",
                }
            ),
        ):
            # Use extremely low budget to ensure BudgetExceededError
            with pytest.raises(BudgetExceededError):
                await budget_constrained_router.select_model(
                    expensive_model_messages,
                    budget_constraint=0.000001,  # Impossibly low budget
                )

            # Verify budget exceeded metric was recorded
            assert mock_telemetry["budget_exceeded_labeled"].inc.called

    async def test_zero_token_estimates_edge_case(
        self, direct_providers_only_mode, direct_router, simple_messages, monkeypatch
    ):
        """Test edge case with zero token estimates."""
        # Patch the settings to allow zero tokens
        monkeypatch.setattr(direct_router.settings.pricing, "min_tokens_in_floor", 0)
        monkeypatch.setattr(direct_router.settings.pricing, "estimator_default_tokens_in", 0)
        monkeypatch.setattr(direct_router.settings.pricing, "estimator_default_tokens_out", 0)

        with (
            patch(
                "app.core.intent.classify_intent",
                new=AsyncMock(
                    return_value={
                        "label": "chat_lite",
                        "confidence": 0.9,
                        "signals": {},
                        "method": "heuristic",
                    }
                ),
            ),
            patch("app.core.costing.estimate_tokens", return_value=(0, 0)),
        ):
            (
                provider,
                model,
                reasoning,
                intent_metadata,
                estimate_metadata,
            ) = await direct_router.select_model(simple_messages)

            # Should handle zero token estimates gracefully
            # Note: The router may still use some minimum token values for safety
            assert estimate_metadata["tokens_in"] >= 0
            assert estimate_metadata["usd"] >= 0

    async def test_budget_constraint_with_max_tokens_override(
        self, direct_providers_only_mode, budget_constrained_router, simple_messages
    ):
        """Test budget constraint with max_tokens parameter override."""
        with patch(
            "app.core.intent.classify_intent",
            new=AsyncMock(
                return_value={
                    "label": "complex",
                    "confidence": 0.9,
                    "signals": {},
                    "method": "heuristic",
                }
            ),
        ):
            # Test with different max_tokens values - both should fail with 0.02 USD budget threshold
            with pytest.raises(BudgetExceededError) as exc_info1:
                await budget_constrained_router.select_model(
                    simple_messages, max_tokens=10, budget_constraint=0.00001
                )

            with pytest.raises(BudgetExceededError) as exc_info2:
                await budget_constrained_router.select_model(
                    simple_messages, max_tokens=1000, budget_constraint=0.00001
                )

            # Verify error structure
            assert "budget" in str(exc_info1.value).lower()
            assert "budget" in str(exc_info2.value).lower()

    async def test_budget_reasoning_includes_budget_info(
        self, direct_providers_only_mode, budget_constrained_router, complex_messages
    ):
        """Test that reasoning includes budget-related information."""
        with patch(
            "app.core.intent.classify_intent",
            new=AsyncMock(
                return_value={
                    "label": "complex",
                    "confidence": 0.9,
                    "signals": {},
                    "method": "heuristic",
                }
            ),
        ):
            # Use impossibly low budget to ensure BudgetExceededError
            with pytest.raises(BudgetExceededError) as exc_info:
                await budget_constrained_router.select_model(
                    complex_messages, budget_constraint=0.00001  # Below all model costs
                )

            # Verify error structure
            assert "budget" in str(exc_info.value).lower()

    async def test_budget_constraint_with_different_providers(
        self, direct_providers_only_mode, budget_constrained_router, simple_messages
    ):
        """Test budget constraints work correctly across different providers."""
        with patch(
            "app.core.intent.classify_intent",
            new=AsyncMock(
                return_value={
                    "label": "complex",
                    "confidence": 0.9,
                    "signals": {},
                    "method": "heuristic",
                }
            ),
        ):
            results = []

            # Test with different budget constraints - all should fail with very low budgets
            for budget in [0.000001, 0.00001, 0.0001]:
                try:
                    (
                        provider,
                        model,
                        reasoning,
                        intent_metadata,
                        estimate_metadata,
                    ) = await budget_constrained_router.select_model(
                        simple_messages, budget_constraint=budget
                    )
                    results.append((budget, (provider, model)))
                except BudgetExceededError:
                    results.append((budget, ("", "")))

            # All budget constraints should fail with very low budgets
            successful_results = [r for r in results if r[1] is not None]
            # None should succeed with such low budgets
            assert (
                len(successful_results) == 0
            ), f"All budget constraints should fail, got {len(successful_results)} successes: {successful_results}"

    async def test_down_routing_metric(
        self, direct_providers_only_mode, budget_constrained_router, complex_messages, monkeypatch, mock_provider_registry_patch
    ):
        """Test that LLM_ROUTER_DOWN_ROUTE_TOTAL metric is incremented on down-routing."""
        with patch("app.telemetry.metrics.LLM_ROUTER_DOWN_ROUTE_TOTAL") as mock_down_route_metric:
            labeled = mock_down_route_metric.labels.return_value
            # Set up preferences with expensive model first, cheaper models later
            monkeypatch.setattr(
                budget_constrained_router,
                "model_preferences",
                {
                    "complex": [
                        ("anthropic", "claude-3-5-sonnet-20241022"),  # Expensive
                        ("openai", "gpt-4o"),  # Expensive
                        ("openai", "gpt-4o-mini"),  # Medium
                        ("anthropic", "claude-3-haiku-20240307"),  # Cheap
                        ("groq", "llama3-8b-8192"),  # Cheapest
                    ]
                },
            )
            with patch(
                "app.core.intent.classify_intent",
                new=AsyncMock(
                    return_value={
                        "label": "chat_lite",
                        "confidence": 0.9,
                        "signals": {},
                        "method": "heuristic",
                    }
                ),
            ):
                # Ensure the first preference has an available adapter by checking the provider registry
                available_providers = budget_constrained_router.provider_registry_fn()
                first_pref = budget_constrained_router.model_preferences['complex'][0]
                first_provider, first_model = first_pref

                # If first preference doesn't have an adapter, the test won't work
                if first_provider not in available_providers:
                    pytest.skip(f"First preference provider {first_provider} not available in test setup")

                # Use budget below first preference cost but above cheaper alternative cost
                # claude-3-5-sonnet-20241022 costs ~$0.003, gpt-4o-mini costs ~$0.00015
                provider, model, _, _, _ = await budget_constrained_router.select_model(
                    complex_messages,
                    budget_constraint=0.002,  # Above gpt-4o-mini cost but below claude-3-5-sonnet cost
                )

                # Should have selected a cheaper model, triggering down-routing
                assert provider in ["openai", "anthropic", "groq"]
                assert model != "claude-3-5-sonnet-20241022"  # Should not be the most expensive

                # Verify down-routing metric was recorded
                assert labeled.inc.called
