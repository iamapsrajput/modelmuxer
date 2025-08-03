#!/usr/bin/env python3
# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Unit tests for the routing logic.
"""

import unittest

from app.models import ChatMessage
from app.router import HeuristicRouter


class TestHeuristicRouter(unittest.TestCase):
    """Test cases for the HeuristicRouter class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.router = HeuristicRouter()

    def test_code_detection(self) -> None:
        """Test code detection in prompts."""
        # Test with code block
        messages = [
            ChatMessage(
                role="user",
                content="Here's some code:\n```python\ndef hello():\n    print('Hello')\n```",
            )
        ]
        analysis = self.router.analyze_prompt(messages)
        self.assertTrue(analysis["has_code"])
        self.assertEqual(analysis["task_type"], "code")

        # Test with inline code
        messages = [
            ChatMessage(role="user", content="Use the `print()` function in Python", name=None)
        ]
        analysis = self.router.analyze_prompt(messages)
        self.assertTrue(analysis["has_code"])

        # Test without code
        messages = [ChatMessage(role="user", content="What is the weather like today?", name=None)]
        analysis = self.router.analyze_prompt(messages)
        self.assertFalse(analysis["has_code"])

    def test_complexity_detection(self) -> None:
        """Test complexity detection in prompts."""
        # Test complex analysis request
        messages = [
            ChatMessage(
                role="user", content="Analyze the algorithm complexity and explain the trade-offs"
            )
        ]
        analysis = self.router.analyze_prompt(messages)
        self.assertTrue(analysis["has_complexity"])
        self.assertEqual(analysis["task_type"], "complex")

        # Test simple request
        messages = [ChatMessage(role="user", content="What is 2+2?", name=None)]
        analysis = self.router.analyze_prompt(messages)
        self.assertFalse(analysis["has_complexity"])

    def test_simple_query_detection(self) -> None:
        """Test simple query detection."""
        # Test simple question
        messages = [ChatMessage(role="user", content="What is Python?", name=None)]
        analysis = self.router.analyze_prompt(messages)
        self.assertTrue(analysis["is_simple"])
        self.assertEqual(analysis["task_type"], "simple")

        # Test long complex query
        messages = [
            ChatMessage(
                role="user",
                content="Explain in detail the differences between various sorting algorithms, their time complexities, space complexities, and when to use each one in different scenarios.",
            )
        ]
        analysis = self.router.analyze_prompt(messages)
        self.assertFalse(analysis["is_simple"])

    def test_model_selection_code(self) -> None:
        """Test model selection for code-related prompts."""
        messages = [
            ChatMessage(
                role="user", content="Write a Python function to implement binary search", name=None
            )
        ]
        provider, model, reason = self.router.select_model(messages)

        # Should select a high-quality model for code
        self.assertIn(provider, ["openai", "anthropic"])
        self.assertIn("code", reason.lower())

    def test_model_selection_simple(self) -> None:
        """Test model selection for simple prompts."""
        messages = [ChatMessage(role="user", content="What is 5+3?", name=None)]
        provider, model, reason = self.router.select_model(messages)

        # Should select cost-effective model for simple queries
        self.assertIn(provider, ["mistral", "anthropic", "openai"])
        self.assertIn("simple", reason.lower())

    def test_model_selection_complex(self) -> None:
        """Test model selection for complex analysis."""
        messages = [
            ChatMessage(
                role="user",
                content="Analyze the performance characteristics of different database indexing strategies",
            )
        ]
        provider, model, reason = self.router.select_model(messages)

        # Should select high-quality model for complex analysis
        self.assertIn(provider, ["openai", "anthropic"])
        self.assertIn("complex", reason.lower())

    def test_multi_message_analysis(self) -> None:
        """Test analysis with multiple messages."""
        messages = [
            ChatMessage(role="user", content="I need help with programming", name=None),
            ChatMessage(
                role="assistant", content="I'd be happy to help! What programming language?"
            ),
            ChatMessage(
                role="user", content="Python. Can you write a function to reverse a string?"
            ),
        ]

        analysis = self.router.analyze_prompt(messages)
        self.assertEqual(analysis["message_count"], 3)
        self.assertTrue(analysis["has_code"])  # Should detect programming context

    def test_budget_constraint(self) -> None:
        """Test model selection with budget constraints."""
        messages = [ChatMessage(role="user", content="Explain quantum computing", name=None)]

        # Test with very low budget
        provider, model, reason = self.router.select_model(messages, budget_constraint=0.0001)
        self.assertIn(provider, ["mistral"])  # Should select cheapest option

        # Test with normal budget
        provider, model, reason = self.router.select_model(messages, budget_constraint=0.01)
        # Should allow more expensive models
        self.assertIn(provider, ["openai", "anthropic", "mistral"])


class TestCostTracker(unittest.TestCase):
    """Test cases for cost tracking functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from app.cost_tracker import CostTracker

        self.cost_tracker = CostTracker()

    def test_token_counting(self) -> None:
        """Test token counting functionality."""
        messages = [ChatMessage(role="user", content="Hello, how are you?", name=None)]
        tokens = self.cost_tracker.count_tokens(messages, "openai", "gpt-3.5-turbo")
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, 100)  # Should be reasonable for short message

    def test_cost_calculation(self) -> None:
        """Test cost calculation."""
        cost = self.cost_tracker.calculate_cost("openai", "gpt-3.5-turbo", 1000, 500)
        self.assertGreater(cost, 0)
        self.assertLess(cost, 1.0)  # Should be reasonable cost

        # Test that GPT-4o is more expensive than GPT-3.5
        cost_gpt4 = self.cost_tracker.calculate_cost("openai", "gpt-4o", 1000, 500)
        cost_gpt35 = self.cost_tracker.calculate_cost("openai", "gpt-3.5-turbo", 1000, 500)
        self.assertGreater(cost_gpt4, cost_gpt35)

    def test_cost_estimation(self) -> None:
        """Test request cost estimation."""
        messages = [ChatMessage(role="user", content="Write a simple Python function", name=None)]
        estimate = self.cost_tracker.estimate_request_cost(messages, "openai", "gpt-3.5-turbo")

        self.assertIn("input_tokens", estimate)
        self.assertIn("estimated_cost", estimate)
        self.assertGreater(estimate["input_tokens"], 0)
        self.assertGreater(estimate["estimated_cost"], 0)


def run_unit_tests() -> None:
    """Run all unit tests."""
    print("🧪 Running Unit Tests for ModelMuxer Router\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestHeuristicRouter))
    suite.addTests(loader.loadTestsFromTestCase(TestCostTracker))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*50}")
    print("📊 UNIT TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\n❌ ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print("\n🎉 All unit tests passed!")
    else:
        print("\n⚠️  Some unit tests failed.")

    return success


if __name__ == "__main__":
    import sys

    success = run_unit_tests()
    sys.exit(0 if success else 1)
