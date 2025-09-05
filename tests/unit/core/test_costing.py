# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Unit tests for the costing module.

Tests the price table loader, latency priors, and estimator functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.costing import (
    Estimate,
    Estimator,
    LatencyPriors,
    Price,
    estimate_tokens,
    load_price_table,
)
from app.models import ChatMessage


class TestPrice:
    """Test the Price Pydantic model."""

    def test_valid_price(self):
        """Test creating a valid Price object."""
        price = Price(input_per_1k_usd=2.50, output_per_1k_usd=10.00)
        assert price.input_per_1k_usd == 2.50
        assert price.output_per_1k_usd == 10.00

    def test_zero_price_allowed(self):
        """Test that zero prices are allowed."""
        price = Price(input_per_1k_usd=0.0, output_per_1k_usd=0.0)
        assert price.input_per_1k_usd == 0.0
        assert price.output_per_1k_usd == 0.0

    def test_invalid_price_negative(self):
        """Test that negative prices raise validation error."""
        with pytest.raises(ValueError):
            Price(input_per_1k_usd=-1.0, output_per_1k_usd=5.0)


class TestLoadPriceTable:
    """Test the price table loading functionality."""

    def test_load_valid_price_table(self):
        """Test loading a valid price table JSON file."""
        price_data = {
            "openai:gpt-4o": {"input_per_1k_usd": 2.50, "output_per_1k_usd": 10.00},
            "anthropic:claude-3-sonnet": {"input_per_1k_usd": 3.00, "output_per_1k_usd": 15.00},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(price_data, f)
            temp_path = f.name

        try:
            prices = load_price_table(temp_path)
            assert len(prices) == 2
            assert "openai:gpt-4o" in prices
            assert "anthropic:claude-3-sonnet" in prices
            assert prices["openai:gpt-4o"].input_per_1k_usd == 2.50
            assert prices["openai:gpt-4o"].output_per_1k_usd == 10.00
        finally:
            Path(temp_path).unlink()

    def test_load_missing_file(self):
        """Test handling of missing price table file."""
        prices = load_price_table("/nonexistent/path/prices.json")
        assert prices == {}

    def test_load_invalid_json(self):
        """Test handling of invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name

        try:
            prices = load_price_table(temp_path)
            assert prices == {}
        finally:
            Path(temp_path).unlink()

    def test_load_with_metadata(self):
        """Test that metadata keys (starting with _) are filtered out."""
        price_data = {
            "_comment": "This is a comment",
            "_format": "provider:model format",
            "openai:gpt-4o": {"input_per_1k_usd": 2.50, "output_per_1k_usd": 10.00},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(price_data, f)
            temp_path = f.name

        try:
            prices = load_price_table(temp_path)
            assert len(prices) == 1
            assert "openai:gpt-4o" in prices
            assert "_comment" not in prices
            assert "_format" not in prices
        finally:
            Path(temp_path).unlink()

    def test_load_invalid_price_entry(self):
        """Test handling of invalid price entries."""
        price_data = {
            "openai:gpt-4o": {"input_per_1k_usd": 2.50, "output_per_1k_usd": 10.00},
            "invalid:model": {"invalid_field": "invalid_value"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(price_data, f)
            temp_path = f.name

        try:
            prices = load_price_table(temp_path)
            assert len(prices) == 1
            assert "openai:gpt-4o" in prices
            assert "invalid:model" not in prices
        finally:
            Path(temp_path).unlink()


class TestLatencyPriors:
    """Test the LatencyPriors class."""

    def test_update_and_get(self):
        """Test updating and retrieving latency measurements."""
        priors = LatencyPriors(window_seconds=3600)

        # Add some measurements
        priors.update("openai:gpt-4o", 1000)
        priors.update("openai:gpt-4o", 1200)
        priors.update("openai:gpt-4o", 800)
        priors.update("openai:gpt-4o", 1500)
        priors.update("openai:gpt-4o", 1100)

        stats = priors.get("openai:gpt-4o")
        assert "p95" in stats
        assert "p99" in stats
        assert stats["p95"] > 0
        assert stats["p99"] > 0

    def test_empty_buffer_defaults(self):
        """Test that empty buffer returns sensible defaults."""
        priors = LatencyPriors()

        # Test high-end model defaults
        stats = priors.get("openai:gpt-4")
        assert stats["p95"] == 1500
        assert stats["p99"] == 3000

        # Test regular model defaults
        stats = priors.get("openai:gpt-3.5-turbo")
        assert stats["p95"] == 800
        assert stats["p99"] == 1500

    def test_multiple_models(self):
        """Test that multiple models are tracked independently."""
        priors = LatencyPriors()

        priors.update("openai:gpt-4o", 1000)
        priors.update("anthropic:claude-3-sonnet", 800)

        stats1 = priors.get("openai:gpt-4o")
        stats2 = priors.get("anthropic:claude-3-sonnet")

        assert stats1["p95"] == 1000
        assert stats2["p95"] == 800

    def test_single_measurement(self):
        """Test behavior with single measurement."""
        priors = LatencyPriors()
        priors.update("test:model", 500)

        stats = priors.get("test:model")
        assert stats["p95"] == 500
        assert stats["p99"] == 500

    def test_identical_measurements(self):
        """Test behavior with identical measurements."""
        priors = LatencyPriors()

        for _ in range(5):
            priors.update("test:model", 1000)

        stats = priors.get("test:model")
        assert stats["p95"] == 1000
        assert stats["p99"] == 1000

    @patch("time.time")
    def test_windowing_behavior(self, mock_time):
        """Test that old measurements expire outside the window."""
        priors = LatencyPriors(window_seconds=100)

        # Set initial time
        mock_time.return_value = 1000

        # Add measurements
        priors.update("test:model", 500)
        priors.update("test:model", 600)

        # Advance time beyond window
        mock_time.return_value = 1200

        # Add new measurement
        priors.update("test:model", 700)

        # Should only have the new measurement
        stats = priors.get("test:model")
        assert stats["p95"] == 700
        assert stats["p99"] == 700

    def test_small_sample_percentiles(self):
        """Test percentile calculation for small sample sizes (n=1,2,3)."""
        priors = LatencyPriors()

        # Test n=1
        priors.update("test:n1", 100)
        stats = priors.get("test:n1")
        assert stats["p95"] == 100
        assert stats["p99"] == 100

        # Test n=2
        priors.update("test:n2", 100)
        priors.update("test:n2", 200)
        stats = priors.get("test:n2")
        # With n=2, ceil(0.95 * (2-1)) = ceil(0.95) = 1, so p95 = latencies[1] = 200
        assert stats["p95"] == 200
        assert stats["p99"] == 200

        # Test n=3
        priors.update("test:n3", 100)
        priors.update("test:n3", 200)
        priors.update("test:n3", 300)
        stats = priors.get("test:n3")
        # With n=3, ceil(0.95 * (3-1)) = ceil(1.9) = 2, so p95 = latencies[2] = 300
        assert stats["p95"] == 300
        assert stats["p99"] == 300


class TestEstimator:
    """Test the Estimator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.prices = {
            "openai:gpt-4o": Price(input_per_1k_usd=2.50, output_per_1k_usd=10.00),
            "anthropic:claude-3-sonnet": Price(input_per_1k_usd=3.00, output_per_1k_usd=15.00),
            "mistral:mistral-small": Price(input_per_1k_usd=0.14, output_per_1k_usd=0.42),
            "free:model": Price(input_per_1k_usd=0.0, output_per_1k_usd=0.0),
        }
        self.latency_priors = LatencyPriors()
        self.settings = type(
            "Settings",
            (),
            {
                "pricing": type(
                    "PricingSettings",
                    (),
                    {"estimator_default_tokens_in": 400, "estimator_default_tokens_out": 300},
                )()
            },
        )()
        self.estimator = Estimator(self.prices, self.latency_priors, self.settings)

    def test_cost_calculation(self):
        """Test cost calculation math correctness."""
        estimate = self.estimator.estimate("openai:gpt-4o", 1000, 500)

        # Expected: (1000/1000) * 2.50 + (500/1000) * 10.00 = 2.50 + 5.00 = 7.50
        expected_cost = (1000 / 1000) * 2.50 + (500 / 1000) * 10.00
        assert estimate.usd == pytest.approx(expected_cost, rel=1e-6)
        assert estimate.model_key == "openai:gpt-4o"
        assert estimate.tokens_in == 1000
        assert estimate.tokens_out == 500

    def test_cost_monotonicity(self):
        """Test that more tokens = higher cost."""
        estimate1 = self.estimator.estimate("openai:gpt-4o", 100, 50)
        estimate2 = self.estimator.estimate("openai:gpt-4o", 200, 100)

        assert estimate2.usd > estimate1.usd

    def test_none_price_unknown_model(self):
        """Test that unknown models return None cost."""
        estimate = self.estimator.estimate("unknown:model", 1000, 500)

        assert estimate.usd is None
        assert estimate.eta_ms == 800  # Default latency
        assert estimate.model_key == "unknown:model"

    def test_token_estimation_fallbacks(self):
        """Test token estimation fallbacks when tokens not provided."""
        estimate = self.estimator.estimate("openai:gpt-4o")

        assert estimate.tokens_in == 400  # Default from settings
        assert estimate.tokens_out == 300  # Default from settings

    def test_eta_from_latency_priors(self):
        """Test ETA calculation from latency priors."""
        # Add some latency measurements
        self.latency_priors.update("openai:gpt-4o", 1000)
        self.latency_priors.update("openai:gpt-4o", 1200)
        self.latency_priors.update("openai:gpt-4o", 800)

        estimate = self.estimator.estimate("openai:gpt-4o", 1000, 500)

        # Should use p95 from latency priors
        stats = self.latency_priors.get("openai:gpt-4o")
        assert estimate.eta_ms == stats["p95"]

    def test_model_key_format(self):
        """Test model key format handling."""
        estimate = self.estimator.estimate("provider:model", 100, 50)

        assert estimate.model_key == "provider:model"

    def test_integration_with_price_table(self):
        """Test integration with price table and latency priors."""
        # Test with different models
        estimate1 = self.estimator.estimate("openai:gpt-4o", 1000, 500)
        estimate2 = self.estimator.estimate("anthropic:claude-3-sonnet", 1000, 500)
        estimate3 = self.estimator.estimate("mistral:mistral-small", 1000, 500)

        # All should have different costs
        assert estimate1.usd != estimate2.usd
        assert estimate2.usd != estimate3.usd
        assert estimate1.usd != estimate3.usd

        # All should have valid model keys
        assert estimate1.model_key == "openai:gpt-4o"
        assert estimate2.model_key == "anthropic:claude-3-sonnet"
        assert estimate3.model_key == "mistral:mistral-small"


class TestEstimate:
    """Test the Estimate dataclass."""

    def test_estimate_creation(self):
        """Test creating an Estimate object."""
        estimate = Estimate(
            usd=5.25, eta_ms=1200, model_key="openai:gpt-4o", tokens_in=1000, tokens_out=500
        )

        assert estimate.usd == 5.25
        assert estimate.eta_ms == 1200
        assert estimate.model_key == "openai:gpt-4o"
        assert estimate.tokens_in == 1000
        assert estimate.tokens_out == 500

    def test_estimate_immutability(self):
        """Test that Estimate objects are immutable."""
        estimate = Estimate(
            usd=5.25, eta_ms=1200, model_key="openai:gpt-4o", tokens_in=1000, tokens_out=500
        )

        # Should not be able to modify fields
        with pytest.raises(Exception):  # noqa: B017
            estimate.usd = 10.0


class TestZeroPriceModel:
    """Test zero-price model handling."""

    def test_zero_price_model_zero_cost_monotonicity(self):
        """Test that a zero-priced model yields zero cost and doesn't break monotonicity."""
        prices = {
            "free:model": Price(input_per_1k_usd=0.0, output_per_1k_usd=0.0),
            "mistral:mistral-small": Price(input_per_1k_usd=0.14, output_per_1k_usd=0.42),
        }
        latency_priors = LatencyPriors()
        settings = type(
            "Settings",
            (),
            {
                "pricing": type(
                    "PricingSettings",
                    (),
                    {"estimator_default_tokens_in": 400, "estimator_default_tokens_out": 300},
                )()
            },
        )()
        estimator = Estimator(prices, latency_priors, settings)

        # Test zero cost for zero-priced model
        estimate_zero = estimator.estimate("free:model", 1000, 500)
        assert estimate_zero.usd == 0.0
        assert estimate_zero.model_key == "free:model"

        # Test monotonicity still works - more tokens should still result in zero cost
        estimate_more = estimator.estimate("free:model", 2000, 1000)
        assert estimate_more.usd == 0.0

        # Test that zero-priced model cost is always <= paid models
        estimate_paid = estimator.estimate("mistral:mistral-small", 1000, 500)
        assert estimate_zero.usd <= estimate_paid.usd


class TestEstimateTokens:
    """Test the estimate_tokens helper function."""

    def test_basic_token_estimation(self):
        """Test basic token estimation with simple messages."""
        messages = [
            ChatMessage(role="user", content="Hello, how are you?"),
            ChatMessage(role="assistant", content="I'm doing well, thank you!"),
        ]

        settings = type(
            "Settings",
            (),
            {
                "pricing": type(
                    "PricingSettings",
                    (),
                    {"estimator_default_tokens_in": 400, "estimator_default_tokens_out": 300},
                )()
            },
        )()

        tokens_in, tokens_out = estimate_tokens(messages, settings, 50)

        # Should be roughly 4 characters per token
        expected_tokens = len("Hello, how are you? I'm doing well, thank you!") // 4
        assert tokens_in == max(expected_tokens, 50)  # Should respect floor
        assert tokens_out == 300  # Should use default

    def test_empty_messages(self):
        """Test token estimation with empty messages."""
        messages = [
            ChatMessage(role="user", content=""),
            ChatMessage(role="assistant", content=""),
        ]

        settings = type(
            "Settings",
            (),
            {
                "pricing": type(
                    "PricingSettings",
                    (),
                    {"estimator_default_tokens_in": 400, "estimator_default_tokens_out": 300},
                )()
            },
        )()

        tokens_in, tokens_out = estimate_tokens(messages, settings, 50)

        assert tokens_in == 400  # Should use default when no content
        assert tokens_out == 300  # Should use default

    def test_token_floor_respect(self):
        """Test that token estimation respects the minimum floor."""
        messages = [
            ChatMessage(role="user", content="Hi"),  # Very short message
        ]

        settings = type(
            "Settings",
            (),
            {
                "pricing": type(
                    "PricingSettings",
                    (),
                    {"estimator_default_tokens_in": 400, "estimator_default_tokens_out": 300},
                )()
            },
        )()

        tokens_in, tokens_out = estimate_tokens(messages, settings, 100)

        assert tokens_in == 100  # Should use floor when estimate is below it
        assert tokens_out == 300  # Should use default

    def test_long_message_estimation(self):
        """Test token estimation with a long message."""
        long_content = "This is a very long message " * 100  # 2600 characters
        messages = [
            ChatMessage(role="user", content=long_content),
        ]

        settings = type(
            "Settings",
            (),
            {
                "pricing": type(
                    "PricingSettings",
                    (),
                    {"estimator_default_tokens_in": 400, "estimator_default_tokens_out": 300},
                )()
            },
        )()

        tokens_in, tokens_out = estimate_tokens(messages, settings, 50)

        expected_tokens = len(long_content) // 4  # Should be around 650
        assert tokens_in == expected_tokens
        assert tokens_out == 300  # Should use default

    def test_mixed_content_messages(self):
        """Test token estimation with messages containing empty content."""
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="system", content=""),  # Empty content
            ChatMessage(role="assistant", content="Hi there"),
        ]

        settings = type(
            "Settings",
            (),
            {
                "pricing": type(
                    "PricingSettings",
                    (),
                    {"estimator_default_tokens_in": 400, "estimator_default_tokens_out": 300},
                )()
            },
        )()

        tokens_in, tokens_out = estimate_tokens(messages, settings, 50)

        # Should only count non-empty content
        expected_tokens = len("Hello Hi there") // 4
        assert tokens_in == max(expected_tokens, 50)
        assert tokens_out == 300
