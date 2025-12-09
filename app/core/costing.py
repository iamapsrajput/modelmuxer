# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Cost estimation and budget management for ModelMuxer.

This module provides:
- Price table loading and validation
- Latency priors tracking with ring buffer
- Cost estimation with token heuristics and latency priors
"""

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.models import ChatMessage
from app.settings import Settings

# Re-export for tests that patch load_price_table at routing modules
__all__ = ["load_price_table", "LatencyPriors", "Estimator", "Price", "Settings"]

logger = logging.getLogger(__name__)


class Price(BaseModel):
    """Price model for validation of price table entries."""

    input_per_1k_usd: float = Field(..., ge=0, description="Input price per 1k tokens in USD")
    output_per_1k_usd: float = Field(..., ge=0, description="Output price per 1k tokens in USD")


def load_price_table(price_table_path: str) -> Dict[str, Price]:
    """
    Load and validate price table from JSON file.

    Args:
        price_table_path: Path to the price table JSON file

    Returns:
        Dictionary mapping model keys to Price objects

    Note:
        Returns empty dict on any error, with error logged
    """
    try:
        path = Path(price_table_path)
        if not path.exists():
            logger.warning("Price table file not found: %s", price_table_path)
            return {}

        with open(path) as f:
            data = json.load(f)

        # Filter out metadata keys starting with underscore
        price_data = {k: v for k, v in data.items() if not k.startswith("_")}

        # Validate each entry
        validated_prices = {}
        for model_key, price_dict in price_data.items():
            try:
                validated_prices[model_key] = Price(**price_dict)
            except Exception as e:
                logger.warning("Invalid price entry for %s: %s", model_key, e)
                continue

        logger.info(
            "Loaded %d valid price entries from %s", len(validated_prices), price_table_path
        )
        return validated_prices

    except Exception as e:
        logger.error("Failed to load price table from %s: %s", price_table_path, e)
        return {}


class LatencyPriors:
    """
    In-memory ring buffer for tracking recent latency measurements per model.

    Provides p95 and p99 percentile estimates for ETA calculation.

    **Important:** This implementation is in-memory only and resets on application restart.
    All latency measurements are lost when the service restarts. For production deployments
    requiring persistent latency tracking, consider implementing a Redis-backed version
    that can persist measurements across restarts.

    **Future Enhancement:** This class is designed to be easily replaceable with a Redis-backed
    implementation that would persist latency measurements in Redis with the same interface.
    """

    def __init__(self, window_seconds: int = 1800):
        """
        Initialize latency priors with configurable window.

        Args:
            window_seconds: Time window for measurements in seconds (default: 1800s = 30min)
        """
        self.window_seconds = window_seconds
        self.measurements: Dict[str, deque[tuple[float, int]]] = {}

    def update(self, model_key: str, ms: int) -> None:
        """
        Record a new latency measurement for a model.

        Args:
            model_key: Model identifier (e.g., "openai:gpt-4o")
            ms: Latency in milliseconds
        """
        if model_key not in self.measurements:
            self.measurements[model_key] = deque()

        # Add timestamp and measurement
        import time

        timestamp = time.time()
        self.measurements[model_key].append((timestamp, ms))

        # Remove old measurements outside window
        cutoff = timestamp - self.window_seconds
        while self.measurements[model_key] and self.measurements[model_key][0][0] < cutoff:
            self.measurements[model_key].popleft()

    def get(self, model_key: str) -> Dict[str, int]:
        """
        Get latency percentiles for a model.

        Args:
            model_key: Model identifier

        Returns:
            Dictionary with p95 and p99 latency estimates in milliseconds
        """
        # Prune stale entries if model exists
        if model_key in self.measurements and self.measurements[model_key]:
            import time

            cutoff = time.time() - self.window_seconds
            dq = self.measurements[model_key]
            while dq and dq[0][0] < cutoff:
                dq.popleft()

        if model_key not in self.measurements or not self.measurements[model_key]:
            # Return sensible defaults based on model class
            # High-end models with more sophisticated defaults
            high_end_patterns = [
                "gpt-4",
                "gpt-4o",
                "gpt-4-turbo",  # OpenAI high-end
                "claude-3-opus",
                "claude-3-5-sonnet",
                "claude-4-sonnet",
                "opus",  # Generic opus models
                "mistral-large",
                "mistral-medium",  # Mistral high-end
                "gemini-1.5-pro",  # Google high-end
                "o1",  # OpenAI reasoning model
            ]

            if any(pattern in model_key.lower() for pattern in high_end_patterns):
                return {"p95": 1500, "p99": 3000}
            else:
                return {"p95": 800, "p99": 1500}

        # Extract just the latency values
        latencies = [ms for _, ms in self.measurements[model_key]]
        latencies.sort()

        # Calculate percentiles with bias correction for small samples
        import math

        n = len(latencies)

        # Use ceiling to avoid bias for small sample sizes and clamp to bounds
        p95_idx = min(math.ceil(0.95 * (n - 1)), n - 1)
        p99_idx = min(math.ceil(0.99 * (n - 1)), n - 1)

        p95 = latencies[p95_idx]
        p99 = latencies[p99_idx]

        return {"p95": p95, "p99": p99}


@dataclass(frozen=True)
class Estimate:
    """Cost and latency estimate for a model request."""

    usd: float | None
    eta_ms: int
    model_key: str
    tokens_in: int
    tokens_out: int


class Estimator:
    """
    Cost and latency estimator using price table and latency priors.

    Provides deterministic estimates suitable for offline testing and
    budget enforcement in routing decisions.
    """

    def __init__(self, prices: Dict[str, Price], latency_priors: LatencyPriors, settings: Settings):
        """
        Initialize estimator with price table and latency priors.

        Args:
            prices: Validated price table from load_price_table()
            latency_priors: LatencyPriors instance for ETA estimation
            settings: Application settings for defaults
        """
        self.prices = prices
        self.latency_priors = latency_priors
        self.settings = settings

    def estimate(
        self, model_key: str, tokens_in: int | None = None, tokens_out: int | None = None
    ) -> Estimate:
        """
        Estimate cost and latency for a model request.

        Args:
            model_key: Model identifier (e.g., "openai:gpt-4o")
            tokens_in: Input tokens (uses default if None)
            tokens_out: Output tokens (uses default if None)

        Returns:
            Estimate object with cost, ETA, and token counts

        Note:
            Returns zero cost for unknown models
        """
        # Use defaults if tokens not provided
        if tokens_in is None:
            tokens_in = self.settings.pricing.estimator_default_tokens_in
        if tokens_out is None:
            tokens_out = self.settings.pricing.estimator_default_tokens_out

        # Get price for model
        price = self.prices.get(model_key)

        # Get ETA from latency priors (use p95 for conservative estimate)
        latency_stats = self.latency_priors.get(model_key)
        eta_ms = latency_stats["p95"]

        if price is None:
            # Unknown model - return None cost to indicate no pricing available
            # This will cause router to skip this candidate
            return Estimate(
                usd=None,  # Sentinel value to indicate no pricing
                eta_ms=eta_ms,  # Use latency priors even when pricing is missing
                model_key=model_key,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )

        # Calculate cost: (tokens_in/1000) * input_price + (tokens_out/1000) * output_price
        # Note: prices are per 1k tokens
        input_cost = (tokens_in / 1000) * price.input_per_1k_usd
        output_cost = (tokens_out / 1000) * price.output_per_1k_usd
        total_cost = input_cost + output_cost

        return Estimate(
            usd=total_cost,
            eta_ms=eta_ms,
            model_key=model_key,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


def estimate_tokens(messages: List[ChatMessage], defaults: Settings, floor: int) -> tuple[int, int]:
    """
    Estimate input and output tokens for a list of chat messages.

    This function encapsulates the token estimation heuristic used throughout the application
    to ensure consistency and testability.

    Args:
        messages: List of chat messages to estimate tokens for
        defaults: Settings object containing default token values
        floor: Minimum token floor for input tokens

    Returns:
        Tuple of (input_tokens, output_tokens)

    Note:
        Uses character-based estimation (roughly 4 characters per token) for input tokens.
        Output tokens default to the configured default unless max_tokens is specified.
    """
    # Combine all message content for token estimation
    full_text = " ".join([msg.content for msg in messages if msg.content])

    # Use character-based estimation: roughly 4 characters per token
    tokens_in_estimate = max(len(full_text) // 4, floor)

    # Use the estimate or fall back to default if no content
    tokens_in = (
        tokens_in_estimate
        if full_text.strip()
        else int(defaults.pricing.estimator_default_tokens_in)
    )

    # Output tokens default to configured default
    tokens_out = int(defaults.pricing.estimator_default_tokens_out)

    return tokens_in, tokens_out
