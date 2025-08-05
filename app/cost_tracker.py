# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Cost calculation and tracking utilities with enhanced features.

This module provides both basic and advanced cost tracking capabilities including:
- Basic token counting and cost calculation
- Real-time budget monitoring with Redis (enhanced mode)
- Multi-level budget alerts (daily/weekly/monthly/yearly)
- Cascade routing cost optimization tracking
- Comprehensive analytics and reporting
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import Any

import tiktoken

from .config import settings
from .models import ChatMessage

# Enhanced imports (optional)
try:
    import redis
    import structlog

    ENHANCED_FEATURES_AVAILABLE = True
    logger = structlog.get_logger(__name__)
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    logger = None
    redis = None


class MockRedisClient:
    """Mock Redis client for testing when Redis is not available."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def incrbyfloat(self, key: str, amount: float) -> float:
        current = float(self._data.get(key, 0.0))
        self._data[key] = current + amount
        return float(self._data[key])

    def expire(self, key: str, seconds: int) -> None:
        pass  # Mock implementation

    def setex(self, key: str, seconds: int, value: str) -> None:
        self._data[key] = value

    def get(self, key: str) -> Any:
        return self._data.get(key)


class BudgetPeriod(Enum):
    """Budget period types for enhanced cost tracking."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


@dataclass
class BudgetAlert:
    """Budget alert configuration."""

    threshold_percent: float
    message: str
    severity: str  # "info", "warning", "critical"


class CostTracker:
    """Handles cost calculation and token counting for different providers."""

    def __init__(self, enhanced_mode: bool = False):
        self.enhanced_mode = enhanced_mode
        self.pricing = settings.get_provider_pricing()
        # Initialize tokenizers for different providers
        self._tokenizers: dict[str, Any] = {}

        # Enhanced features (only initialized if enhanced_mode is True)
        self.redis_client: Any = None
        self.db_path: str | None = None

        if enhanced_mode and ENHANCED_FEATURES_AVAILABLE:
            self._initialize_enhanced_features()

    def _initialize_enhanced_features(self) -> None:
        """Initialize enhanced cost tracking features."""
        # This will be implemented when we add the AdvancedCostTracker functionality
        pass

    def get_tokenizer(self, provider: str, model: str) -> None:
        """Get or create tokenizer for a specific model."""
        key = f"{provider}:{model}"
        if key not in self._tokenizers:
            try:
                if provider == "openai":
                    if "gpt-4" in model:
                        self._tokenizers[key] = tiktoken.encoding_for_model("gpt-4")
                    else:
                        self._tokenizers[key] = tiktoken.encoding_for_model("gpt-3.5-turbo")
                elif provider == "anthropic":
                    # Anthropic uses a similar tokenizer to GPT-3.5
                    self._tokenizers[key] = tiktoken.encoding_for_model("gpt-3.5-turbo")
                elif provider == "mistral":
                    # Mistral uses a similar tokenizer to GPT-3.5
                    self._tokenizers[key] = tiktoken.encoding_for_model("gpt-3.5-turbo")
                else:
                    # Default fallback
                    self._tokenizers[key] = tiktoken.encoding_for_model("gpt-3.5-turbo")
            except Exception:
                # Fallback to cl100k_base encoding
                self._tokenizers[key] = tiktoken.get_encoding("cl100k_base")

        return self._tokenizers[key]

    def count_tokens(self, messages: list[ChatMessage], provider: str, model: str) -> int:
        """Count tokens in a list of messages."""
        tokenizer = self.get_tokenizer(provider, model)

        total_tokens = 0
        for message in messages:
            # Add tokens for role and content
            total_tokens += len(tokenizer.encode(message.role))
            total_tokens += len(tokenizer.encode(message.content))
            # Add overhead tokens (varies by provider, using OpenAI's format)
            total_tokens += 4  # Overhead per message

        # Add overhead for the conversation
        total_tokens += 2

        return total_tokens

    def estimate_output_tokens(self, max_tokens: int | None = None) -> int:
        """Estimate output tokens based on max_tokens parameter."""
        if max_tokens:
            return min(max_tokens, 1000)  # Cap at reasonable default
        return settings.max_tokens_default // 2  # Estimate half of max as typical output

    def calculate_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request."""
        if provider not in self.pricing:
            return 0.0

        if model not in self.pricing[provider]:
            return 0.0

        model_pricing = self.pricing[provider][model]

        # Calculate cost per million tokens
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

        return input_cost + output_cost

    def estimate_request_cost(
        self,
        messages: list[ChatMessage],
        provider: str,
        model: str,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Estimate the cost of a request before making it."""
        input_tokens = self.count_tokens(messages, provider, model)
        estimated_output_tokens = self.estimate_output_tokens(max_tokens)
        estimated_cost = self.calculate_cost(provider, model, input_tokens, estimated_output_tokens)

        return {
            "input_tokens": input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_total_tokens": input_tokens + estimated_output_tokens,
            "estimated_cost": estimated_cost,
            "provider": provider,
            "model": model,
        }

    def get_cheapest_model_for_task(self, task_type: str = "general") -> dict[str, str]:
        """Get the cheapest model for a given task type."""
        # Define model preferences by task type
        task_models = {
            "simple": [
                ("mistral", "mistral-small-latest"),
                ("anthropic", "claude-3-haiku-20240307"),
                ("openai", "gpt-3.5-turbo"),
            ],
            "code": [
                ("openai", "gpt-4o"),
                ("anthropic", "claude-3-sonnet-20240229"),
                ("openai", "gpt-3.5-turbo"),
            ],
            "complex": [
                ("openai", "gpt-4o"),
                ("anthropic", "claude-3-sonnet-20240229"),
                ("openai", "gpt-3.5-turbo"),
            ],
            "general": [
                ("openai", "gpt-3.5-turbo"),
                ("anthropic", "claude-3-haiku-20240307"),
                ("mistral", "mistral-small-latest"),
            ],
        }

        models = task_models.get(task_type, task_models["general"])

        # Return the first available model (they're ordered by preference)
        for provider, model in models:
            if provider in self.pricing and model in self.pricing[provider]:
                return {"provider": provider, "model": model}

        # Fallback
        return {"provider": "openai", "model": "gpt-3.5-turbo"}

    def compare_model_costs(
        self,
        messages: list[ChatMessage],
        models: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """Compare costs across multiple models for the same request."""
        comparisons = []

        for model_info in models:
            provider = model_info["provider"]
            model = model_info["model"]

            estimate = self.estimate_request_cost(messages, provider, model, max_tokens)
            estimate.update(model_info)
            comparisons.append(estimate)

        # Sort by estimated cost
        return sorted(comparisons, key=lambda x: x["estimated_cost"])

    def get_model_info(self, provider: str, model: str) -> dict[str, Any]:
        """Get detailed information about a model including pricing."""
        if provider not in self.pricing or model not in self.pricing[provider]:
            return {}

        pricing = self.pricing[provider][model]
        return {
            "provider": provider,
            "model": model,
            "input_price_per_million": pricing["input"],
            "output_price_per_million": pricing["output"],
            "total_price_per_million": pricing["input"] + pricing["output"],
        }


class AdvancedCostTracker(CostTracker):
    """
    Enhanced cost tracker with budget management and cascade-aware tracking.

    Extends the basic CostTracker with advanced features including:
    - Real-time budget monitoring with Redis
    - Multi-level budget alerts
    - Cascade routing cost optimization tracking
    - Comprehensive analytics and reporting
    """

    def __init__(self, db_path: str = "cost_tracker.db", redis_url: str = "redis://localhost:6379/0"):
        super().__init__(enhanced_mode=True)

        # Set database path for enhanced features
        self.db_path = db_path

        # Store redis URL for enhanced features
        self.redis_url = redis_url

        # Configure Redis client for enhanced features
        if ENHANCED_FEATURES_AVAILABLE and redis:
            try:
                redis_client = redis.from_url(redis_url, decode_responses=True)
                # Test connection
                redis_client.ping()
                self.redis_client = redis_client
                if logger:
                    logger.info("redis_connected", url=redis_url)
            except Exception as e:
                if logger:
                    logger.warning("redis_connection_failed", error=str(e), fallback="mock")
                self.redis_client = MockRedisClient()
            else:
                self.redis_client = MockRedisClient()

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database for cost tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables for enhanced tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cost_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT NOT NULL,
                session_id TEXT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                cost REAL NOT NULL,
                routing_strategy TEXT,
                cascade_type TEXT,
                cascade_steps INTEGER,
                quality_score REAL,
                success BOOLEAN DEFAULT TRUE,
                error_message TEXT,
                metadata TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_budgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                budget_type TEXT NOT NULL,
                budget_limit REAL NOT NULL,
                provider TEXT,
                model TEXT,
                alert_thresholds TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, budget_type, provider, model)
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cost_requests_user_timestamp
            ON cost_requests(user_id, timestamp)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cost_requests_provider_model
            ON cost_requests(provider, model)
        """
        )

        conn.commit()
        conn.close()

        if logger:
            logger.info("database_initialized", db_path=self.db_path)

    async def log_request_with_cascade(
        self,
        user_id: str,
        session_id: str,
        cascade_metadata: dict[str, Any],
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        """Log a cascade routing request with detailed metadata."""
        if not ENHANCED_FEATURES_AVAILABLE:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Extract cascade information
            cascade_type = cascade_metadata.get("type", "unknown")
            cascade_steps = len(cascade_metadata.get("steps", []))
            total_cost = sum(step.get("cost", 0) for step in cascade_metadata.get("steps", []))
            quality_score = cascade_metadata.get("final_quality_score")

            # Get the final successful provider/model
            final_step = None
            for step in reversed(cascade_metadata.get("steps", [])):
                if step.get("success", False):
                    final_step = step
                    break

            if final_step:
                provider = final_step.get("provider", "unknown")
                model = final_step.get("model", "unknown")
                prompt_tokens = final_step.get("prompt_tokens", 0)
                completion_tokens = final_step.get("completion_tokens", 0)
            else:
                provider = "failed"
                model = "failed"
                prompt_tokens = 0
                completion_tokens = 0

            cursor.execute(
                """
                INSERT INTO cost_requests (
                    user_id, session_id, provider, model, prompt_tokens, completion_tokens,
                    total_tokens, cost, routing_strategy, cascade_type, cascade_steps,
                    quality_score, success, error_message, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user_id,
                    session_id,
                    provider,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    prompt_tokens + completion_tokens,
                    total_cost,
                    "cascade",
                    cascade_type,
                    cascade_steps,
                    quality_score,
                    success,
                    error_message,
                    json.dumps(cascade_metadata),
                ),
            )

            conn.commit()

            # Update real-time usage in Redis
            await self._update_usage_cache(user_id, total_cost, provider, model)

            # Check budget alerts
            await self._check_budget_alerts(user_id, total_cost, provider, model)

            if logger:
                logger.info(
                    "cascade_request_logged",
                    user_id=user_id,
                    cascade_type=cascade_type,
                    total_cost=total_cost,
                    cascade_steps=cascade_steps,
                    success=success,
                )

        finally:
            conn.close()

    async def log_simple_request(
        self,
        user_id: str,
        session_id: str,
        provider: str,
        model: str,
        cost: float,
        success: bool,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error_message: str | None = None,
    ) -> None:
        """Log a simple (non-cascade) request."""
        if not ENHANCED_FEATURES_AVAILABLE:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO cost_requests (
                    user_id, session_id, provider, model, prompt_tokens, completion_tokens,
                    total_tokens, cost, routing_strategy, success, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user_id,
                    session_id,
                    provider,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    prompt_tokens + completion_tokens,
                    cost,
                    "single",
                    success,
                    error_message,
                ),
            )

            conn.commit()

            # Update real-time usage in Redis
            await self._update_usage_cache(user_id, cost, provider, model)

            # Check budget alerts
            await self._check_budget_alerts(user_id, cost, provider, model)

        finally:
            conn.close()

    async def _update_usage_cache(self, user_id: str, cost: float, provider: str, model: str) -> None:
        """Update real-time usage cache in Redis."""
        if not self.redis_client:
            return

        try:
            today = date.today()

            # Update daily usage
            daily_key = f"usage:daily:{user_id}:{today}"
            self.redis_client.incrbyfloat(daily_key, cost)
            self.redis_client.expire(daily_key, 86400 * 2)  # 2 days TTL

            # Update monthly usage
            monthly_key = f"usage:monthly:{user_id}:{today.year}-{today.month:02d}"
            self.redis_client.incrbyfloat(monthly_key, cost)
            self.redis_client.expire(monthly_key, 86400 * 35)  # 35 days TTL

            # Update provider-specific usage
            provider_key = f"usage:provider:{user_id}:{provider}:{today}"
            self.redis_client.incrbyfloat(provider_key, cost)
            self.redis_client.expire(provider_key, 86400 * 7)  # 7 days TTL

        except Exception as e:
            if logger:
                logger.warning("usage_cache_update_failed", error=str(e))

    async def _check_budget_alerts(self, user_id: str, cost: float, provider: str, model: str) -> None:
        """Check if budget alerts should be triggered."""
        # Implementation would check against user budgets and send alerts
        # This is a simplified version
        pass

    async def set_budget(
        self,
        user_id: str,
        budget_type: str,
        budget_limit: float,
        provider: str | None = None,
        model: str | None = None,
        alert_thresholds: list[float] | None = None,
    ) -> None:
        """Set budget for a user."""
        if not ENHANCED_FEATURES_AVAILABLE:
            return

        if alert_thresholds is None:
            alert_thresholds = [50.0, 80.0, 95.0]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO user_budgets (
                    user_id, budget_type, budget_limit, provider, model, alert_thresholds, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (
                    user_id,
                    budget_type,
                    budget_limit,
                    provider,
                    model,
                    json.dumps(alert_thresholds),
                ),
            )

            conn.commit()

            if logger:
                logger.info(
                    "budget_set",
                    user_id=user_id,
                    budget_type=budget_type,
                    budget_limit=budget_limit,
                    provider=provider,
                    model=model,
                )

        finally:
            conn.close()

    async def get_budget_status(self, user_id: str, budget_type: str | None = None) -> list[dict[str, Any]]:
        """Get budget status for a user."""
        if not ENHANCED_FEATURES_AVAILABLE:
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get budgets
            if budget_type:
                cursor.execute(
                    """
                    SELECT budget_type, budget_limit, provider, model, alert_thresholds
                    FROM user_budgets
                    WHERE user_id = ? AND budget_type = ?
                """,
                    (user_id, budget_type),
                )
            else:
                cursor.execute(
                    """
                    SELECT budget_type, budget_limit, provider, model, alert_thresholds
                    FROM user_budgets
                    WHERE user_id = ?
                """,
                    (user_id,),
                )

            budgets = cursor.fetchall()
            budget_statuses = []

            for budget in budgets:
                budget_type_val, budget_limit, provider, model, alert_thresholds_json = budget
                alert_thresholds = json.loads(alert_thresholds_json) if alert_thresholds_json else []

                # Calculate current usage based on budget type
                current_usage = await self._get_current_usage(user_id, budget_type_val, provider, model)

                usage_percentage = (current_usage / budget_limit * 100) if budget_limit > 0 else 0
                remaining_budget = max(0, budget_limit - current_usage)

                # Generate alerts
                alerts = []
                for threshold in alert_thresholds:
                    if usage_percentage >= threshold:
                        alert_type = "critical" if threshold >= 90 else "warning"
                        alerts.append(
                            {
                                "type": alert_type,
                                "message": f"Budget usage at {usage_percentage:.1f}% (threshold: {threshold}%)",
                                "threshold": threshold,
                                "current_usage": usage_percentage,
                            }
                        )

                # Calculate period dates
                period_start, period_end = self._get_budget_period_dates(budget_type_val)

                budget_statuses.append(
                    {
                        "budget_type": budget_type_val,
                        "budget_limit": budget_limit,
                        "current_usage": current_usage,
                        "usage_percentage": usage_percentage,
                        "remaining_budget": remaining_budget,
                        "provider": provider,
                        "model": model,
                        "alerts": alerts,
                        "period_start": period_start,
                        "period_end": period_end,
                    }
                )

            return budget_statuses

        finally:
            conn.close()

    async def _get_current_usage(
        self, user_id: str, budget_type: str, provider: str | None, model: str | None
    ) -> float:
        """Get current usage for a budget period."""
        today = date.today()

        if budget_type == "daily":
            start_date = today
            end_date = today
        elif budget_type == "weekly":
            # Get start of week (Monday)
            start_date = today - timedelta(days=today.weekday())
            end_date = start_date + timedelta(days=6)
        elif budget_type == "monthly":
            start_date = today.replace(day=1)
            # Get last day of month
            if today.month == 12:
                end_date = date(today.year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = date(today.year, today.month + 1, 1) - timedelta(days=1)
        elif budget_type == "yearly":
            start_date = today.replace(month=1, day=1)
            end_date = today.replace(month=12, day=31)
        else:
            return 0.0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            query = """
                SELECT COALESCE(SUM(cost), 0)
                FROM cost_requests
                WHERE user_id = ? AND DATE(timestamp) BETWEEN ? AND ?
            """
            params = [user_id, start_date.isoformat(), end_date.isoformat()]

            if provider:
                query += " AND provider = ?"
                params.append(provider)

            if model:
                query += " AND model = ?"
                params.append(model)

            cursor.execute(query, params)
            result = cursor.fetchone()
            return result[0] if result else 0.0

        finally:
            conn.close()

    def _get_budget_period_dates(self, budget_type: str) -> tuple[str, str]:
        """Get start and end dates for a budget period."""
        today = date.today()

        if budget_type == "daily":
            return today.isoformat(), today.isoformat()
        elif budget_type == "weekly":
            start_date = today - timedelta(days=today.weekday())
            end_date = start_date + timedelta(days=6)
            return start_date.isoformat(), end_date.isoformat()
        elif budget_type == "monthly":
            start_date = today.replace(day=1)
            if today.month == 12:
                end_date = date(today.year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = date(today.year, today.month + 1, 1) - timedelta(days=1)
            return start_date.isoformat(), end_date.isoformat()
        elif budget_type == "yearly":
            start_date = today.replace(month=1, day=1)
            end_date = today.replace(month=12, day=31)
            return start_date.isoformat(), end_date.isoformat()
        else:
            return today.isoformat(), today.isoformat()


# Global cost tracker instance (basic mode by default)
cost_tracker = CostTracker()


# Function to create enhanced cost tracker
def create_advanced_cost_tracker(
    db_path: str = "cost_tracker.db", redis_url: str = "redis://localhost:6379/0"
) -> AdvancedCostTracker:
    """Create an advanced cost tracker instance."""
    return AdvancedCostTracker(db_path=db_path, redis_url=redis_url)
