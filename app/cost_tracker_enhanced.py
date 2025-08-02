# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Enhanced Cost Tracker with budget management and cascade-aware tracking.

This module provides advanced cost tracking capabilities including:
- Real-time budget monitoring with Redis
- Multi-level budget alerts (daily/weekly/monthly/yearly)
- Cascade routing cost optimization tracking
- Comprehensive analytics and reporting
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

import redis
import structlog

logger = structlog.get_logger(__name__)


class MockRedisClient:
    """Mock Redis client for testing when Redis is not available."""

    def __init__(self):
        self._data = {}

    def incrbyfloat(self, key: str, amount: float):
        current = self._data.get(key, 0.0)
        self._data[key] = current + amount
        return self._data[key]

    def expire(self, key: str, seconds: int):
        pass  # Mock implementation

    def setex(self, key: str, seconds: int, value: str):
        self._data[key] = value

    def get(self, key: str):
        return self._data.get(key)


class BudgetPeriod(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


@dataclass
class Budget:
    user_id: str
    budget_type: BudgetPeriod
    budget_limit: float
    provider: str | None = None
    model: str | None = None
    alert_thresholds: list[float] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = [50.0, 80.0, 95.0]
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class CascadeRequestLog:
    user_id: str
    session_id: str
    timestamp: datetime
    cascade_type: str
    total_cost: float
    cascade_steps: int
    quality_score: float | None
    confidence_score: float | None
    response_time: float
    success: bool
    final_model: str | None = None
    escalation_reasons: list[str] = None
    error_message: str | None = None


class AdvancedCostTracker:
    """Enhanced cost tracker with budget management and cascade analytics."""

    def __init__(
        self, db_path: str = "cost_tracker.db", redis_url: str = "redis://localhost:6379/0"
    ):
        self.db_path = db_path
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
        except Exception as e:
            logger.warning("redis_connection_failed", error=str(e))
            # Create a mock Redis client for testing
            self.redis_client = MockRedisClient()
        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database with enhanced schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enhanced requests table with cascade metadata
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_cost REAL NOT NULL,
                success BOOLEAN NOT NULL,
                routing_strategy TEXT,
                cascade_type TEXT,
                cascade_steps INTEGER,
                quality_score REAL,
                confidence_score REAL,
                response_time REAL,
                escalation_reasons TEXT,
                error_message TEXT
            )
        """
        )

        # Budgets table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS budgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                budget_type TEXT NOT NULL,
                budget_limit REAL NOT NULL,
                provider TEXT,
                model TEXT,
                alert_thresholds TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                UNIQUE(user_id, budget_type, provider, model)
            )
        """
        )

        # Budget alerts table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS budget_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                budget_id INTEGER NOT NULL,
                alert_threshold REAL NOT NULL,
                triggered_at DATETIME NOT NULL,
                current_usage REAL NOT NULL,
                FOREIGN KEY (budget_id) REFERENCES budgets (id)
            )
        """
        )

        # Create indexes for performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_user_timestamp ON requests(user_id, timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_cascade ON requests(cascade_type, timestamp)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_budgets_user ON budgets(user_id)")

        conn.commit()
        conn.close()

        logger.info("enhanced_cost_tracker_initialized", db_path=self.db_path)

    async def set_budget(
        self,
        user_id: str,
        budget_type: BudgetPeriod,
        budget_limit: float,
        provider: str | None = None,
        model: str | None = None,
        alert_thresholds: list[float] = None,
    ):
        """Set or update user budget."""
        if alert_thresholds is None:
            alert_thresholds = [50.0, 80.0, 95.0]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO budgets
                (user_id, budget_type, budget_limit, provider, model, alert_thresholds, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user_id,
                    budget_type.value,
                    budget_limit,
                    provider,
                    model,
                    json.dumps(alert_thresholds),
                    datetime.now(),
                    datetime.now(),
                ),
            )

            conn.commit()

            # Update Redis cache
            try:
                cache_key = (
                    f"budget:{user_id}:{budget_type.value}:{provider or 'all'}:{model or 'all'}"
                )
                budget_data = {
                    "budget_limit": budget_limit,
                    "alert_thresholds": alert_thresholds,
                    "updated_at": datetime.now().isoformat(),
                }
                self.redis_client.setex(cache_key, 3600, json.dumps(budget_data))
            except Exception as e:
                logger.warning("redis_budget_cache_failed", error=str(e))

            logger.info(
                "budget_set",
                user_id=user_id,
                budget_type=budget_type.value,
                budget_limit=budget_limit,
                provider=provider,
                model=model,
            )

        finally:
            conn.close()

    async def log_request_with_cascade(
        self,
        user_id: str,
        session_id: str,
        cascade_metadata: dict[str, Any],
        success: bool,
        error_message: str | None = None,
    ):
        """Log a cascade request with detailed metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Extract cascade metadata
            cascade_type = cascade_metadata.get("cascade_type", "unknown")
            total_cost = cascade_metadata.get("total_cost", 0.0)
            cascade_steps = len(cascade_metadata.get("steps_attempted", []))
            quality_score = cascade_metadata.get("quality_score")
            confidence_score = cascade_metadata.get("confidence_score")
            response_time = cascade_metadata.get("response_time", 0.0)
            escalation_reasons = cascade_metadata.get("escalation_reasons", [])

            # Get the final successful step for provider/model info
            successful_steps = [
                s for s in cascade_metadata.get("steps_attempted", []) if s.get("success")
            ]
            if successful_steps:
                final_step = successful_steps[-1]
                provider = final_step["provider"]
                model = final_step["model"]
                # Estimate tokens (simplified)
                prompt_tokens = 50  # Placeholder
                completion_tokens = 20  # Placeholder
            else:
                provider = "unknown"
                model = "unknown"
                prompt_tokens = 0
                completion_tokens = 0

            cursor.execute(
                """
                INSERT INTO requests
                (user_id, session_id, timestamp, provider, model, prompt_tokens, completion_tokens,
                 total_cost, success, routing_strategy, cascade_type, cascade_steps, quality_score,
                 confidence_score, response_time, escalation_reasons, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user_id,
                    session_id,
                    datetime.now(),
                    provider,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    total_cost,
                    success,
                    "cascade",
                    cascade_type,
                    cascade_steps,
                    quality_score,
                    confidence_score,
                    response_time,
                    json.dumps(escalation_reasons),
                    error_message,
                ),
            )

            conn.commit()

            # Update real-time usage in Redis
            await self._update_usage_cache(user_id, total_cost, provider, model)

            # Check budget alerts
            await self._check_budget_alerts(user_id, total_cost, provider, model)

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
    ):
        """Log a simple (non-cascade) request."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO requests
                (user_id, session_id, timestamp, provider, model, prompt_tokens, completion_tokens,
                 total_cost, success, routing_strategy, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user_id,
                    session_id,
                    datetime.now(),
                    provider,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    cost,
                    success,
                    "single",
                    error_message,
                ),
            )

            conn.commit()

            # Update real-time usage in Redis
            await self._update_usage_cache(user_id, cost, provider, model)

            # Check budget alerts
            await self._check_budget_alerts(user_id, cost, provider, model)

            logger.info(
                "simple_request_logged",
                user_id=user_id,
                provider=provider,
                model=model,
                cost=cost,
                success=success,
            )

        finally:
            conn.close()

    async def _update_usage_cache(self, user_id: str, cost: float, provider: str, model: str):
        """Update real-time usage cache in Redis."""
        try:
            # Update daily usage
            today = datetime.now().date().isoformat()
            daily_key = f"usage:daily:{user_id}:{today}"
            self.redis_client.incrbyfloat(daily_key, cost)
            self.redis_client.expire(daily_key, 86400 * 7)  # Keep for 7 days

            # Update provider-specific usage
            provider_key = f"usage:daily:{user_id}:{today}:{provider}"
            self.redis_client.incrbyfloat(provider_key, cost)
            self.redis_client.expire(provider_key, 86400 * 7)

            # Update model-specific usage
            model_key = f"usage:daily:{user_id}:{today}:{provider}:{model}"
            self.redis_client.incrbyfloat(model_key, cost)
            self.redis_client.expire(model_key, 86400 * 7)

        except Exception as e:
            logger.warning("redis_usage_update_failed", error=str(e))

    async def _check_budget_alerts(self, user_id: str, cost: float, provider: str, model: str):
        """Check if budget alerts should be triggered."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get all budgets for this user
            cursor.execute(
                """
                SELECT id, budget_type, budget_limit, provider, model, alert_thresholds
                FROM budgets WHERE user_id = ?
            """,
                (user_id,),
            )

            budgets = cursor.fetchall()

            for (
                budget_id,
                budget_type,
                budget_limit,
                budget_provider,
                budget_model,
                alert_thresholds_json,
            ) in budgets:
                # Check if this budget applies to the current request
                if budget_provider and budget_provider != provider:
                    continue
                if budget_model and budget_model != model:
                    continue

                # Get current usage for this budget period
                current_usage = await self._get_current_usage(
                    user_id, BudgetPeriod(budget_type), budget_provider, budget_model
                )
                utilization_percent = (current_usage / budget_limit) * 100

                alert_thresholds = json.loads(alert_thresholds_json)

                # Check each alert threshold
                for threshold in alert_thresholds:
                    if utilization_percent >= threshold:
                        # Check if alert already triggered for this threshold today
                        today = datetime.now().date()
                        cursor.execute(
                            """
                            SELECT COUNT(*) FROM budget_alerts
                            WHERE budget_id = ? AND alert_threshold = ? AND DATE(triggered_at) = ?
                        """,
                            (budget_id, threshold, today),
                        )

                        if cursor.fetchone()[0] == 0:
                            # Trigger alert
                            cursor.execute(
                                """
                                INSERT INTO budget_alerts (user_id, budget_id, alert_threshold, triggered_at, current_usage)
                                VALUES (?, ?, ?, ?, ?)
                            """,
                                (user_id, budget_id, threshold, datetime.now(), current_usage),
                            )

                            logger.warning(
                                "budget_alert_triggered",
                                user_id=user_id,
                                budget_type=budget_type,
                                threshold=threshold,
                                current_usage=current_usage,
                                budget_limit=budget_limit,
                                utilization_percent=utilization_percent,
                            )

            conn.commit()

        finally:
            conn.close()

    async def _get_current_usage(
        self,
        user_id: str,
        budget_type: BudgetPeriod,
        provider: str | None = None,
        model: str | None = None,
    ) -> float:
        """Get current usage for a specific budget period."""
        now = datetime.now()

        # Calculate period start based on budget type
        if budget_type == BudgetPeriod.DAILY:
            period_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif budget_type == BudgetPeriod.WEEKLY:
            days_since_monday = now.weekday()
            period_start = (now - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        elif budget_type == BudgetPeriod.MONTHLY:
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif budget_type == BudgetPeriod.YEARLY:
            period_start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            period_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Query database for usage in this period
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            query = """
                SELECT SUM(total_cost) FROM requests
                WHERE user_id = ? AND timestamp >= ? AND success = 1
            """
            params = [user_id, period_start]

            if provider:
                query += " AND provider = ?"
                params.append(provider)

            if model:
                query += " AND model = ?"
                params.append(model)

            cursor.execute(query, params)
            result = cursor.fetchone()[0]
            return result or 0.0

        finally:
            conn.close()

    async def get_budget_status(self, user_id: str) -> dict[str, Any]:
        """Get current budget status and alerts for a user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Get all budgets
            cursor.execute(
                """
                SELECT budget_type, budget_limit, provider, model, alert_thresholds
                FROM budgets WHERE user_id = ?
            """,
                (user_id,),
            )

            budgets = []
            for (
                budget_type,
                budget_limit,
                provider,
                model,
                alert_thresholds_json,
            ) in cursor.fetchall():
                current_usage = await self._get_current_usage(
                    user_id, BudgetPeriod(budget_type), provider, model
                )
                utilization_percent = (current_usage / budget_limit) * 100

                budgets.append(
                    {
                        "type": budget_type,
                        "limit": budget_limit,
                        "current_usage": round(current_usage, 4),
                        "utilization_percent": round(utilization_percent, 1),
                        "provider": provider,
                        "model": model,
                        "alert_thresholds": json.loads(alert_thresholds_json),
                        "status": (
                            "exceeded"
                            if utilization_percent >= 100
                            else "warning" if utilization_percent >= 80 else "normal"
                        ),
                    }
                )

            # Get recent alerts
            cursor.execute(
                """
                SELECT ba.alert_threshold, ba.triggered_at, ba.current_usage, b.budget_type, b.budget_limit
                FROM budget_alerts ba
                JOIN budgets b ON ba.budget_id = b.id
                WHERE b.user_id = ? AND ba.triggered_at >= ?
                ORDER BY ba.triggered_at DESC
                LIMIT 10
            """,
                (user_id, datetime.now() - timedelta(days=7)),
            )

            recent_alerts = []
            for threshold, triggered_at, usage, budget_type, limit in cursor.fetchall():
                recent_alerts.append(
                    {
                        "threshold": threshold,
                        "triggered_at": triggered_at,
                        "usage": usage,
                        "budget_type": budget_type,
                        "budget_limit": limit,
                    }
                )

            return {
                "budgets": budgets,
                "recent_alerts": recent_alerts,
                "total_budgets": len(budgets),
                "exceeded_budgets": len([b for b in budgets if b["status"] == "exceeded"]),
                "warning_budgets": len([b for b in budgets if b["status"] == "warning"]),
            }

        finally:
            conn.close()

    async def get_cost_analytics(
        self,
        user_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
        group_by: str = "day",
    ) -> dict[str, Any]:
        """Get detailed cost analytics."""
        if not start_date:
            start_date = datetime.now().date() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now().date()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Base query
            base_query = """
                SELECT * FROM requests
                WHERE user_id = ? AND DATE(timestamp) BETWEEN ? AND ? AND success = 1
            """

            cursor.execute(base_query, (user_id, start_date, end_date))
            requests = cursor.fetchall()

            # Column names for easier access
            columns = [desc[0] for desc in cursor.description]

            # Convert to list of dicts
            request_data = []
            for row in requests:
                request_data.append(dict(zip(columns, row, strict=False)))

            # Calculate analytics
            total_cost = sum(r["total_cost"] for r in request_data)
            total_requests = len(request_data)

            # Group by analysis
            if group_by == "provider":
                grouped = {}
                for r in request_data:
                    provider = r["provider"]
                    if provider not in grouped:
                        grouped[provider] = {"cost": 0, "requests": 0}
                    grouped[provider]["cost"] += r["total_cost"]
                    grouped[provider]["requests"] += 1

            elif group_by == "model":
                grouped = {}
                for r in request_data:
                    model = f"{r['provider']}/{r['model']}"
                    if model not in grouped:
                        grouped[model] = {"cost": 0, "requests": 0}
                    grouped[model]["cost"] += r["total_cost"]
                    grouped[model]["requests"] += 1

            else:  # group by day
                grouped = {}
                for r in request_data:
                    day = r["timestamp"][:10]  # Extract date part
                    if day not in grouped:
                        grouped[day] = {"cost": 0, "requests": 0}
                    grouped[day]["cost"] += r["total_cost"]
                    grouped[day]["requests"] += 1

            # Cascade vs single routing comparison
            cascade_requests = [r for r in request_data if r["routing_strategy"] == "cascade"]
            single_requests = [r for r in request_data if r["routing_strategy"] == "single"]

            cascade_stats = {
                "total_requests": len(cascade_requests),
                "total_cost": sum(r["total_cost"] for r in cascade_requests),
                "avg_cost": (
                    sum(r["total_cost"] for r in cascade_requests) / len(cascade_requests)
                    if cascade_requests
                    else 0
                ),
                "avg_steps": (
                    sum(r["cascade_steps"] or 0 for r in cascade_requests) / len(cascade_requests)
                    if cascade_requests
                    else 0
                ),
                "avg_quality": (
                    sum(r["quality_score"] or 0 for r in cascade_requests) / len(cascade_requests)
                    if cascade_requests
                    else 0
                ),
            }

            single_stats = {
                "total_requests": len(single_requests),
                "total_cost": sum(r["total_cost"] for r in single_requests),
                "avg_cost": (
                    sum(r["total_cost"] for r in single_requests) / len(single_requests)
                    if single_requests
                    else 0
                ),
            }

            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": (end_date - start_date).days + 1,
                },
                "summary": {
                    "total_cost": round(total_cost, 4),
                    "total_requests": total_requests,
                    "avg_cost_per_request": (
                        round(total_cost / total_requests, 4) if total_requests > 0 else 0
                    ),
                },
                "grouped_data": {
                    k: {"cost": round(v["cost"], 4), "requests": v["requests"]}
                    for k, v in grouped.items()
                },
                "routing_comparison": {"cascade": cascade_stats, "single": single_stats},
                "generated_at": datetime.now().isoformat(),
            }

        finally:
            conn.close()
