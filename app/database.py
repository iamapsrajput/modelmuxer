# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Database operations for request logging and cost tracking.
"""

import hashlib
from datetime import date
from typing import Any

import aiosqlite


class Database:
    """Async SQLite database manager for the LLM router."""

    def __init__(self, db_path: str = None):
        # Use centralized settings for database configuration
        from .settings import settings as app_settings

        database_url = app_settings.db.database_url
        self.db_path = db_path or database_url.replace("sqlite:///", "")

    async def init_database(self) -> None:
        """Initialize database tables."""
        async with aiosqlite.connect(self.db_path) as db:
            # Requests table for logging all API calls
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    cost REAL NOT NULL,
                    response_time_ms REAL NOT NULL,
                    routing_reason TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT,
                    intent_label TEXT,
                    intent_method TEXT,
                    routing_rule TEXT
                )
            """
            )

            # Users table for budget tracking
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    daily_budget REAL DEFAULT 10.0,
                    monthly_budget REAL DEFAULT 100.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Daily usage tracking
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    date DATE NOT NULL,
                    total_cost REAL NOT NULL DEFAULT 0.0,
                    request_count INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(user_id, date),
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """
            )

            # Monthly usage tracking
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS monthly_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    month INTEGER NOT NULL,
                    total_cost REAL NOT NULL DEFAULT 0.0,
                    request_count INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(user_id, year, month),
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """
            )

            # Create indexes for better performance
            await db.execute("CREATE INDEX IF NOT EXISTS idx_requests_user_id ON requests(user_id)")
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_daily_usage_user_date ON daily_usage(user_id, date)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_monthly_usage_user_month ON monthly_usage(user_id, year, month)"
            )

            await self._ensure_routing_columns(db)
            await db.commit()

    async def _ensure_routing_columns(self, db: aiosqlite.Connection) -> None:
        """Add routing analytics columns to existing databases."""
        cursor = await db.execute("PRAGMA table_info(requests)")
        columns = {row[1] for row in await cursor.fetchall()}
        migrations = [
            ("intent_label", "TEXT"),
            ("intent_method", "TEXT"),
            ("routing_rule", "TEXT"),
        ]
        for column_name, column_type in migrations:
            if column_name not in columns:
                await db.execute(f"ALTER TABLE requests ADD COLUMN {column_name} {column_type}")

    async def ensure_user_exists(self, user_id: str) -> None:
        """Ensure user exists in the database with default budgets."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR IGNORE INTO users (user_id, daily_budget, monthly_budget)
                VALUES (?, ?, ?)
            """,
                (user_id, 100.0, 1000.0),  # Default budgets
            )
            await db.commit()

    async def log_request(
        self,
        user_id: str,
        provider: str,
        model: str,
        messages: list[dict],
        input_tokens: int,
        output_tokens: int,
        cost: float,
        response_time_ms: float,
        routing_reason: str,
        success: bool = True,
        error_message: str | None = None,
        intent_label: str | None = None,
        intent_method: str | None = None,
        routing_rule: str | None = None,
    ) -> int:
        """Log a request to the database."""
        # Create hash of the prompt for analytics (privacy-preserving)
        prompt_text = " ".join([msg.get("content", "") for msg in messages])
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]

        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_routing_columns(db)
            cursor = await db.execute(
                """
                INSERT INTO requests (
                    user_id, provider, model, prompt_hash, input_tokens,
                    output_tokens, total_tokens, cost, response_time_ms,
                    routing_reason, success, error_message,
                    intent_label, intent_method, routing_rule
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user_id,
                    provider,
                    model,
                    prompt_hash,
                    input_tokens,
                    output_tokens,
                    input_tokens + output_tokens,
                    cost,
                    response_time_ms,
                    routing_reason,
                    success,
                    error_message,
                    intent_label,
                    intent_method,
                    routing_rule,
                ),
            )

            request_id = cursor.lastrowid
            await db.commit()

            # Update usage tracking if successful
            if success:
                await self._update_usage_tracking(db, user_id, cost)

            return request_id

    async def _update_usage_tracking(self, db: aiosqlite.Connection, user_id: str, cost: float):
        """Update daily and monthly usage tracking."""
        today = date.today()
        current_year = today.year
        current_month = today.month

        # Update daily usage
        await db.execute(
            """
            INSERT INTO daily_usage (user_id, date, total_cost, request_count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(user_id, date) DO UPDATE SET
                total_cost = total_cost + ?,
                request_count = request_count + 1
        """,
            (user_id, today, cost, cost),
        )

        # Update monthly usage
        await db.execute(
            """
            INSERT INTO monthly_usage (user_id, year, month, total_cost, request_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(user_id, year, month) DO UPDATE SET
                total_cost = total_cost + ?,
                request_count = request_count + 1
        """,
            (user_id, current_year, current_month, cost, cost),
        )

        await db.commit()

    async def check_budget(self, user_id: str, estimated_cost: float) -> dict[str, Any]:
        """Check if user has budget remaining for the estimated cost."""
        await self.ensure_user_exists(user_id)

        async with aiosqlite.connect(self.db_path) as db:
            # Get user budgets
            cursor = await db.execute(
                """
                SELECT daily_budget, monthly_budget FROM users WHERE user_id = ?
            """,
                (user_id,),
            )
            user_data = await cursor.fetchone()

            if not user_data:
                return {"allowed": False, "reason": "User not found"}

            daily_budget, monthly_budget = user_data

            # Handle NULL budget values - treat as unlimited if NULL
            if daily_budget is None:
                daily_budget = float("inf")
            if monthly_budget is None:
                monthly_budget = float("inf")

            # Get current usage
            today = date.today()
            cursor = await db.execute(
                """
                SELECT COALESCE(total_cost, 0) FROM daily_usage
                WHERE user_id = ? AND date = ?
            """,
                (user_id, today),
            )
            daily_result = await cursor.fetchone()
            daily_usage = daily_result[0] if daily_result else 0.0

            cursor = await db.execute(
                """
                SELECT COALESCE(total_cost, 0) FROM monthly_usage
                WHERE user_id = ? AND year = ? AND month = ?
            """,
                (user_id, today.year, today.month),
            )
            monthly_result = await cursor.fetchone()
            monthly_usage = monthly_result[0] if monthly_result else 0.0

            # Check budgets
            daily_remaining = daily_budget - daily_usage
            monthly_remaining = monthly_budget - monthly_usage

            if daily_usage + estimated_cost > daily_budget:
                return {
                    "allowed": False,
                    "reason": "Daily budget exceeded",
                    "daily_used": daily_usage,
                    "daily_budget": daily_budget,
                    "estimated_cost": estimated_cost,
                }

            if monthly_usage + estimated_cost > monthly_budget:
                return {
                    "allowed": False,
                    "reason": "Monthly budget exceeded",
                    "monthly_used": monthly_usage,
                    "monthly_budget": monthly_budget,
                    "estimated_cost": estimated_cost,
                }

            return {
                "allowed": True,
                "daily_remaining": daily_remaining,
                "monthly_remaining": monthly_remaining,
                "estimated_cost": estimated_cost,
            }

    async def get_user_stats(self, user_id: str, days: int = 30) -> dict[str, Any]:
        """Get user usage statistics."""
        await self.ensure_user_exists(user_id)

        async with aiosqlite.connect(self.db_path) as db:
            # Get user budgets
            cursor = await db.execute(
                """
                SELECT daily_budget, monthly_budget FROM users WHERE user_id = ?
            """,
                (user_id,),
            )
            user_data = await cursor.fetchone()
            daily_budget, monthly_budget = user_data

            # Get current usage
            today = date.today()
            cursor = await db.execute(
                """
                SELECT COALESCE(total_cost, 0), COALESCE(request_count, 0)
                FROM daily_usage WHERE user_id = ? AND date = ?
            """,
                (user_id, today),
            )
            daily_data = await cursor.fetchone()
            daily_cost, daily_requests = daily_data or (0.0, 0)

            cursor = await db.execute(
                """
                SELECT COALESCE(total_cost, 0), COALESCE(request_count, 0)
                FROM monthly_usage WHERE user_id = ? AND year = ? AND month = ?
            """,
                (user_id, today.year, today.month),
            )
            monthly_data = await cursor.fetchone()
            monthly_cost, monthly_requests = monthly_data or (0.0, 0)

            # Get total stats
            cursor = await db.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(cost), 0) FROM requests
                WHERE user_id = ? AND success = TRUE
            """,
                (user_id,),
            )
            total_requests, total_cost = await cursor.fetchone()

            # Get favorite model
            cursor = await db.execute(
                """
                SELECT model, COUNT(*) as count FROM requests
                WHERE user_id = ? AND success = TRUE
                GROUP BY model ORDER BY count DESC LIMIT 1
            """,
                (user_id,),
            )
            favorite_data = await cursor.fetchone()
            favorite_model = favorite_data[0] if favorite_data else None

            return {
                "user_id": user_id,
                "total_requests": total_requests,
                "total_cost": total_cost,
                "daily_cost": daily_cost,
                "monthly_cost": monthly_cost,
                "daily_budget": daily_budget,
                "monthly_budget": monthly_budget,
                "favorite_model": favorite_model,
            }

    async def get_system_metrics(self) -> dict[str, Any]:
        """Get system-wide metrics."""
        async with aiosqlite.connect(self.db_path) as db:
            # Total requests and cost
            cursor = await db.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(cost), 0), COALESCE(AVG(response_time_ms), 0)
                FROM requests WHERE success = TRUE
            """
            )
            total_requests, total_cost, avg_response_time = await cursor.fetchone()

            # Active users (users with requests in last 30 days)
            cursor = await db.execute(
                """
                SELECT COUNT(DISTINCT user_id) FROM requests
                WHERE created_at >= datetime('now', '-30 days')
            """
            )
            active_users = (await cursor.fetchone())[0]

            # Provider usage
            cursor = await db.execute(
                """
                SELECT provider, COUNT(*) FROM requests
                WHERE success = TRUE GROUP BY provider
            """
            )
            provider_usage: dict[str, float] = dict(await cursor.fetchall())

            # Model usage
            cursor = await db.execute(
                """
                SELECT model, COUNT(*) FROM requests
                WHERE success = TRUE GROUP BY model
            """
            )
            model_usage: dict[str, float] = dict(await cursor.fetchall())

            return {
                "total_requests": total_requests,
                "total_cost": total_cost,
                "active_users": active_users,
                "provider_usage": provider_usage,
                "model_usage": model_usage,
                "average_response_time": avg_response_time,
            }

    def _analytics_filters(
        self,
        user_id: str,
        days: int,
        provider: str | None = None,
        model: str | None = None,
    ) -> tuple[str, list[Any]]:
        """Build shared WHERE clause and params for analytics queries."""
        clauses = [
            "user_id = ?",
            "success = TRUE",
            "created_at >= datetime('now', ?)",
        ]
        params: list[Any] = [user_id, f"-{days} days"]
        if provider:
            clauses.append("provider = ?")
            params.append(provider)
        if model:
            clauses.append("model = ?")
            params.append(model)
        return " AND ".join(clauses), params

    async def get_cost_analytics(
        self,
        user_id: str,
        days: int = 30,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Get cost analytics with routing metadata for dashboard views."""
        await self.ensure_user_exists(user_id)
        where_clause, params = self._analytics_filters(user_id, days, provider, model)

        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_routing_columns(db)
            cursor = await db.execute(
                f"""
                SELECT COUNT(*), COALESCE(SUM(cost), 0)
                FROM requests
                WHERE {where_clause}
            """,  # noqa: S608
                params,
            )
            total_requests, total_cost = await cursor.fetchone()

            cursor = await db.execute(
                f"""
                SELECT provider, COALESCE(SUM(cost), 0)
                FROM requests
                WHERE {where_clause}
                GROUP BY provider
                ORDER BY SUM(cost) DESC
            """,
                params,
            )
            cost_by_provider = {row[0]: float(row[1]) for row in await cursor.fetchall()}

            cursor = await db.execute(
                f"""
                SELECT model, COALESCE(SUM(cost), 0)
                FROM requests
                WHERE {where_clause}
                GROUP BY model
                ORDER BY SUM(cost) DESC
            """,
                params,
            )
            cost_by_model = {row[0]: float(row[1]) for row in await cursor.fetchall()}

            cursor = await db.execute(
                f"""
                SELECT DATE(created_at) as day, COALESCE(SUM(cost), 0), COUNT(*)
                FROM requests
                WHERE {where_clause}
                GROUP BY DATE(created_at)
                ORDER BY day ASC
            """,
                params,
            )
            daily_breakdown = [
                {"date": row[0], "cost": float(row[1]), "requests": row[2]}
                for row in await cursor.fetchall()
            ]

            cursor = await db.execute(
                f"""
                SELECT strftime('%Y-W%W', created_at) as week, COALESCE(SUM(cost), 0), COUNT(*)
                FROM requests
                WHERE {where_clause}
                GROUP BY strftime('%Y-W%W', created_at)
                ORDER BY week ASC
            """,
                params,
            )
            weekly_breakdown = [
                {"week": row[0], "cost": float(row[1]), "requests": row[2]}
                for row in await cursor.fetchall()
            ]

            cursor = await db.execute(
                f"""
                SELECT provider, model, cost, routing_reason,
                       intent_label, intent_method, routing_rule, created_at
                FROM requests
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT 25
            """,
                params,
            )
            recent_requests = [
                {
                    "provider": row[0],
                    "model": row[1],
                    "cost": float(row[2]),
                    "routing_reason": row[3],
                    "intent_label": row[4],
                    "intent_method": row[5],
                    "routing_rule": row[6],
                    "created_at": row[7],
                }
                for row in await cursor.fetchall()
            ]

        return {
            "user_id": user_id,
            "period_days": days,
            "total_cost": float(total_cost or 0.0),
            "total_requests": int(total_requests or 0),
            "cost_by_provider": cost_by_provider,
            "cost_by_model": cost_by_model,
            "daily_breakdown": daily_breakdown,
            "weekly_breakdown": weekly_breakdown,
            "recent_requests": recent_requests,
        }

    async def get_routing_analytics(self, user_id: str, days: int = 30) -> dict[str, Any]:
        """Get routing decision aggregates for dashboard views."""
        await self.ensure_user_exists(user_id)
        where_clause, params = self._analytics_filters(user_id, days)

        async with aiosqlite.connect(self.db_path) as db:
            await self._ensure_routing_columns(db)
            cursor = await db.execute(
                f"SELECT COUNT(*) FROM requests WHERE {where_clause}",
                params,
            )
            total_requests = (await cursor.fetchone())[0]

            cursor = await db.execute(
                f"""
                SELECT COALESCE(intent_label, 'unknown'), COUNT(*)
                FROM requests
                WHERE {where_clause}
                GROUP BY COALESCE(intent_label, 'unknown')
                ORDER BY COUNT(*) DESC
            """,
                params,
            )
            requests_by_intent = {row[0]: row[1] for row in await cursor.fetchall()}

            cursor = await db.execute(
                f"""
                SELECT provider, COUNT(*)
                FROM requests
                WHERE {where_clause}
                GROUP BY provider
                ORDER BY COUNT(*) DESC
            """,
                params,
            )
            provider_mix = {row[0]: row[1] for row in await cursor.fetchall()}

            cursor = await db.execute(
                f"""
                SELECT
                    SUM(CASE WHEN provider = 'ollama' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN provider != 'ollama' THEN 1 ELSE 0 END)
                FROM requests
                WHERE {where_clause}
            """,
                params,
            )
            local_count, cloud_count = await cursor.fetchone()
            local_count = int(local_count or 0)
            cloud_count = int(cloud_count or 0)

            cursor = await db.execute(
                f"""
                SELECT COALESCE(routing_rule, '_none'), COUNT(*)
                FROM requests
                WHERE {where_clause}
                GROUP BY COALESCE(routing_rule, '_none')
                ORDER BY COUNT(*) DESC
            """,
                params,
            )
            routing_rule_hits = {row[0]: row[1] for row in await cursor.fetchall()}

            cursor = await db.execute(
                f"""
                SELECT COALESCE(intent_method, 'unknown'), COUNT(*)
                FROM requests
                WHERE {where_clause}
                GROUP BY COALESCE(intent_method, 'unknown')
                ORDER BY COUNT(*) DESC
            """,
                params,
            )
            intent_method_mix = {row[0]: row[1] for row in await cursor.fetchall()}

            cursor = await db.execute(
                f"""
                SELECT model, COUNT(*), COALESCE(SUM(cost), 0)
                FROM requests
                WHERE {where_clause}
                GROUP BY model
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """,
                params,
            )
            top_models = [
                {"model": row[0], "count": row[1], "cost": float(row[2])}
                for row in await cursor.fetchall()
            ]

            cursor = await db.execute(
                f"""
                SELECT provider, COUNT(*), COALESCE(SUM(cost), 0)
                FROM requests
                WHERE {where_clause}
                GROUP BY provider
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """,
                params,
            )
            top_providers = [
                {"provider": row[0], "count": row[1], "cost": float(row[2])}
                for row in await cursor.fetchall()
            ]

        return {
            "user_id": user_id,
            "period_days": days,
            "total_requests": int(total_requests or 0),
            "requests_by_intent": requests_by_intent,
            "provider_mix": provider_mix,
            "local_vs_cloud": {"local": local_count, "cloud": cloud_count},
            "routing_rule_hits": routing_rule_hits,
            "intent_method_mix": intent_method_mix,
            "top_models": top_models,
            "top_providers": top_providers,
        }


# Global database instance
db = Database()

# Import settings for test compatibility
from .settings import settings
