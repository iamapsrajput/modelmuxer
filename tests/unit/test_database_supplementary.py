# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""
Supplementary tests for app/database.py to cover missing lines.
Focuses on edge cases in check_budget method.
"""

import tempfile
from datetime import date
from unittest.mock import patch

import aiosqlite
import pytest

from app.database import Database
from tests.fixtures.temp_files import temp_database


class TestDatabaseSupplementary:
    """Supplementary tests for database module edge cases."""

    @pytest.fixture
    async def temp_db(self):
        """Create a temporary database for testing."""
        with temp_database(suffix=".db") as db_path:
            # Initialize database
            db = Database(db_path)
            await db.init_database()
            yield db

    @pytest.mark.asyncio
    async def test_check_budget_user_not_found_returns_early(self, temp_db):
        """Test check_budget when user doesn't exist in database (covers line 219)."""
        # Mock ensure_user_exists to do nothing, simulating a failure to create user
        with patch.object(temp_db, "ensure_user_exists") as mock_ensure:
            # Make ensure_user_exists do nothing
            mock_ensure.return_value = None

            # Directly manipulate the database to ensure user doesn't exist
            async with aiosqlite.connect(temp_db.db_path) as db:
                # Delete any existing user to force the not found condition
                await db.execute("DELETE FROM users WHERE user_id = ?", ("nonexistent_user",))
                await db.commit()

            # Now check budget - should return user not found since ensure_user_exists failed
            result = await temp_db.check_budget("nonexistent_user", 0.01)

            # Should return user not found since ensure_user_exists was mocked to do nothing
            assert result["allowed"] is False
            assert result["reason"] == "User not found"

    @pytest.mark.asyncio
    async def test_check_budget_monthly_limit_exceeded(self, temp_db):
        """Test check_budget when monthly budget is exceeded (covers line 259)."""
        user_id = "test_user"

        # Create user with small monthly budget
        await temp_db.ensure_user_exists(user_id)

        # Update user's monthly budget to a small value
        async with aiosqlite.connect(temp_db.db_path) as db:
            await db.execute("UPDATE users SET monthly_budget = 10.0 WHERE user_id = ?", (user_id,))
            await db.commit()

        # Add usage that's within daily but exceeds monthly
        async with aiosqlite.connect(temp_db.db_path) as db:
            # Add monthly usage
            await db.execute(
                """
                INSERT OR REPLACE INTO monthly_usage (user_id, year, month, total_cost, request_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, date.today().year, date.today().month, 9.5, 10),
            )
            await db.commit()

        # Try to make a request that would exceed monthly budget
        result = await temp_db.check_budget(user_id, 1.0)

        assert result["allowed"] is False
        assert result["reason"] == "Monthly budget exceeded"
        assert result["monthly_budget"] == 10.0
        assert result["monthly_used"] == 9.5

    @pytest.mark.asyncio
    async def test_check_budget_daily_limit_with_existing_usage(self, temp_db):
        """Test check_budget when daily limit is reached with existing usage."""
        user_id = "test_user"

        # Create user
        await temp_db.ensure_user_exists(user_id)

        # Add daily usage close to limit
        async with aiosqlite.connect(temp_db.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO daily_usage (user_id, date, total_cost, request_count)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, date.today(), 99.5, 50),
            )
            await db.commit()

        # Try to make a request that would exceed daily budget
        result = await temp_db.check_budget(user_id, 1.0)

        assert result["allowed"] is False
        assert result["reason"] == "Daily budget exceeded"
        assert result["daily_budget"] == 100.0
        assert result["daily_used"] == 99.5

    @pytest.mark.asyncio
    async def test_check_budget_both_limits_exceeded_daily_first(self, temp_db):
        """Test check_budget when both limits would be exceeded (daily check comes first)."""
        user_id = "test_user"

        # Create user
        await temp_db.ensure_user_exists(user_id)

        # Add usage close to both limits
        async with aiosqlite.connect(temp_db.db_path) as db:
            # Daily usage
            await db.execute(
                """
                INSERT OR REPLACE INTO daily_usage (user_id, date, total_cost, request_count)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, date.today(), 99.5, 50),
            )
            # Monthly usage
            await db.execute(
                """
                INSERT OR REPLACE INTO monthly_usage (user_id, year, month, total_cost, request_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, date.today().year, date.today().month, 999.5, 500),
            )
            await db.commit()

        # Try to make a request that would exceed both budgets
        result = await temp_db.check_budget(user_id, 1.0)

        # Daily check comes first in the code, so it should report daily exceeded
        assert result["allowed"] is False
        assert result["reason"] == "Daily budget exceeded"

    @pytest.mark.asyncio
    async def test_check_budget_exactly_at_monthly_limit(self, temp_db):
        """Test check_budget when exactly at monthly limit."""
        user_id = "test_user"

        # Create user
        await temp_db.ensure_user_exists(user_id)

        # Add usage exactly at monthly limit
        async with aiosqlite.connect(temp_db.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO monthly_usage (user_id, year, month, total_cost, request_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, date.today().year, date.today().month, 999.0, 100),
            )
            await db.commit()

        # Request that would put us exactly at limit
        result = await temp_db.check_budget(user_id, 1.0)

        # Should be allowed (exactly at limit, not over)
        assert result["allowed"] is True
        assert result["monthly_remaining"] == 1.0

    @pytest.mark.asyncio
    async def test_check_budget_with_zero_budgets(self, temp_db):
        """Test check_budget when user has zero budgets set."""
        user_id = "test_user"

        # Create user
        await temp_db.ensure_user_exists(user_id)

        # Set budgets to zero
        async with aiosqlite.connect(temp_db.db_path) as db:
            await db.execute(
                "UPDATE users SET daily_budget = 0.0, monthly_budget = 0.0 WHERE user_id = ?",
                (user_id,),
            )
            await db.commit()

        # Any request should be denied
        result = await temp_db.check_budget(user_id, 0.01)

        assert result["allowed"] is False
        assert result["reason"] == "Daily budget exceeded"

    @pytest.mark.asyncio
    async def test_check_budget_with_negative_budgets(self, temp_db):
        """Test check_budget when user has negative budgets (edge case)."""
        user_id = "test_user"

        # Create user
        await temp_db.ensure_user_exists(user_id)

        # Set budgets to negative (shouldn't happen but testing edge case)
        async with aiosqlite.connect(temp_db.db_path) as db:
            await db.execute(
                "UPDATE users SET daily_budget = -10.0, monthly_budget = -100.0 WHERE user_id = ?",
                (user_id,),
            )
            await db.commit()

        # Any positive request should be denied
        result = await temp_db.check_budget(user_id, 0.01)

        assert result["allowed"] is False

    @pytest.mark.asyncio
    async def test_check_budget_with_very_large_request(self, temp_db):
        """Test check_budget with very large request amount."""
        user_id = "test_user"

        # Create user
        await temp_db.ensure_user_exists(user_id)

        # Try a very large request
        result = await temp_db.check_budget(user_id, 1000000.0)

        assert result["allowed"] is False
        assert result["reason"] == "Daily budget exceeded"
        assert result["estimated_cost"] == 1000000.0

    @pytest.mark.asyncio
    async def test_check_budget_creates_missing_usage_records(self, temp_db):
        """Test that check_budget handles missing usage records gracefully."""
        user_id = "test_user"

        # Create user
        await temp_db.ensure_user_exists(user_id)

        # Ensure no usage records exist
        async with aiosqlite.connect(temp_db.db_path) as db:
            await db.execute("DELETE FROM daily_usage WHERE user_id = ?", (user_id,))
            await db.execute("DELETE FROM monthly_usage WHERE user_id = ?", (user_id,))
            await db.commit()

        # Should handle missing records and treat as zero usage
        result = await temp_db.check_budget(user_id, 10.0)

        assert result["allowed"] is True
        assert result["daily_remaining"] == 100.0
        assert result["monthly_remaining"] == 1000.0

    @pytest.mark.asyncio
    async def test_check_budget_with_future_date_usage(self, temp_db):
        """Test check_budget with usage from future date (shouldn't happen but edge case)."""
        user_id = "test_user"

        # Create user
        await temp_db.ensure_user_exists(user_id)

        # Add usage for tomorrow (edge case)
        from datetime import timedelta

        tomorrow = date.today() + timedelta(days=1)

        async with aiosqlite.connect(temp_db.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO daily_usage (user_id, date, total_cost, request_count)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, tomorrow, 50.0, 25),
            )
            await db.commit()

        # Today's budget should not be affected by tomorrow's usage
        result = await temp_db.check_budget(user_id, 10.0)

        assert result["allowed"] is True
        assert result["daily_remaining"] == 100.0  # Full daily budget

    @pytest.mark.asyncio
    async def test_ensure_user_exists_with_null_budgets(self, temp_db):
        """Test ensure_user_exists handles NULL budget values."""
        user_id = "test_user"

        # Manually insert user with NULL budgets
        async with aiosqlite.connect(temp_db.db_path) as db:
            await db.execute(
                "INSERT INTO users (user_id, daily_budget, monthly_budget) VALUES (?, NULL, NULL)",
                (user_id,),
            )
            await db.commit()

        # ensure_user_exists should handle this gracefully
        await temp_db.ensure_user_exists(user_id)

        # Check that user still exists (shouldn't crash)
        async with aiosqlite.connect(temp_db.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM users WHERE user_id = ?", (user_id,))
            row = await cursor.fetchone()
            assert row is not None
            count = row[0]
            assert count == 1
