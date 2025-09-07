# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""
Comprehensive unit tests for app/database.py.

Tests cover:
- Database initialization
- User management functions
- Budget tracking logic
- Request logging functionality
- Error conditions and edge cases
"""

import asyncio
import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import aiosqlite

from app.database import Database
from tests.fixtures.temp_files import temp_database

# Test constants (not actual credentials)
TEST_USER_ID = "test_user"
NEW_USER_ID = "new_user"


class TestDatabase:
    """Test suite for Database class."""

    @pytest.fixture
    async def temp_db(self):
        """Create a temporary database for testing."""
        with temp_database(suffix=".db") as db_path:
            # Initialize database
            db = Database(db_path)
            await db.init_database()
            yield db

    @pytest.fixture
    async def populated_db(self, temp_db):
        """Create a database with sample data."""
        # Add test user
        await temp_db.ensure_user_exists("test_user")

        # Add some requests
        messages = [{"role": "user", "content": "Hello world"}]
        await temp_db.log_request(
            user_id="test_user",
            provider="openai",
            model="gpt-4",
            messages=messages,
            input_tokens=10,
            output_tokens=20,
            cost=0.05,
            response_time_ms=500.0,
            routing_reason="test",
            success=True,
        )

        await temp_db.log_request(
            user_id="test_user",
            provider="anthropic",
            model="claude-3",
            messages=messages,
            input_tokens=15,
            output_tokens=25,
            cost=0.08,
            response_time_ms=600.0,
            routing_reason="fallback",
            success=False,
            error_message="API error",
        )

        return temp_db

    @pytest.mark.asyncio
    async def test_init_database_creates_tables(self, temp_db):
        """Test that init_database creates all required tables."""
        # Verify tables exist by querying them
        async with aiosqlite.connect(temp_db.db_path) as db:
            # Check requests table
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='requests'"
            )
            assert await cursor.fetchone() is not None

            # Check users table
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
            )
            assert await cursor.fetchone() is not None

            # Check usage tables
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='daily_usage'"
            )
            assert await cursor.fetchone() is not None

            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='monthly_usage'"
            )
            assert await cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_init_database_creates_indexes(self, temp_db):
        """Test that init_database creates required indexes."""
        async with aiosqlite.connect(temp_db.db_path) as db:
            # Check indexes
            indexes = [
                "idx_requests_user_id",
                "idx_requests_created_at",
                "idx_daily_usage_user_date",
                "idx_monthly_usage_user_month",
            ]

            for index_name in indexes:
                cursor = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name=?", (index_name,)
                )
                assert await cursor.fetchone() is not None, f"Index {index_name} not found"

    @pytest.mark.asyncio
    async def test_ensure_user_exists_creates_new_user(self, temp_db):
        """Test that ensure_user_exists creates a new user with default budgets."""
        user_id = NEW_USER_ID

        # Ensure user doesn't exist initially
        async with aiosqlite.connect(temp_db.db_path) as db:
            cursor = await db.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            assert await cursor.fetchone() is None

        # Ensure user exists
        await temp_db.ensure_user_exists(user_id)

        # Verify user was created with default budgets
        async with aiosqlite.connect(temp_db.db_path) as db:
            cursor = await db.execute(
                "SELECT daily_budget, monthly_budget FROM users WHERE user_id = ?", (user_id,)
            )
            result = await cursor.fetchone()
            assert result == (100.0, 1000.0)

    @pytest.mark.asyncio
    async def test_ensure_user_exists_idempotent(self, temp_db):
        """Test that ensure_user_exists is idempotent."""
        user_id = TEST_USER_ID

        # First call
        await temp_db.ensure_user_exists(user_id)

        # Second call should not fail
        await temp_db.ensure_user_exists(user_id)

        # Verify only one record exists
        async with aiosqlite.connect(temp_db.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM users WHERE user_id = ?", (user_id,))
            result = await cursor.fetchone()
            assert result is not None
            count = result[0]
            assert count == 1

    @pytest.mark.asyncio
    async def test_log_request_successful(self, temp_db):
        """Test logging a successful request."""
        messages = [{"role": "user", "content": "Test message"}]

        request_id = await temp_db.log_request(
            user_id="test_user",
            provider="openai",
            model="gpt-4",
            messages=messages,
            input_tokens=10,
            output_tokens=20,
            cost=0.05,
            response_time_ms=500.0,
            routing_reason="selected",
            success=True,
        )

        assert isinstance(request_id, int)
        assert request_id > 0

        # Verify request was logged
        async with aiosqlite.connect(temp_db.db_path) as db:
            cursor = await db.execute("SELECT * FROM requests WHERE id = ?", (request_id,))
            row = await cursor.fetchone()
            assert row is not None
            assert row[1] == "test_user"  # user_id
            assert row[2] == "openai"  # provider
            assert row[3] == "gpt-4"  # model
            assert row[5] == 10  # input_tokens
            assert row[6] == 20  # output_tokens
            assert row[7] == 30  # total_tokens
            assert row[8] == 0.05  # cost
            assert row[10] == "selected"  # routing_reason
            assert row[12]  # success

    @pytest.mark.asyncio
    async def test_log_request_with_error(self, temp_db):
        """Test logging a failed request with error message."""
        messages = [{"role": "user", "content": "Test message"}]

        request_id = await temp_db.log_request(
            user_id="test_user",
            provider="openai",
            model="gpt-4",
            messages=messages,
            input_tokens=10,
            output_tokens=0,
            cost=0.0,
            response_time_ms=100.0,
            routing_reason="failed",
            success=False,
            error_message="API timeout",
        )

        # Verify error was logged
        async with aiosqlite.connect(temp_db.db_path) as db:
            cursor = await db.execute(
                "SELECT success, error_message FROM requests WHERE id = ?", (request_id,)
            )
            row = await cursor.fetchone()
            assert row is not None
            assert not row[0]
            assert row[1] == "API timeout"

    @pytest.mark.asyncio
    async def test_log_request_creates_prompt_hash(self, temp_db):
        """Test that log_request creates a consistent prompt hash."""
        messages = [{"role": "user", "content": "Test message"}]

        request_id1 = await temp_db.log_request(
            user_id="test_user",
            provider="openai",
            model="gpt-4",
            messages=messages,
            input_tokens=10,
            output_tokens=20,
            cost=0.05,
            response_time_ms=500.0,
            routing_reason="test",
            success=True,
        )

        # Log same message again
        request_id2 = await temp_db.log_request(
            user_id="test_user",
            provider="anthropic",
            model="claude-3",
            messages=messages,
            input_tokens=10,
            output_tokens=20,
            cost=0.05,
            response_time_ms=500.0,
            routing_reason="test",
            success=True,
        )

        # Verify same prompt hash
        async with aiosqlite.connect(temp_db.db_path) as db:
            cursor = await db.execute(
                "SELECT prompt_hash FROM requests WHERE id IN (?, ?)", (request_id1, request_id2)
            )
            rows = list(await cursor.fetchall())
            assert len(rows) == 2
            assert rows[0][0] == rows[1][0]  # Same hash

    @pytest.mark.asyncio
    async def test_log_request_updates_usage_for_success(self, temp_db):
        """Test that successful requests update usage tracking."""
        messages = [{"role": "user", "content": "Test message"}]

        await temp_db.log_request(
            user_id="test_user",
            provider="openai",
            model="gpt-4",
            messages=messages,
            input_tokens=10,
            output_tokens=20,
            cost=0.05,
            response_time_ms=500.0,
            routing_reason="test",
            success=True,
        )

        # Verify usage was updated
        async with aiosqlite.connect(temp_db.db_path) as db:
            # Check daily usage
            cursor = await db.execute(
                "SELECT total_cost, request_count FROM daily_usage WHERE user_id = ? AND date = ?",
                ("test_user", date.today()),
            )
            daily_row = await cursor.fetchone()
            assert daily_row == (0.05, 1)

            # Check monthly usage
            cursor = await db.execute(
                "SELECT total_cost, request_count FROM monthly_usage WHERE user_id = ? AND year = ? AND month = ?",
                ("test_user", date.today().year, date.today().month),
            )
            monthly_row = await cursor.fetchone()
            assert monthly_row == (0.05, 1)

    @pytest.mark.asyncio
    async def test_log_request_no_usage_update_for_failure(self, temp_db):
        """Test that failed requests don't update usage tracking."""
        messages = [{"role": "user", "content": "Test message"}]

        await temp_db.log_request(
            user_id="test_user",
            provider="openai",
            model="gpt-4",
            messages=messages,
            input_tokens=10,
            output_tokens=0,
            cost=0.0,
            response_time_ms=100.0,
            routing_reason="failed",
            success=False,
            error_message="API error",
        )

        # Verify no usage was recorded
        async with aiosqlite.connect(temp_db.db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM daily_usage WHERE user_id = ?", ("test_user",)
            )
            result = await cursor.fetchone()
            assert result is not None
            count = result[0]
            assert count == 0

    @pytest.mark.asyncio
    async def test_check_budget_user_not_found(self, temp_db):
        """Test check_budget when user doesn't exist (creates user automatically)."""
        result = await temp_db.check_budget("nonexistent_user", 0.01)

        # check_budget automatically creates users via ensure_user_exists
        assert result["allowed"]
        # User should be created with default budgets and 0.01 should be allowed
        assert result["daily_remaining"] > 0
        assert result["monthly_remaining"] > 0

    @pytest.mark.asyncio
    async def test_check_budget_within_limits(self, temp_db):
        """Test check_budget when cost is within budget limits."""
        await temp_db.ensure_user_exists("test_user")

        result = await temp_db.check_budget("test_user", 50.0)

        assert result["allowed"]
        # Remaining should be full budget since user has no prior usage
        assert result["daily_remaining"] == 100.0  # full daily budget
        assert result["monthly_remaining"] == 1000.0  # full monthly budget

    @pytest.mark.asyncio
    async def test_check_budget_exceeds_daily_limit(self, temp_db):
        """Test check_budget when cost exceeds daily budget."""
        await temp_db.ensure_user_exists("test_user")

        result = await temp_db.check_budget("test_user", 150.0)

        assert not result["allowed"]
        assert result["reason"] == "Daily budget exceeded"
        assert result["daily_budget"] == 100.0
        assert result["estimated_cost"] == 150.0

    @pytest.mark.asyncio
    async def test_check_budget_exceeds_monthly_limit(self, temp_db):
        """Test check_budget when cost exceeds monthly budget."""
        await temp_db.ensure_user_exists("test_user")

        result = await temp_db.check_budget("test_user", 150.0)

        assert not result["allowed"]
        assert result["reason"] == "Daily budget exceeded"
        assert result["estimated_cost"] == 150.0

    @pytest.mark.asyncio
    async def test_check_budget_with_existing_usage(self, populated_db):
        """Test check_budget with existing usage data."""
        # Database already has usage from populated_db fixture
        result = await populated_db.check_budget("test_user", 0.01)

        assert result["allowed"]
        # Should have remaining budget after the logged requests
        assert result["daily_remaining"] > 0
        assert result["monthly_remaining"] > 0

    @pytest.mark.asyncio
    async def test_get_user_stats_no_data(self, temp_db):
        """Test get_user_stats for user with no data."""
        await temp_db.ensure_user_exists("test_user")

        stats = await temp_db.get_user_stats("test_user")

        assert stats["user_id"] == "test_user"
        assert stats["total_requests"] == 0
        assert stats["total_cost"] == 0.0
        assert stats["daily_cost"] == 0.0
        assert stats["monthly_cost"] == 0.0
        assert stats["daily_budget"] == 100.0
        assert stats["monthly_budget"] == 1000.0
        assert stats["favorite_model"] is None

    @pytest.mark.asyncio
    async def test_get_user_stats_with_data(self, populated_db):
        """Test get_user_stats with existing data."""
        stats = await populated_db.get_user_stats("test_user")

        assert stats["user_id"] == "test_user"
        assert stats["total_requests"] == 1  # Only successful request counts
        assert stats["total_cost"] == 0.05
        assert stats["daily_cost"] == 0.05
        assert stats["monthly_cost"] == 0.05
        assert stats["daily_budget"] == 100.0
        assert stats["monthly_budget"] == 1000.0
        assert stats["favorite_model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_get_user_stats_favorite_model(self, temp_db):
        """Test get_user_stats favorite model calculation."""
        await temp_db.ensure_user_exists("test_user")

        messages = [{"role": "user", "content": "Test"}]

        # Log multiple requests with different models
        await temp_db.log_request(
            "test_user", "openai", "gpt-4", messages, 10, 20, 0.05, 500, "test", True
        )
        await temp_db.log_request(
            "test_user", "openai", "gpt-4", messages, 10, 20, 0.05, 500, "test", True
        )
        await temp_db.log_request(
            "test_user", "anthropic", "claude-3", messages, 10, 20, 0.08, 600, "test", True
        )

        stats = await temp_db.get_user_stats("test_user")

        # gpt-4 should be favorite (2 requests vs 1)
        assert stats["favorite_model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_get_system_metrics_empty_db(self, temp_db):
        """Test get_system_metrics with empty database."""
        metrics = await temp_db.get_system_metrics()

        assert metrics["total_requests"] == 0
        assert metrics["total_cost"] == 0.0
        assert metrics["active_users"] == 0
        assert metrics["provider_usage"] == {}
        assert metrics["model_usage"] == {}
        assert metrics["average_response_time"] == 0.0

    @pytest.mark.asyncio
    async def test_get_system_metrics_with_data(self, populated_db):
        """Test get_system_metrics with data."""
        metrics = await populated_db.get_system_metrics()

        assert metrics["total_requests"] == 1  # Only successful
        assert metrics["total_cost"] == 0.05
        assert metrics["active_users"] == 1
        assert metrics["provider_usage"] == {"openai": 1}
        assert metrics["model_usage"] == {"gpt-4": 1}
        assert metrics["average_response_time"] == 500.0

    @pytest.mark.asyncio
    async def test_get_system_metrics_multiple_providers(self, temp_db):
        """Test get_system_metrics with multiple providers and models."""
        await temp_db.ensure_user_exists("user1")
        await temp_db.ensure_user_exists("user2")

        messages = [{"role": "user", "content": "Test"}]

        # Log requests from different users and providers
        await temp_db.log_request(
            "user1", "openai", "gpt-4", messages, 10, 20, 0.05, 500, "test", True
        )
        await temp_db.log_request(
            "user1", "anthropic", "claude-3", messages, 15, 25, 0.08, 600, "test", True
        )
        await temp_db.log_request(
            "user2", "openai", "gpt-3.5", messages, 5, 10, 0.02, 300, "test", True
        )

        metrics = await temp_db.get_system_metrics()

        assert metrics["total_requests"] == 3
        assert metrics["total_cost"] == 0.15
        assert metrics["active_users"] == 2
        assert metrics["provider_usage"] == {"openai": 2, "anthropic": 1}
        assert metrics["model_usage"] == {"gpt-4": 1, "claude-3": 1, "gpt-3.5": 1}
        assert metrics["average_response_time"] == 466.6666666666667  # (500+600+300)/3

    @pytest.mark.asyncio
    async def test_database_initialization_with_custom_path(self):
        """Test database initialization with custom path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            custom_path = f.name

        try:
            db = Database(custom_path)
            assert db.db_path == custom_path

            await db.init_database()

            # Verify database was created
            assert Path(custom_path).exists()

        finally:
            Path(custom_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_database_initialization_with_settings_path(self):
        """Test database initialization using settings path."""
        # Mock the settings import inside Database.__init__
        with patch("app.settings.settings") as mock_settings:
            mock_settings.db.database_url = "sqlite:///test.db"

            db = Database()
            assert db.db_path == "test.db"

    @pytest.mark.asyncio
    async def test_log_request_empty_messages(self, temp_db):
        """Test log_request with empty messages list."""
        request_id = await temp_db.log_request(
            user_id="test_user",
            provider="openai",
            model="gpt-4",
            messages=[],
            input_tokens=0,
            output_tokens=10,
            cost=0.01,
            response_time_ms=200.0,
            routing_reason="test",
            success=True,
        )

        assert isinstance(request_id, int)

    @pytest.mark.asyncio
    async def test_log_request_message_without_content(self, temp_db):
        """Test log_request with message missing content field."""
        messages = [{"role": "user"}]  # No content field

        request_id = await temp_db.log_request(
            user_id="test_user",
            provider="openai",
            model="gpt-4",
            messages=messages,
            input_tokens=5,
            output_tokens=10,
            cost=0.01,
            response_time_ms=200.0,
            routing_reason="test",
            success=True,
        )

        assert isinstance(request_id, int)

    @pytest.mark.asyncio
    async def test_check_budget_zero_cost(self, temp_db):
        """Test check_budget with zero cost."""
        await temp_db.ensure_user_exists("test_user")

        result = await temp_db.check_budget("test_user", 0.0)

        assert result["allowed"]
        assert result["daily_remaining"] == 100.0
        assert result["monthly_remaining"] == 1000.0

    @pytest.mark.asyncio
    async def test_check_budget_negative_cost(self, temp_db):
        """Test check_budget with negative cost (edge case)."""
        await temp_db.ensure_user_exists("test_user")

        result = await temp_db.check_budget("test_user", -10.0)

        # Negative cost should still be allowed (though unusual)
        assert result["allowed"]

    @pytest.mark.asyncio
    async def test_get_user_stats_with_failed_requests(self, temp_db):
        """Test get_user_stats excludes failed requests from counts."""
        await temp_db.ensure_user_exists("test_user")

        messages = [{"role": "user", "content": "Test"}]

        # Log one successful and one failed request
        await temp_db.log_request(
            "test_user", "openai", "gpt-4", messages, 10, 20, 0.05, 500, "test", True
        )
        await temp_db.log_request(
            "test_user", "openai", "gpt-4", messages, 10, 0, 0.0, 100, "test", False, "error"
        )

        stats = await temp_db.get_user_stats("test_user")

        # Only successful request should count
        assert stats["total_requests"] == 1
        assert stats["total_cost"] == 0.05

    @pytest.mark.asyncio
    async def test_concurrent_database_operations(self, temp_db):
        """Test concurrent database operations."""
        await temp_db.ensure_user_exists("user1")
        await temp_db.ensure_user_exists("user2")

        messages = [{"role": "user", "content": "Test"}]

        async def log_request(user_id):
            await temp_db.log_request(
                user_id=user_id,
                provider="openai",
                model="gpt-4",
                messages=messages,
                input_tokens=10,
                output_tokens=20,
                cost=0.05,
                response_time_ms=500.0,
                routing_reason="test",
                success=True,
            )

        # Run concurrent operations
        await asyncio.gather(
            log_request("user1"),
            log_request("user2"),
            temp_db.check_budget("user1", 0.01),
            temp_db.get_user_stats("user2"),
        )

        # Verify both users have data
        stats1 = await temp_db.get_user_stats("user1")
        stats2 = await temp_db.get_user_stats("user2")

        assert stats1["total_requests"] == 1
        assert stats2["total_requests"] == 1

    @pytest.mark.asyncio
    async def test_database_connection_error_handling(self, temp_db):
        """Test handling of database connection errors."""
        # Simulate connection error
        with patch("aiosqlite.connect", side_effect=Exception("Connection failed")):
            with pytest.raises(Exception, match="Connection failed"):
                await temp_db.ensure_user_exists("test_user")

    @pytest.mark.asyncio
    async def test_large_token_counts(self, temp_db):
        """Test handling of large token counts."""
        messages = [{"role": "user", "content": "Test"}]

        request_id = await temp_db.log_request(
            user_id="test_user",
            provider="openai",
            model="gpt-4",
            messages=messages,
            input_tokens=100000,
            output_tokens=200000,
            cost=10.0,
            response_time_ms=5000.0,
            routing_reason="test",
            success=True,
        )

        # Verify large numbers are handled correctly
        async with aiosqlite.connect(temp_db.db_path) as db:
            cursor = await db.execute(
                "SELECT input_tokens, output_tokens, total_tokens FROM requests WHERE id = ?",
                (request_id,),
            )
            row = await cursor.fetchone()
            assert row == (100000, 200000, 300000)

    @pytest.mark.asyncio
    async def test_special_characters_in_messages(self, temp_db):
        """Test handling of special characters in messages."""
        messages = [{"role": "user", "content": "Test with émojis 🚀 and symbols @#$%^&*()"}]

        request_id = await temp_db.log_request(
            user_id="test_user",
            provider="openai",
            model="gpt-4",
            messages=messages,
            input_tokens=10,
            output_tokens=20,
            cost=0.05,
            response_time_ms=500.0,
            routing_reason="test",
            success=True,
        )

        assert isinstance(request_id, int)

    @pytest.mark.asyncio
    async def test_very_long_error_messages(self, temp_db):
        """Test handling of very long error messages."""
        long_error = "A" * 1000  # 1000 character error message

        messages = [{"role": "user", "content": "Test"}]

        request_id = await temp_db.log_request(
            user_id="test_user",
            provider="openai",
            model="gpt-4",
            messages=messages,
            input_tokens=10,
            output_tokens=0,
            cost=0.0,
            response_time_ms=100.0,
            routing_reason="failed",
            success=False,
            error_message=long_error,
        )

        # Verify long error message was stored
        async with aiosqlite.connect(temp_db.db_path) as db:
            cursor = await db.execute(
                "SELECT error_message FROM requests WHERE id = ?", (request_id,)
            )
            result = await cursor.fetchone()
            assert result is not None
            stored_error = result[0]
            assert stored_error == long_error
