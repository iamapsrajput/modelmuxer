# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""
Tests for rate limiting middleware.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request

from app.middleware.rate_limit_middleware import RateLimitMiddleware


class TestRateLimitMiddleware:
    """Test the rate limiting middleware functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "algorithm": "sliding_window",
            "enable_global_limits": False,  # Disable for individual tests
            "enable_per_endpoint_limits": True,
            "global_limits": {"requests_per_second": 1000, "requests_per_minute": 10000},
            "endpoint_limits": {"/v1/chat/completions": {"requests_per_minute": 50}},
        }
        self.middleware = RateLimitMiddleware(self.config)

    def test_initialization(self):
        """Test middleware initialization with config."""
        assert self.middleware.algorithm == "sliding_window"
        assert self.middleware.enable_global_limits is False  # Disabled for tests
        assert self.middleware.enable_per_endpoint_limits is True
        assert self.middleware.global_limits["requests_per_second"] == 1000
        assert "/v1/chat/completions" in self.middleware.endpoint_limits

    def test_initialization_defaults(self):
        """Test middleware initialization with default config."""
        middleware = RateLimitMiddleware()
        assert middleware.algorithm == "sliding_window"
        assert middleware.enable_global_limits is True
        assert middleware.enable_per_endpoint_limits is True

    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self):
        """Test rate limit check when request is allowed."""
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/v1/chat/completions"

        user_id = "test_user"
        result = await self.middleware.check_rate_limit(mock_request, user_id)

        assert result["allowed"] is True
        assert "remaining" in result
        assert "limit" in result
        assert "reset_time" in result

    @pytest.mark.asyncio
    async def test_check_rate_limit_denied(self):
        """Test rate limit check when request is denied."""
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/v1/chat/completions"

        user_id = "test_user"
        user_limits = {"requests_per_minute": 1}

        # First request should be allowed
        result1 = await self.middleware.check_rate_limit(mock_request, user_id, user_limits)
        assert result1["allowed"] is True

        # Second request should be denied
        with pytest.raises(HTTPException) as exc_info:
            await self.middleware.check_rate_limit(mock_request, user_id, user_limits)

        assert exc_info.value.status_code == 429
        assert "rate_limit_error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_global_rate_limit_exceeded(self):
        """Test global rate limit enforcement."""
        # Create middleware with global limits enabled
        config = {
            "algorithm": "sliding_window",
            "enable_global_limits": True,
            "enable_per_endpoint_limits": False,
            "global_limits": {"requests_per_second": 10, "requests_per_minute": 100},
        }
        middleware = RateLimitMiddleware(config)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/v1/chat/completions"

        # Exhaust global per-second limit
        for i in range(11):  # More than the limit of 10
            try:
                await middleware.check_rate_limit(mock_request, f"user_{i}")
            except HTTPException:
                if i >= 10:  # Should start failing after 10 requests
                    assert True
                    return

        # If we get here, the test failed
        assert False, "Global rate limit was not enforced"

    @pytest.mark.asyncio
    async def test_endpoint_rate_limit_exceeded(self):
        """Test per-endpoint rate limit enforcement."""
        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/v1/chat/completions"

        # Exhaust endpoint limit
        for i in range(51):  # More than the limit of 50
            try:
                await self.middleware.check_rate_limit(mock_request, f"user_{i}")
            except HTTPException as e:
                if i >= 50:  # Should start failing after 50 requests
                    assert "endpoint_rate_limit_exceeded" in str(e.detail)
                    return

        # If we get here, the test failed
        assert False, "Endpoint rate limit was not enforced"

    @pytest.mark.asyncio
    async def test_token_bucket_algorithm(self):
        """Test token bucket rate limiting algorithm."""
        config = {"algorithm": "token_bucket"}
        middleware = RateLimitMiddleware(config)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/v1/chat/completions"

        user_limits = {"burst_size": 5, "requests_per_second": 1}

        # Should allow burst requests
        for i in range(5):
            result = await middleware.check_rate_limit(mock_request, "test_user", user_limits)
            assert result["allowed"] is True

        # Next request should be denied
        with pytest.raises(HTTPException):
            await middleware.check_rate_limit(mock_request, "test_user", user_limits)

    @pytest.mark.asyncio
    async def test_fixed_window_algorithm(self):
        """Test fixed window rate limiting algorithm."""
        config = {"algorithm": "fixed_window"}
        middleware = RateLimitMiddleware(config)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/v1/chat/completions"

        user_limits = {"requests_per_minute": 2}

        # Should allow 2 requests in the window
        result1 = await middleware.check_rate_limit(mock_request, "test_user", user_limits)
        assert result1["allowed"] is True

        result2 = await middleware.check_rate_limit(mock_request, "test_user", user_limits)
        assert result2["allowed"] is True

        # Third request should be denied
        with pytest.raises(HTTPException):
            await middleware.check_rate_limit(mock_request, "test_user", user_limits)

    @pytest.mark.asyncio
    async def test_adaptive_throttling_disabled(self):
        """Test that adaptive throttling is disabled by default."""
        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/v1/chat/completions"

        result = await self.middleware.check_rate_limit(mock_request, "test_user")
        assert result["allowed"] is True
        assert self.middleware.enable_adaptive_limits is False

    @pytest.mark.asyncio
    async def test_adaptive_throttling_enabled(self):
        """Test adaptive throttling when enabled."""
        config = {
            "algorithm": "sliding_window",
            "enable_adaptive_limits": True,
            "system_load_threshold": 0.5,
        }
        middleware = RateLimitMiddleware(config)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/v1/chat/completions"

        # Mock psutil at the module level
        with patch("psutil.cpu_percent", return_value=90.0), \
             patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value.percent = 85.0

            result = await middleware.check_rate_limit(mock_request, "test_user")

            # Should still allow but with reduced remaining count
            assert result["allowed"] is True
            assert result["remaining"] < 60  # Should be reduced due to high load

    def test_get_rate_limit_stats(self):
        """Test getting rate limit statistics."""
        stats = self.middleware.get_rate_limit_stats()

        assert isinstance(stats, dict)
        assert "algorithm" in stats
        assert "active_users" in stats
        assert "global_counters" in stats
        assert "endpoint_limits_active" in stats
        assert "adaptive_throttling" in stats

        assert stats["algorithm"] == "sliding_window"
        assert stats["adaptive_throttling"] is False

    def test_reset_user_limits(self):
        """Test resetting user rate limits."""
        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/v1/chat/completions"

        user_id = "test_user"

        # Make a request to create user state
        asyncio.run(self.middleware.check_rate_limit(mock_request, user_id))

        # Reset user limits
        result = self.middleware.reset_user_limits(user_id)
        assert result is True

        # Verify user state was removed
        assert f"user:{user_id}" not in self.middleware.sliding_windows

    def test_reset_nonexistent_user(self):
        """Test resetting limits for non-existent user."""
        result = self.middleware.reset_user_limits("nonexistent_user")
        assert result is False

    @pytest.mark.asyncio
    async def test_unknown_algorithm_raises_error(self):
        """Test that unknown algorithm raises ValueError."""
        config = {"algorithm": "unknown_algorithm"}
        middleware = RateLimitMiddleware(config)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/v1/chat/completions"

        with pytest.raises(ValueError, match="Unknown rate limiting algorithm"):
            await middleware.check_rate_limit(mock_request, "test_user")

    @pytest.mark.asyncio
    async def test_rate_limit_headers(self):
        """Test that proper headers are set on rate limit exceeded."""
        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/v1/chat/completions"

        user_limits = {"requests_per_minute": 1}

        # First request
        await self.middleware.check_rate_limit(mock_request, "test_user", user_limits)

        # Second request should fail with headers
        with pytest.raises(HTTPException) as exc_info:
            await self.middleware.check_rate_limit(mock_request, "test_user", user_limits)

        # Check that headers are present
        assert exc_info.value.headers is not None
        assert "Retry-After" in exc_info.value.headers
        assert "X-RateLimit-Limit" in exc_info.value.headers
        assert "X-RateLimit-Remaining" in exc_info.value.headers
        assert "X-RateLimit-Reset" in exc_info.value.headers

    @pytest.mark.asyncio
    async def test_different_users_independent_limits(self):
        """Test that different users have independent rate limits."""
        # Create middleware with endpoint limits disabled for this test
        config = {
            "algorithm": "sliding_window",
            "enable_global_limits": False,
            "enable_per_endpoint_limits": False,
        }
        middleware = RateLimitMiddleware(config)

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/v1/chat/completions"

        user_limits = {"requests_per_minute": 1}

        # User 1 can make request
        result1 = await middleware.check_rate_limit(mock_request, "user1", user_limits)
        assert result1["allowed"] is True

        # User 2 can also make request (independent of user 1)
        result2 = await middleware.check_rate_limit(mock_request, "user2", user_limits)
        assert result2["allowed"] is True

        # User 1 cannot make another request
        with pytest.raises(HTTPException):
            await middleware.check_rate_limit(mock_request, "user1", user_limits)

        # User 2 still can make another request (different user, independent limits)
        user_limits_2 = {"requests_per_minute": 2}  # Allow 2 requests for user2
        result3 = await middleware.check_rate_limit(mock_request, "user2", user_limits_2)
        assert result3["allowed"] is True