# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException

from app.core.exceptions import AuthenticationError
from app.middleware.auth_middleware import AuthMiddleware


class TestAuthMiddleware:
    """Test suite for AuthMiddleware."""

    @pytest.fixture
    def auth_middleware(self):
        """Create AuthMiddleware instance for testing."""
        return AuthMiddleware()

    def test_init(self, auth_middleware):
        """Test middleware initialization."""
        assert auth_middleware.config == {}
        assert auth_middleware.users == {}
        assert auth_middleware.rate_limit_storage == {}

    def test_init_with_config(self):
        """Test middleware initialization with config."""
        config = {"jwt_secret": "test-secret"}
        middleware = AuthMiddleware(config)
        assert middleware.config == config

    def test_load_users_empty(self, auth_middleware):
        """Test loading users with no users configured."""
        auth_middleware.load_users()
        assert auth_middleware.users == {}

    def test_load_users_from_config(self):
        """Test loading users from config."""
        config = {
            "users": {
                "user1": {"name": "User One", "rate_limits": {"requests_per_minute": 100}},
                "user2": {"name": "User Two", "rate_limits": {"requests_per_minute": 200}},
            }
        }
        middleware = AuthMiddleware(config)
        middleware.load_users()
        assert "user1" in middleware.users
        assert "user2" in middleware.users
        assert middleware.users["user1"]["name"] == "User One"
        assert middleware.users["user1"]["rate_limits"]["requests_per_minute"] == 100

    @pytest.mark.asyncio
    async def test_authenticate_request_no_authorization(self, auth_middleware):
        """Test authentication with no authorization header."""
        mock_request = Mock()
        mock_request.url.scheme = "http"
        with pytest.raises(HTTPException):  # Should raise authentication error
            await auth_middleware.authenticate_request(mock_request, None)

    @pytest.mark.asyncio
    async def test_authenticate_request_invalid_format(self, auth_middleware):
        """Test authentication with invalid authorization format."""
        mock_request = Mock()
        mock_request.url.scheme = "http"
        with pytest.raises(HTTPException):  # Should raise authentication error
            await auth_middleware.authenticate_request(mock_request, "InvalidFormat")

    @pytest.mark.asyncio
    async def test_authenticate_request_unsupported_scheme(self, auth_middleware):
        """Test authentication with unsupported scheme."""
        mock_request = Mock()
        mock_request.url.scheme = "http"
        with pytest.raises(HTTPException):  # Should raise authentication error
            await auth_middleware.authenticate_request(mock_request, "Unsupported key")

    @pytest.mark.asyncio
    async def test_authenticate_api_key_valid(self, auth_middleware):
        """Test API key authentication with valid key."""
        # Setup API keys
        auth_middleware.api_keys = {"valid-key"}

        result = await auth_middleware._authenticate_api_key("Bearer valid-key")
        assert result["user_id"] is not None
        assert result["api_key"] == "valid-key"
        assert "rate_limits" in result

    @pytest.mark.asyncio
    async def test_authenticate_api_key_invalid(self, auth_middleware):
        """Test API key authentication with invalid key."""
        with pytest.raises(AuthenticationError):  # Should raise authentication error
            await auth_middleware._authenticate_api_key("Bearer invalid-key")

    @pytest.mark.asyncio
    async def test_authenticate_jwt_valid(self, auth_middleware):
        """Test JWT authentication with valid token."""
        # Mock JWT decoding
        with patch("jwt.decode") as mock_decode:
            mock_decode.return_value = {"sub": "testuser", "exp": 2000000000}
            result = await auth_middleware._authenticate_jwt("Bearer valid-jwt")
            assert result["user_id"] == "testuser"

    @pytest.mark.asyncio
    async def test_authenticate_jwt_expired(self, auth_middleware):
        """Test JWT authentication with expired token."""
        with patch("jwt.decode") as mock_decode:
            mock_decode.side_effect = ValueError("Token expired")
            with pytest.raises(ValueError):  # Should raise exception from jwt.decode
                await auth_middleware._authenticate_jwt("Bearer expired-jwt")

    @pytest.mark.asyncio
    async def test_check_rate_limits_under_limit(self, auth_middleware):
        """Test rate limit check when under limit."""
        user_info = {"user_id": "testuser", "rate_limit": {"allowed": True}}
        # Should not raise exception
        await auth_middleware._check_rate_limits(user_info)

    @pytest.mark.asyncio
    async def test_check_rate_limits_exceeded(self, auth_middleware):
        """Test rate limit check when limit exceeded."""
        # Set up rate limits to exceed
        user_info = {
            "user_id": "testuser",
            "rate_limits": {"requests_per_minute": 0},  # Set to 0 to force exceed
        }
        with pytest.raises(HTTPException):  # Should raise rate limit error
            await auth_middleware._check_rate_limits(user_info)

    def test_create_jwt_token(self, auth_middleware):
        """Test JWT token creation."""
        with patch("jwt.encode") as mock_encode:
            mock_encode.return_value = "mock-jwt-token"
            token = auth_middleware.create_jwt_token("testuser")
            assert token == "mock-jwt-token"
            mock_encode.assert_called_once()

    def test_add_user_success(self, auth_middleware):
        """Test adding a user successfully."""
        user_data = {"user_id": "newuser", "api_key": "new-key", "rate_limit": 100}
        result = auth_middleware.add_user(user_data)
        assert result is True
        assert len(auth_middleware.users) == 1

    def test_add_user_duplicate(self, auth_middleware):
        """Test adding a duplicate user."""
        user_data = {"api_key": "existing-key", "rate_limit": 100}
        auth_middleware.add_user(user_data)
        result = auth_middleware.add_user(user_data)  # Try to add again
        assert result is False  # Should fail for duplicate

    def test_get_rate_limit_stats(self, auth_middleware):
        """Test getting rate limit statistics."""
        # Setup some rate limit data
        auth_middleware.rate_limit_storage = {"user1": {"minute_requests": {1000000: 10}}}
        stats = auth_middleware.get_rate_limit_stats()
        assert stats["total_users"] == 1
        assert stats["active_users"] == 1

    def test_get_rate_limit_stats_empty(self, auth_middleware):
        """Test getting rate limit statistics when empty."""
        stats = auth_middleware.get_rate_limit_stats()
        assert stats["total_users"] == 0
        assert stats["active_users"] == 0
