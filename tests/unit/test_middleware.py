# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Comprehensive tests for middleware modules to improve coverage.
"""

import json
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import Request, Response
from starlette.datastructures import Headers
from starlette.middleware.base import RequestResponseEndpoint

from app.middleware.auth_middleware import AuthMiddleware
from app.middleware.logging_middleware import LoggingMiddleware
from app.middleware.rate_limit_middleware import RateLimitMiddleware


class TestLoggingMiddleware:
    """Test LoggingMiddleware functionality."""

    @pytest.fixture
    def middleware_config(self):
        """Create middleware configuration."""
        return Mock(
            enabled=True,
            log_requests=True,
            log_responses=True,
            log_errors=True,
            exclude_paths=["/health", "/metrics"],
            include_headers=True,
            include_body=True,
            max_body_size=1000,
            mask_sensitive_data=True,
            sensitive_fields=["password", "api_key", "token"],
        )

    @pytest.fixture
    def middleware(self, middleware_config):
        """Create LoggingMiddleware instance."""
        return LoggingMiddleware(middleware_config)

    @pytest.fixture
    def mock_request(self):
        """Create a mock request."""
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/v1/chat/completions"
        request.headers = Headers({"content-type": "application/json", "authorization": "Bearer test-key"})
        request.client = Mock(host="127.0.0.1", port=12345)
        request.path_params = {}
        request.query_params = {}
        return request

    async def test_dispatch_success(self, middleware, mock_request):
        """Test successful request dispatch."""

        async def call_next(request):
            return Response(content=json.dumps({"result": "success"}), status_code=200)

        with patch("app.middleware.logging_middleware.logger") as mock_logger:
            response = await middleware.dispatch(mock_request, call_next)

            assert response.status_code == 200
            # Check that request was logged
            assert mock_logger.info.called

    async def test_dispatch_with_error(self, middleware, mock_request):
        """Test request dispatch with error."""

        async def call_next(request):
            raise ValueError("Test error")

        with patch("app.middleware.logging_middleware.logger") as mock_logger:
            with pytest.raises(ValueError):
                await middleware.dispatch(mock_request, call_next)

            # Check that error was logged
            assert mock_logger.error.called

    async def test_dispatch_excluded_path(self, middleware):
        """Test that excluded paths are not logged."""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/health"
        request.headers = Headers({})

        async def call_next(request):
            return Response(content="OK", status_code=200)

        with patch("app.middleware.logging_middleware.logger") as mock_logger:
            response = await middleware.dispatch(request, call_next)

            assert response.status_code == 200
            # Should not log excluded paths
            assert not mock_logger.info.called

    async def test_log_request(self, middleware, mock_request):
        """Test request logging."""
        with patch("app.middleware.logging_middleware.logger") as mock_logger:
            await middleware._log_request(mock_request)

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "request_received" in call_args[0][0]

    async def test_log_response(self, middleware, mock_request):
        """Test response logging."""
        response = Response(content=json.dumps({"result": "success"}), status_code=200)

        with patch("app.middleware.logging_middleware.logger") as mock_logger:
            await middleware._log_response(mock_request, response, 100.5)

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "request_completed" in call_args[0][0]

    async def test_log_error(self, middleware, mock_request):
        """Test error logging."""
        error = Exception("Test error")

        with patch("app.middleware.logging_middleware.logger") as mock_logger:
            await middleware._log_error(mock_request, error, 100.5)

            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert "request_failed" in call_args[0][0]

    async def test_get_request_body(self, middleware, mock_request):
        """Test getting request body."""
        body_data = {"messages": [{"role": "user", "content": "Hello"}]}
        mock_request.body = AsyncMock(return_value=json.dumps(body_data).encode())

        body = await middleware._get_request_body(mock_request)
        assert body == json.dumps(body_data)

    async def test_get_request_body_large(self, middleware, mock_request):
        """Test getting large request body (should be truncated)."""
        large_body = "x" * 2000
        mock_request.body = AsyncMock(return_value=large_body.encode())

        body = await middleware._get_request_body(mock_request)
        assert len(body) <= middleware.config.max_body_size
        assert body.endswith("...")

    async def test_get_response_body(self, middleware):
        """Test getting response body."""
        response_data = {"result": "success"}
        response = Response(content=json.dumps(response_data), status_code=200)

        body = await middleware._get_response_body(response)
        assert json.loads(body) == response_data

    def test_mask_sensitive_data(self, middleware):
        """Test masking sensitive data."""
        data = {
            "password": "secret123",
            "api_key": "sk-12345",
            "token": "bearer-token",
            "normal_field": "visible",
        }

        masked = middleware._mask_sensitive_data(data)
        assert masked["password"] == "***"
        assert masked["api_key"] == "***"
        assert masked["token"] == "***"
        assert masked["normal_field"] == "visible"

    def test_mask_sensitive_data_nested(self, middleware):
        """Test masking sensitive data in nested structures."""
        data = {
            "user": {"name": "John", "password": "secret123"},
            "config": {"api_key": "sk-12345"},
        }

        masked = middleware._mask_sensitive_data(data)
        assert masked["user"]["password"] == "***"
        assert masked["config"]["api_key"] == "***"
        assert masked["user"]["name"] == "John"

    def test_format_headers(self, middleware):
        """Test formatting headers."""
        headers = Headers({
            "content-type": "application/json",
            "authorization": "Bearer secret-token",
            "x-custom-header": "value",
        })

        formatted = middleware._format_headers(headers)
        assert formatted["content-type"] == "application/json"
        assert formatted["authorization"] == "Bearer ***"  # Should be masked
        assert formatted["x-custom-header"] == "value"

    def test_should_log_path(self, middleware):
        """Test path filtering logic."""
        assert middleware._should_log_path("/v1/chat/completions") is True
        assert middleware._should_log_path("/health") is False
        assert middleware._should_log_path("/metrics") is False
        assert middleware._should_log_path("/api/users") is True


class TestAuthMiddleware:
    """Test AuthMiddleware functionality."""

    @pytest.fixture
    def auth_config(self):
        """Create auth middleware configuration."""
        return Mock(
            enabled=True,
            require_api_key=True,
            api_key_header="X-API-Key",
            bearer_token_header="Authorization",
            validate_jwt=False,
            jwt_secret=None,
            jwt_algorithm="HS256",
            exempt_paths=["/health", "/docs"],
        )

    @pytest.fixture
    def middleware(self, auth_config):
        """Create AuthMiddleware instance."""
        return AuthMiddleware(auth_config)

    @pytest.fixture
    def mock_request(self):
        """Create a mock request."""
        request = Mock(spec=Request)
        request.url = Mock()
        request.url.path = "/v1/chat/completions"
        request.headers = Headers({"x-api-key": "test-api-key"})
        return request

    async def test_dispatch_authenticated(self, middleware, mock_request):
        """Test dispatch with authenticated request."""

        async def call_next(request):
            return Response(content="OK", status_code=200)

        with patch.object(middleware, "_authenticate", return_value=True):
            response = await middleware.dispatch(mock_request, call_next)
            assert response.status_code == 200

    async def test_dispatch_unauthenticated(self, middleware, mock_request):
        """Test dispatch with unauthenticated request."""

        async def call_next(request):
            return Response(content="OK", status_code=200)

        with patch.object(middleware, "_authenticate", return_value=False):
            response = await middleware.dispatch(mock_request, call_next)
            assert response.status_code == 401
            content = json.loads(response.body)
            assert "error" in content

    async def test_dispatch_exempt_path(self, middleware):
        """Test dispatch with exempt path."""
        request = Mock(spec=Request)
        request.url = Mock()
        request.url.path = "/health"
        request.headers = Headers({})

        async def call_next(request):
            return Response(content="OK", status_code=200)

        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 200

    async def test_authenticate_with_api_key(self, middleware, mock_request):
        """Test authentication with API key."""
        with patch.object(middleware, "_validate_api_key", return_value=True):
            result = await middleware._authenticate(mock_request)
            assert result is True

    async def test_authenticate_with_bearer_token(self, middleware):
        """Test authentication with bearer token."""
        request = Mock(spec=Request)
        request.headers = Headers({"authorization": "Bearer test-token"})

        with patch.object(middleware, "_validate_bearer_token", return_value=True):
            result = await middleware._authenticate(request)
            assert result is True

    async def test_authenticate_no_credentials(self, middleware):
        """Test authentication with no credentials."""
        request = Mock(spec=Request)
        request.headers = Headers({})

        result = await middleware._authenticate(request)
        assert result is False

    async def test_validate_api_key_valid(self, middleware):
        """Test API key validation with valid key."""
        # Mock database or key store check
        with patch("app.middleware.auth_middleware.validate_api_key_in_db", return_value=True):
            result = await middleware._validate_api_key("valid-api-key")
            assert result is True

    async def test_validate_api_key_invalid(self, middleware):
        """Test API key validation with invalid key."""
        with patch("app.middleware.auth_middleware.validate_api_key_in_db", return_value=False):
            result = await middleware._validate_api_key("invalid-api-key")
            assert result is False

    async def test_validate_bearer_token(self, middleware):
        """Test bearer token validation."""
        with patch("app.middleware.auth_middleware.validate_bearer_token_in_db", return_value=True):
            result = await middleware._validate_bearer_token("Bearer valid-token")
            assert result is True

    def test_is_exempt_path(self, middleware):
        """Test exempt path checking."""
        assert middleware._is_exempt_path("/health") is True
        assert middleware._is_exempt_path("/docs") is True
        assert middleware._is_exempt_path("/api/users") is False


class TestRateLimitMiddleware:
    """Test RateLimitMiddleware functionality."""

    @pytest.fixture
    def rate_limit_config(self):
        """Create rate limit middleware configuration."""
        return Mock(
            enabled=True,
            requests_per_minute=60,
            requests_per_hour=1000,
            requests_per_day=10000,
            burst_size=10,
            exempt_paths=["/health", "/metrics"],
            by_api_key=True,
            by_ip=True,
            redis_url=None,
            use_memory_store=True,
        )

    @pytest.fixture
    def middleware(self, rate_limit_config):
        """Create RateLimitMiddleware instance."""
        return RateLimitMiddleware(rate_limit_config)

    @pytest.fixture
    def mock_request(self):
        """Create a mock request."""
        request = Mock(spec=Request)
        request.url = Mock()
        request.url.path = "/v1/chat/completions"
        request.headers = Headers({"x-api-key": "test-key"})
        request.client = Mock(host="127.0.0.1")
        return request

    async def test_dispatch_within_limit(self, middleware, mock_request):
        """Test dispatch within rate limit."""

        async def call_next(request):
            return Response(content="OK", status_code=200)

        with patch.object(middleware, "_check_rate_limit", return_value=True):
            response = await middleware.dispatch(mock_request, call_next)
            assert response.status_code == 200

    async def test_dispatch_exceeded_limit(self, middleware, mock_request):
        """Test dispatch when rate limit exceeded."""

        async def call_next(request):
            return Response(content="OK", status_code=200)

        with patch.object(middleware, "_check_rate_limit", return_value=False):
            response = await middleware.dispatch(mock_request, call_next)
            assert response.status_code == 429
            content = json.loads(response.body)
            assert "error" in content
            assert content["error"]["type"] == "rate_limit_error"

    async def test_dispatch_exempt_path(self, middleware):
        """Test dispatch with exempt path."""
        request = Mock(spec=Request)
        request.url = Mock()
        request.url.path = "/health"

        async def call_next(request):
            return Response(content="OK", status_code=200)

        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 200

    async def test_check_rate_limit_by_api_key(self, middleware, mock_request):
        """Test rate limit checking by API key."""
        result = await middleware._check_rate_limit(mock_request)
        # First request should pass
        assert result is True

    async def test_check_rate_limit_burst(self, middleware, mock_request):
        """Test burst rate limiting."""
        # Simulate burst requests
        for _i in range(middleware.config.burst_size):
            result = await middleware._check_rate_limit(mock_request)
            assert result is True

        # Next request should be rate limited
        result = await middleware._check_rate_limit(mock_request)
        assert result is False

    def test_get_client_id_with_api_key(self, middleware, mock_request):
        """Test getting client ID from API key."""
        client_id = middleware._get_client_id(mock_request)
        assert "test-key" in client_id

    def test_get_client_id_with_ip(self, middleware):
        """Test getting client ID from IP address."""
        request = Mock(spec=Request)
        request.headers = Headers({})
        request.client = Mock(host="192.168.1.1")

        client_id = middleware._get_client_id(request)
        assert "192.168.1.1" in client_id

    def test_get_client_id_no_identifier(self, middleware):
        """Test getting client ID with no identifier."""
        request = Mock(spec=Request)
        request.headers = Headers({})
        request.client = None

        client_id = middleware._get_client_id(request)
        assert client_id == "anonymous"

    async def test_increment_counter(self, middleware):
        """Test incrementing request counter."""
        client_id = "test-client"

        # First increment
        count = await middleware._increment_counter(client_id, "minute")
        assert count == 1

        # Second increment
        count = await middleware._increment_counter(client_id, "minute")
        assert count == 2

    async def test_get_current_counts(self, middleware):
        """Test getting current request counts."""
        client_id = "test-client"

        # Increment some counters
        await middleware._increment_counter(client_id, "minute")
        await middleware._increment_counter(client_id, "hour")

        counts = await middleware._get_current_counts(client_id)
        assert counts["minute"] >= 1
        assert counts["hour"] >= 1
        assert "day" in counts

    def test_is_exempt_path(self, middleware):
        """Test exempt path checking."""
        assert middleware._is_exempt_path("/health") is True
        assert middleware._is_exempt_path("/metrics") is True
        assert middleware._is_exempt_path("/api/endpoint") is False

    async def test_cleanup_old_entries(self, middleware):
        """Test cleanup of old rate limit entries."""
        # Add some entries
        client_id = "test-client"
        await middleware._increment_counter(client_id, "minute")

        # Simulate time passing
        with patch("time.time", return_value=time.time() + 3600):
            await middleware._cleanup_old_entries()

        # Old entries should be cleaned up
        counts = await middleware._get_current_counts(client_id)
        assert counts["minute"] == 0

    def test_format_retry_after(self, middleware):
        """Test formatting retry-after header."""
        retry_after = middleware._format_retry_after("minute")
        assert isinstance(retry_after, int)
        assert retry_after > 0
        assert retry_after <= 60

    def test_format_rate_limit_headers(self, middleware):
        """Test formatting rate limit headers."""
        headers = middleware._format_rate_limit_headers(limit=60, remaining=10, reset_time=int(time.time()) + 60)

        assert "X-RateLimit-Limit" in headers
        assert headers["X-RateLimit-Limit"] == "60"
        assert "X-RateLimit-Remaining" in headers
        assert headers["X-RateLimit-Remaining"] == "10"
        assert "X-RateLimit-Reset" in headers
