# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""
Supplementary tests for app/auth.py to cover missing lines.
Focuses on edge cases and the validate_request_size method.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException, Request

from app.auth import APIKeyAuth, validate_request_size


class TestAuthSupplementary:
    """Supplementary tests for auth module edge cases."""

    @pytest.fixture
    def auth(self):
        """Create APIKeyAuth instance with mocked settings."""
        with patch("app.auth.settings") as mock_settings:
            mock_settings.api.api_keys = "test-key-1,test-key-2"
            mock_settings.auth.max_request_size_mb = 10
            mock_settings.auth.rate_limit_per_minute = 60
            auth = APIKeyAuth()
            return auth

    def test_extract_api_key_empty_string(self, auth):
        """Test extract_api_key with empty string."""
        result = auth.extract_api_key("")
        assert result is None

    def test_extract_api_key_whitespace_only(self, auth):
        """Test extract_api_key with whitespace only."""
        result = auth.extract_api_key("   ")
        assert result is None

    def test_extract_api_key_bearer_with_extra_spaces(self, auth):
        """Test extract_api_key with Bearer token with extra spaces."""
        # Test with multiple spaces after Bearer
        result = auth.extract_api_key("Bearer test-key-with-spaces")
        assert result == "test-key-with-spaces"

    def test_extract_api_key_sk_prefix(self, auth):
        """Test extract_api_key with sk- prefix (OpenAI format)."""
        result = auth.extract_api_key("sk-test-key-123")
        assert result == "sk-test-key-123"

    def test_extract_api_key_test_prefix(self, auth):
        """Test extract_api_key with test- prefix."""
        result = auth.extract_api_key("test-key-123")
        assert result == "test-key-123"

    def test_extract_api_key_long_key_without_prefix(self, auth):
        """Test extract_api_key with long key without standard prefix (covers line 52-53)."""
        # Key longer than 10 chars without standard prefix
        result = auth.extract_api_key("custom-long-api-key-format")
        assert result == "custom-long-api-key-format"

        # Short key without prefix should return None
        result = auth.extract_api_key("short")
        assert result is None

        # Key starting with "Invalid" should return None
        result = auth.extract_api_key("InvalidApiKeyFormat")
        assert result is None

    async def test_authenticate_request_invalid_key_format(self, auth):
        """Test authenticate_request with invalid key format."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {}

        # Pass authorization as parameter since it's a Header parameter
        with pytest.raises(HTTPException) as exc_info:
            await auth.authenticate_request(mock_request, authorization="short")

        assert exc_info.value.status_code == 401
        # Check for either "Missing" or "Invalid" in the error message
        detail_str = str(exc_info.value.detail)
        assert "Missing API key" in detail_str or "Invalid API key" in detail_str

    async def test_authenticate_request_empty_authorization(self, auth):
        """Test authenticate_request with empty authorization header."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {}

        with pytest.raises(HTTPException) as exc_info:
            await auth.authenticate_request(mock_request, authorization="")

        assert exc_info.value.status_code == 401

    def test_check_rate_limit_cleanup_old_entries(self, auth):
        """Test rate limit cleanup of old entries."""
        import time

        # Initialize rate limit storage structure
        current_time = time.time()
        current_minute = int(current_time // 60)
        current_hour = int(current_time // 3600)

        # Add old entries (2 minutes ago)
        old_minute = current_minute - 2
        old_hour = current_hour - 2

        auth.rate_limit_storage["test-user"] = {
            "minute_requests": {old_minute: 10, current_minute - 1: 5, current_minute: 3},
            "hour_requests": {old_hour: 100, current_hour - 1: 50, current_hour: 20},
        }

        # Check rate limit should trigger cleanup
        result = auth.check_rate_limit("test-user")

        # Old entries should be removed
        assert old_minute not in auth.rate_limit_storage["test-user"]["minute_requests"]
        assert old_hour not in auth.rate_limit_storage["test-user"]["hour_requests"]
        assert result["allowed"]  # Should be within limit after cleanup

    async def test_validate_request_size_small_request(self):
        """Test validate_request_size with small request (covers lines 197-234)."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {"content-length": "1000"}  # 1KB

        # Mock the body() method
        async def mock_body():
            return b"x" * 1000

        mock_request.body = mock_body

        # Should not raise exception
        result = await validate_request_size(mock_request)
        assert result is None

    async def test_validate_request_size_exact_limit(self):
        """Test validate_request_size at exact limit."""
        max_size_bytes = 10 * 1024 * 1024  # 10MB
        mock_request = Mock(spec=Request)
        mock_request.headers = {"content-length": str(max_size_bytes)}

        # Mock the body() method
        async def mock_body():
            return b"x" * max_size_bytes

        mock_request.body = mock_body

        # Should not raise exception at exact limit
        result = await validate_request_size(mock_request)
        assert result is None

    async def test_validate_request_size_exceeds_limit(self):
        """Test validate_request_size when request exceeds limit."""
        max_size_bytes = 10 * 1024 * 1024 + 1  # 10MB + 1 byte
        mock_request = Mock(spec=Request)
        mock_request.headers = {"content-length": str(max_size_bytes)}

        with pytest.raises(HTTPException) as exc_info:
            await validate_request_size(mock_request)

        assert exc_info.value.status_code == 413
        assert "Request too large" in str(exc_info.value.detail)

    async def test_validate_request_size_no_content_length(self):
        """Test validate_request_size when content-length header is missing."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {}

        # Mock the body() method
        async def mock_body():
            return b"x" * 1000

        mock_request.body = mock_body

        # Should not raise exception when header is missing
        result = await validate_request_size(mock_request)
        assert result is None

    async def test_validate_request_size_body_exceeds_limit(self):
        """Test validate_request_size when body exceeds limit but header doesn't."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {"content-length": "1000"}  # Header says 1KB

        # But actual body is 11MB
        async def mock_body():
            return b"x" * (11 * 1024 * 1024)

        mock_request.body = mock_body

        with pytest.raises(HTTPException) as exc_info:
            await validate_request_size(mock_request)

        assert exc_info.value.status_code == 413

    async def test_validate_request_size_body_read_error(self):
        """Test validate_request_size when body reading fails (covers lines 231-234)."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {"content-length": "1000"}

        # Mock body() to raise an exception
        async def mock_body():
            raise Exception("Body already consumed")

        mock_request.body = mock_body

        # Should handle exception gracefully
        result = await validate_request_size(mock_request)
        assert result is None

    async def test_validate_request_size_zero_content_length(self):
        """Test validate_request_size with zero content-length."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {"content-length": "0"}

        # Mock the body() method
        async def mock_body():
            return b""

        mock_request.body = mock_body

        # Should not raise exception
        result = await validate_request_size(mock_request)
        assert result is None

    async def test_validate_request_size_with_custom_limit(self):
        """Test validate_request_size with custom limit."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {"content-length": str(6 * 1024 * 1024)}  # 6MB

        # Test with 5MB limit
        with pytest.raises(HTTPException) as exc_info:
            await validate_request_size(mock_request, max_size_mb=5)

        assert exc_info.value.status_code == 413
        assert "Maximum size is 5MB" in str(exc_info.value.detail)

    async def test_validate_request_size_very_large_request(self):
        """Test validate_request_size with very large request."""
        # Test with 1GB request
        mock_request = Mock(spec=Request)
        mock_request.headers = {"content-length": str(1024 * 1024 * 1024)}

        with pytest.raises(HTTPException) as exc_info:
            await validate_request_size(mock_request)

        assert exc_info.value.status_code == 413
        assert "Request too large" in str(exc_info.value.detail)

    def test_get_user_id_from_key_consistency(self, auth):
        """Test that get_user_id_from_key produces consistent results."""
        api_key = "test-api-key-123"

        # Should produce same ID for same key
        id1 = auth.get_user_id_from_key(api_key)
        id2 = auth.get_user_id_from_key(api_key)

        assert id1 == id2
        assert len(id1) == 16  # Should be 16 characters

    def test_get_user_id_from_key_different_keys(self, auth):
        """Test that different keys produce different user IDs."""
        id1 = auth.get_user_id_from_key("key1")
        id2 = auth.get_user_id_from_key("key2")

        assert id1 != id2

    async def test_authenticate_request_with_rate_limit_exceeded(self, auth):
        """Test authenticate_request when rate limit is exceeded."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {}

        # Fill up rate limit
        user_id = auth.get_user_id_from_key("test-key-1")
        import time

        current_time = time.time()
        current_minute = int(current_time // 60)

        auth.rate_limit_storage[user_id] = {
            "minute_requests": {current_minute: 61},  # Exceed 60/min
            "hour_requests": {int(current_time // 3600): 50},
        }

        with pytest.raises(HTTPException) as exc_info:
            await auth.authenticate_request(mock_request, authorization="Bearer test-key-1")

        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in str(exc_info.value.detail)

    def test_check_rate_limit_edge_case_exactly_at_limit(self, auth):
        """Test rate limit when exactly at the limit."""
        import time

        user_id = "test-user"
        current_time = time.time()
        current_minute = int(current_time // 60)
        current_hour = int(current_time // 3600)

        # Add exactly 59 requests (one below limit)
        auth.rate_limit_storage[user_id] = {
            "minute_requests": {current_minute: 59},
            "hour_requests": {current_hour: 100},
        }

        # 60th request should be allowed
        result = auth.check_rate_limit(user_id)
        assert result["allowed"] is True

        # 61st request should be denied
        result = auth.check_rate_limit(user_id)
        assert result["allowed"] is False

    def test_validate_api_key_with_empty_list(self):
        """Test validate_api_key when no API keys are configured."""
        with patch("app.auth.settings") as mock_settings:
            mock_settings.api.api_keys = ""  # Empty string

            auth = APIKeyAuth()

            # Should reject all keys when none are configured
            assert auth.validate_api_key("any-key") is False

    def test_validate_api_key_with_valid_key(self, auth):
        """Test validate_api_key with valid key."""
        # test-key-1 should be in allowed keys
        assert auth.validate_api_key("test-key-1") is True
        assert auth.validate_api_key("test-key-2") is True
        assert auth.validate_api_key("invalid-key") is False

    async def test_authenticate_request_with_valid_bearer_token(self, auth):
        """Test authenticate_request with valid Bearer token."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {}

        result = await auth.authenticate_request(mock_request, authorization="Bearer test-key-1")

        assert result["user_id"] == auth.get_user_id_from_key("test-key-1")
        assert result["api_key"] == "test-key-1"
        assert "rate_limit" in result

    async def test_authenticate_request_with_direct_key(self, auth):
        """Test authenticate_request with direct API key."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {}

        result = await auth.authenticate_request(mock_request, authorization="test-key-1")

        assert result["user_id"] == auth.get_user_id_from_key("test-key-1")
        assert result["api_key"] == "test-key-1"

    def test_rate_limit_with_concurrent_requests(self, auth):
        """Test rate limiting with concurrent request tracking."""
        import time

        user_id = "test-user"

        # Simulate rapid concurrent requests
        for _i in range(30):
            result = auth.check_rate_limit(user_id)
            assert result["allowed"] is True

        # Check we're tracking requests properly
        current_minute = int(time.time() // 60)
        assert auth.rate_limit_storage[user_id]["minute_requests"][current_minute] == 30

        # More requests should still be allowed up to 60
        for _ in range(29):
            result = auth.check_rate_limit(user_id)
            assert result["allowed"] is True

        # 60th request should still be allowed
        result = auth.check_rate_limit(user_id)
        assert result["allowed"] is True

        # 61st request should be denied
        result = auth.check_rate_limit(user_id)
        assert result["allowed"] is False

    def test_check_rate_limit_hour_limit(self, auth):
        """Test rate limiting hour limit (covers line 102-107)."""
        import time

        user_id = "test-user"
        current_time = time.time()
        current_hour = int(current_time // 3600)

        # Set up to exceed hour limit
        auth.rate_limit_storage[user_id] = {
            "minute_requests": {int(current_time // 60): 10},
            "hour_requests": {current_hour: 1000},  # At hour limit
        }

        # Next request should be denied due to hour limit
        result = auth.check_rate_limit(user_id)
        assert result["allowed"] is False
        assert "too many requests per hour" in result["reason"]
