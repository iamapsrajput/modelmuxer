# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""
Extra tests for app/auth.py to increase coverage to 70%.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException

from app.auth import APIKeyAuth, SecurityHeaders, sanitize_user_input


class TestAuthExtra:
    """Extra tests for auth module."""

    def test_security_headers_get_security_headers(self):
        """Test SecurityHeaders.get_security_headers method."""
        headers = SecurityHeaders.get_security_headers()

        assert isinstance(headers, dict)
        assert "X-Content-Type-Options" in headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in headers
        assert headers["X-Frame-Options"] == "DENY"
        assert "X-XSS-Protection" in headers
        assert "Strict-Transport-Security" in headers
        assert "Content-Security-Policy" in headers
        assert "Referrer-Policy" in headers

    def test_sanitize_user_input_basic(self):
        """Test sanitize_user_input with basic inputs."""
        # Normal text
        result = sanitize_user_input("Hello World")
        assert result == "Hello World"

        # Text with null bytes
        result = sanitize_user_input("Hello\x00World")
        assert "\x00" not in result
        assert "Hello" in result
        assert "World" in result

        # Text with CRLF
        result = sanitize_user_input("Line1\r\nLine2")
        assert result == "Line1\nLine2"

        # Empty string
        result = sanitize_user_input("")
        assert result == ""

        # Single line
        result = sanitize_user_input("Single line text")
        assert result == "Single line text"

    def test_sanitize_user_input_whitespace(self):
        """Test sanitize_user_input with whitespace handling."""
        # Multiple newlines
        result = sanitize_user_input("Line1\n\n\n\nLine2")
        assert "Line1" in result
        assert "Line2" in result

        # Leading/trailing whitespace
        result = sanitize_user_input("  text  ")
        assert "text" in result

        # Only whitespace
        result = sanitize_user_input("   \n\n   ")
        assert result == "   "  # Keeps first line even if whitespace

    def test_sanitize_user_input_max_length(self):
        """Test sanitize_user_input with max length."""
        # Text within limit
        text = "a" * 100
        result = sanitize_user_input(text, max_length=1000)
        assert result == text

        # Text exceeding limit
        text = "a" * 1001
        with pytest.raises(HTTPException) as exc_info:
            sanitize_user_input(text, max_length=1000)

        assert exc_info.value.status_code == 400
        assert "too long" in str(exc_info.value.detail).lower()

    def test_api_key_auth_initialization(self):
        """Test APIKeyAuth initialization."""
        with patch("app.auth.settings") as mock_settings:
            mock_settings.api.api_keys = "key1,key2,key3"

            auth = APIKeyAuth()

            assert len(auth.allowed_keys) == 3
            assert "key1" in auth.allowed_keys
            assert "key2" in auth.allowed_keys
            assert "key3" in auth.allowed_keys

    def test_api_key_auth_empty_keys(self):
        """Test APIKeyAuth with no API keys configured."""
        with patch("app.auth.settings") as mock_settings:
            mock_settings.api.api_keys = ""

            auth = APIKeyAuth()

            assert len(auth.allowed_keys) == 0

    def test_api_key_auth_none_keys(self):
        """Test APIKeyAuth with None API keys."""
        with patch("app.auth.settings") as mock_settings:
            mock_settings.api.api_keys = None

            auth = APIKeyAuth()

            assert len(auth.allowed_keys) == 0

    def test_api_key_auth_whitespace_handling(self):
        """Test APIKeyAuth handles whitespace in keys."""
        with patch("app.auth.settings") as mock_settings:
            mock_settings.api.api_keys = " key1 , key2 , key3 "

            auth = APIKeyAuth()

            assert len(auth.allowed_keys) == 3
            assert "key1" in auth.allowed_keys
            assert "key2" in auth.allowed_keys
            assert "key3" in auth.allowed_keys
            # Whitespace should be stripped
            assert " key1 " not in auth.allowed_keys
