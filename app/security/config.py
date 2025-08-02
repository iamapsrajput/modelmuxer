# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
# Security configuration and utilities for ModelMuxer.
# Addresses common SNYK security warnings and best practices.

import hashlib
import secrets
import ssl
from typing import Any

import httpx


class SecurityConfig:
    """Centralized security configuration."""

    @staticmethod
    def get_ssl_context() -> ssl.SSLContext:
        """Get a secure SSL context for HTTP clients."""
        context = ssl.create_default_context()
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        # Disable weak protocols
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1
        # Use strong ciphers only
        context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS")
        return context

    @staticmethod
    def get_secure_httpx_client(**kwargs) -> httpx.AsyncClient:
        """Get a securely configured httpx client."""
        default_config = {
            "verify": True,  # Always verify SSL certificates
            "follow_redirects": False,  # Disable automatic redirects
            "timeout": httpx.Timeout(60.0, connect=10.0),  # Set reasonable timeouts
            "limits": httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=30.0),
        }
        # Merge with user-provided kwargs
        config = {**default_config, **kwargs}
        return httpx.AsyncClient(**config)

    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate a cryptographically secure random token."""
        return secrets.token_urlsafe(length)

    @staticmethod
    def secure_hash(data: str, salt: str | None = None) -> str:
        """Create a secure hash of the given data."""
        if salt is None:
            salt = secrets.token_hex(16)

        # Use SHA-256 with salt
        hash_obj = hashlib.sha256()
        hash_obj.update(f"{salt}{data}".encode())
        return f"{salt}:{hash_obj.hexdigest()}"

    @staticmethod
    def verify_hash(data: str, hashed: str) -> bool:
        """Verify data against a secure hash."""
        try:
            salt, hash_value = hashed.split(":", 1)
            expected_hash = SecurityConfig.secure_hash(data, salt)
            return secrets.compare_digest(expected_hash, hashed)
        except ValueError:
            return False

    @staticmethod
    def sanitize_headers(headers: dict[str, Any]) -> dict[str, Any]:
        """Sanitize headers by removing or masking sensitive information."""
        sensitive_keys = {"authorization", "x-api-key", "cookie", "set-cookie"}
        sanitized = {}

        for key, value in headers.items():
            lower_key = key.lower()
            if any(sensitive in lower_key for sensitive in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value

        return sanitized


# Security constants
SECURE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}

# Minimum password/key requirements
MIN_API_KEY_LENGTH = 20
MIN_JWT_SECRET_LENGTH = 32
MIN_PASSWORD_LENGTH = 12

# Rate limiting defaults
DEFAULT_RATE_LIMITS = {"requests_per_minute": 60, "requests_per_hour": 1000, "requests_per_day": 10000}
