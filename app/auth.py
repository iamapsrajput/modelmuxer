# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Authentication and security utilities.
"""

import hashlib
import time
from typing import Any, Dict, Optional

from fastapi import Header, HTTPException, Request
# Removed unused imports: HTTPAuthorizationCredentials, HTTPBearer

from .config import settings


class APIKeyAuth:
    """API key authentication handler."""

    def __init__(self):
        self.allowed_keys = set(settings.get_allowed_api_keys())
        # Simple rate limiting storage (in production, use Redis)
        self.rate_limit_storage: Dict[str, Dict[str, Any]] = {}

    def extract_api_key(self, authorization: Optional[str] = None) -> Optional[str]:
        """Extract API key from Authorization header."""
        if not authorization:
            return None

        # Support both "Bearer sk-..." and "sk-..." formats
        if authorization.startswith("Bearer "):
            return authorization[7:]  # Remove "Bearer " prefix
        elif authorization.startswith("sk-"):
            return authorization

        return None

    def validate_api_key(self, api_key: str) -> bool:
        """Validate if the API key is allowed."""
        return api_key in self.allowed_keys

    def get_user_id_from_key(self, api_key: str) -> str:
        """Generate a consistent user ID from API key."""
        # Create a hash of the API key for user identification
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]

    def check_rate_limit(
        self, user_id: str, requests_per_minute: int = 60, requests_per_hour: int = 1000
    ) -> Dict[str, Any]:
        """
        Simple rate limiting check.
        In production, use Redis with sliding window.
        """
        current_time = time.time()
        current_minute = int(current_time // 60)
        current_hour = int(current_time // 3600)

        if user_id not in self.rate_limit_storage:
            self.rate_limit_storage[user_id] = {"minute_requests": {}, "hour_requests": {}}

        user_limits = self.rate_limit_storage[user_id]

        # Clean old entries (keep last 2 minutes and 2 hours)
        user_limits["minute_requests"] = {
            k: v for k, v in user_limits["minute_requests"].items() if k >= current_minute - 1
        }
        user_limits["hour_requests"] = {k: v for k, v in user_limits["hour_requests"].items() if k >= current_hour - 1}

        # Count current requests
        minute_count = user_limits["minute_requests"].get(current_minute, 0)
        hour_count = user_limits["hour_requests"].get(current_hour, 0)

        # Check limits
        if minute_count >= requests_per_minute:
            return {
                "allowed": False,
                "reason": "Rate limit exceeded: too many requests per minute",
                "retry_after": 60 - (current_time % 60),
            }

        if hour_count >= requests_per_hour:
            return {
                "allowed": False,
                "reason": "Rate limit exceeded: too many requests per hour",
                "retry_after": 3600 - (current_time % 3600),
            }

        # Increment counters
        user_limits["minute_requests"][current_minute] = minute_count + 1
        user_limits["hour_requests"][current_hour] = hour_count + 1

        return {
            "allowed": True,
            "remaining_minute": requests_per_minute - minute_count - 1,
            "remaining_hour": requests_per_hour - hour_count - 1,
        }

    async def authenticate_request(
        self, request: Request, authorization: Optional[str] = Header(None)
    ) -> Dict[str, Any]:
        """
        Authenticate a request and return user information.

        Returns:
            Dict with user_id and authentication info

        Raises:
            HTTPException: If authentication fails
        """
        # Extract API key
        api_key = self.extract_api_key(authorization)

        if not api_key:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "Missing API key. Please provide a valid API key in the Authorization header.",
                        "type": "authentication_error",
                        "code": "missing_api_key",
                    }
                },
            )

        # Validate API key
        if not self.validate_api_key(api_key):
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "Invalid API key provided.",
                        "type": "authentication_error",
                        "code": "invalid_api_key",
                    }
                },
            )

        # Get user ID
        user_id = self.get_user_id_from_key(api_key)

        # Check rate limits
        rate_limit_result = self.check_rate_limit(user_id)

        if not rate_limit_result["allowed"]:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": {
                        "message": rate_limit_result["reason"],
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
                headers={"Retry-After": str(int(rate_limit_result["retry_after"]))},
            )

        return {"user_id": user_id, "api_key": api_key, "rate_limit": rate_limit_result}


class SecurityHeaders:
    """Security headers middleware."""

    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Get security headers to add to responses."""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }


def validate_request_size(request: Request, max_size_mb: int = 10) -> None:
    """Validate request size to prevent abuse."""
    content_length = request.headers.get("content-length")

    if content_length:
        size_mb = int(content_length) / (1024 * 1024)
        if size_mb > max_size_mb:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": {
                        "message": f"Request too large. Maximum size is {max_size_mb}MB.",
                        "type": "request_too_large",
                        "code": "payload_too_large",
                    }
                },
            )


def sanitize_user_input(text: str, max_length: int = 100000) -> str:
    """Sanitize user input to prevent injection attacks."""
    if len(text) > max_length:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Input too long. Maximum length is {max_length} characters.",
                    "type": "invalid_request_error",
                    "code": "input_too_long",
                }
            },
        )

    # Basic sanitization - remove null bytes and control characters
    sanitized = text.replace("\x00", "").replace("\r\n", "\n")

    # Remove excessive whitespace
    lines = sanitized.split("\n")
    sanitized_lines = []
    for line in lines:
        # Keep reasonable amount of whitespace
        if len(line.strip()) > 0 or len(sanitized_lines) == 0:
            sanitized_lines.append(line)

    return "\n".join(sanitized_lines)


# Global auth instance
auth = APIKeyAuth()
