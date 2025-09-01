# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Enhanced authentication middleware for ModelMuxer.

This module provides comprehensive authentication and authorization
middleware with support for multiple authentication methods.
"""

import hashlib
import time
from typing import Any

import jwt
import structlog
from fastapi import Header, HTTPException, Request

from ..core.exceptions import AuthenticationError
from ..core.utils import generate_request_id

logger = structlog.get_logger(__name__)


class AuthMiddleware:
    """
    Enhanced authentication middleware with multiple auth methods.

    Supports API key authentication, JWT tokens, and custom authentication
    schemes with rate limiting and user management.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

        # Authentication methods
        self.auth_methods = self.config.get("auth_methods", ["api_key"])
        self.api_keys = set(self.config.get("api_keys", []))
        self.jwt_secret = self.config.get("jwt_secret_key", "default-secret-change-in-production")
        self.jwt_algorithm = self.config.get("jwt_algorithm", "HS256")

        # Rate limiting
        self.enable_rate_limiting = self.config.get("enable_rate_limiting", True)
        self.rate_limit_storage: dict[str, dict[str, Any]] = {}
        self.default_rate_limits = {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "requests_per_day": 10000,
        }

        # User management
        self.users: dict[str, dict[str, Any]] = {}
        self.load_users()

        # Security settings
        self.require_https = self.config.get("require_https", False)
        self.allowed_origins = self.config.get("allowed_origins", ["*"])

        logger.info(
            "auth_middleware_initialized",
            auth_methods=self.auth_methods,
            rate_limiting=self.enable_rate_limiting,
            api_keys_count=len(self.api_keys),
        )

    def load_users(self) -> None:
        """Load user configurations."""
        users_config = self.config.get("users", {})

        for user_id, user_data in users_config.items():
            self.users[user_id] = {
                "user_id": user_id,
                "name": user_data.get("name", user_id),
                "email": user_data.get("email", ""),
                "role": user_data.get("role", "user"),
                "permissions": user_data.get("permissions", []),
                "rate_limits": user_data.get("rate_limits", self.default_rate_limits),
                "enabled": user_data.get("enabled", True),
                "created_at": user_data.get("created_at", time.time()),
            }

    async def authenticate_request(
        self, request: Request, authorization: str | None = Header(None)
    ) -> dict[str, Any]:
        """
        Authenticate a request using configured methods.

        Args:
            request: FastAPI request object
            authorization: Authorization header value

        Returns:
            User information dictionary

        Raises:
            HTTPException: If authentication fails
        """
        # Security checks
        if self.require_https and request.url.scheme != "https":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "HTTPS is required",
                        "type": "security_error",
                        "code": "https_required",
                    }
                },
            )

        # Try each authentication method
        user_info = None
        auth_method_used = None

        for method in self.auth_methods:
            try:
                if method == "api_key":
                    user_info = await self._authenticate_api_key(authorization)
                    auth_method_used = "api_key"
                    break
                elif method == "jwt":
                    user_info = await self._authenticate_jwt(authorization)
                    auth_method_used = "jwt"
                    break
                elif method == "custom":
                    user_info = await self._authenticate_custom(request, authorization)
                    auth_method_used = "custom"
                    break
            except AuthenticationError:
                continue  # Try next method

        if not user_info:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "Authentication failed. Please provide a valid API key or token.",
                        "type": "authentication_error",
                        "code": "authentication_failed",
                    }
                },
            )

        # Add authentication metadata
        user_info["auth_method"] = auth_method_used
        user_info["request_id"] = generate_request_id()
        user_info["authenticated_at"] = time.time()

        # Check rate limits
        if self.enable_rate_limiting:
            await self._check_rate_limits(user_info)

        # Log successful authentication
        logger.info(
            "request_authenticated",
            user_id=user_info.get("user_id"),
            auth_method=auth_method_used,
            request_id=user_info["request_id"],
        )

        return user_info

    async def _authenticate_api_key(self, authorization: str | None) -> dict[str, Any]:
        """Authenticate using API key."""
        if not authorization:
            raise AuthenticationError("Missing authorization header")

        # Extract API key
        api_key = None
        if authorization.startswith("Bearer "):
            api_key = authorization[7:]
        elif authorization.startswith("sk-"):
            api_key = authorization
        else:
            raise AuthenticationError("Invalid authorization format")

        # Validate API key
        if api_key not in self.api_keys:
            raise AuthenticationError("Invalid API key")

        # Generate user ID from API key
        user_id = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        # Get or create user info
        if user_id in self.users:
            user_info = self.users[user_id].copy()
        else:
            user_info = {
                "user_id": user_id,
                "name": f"API User {user_id[:8]}",
                "email": "",
                "role": "api_user",
                "permissions": ["chat_completion"],
                "rate_limits": self.default_rate_limits,
                "enabled": True,
                "created_at": time.time(),
            }

        user_info["api_key"] = api_key
        return user_info

    async def _authenticate_jwt(self, authorization: str | None) -> dict[str, Any]:
        """Authenticate using JWT token."""
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthenticationError("Missing or invalid JWT token")

        token = authorization[7:]

        try:
            # Decode JWT token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])

            # Extract user information
            user_id = payload.get("sub")
            if not user_id:
                raise AuthenticationError("Invalid token: missing user ID")

            # Check token expiration
            exp = payload.get("exp")
            if exp and exp < time.time():
                raise AuthenticationError("Token expired")

            # Get user info
            if user_id in self.users:
                user_info = self.users[user_id].copy()
            else:
                # Create user from token payload
                user_info = {
                    "user_id": user_id,
                    "name": payload.get("name", user_id),
                    "email": payload.get("email", ""),
                    "role": payload.get("role", "user"),
                    "permissions": payload.get("permissions", ["chat_completion"]),
                    "rate_limits": self.default_rate_limits,
                    "enabled": True,
                    "created_at": time.time(),
                }

            # Check if user is enabled
            if not user_info.get("enabled", True):
                raise AuthenticationError("User account disabled")

            user_info["token"] = token
            return user_info

        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid JWT token: {str(e)}") from e

    async def _authenticate_custom(
        self, request: Request, authorization: str | None
    ) -> dict[str, Any]:
        """Custom authentication method (placeholder for extension)."""
        # This is a placeholder for custom authentication logic
        # Organizations can extend this method for their specific needs
        raise AuthenticationError("Custom authentication not implemented")

    async def _check_rate_limits(self, user_info: dict[str, Any]) -> None:
        """Check rate limits for the user."""
        user_id = user_info["user_id"]
        rate_limits = user_info.get("rate_limits", self.default_rate_limits)

        current_time = time.time()
        current_minute = int(current_time // 60)
        current_hour = int(current_time // 3600)
        current_day = int(current_time // 86400)

        # Initialize user rate limit storage
        if user_id not in self.rate_limit_storage:
            self.rate_limit_storage[user_id] = {
                "minute_requests": {},
                "hour_requests": {},
                "day_requests": {},
            }

        user_limits = self.rate_limit_storage[user_id]

        # Clean old entries
        user_limits["minute_requests"] = {
            k: v for k, v in user_limits["minute_requests"].items() if k >= current_minute - 1
        }
        user_limits["hour_requests"] = {
            k: v for k, v in user_limits["hour_requests"].items() if k >= current_hour - 1
        }
        user_limits["day_requests"] = {
            k: v for k, v in user_limits["day_requests"].items() if k >= current_day - 1
        }

        # Count current requests
        minute_count = user_limits["minute_requests"].get(current_minute, 0)
        hour_count = user_limits["hour_requests"].get(current_hour, 0)
        day_count = user_limits["day_requests"].get(current_day, 0)

        # Check limits
        if minute_count >= rate_limits.get("requests_per_minute", float("inf")):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": {
                        "message": "Rate limit exceeded: too many requests per minute",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
                headers={"Retry-After": str(60 - int(current_time % 60))},
            )

        if hour_count >= rate_limits.get("requests_per_hour", float("inf")):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": {
                        "message": "Rate limit exceeded: too many requests per hour",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
                headers={"Retry-After": str(3600 - int(current_time % 3600))},
            )

        if day_count >= rate_limits.get("requests_per_day", float("inf")):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": {
                        "message": "Rate limit exceeded: too many requests per day",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
                headers={"Retry-After": str(86400 - int(current_time % 86400))},
            )

        # Increment counters
        user_limits["minute_requests"][current_minute] = minute_count + 1
        user_limits["hour_requests"][current_hour] = hour_count + 1
        user_limits["day_requests"][current_day] = day_count + 1

    def create_jwt_token(
        self,
        user_id: str,
        expires_in: int = 3600,
        additional_claims: dict[str, Any] | None = None,
    ) -> str:
        """Create a JWT token for a user."""
        payload = {"sub": user_id, "iat": int(time.time()), "exp": int(time.time()) + expires_in}

        if user_id in self.users:
            user_info = self.users[user_id]
            payload.update({
                "name": user_info["name"],
                "email": user_info["email"],
                "role": user_info["role"],
                "permissions": user_info["permissions"],
            })

        if additional_claims:
            payload.update(additional_claims)

        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    def get_user_info(self, user_id: str) -> dict[str, Any] | None:
        """Get user information."""
        return self.users.get(user_id)

    def add_user(self, user_data: dict[str, Any]) -> bool:
        """Add a new user."""
        user_id = user_data.get("user_id")
        if not user_id:
            return False

        self.users[user_id] = {
            "user_id": user_id,
            "name": user_data.get("name", user_id),
            "email": user_data.get("email", ""),
            "role": user_data.get("role", "user"),
            "permissions": user_data.get("permissions", ["chat_completion"]),
            "rate_limits": user_data.get("rate_limits", self.default_rate_limits),
            "enabled": user_data.get("enabled", True),
            "created_at": time.time(),
        }

        logger.info("user_added", user_id=user_id)
        return True

    def get_rate_limit_stats(self) -> dict[str, Any]:
        """Get rate limiting statistics."""
        total_users = len(self.rate_limit_storage)
        active_users = sum(
            1 for user_data in self.rate_limit_storage.values() if any(user_data.values())
        )

        return {
            "total_users": total_users,
            "active_users": active_users,
            "rate_limiting_enabled": self.enable_rate_limiting,
            "default_limits": self.default_rate_limits,
        }
