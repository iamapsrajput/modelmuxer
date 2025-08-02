# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Advanced rate limiting middleware for ModelMuxer.

This module provides sophisticated rate limiting with multiple algorithms,
distributed support, and intelligent throttling.
"""

import time
from collections import deque
from typing import Any

import structlog
from fastapi import HTTPException, Request

logger = structlog.get_logger(__name__)


class RateLimitMiddleware:
    """
    Advanced rate limiting middleware with multiple algorithms.

    Supports token bucket, sliding window, and fixed window algorithms
    with per-user, per-endpoint, and global rate limiting.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

        # Rate limiting algorithms
        self.algorithm = self.config.get(
            "algorithm", "sliding_window"
        )  # token_bucket, sliding_window, fixed_window
        self.enable_global_limits = self.config.get("enable_global_limits", True)
        self.enable_per_endpoint_limits = self.config.get("enable_per_endpoint_limits", True)

        # Default limits
        self.default_limits = {
            "requests_per_second": 10,
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "burst_size": 20,  # For token bucket algorithm
        }

        # Storage for different algorithms
        self.token_buckets: dict[str, dict[str, Any]] = {}
        self.sliding_windows: dict[str, deque] = {}
        self.fixed_windows: dict[str, dict[str, Any]] = {}

        # Global rate limiting
        self.global_limits = self.config.get(
            "global_limits", {"requests_per_second": 1000, "requests_per_minute": 10000}
        )
        self.global_counters = {
            "current_second": {"count": 0, "timestamp": int(time.time())},
            "current_minute": {"count": 0, "timestamp": int(time.time() // 60)},
        }

        # Per-endpoint limits
        self.endpoint_limits = self.config.get(
            "endpoint_limits",
            {"/v1/chat/completions": {"requests_per_second": 50, "requests_per_minute": 500}},
        )

        # Intelligent throttling
        self.enable_adaptive_limits = self.config.get("enable_adaptive_limits", False)
        self.system_load_threshold = self.config.get("system_load_threshold", 0.8)

        logger.info(
            "rate_limit_middleware_initialized",
            algorithm=self.algorithm,
            global_limits=self.enable_global_limits,
            per_endpoint_limits=self.enable_per_endpoint_limits,
        )

    async def check_rate_limit(
        self, request: Request, user_id: str, user_limits: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Check rate limits for a request.

        Args:
            request: FastAPI request object
            user_id: User identifier
            user_limits: User-specific rate limits

        Returns:
            Rate limit status information

        Raises:
            HTTPException: If rate limit is exceeded
        """
        current_time = time.time()
        endpoint = request.url.path

        # Use user-specific limits or defaults
        limits = user_limits or self.default_limits

        # Check global rate limits first
        if self.enable_global_limits:
            await self._check_global_limits(current_time)

        # Check per-endpoint limits
        if self.enable_per_endpoint_limits and endpoint in self.endpoint_limits:
            await self._check_endpoint_limits(endpoint, current_time)

        # Check user-specific limits
        rate_limit_key = f"user:{user_id}"

        if self.algorithm == "token_bucket":
            result = await self._check_token_bucket(rate_limit_key, limits, current_time)
        elif self.algorithm == "sliding_window":
            result = await self._check_sliding_window(rate_limit_key, limits, current_time)
        elif self.algorithm == "fixed_window":
            result = await self._check_fixed_window(rate_limit_key, limits, current_time)
        else:
            raise ValueError(f"Unknown rate limiting algorithm: {self.algorithm}")

        # Apply adaptive throttling if enabled
        if self.enable_adaptive_limits:
            result = await self._apply_adaptive_throttling(result, current_time)

        if not result["allowed"]:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": {
                        "message": result["reason"],
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
                headers={
                    "Retry-After": str(int(result.get("retry_after", 60))),
                    "X-RateLimit-Limit": str(result.get("limit", 0)),
                    "X-RateLimit-Remaining": str(result.get("remaining", 0)),
                    "X-RateLimit-Reset": str(int(result.get("reset_time", current_time + 60))),
                },
            )

        return result

    async def _check_global_limits(self, current_time: float) -> None:
        """Check global rate limits."""
        current_second = int(current_time)
        current_minute = int(current_time // 60)

        # Check per-second limit
        if self.global_counters["current_second"]["timestamp"] != current_second:
            self.global_counters["current_second"] = {"count": 0, "timestamp": current_second}

        if self.global_counters["current_second"]["count"] >= self.global_limits.get(
            "requests_per_second", float("inf")
        ):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": {
                        "message": "Global rate limit exceeded: too many requests per second",
                        "type": "rate_limit_error",
                        "code": "global_rate_limit_exceeded",
                    }
                },
                headers={"Retry-After": "1"},
            )

        # Check per-minute limit
        if self.global_counters["current_minute"]["timestamp"] != current_minute:
            self.global_counters["current_minute"] = {"count": 0, "timestamp": current_minute}

        if self.global_counters["current_minute"]["count"] >= self.global_limits.get(
            "requests_per_minute", float("inf")
        ):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": {
                        "message": "Global rate limit exceeded: too many requests per minute",
                        "type": "rate_limit_error",
                        "code": "global_rate_limit_exceeded",
                    }
                },
                headers={"Retry-After": str(60 - int(current_time % 60))},
            )

        # Increment counters
        self.global_counters["current_second"]["count"] += 1
        self.global_counters["current_minute"]["count"] += 1

    async def _check_endpoint_limits(self, endpoint: str, current_time: float) -> None:
        """Check per-endpoint rate limits."""
        endpoint_key = f"endpoint:{endpoint}"
        limits = self.endpoint_limits[endpoint]

        # Use sliding window for endpoint limits
        if endpoint_key not in self.sliding_windows:
            self.sliding_windows[endpoint_key] = deque()

        window = self.sliding_windows[endpoint_key]

        # Clean old entries
        cutoff_time = current_time - 60  # 1 minute window
        while window and window[0] < cutoff_time:
            window.popleft()

        # Check limits
        if len(window) >= limits.get("requests_per_minute", float("inf")):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": {
                        "message": f"Endpoint rate limit exceeded for {endpoint}",
                        "type": "rate_limit_error",
                        "code": "endpoint_rate_limit_exceeded",
                    }
                },
                headers={"Retry-After": str(int(60 - (current_time - window[0])))},
            )

        # Add current request
        window.append(current_time)

    async def _check_token_bucket(
        self, key: str, limits: dict[str, Any], current_time: float
    ) -> dict[str, Any]:
        """Check rate limit using token bucket algorithm."""
        if key not in self.token_buckets:
            self.token_buckets[key] = {
                "tokens": limits.get("burst_size", 20),
                "last_refill": current_time,
                "capacity": limits.get("burst_size", 20),
                "refill_rate": limits.get("requests_per_second", 10),
            }

        bucket = self.token_buckets[key]

        # Refill tokens based on time elapsed
        time_elapsed = current_time - bucket["last_refill"]
        tokens_to_add = time_elapsed * bucket["refill_rate"]
        bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = current_time

        # Check if request can be allowed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return {
                "allowed": True,
                "remaining": int(bucket["tokens"]),
                "limit": bucket["capacity"],
                "reset_time": current_time
                + (bucket["capacity"] - bucket["tokens"]) / bucket["refill_rate"],
            }
        else:
            return {
                "allowed": False,
                "reason": "Token bucket empty",
                "remaining": 0,
                "limit": bucket["capacity"],
                "retry_after": (1 - bucket["tokens"]) / bucket["refill_rate"],
                "reset_time": current_time + (1 - bucket["tokens"]) / bucket["refill_rate"],
            }

    async def _check_sliding_window(
        self, key: str, limits: dict[str, Any], current_time: float
    ) -> dict[str, Any]:
        """Check rate limit using sliding window algorithm."""
        if key not in self.sliding_windows:
            self.sliding_windows[key] = deque()

        window = self.sliding_windows[key]
        window_size = 60  # 1 minute window
        limit = limits.get("requests_per_minute", 60)

        # Clean old entries
        cutoff_time = current_time - window_size
        while window and window[0] < cutoff_time:
            window.popleft()

        # Check if request can be allowed
        if len(window) < limit:
            window.append(current_time)
            return {
                "allowed": True,
                "remaining": limit - len(window),
                "limit": limit,
                "reset_time": current_time + window_size,
            }
        else:
            oldest_request = window[0]
            retry_after = window_size - (current_time - oldest_request)
            return {
                "allowed": False,
                "reason": "Sliding window limit exceeded",
                "remaining": 0,
                "limit": limit,
                "retry_after": retry_after,
                "reset_time": oldest_request + window_size,
            }

    async def _check_fixed_window(
        self, key: str, limits: dict[str, Any], current_time: float
    ) -> dict[str, Any]:
        """Check rate limit using fixed window algorithm."""
        window_size = 60  # 1 minute window
        current_window = int(current_time // window_size)
        limit = limits.get("requests_per_minute", 60)

        if key not in self.fixed_windows:
            self.fixed_windows[key] = {}

        windows = self.fixed_windows[key]

        # Clean old windows
        windows = {w: count for w, count in windows.items() if w >= current_window - 1}
        self.fixed_windows[key] = windows

        # Get current window count
        current_count = windows.get(current_window, 0)

        # Check if request can be allowed
        if current_count < limit:
            windows[current_window] = current_count + 1
            return {
                "allowed": True,
                "remaining": limit - current_count - 1,
                "limit": limit,
                "reset_time": (current_window + 1) * window_size,
            }
        else:
            return {
                "allowed": False,
                "reason": "Fixed window limit exceeded",
                "remaining": 0,
                "limit": limit,
                "retry_after": (current_window + 1) * window_size - current_time,
                "reset_time": (current_window + 1) * window_size,
            }

    async def _apply_adaptive_throttling(
        self, result: dict[str, Any], current_time: float
    ) -> dict[str, Any]:
        """Apply adaptive throttling based on system load."""
        try:
            import psutil

            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent

            # Calculate system load
            system_load = max(cpu_percent / 100, memory_percent / 100)

            # Apply throttling if system is under high load
            if system_load > self.system_load_threshold:
                throttle_factor = (system_load - self.system_load_threshold) / (
                    1 - self.system_load_threshold
                )

                if result["allowed"]:
                    # Reduce remaining requests based on system load
                    original_remaining = result.get("remaining", 0)
                    result["remaining"] = int(original_remaining * (1 - throttle_factor))

                    # Potentially deny request if system is very loaded
                    if system_load > 0.95 and throttle_factor > 0.8:
                        result["allowed"] = False
                        result["reason"] = "System under high load - adaptive throttling applied"
                        result["retry_after"] = 30

                logger.warning(
                    "adaptive_throttling_applied",
                    system_load=system_load,
                    throttle_factor=throttle_factor,
                    request_allowed=result["allowed"],
                )

        except ImportError:
            # psutil not available, skip adaptive throttling
            pass
        except Exception as e:
            logger.warning("adaptive_throttling_error", error=str(e))

        return result

    def get_rate_limit_stats(self) -> dict[str, Any]:
        """Get rate limiting statistics."""
        return {
            "algorithm": self.algorithm,
            "active_users": len(self.token_buckets)
            + len(self.sliding_windows)
            + len(self.fixed_windows),
            "global_counters": self.global_counters,
            "endpoint_limits_active": len(self.endpoint_limits),
            "adaptive_throttling": self.enable_adaptive_limits,
        }

    def reset_user_limits(self, user_id: str) -> bool:
        """Reset rate limits for a specific user."""
        user_key = f"user:{user_id}"

        removed = False
        if user_key in self.token_buckets:
            del self.token_buckets[user_key]
            removed = True
        if user_key in self.sliding_windows:
            del self.sliding_windows[user_key]
            removed = True
        if user_key in self.fixed_windows:
            del self.fixed_windows[user_key]
            removed = True

        if removed:
            logger.info("user_rate_limits_reset", user_id=user_id)

        return removed
