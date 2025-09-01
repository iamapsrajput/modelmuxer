# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
# ModelMuxer Security Middleware
# Comprehensive security middleware for API protection

import json
import re
import time
from datetime import datetime
from typing import Any

import redis
import structlog
from fastapi import HTTPException, Request, Response, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.types import ASGIApp

from .auth import SecurityManager
from ..core.exceptions import ErrorResponse
from .pii_protection import PIIProtector

logger = structlog.get_logger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security middleware for API protection."""

    def __init__(
        self,
        app: ASGIApp,
        redis_client: redis.Redis,
        security_manager: SecurityManager,
        pii_protector: PIIProtector,
        config: dict[str, Any],
    ):
        super().__init__(app)
        self.redis_client = redis_client
        self.security_manager = security_manager
        self.pii_protector = pii_protector
        self.config = config

        # Security configuration
        self.rate_limits = config.get("rate_limits", {})
        self.blocked_ips = set(config.get("blocked_ips", []))
        self.allowed_origins = set(config.get("allowed_origins", ["*"]))
        self.max_request_size = config.get("max_request_size", 10 * 1024 * 1024)  # 10MB
        self.enable_pii_protection = config.get("enable_pii_protection", True)
        self.suspicious_patterns = self._compile_suspicious_patterns()

    def _compile_suspicious_patterns(self) -> list[re.Pattern]:
        """Compile patterns for detecting suspicious requests."""
        patterns = [
            # XSS - More robust script tag detection that handles malformed tags
            # Matches <script> tags with various attributes and malformed closing tags
            re.compile(r"<script\b[^>]*>.*?</script\s*[^>]*>", re.IGNORECASE | re.DOTALL),
            # Also catch script tags without proper closing
            re.compile(r"<script\b[^>]*>(?!.*</script)", re.IGNORECASE | re.DOTALL),
            # Catch other dangerous HTML elements
            re.compile(r"<iframe\b[^>]*>", re.IGNORECASE),
            re.compile(r"<object\b[^>]*>", re.IGNORECASE),
            re.compile(r"<embed\b[^>]*>", re.IGNORECASE),
            re.compile(r"javascript:", re.IGNORECASE),
            re.compile(r"vbscript:", re.IGNORECASE),
            re.compile(r"on\w+\s*=", re.IGNORECASE),  # Event handlers like onclick, onload
            re.compile(r"union\s+select", re.IGNORECASE),  # SQL injection
            re.compile(r"drop\s+table", re.IGNORECASE),  # SQL injection
            re.compile(r"exec\s*\(", re.IGNORECASE),  # Code injection
            re.compile(r"eval\s*\(", re.IGNORECASE),  # Code injection
            re.compile(r"\.\./", re.IGNORECASE),  # Path traversal
            re.compile(r"%2e%2e%2f", re.IGNORECASE),  # Encoded path traversal
        ]
        return patterns

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Main middleware dispatch method."""
        start_time = time.time()
        client_ip = self._get_client_ip(request)

        try:
            # Security checks
            await self._check_ip_blocking(client_ip)
            await self._check_rate_limiting(request, client_ip)
            await self._check_request_size(request)
            await self._check_suspicious_patterns(request)

            # Process request
            response = await call_next(request)

            # Post-processing
            await self._log_request(request, response, start_time, client_ip)
            self._add_security_headers(response)

            return response

        except HTTPException as e:
            await self._log_security_event(request, client_ip, "blocked", str(e.detail))
            return JSONResponse(
                status_code=e.status_code,
                content={"error": e.detail, "timestamp": datetime.utcnow().isoformat()},
            )
        except Exception as e:
            logger.error("security_middleware_error", error=str(e), client_ip=client_ip)
            await self._log_security_event(request, client_ip, "error", str(e))
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal security error",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (behind proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct connection
        return request.client.host if request.client else "unknown"

    async def _check_ip_blocking(self, client_ip: str) -> None:
        """Check if IP is blocked."""
        if client_ip in self.blocked_ips:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=ErrorResponse.create(
                    message="IP address is blocked", error_type="forbidden", code="ip_blocked"
                ).dict(),
            )

        # Check dynamic IP blocking (from Redis)
        if self.redis_client.exists(f"blocked_ip:{client_ip}"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=ErrorResponse.create(
                    message="IP address is temporarily blocked",
                    error_type="forbidden",
                    code="ip_temporarily_blocked",
                ).dict(),
            )

    async def _check_rate_limiting(self, request: Request, client_ip: str) -> None:
        """Check rate limiting for the request."""
        path = request.url.path
        method = request.method

        # Determine rate limit based on endpoint
        rate_limit_key = self._get_rate_limit_key(path, method)
        if not rate_limit_key:
            return

        limit_config = self.rate_limits.get(rate_limit_key, {})
        if not limit_config:
            return

        requests_per_minute = limit_config.get("requests_per_minute", 60)
        requests_per_hour = limit_config.get("requests_per_hour", 1000)

        # Check minute-based rate limit
        minute_key = f"rate_limit:{client_ip}:{rate_limit_key}:minute:{int(time.time() // 60)}"
        minute_count = self.redis_client.incr(minute_key)
        self.redis_client.expire(minute_key, 60)

        if minute_count > requests_per_minute:
            await self._handle_rate_limit_exceeded(
                client_ip, "minute", minute_count, requests_per_minute
            )

        # Check hour-based rate limit
        hour_key = f"rate_limit:{client_ip}:{rate_limit_key}:hour:{int(time.time() // 3600)}"
        hour_count = self.redis_client.incr(hour_key)
        self.redis_client.expire(hour_key, 3600)

        if hour_count > requests_per_hour:
            await self._handle_rate_limit_exceeded(client_ip, "hour", hour_count, requests_per_hour)

    def _get_rate_limit_key(self, path: str, method: str) -> str | None:
        """Determine rate limit key based on path and method."""
        if path.startswith("/v1/chat/completions"):
            return "chat_completions"
        elif path.startswith("/auth/"):
            return "auth"
        elif path.startswith("/v1/analytics/"):
            return "analytics"
        elif method == "POST":
            return "post_requests"
        elif method == "GET":
            return "get_requests"
        return None

    async def _handle_rate_limit_exceeded(
        self, client_ip: str, window: str, current: int, limit: int
    ) -> None:
        """Handle rate limit exceeded."""
        logger.warning(
            "rate_limit_exceeded", client_ip=client_ip, window=window, current=current, limit=limit
        )

        # Temporarily block IP if severely exceeding limits
        if current > limit * 2:
            self.redis_client.setex(
                f"blocked_ip:{client_ip}", 300, "rate_limit_exceeded"
            )  # 5 minutes
            logger.warning(
                "ip_temporarily_blocked", client_ip=client_ip, reason="severe_rate_limit_violation"
            )

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=ErrorResponse.create(
                message=f"Rate limit exceeded: {current}/{limit} requests per {window}",
                error_type="rate_limit_exceeded",
                code="security_rate_limit",
                details={"current": current, "limit": limit, "window": window},
            ).dict(),
            headers={"Retry-After": "60" if window == "minute" else "3600"},
        )

    async def _check_request_size(self, request: Request) -> None:
        """Check request size limits."""
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=ErrorResponse.create(
                    message=f"Request too large: {content_length} bytes (max: {self.max_request_size})",
                    error_type="request_too_large",
                    code="payload_too_large",
                    details={"size": int(content_length), "max_size": self.max_request_size},
                ).dict(),
            )

    async def _check_suspicious_patterns(self, request: Request) -> None:
        """Check for suspicious patterns in request."""
        # Check URL path
        path = request.url.path
        query = str(request.url.query) if request.url.query else ""

        for pattern in self.suspicious_patterns:
            if pattern.search(path) or pattern.search(query):
                client_ip = self._get_client_ip(request)
                logger.warning(
                    "suspicious_pattern_detected",
                    client_ip=client_ip,
                    path=path,
                    pattern=pattern.pattern,
                )

                # Increase suspicion score
                suspicion_key = f"suspicion:{client_ip}"
                suspicion_score = self.redis_client.incr(suspicion_key)
                self.redis_client.expire(suspicion_key, 3600)  # 1 hour

                if suspicion_score > 5:
                    # Temporarily block highly suspicious IPs
                    self.redis_client.setex(
                        f"blocked_ip:{client_ip}", 1800, "suspicious_activity"
                    )  # 30 minutes
                    logger.warning(
                        "ip_blocked_suspicious_activity", client_ip=client_ip, score=suspicion_score
                    )

                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ErrorResponse.create(
                        message="Suspicious request pattern detected",
                        error_type="security_violation",
                        code="suspicious_pattern",
                    ).dict(),
                )

    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response."""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "X-Permitted-Cross-Domain-Policies": "none",
        }

        for header, value in security_headers.items():
            response.headers[header] = value

    async def _log_request(
        self, request: Request, response: Response, start_time: float, client_ip: str
    ) -> None:
        """Log request details for security monitoring."""
        duration = time.time() - start_time

        log_data = {
            "client_ip": client_ip,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration": round(duration, 3),
            "user_agent": request.headers.get("user-agent", ""),
            "referer": request.headers.get("referer", ""),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Log authentication info if available
        auth_header = request.headers.get("authorization")
        if auth_header:
            log_data["auth_method"] = "bearer" if auth_header.startswith("Bearer") else "other"

        logger.info("api_request", **log_data)

    async def _log_security_event(
        self, request: Request, client_ip: str, event_type: str, details: str
    ) -> None:
        """Log security events for monitoring and analysis."""
        event_data = {
            "event_type": event_type,
            "client_ip": client_ip,
            "method": request.method,
            "path": request.url.path,
            "details": details,
            "user_agent": request.headers.get("user-agent", ""),
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.warning("security_event", **event_data)

        # Store in Redis for analysis
        event_key = f"security_event:{int(time.time())}"
        self.redis_client.setex(event_key, 86400, json.dumps(event_data))  # 24 hours


class PIIProtectionMiddleware(BaseHTTPMiddleware):
    """Middleware for PII protection in requests and responses."""

    def __init__(self, app: ASGIApp, pii_protector: PIIProtector, enabled_paths: set[str]):
        super().__init__(app)
        self.pii_protector = pii_protector
        self.enabled_paths = enabled_paths

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request for PII protection."""
        if not self._should_protect_path(request.url.path):
            return await call_next(request)

        try:
            # Protect request body if present
            if request.method in ["POST", "PUT", "PATCH"]:
                await self._protect_request_body(request)

            response = await call_next(request)

            # Protect response body if needed
            response = await self._protect_response_body(response, request)

            return response

        except ValueError as e:
            # PII protection blocked the request
            logger.warning("request_blocked_pii", path=request.url.path, error=str(e))
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "Request contains sensitive information that cannot be processed"
                },
            )

    def _should_protect_path(self, path: str) -> bool:
        """Check if path should be protected for PII."""
        return any(path.startswith(protected_path) for protected_path in self.enabled_paths)

    async def _protect_request_body(self, request: Request) -> None:
        """Protect PII in request body."""
        try:
            body = await request.body()
            if body:
                body_text = body.decode("utf-8")
                protected_text, detections = self.pii_protector.protect_text(body_text)

                if detections:
                    logger.info(
                        "pii_protected_request",
                        path=request.url.path,
                        detections_count=len(detections),
                        types=[d.pii_type.value for d in detections],
                    )

                # Replace request body with protected version
                request._body = protected_text.encode("utf-8")

        except Exception as e:
            logger.error("pii_protection_error", error=str(e), path=request.url.path)

    async def _protect_response_body(self, response: Response, request: Request) -> Response:
        """Protect PII in response body."""
        # This would require streaming response modification
        # For now, we'll log that response protection is needed
        logger.debug("response_pii_protection_needed", path=request.url.path)
        return response
