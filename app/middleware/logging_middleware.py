# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Advanced logging middleware for ModelMuxer.

This module provides comprehensive request/response logging with
structured logging, performance metrics, and audit trails.
"""

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from ..core.utils import generate_request_id, sanitize_input

logger = structlog.get_logger(__name__)


class LoggingMiddleware:
    """
    Advanced logging middleware with structured logging and metrics.

    Provides comprehensive request/response logging, performance tracking,
    error monitoring, and audit trails for compliance.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Logging configuration
        self.log_requests = self.config.get("log_requests", True)
        self.log_responses = self.config.get("log_responses", True)
        self.log_request_body = self.config.get("log_request_body", False)
        self.log_response_body = self.config.get("log_response_body", False)
        self.log_headers = self.config.get("log_headers", False)

        # Performance tracking
        self.track_performance = self.config.get("track_performance", True)
        self.slow_request_threshold = self.config.get("slow_request_threshold", 5.0)  # seconds

        # Security and privacy
        self.sanitize_sensitive_data = self.config.get("sanitize_sensitive_data", True)
        self.sensitive_headers = set(
            self.config.get(
                "sensitive_headers", ["authorization", "x-api-key", "cookie", "x-auth-token"]
            )
        )
        self.sensitive_fields = set(
            self.config.get(
                "sensitive_fields", ["password", "token", "api_key", "secret", "private_key"]
            )
        )

        # Audit logging
        self.enable_audit_log = self.config.get("enable_audit_log", True)
        self.audit_events = set(
            self.config.get(
                "audit_events",
                ["authentication", "authorization", "rate_limit", "error", "completion"],
            )
        )

        # Request tracking
        self.request_metrics: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}

        logger.info(
            "logging_middleware_initialized",
            log_requests=self.log_requests,
            log_responses=self.log_responses,
            track_performance=self.track_performance,
            audit_logging=self.enable_audit_log,
        )

    async def log_request(
        self, request: Request, user_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log incoming request with metadata.

        Args:
            request: FastAPI request object
            user_info: Authenticated user information

        Returns:
            Request context for response logging
        """
        request_id = generate_request_id()
        start_time = time.time()

        # Extract request information
        request_data = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent", ""),
            "content_type": request.headers.get("content-type", ""),
            "content_length": request.headers.get("content-length", 0),
        }

        # Add user information if available
        if user_info:
            request_data["user_id"] = user_info.get("user_id")
            request_data["user_role"] = user_info.get("role")
            request_data["auth_method"] = user_info.get("auth_method")

        # Add headers if configured
        if self.log_headers:
            request_data["headers"] = self._sanitize_headers(dict(request.headers))

        # Add request body if configured
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    if request_data["content_type"].startswith("application/json"):
                        try:
                            body_json = json.loads(body.decode())
                            request_data["body"] = self._sanitize_data(body_json)
                        except json.JSONDecodeError:
                            request_data["body"] = "<invalid_json>"
                    else:
                        request_data["body"] = f"<{len(body)} bytes>"
            except Exception as e:
                request_data["body_error"] = str(e)

        # Log the request
        if self.log_requests:
            logger.info("request_received", **request_data)

        # Audit log for authentication events
        if self.enable_audit_log and "authentication" in self.audit_events:
            if user_info:
                self._audit_log(
                    "authentication",
                    {
                        "request_id": request_id,
                        "user_id": user_info.get("user_id"),
                        "auth_method": user_info.get("auth_method"),
                        "client_ip": request_data["client_ip"],
                        "endpoint": request_data["path"],
                    },
                )

        # Return context for response logging
        return {
            "request_id": request_id,
            "start_time": start_time,
            "user_info": user_info,
            "endpoint": request_data["path"],
            "method": request_data["method"],
            "client_ip": request_data["client_ip"],
        }

    async def log_response(
        self, response: Response, request_context: Dict[str, Any], error: Optional[Exception] = None
    ) -> None:
        """
        Log response with performance metrics.

        Args:
            response: FastAPI response object
            request_context: Context from request logging
            error: Exception if request failed
        """
        end_time = time.time()
        duration = end_time - request_context["start_time"]

        # Extract response information
        response_data = {
            "request_id": request_context["request_id"],
            "timestamp": datetime.now().isoformat(),
            "status_code": response.status_code if response else 500,
            "duration_ms": round(duration * 1000, 2),
            "endpoint": request_context["endpoint"],
            "method": request_context["method"],
            "client_ip": request_context["client_ip"],
        }

        # Add user information
        if request_context.get("user_info"):
            user_info = request_context["user_info"]
            response_data["user_id"] = user_info.get("user_id")

        # Add error information
        if error:
            response_data["error"] = {"type": type(error).__name__, "message": str(error)}

            # Track error counts
            error_key = f"{request_context['endpoint']}:{type(error).__name__}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        # Add response headers if configured
        if self.log_headers and response:
            response_data["response_headers"] = dict(response.headers)

        # Add response body if configured (be careful with streaming responses)
        if self.log_response_body and response and not isinstance(response, StreamingResponse):
            try:
                if hasattr(response, "body") and response.body:
                    if response.headers.get("content-type", "").startswith("application/json"):
                        try:
                            body_json = json.loads(response.body.decode())
                            response_data["response_body"] = self._sanitize_data(body_json)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            response_data["response_body"] = "<non_json_response>"
                    else:
                        response_data["response_body"] = f"<{len(response.body)} bytes>"
            except Exception as e:
                response_data["response_body_error"] = str(e)

        # Performance tracking
        if self.track_performance:
            endpoint = request_context["endpoint"]
            if endpoint not in self.request_metrics:
                self.request_metrics[endpoint] = []

            self.request_metrics[endpoint].append(duration)

            # Keep only recent metrics (last 1000 requests per endpoint)
            if len(self.request_metrics[endpoint]) > 1000:
                self.request_metrics[endpoint] = self.request_metrics[endpoint][-1000:]

            # Log slow requests
            if duration > self.slow_request_threshold:
                logger.warning(
                    "slow_request_detected",
                    request_id=request_context["request_id"],
                    endpoint=endpoint,
                    duration_ms=response_data["duration_ms"],
                    threshold_ms=self.slow_request_threshold * 1000,
                )

        # Log the response
        if self.log_responses:
            if error:
                logger.error("request_failed", **response_data)
            elif response and response.status_code >= 400:
                logger.warning("request_error", **response_data)
            else:
                logger.info("request_completed", **response_data)

        # Audit logging
        if self.enable_audit_log:
            if error and "error" in self.audit_events:
                self._audit_log(
                    "error",
                    {
                        "request_id": request_context["request_id"],
                        "user_id": request_context.get("user_info", {}).get("user_id"),
                        "endpoint": request_context["endpoint"],
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "client_ip": request_context["client_ip"],
                    },
                )
            elif response and response.status_code == 200 and "completion" in self.audit_events:
                self._audit_log(
                    "completion",
                    {
                        "request_id": request_context["request_id"],
                        "user_id": request_context.get("user_info", {}).get("user_id"),
                        "endpoint": request_context["endpoint"],
                        "duration_ms": response_data["duration_ms"],
                        "client_ip": request_context["client_ip"],
                    },
                )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fallback to direct client IP
        if hasattr(request, "client") and request.client:
            return request.client.host

        return "unknown"

    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize sensitive headers."""
        if not self.sanitize_sensitive_data:
            return headers

        sanitized = {}
        for key, value in headers.items():
            if key.lower() in self.sensitive_headers:
                sanitized[key] = "<redacted>"
            else:
                sanitized[key] = value

        return sanitized

    def _sanitize_data(self, data: Any) -> Any:
        """Recursively sanitize sensitive data."""
        if not self.sanitize_sensitive_data:
            return data

        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if key.lower() in self.sensitive_fields:
                    sanitized[key] = "<redacted>"
                else:
                    sanitized[key] = self._sanitize_data(value)
            return sanitized
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        else:
            return data

    def _audit_log(self, event_type: str, data: Dict[str, Any]) -> None:
        """Write audit log entry."""
        audit_entry = {"audit_event": event_type, "timestamp": datetime.now().isoformat(), **data}

        # Use a separate audit logger if configured
        audit_logger = structlog.get_logger("audit")
        audit_logger.info("audit_event", **audit_entry)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all endpoints."""
        metrics = {}

        for endpoint, durations in self.request_metrics.items():
            if durations:
                metrics[endpoint] = {
                    "request_count": len(durations),
                    "avg_duration_ms": round(sum(durations) * 1000 / len(durations), 2),
                    "min_duration_ms": round(min(durations) * 1000, 2),
                    "max_duration_ms": round(max(durations) * 1000, 2),
                    "slow_requests": sum(1 for d in durations if d > self.slow_request_threshold),
                }

        return {
            "endpoint_metrics": metrics,
            "error_counts": self.error_counts.copy(),
            "total_requests": sum(len(durations) for durations in self.request_metrics.values()),
            "total_errors": sum(self.error_counts.values()),
        }

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.request_metrics.clear()
        self.error_counts.clear()
        logger.info("performance_metrics_reset")
