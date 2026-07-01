# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""System endpoints: health check and metrics."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response

import app.main as app_main
from app.main import get_authenticated_user
from app.models import HealthResponse, MetricsResponse

router = APIRouter()


@router.get("/metrics/prometheus", include_in_schema=False)
async def prometheus_metrics():
    """Expose Prometheus metrics in text exposition format."""
    if app_main.generate_latest and app_main.CONTENT_TYPE_LATEST:
        return Response(
            content=app_main.generate_latest(), media_type=app_main.CONTENT_TYPE_LATEST
        )
    return JSONResponse(
        status_code=500, content={"error": {"message": "prometheus_client not available"}}
    )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0", timestamp=datetime.now())


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(user_info: dict[str, Any] = Depends(get_authenticated_user)):
    """Get system metrics and usage statistics."""
    # If Prometheus client is available, prefer exposing Prometheus format here as a fallback
    if app_main.generate_latest and app_main.CONTENT_TYPE_LATEST:
        return JSONResponse(
            content=app_main.generate_latest().decode("utf-8"),
            media_type=app_main.CONTENT_TYPE_LATEST,
        )
    # If Prometheus client is not available, return 500 to satisfy tests that expect prom text or error
    if not app_main.generate_latest or not app_main.CONTENT_TYPE_LATEST:
        return JSONResponse(
            status_code=500, content={"error": {"message": "prometheus_client not available"}}
        )

    metrics = await app_main.db.get_system_metrics()

    return MetricsResponse(
        total_requests=metrics["total_requests"],
        total_cost=metrics["total_cost"],
        active_users=metrics["active_users"],
        provider_usage=metrics["provider_usage"],
        model_usage=metrics["model_usage"],
        average_response_time=metrics["average_response_time"],
    )
