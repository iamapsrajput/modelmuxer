# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""Analytics endpoints: cost analytics, budgets, and user statistics."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

import app.main as app_main
from app.main import get_authenticated_user
from app.models import ErrorResponse, UserStats

router = APIRouter()


@router.get("/user/stats", response_model=UserStats)
async def get_user_stats(user_info: dict[str, Any] = Depends(get_authenticated_user)):
    """Get user-specific usage statistics."""
    user_id = user_info["user_id"]
    stats = await app_main.db.get_user_stats(user_id)

    return UserStats(**stats)


@router.get("/v1/analytics/costs")
async def get_cost_analytics(
    user_info: dict[str, Any] = Depends(get_authenticated_user),
    days: int = 30,
    provider: str | None = None,
    model: str | None = None,
):
    """Get cost analytics for the user."""
    user_id = user_info["user_id"]
    analytics = await app_main.db.get_cost_analytics(
        user_id=user_id,
        days=days,
        provider=provider,
        model=model,
    )

    tracker = getattr(app_main.model_muxer, "advanced_cost_tracker", None)
    if tracker and hasattr(tracker, "get_cost_analytics"):
        try:
            tracker_analytics = await tracker.get_cost_analytics(
                user_id=user_id,
                days=days,
                provider=provider,
                model=model,
            )
            analytics = _merge_cost_analytics(analytics, tracker_analytics)
        except Exception as e:
            app_main.logger.warning(
                "cost_tracker_analytics_merge_failed", error=str(e), user_id=user_id
            )

    return analytics


@router.get("/v1/analytics/routing")
async def get_routing_analytics(
    user_info: dict[str, Any] = Depends(get_authenticated_user),
    days: int = 30,
):
    """Get routing decision aggregates for the user."""
    user_id = user_info["user_id"]
    return await app_main.db.get_routing_analytics(user_id=user_id, days=days)


def _merge_cost_analytics(primary: dict[str, Any], supplemental: dict[str, Any]) -> dict[str, Any]:
    """Merge DB analytics with advanced cost tracker totals when both exist."""
    if not supplemental or supplemental.get("total_requests", 0) == 0:
        return primary
    if primary.get("total_requests", 0) > 0:
        return primary

    merged = dict(primary)
    merged["total_cost"] = supplemental.get("total_cost", 0.0)
    merged["total_requests"] = supplemental.get("total_requests", 0)
    merged["cost_by_provider"] = supplemental.get("cost_by_provider", {})
    merged["cost_by_model"] = supplemental.get("cost_by_model", {})
    merged["daily_breakdown"] = supplemental.get("daily_breakdown", [])
    merged["weekly_breakdown"] = supplemental.get("weekly_breakdown", [])
    return merged


@router.get("/v1/analytics/budgets")
async def get_budget_status(
    user_info: dict[str, Any] = Depends(get_authenticated_user),
    budget_type: str | None = None,
):
    """Get budget status and alerts for the user."""
    user_id = user_info["user_id"]
    model_muxer = app_main.model_muxer

    try:
        # Get budget status from advanced cost tracker
        if hasattr(model_muxer, "advanced_cost_tracker") and model_muxer.advanced_cost_tracker:
            budget_statuses = await model_muxer.advanced_cost_tracker.get_budget_status(
                user_id, budget_type
            )

            # Convert to response format
            response_budgets = []
            for status in budget_statuses:
                alerts = [
                    {
                        "type": alert["type"],
                        "message": alert["message"],
                        "threshold": alert["threshold"],
                        "current_usage": alert["current_usage"],
                    }
                    for alert in status["alerts"]
                ]

                response_budgets.append(
                    {
                        "budget_type": status["budget_type"],
                        "budget_limit": status["budget_limit"],
                        "current_usage": status["current_usage"],
                        "usage_percentage": status["usage_percentage"],
                        "remaining_budget": status["remaining_budget"],
                        "provider": status["provider"],
                        "model": status["model"],
                        "alerts": alerts,
                        "period_start": status["period_start"],
                        "period_end": status["period_end"],
                    }
                )

            return {
                "message": "Budget status retrieved successfully",
                "budgets": response_budgets,
                "total_budgets": len(response_budgets),
            }
        else:
            return {
                "message": "Budget tracking not initialized",
                "budgets": [],
                "total_budgets": 0,
            }

    except Exception as e:
        app_main.logger.error("budget_status_error", error=str(e), user_id=user_id)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.create(
                message="Failed to retrieve budget status",
                error_type="internal_error",
                code="budget_retrieval_failed",
            ).dict(),
        ) from e


@router.post("/v1/analytics/budgets")
async def set_budget(
    request: dict[str, Any],
    user_info: dict[str, Any] = Depends(get_authenticated_user),
):
    """Set budget limits and alert thresholds for the user."""
    user_id = user_info["user_id"]
    model_muxer = app_main.model_muxer

    try:
        # Validate request
        budget_type = request.get("budget_type")
        budget_limit = request.get("budget_limit")

        if not budget_type or budget_limit is None:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse.create(
                    message="budget_type and budget_limit are required",
                    error_type="invalid_request",
                    code="missing_parameters",
                ).dict(),
            )

        if budget_type not in ["daily", "weekly", "monthly", "yearly"]:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse.create(
                    message="budget_type must be one of: daily, weekly, monthly, yearly",
                    error_type="invalid_request",
                    code="invalid_budget_type",
                ).dict(),
            )

        if not isinstance(budget_limit, int | float) or budget_limit <= 0:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse.create(
                    message="budget_limit must be a positive number",
                    error_type="invalid_request",
                    code="invalid_budget_limit",
                ).dict(),
            )

        provider = request.get("provider")
        model = request.get("model")
        alert_thresholds = request.get("alert_thresholds", [50.0, 80.0, 95.0])

        # Set budget using advanced cost tracker
        if hasattr(model_muxer, "advanced_cost_tracker") and model_muxer.advanced_cost_tracker:
            await model_muxer.advanced_cost_tracker.set_budget(
                user_id=user_id,
                budget_type=budget_type,
                budget_limit=float(budget_limit),
                provider=provider,
                model=model,
                alert_thresholds=alert_thresholds,
            )

            return {
                "message": "Budget set successfully",
                "budget": {
                    "budget_type": budget_type,
                    "budget_limit": budget_limit,
                    "provider": provider,
                    "model": model,
                    "alert_thresholds": alert_thresholds,
                },
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse.create(
                    message="Budget tracking not initialized",
                    error_type="internal_error",
                    code="budget_not_initialized",
                ).dict(),
            )

    except HTTPException:
        raise
    except Exception as e:
        app_main.logger.error("set_budget_error", error=str(e), user_id=user_id)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.create(
                message="Failed to set budget",
                error_type="internal_error",
                code="budget_set_failed",
            ).dict(),
        ) from e
