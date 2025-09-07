# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Centralized database mock configurations.

This module provides standardized database mocks and logging
configurations to ensure consistent testing behavior.
"""

from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List


def create_mock_database(
    log_success: bool = True,
    check_budget_result: Dict[str, Any] = None,
) -> Mock:
    """Create a standardized mock database."""
    db = Mock()

    # Mock log_request method
    if log_success:
        db.log_request = AsyncMock(return_value=True)
    else:
        db.log_request = AsyncMock(side_effect=Exception("Database error"))

    # Mock check_budget method
    if check_budget_result is None:
        check_budget_result = {
            "within_limits": True,
            "daily_usage": 5.0,
            "daily_limit": 100.0,
            "monthly_usage": 50.0,
            "monthly_limit": 1000.0,
        }

    db.check_budget = AsyncMock(return_value=check_budget_result)

    # Mock other database methods
    db.get_user_stats = AsyncMock(
        return_value={
            "total_requests": 100,
            "total_cost": 10.50,
            "avg_response_time": 250.0,
        }
    )

    db.set_budget = AsyncMock(return_value=True)
    db.get_budget = AsyncMock(return_value=check_budget_result)

    return db


def create_budget_exceeded_database(
    daily_limit: float = 10.0,
    daily_usage: float = 15.0,
) -> Mock:
    """Create a database mock that indicates budget exceeded."""
    return create_mock_database(
        check_budget_result={
            "within_limits": False,
            "daily_usage": daily_usage,
            "daily_limit": daily_limit,
            "monthly_usage": daily_usage * 30,
            "monthly_limit": daily_limit * 30,
        }
    )


def create_database_error_mock() -> Mock:
    """Create a database mock that raises errors."""
    db = Mock()
    db.log_request = AsyncMock(side_effect=Exception("Database connection failed"))
    db.check_budget = AsyncMock(side_effect=Exception("Budget check failed"))
    db.get_user_stats = AsyncMock(side_effect=Exception("Stats retrieval failed"))
    db.set_budget = AsyncMock(side_effect=Exception("Budget update failed"))
    db.get_budget = AsyncMock(side_effect=Exception("Budget retrieval failed"))

    return db


# Standard database configurations
MOCK_SUCCESSFUL_DATABASE = create_mock_database()
MOCK_BUDGET_EXCEEDED_DATABASE = create_budget_exceeded_database()
MOCK_DATABASE_ERROR = create_database_error_mock()
