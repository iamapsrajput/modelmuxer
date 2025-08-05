# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Tests for budget management functionality.

This module tests the budget management API endpoints and cost tracker
budget functionality including setting budgets, getting status, and alerts.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.cost_tracker import AdvancedCostTracker
from app.main import app


class TestBudgetManagement:
    """Test budget management API endpoints."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.test_user_id = "test_user_123"
        self.test_headers = {"Authorization": "Bearer test_token"}

    @patch("app.main.model_muxer")
    def test_get_budget_status_basic_mode(self, mock_model_muxer):
        """Test getting budget status in basic mode."""
        # Mock basic mode
        mock_model_muxer.enhanced_mode = False

        # Use dependency override for authentication
        def mock_auth():
            return {"user_id": self.test_user_id}

        from app.main import app, get_authenticated_user

        app.dependency_overrides[get_authenticated_user] = mock_auth

        try:
            response = self.client.get("/v1/analytics/budgets")
            assert response.status_code == 501
            assert "enhanced mode" in response.json()["error"]["message"].lower()
        finally:
            app.dependency_overrides.clear()

    @patch("app.main.model_muxer")
    def test_get_budget_status_enhanced_mode(self, mock_model_muxer):
        """Test getting budget status in enhanced mode."""
        # Mock enhanced mode with advanced cost tracker
        mock_model_muxer.enhanced_mode = True
        mock_advanced_tracker = AsyncMock()
        mock_model_muxer.advanced_cost_tracker = mock_advanced_tracker

        # Mock budget status response
        mock_budget_status = [
            {
                "budget_type": "daily",
                "budget_limit": 10.0,
                "current_usage": 2.5,
                "usage_percentage": 25.0,
                "remaining_budget": 7.5,
                "provider": None,
                "model": None,
                "alerts": [
                    {
                        "type": "warning",
                        "message": "Budget usage at 25.0% (threshold: 50%)",
                        "threshold": 50.0,
                        "current_usage": 25.0,
                    }
                ],
                "period_start": "2024-01-15",
                "period_end": "2024-01-15",
            }
        ]
        mock_advanced_tracker.get_budget_status.return_value = mock_budget_status

        # Use dependency override for authentication
        def mock_auth():
            return {"user_id": self.test_user_id}

        from app.main import app, get_authenticated_user

        app.dependency_overrides[get_authenticated_user] = mock_auth

        try:
            response = self.client.get("/v1/analytics/budgets")
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Budget status retrieved successfully"
            assert len(data["budgets"]) == 1
            assert data["total_budgets"] == 1
            assert data["budgets"][0]["budget_type"] == "daily"
            assert data["budgets"][0]["budget_limit"] == 10.0
        finally:
            app.dependency_overrides.clear()

    @patch("app.main.get_authenticated_user")
    @patch("app.main.model_muxer")
    def test_set_budget_basic_mode(self, mock_model_muxer, mock_auth):
        """Test setting budget in basic mode."""
        # Mock authentication
        mock_auth.return_value = {"user_id": self.test_user_id}

        # Mock basic mode
        mock_model_muxer.enhanced_mode = False

        budget_request = {"budget_type": "daily", "budget_limit": 15.0, "alert_thresholds": [50.0, 80.0, 95.0]}

        response = self.client.post("/v1/analytics/budgets", headers=self.test_headers, json=budget_request)

        assert response.status_code == 501
        assert "enhanced mode" in response.json()["error"]["message"].lower()

    @patch("app.main.get_authenticated_user")
    @patch("app.main.model_muxer")
    async def test_set_budget_enhanced_mode(self, mock_model_muxer, mock_auth):
        """Test setting budget in enhanced mode."""
        # Mock authentication
        mock_auth.return_value = {"user_id": self.test_user_id}

        # Mock enhanced mode with advanced cost tracker
        mock_model_muxer.enhanced_mode = True
        mock_advanced_tracker = AsyncMock()
        mock_model_muxer.advanced_cost_tracker = mock_advanced_tracker

        budget_request = {
            "budget_type": "daily",
            "budget_limit": 15.0,
            "provider": "openai",
            "model": "gpt-4o",
            "alert_thresholds": [50.0, 80.0, 95.0],
        }

        response = self.client.post("/v1/analytics/budgets", headers=self.test_headers, json=budget_request)

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Budget set successfully"
        assert data["budget"]["budget_type"] == "daily"
        assert data["budget"]["budget_limit"] == 15.0
        assert data["budget"]["provider"] == "openai"
        assert data["budget"]["model"] == "gpt-4o"

        # Verify the advanced tracker was called
        mock_advanced_tracker.set_budget.assert_called_once_with(
            user_id=self.test_user_id,
            budget_type="daily",
            budget_limit=15.0,
            provider="openai",
            model="gpt-4o",
            alert_thresholds=[50.0, 80.0, 95.0],
        )

    @patch("app.main.get_authenticated_user")
    @patch("app.main.model_muxer")
    def test_set_budget_invalid_request(self, mock_model_muxer, mock_auth):
        """Test setting budget with invalid request data."""
        # Mock authentication
        mock_auth.return_value = {"user_id": self.test_user_id}

        # Mock enhanced mode
        mock_model_muxer.enhanced_mode = True
        mock_model_muxer.advanced_cost_tracker = AsyncMock()

        # Test missing budget_type
        response = self.client.post("/v1/analytics/budgets", headers=self.test_headers, json={"budget_limit": 15.0})
        assert response.status_code == 400

        # Test missing budget_limit
        response = self.client.post("/v1/analytics/budgets", headers=self.test_headers, json={"budget_type": "daily"})
        assert response.status_code == 400

        # Test invalid budget_type
        response = self.client.post(
            "/v1/analytics/budgets", headers=self.test_headers, json={"budget_type": "invalid", "budget_limit": 15.0}
        )
        assert response.status_code == 400

        # Test invalid budget_limit
        response = self.client.post(
            "/v1/analytics/budgets", headers=self.test_headers, json={"budget_type": "daily", "budget_limit": -5.0}
        )
        assert response.status_code == 400


class TestAdvancedCostTrackerBudgets:
    """Test AdvancedCostTracker budget functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.test_user_id = "test_user_123"

    @patch("app.cost_tracker.ENHANCED_FEATURES_AVAILABLE", True)
    @patch("app.cost_tracker.sqlite3")
    async def test_set_budget(self, mock_sqlite):
        """Test setting a budget."""
        # Mock database
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        tracker = AdvancedCostTracker()

        await tracker.set_budget(
            user_id=self.test_user_id,
            budget_type="daily",
            budget_limit=10.0,
            provider="openai",
            model="gpt-4o",
            alert_thresholds=[50.0, 80.0, 95.0],
        )

        # Verify database call
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        assert "INSERT OR REPLACE INTO user_budgets" in call_args[0]
        assert call_args[1][0] == self.test_user_id
        assert call_args[1][1] == "daily"
        assert call_args[1][2] == 10.0

    @patch("app.cost_tracker.ENHANCED_FEATURES_AVAILABLE", False)
    async def test_set_budget_basic_mode(self):
        """Test setting budget in basic mode (should do nothing)."""
        tracker = AdvancedCostTracker()

        # Should not raise an exception, just return early
        await tracker.set_budget(user_id=self.test_user_id, budget_type="daily", budget_limit=10.0)

        # No assertions needed - just verify it doesn't crash

    @patch("app.cost_tracker.ENHANCED_FEATURES_AVAILABLE", True)
    @patch("app.cost_tracker.sqlite3")
    async def test_get_budget_status(self, mock_sqlite):
        """Test getting budget status."""
        # Mock database
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock budget data
        mock_cursor.fetchall.return_value = [("daily", 10.0, None, None, json.dumps([50.0, 80.0, 95.0]))]

        tracker = AdvancedCostTracker()

        # Mock the _get_current_usage method
        with patch.object(tracker, "_get_current_usage", return_value=2.5):
            budget_statuses = await tracker.get_budget_status(self.test_user_id)

        assert len(budget_statuses) == 1
        status = budget_statuses[0]
        assert status["budget_type"] == "daily"
        assert status["budget_limit"] == 10.0
        assert status["current_usage"] == 2.5
        assert status["usage_percentage"] == 25.0
        assert status["remaining_budget"] == 7.5
