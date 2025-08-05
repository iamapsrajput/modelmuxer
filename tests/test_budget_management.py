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
        self.test_user_id = "test_user_123"
        self.test_headers = {"Authorization": "Bearer test_token"}

        # Mock authentication for all tests
        def mock_auth():
            return {"user_id": self.test_user_id}

        from app.main import get_authenticated_user

        app.dependency_overrides[get_authenticated_user] = mock_auth
        self.client = TestClient(app)

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        app.dependency_overrides.clear()

    @patch("app.main.model_muxer")
    def test_get_budget_status_basic_mode(self, mock_model_muxer):
        """Test getting budget status in basic mode."""
        # Mock basic mode
        mock_model_muxer.enhanced_mode = False

        response = self.client.get("/v1/analytics/budgets")
        assert response.status_code == 501
        assert "enhanced mode" in response.json()["error"]["message"].lower()

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

        response = self.client.get("/v1/analytics/budgets")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Budget status retrieved successfully"
        assert len(data["budgets"]) == 1
        assert data["total_budgets"] == 1
        assert data["budgets"][0]["budget_type"] == "daily"
        assert data["budgets"][0]["budget_limit"] == 10.0

    @patch("app.main.model_muxer")
    def test_set_budget_basic_mode(self, mock_model_muxer):
        """Test setting budget in basic mode."""
        # Mock basic mode
        mock_model_muxer.enhanced_mode = False

        budget_request = {
            "budget_type": "daily",
            "budget_limit": 15.0,
            "alert_thresholds": [50.0, 80.0, 95.0],
        }

        response = self.client.post(
            "/v1/analytics/budgets", headers=self.test_headers, json=budget_request
        )

        assert response.status_code == 501
        assert "enhanced mode" in response.json()["error"]["message"].lower()

    @patch("app.main.model_muxer")
    def test_set_budget_enhanced_mode(self, mock_model_muxer):
        """Test setting budget in enhanced mode."""
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

        response = self.client.post(
            "/v1/analytics/budgets", headers=self.test_headers, json=budget_request
        )

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

    @patch("app.main.model_muxer")
    def test_set_budget_invalid_request(self, mock_model_muxer):
        """Test setting budget with invalid request data."""
        # Mock enhanced mode
        mock_model_muxer.enhanced_mode = True
        mock_model_muxer.advanced_cost_tracker = AsyncMock()

        # Test missing budget_type
        response = self.client.post(
            "/v1/analytics/budgets", headers=self.test_headers, json={"budget_limit": 15.0}
        )
        assert response.status_code == 400

        # Test missing budget_limit
        response = self.client.post(
            "/v1/analytics/budgets", headers=self.test_headers, json={"budget_type": "daily"}
        )
        assert response.status_code == 400

        # Test invalid budget_type
        response = self.client.post(
            "/v1/analytics/budgets",
            headers=self.test_headers,
            json={"budget_type": "invalid", "budget_limit": 15.0},
        )
        assert response.status_code == 400

        # Test invalid budget_limit
        response = self.client.post(
            "/v1/analytics/budgets",
            headers=self.test_headers,
            json={"budget_type": "daily", "budget_limit": -5.0},
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

        # Verify database call was made - check that execute was called with budget data
        assert mock_cursor.execute.call_count >= 1

        # Find the INSERT OR REPLACE call among all calls
        insert_call_found = False
        for call in mock_cursor.execute.call_args_list:
            call_args = call[0]
            if len(call_args) >= 2 and "INSERT OR REPLACE INTO user_budgets" in call_args[0]:
                # Verify the parameters
                params = call_args[1]
                assert params[0] == self.test_user_id  # user_id
                assert params[1] == "daily"  # budget_type
                assert params[2] == 10.0  # budget_limit
                assert params[3] == "openai"  # provider
                assert params[4] == "gpt-4o"  # model
                assert params[5] == json.dumps([50.0, 80.0, 95.0])  # alert_thresholds
                insert_call_found = True
                break

        assert insert_call_found, "INSERT OR REPLACE INTO user_budgets call not found"

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
        mock_cursor.fetchall.return_value = [
            ("daily", 10.0, None, None, json.dumps([50.0, 80.0, 95.0]))
        ]

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
