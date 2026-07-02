# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""Tests for routing-aware analytics dashboard endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.database import Database
from tests.fixtures import temp_database


@pytest.fixture
def test_client():
    from app.main import app

    return TestClient(app)


@pytest.fixture
def mock_auth():
    with patch("app.main.auth.authenticate_request") as mock:
        mock.return_value = {"user_id": "dashboard-user", "scopes": ["api_access"]}
        yield mock


@pytest.fixture
async def analytics_db():
    with temp_database(suffix=".db") as db_path:
        db = Database(db_path)
        await db.init_database()
        await db.ensure_user_exists("dashboard-user")

        messages = [{"role": "user", "content": "Write a Python function"}]
        await db.log_request(
            user_id="dashboard-user",
            provider="openai",
            model="gpt-4o-mini",
            messages=messages,
            input_tokens=20,
            output_tokens=40,
            cost=0.02,
            response_time_ms=450.0,
            routing_reason="Intent code_gen matched rule code_cheap",
            intent_label="code_gen",
            intent_method="cheap_llm",
            routing_rule="code_cheap",
        )
        await db.log_request(
            user_id="dashboard-user",
            provider="ollama",
            model="llama3.2",
            messages=messages,
            input_tokens=20,
            output_tokens=40,
            cost=0.0,
            response_time_ms=900.0,
            routing_reason="Local model preferred via PREFER_LOCAL",
            intent_label="chat_lite",
            intent_method="heuristic",
            routing_rule=None,
        )
        yield db


class TestAnalyticsDashboardEndpoints:
    def test_get_cost_analytics_returns_real_data(self, test_client, mock_auth, analytics_db):
        with patch("app.main.db", analytics_db):
            response = test_client.get(
                "/v1/analytics/costs?days=30",
                headers={"Authorization": "Bearer test-key"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "dashboard-user"
        assert data["period_days"] == 30
        assert data["total_requests"] == 2
        assert data["total_cost"] == pytest.approx(0.02)
        assert data["cost_by_provider"]["openai"] == pytest.approx(0.02)
        assert data["cost_by_provider"]["ollama"] == pytest.approx(0.0)
        assert len(data["daily_breakdown"]) >= 1
        assert len(data["recent_requests"]) == 2
        assert data["recent_requests"][0]["intent_label"] in {"code_gen", "chat_lite"}

    def test_get_cost_analytics_supports_filters(self, test_client, mock_auth, analytics_db):
        with patch("app.main.db", analytics_db):
            response = test_client.get(
                "/v1/analytics/costs?days=7&provider=ollama",
                headers={"Authorization": "Bearer test-key"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 1
        assert data["cost_by_provider"] == {"ollama": 0.0}

    def test_get_routing_analytics_returns_aggregates(self, test_client, mock_auth, analytics_db):
        with patch("app.main.db", analytics_db):
            response = test_client.get(
                "/v1/analytics/routing?days=30",
                headers={"Authorization": "Bearer test-key"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 2
        assert data["requests_by_intent"]["code_gen"] == 1
        assert data["requests_by_intent"]["chat_lite"] == 1
        assert data["local_vs_cloud"]["local"] == 1
        assert data["local_vs_cloud"]["cloud"] == 1
        assert data["routing_rule_hits"]["code_cheap"] == 1
        assert data["routing_rule_hits"]["_none"] == 1
        assert data["intent_method_mix"]["cheap_llm"] == 1
        assert data["intent_method_mix"]["heuristic"] == 1
        assert len(data["top_models"]) == 2
        assert len(data["top_providers"]) == 2

    def test_get_cost_analytics_empty_but_valid(self, test_client, mock_auth):
        empty_db = AsyncMock()
        empty_db.get_cost_analytics = AsyncMock(
            return_value={
                "user_id": "dashboard-user",
                "period_days": 30,
                "total_cost": 0.0,
                "total_requests": 0,
                "cost_by_provider": {},
                "cost_by_model": {},
                "daily_breakdown": [],
                "weekly_breakdown": [],
                "recent_requests": [],
            }
        )

        with patch("app.main.db", empty_db):
            response = test_client.get(
                "/v1/analytics/costs",
                headers={"Authorization": "Bearer test-key"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 0
        assert data["cost_by_provider"] == {}

    def test_dashboard_static_files_served(self, test_client):
        response = test_client.get("/dashboard/", follow_redirects=False)
        assert response.status_code == 200
        assert "ModelMuxer Dashboard" in response.text

        assets = ["styles.css", "app.js"]
        for asset in assets:
            asset_response = test_client.get(f"/dashboard/{asset}")
            assert asset_response.status_code == 200

    def test_dashboard_root_redirects(self, test_client):
        response = test_client.get("/dashboard", follow_redirects=False)
        assert response.status_code in {307, 308}
        assert response.headers["location"] == "/dashboard/"
