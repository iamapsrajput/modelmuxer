# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""Unit tests for database analytics helpers."""

import pytest

from app.database import Database
from tests.fixtures import temp_database


@pytest.fixture
async def analytics_db():
    with temp_database(suffix=".db") as db_path:
        db = Database(db_path)
        await db.init_database()
        await db.ensure_user_exists("analytics-user")

        messages = [{"role": "user", "content": "hello"}]
        await db.log_request(
            user_id="analytics-user",
            provider="anthropic",
            model="claude-3-5-haiku",
            messages=messages,
            input_tokens=10,
            output_tokens=20,
            cost=0.01,
            response_time_ms=300.0,
            routing_reason="rule match",
            intent_label="chat_lite",
            intent_method="heuristic",
            routing_rule="chat_fast",
        )
        yield db


@pytest.mark.asyncio
async def test_get_cost_analytics_includes_routing_fields(analytics_db):
    result = await analytics_db.get_cost_analytics("analytics-user", days=30)

    assert result["total_requests"] == 1
    assert result["total_cost"] == pytest.approx(0.01)
    assert result["recent_requests"][0]["intent_label"] == "chat_lite"
    assert result["recent_requests"][0]["routing_rule"] == "chat_fast"


@pytest.mark.asyncio
async def test_get_routing_analytics_aggregates(analytics_db):
    result = await analytics_db.get_routing_analytics("analytics-user", days=30)

    assert result["requests_by_intent"]["chat_lite"] == 1
    assert result["provider_mix"]["anthropic"] == 1
    assert result["local_vs_cloud"]["cloud"] == 1
    assert result["routing_rule_hits"]["chat_fast"] == 1
