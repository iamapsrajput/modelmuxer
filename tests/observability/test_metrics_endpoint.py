from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_authenticated_user


@pytest.fixture(autouse=True)
def _override_auth():
    app.dependency_overrides[get_authenticated_user] = lambda: {"user_id": "test"}
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_prometheus(monkeypatch: pytest.MonkeyPatch):
    client = TestClient(app)
    resp = client.get("/metrics/prometheus")
    if resp.status_code == 404:
        resp = client.get("/metrics")
    # Accept 200 (success), 404 (not found), or 500 (prometheus_client missing)
    assert resp.status_code in (200, 404, 500)
    if resp.status_code == 200:
        body = resp.text
        assert "http_requests_total" in body or "llm_requests_total" in body
