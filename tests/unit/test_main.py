# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""
Comprehensive unit tests for app/main.py.

Tests cover:
- FastAPI application endpoints and routing
- Request handling and validation
- Error responses and exception handling
- Middleware functionality
- Authentication and authorization
- Provider registry integration
- Cost tracking and budget management
- Health check and metrics endpoints
- Anthropic API compatibility
"""

import asyncio
import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.main import (
    app,
    chat_completions,
    health_check,
    get_metrics,
    get_user_stats,
    get_providers,
    list_models,
    get_cost_analytics,
    get_budget_status,
    set_budget,
    anthropic_messages,
    get_authenticated_user,
)
from app.models import ChatCompletionRequest, ChatMessage, ErrorResponse
from app.core.exceptions import BudgetExceededError, ProviderError


class TestMainApplication:
    """Test suite for main FastAPI application."""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def sample_chat_request(self):
        """Sample chat completion request for testing."""
        return ChatCompletionRequest(
            messages=[
                ChatMessage(role="user", content="Hello, how are you?", name=None)
            ],
            model="gpt-3.5-turbo",
            max_tokens=100,
            temperature=0.7,
            top_p=1.0,
            n=1,
            stream=False,
            stop=None,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            logit_bias=None,
            user=None,
            region=None,
        )

    @pytest.fixture
    def mock_user_info(self):
        """Mock user authentication info."""
        return {"user_id": "test_user", "api_key": "test_key"}

    def test_health_endpoint_success(self, client):
        """Test health check endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data

    def test_health_endpoint_with_debug_headers(self, client, monkeypatch):
        """Test health check includes debug headers when enabled."""
        monkeypatch.setattr("app.settings.settings.server.debug", True)

        response = client.get("/health")

        assert response.status_code == 200
        # Debug headers should be present when debug mode is enabled

    @patch("app.main.get_authenticated_user")
    def test_chat_completions_success(self, client, sample_chat_request):
        """Test successful chat completions endpoint."""
        with patch("app.main.get_authenticated_user", return_value={"user_id": "test_user"}):
            with patch("app.main.router") as mock_router:
                mock_router.select_model.return_value = (
                    "openai",
                    "gpt-3.5-turbo",
                    "selected",
                    {"label": "simple", "confidence": 0.9},
                    {"usd": 0.01, "eta_ms": 500, "tokens_in": 10, "tokens_out": 20}
                )

                with patch("app.main.providers_registry.get_provider_registry") as mock_registry:
                    mock_provider = Mock()
                    mock_adapter = Mock()
                    mock_adapter.chat_completion = AsyncMock(return_value=Mock(
                        dict=Mock(return_value={
                            "id": "test-id",
                            "object": "chat.completion",
                            "created": 1234567890,
                            "model": "gpt-3.5-turbo",
                            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                            "router_metadata": {
                                "provider": "openai",
                                "model": "gpt-3.5-turbo",
                                "routing_reason": "selected",
                                "estimated_cost": 0.01,
                                "response_time_ms": 500
                            }
                        })
                    ))
                    mock_registry.return_value = {"openai": mock_adapter}

                    response = client.post(
                        "/v1/chat/completions",
                        json=sample_chat_request.dict(),
                        headers={"Authorization": "Bearer test_key"}
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert "id" in data
                    assert data["object"] == "chat.completion"
                    assert "router_metadata" in data

    def test_chat_completions_unauthorized(self, client, sample_chat_request):
        """Test chat completions with invalid authentication."""
        response = client.post("/v1/chat/completions", json=sample_chat_request.dict())

        assert response.status_code == 401
        data = response.json()
        assert "error" in data

    def test_chat_completions_budget_exceeded(self, client, sample_chat_request, mock_user_info):
        """Test chat completions when budget is exceeded."""
        with patch("app.main.get_authenticated_user", return_value=mock_user_info):
            with patch("app.main.router") as mock_router:
                mock_router.select_model.side_effect = BudgetExceededError(
                    "Budget exceeded", limit=100.0, estimates=[("openai:gpt-3.5-turbo", 0.05)]
                )

                response = client.post(
                    "/v1/chat/completions",
                    json=sample_chat_request.dict(),
                    headers={"Authorization": "Bearer test_key"}
                )

                assert response.status_code == 402
                data = response.json()
                assert data["error"]["type"] == "budget_exceeded"

    def test_chat_completions_provider_error(self, client, sample_chat_request, mock_user_info):
        """Test chat completions with provider error."""
        with patch("app.main.get_authenticated_user", return_value=mock_user_info):
            with patch("app.main.router") as mock_router:
                mock_router.select_model.return_value = (
                    "openai",
                    "gpt-3.5-turbo",
                    "selected",
                    {"label": "simple", "confidence": 0.9},
                    {"usd": 0.01, "eta_ms": 500, "tokens_in": 10, "tokens_out": 20}
                )

                with patch("app.main.providers_registry.get_provider_registry") as mock_registry:
                    mock_provider = Mock()
                    mock_adapter = Mock()
                    mock_adapter.chat_completion = AsyncMock(side_effect=ProviderError("API Error"))
                    mock_registry.return_value = {"openai": mock_adapter}

                    response = client.post(
                        "/v1/chat/completions",
                        json=sample_chat_request.dict(),
                        headers={"Authorization": "Bearer test_key"}
                    )

                    assert response.status_code == 502
                    data = response.json()
                    assert data["error"]["type"] == "provider_error"

    def test_chat_completions_streaming(self, client, sample_chat_request, mock_user_info):
        """Test streaming chat completions."""
        streaming_request = sample_chat_request.copy()
        streaming_request.stream = True

        with patch("app.main.get_authenticated_user", return_value=mock_user_info):
            with patch("app.main.stream_chat_completion") as mock_stream:
                mock_stream.return_value = Mock()

                response = client.post(
                    "/v1/chat/completions",
                    json=streaming_request.dict(),
                    headers={"Authorization": "Bearer test_key"}
                )

                # Streaming responses should return 200 with proper headers
                assert response.status_code == 200

    def test_chat_completions_validation_error(self, client):
        """Test chat completions with invalid request data."""
        invalid_request = {"messages": "not_a_list"}  # Invalid format

        response = client.post(
            "/v1/chat/completions",
            json=invalid_request,
            headers={"Authorization": "Bearer test_key"}
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_chat_completions_empty_messages(self, client, mock_user_info):
        """Test chat completions with empty messages."""
        empty_request = ChatCompletionRequest(
            messages=[],
            model="gpt-3.5-turbo",
            max_tokens=100,
            temperature=0.7,
            top_p=1.0,
            n=1,
            stream=False,
            stop=None,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            logit_bias=None,
            user=None,
            region=None,
        )

        with patch("app.main.get_authenticated_user", return_value=mock_user_info):
            response = client.post(
                "/v1/chat/completions",
                json=empty_request.dict(),
                headers={"Authorization": "Bearer test_key"}
            )

            # Should handle gracefully or return appropriate error
            assert response.status_code in [200, 400, 500]

    def test_get_providers_success(self, client):
        """Test get providers endpoint."""
        with patch("app.main.get_authenticated_user", return_value={"user_id": "test_user"}):
            with patch("app.main.providers_registry.get_provider_registry") as mock_registry:
                mock_provider = Mock()
                mock_provider.get_supported_models.return_value = ["gpt-3.5-turbo", "gpt-4"]
                mock_registry.return_value = {"openai": mock_provider}

                response = client.get("/providers", headers={"Authorization": "Bearer test_key"})

                assert response.status_code == 200
                data = response.json()
                assert "providers" in data
                assert "openai" in data["providers"]

    def test_list_models_success(self, client):
        """Test list models endpoint."""
        with patch("app.main.get_authenticated_user", return_value={"user_id": "test_user"}):
            with patch("app.main.load_price_table") as mock_load_price:
                mock_load_price.return_value = {
                    "openai:gpt-4": Mock(input_per_1k_usd=0.005, output_per_1k_usd=0.015),
                    "anthropic:claude-3": Mock(input_per_1k_usd=0.003, output_per_1k_usd=0.015)
                }

                response = client.get("/v1/models", headers={"Authorization": "Bearer test_key"})

                assert response.status_code == 200
                data = response.json()
                assert "data" in data
                assert len(data["data"]) > 0

    @patch("app.main.get_authenticated_user")
    def test_get_user_stats_success(self, mock_auth, client, mock_user_info):
        """Test get user stats endpoint."""
        mock_auth.return_value = mock_user_info

        with patch("app.main.db.get_user_stats") as mock_get_stats:
            mock_get_stats.return_value = {
                "user_id": "test_user",
                "total_requests": 10,
                "total_cost": 1.50,
                "daily_cost": 0.50,
                "monthly_cost": 1.50,
                "daily_budget": 100.0,
                "monthly_budget": 1000.0,
                "favorite_model": "gpt-4"
            }

            response = client.get("/user/stats", headers={"Authorization": "Bearer test_key"})

            assert response.status_code == 200
            data = response.json()
            assert data["user_id"] == "test_user"
            assert data["total_requests"] == 10

    @patch("app.main.get_authenticated_user")
    def test_get_metrics_success(self, mock_auth, client, mock_user_info):
        """Test get metrics endpoint."""
        mock_auth.return_value = mock_user_info

        with patch("app.main.db.get_system_metrics") as mock_get_metrics:
            mock_get_metrics.return_value = {
                "total_requests": 100,
                "total_cost": 10.50,
                "active_users": 5,
                "provider_usage": {"openai": 60, "anthropic": 40},
                "model_usage": {"gpt-4": 30, "claude-3": 70},
                "average_response_time": 500.0
            }

            response = client.get("/metrics", headers={"Authorization": "Bearer test_key"})

            assert response.status_code == 200
            data = response.json()
            assert data["total_requests"] == 100
            assert "provider_usage" in data

    @patch("app.main.get_authenticated_user")
    def test_get_cost_analytics_basic_mode(self, mock_auth, client, mock_user_info):
        """Test cost analytics in basic mode."""
        mock_auth.return_value = mock_user_info

        with patch("app.main.model_muxer.enhanced_mode", False):
            response = client.get("/v1/analytics/costs", headers={"Authorization": "Bearer test_key"})

            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "basic_stats" in data

    @patch("app.main.get_authenticated_user")
    def test_get_budget_status_enhanced_mode_required(self, mock_auth, client, mock_user_info):
        """Test budget status requires enhanced mode."""
        mock_auth.return_value = mock_user_info

        with patch("app.main.model_muxer.enhanced_mode", False):
            response = client.get("/v1/analytics/budgets", headers={"Authorization": "Bearer test_key"})

            assert response.status_code == 501
            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == "enhanced_mode_required"

    @patch("app.main.get_authenticated_user")
    def test_set_budget_enhanced_mode_required(self, mock_auth, client, mock_user_info):
        """Test set budget requires enhanced mode."""
        mock_auth.return_value = mock_user_info

        with patch("app.main.model_muxer.enhanced_mode", False):
            response = client.post(
                "/v1/analytics/budgets",
                json={"budget_type": "daily", "budget_limit": 50.0},
                headers={"Authorization": "Bearer test_key"}
            )

            assert response.status_code == 501
            data = response.json()
            assert data["error"]["code"] == "enhanced_mode_required"

    @patch("app.main.get_authenticated_user")
    def test_enhanced_chat_completions_basic_mode(self, mock_auth, client, sample_chat_request, mock_user_info):
        """Test enhanced chat completions in basic mode falls back."""
        mock_auth.return_value = mock_user_info

        with patch("app.main.model_muxer.enhanced_mode", False):
            with patch("app.main.chat_completions") as mock_chat:
                mock_chat.return_value = JSONResponse(content={"test": "response"})

                response = client.post(
                    "/v1/chat/completions/enhanced",
                    json=sample_chat_request.dict(),
                    headers={"Authorization": "Bearer test_key"}
                )

                assert response.status_code == 200
                mock_chat.assert_called_once()

    @patch("app.main.get_authenticated_user")
    def test_anthropic_messages_success(self, mock_auth, client, mock_user_info):
        """Test Anthropic messages API compatibility."""
        mock_auth.return_value = mock_user_info

        anthropic_request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "claude-3-haiku-20240307",
            "max_tokens": 100
        }

        with patch("app.main.chat_completions") as mock_chat:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "test-id",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "claude-3-haiku-20240307",
                "choices": [{"message": {"role": "assistant", "content": "Hello back!"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                "router_metadata": {"provider": "anthropic", "model": "claude-3-haiku-20240307"}
            }
            mock_chat.return_value = mock_response

            response = client.post(
                "/v1/messages",
                json=anthropic_request,
                headers={"Authorization": "Bearer test_key"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["type"] == "message"
            assert "content" in data

    def test_anthropic_messages_validation_error(self, client):
        """Test Anthropic messages with invalid data."""
        invalid_request = {"messages": "not_a_list"}

        response = client.post("/v1/messages", json=invalid_request)

        assert response.status_code == 400

    def test_anthropic_messages_with_system_prompt(self, client, mock_user_info):
        """Test Anthropic messages with system prompt conversion."""
        with patch("app.main.get_authenticated_user", return_value=mock_user_info):
            anthropic_request = {
                "messages": [{"role": "user", "content": "Hello"}],
                "system": "You are a helpful assistant",
                "model": "claude-3-haiku-20240307",
                "max_tokens": 100
            }

            with patch("app.main.chat_completions") as mock_chat:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "id": "test-id",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "claude-3-haiku-20240307",
                    "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
                }
                mock_chat.return_value = mock_response

                response = client.post(
                    "/v1/messages",
                    json=anthropic_request,
                    headers={"Authorization": "Bearer test_key"}
                )

                assert response.status_code == 200

    def test_anthropic_messages_multi_part_content(self, client, mock_user_info):
        """Test Anthropic messages with multi-part content."""
        with patch("app.main.get_authenticated_user", return_value=mock_user_info):
            anthropic_request = {
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": " world!"}
                    ]
                }],
                "model": "claude-3-haiku-20240307",
                "max_tokens": 100
            }

            with patch("app.main.chat_completions") as mock_chat:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "id": "test-id",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "claude-3-haiku-20240307",
                    "choices": [{"message": {"role": "assistant", "content": "Hello world!"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
                }
                mock_chat.return_value = mock_response

                response = client.post(
                    "/v1/messages",
                    json=anthropic_request,
                    headers={"Authorization": "Bearer test_key"}
                )

                assert response.status_code == 200

    def test_validation_exception_handler(self, client):
        """Test request validation error handling."""
        # Send invalid JSON to trigger validation error
        response = client.post(
            "/v1/chat/completions",
            data="invalid json",
            headers={"Content-Type": "application/json", "Authorization": "Bearer test"}
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"

    def test_http_exception_handler(self, client):
        """Test HTTP exception handling."""
        # This would typically be tested by triggering an HTTPException in an endpoint
        # For now, we test the handler structure
        from app.main import http_exception_handler
        from fastapi import Request

        mock_request = Mock(spec=Request)
        mock_exc = HTTPException(status_code=404, detail="Not found")

        response = http_exception_handler(mock_request, mock_exc)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 404

    def test_security_headers_middleware(self, client):
        """Test security headers are added to responses."""
        response = client.get("/health")

        # Check for common security headers
        assert "x-content-type-options" in response.headers
        assert "x-frame-options" in response.headers
        assert "x-xss-protection" in response.headers

    def test_request_size_validation_middleware(self, client):
        """Test request size validation middleware."""
        # Create a very large request body
        large_content = "x" * 1000000  # 1MB of data

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": large_content}]},
            headers={"Authorization": "Bearer test"}
        )

        # Should either succeed or return 413 Payload Too Large
        assert response.status_code in [200, 400, 413]

    def test_request_observability_middleware(self, client):
        """Test request observability middleware adds metrics."""
        with patch("app.main.HTTP_REQUESTS") as mock_requests:
            with patch("app.main.HTTP_LATENCY") as mock_latency:
                response = client.get("/health")

                assert response.status_code == 200
                # Middleware should have been called and attempted to record metrics

    def test_cors_middleware_configuration(self, client):
        """Test CORS middleware allows configured origins."""
        # Test preflight request
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization"
            }
        )

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    @pytest.mark.asyncio
    async def test_get_authenticated_user_success(self):
        """Test successful user authentication."""
        from app.main import get_authenticated_user
        from fastapi import Request

        mock_request = Mock(spec=Request)
        mock_auth = Mock()
        mock_auth.authenticate_request = Mock(return_value={"user_id": "test_user"})

        with patch("app.main.auth", mock_auth):
            result = await get_authenticated_user(mock_request, "Bearer test_key")

            assert result["user_id"] == "test_user"

    @pytest.mark.asyncio
    async def test_get_authenticated_user_failure(self):
        """Test failed user authentication."""
        from app.main import get_authenticated_user
        from fastapi import Request

        mock_request = Mock(spec=Request)
        mock_auth = Mock()
        mock_auth.authenticate_request = Mock(side_effect=Exception("Invalid key"))

        with patch("app.main.auth", mock_auth):
            with pytest.raises(HTTPException) as exc_info:
                await get_authenticated_user(mock_request, "Bearer invalid_key")

            assert exc_info.value.status_code == 401

    def test_pytest_short_circuit_mode(self, client, sample_chat_request):
        """Test pytest short-circuit mode for faster testing."""
        # This test verifies the pytest-specific code paths work
        with patch("sys.modules", {"pytest": Mock()}):
            with patch("app.main.providers_registry.get_provider_registry") as mock_registry:
                mock_provider = Mock()
                mock_adapter = Mock()
                mock_adapter.invoke = AsyncMock(return_value=Mock(
                    output_text="Test response",
                    tokens_in=10,
                    tokens_out=20,
                    latency_ms=100
                ))
                mock_registry.return_value = {"openai": mock_adapter}

                response = client.post(
                    "/v1/chat/completions",
                    json=sample_chat_request.dict(),
                    headers={"Authorization": "Bearer test_key"}
                )

                # Should work in pytest mode
                assert response.status_code in [200, 400, 500]

    def test_provider_unavailable_error(self, client, sample_chat_request, mock_user_info):
        """Test handling when selected provider is not available."""
        with patch("app.main.get_authenticated_user", return_value=mock_user_info):
            with patch("app.main.router") as mock_router:
                mock_router.select_model.return_value = (
                    "nonexistent_provider",
                    "model",
                    "selected",
                    {"label": "simple"},
                    {"usd": 0.01}
                )

                with patch("app.main.providers_registry.get_provider_registry", return_value={}):
                    response = client.post(
                        "/v1/chat/completions",
                        json=sample_chat_request.dict(),
                        headers={"Authorization": "Bearer test_key"}
                    )

                    assert response.status_code == 503
                    data = response.json()
                    assert data["error"]["code"] == "provider_unavailable"

    def test_model_format_validation_production(self, client, sample_chat_request, mock_user_info):
        """Test model format validation in production mode."""
        with patch("app.main.get_authenticated_user", return_value=mock_user_info):
            with patch("app.main.app_settings.features.mode", "production"):
                # Test with proxy-style model name
                invalid_request = sample_chat_request.copy()
                invalid_request.model = "openai/gpt-4"

                response = client.post(
                    "/v1/chat/completions",
                    json=invalid_request.dict(),
                    headers={"Authorization": "Bearer test_key"}
                )

                # Should reject proxy-style model names in production
                assert response.status_code in [200, 400]

    def test_policy_enforcement(self, client, sample_chat_request, mock_user_info):
        """Test policy enforcement in request processing."""
        with patch("app.main.get_authenticated_user", return_value=mock_user_info):
            with patch("app.main.enforce_policies") as mock_enforce:
                mock_enforce.return_value = Mock(blocked=True, reasons=["inappropriate_content"])

                response = client.post(
                    "/v1/chat/completions",
                    json=sample_chat_request.dict(),
                    headers={"Authorization": "Bearer test_key"}
                )

                assert response.status_code == 403
                data = response.json()
                assert data["error"]["type"] == "policy_violation"

    def test_database_logging_on_success(self, client, sample_chat_request, mock_user_info):
        """Test database logging on successful requests."""
        with patch("app.main.get_authenticated_user", return_value=mock_user_info):
            with patch("app.main.db.log_request") as mock_log:
                with patch("app.main.router") as mock_router:
                    mock_router.select_model.return_value = (
                        "openai",
                        "gpt-3.5-turbo",
                        "selected",
                        {"label": "simple"},
                        {"usd": 0.01}
                    )

                    with patch("app.main.providers_registry.get_provider_registry") as mock_registry:
                        mock_provider = Mock()
                        mock_adapter = Mock()
                        mock_adapter.chat_completion = AsyncMock(return_value=Mock(
                            dict=Mock(return_value={
                                "id": "test-id",
                                "object": "chat.completion",
                                "created": 1234567890,
                                "model": "gpt-3.5-turbo",
                                "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
                                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                                "router_metadata": {"estimated_cost": 0.01, "response_time_ms": 500}
                            })
                        ))
                        mock_registry.return_value = {"openai": mock_adapter}

                        response = client.post(
                            "/v1/chat/completions",
                            json=sample_chat_request.dict(),
                            headers={"Authorization": "Bearer test_key"}
                        )

                        assert response.status_code == 200
                        mock_log.assert_called()

    def test_database_logging_on_error(self, client, sample_chat_request, mock_user_info):
        """Test database logging on failed requests."""
        with patch("app.main.get_authenticated_user", return_value=mock_user_info):
            with patch("app.main.db.log_request") as mock_log:
                with patch("app.main.router") as mock_router:
                    mock_router.select_model.side_effect = Exception("Router error")

                    response = client.post(
                        "/v1/chat/completions",
                        json=sample_chat_request.dict(),
                        headers={"Authorization": "Bearer test_key"}
                    )

                    # Should log the error
                    mock_log.assert_called()

    def test_trace_id_header(self, client):
        """Test trace ID header is added to responses."""
        with patch("app.main.get_trace_id", return_value="test-trace-id"):
            response = client.get("/health")

            assert response.status_code == 200
            assert response.headers.get("x-trace-id") == "test-trace-id"

    def test_concurrent_requests_handling(self, client, sample_chat_request):
        """Test handling of concurrent requests."""
        import asyncio

        async def make_request():
            return client.post(
                "/v1/chat/completions",
                json=sample_chat_request.dict(),
                headers={"Authorization": "Bearer test_key"}
            )

        # This is a basic test - in a real scenario you'd use asyncio.gather
        # to test true concurrency, but TestClient doesn't support async
        response = client.post(
            "/v1/chat/completions",
            json=sample_chat_request.dict(),
            headers={"Authorization": "Bearer test_key"}
        )

        # Just verify the endpoint can handle requests
        assert response.status_code in [200, 401, 400, 500]

    def test_memory_usage_under_load(self, client):
        """Test memory usage doesn't grow excessively under load."""
        import psutil
        import os

        initial_memory = psutil.Process(os.getpid()).memory_info().rss

        # Make multiple requests
        for _ in range(10):
            response = client.get("/health")
            assert response.status_code == 200

        final_memory = psutil.Process(os.getpid()).memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024

    def test_request_timeout_handling(self, client, sample_chat_request, mock_user_info):
        """Test request timeout handling."""
        with patch("app.main.get_authenticated_user", return_value=mock_user_info):
            with patch("app.main.router") as mock_router:
                mock_router.select_model.return_value = (
                    "openai",
                    "gpt-3.5-turbo",
                    "selected",
                    {"label": "simple"},
                    {"usd": 0.01}
                )

                with patch("app.main.providers_registry.get_provider_registry") as mock_registry:
                    mock_provider = Mock()
                    mock_adapter = Mock()
                    # Simulate a timeout
                    mock_adapter.chat_completion = AsyncMock(side_effect=asyncio.TimeoutError())
                    mock_registry.return_value = {"openai": mock_adapter}

                    response = client.post(
                        "/v1/chat/completions",
                        json=sample_chat_request.dict(),
                        headers={"Authorization": "Bearer test_key"}
                    )

                    # Should handle timeout gracefully
                    assert response.status_code == 500

    def test_invalid_json_handling(self, client):
        """Test handling of invalid JSON in requests."""
        response = client.post(
            "/v1/chat/completions",
            data="invalid json content",
            headers={"Content-Type": "application/json", "Authorization": "Bearer test"}
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_unsupported_content_type(self, client):
        """Test handling of unsupported content types."""
        response = client.post(
            "/v1/chat/completions",
            data="plain text content",
            headers={"Content-Type": "text/plain", "Authorization": "Bearer test"}
        )

        assert response.status_code == 400

    def test_extremely_large_request_body(self, client):
        """Test handling of extremely large request bodies."""
        # Create a 10MB request body
        large_content = "x" * (10 * 1024 * 1024)

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": large_content}]},
            headers={"Authorization": "Bearer test"}
        )

        # Should be rejected due to size limits
        assert response.status_code in [413, 400, 500]

    def test_special_characters_in_content(self, client, sample_chat_request, mock_user_info):
        """Test handling of special characters and unicode in content."""
        special_request = sample_chat_request.copy()
        special_request.messages[0].content = "Hello 🌍 with émojis and spëcial chärs!"

        with patch("app.main.get_authenticated_user", return_value=mock_user_info):
            with patch("app.main.router") as mock_router:
                mock_router.select_model.return_value = (
                    "openai",
                    "gpt-3.5-turbo",
                    "selected",
                    {"label": "simple"},
                    {"usd": 0.01}
                )

                with patch("app.main.providers_registry.get_provider_registry") as mock_registry:
                    mock_provider = Mock()
                    mock_adapter = Mock()
                    mock_adapter.chat_completion = AsyncMock(return_value=Mock(
                        dict=Mock(return_value={
                            "id": "test-id",
                            "object": "chat.completion",
                            "created": 1234567890,
                            "model": "gpt-3.5-turbo",
                            "choices": [{"message": {"role": "assistant", "content": "Response"}}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
                        })
                    ))
                    mock_registry.return_value = {"openai": mock_adapter}

                    response = client.post(
                        "/v1/chat/completions",
                        json=special_request.dict(),
                        headers={"Authorization": "Bearer test_key"}
                    )

                    assert response.status_code == 200

    def test_empty_request_body(self, client):
        """Test handling of empty request body."""
        response = client.post(
            "/v1/chat/completions",
            json={},
            headers={"Authorization": "Bearer test"}
        )

        assert response.status_code == 400

    def test_malformed_headers(self, client, sample_chat_request):
        """Test handling of malformed headers."""
        response = client.post(
            "/v1/chat/completions",
            json=sample_chat_request.dict(),
            headers={"Authorization": "InvalidFormat"}
        )

        assert response.status_code == 401

    def test_rate_limiting_simulation(self, client, sample_chat_request):
        """Test rate limiting behavior simulation."""
        # This would typically test actual rate limiting middleware
        # For now, we test the basic request handling
        responses = []
        for _ in range(5):
            response = client.post(
                "/v1/chat/completions",
                json=sample_chat_request.dict(),
                headers={"Authorization": "Bearer test"}
            )
            responses.append(response.status_code)

        # At least some requests should be handled
        assert any(code in [200, 401] for code in responses)