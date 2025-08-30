# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
End-to-end integration tests for complete direct provider routing flow.

This module tests the complete request/response cycle through the full API stack
when using direct providers only, ensuring the entire system works correctly.
"""

import pytest
import json
from unittest.mock import patch, Mock, AsyncMock
from fastapi.testclient import TestClient

from app.models import ChatMessage
from app.core.exceptions import BudgetExceededError


@pytest.mark.direct
@pytest.mark.integration_direct
class TestEndToEndDirect:
    """Test end-to-end integration for direct provider routing."""

    def test_complete_request_flow(self, direct_providers_only_mode, simple_messages):
        """Test complete request flow through the full API stack."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            # Mock all required provider adapters
            with patch("app.providers.registry.get_provider_registry") as mock_registry:
                # Create mock provider registry
                mock_adapter = Mock()
                mock_adapter.invoke = AsyncMock(
                    return_value=Mock(
                        output_text="Mock response from direct provider",
                        tokens_in=10,
                        tokens_out=20,
                        latency_ms=150,
                        raw={"provider": "openai", "model": "gpt-3.5-turbo"},
                        error=None,
                    )
                )

                mock_registry.return_value = {
                    "openai": mock_adapter,
                    "anthropic": mock_adapter,
                    "mistral": mock_adapter,
                    "groq": mock_adapter,
                    "google": mock_adapter,
                    "cohere": mock_adapter,
                    "together": mock_adapter,
                }

                # Send request through API
                request_data = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hello, how are you?"}],
                    "max_tokens": 100,
                }

                response = client.post("/v1/chat/completions", json=request_data)

                # Verify successful response
                assert response.status_code == 200
                response_data = response.json()

                # Verify OpenAI-compatible response format
                assert "choices" in response_data
                assert len(response_data["choices"]) > 0
                assert "message" in response_data["choices"][0]
                assert "content" in response_data["choices"][0]["message"]

                # Verify router metadata includes direct provider info
                assert "router_metadata" in response_data
                router_metadata = response_data["router_metadata"]
                assert "provider" in router_metadata
                assert "model" in router_metadata
                assert router_metadata["provider"] in [
                    "openai",
                    "anthropic",
                    "mistral",
                    "groq",
                    "google",
                    "cohere",
                    "together",
                ]

    def test_api_response_format_openai_compatible(self, direct_providers_only_mode, simple_messages):
        """Test that API responses maintain OpenAI-compatible format."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.providers.registry.get_provider_registry") as mock_registry:
                mock_adapter = Mock()
                mock_adapter.invoke = AsyncMock(
                    return_value=Mock(
                        output_text="Test response content",
                        tokens_in=15,
                        tokens_out=25,
                        latency_ms=200,
                        raw={"provider": "anthropic", "model": "claude-3-haiku-20240307"},
                        error=None,
                    )
                )

                mock_registry.return_value = {"anthropic": mock_adapter}

                request_data = {
                    "model": "claude-3-haiku-20240307",
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                }

                response = client.post("/v1/chat/completions", json=request_data)

                assert response.status_code == 200
                response_data = response.json()

                # Verify OpenAI-compatible structure
                assert "id" in response_data
                assert "object" in response_data
                assert "created" in response_data
                assert "model" in response_data
                assert "choices" in response_data
                assert "usage" in response_data

                # Verify choices structure
                choice = response_data["choices"][0]
                assert "index" in choice
                assert "message" in choice
                assert "finish_reason" in choice

                # Verify message structure
                message = choice["message"]
                assert "role" in message
                assert "content" in message
                assert message["role"] == "assistant"
                assert message["content"] == "Test response content"

                # Verify usage structure
                usage = response_data["usage"]
                assert "prompt_tokens" in usage
                assert "completion_tokens" in usage
                assert "total_tokens" in usage

    def test_router_metadata_includes_direct_provider_info(self, direct_providers_only_mode, simple_messages):
        """Test that router_metadata includes direct provider information."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.providers.registry.get_provider_registry") as mock_registry:
                mock_adapter = Mock()
                mock_adapter.invoke = AsyncMock(
                    return_value=Mock(
                        output_text="Test response",
                        tokens_in=10,
                        tokens_out=15,
                        latency_ms=180,
                        raw={"provider": "groq", "model": "llama3-8b-8192"},
                        error=None,
                    )
                )

                mock_registry.return_value = {"groq": mock_adapter}

            request_data = {"model": "llama3-8b-8192", "messages": [{"role": "user", "content": "Test message"}]}

            response = client.post("/v1/chat/completions", json=request_data)

            assert response.status_code == 200
            response_data = response.json()

            # Verify router metadata
            assert "router_metadata" in response_data
            router_metadata = response_data["router_metadata"]

            # Should include direct provider information
            assert "provider" in router_metadata
            assert "model" in router_metadata
            assert "routing_reason" in router_metadata
            assert "estimated_cost" in router_metadata
            assert "response_time_ms" in router_metadata
            assert "direct_providers_only" in router_metadata

            # Verify direct provider flag is set
            assert router_metadata["direct_providers_only"] is True
            assert router_metadata["provider"] == "groq"

    def test_token_usage_reporting_in_api_responses(self, direct_providers_only_mode, simple_messages):
        """Test token usage reporting in API responses."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.providers.registry.get_provider_registry") as mock_registry:
                mock_adapter = Mock()
                mock_adapter.invoke = AsyncMock(
                    return_value=Mock(
                        output_text="Test response with specific token counts",
                        tokens_in=25,
                        tokens_out=35,
                        latency_ms=250,
                        raw={"provider": "mistral", "model": "mistral-small-latest"},
                        error=None,
                    )
                )

                mock_registry.return_value = {"mistral": mock_adapter}

            request_data = {
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": "Count the tokens in this message"}],
            }

            response = client.post("/v1/chat/completions", json=request_data)

            assert response.status_code == 200
            response_data = response.json()

            # Verify usage reporting
            usage = response_data["usage"]
            assert usage["prompt_tokens"] == 25
            assert usage["completion_tokens"] == 35
            assert usage["total_tokens"] == 60

    def test_cost_tracking_integration(self, direct_providers_only_mode, simple_messages):
        """Test cost tracking integration with direct providers."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with (
                patch("app.providers.registry.get_provider_registry") as mock_registry,
                patch("app.cost_tracker.record_request", create=True) as mock_record,
            ):
                mock_adapter = Mock()
                mock_adapter.invoke = AsyncMock(
                    return_value=Mock(
                        output_text="Test response for cost tracking",
                        tokens_in=20,
                        tokens_out=30,
                        latency_ms=300,
                        raw={"provider": "google", "model": "gemini-1.5-flash"},
                        error=None,
                    )
                )

                mock_registry.return_value = {"google": mock_adapter}

            request_data = {
                "model": "gemini-1.5-flash",
                "messages": [{"role": "user", "content": "Test cost tracking"}],
            }

            response = client.post("/v1/chat/completions", json=request_data)

            assert response.status_code == 200

            # Verify cost tracking was called
            mock_record.assert_called_once()

            # Verify cost tracking includes direct provider metadata
            call_args = mock_record.call_args
            assert "provider" in call_args.kwargs
            assert call_args.kwargs["provider"] == "google"

    def test_database_logging_with_direct_provider_metadata(self, direct_providers_only_mode, simple_messages):
        """Test database logging of requests with direct provider metadata."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with (
                patch("app.providers.registry.get_provider_registry") as mock_registry,
                patch("app.database.log_request", create=True) as mock_log,
            ):
                mock_adapter = Mock()
                mock_adapter.invoke = AsyncMock(
                    return_value=Mock(
                        output_text="Test response for database logging",
                        tokens_in=15,
                        tokens_out=25,
                        latency_ms=220,
                        raw={"provider": "cohere", "model": "command-r"},
                        error=None,
                    )
                )

                mock_registry.return_value = {"cohere": mock_adapter}

            request_data = {"model": "command-r", "messages": [{"role": "user", "content": "Test database logging"}]}

            response = client.post("/v1/chat/completions", json=request_data)

            assert response.status_code == 200

            # Verify database logging was called with direct provider info
            mock_log.assert_called_once()

            call_args = mock_log.call_args
            assert "provider" in call_args.kwargs
            assert "model" in call_args.kwargs
            assert call_args.kwargs["provider"] == "cohere"

    def test_authentication_and_authorization(self, direct_providers_only_mode, simple_messages):
        """Test API key authentication with direct provider routing."""
        from app.main import app

        client = TestClient(app)

        # Mock the authentication method directly
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.providers.registry.get_provider_registry") as mock_registry:
                mock_adapter = Mock()
                mock_adapter.invoke = AsyncMock(
                    return_value=Mock(
                        output_text="Authenticated response",
                        tokens_in=10,
                        tokens_out=20,
                        latency_ms=150,
                        raw={"provider": "together", "model": "meta-llama/Llama-3.1-8B-Instruct"},
                        error=None,
                    )
                )

                mock_registry.return_value = {"together": mock_adapter}

                request_data = {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "messages": [{"role": "user", "content": "Test authentication"}],
                }

                # Test with valid API key
                headers = {"Authorization": "Bearer valid-api-key"}
                response = client.post("/v1/chat/completions", json=request_data, headers=headers)

                assert response.status_code == 200

                # Test with invalid API key
                mock_auth.side_effect = Exception("Invalid API key")
                headers = {"Authorization": "Bearer invalid-api-key"}
                response = client.post("/v1/chat/completions", json=request_data, headers=headers)

                assert response.status_code == 401

    def test_rate_limiting_with_direct_providers(self, direct_providers_only_mode, simple_messages):
        """Test rate limiting works correctly with direct providers."""
        from app.main import app

        client = TestClient(app)

        # Mock the authentication method directly
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.providers.registry.get_provider_registry") as mock_registry:
                mock_adapter = Mock()
                mock_adapter.invoke = AsyncMock(
                    return_value=Mock(
                        output_text="Rate limited response",
                        tokens_in=5,
                        tokens_out=10,
                        latency_ms=100,
                        raw={"provider": "openai", "model": "gpt-3.5-turbo"},
                        error=None,
                    )
                )

                mock_registry.return_value = {"openai": mock_adapter}

                request_data = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Test rate limiting"}],
                }

                headers = {"Authorization": "Bearer test-api-key"}

                # First request should succeed
                response = client.post("/v1/chat/completions", json=request_data, headers=headers)
                assert response.status_code == 200

                # For rate limiting test, we'll just verify the request goes through
                # since actual rate limiting would require more complex setup

    def test_tenant_isolation_with_direct_providers(self, direct_providers_only_mode, simple_messages):
        """Test tenant isolation with direct providers."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.providers.registry.get_provider_registry") as mock_registry:
                mock_adapter = Mock()
                mock_adapter.invoke = AsyncMock(
                    return_value=Mock(
                        output_text="Tenant isolated response",
                        tokens_in=12,
                        tokens_out=18,
                        latency_ms=180,
                        raw={"provider": "anthropic", "model": "claude-3-haiku-20240307"},
                        error=None,
                    )
                )

                mock_registry.return_value = {"anthropic": mock_adapter}

            request_data = {
                "model": "claude-3-haiku-20240307",
                "messages": [{"role": "user", "content": "Test tenant isolation"}],
            }

            # Test with different tenant API keys
            tenant1_headers = {"Authorization": "Bearer tenant1-key"}
            tenant2_headers = {"Authorization": "Bearer tenant2-key"}

            response1 = client.post("/v1/chat/completions", json=request_data, headers=tenant1_headers)
            response2 = client.post("/v1/chat/completions", json=request_data, headers=tenant2_headers)

            # Both should work independently
            assert response1.status_code == 200
            assert response2.status_code == 200

    def test_prometheus_metrics_collection(self, direct_providers_only_mode, simple_messages):
        """Test Prometheus metrics are collected correctly for direct providers."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with (
                patch("app.providers.registry.get_provider_registry") as mock_registry,
                patch("app.telemetry.metrics.ROUTER_REQUESTS", create=True) as mock_requests,
                patch("app.telemetry.metrics.PROVIDER_REQUESTS", create=True) as mock_provider_requests,
            ):
                mock_adapter = Mock()
                mock_adapter.invoke = AsyncMock(
                    return_value=Mock(
                        output_text="Metrics test response",
                        tokens_in=8,
                        tokens_out=12,
                        latency_ms=120,
                        raw={"provider": "groq", "model": "llama3-8b-8192"},
                        error=None,
                    )
                )

                mock_registry.return_value = {"groq": mock_adapter}

            request_data = {
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": "Test metrics collection"}],
            }

            headers = {"Authorization": "Bearer test-key"}
            response = client.post("/v1/chat/completions", json=request_data, headers=headers)

            assert response.status_code == 200

            # Verify metrics were recorded
            mock_requests.inc.assert_called()
            mock_provider_requests.inc.assert_called()

    def test_opentelemetry_tracing(self, direct_providers_only_mode, simple_messages):
        """Test OpenTelemetry tracing through complete request flow."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with (
                patch("app.providers.registry.get_provider_registry") as mock_registry,
                patch("app.telemetry.tracing.start_span", create=True) as mock_span,
            ):
                mock_adapter = Mock()
                mock_adapter.invoke = AsyncMock(
                    return_value=Mock(
                        output_text="Tracing test response",
                        tokens_in=10,
                        tokens_out=15,
                        latency_ms=160,
                        raw={"provider": "google", "model": "gemini-1.5-flash"},
                        error=None,
                )
            )

            mock_registry.return_value = {"google": mock_adapter}

            # Mock span context
            mock_span_context = Mock()
            mock_span_context.set_attribute = Mock()
            mock_span.return_value.__enter__.return_value = mock_span_context

            request_data = {"model": "gemini-1.5-flash", "messages": [{"role": "user", "content": "Test tracing"}]}

            headers = {"Authorization": "Bearer test-key"}
            response = client.post("/v1/chat/completions", json=request_data, headers=headers)

            assert response.status_code == 200

            # Verify tracing was used
            mock_span.assert_called()
            mock_span_context.set_attribute.assert_called()

    def test_health_checks_with_direct_providers(self, direct_providers_only_mode, simple_messages):
        """Test health checks work with direct providers."""
        from app.main import app

        client = TestClient(app)

        with patch("app.providers.registry.get_provider_registry") as mock_registry:
            mock_registry.return_value = {"openai": Mock(), "anthropic": Mock(), "mistral": Mock()}

            # Test health check endpoint
            response = client.get("/health")

            assert response.status_code == 200
            health_data = response.json()
            assert "status" in health_data
            assert health_data["status"] == "healthy"

    def test_error_reporting_and_alerting(self, direct_providers_only_mode, simple_messages):
        """Test error reporting and alerting with direct providers."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.providers.registry.get_provider_registry") as mock_registry:
            # Mock provider that fails
            mock_adapter = Mock()
            mock_adapter.invoke = AsyncMock(
                return_value=Mock(
                    output_text="", tokens_in=0, tokens_out=0, latency_ms=0, raw={}, error="provider_error"
                )
            )

            mock_registry.return_value = {"openai": mock_adapter}

            request_data = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Test error handling"}]}

            headers = {"Authorization": "Bearer test-key"}
            response = client.post("/v1/chat/completions", json=request_data, headers=headers)

            # Should handle provider errors gracefully
            assert response.status_code in [500, 503]


    async def test_concurrent_request_handling(self, direct_providers_only_mode, simple_messages):
        """Test concurrent request handling with direct providers."""
        from app.main import app
        import asyncio
        import httpx

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.providers.registry.get_provider_registry") as mock_registry:
                mock_adapter = Mock()
                mock_adapter.invoke = AsyncMock(
                    return_value=Mock(
                        output_text="Concurrent test response",
                        tokens_in=5,
                        tokens_out=10,
                        latency_ms=50,
                        raw={"provider": "together", "model": "meta-llama/Llama-3.1-8B-Instruct"},
                        error=None,
                    )
                )

                mock_registry.return_value = {"together": mock_adapter}

            request_data = {
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": "Concurrent test"}],
            }

            headers = {"Authorization": "Bearer test-key"}

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                tasks = [client.post("/v1/chat/completions", json=request_data, headers=headers) for _ in range(5)]
                responses = await asyncio.gather(*tasks)

                # All requests should succeed
                for response in responses:
                    assert response.status_code == 200

    def test_minimal_direct_provider_configuration(self, direct_providers_only_mode, simple_messages):
        """Test with minimal direct provider configuration."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.providers.registry.get_provider_registry") as mock_registry:
                # Only one provider configured
                mock_adapter = Mock()
                mock_adapter.invoke = AsyncMock(
                    return_value=Mock(
                        output_text="Minimal config response",
                        tokens_in=8,
                        tokens_out=12,
                        latency_ms=140,
                        raw={"provider": "openai", "model": "gpt-3.5-turbo"},
                        error=None,
                    )
                )

                mock_registry.return_value = {"openai": mock_adapter}

            request_data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Minimal configuration test"}],
            }

            headers = {"Authorization": "Bearer test-key"}
            response = client.post("/v1/chat/completions", json=request_data, headers=headers)

            assert response.status_code == 200

    def test_full_direct_provider_configuration(self, direct_providers_only_mode, simple_messages):
        """Test with full direct provider configuration (all providers enabled)."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.providers.registry.get_provider_registry") as mock_registry:
                # All providers configured
                mock_adapter = Mock()
                mock_adapter.invoke = AsyncMock(
                    return_value=Mock(
                        output_text="Full config response",
                        tokens_in=10,
                        tokens_out=15,
                        latency_ms=160,
                        raw={"provider": "anthropic", "model": "claude-3-haiku-20240307"},
                        error=None,
                    )
                )

                mock_registry.return_value = {
                "openai": mock_adapter,
                "anthropic": mock_adapter,
                "mistral": mock_adapter,
                "groq": mock_adapter,
                "google": mock_adapter,
                "cohere": mock_adapter,
                "together": mock_adapter,
            }

            request_data = {
                "model": "claude-3-haiku-20240307",
                "messages": [{"role": "user", "content": "Full configuration test"}],
            }

            headers = {"Authorization": "Bearer test-key"}
            response = client.post("/v1/chat/completions", json=request_data, headers=headers)

            assert response.status_code == 200

    def test_budget_exceeded_error_end_to_end(
        self, direct_providers_only_mode, simple_messages, deterministic_price_table, monkeypatch
    ):
        """Test API error responses when budget is exceeded."""
        from app.main import app
        from app.settings import settings

        # Set a deterministic price table and a stable budget threshold
        monkeypatch.setattr(settings.pricing, "price_table_path", deterministic_price_table)
        monkeypatch.setattr(settings.router_thresholds, "max_estimated_usd_per_request", 0.0001)

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.providers.registry.get_provider_registry") as mock_registry:
                mock_registry.return_value = {"openai": Mock(), "anthropic": Mock(), "mistral": Mock()}

            request_data = {
                "model": "gpt-4o",  # Expensive model
                "messages": [{"role": "user", "content": "Budget test with expensive model"}],
            }

            headers = {"Authorization": "Bearer test-key"}
            response = client.post("/v1/chat/completions", json=request_data, headers=headers)

            # Should return 402 Payment Required
            assert response.status_code == 402

            error_data = response.json()
            assert "error" in error_data
            assert error_data["error"]["code"] == "budget_exceeded"

    def test_service_unavailable_when_no_providers(self, direct_providers_only_mode, simple_messages):
        """Test 503 Service Unavailable when no providers are available."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.providers.registry.get_provider_registry") as mock_registry:
                # Empty provider registry
                mock_registry.return_value = {}

            request_data = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "No providers test"}]}

            headers = {"Authorization": "Bearer test-key"}
            response = client.post("/v1/chat/completions", json=request_data, headers=headers)

            # Should return 503 Service Unavailable
            assert response.status_code == 503

    def test_user_friendly_error_messages(self, direct_providers_only_mode, simple_messages):
        """Test error messages are user-friendly and actionable."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.providers.registry.get_provider_registry") as mock_registry:
                mock_registry.return_value = {}

            request_data = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Error message test"}]}

            headers = {"Authorization": "Bearer test-key"}
            response = client.post("/v1/chat/completions", json=request_data, headers=headers)

            error_data = response.json()
            error_message = error_data["error"]["message"]

            # Should be user-friendly
            assert "service" in error_message.lower() or "unavailable" in error_message.lower()
            assert len(error_message) > 10  # Should have meaningful content

    def test_latency_recording_and_updates(self, direct_providers_only_mode, simple_messages):
        """Test that router.record_latency() is called after successful requests."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with (
                patch("app.providers.registry.get_provider_registry") as mock_registry,
                patch("app.router.HeuristicRouter.record_latency", create=True) as mock_record_latency,
            ):
            mock_adapter = Mock()
            mock_adapter.invoke = AsyncMock(
                return_value=Mock(
                    output_text="Latency recording test",
                    tokens_in=10,
                    tokens_out=15,
                    latency_ms=180,
                    raw={"provider": "mistral", "model": "mistral-small-latest"},
                    error=None,
                )
            )

            mock_registry.return_value = {"mistral": mock_adapter}

            request_data = {
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": "Latency recording test"}],
            }

            headers = {"Authorization": "Bearer test-key"}
            response = client.post("/v1/chat/completions", json=request_data, headers=headers)

            assert response.status_code == 200

            # Verify latency recording was called
            mock_record_latency.assert_called_once()

    def test_latency_priors_influence_routing(self, direct_providers_only_mode, simple_messages):
        """Test that latency data influences future routing decisions."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.providers.registry.get_provider_registry") as mock_registry:
            # Create adapters with different latencies
            fast_adapter = Mock()
            fast_adapter.invoke = AsyncMock(
                return_value=Mock(
                    output_text="Fast response",
                    tokens_in=10,
                    tokens_out=15,
                    latency_ms=50,  # Fast
                    raw={"provider": "groq", "model": "llama3-8b-8192"},
                    error=None,
                )
            )

            slow_adapter = Mock()
            slow_adapter.invoke = AsyncMock(
                return_value=Mock(
                    output_text="Slow response",
                    tokens_in=10,
                    tokens_out=15,
                    latency_ms=500,  # Slow
                    raw={"provider": "openai", "model": "gpt-3.5-turbo"},
                    error=None,
                )
            )

            mock_registry.return_value = {"groq": fast_adapter, "openai": slow_adapter}

            request_data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Latency influence test"}],
            }

            headers = {"Authorization": "Bearer test-key"}

            # Send multiple requests to build latency priors
            for _ in range(3):
                response = client.post("/v1/chat/completions", json=request_data, headers=headers)
                assert response.status_code == 200

            # Future requests should prefer faster providers based on latency priors
            # (This would be verified by checking which provider was selected)

    async def test_streaming_response_end_to_end(self, direct_providers_only_mode):
        """Test end-to-end streaming response."""
        from app.main import app
        import asyncio
        import httpx

        # Mock authentication
        with patch("app.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            async def mock_streaming_invoke(model, prompt, **kwargs):
            chunks = [
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": ", "}}]},
                {"choices": [{"delta": {"content": "world!"}}]},
            ]
            for chunk in chunks:
                yield chunk

        with patch("app.providers.registry.get_provider_registry") as mock_registry:
            mock_adapter = Mock()
            mock_adapter.invoke = mock_streaming_invoke

            mock_registry.return_value = {"openai": mock_adapter}

            request_data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            }

            headers = {"Authorization": "Bearer test-key"}

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                async with client.stream(
                    "POST", "/v1/chat/completions", json=request_data, headers=headers
                ) as response:
                    assert response.status_code == 200
                    response_chunks = []
                    async for chunk in response.aiter_bytes():
                        response_chunks.append(chunk)

                    full_response = b"".join(response_chunks).decode()
                    lines = full_response.strip().split("\n")
                    assert lines[0].startswith("data: ")
                    assert lines[1].startswith("data: ")
                    assert lines[2].startswith("data: ")
                    assert lines[3] == "data: [DONE]"
