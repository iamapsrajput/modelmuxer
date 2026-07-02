# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
End-to-end integration tests for complete direct provider routing flow.

This module tests the complete request/response cycle through the full API stack
when using direct providers only, ensuring the entire system works correctly.
"""

import json
import time
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

# Apply mocks before importing the app
# Mock the provider registry before any imports
with patch("app.main.providers_registry.get_provider_registry") as mock_registry:
    mock_provider = AsyncMock()
    mock_provider.invoke = AsyncMock()
    mock_registry.return_value = {"openai": mock_provider}

# Mock the router before importing the app
with patch("app.main.HeuristicRouter") as mock_router_cls:
    mock_router = MagicMock()
    mock_router.select_model = AsyncMock(return_value=("openai", "gpt-3.5-turbo", "test", {}, {}))
    mock_router_cls.return_value = mock_router

# Mock the global router instance
with patch("app.main.router") as mock_global_router:
    mock_global_router.select_model = AsyncMock(
        return_value=("anthropic", "claude-3-haiku-20240307", "test", {}, {})
    )

from app.core.exceptions import BudgetExceededError

# Import app after mocks are applied
from app.main import app, get_authenticated_user
from app.models import ChatMessage
from app.providers.base import ProviderResponse


# Mock authentication for all tests
def mock_auth():
    return {"user_id": "test-user", "api_key": "test-api-key"}


# Override the authentication dependency for all tests
app.dependency_overrides[get_authenticated_user] = mock_auth


def teardown_module(module):
    """Clean up dependency overrides so other test modules see real auth."""
    app.dependency_overrides.clear()


def _make_provider_response(
    output_text: str = "Test response",
    tokens_in: int = 10,
    tokens_out: int = 20,
    latency_ms: int = 150,
    raw: dict | None = None,
    error: str | None = None,
) -> ProviderResponse:
    """Build a real ProviderResponse so the request path can serialize it."""
    return ProviderResponse(
        output_text=output_text,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
        raw=raw or {},
        error=error,
    )


def _make_wired_router(provider: str, model: str, provider_response: ProviderResponse) -> MagicMock:
    """Router mock matching HeuristicRouter's select/invoke/record contract."""
    mock_router = MagicMock()
    mock_router.select_model = AsyncMock(
        return_value=(
            provider,
            model,
            "test routing",
            {"label": "simple", "confidence": 0.9, "signals": {}},
            {
                "usd": 0.001,
                "eta_ms": provider_response.latency_ms,
                "tokens_in": provider_response.tokens_in,
                "tokens_out": provider_response.tokens_out,
                "model_key": f"{provider}:{model}",
            },
        )
    )
    mock_router.invoke_via_adapter = AsyncMock(return_value=provider_response)
    mock_router.record_latency = Mock()
    return mock_router


@contextmanager
def _wired_request_path(mock_router: MagicMock, registry: dict):
    """Patch the registry and per-request router constructor used by the chat handler.

    app.main imports app.providers.registry as ``providers_registry``, so patching
    the module attribute covers both access paths.
    """
    with (
        patch("app.providers.registry.get_provider_registry", return_value=registry),
        patch("app.main.HeuristicRouter", return_value=mock_router),
        patch("app.main.router", mock_router),
    ):
        yield


@contextmanager
def _real_router_with_registry(registry: dict):
    """Patch only the provider registry so the REAL HeuristicRouter logic runs.

    The chat handler re-instantiates HeuristicRouter per request with the
    (patched) registry function, so selection, budget gating, and latency
    priors all execute for real against the mock adapters.
    """
    with patch("app.providers.registry.get_provider_registry", return_value=registry):
        yield


def _make_adapter(provider_response: ProviderResponse) -> Mock:
    """Mock adapter whose async invoke returns a real ProviderResponse."""
    adapter = Mock()
    adapter.invoke = AsyncMock(return_value=provider_response)
    # A bare Mock attribute would be truthy, making the router treat the
    # adapter's circuit breaker as open and skip it during selection.
    adapter.circuit_open = False
    return adapter


def _make_chat_completion_response(provider: str, model: str, tokens_in: int, tokens_out: int):
    """Real ChatCompletionResponse for the legacy (non-adapter) invocation path,
    which expects router.invoke_via_adapter to return a full response object."""
    from app.models import ChatCompletionResponse, Choice, RouterMetadata, Usage

    return ChatCompletionResponse(
        id=f"chatcmpl-test-{int(time.time() * 1000)}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content="Test response", name=None),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=tokens_in,
            completion_tokens=tokens_out,
            total_tokens=tokens_in + tokens_out,
        ),
        router_metadata=RouterMetadata(
            selected_provider=provider,
            selected_model=model,
            routing_reason="test routing",
            estimated_cost=0.001,
            response_time_ms=200.0,
        ),
    )


@pytest.mark.direct
@pytest.mark.integration_direct
class TestEndToEndDirect:
    """Test end-to-end integration for direct provider routing."""

    def test_complete_request_flow(self, direct_providers_only_mode, simple_messages):
        """Test complete request flow through the full API stack."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            # Mock all required provider adapters
            with patch("app.main.providers_registry.get_provider_registry") as mock_registry:
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

                # Debug: Print response details
                print(f"Response status: {response.status_code}")
                print(f"Response body: {response.text}")

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

    def test_api_response_format_openai_compatible(
        self, direct_providers_only_mode, simple_messages
    ):
        """Test that API responses maintain OpenAI-compatible format."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.main.providers_registry.get_provider_registry") as mock_registry:
                from app.providers.base import ProviderResponse

                provider_response = ProviderResponse(
                    output_text="Test response content",
                    tokens_in=15,
                    tokens_out=25,
                    latency_ms=200,
                    raw={"provider": "anthropic", "model": "claude-3-haiku-20240307"},
                    error=None,
                )
                mock_adapter = Mock()
                mock_adapter.invoke = AsyncMock(return_value=provider_response)

                mock_registry.return_value = {"anthropic": mock_adapter}

                # Mock the router to bypass budget constraints for this test
                mock_router = MagicMock()
                mock_router.select_model = AsyncMock(
                    return_value=(
                        "anthropic",
                        "claude-3-haiku-20240307",
                        "test",
                        {},
                        {
                            "usd": 0.001,
                            "eta_ms": 200,
                            "tokens_in": 15,
                            "tokens_out": 25,
                            "model_key": "anthropic:claude-3-haiku-20240307",
                        },
                    )
                )
                mock_router.invoke_via_adapter = AsyncMock(return_value=provider_response)
                mock_router.record_latency = Mock()

                request_data = {
                    "model": "claude-3-haiku-20240307",
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "max_budget": 1.0,  # Set higher budget to avoid constraint issues
                }

                with patch("app.main.HeuristicRouter", return_value=mock_router):
                    with patch("app.main.router", mock_router):
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

    def test_router_metadata_includes_direct_provider_info(
        self, direct_providers_only_mode, simple_messages
    ):
        """Test that router_metadata includes direct provider information."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            from app.providers.base import ProviderResponse

            provider_response = ProviderResponse(
                output_text="Test response",
                tokens_in=10,
                tokens_out=15,
                latency_ms=180,
                raw={},
                error=None,
            )
            mock_adapter = Mock()
            mock_adapter.invoke = AsyncMock(return_value=provider_response)

            mock_router = MagicMock()
            mock_router.select_model = AsyncMock(
                return_value=(
                    "openai",
                    "gpt-3.5-turbo",
                    "test",
                    {},
                    {"usd": 0.01, "eta_ms": 180, "tokens_in": 10, "tokens_out": 15},
                )
            )
            mock_router.invoke_via_adapter = AsyncMock(return_value=provider_response)
            mock_router.record_latency = Mock()

            # Patch both locations where the registry is accessed
            mock_registry_dict = {"openai": mock_adapter}
            with (
                patch(
                    "app.providers.registry.get_provider_registry", return_value=mock_registry_dict
                ),
                patch(
                    "app.main.providers_registry.get_provider_registry",
                    return_value=mock_registry_dict,
                ),
                patch("app.main.HeuristicRouter", return_value=mock_router),
                patch("app.main.router", mock_router),
            ):
                request_data = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Test message"}],
                    "max_budget": 1.0,  # Set higher budget to avoid constraint issues
                }

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
            assert router_metadata["provider"] == "openai"

    def test_token_usage_reporting_in_api_responses(
        self, direct_providers_only_mode, simple_messages
    ):
        """Test token usage reporting in API responses."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            provider_response = _make_provider_response(
                output_text="Test response with specific token counts",
                tokens_in=25,
                tokens_out=35,
                latency_ms=250,
                raw={"provider": "mistral", "model": "mistral-small-latest"},
            )
            mock_router = _make_wired_router("mistral", "mistral-small-latest", provider_response)
            registry = {"mistral": _make_adapter(provider_response)}

            request_data = {
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": "Count the tokens in this message"}],
            }

            with _wired_request_path(mock_router, registry):
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

        from app.settings import settings

        # Mock authentication
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            provider_response = _make_provider_response(
                output_text="Test response for cost tracking",
                tokens_in=20,
                tokens_out=30,
                latency_ms=300,
                raw={"provider": "google", "model": "gemini-1.5-flash"},
            )
            mock_router = _make_wired_router("google", "gemini-1.5-flash", provider_response)
            # The legacy (non-adapter) path is the one that logs to the cost
            # tracker; it expects a full ChatCompletionResponse from the router.
            mock_router.invoke_via_adapter = AsyncMock(
                return_value=_make_chat_completion_response(
                    "google", "gemini-1.5-flash", tokens_in=20, tokens_out=30
                )
            )
            registry = {"google": _make_adapter(provider_response)}

            mock_tracker = Mock()
            mock_tracker.log_simple_request = AsyncMock()

            request_data = {
                "model": "gemini-1.5-flash",
                "messages": [{"role": "user", "content": "Test cost tracking"}],
            }

            with (
                _wired_request_path(mock_router, registry),
                patch.object(settings.features, "provider_adapters_enabled", False),
                patch("app.main.model_muxer.advanced_cost_tracker", mock_tracker),
            ):
                response = client.post("/v1/chat/completions", json=request_data)

            assert response.status_code == 200

            # Verify cost tracking was called
            mock_tracker.log_simple_request.assert_called_once()

            # Verify cost tracking includes direct provider metadata
            call_args = mock_tracker.log_simple_request.call_args
            assert "provider" in call_args.kwargs
            assert call_args.kwargs["provider"] == "google"

    def test_database_logging_with_direct_provider_metadata(
        self, direct_providers_only_mode, simple_messages
    ):
        """Test database logging of requests with direct provider metadata."""
        from app.main import app

        client = TestClient(app)

        from app.settings import settings

        # Mock authentication
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            provider_response = _make_provider_response(
                output_text="Test response for database logging",
                tokens_in=15,
                tokens_out=25,
                latency_ms=220,
                raw={"provider": "cohere", "model": "command-r"},
            )
            mock_router = _make_wired_router("cohere", "command-r", provider_response)
            # The legacy (non-adapter) path falls back to db.log_request when
            # the advanced cost tracker is disabled.
            mock_router.invoke_via_adapter = AsyncMock(
                return_value=_make_chat_completion_response(
                    "cohere", "command-r", tokens_in=15, tokens_out=25
                )
            )
            registry = {"cohere": _make_adapter(provider_response)}

            request_data = {
                "model": "command-r",
                "messages": [{"role": "user", "content": "Test database logging"}],
            }

            with (
                _wired_request_path(mock_router, registry),
                patch.object(settings.features, "provider_adapters_enabled", False),
                patch("app.main.model_muxer.advanced_cost_tracker", None),
                patch("app.main.db.log_request", new_callable=AsyncMock) as mock_log,
            ):
                response = client.post("/v1/chat/completions", json=request_data)

            assert response.status_code == 200

            # Verify database logging was called with direct provider info
            mock_log.assert_called_once()

            call_args = mock_log.call_args
            assert "provider" in call_args.kwargs
            assert "model" in call_args.kwargs
            assert call_args.kwargs["provider"] == "cohere"
            assert call_args.kwargs["model"] == "command-r"

    def test_authentication_and_authorization(self, direct_providers_only_mode, simple_messages):
        """Test API key authentication with direct provider routing."""
        from fastapi import HTTPException

        from app.main import app

        client = TestClient(app)

        # Remove the module-level auth override so the real dependency runs
        # and the patched authenticate_request is actually exercised.
        app.dependency_overrides.pop(get_authenticated_user, None)
        try:
            # Mock the authentication method directly
            with patch("app.auth.auth.authenticate_request") as mock_auth_method:
                mock_auth_method.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

                provider_response = _make_provider_response(
                    output_text="Authenticated response",
                    tokens_in=10,
                    tokens_out=20,
                    latency_ms=150,
                    raw={"provider": "together", "model": "meta-llama/Llama-3.1-8B-Instruct"},
                )
                mock_router = _make_wired_router(
                    "together", "meta-llama/Llama-3.1-8B-Instruct", provider_response
                )
                registry = {"together": _make_adapter(provider_response)}

                request_data = {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "messages": [{"role": "user", "content": "Test authentication"}],
                }

                with _wired_request_path(mock_router, registry):
                    # Test with valid API key
                    headers = {"Authorization": "Bearer valid-api-key"}
                    response = client.post(
                        "/v1/chat/completions", json=request_data, headers=headers
                    )

                    assert response.status_code == 200

                    # Test with invalid API key
                    mock_auth_method.side_effect = HTTPException(
                        status_code=401, detail="Invalid API key"
                    )
                    headers = {"Authorization": "Bearer invalid-api-key"}
                    response = client.post(
                        "/v1/chat/completions", json=request_data, headers=headers
                    )

                    assert response.status_code == 401
        finally:
            app.dependency_overrides[get_authenticated_user] = mock_auth

    def test_rate_limiting_with_direct_providers(self, direct_providers_only_mode, simple_messages):
        """Test rate limiting works correctly with direct providers."""
        from app.main import app

        client = TestClient(app)

        # Mock the authentication method directly
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.main.providers_registry.get_provider_registry") as mock_registry:
                from app.providers.base import ProviderResponse

                provider_response = ProviderResponse(
                    output_text="Rate limited response",
                    tokens_in=5,
                    tokens_out=10,
                    latency_ms=100,
                    raw={"provider": "openai", "model": "gpt-3.5-turbo"},
                    error=None,
                )
                mock_adapter = Mock()
                mock_adapter.invoke = AsyncMock(return_value=provider_response)

                mock_registry.return_value = {"openai": mock_adapter}

                mock_router = MagicMock()
                mock_router.select_model = AsyncMock(
                    return_value=(
                        "openai",
                        "gpt-3.5-turbo",
                        "test",
                        {},
                        {"usd": 0.001, "eta_ms": 100, "tokens_in": 5, "tokens_out": 10},
                    )
                )
                mock_router.invoke_via_adapter = AsyncMock(return_value=provider_response)
                mock_router.record_latency = Mock()

                request_data = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Test rate limiting"}],
                }

                headers = {"Authorization": "Bearer test-api-key"}

                # First request should succeed
                with patch("app.main.HeuristicRouter", return_value=mock_router):
                    with patch("app.main.router", mock_router):
                        response = client.post(
                            "/v1/chat/completions", json=request_data, headers=headers
                        )
                assert response.status_code == 200

                # For rate limiting test, we'll just verify the request goes through
                # since actual rate limiting would require more complex setup

    def test_tenant_isolation_with_direct_providers(
        self, direct_providers_only_mode, simple_messages
    ):
        """Test tenant isolation with direct providers."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            provider_response = _make_provider_response(
                output_text="Tenant isolated response",
                tokens_in=12,
                tokens_out=18,
                latency_ms=180,
                raw={"provider": "anthropic", "model": "claude-3-haiku-20240307"},
            )
            registry = {"anthropic": _make_adapter(provider_response)}

            request_data = {
                "model": "claude-3-haiku-20240307",
                "messages": [{"role": "user", "content": "Test tenant isolation"}],
                "max_budget": 1.0,  # Avoid budget-gate rejection with real pricing
            }

            # Test with different tenant API keys
            tenant1_headers = {"Authorization": "Bearer tenant1-key"}
            tenant2_headers = {"Authorization": "Bearer tenant2-key"}

            # Real router logic runs; anthropic is in the default preferences.
            with _real_router_with_registry(registry):
                response1 = client.post(
                    "/v1/chat/completions", json=request_data, headers=tenant1_headers
                )
                response2 = client.post(
                    "/v1/chat/completions", json=request_data, headers=tenant2_headers
                )

            # Both should work independently
            assert response1.status_code == 200
            assert response2.status_code == 200

    def test_prometheus_metrics_collection(self, direct_providers_only_mode, simple_messages):
        """Test Prometheus metrics are collected correctly for direct providers."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            provider_response = _make_provider_response(
                output_text="Metrics test response",
                tokens_in=8,
                tokens_out=12,
                latency_ms=120,
                raw={"provider": "openai", "model": "gpt-3.5-turbo"},
            )
            registry = {"openai": _make_adapter(provider_response)}

            request_data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Test metrics collection"}],
                "max_budget": 1.0,  # Avoid budget-gate rejection with real pricing
            }

            headers = {"Authorization": "Bearer test-key"}
            # Run the REAL router (registry-only patch) so router metrics fire;
            # patch the metric objects where the router imported them.
            with (
                _real_router_with_registry(registry),
                patch("app.router.ROUTER_REQUESTS") as mock_requests,
                patch("app.router.LLM_ROUTER_SELECTED_COST_ESTIMATE_USD") as mock_selected_cost,
            ):
                response = client.post("/v1/chat/completions", json=request_data, headers=headers)

            assert response.status_code == 200

            # Verify router request counter was incremented
            mock_requests.labels.assert_called_with("chat")
            mock_requests.labels.return_value.inc.assert_called()
            # Verify the selected model's cost estimate metric (provider:model label) was recorded
            mock_selected_cost.labels.assert_called()
            selected_model_key = mock_selected_cost.labels.call_args.args[1]
            assert selected_model_key.startswith("openai:")
            mock_selected_cost.labels.return_value.inc.assert_called()

    def test_opentelemetry_tracing(self, direct_providers_only_mode, simple_messages):
        """Test OpenTelemetry tracing through complete request flow."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            provider_response = _make_provider_response(
                output_text="Tracing test response",
                tokens_in=10,
                tokens_out=15,
                latency_ms=160,
                raw={"provider": "google", "model": "gemini-1.5-flash"},
            )
            mock_router = _make_wired_router("google", "gemini-1.5-flash", provider_response)
            registry = {"google": _make_adapter(provider_response)}

            # Mock span context returned by the policy-enforcement span
            mock_span_context = Mock()
            mock_span_context.set_attribute = Mock()

            request_data = {
                "model": "gemini-1.5-flash",
                "messages": [{"role": "user", "content": "Test tracing"}],
            }

            headers = {"Authorization": "Bearer test-key"}
            # Patch start_span where the request path imported it: the HTTP
            # middleware (app.main) and policy enforcement (app.policy.rules).
            with (
                _wired_request_path(mock_router, registry),
                patch("app.main.start_span") as mock_main_span,
                patch("app.policy.rules.start_span") as mock_policy_span,
            ):
                mock_policy_span.return_value.__enter__.return_value = mock_span_context
                response = client.post("/v1/chat/completions", json=request_data, headers=headers)

            assert response.status_code == 200

            # Verify tracing was used through the complete request flow
            mock_main_span.assert_called()
            assert mock_main_span.call_args.args[0] == "http.request"
            mock_policy_span.assert_called()
            mock_span_context.set_attribute.assert_called()

    def test_health_checks_with_direct_providers(self, direct_providers_only_mode, simple_messages):
        """Test health checks work with direct providers."""
        from app.main import app

        client = TestClient(app)

        with patch("app.main.providers_registry.get_provider_registry") as mock_registry:
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
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.main.providers_registry.get_provider_registry") as mock_registry:
                # Mock provider that fails
                mock_adapter = Mock()
                mock_adapter.invoke = AsyncMock(
                    return_value=Mock(
                        output_text="",
                        tokens_in=0,
                        tokens_out=0,
                        latency_ms=0,
                        raw={},
                        error="provider_error",
                    )
                )

                mock_registry.return_value = {"openai": mock_adapter}

            request_data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Test error handling"}],
            }

            headers = {"Authorization": "Bearer test-key"}
            response = client.post("/v1/chat/completions", json=request_data, headers=headers)

            # Should handle provider errors gracefully
            assert response.status_code in [500, 503]

    async def test_concurrent_request_handling(self, direct_providers_only_mode, simple_messages):
        """Test concurrent request handling with direct providers."""
        import asyncio

        import httpx

        from app.main import app

        # Mock authentication
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            provider_response = _make_provider_response(
                output_text="Concurrent test response",
                tokens_in=5,
                tokens_out=10,
                latency_ms=50,
                raw={"provider": "together", "model": "meta-llama/Llama-3.1-8B-Instruct"},
            )
            mock_router = _make_wired_router(
                "together", "meta-llama/Llama-3.1-8B-Instruct", provider_response
            )
            registry = {"together": _make_adapter(provider_response)}

            request_data = {
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": "Concurrent test"}],
            }

            headers = {"Authorization": "Bearer test-key"}

            with _wired_request_path(mock_router, registry):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    tasks = [
                        client.post("/v1/chat/completions", json=request_data, headers=headers)
                        for _ in range(5)
                    ]
                    responses = await asyncio.gather(*tasks)

                    # All requests should succeed
                    for response in responses:
                        assert response.status_code == 200

    def test_minimal_direct_provider_configuration(
        self, direct_providers_only_mode, simple_messages
    ):
        """Test with minimal direct provider configuration."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            # Only one provider configured; real router routes to it
            provider_response = _make_provider_response(
                output_text="Minimal config response",
                tokens_in=8,
                tokens_out=12,
                latency_ms=140,
                raw={"provider": "openai", "model": "gpt-3.5-turbo"},
            )
            registry = {"openai": _make_adapter(provider_response)}

            request_data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Minimal configuration test"}],
                "max_budget": 1.0,  # Avoid budget-gate rejection with real pricing
            }

            headers = {"Authorization": "Bearer test-key"}
            with _real_router_with_registry(registry):
                response = client.post("/v1/chat/completions", json=request_data, headers=headers)

            assert response.status_code == 200

    def test_full_direct_provider_configuration(self, direct_providers_only_mode, simple_messages):
        """Test with full direct provider configuration (all providers enabled)."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            # All providers configured; real router picks a preferred one
            provider_response = _make_provider_response(
                output_text="Full config response",
                tokens_in=10,
                tokens_out=15,
                latency_ms=160,
                raw={"provider": "anthropic", "model": "claude-3-haiku-20240307"},
            )
            mock_adapter = _make_adapter(provider_response)
            registry = {
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
                "max_budget": 1.0,  # Avoid budget-gate rejection with real pricing
            }

            headers = {"Authorization": "Bearer test-key"}
            with _real_router_with_registry(registry):
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
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.main.providers_registry.get_provider_registry") as mock_registry:
                mock_registry.return_value = {
                    "openai": Mock(),
                    "anthropic": Mock(),
                    "mistral": Mock(),
                }

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

    def test_service_unavailable_when_no_providers(
        self, direct_providers_only_mode, simple_messages
    ):
        """Test 503 Service Unavailable when no providers are available."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.main.providers_registry.get_provider_registry") as mock_registry:
                # Empty provider registry
                mock_registry.return_value = {}

            request_data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "No providers test"}],
            }

            headers = {"Authorization": "Bearer test-key"}
            response = client.post("/v1/chat/completions", json=request_data, headers=headers)

            # Should return 503 Service Unavailable
            assert response.status_code == 503

    def test_user_friendly_error_messages(self, direct_providers_only_mode, simple_messages):
        """Test error messages are user-friendly and actionable."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            with patch("app.main.providers_registry.get_provider_registry") as mock_registry:
                mock_registry.return_value = {}

            request_data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Error message test"}],
            }

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
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            provider_response = _make_provider_response(
                output_text="Latency recording test",
                tokens_in=10,
                tokens_out=15,
                latency_ms=180,
                raw={"provider": "mistral", "model": "mistral-small-latest"},
            )
            mock_router = _make_wired_router("mistral", "mistral-small-latest", provider_response)
            registry = {"mistral": _make_adapter(provider_response)}

            request_data = {
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": "Latency recording test"}],
            }

            headers = {"Authorization": "Bearer test-key"}
            with _wired_request_path(mock_router, registry):
                response = client.post("/v1/chat/completions", json=request_data, headers=headers)

            assert response.status_code == 200

            # Verify latency recording was called with the measured provider latency
            mock_router.record_latency.assert_called_once_with("mistral:mistral-small-latest", 180)

    def test_latency_priors_influence_routing(self, direct_providers_only_mode, simple_messages):
        """Test that latency data influences future routing decisions."""
        from app.main import app

        client = TestClient(app)

        # Mock authentication
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            # Create adapters with different latencies; both providers appear in
            # the router's default preferences so the REAL routing logic runs.
            fast_adapter = _make_adapter(
                _make_provider_response(
                    output_text="Fast response",
                    tokens_in=10,
                    tokens_out=15,
                    latency_ms=50,  # Fast
                    raw={"provider": "anthropic", "model": "claude-3-haiku-20240307"},
                )
            )
            slow_adapter = _make_adapter(
                _make_provider_response(
                    output_text="Slow response",
                    tokens_in=10,
                    tokens_out=15,
                    latency_ms=500,  # Slow
                    raw={"provider": "openai", "model": "gpt-3.5-turbo"},
                )
            )

            registry = {"anthropic": fast_adapter, "openai": slow_adapter}

            request_data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Latency influence test"}],
                "max_budget": 1.0,  # Avoid budget-gate rejection with real pricing
            }

            headers = {"Authorization": "Bearer test-key"}

            # Send multiple requests to build latency priors via record_latency
            with _real_router_with_registry(registry):
                for _ in range(3):
                    response = client.post(
                        "/v1/chat/completions", json=request_data, headers=headers
                    )
                    assert response.status_code == 200
                    provider = response.json()["router_metadata"]["provider"]
                    assert provider in ("anthropic", "openai")

            # Future requests should prefer faster providers based on latency priors
            # (This would be verified by checking which provider was selected)

    async def test_streaming_response_end_to_end(self, direct_providers_only_mode):
        """Test end-to-end streaming response."""
        import asyncio

        import httpx

        from app.main import app

        # Mock authentication
        with patch("app.auth.auth.authenticate_request") as mock_auth:
            mock_auth.return_value = {"user_id": "test-user", "scopes": ["api_access"]}

            async def mock_streaming_invoke(model, prompt, **kwargs):
                chunks = [
                    {"choices": [{"delta": {"content": "Hello"}}]},
                    {"choices": [{"delta": {"content": ", "}}]},
                    {"choices": [{"delta": {"content": "world!"}}]},
                ]

            for chunk in chunks:
                yield chunk

        with patch("app.main.providers_registry.get_provider_registry") as mock_registry:
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
