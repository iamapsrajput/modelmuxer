# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
ModelMuxer main application.

This module provides the ModelMuxer FastAPI application with intelligent
LLM routing across multiple providers, cost estimation with budget gating,
cost tracking, and observability.
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import app.providers.registry as providers_registry
from app.core.exceptions import BudgetExceededError
from app.policy.rules import enforce_policies  # resolved via app.main by route modules
from app.telemetry.logging import configure_json_logging
from app.telemetry.metrics import HTTP_LATENCY as HTTP_LATENCY
from app.telemetry.metrics import HTTP_REQUESTS_TOTAL as HTTP_REQUESTS
from app.telemetry.tracing import get_trace_id, setup_tracing, start_span

# Core imports
from .auth import SecurityHeaders, auth, sanitize_user_input, validate_request_size
from .cost_tracker import cost_tracker
from .database import db
from .models import ErrorResponse
from .providers.base import ProviderError
from .router import HeuristicRouter
from .settings import settings as app_settings

try:  # optional dependency in test/basic envs
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
except Exception:  # pragma: no cover
    CONTENT_TYPE_LATEST = None  # type: ignore[assignment]
    generate_latest = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)

# Global providers registry is now managed in app.providers.registry
# Use get_provider_registry() to access provider adapters

# Global router instance
router = None


class ModelMuxer:
    """
    Central ModelMuxer orchestrator.

    Holds shared application services: configuration and cost tracking.
    Providers are managed through the registry system and routing through
    the global HeuristicRouter instance.
    """

    def __init__(self) -> None:
        self.config = app_settings
        self.cost_tracker: Any = cost_tracker
        self.advanced_cost_tracker: Any = None
        self._initialize_cost_tracking()

    def _initialize_cost_tracking(self) -> None:
        """Initialize the advanced cost tracker (budgets, analytics).

        Falls back to the basic cost tracker when initialization fails;
        Redis is optional and degrades to an in-memory mock internally.
        """
        try:
            # Import here to avoid circular imports
            from .cost_tracker import create_advanced_cost_tracker

            redis_url = app_settings.cache.redis_url or "redis://localhost:6379/0"
            self.advanced_cost_tracker = create_advanced_cost_tracker(
                db_path="cost_tracker_enhanced.db", redis_url=redis_url
            )
            self.cost_tracker = self.advanced_cost_tracker
            logger.info("cost_tracker_initialized", redis_url=redis_url)
        except Exception as e:
            logger.warning("cost_tracker_init_failed", error=str(e))
            # Fallback to basic cost tracker
            self.cost_tracker = cost_tracker


# Global model muxer instance
model_muxer = ModelMuxer()

# Import load_price_table for test compatibility
from .core.costing import load_price_table


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    # Starting ModelMuxer LLM Router (logged via structlog if available)

    # Initialize database
    await db.init_database()
    # Database initialized (logged via structlog if available)

    # Initialize providers through the registry system
    # The registry automatically detects and initializes providers based on available API keys
    current_providers = providers_registry.get_provider_registry()

    if not current_providers:
        error_msg = (
            "No providers initialized! ModelMuxer requires at least one configured provider. "
            "Please ensure you have valid API keys configured in your environment."
        )
        logger.error(
            error_msg,
            extra={
                "provider_config_help": {
                    "openai": "Set OPENAI_API_KEY=sk-... for GPT models",
                    "anthropic": "Set ANTHROPIC_API_KEY=sk-ant-... for Claude models",
                    "mistral": "Set MISTRAL_API_KEY=... for Mistral models",
                    "google": "Set GOOGLE_API_KEY=... for Gemini models",
                    "groq": "Set GROQ_API_KEY=gsk_... for Groq models",
                    "together": "Set TOGETHER_API_KEY=... for Together AI models",
                    "cohere": "Set COHERE_API_KEY=... for Cohere models",
                },
                "note": "API keys should not contain placeholder text like 'your-key-here'",
            },
        )

        # Hard fail in production mode if no providers are available
        if app_settings.features.mode == "production":
            from app.core.exceptions import ConfigurationError

            raise ConfigurationError(error_msg)
    else:
        logger.info(
            "Initialized %d providers: %s", len(current_providers), list(current_providers.keys())
        )

    # Initialize router with provider registry function
    global router
    router = HeuristicRouter(provider_registry_fn=providers_registry.get_provider_registry)

    # Validate price table completeness for router preferences
    try:
        router._validate_model_keys()
        logger.info("Router configuration validation passed")
    except Exception as e:
        error_msg = f"Router configuration validation failed: {e}"
        logger.error(error_msg)
        if app_settings.features.mode == "production":
            logger.error("Router configuration errors are not allowed in production mode")
            raise RuntimeError(f"Startup validation failed: {error_msg}") from e
        else:
            logger.warning(
                "Router configuration issues detected - continuing in non-production mode"
            )

    # Router ready with providers (logged via structlog if available)

    yield

    # Shutdown
    # Shutting down ModelMuxer (logged via structlog if available)

    # Clean up provider registry adapters
    try:
        current_registry = providers_registry.get_provider_registry()
        logger.info("Starting cleanup of %d provider adapters", len(current_registry))
        await providers_registry.cleanup_provider_registry()
        logger.info("Provider registry cleanup completed successfully")
    except Exception as e:
        if logger:
            logger.warning("Failed to cleanup provider registry", error=str(e))

    # Cleanup complete (logged via structlog if available)


# Initialize tracing and logging
try:
    if app_settings.observability.enable_tracing:
        setup_tracing(
            service_name="modelmuxer",
            sampling_ratio=app_settings.observability.sampling_ratio,
            otlp_endpoint=(
                str(app_settings.observability.otel_exporter_otlp_endpoint)
                if app_settings.observability.otel_exporter_otlp_endpoint
                else None
            ),
        )
    configure_json_logging(app_settings.observability.log_level)
except Exception:
    pass

# Create FastAPI app
app = FastAPI(
    title="ModelMuxer - LLM Router API",
    description="Intelligent LLM routing service that optimizes cost and quality",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware with secure configuration from centralized settings
allowed_origins = app_settings.observability.cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Secure: specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Specific methods only
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-API-Key",
        "X-Request-ID",
    ],  # Specific headers only
)


# Custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse.create(
            message=f"Invalid request: {str(exc)}",
            error_type="invalid_request_error",
            code="validation_error",
        ).dict(),
    )


@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=(
            exc.detail
            if isinstance(exc.detail, dict)
            else ErrorResponse.create(message=str(exc.detail), error_type="http_error").dict()
        ),
        headers=getattr(exc, "headers", None),
    )


@app.middleware("http")
async def add_security_headers(request: Request, call_next) -> None:
    """Add security headers to all responses."""
    response = await call_next(request)

    # Add security headers
    for header, value in SecurityHeaders.get_security_headers().items():
        response.headers[header] = value

    return response


@app.middleware("http")
async def validate_request_middleware(request: Request, call_next) -> None:
    """Validate request size and other security checks."""
    # Skip validation for health check
    if hasattr(request.url, "path") and request.url.path == "/health":
        return await call_next(request)

    # Validate request size
    await validate_request_size(request)

    return await call_next(request)


@app.middleware("http")
async def request_observability_middleware(request: Request, call_next):
    route = request.scope.get("route").path if request.scope.get("route") else request.url.path
    method = request.method
    start_ts = time.perf_counter()
    with start_span("http.request", route=route, method=method):
        try:
            response = await call_next(request)
            status = str(response.status_code)
        except HTTPException as http_exc:  # type: ignore[name-defined]
            # Allow FastAPI exception handlers to process HTTPException
            status = str(http_exc.status_code)
            raise
        except (ProviderError, BudgetExceededError):
            # Let specific errors pass through to application handlers
            raise
        except Exception:
            status = "500"
            response = JSONResponse({"error": {"message": "internal error"}}, status_code=500)
        finally:
            elapsed_ms = (time.perf_counter() - start_ts) * 1000
            try:
                HTTP_REQUESTS.labels(route, method, status).inc()
                HTTP_LATENCY.labels(route, method).observe(elapsed_ms)
            except Exception:
                pass
    trace_id = get_trace_id()
    if trace_id:
        response.headers["x-trace-id"] = trace_id
    return response


async def get_authenticated_user(
    request: Request, authorization: str | None = Header(None)
) -> dict[str, Any]:
    """Dependency to authenticate requests.

    Supports tests that patch `auth.authenticate_request` with a synchronous mock
    by detecting awaitability at runtime.
    """
    # Dynamically resolve the current auth instance so test patches apply
    try:
        from importlib import import_module

        _auth_mod = import_module("app.auth")
        _auth_inst = getattr(_auth_mod, "auth", auth)
    except Exception:
        _auth_inst = auth
    result = _auth_inst.authenticate_request(request, authorization)
    try:
        import inspect

        if inspect.isawaitable(result):
            return await result  # type: ignore[no-any-return]
        return result  # type: ignore[return-value]
    except Exception as e:
        # Normalize unexpected auth errors to 401 for tests
        raise HTTPException(
            status_code=401,
            detail=ErrorResponse.create(
                message=str(e) or "Invalid API key provided.",
                error_type="authentication_error",
                code="invalid_api_key",
            ).dict(),
        ) from e


# Route modules are imported after the app, singletons, and
# get_authenticated_user are defined so they can bind to this module safely.
from app.api.routes import analytics as _analytics_routes  # noqa: E402
from app.api.routes import chat as _chat_routes  # noqa: E402
from app.api.routes import providers as _providers_routes  # noqa: E402
from app.api.routes import system as _system_routes  # noqa: E402

app.include_router(_system_routes.router)
app.include_router(_chat_routes.router)
app.include_router(_analytics_routes.router)
app.include_router(_providers_routes.router)

DASHBOARD_DIR = Path(__file__).resolve().parent / "static" / "dashboard"
if DASHBOARD_DIR.is_dir():
    app.mount(
        "/dashboard",
        StaticFiles(directory=str(DASHBOARD_DIR), html=True),
        name="dashboard",
    )

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard_root() -> RedirectResponse:
        return RedirectResponse(url="/dashboard/")

else:
    logger.warning("dashboard_static_dir_missing", path=str(DASHBOARD_DIR))

# Re-export handlers so existing imports and patch targets (app.main.<name>)
# keep working after the split into app/api/routes/.
chat_completions = _chat_routes.chat_completions
stream_chat_completion = _chat_routes.stream_chat_completion
anthropic_messages = _chat_routes.anthropic_messages
prometheus_metrics = _system_routes.prometheus_metrics
health_check = _system_routes.health_check
get_metrics = _system_routes.get_metrics
get_user_stats = _analytics_routes.get_user_stats
get_cost_analytics = _analytics_routes.get_cost_analytics
get_routing_analytics = _analytics_routes.get_routing_analytics
get_budget_status = _analytics_routes.get_budget_status
set_budget = _analytics_routes.set_budget
get_providers = _providers_routes.get_providers
list_models = _providers_routes.list_models


def cli():
    """CLI entry point for ModelMuxer."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="ModelMuxer LLM Router")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: localhost; use 0.0.0.0 for external access)",
    )  # safer default
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument(
        "--mode",
        choices=["basic", "production"],
        default="basic",
        help="Deployment mode (production enables strict validation)",
    )

    args = parser.parse_args()

    # Set mode environment variable
    os.environ["MODELMUXER_MODE"] = args.mode

    # Starting ModelMuxer in specified mode (logged via structlog if available)

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


def main():
    """Main entry point for ModelMuxer server."""
    import uvicorn

    config = model_muxer.config
    uvicorn.run(
        "app.main:app",
        host=config.host if hasattr(config, "host") else "127.0.0.1",  # safer default
        port=config.port if hasattr(config, "port") else 8000,
        reload=config.debug if hasattr(config, "debug") else False,
        log_level=(
            config.logging.level.lower()
            if hasattr(config, "logging") and hasattr(config.logging, "level")
            else "info"
        ),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app", host=app_settings.host, port=app_settings.port, reload=app_settings.debug
    )
