# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
ModelMuxer main application with comprehensive features.

This module provides the complete ModelMuxer application with both basic and advanced features,
including intelligent routing, multiple providers, ML-based classification, comprehensive
monitoring, cost tracking, and enterprise features. The application automatically detects
the deployment mode and enables appropriate features.
"""

import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Optional prometheus import
try:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"

    def generate_latest() -> str:
        """Fallback when prometheus is not available."""
        return "# Prometheus not available\n"


# Core imports
from .auth import SecurityHeaders, auth, sanitize_user_input, validate_request_size
from .config import settings
from .cost_tracker import cost_tracker
from .database import db
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    UserStats,
)
from .providers import AnthropicProvider, MistralProvider, OpenAIProvider
from .providers.base import AuthenticationError, ProviderError, RateLimitError
from .router import router

# Enhanced imports (optional - loaded based on availability and configuration)
try:
    # Cache imports
    from .cache.memory_cache import MemoryCache
    from .cache.redis_cache import RedisCache

    # Classification imports
    from .classification.embeddings import EmbeddingManager
    from .classification.prompt_classifier import PromptClassifier
    from .config import enhanced_config

    # Middleware imports
    from .middleware.auth_middleware import AuthMiddleware
    from .middleware.logging_middleware import LoggingMiddleware
    from .middleware.rate_limit_middleware import RateLimitMiddleware

    # Monitoring imports
    from .monitoring.health_checker import HealthChecker
    from .monitoring.metrics_collector import MetricsCollector

    # Routing imports
    from .routing.cascade_router import CascadeRouter
    from .routing.heuristic_router import HeuristicRouter
    from .routing.hybrid_router import HybridRouter
    from .routing.semantic_router import SemanticRouter

    ENHANCED_FEATURES_AVAILABLE = True
    logger = structlog.get_logger(__name__)

except ImportError as e:
    ENHANCED_FEATURES_AVAILABLE = False
    logger = None
    enhanced_config = None
    print(f"Enhanced features not available: {e}")
    print("Running in basic mode...")

# Determine if we should run in enhanced mode
ENHANCED_MODE = (
    ENHANCED_FEATURES_AVAILABLE
    and os.getenv("MODELMUXER_MODE", "basic").lower() in ["enhanced", "production"]
    and enhanced_config is not None
)

# Provider instances
providers = {}


class ModelMuxer:
    """
    Unified ModelMuxer with both basic and enhanced features.

    Automatically detects deployment mode and enables appropriate features.
    Provides intelligent LLM routing with multiple providers, optional ML-based
    classification, monitoring, caching, authentication, and rate limiting.
    """

    def __init__(self, enhanced_mode: bool = None) -> None:
        # Determine mode
        if enhanced_mode is None:
            enhanced_mode = ENHANCED_MODE

        self.enhanced_mode = enhanced_mode
        self.config = enhanced_config if enhanced_mode and enhanced_config else settings

        # Initialize components
        self.providers: dict[str, Any] = {}
        self.routers: dict[str, Any] = {}
        self.cache: Any = None
        self.embedding_manager: Any = None
        self.classifier: Any = None
        self.metrics_collector: Any = None
        self.health_checker: Any = None
        self.cost_tracker: Any = None
        self.advanced_cost_tracker: Any = None

        # Middleware (enhanced mode only)
        self.auth_middleware: Any = None
        self.rate_limit_middleware: Any = None
        self.logging_middleware: Any = None

        # Initialize components based on mode
        if self.enhanced_mode:
            self._initialize_enhanced_components()
        else:
            self._initialize_basic_components()

    def _initialize_basic_components(self) -> None:
        """Initialize basic components for simple deployment."""
        # Use global cost_tracker for basic mode
        self.cost_tracker = cost_tracker
        print("✅ Basic ModelMuxer initialized")

    def _initialize_enhanced_components(self) -> None:
        """Initialize all enhanced components."""
        if not ENHANCED_FEATURES_AVAILABLE:
            print("⚠️ Enhanced features requested but not available, falling back to basic mode")
            self._initialize_basic_components()
            return

        try:
            self._initialize_cache()
            self._initialize_classification()
            self._initialize_routing()
            self._initialize_cost_tracking()
            self._initialize_monitoring()
            self._initialize_middleware()

            if logger:
                logger.info(
                    "enhanced_modelmuxer_initialized",
                    cache_enabled=self.cache is not None,
                    classification_enabled=self.classifier is not None,
                    monitoring_enabled=self.metrics_collector is not None,
                )
            print("✅ Enhanced ModelMuxer initialized")

        except Exception as e:
            print(f"⚠️ Enhanced initialization failed: {e}")
            print("Falling back to basic mode...")
            self._initialize_basic_components()

    def _initialize_cache(self) -> None:
        """Initialize caching system."""
        if not hasattr(self.config, "cache") or not self.config.cache.enabled:
            return

        try:
            if self.config.cache.backend == "redis":
                # Parse Redis URL for connection details
                import urllib.parse

                parsed = urllib.parse.urlparse(self.config.cache.redis_url)

                self.cache = RedisCache(
                    host=parsed.hostname or "localhost",
                    port=parsed.port or 6379,
                    db=self.config.cache.redis_db,
                    password=parsed.password,
                    default_ttl=self.config.cache.default_ttl,
                )
            else:
                self.cache = MemoryCache(
                    max_size=self.config.cache.memory_max_size,
                    max_memory_mb=self.config.cache.memory_max_memory_mb,
                    default_ttl=self.config.cache.default_ttl,
                )

            if logger:
                logger.info("cache_initialized", cache_type=self.config.cache.backend)
        except Exception as e:
            if logger:
                logger.warning("cache_init_failed", error=str(e))
            # Fallback to memory cache with basic settings
            self.cache = MemoryCache(max_size=1000, default_ttl=300)

    def _initialize_classification(self) -> None:
        """Initialize ML classification system."""
        if not hasattr(self.config, "classification") or not self.config.classification.enabled:
            return

        try:
            self.embedding_manager = EmbeddingManager(
                model_name=self.config.classification.embedding_model,
                cache_dir=self.config.classification.cache_dir,
            )

            self.classifier = PromptClassifier(
                embedding_manager=self.embedding_manager,
                config=self.config.classification,
            )

            if logger:
                logger.info("classification_initialized")
        except Exception as e:
            if logger:
                logger.warning("classification_init_failed", error=str(e))

    def _initialize_routing(self) -> None:
        """Initialize advanced routing system."""
        if not hasattr(self.config, "routing"):
            return

        try:
            # Initialize different router types
            if hasattr(self.config.routing, "heuristic") and self.config.routing.heuristic.enabled:
                self.routers["heuristic"] = HeuristicRouter(self.config.routing.heuristic.dict())

            if hasattr(self.config.routing, "semantic") and self.config.routing.semantic.enabled:
                self.routers["semantic"] = SemanticRouter(
                    embedding_manager=self.embedding_manager,
                    config=self.config.routing.semantic.dict(),
                )

            if hasattr(self.config.routing, "cascade") and self.config.routing.cascade.enabled:
                self.routers["cascade"] = CascadeRouter(self.config.routing.cascade.dict())

            if hasattr(self.config.routing, "hybrid") and self.config.routing.hybrid.enabled:
                self.routers["hybrid"] = HybridRouter(
                    heuristic_router=self.routers.get("heuristic"),
                    semantic_router=self.routers.get("semantic"),
                    config=self.config.routing.hybrid.dict(),
                )

            if logger:
                logger.info("routing_initialized", routers=list(self.routers.keys()))
        except Exception as e:
            if logger:
                logger.warning("routing_init_failed", error=str(e))

    def _initialize_cost_tracking(self) -> None:
        """Initialize enhanced cost tracking system."""
        if not ENHANCED_FEATURES_AVAILABLE:
            self.cost_tracker = cost_tracker
            return

        try:
            # Import here to avoid circular imports
            from .cost_tracker import create_advanced_cost_tracker

            # Use Redis URL from cache config if available
            redis_url = "redis://localhost:6379/0"
            if hasattr(self.config, "cache") and hasattr(self.config.cache, "redis_url"):
                redis_url = self.config.cache.redis_url
            elif hasattr(self.config, "cache"):
                redis_url = f"redis://{self.config.cache.redis_host}:{self.config.cache.redis_port}/{self.config.cache.redis_db}"

            self.advanced_cost_tracker = create_advanced_cost_tracker(
                db_path="cost_tracker.db", redis_url=redis_url
            )
            self.cost_tracker = self.advanced_cost_tracker  # For backward compatibility
            if logger:
                logger.info("enhanced_cost_tracker_initialized")
        except Exception as e:
            if logger:
                logger.warning("cost_tracker_init_failed", error=str(e))
            # Fallback to basic cost tracker
            self.cost_tracker = cost_tracker

    def _initialize_monitoring(self) -> None:
        """Initialize monitoring and metrics collection."""
        if not hasattr(self.config, "monitoring") or not self.config.monitoring.enabled:
            return

        try:
            self.metrics_collector = MetricsCollector(
                enabled=self.config.monitoring.metrics_enabled,
                prometheus_enabled=PROMETHEUS_AVAILABLE
                and self.config.monitoring.prometheus_enabled,
            )

            self.health_checker = HealthChecker(
                check_interval=self.config.monitoring.health_check_interval,
                providers=self.providers,
            )

            if logger:
                logger.info("monitoring_initialized")
        except Exception as e:
            if logger:
                logger.warning("monitoring_init_failed", error=str(e))

    def _initialize_middleware(self) -> None:
        """Initialize middleware components."""
        if not hasattr(self.config, "middleware"):
            return

        try:
            if hasattr(self.config.middleware, "auth") and self.config.middleware.auth.enabled:
                self.auth_middleware = AuthMiddleware(self.config.middleware.auth)

            if (
                hasattr(self.config.middleware, "rate_limit")
                and self.config.middleware.rate_limit.enabled
            ):
                self.rate_limit_middleware = RateLimitMiddleware(self.config.middleware.rate_limit)

            if (
                hasattr(self.config.middleware, "logging")
                and self.config.middleware.logging.enabled
            ):
                self.logging_middleware = LoggingMiddleware(self.config.middleware.logging)

            if logger:
                logger.info("middleware_initialized")
        except Exception as e:
            if logger:
                logger.warning("middleware_init_failed", error=str(e))


# Global model muxer instance
model_muxer = ModelMuxer()


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """Application lifespan manager."""
    # Startup
    print("🚀 Starting ModelMuxer LLM Router...")

    # Initialize database
    await db.init_database()
    print("✅ Database initialized")

    # Initialize providers with API keys from settings
    import os

    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and not openai_key.startswith("your-") and not openai_key.endswith("-here"):
        try:
            providers["openai"] = OpenAIProvider(api_key=openai_key)
            print("✅ OpenAI provider initialized")
        except Exception as e:
            print(f"⚠️  OpenAI provider failed to initialize: {e}")

    # Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if (
        anthropic_key
        and not anthropic_key.startswith("your-")
        and not anthropic_key.endswith("-here")
    ):
        try:
            providers["anthropic"] = AnthropicProvider(api_key=anthropic_key)
            print("✅ Anthropic provider initialized")
        except Exception as e:
            print(f"⚠️  Anthropic provider failed to initialize: {e}")

    # Mistral
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if mistral_key and not mistral_key.startswith("your-") and not mistral_key.endswith("-here"):
        try:
            providers["mistral"] = MistralProvider(api_key=mistral_key)
            print("✅ Mistral provider initialized")
        except Exception as e:
            print(f"⚠️  Mistral provider failed to initialize: {e}")

    if not providers:
        print("❌ No providers initialized! Check your API keys.")
        print("💡 Make sure you have valid API keys in your .env file:")
        print("   - OPENAI_API_KEY=sk-...")
        print("   - ANTHROPIC_API_KEY=sk-ant-...")
        print("   - MISTRAL_API_KEY=...")
        print("   (Keys should not contain placeholder text like 'your-key-here')")

    print(f"🎯 Router ready with {len(providers)} providers")

    yield

    # Shutdown
    print("🛑 Shutting down ModelMuxer...")
    for provider in providers.values():
        if hasattr(provider, "client"):
            await provider.client.aclose()
    print("✅ Cleanup complete")


# Create FastAPI app
app = FastAPI(
    title="ModelMuxer - LLM Router API",
    description="Intelligent LLM routing service that optimizes cost and quality",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware with secure configuration
# Get allowed origins from environment or use secure defaults
allowed_origins = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:8080,https://modelmuxer.com"
).split(",")

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
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> None:
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
async def http_exception_handler(request: Request, exc: HTTPException) -> None:
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
    if request.url.path == "/health":
        return await call_next(request)

    # Validate request size
    validate_request_size(request)

    return await call_next(request)


async def get_authenticated_user(
    request: Request, authorization: str | None = Header(None)
) -> dict[str, Any]:
    """Dependency to authenticate requests."""
    return await auth.authenticate_request(request, authorization)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest, user_info: dict[str, Any] = Depends(get_authenticated_user)
):
    """
    Create a chat completion using the optimal LLM provider.

    This endpoint is compatible with OpenAI's chat completions API.
    """
    start_time = time.time()
    user_id = user_info["user_id"]

    try:
        # Sanitize input messages
        for message in request.messages:
            message.content = sanitize_user_input(message.content)

        # Route the request to the best provider/model
        provider_name, model_name, routing_reason = router.select_model(
            messages=request.messages, user_id=user_id
        )

        # Check if provider is available
        if provider_name not in providers:
            raise HTTPException(
                status_code=503,
                detail=ErrorResponse.create(
                    message=f"Provider {provider_name} is not available",
                    error_type="service_unavailable",
                    code="provider_unavailable",
                ).dict(),
            )

        provider = providers[provider_name]

        # Estimate cost and check budget
        cost_estimate = cost_tracker.estimate_request_cost(
            messages=request.messages,
            provider=provider_name,
            model=model_name,
            max_tokens=request.max_tokens,
        )

        budget_check = await db.check_budget(user_id, cost_estimate["estimated_cost"])
        if not budget_check["allowed"]:
            raise HTTPException(
                status_code=402,
                detail=ErrorResponse.create(
                    message=f"Budget exceeded: {budget_check['reason']}",
                    error_type="budget_exceeded",
                    code="insufficient_budget",
                ).dict(),
            )

        # Handle streaming vs non-streaming
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(
                    provider, request, model_name, routing_reason, user_id, start_time
                ),
                media_type="text/plain",
            )
        else:
            # Make the request to the provider
            response = await provider.chat_completion(
                messages=request.messages,
                model=model_name,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                **{
                    k: v
                    for k, v in request.dict().items()
                    if k not in ["messages", "model", "max_tokens", "temperature", "stream"]
                    and v is not None
                },
            )

            # Update routing reason in metadata
            response.router_metadata.routing_reason = routing_reason

            # Log the request
            await db.log_request(
                user_id=user_id,
                provider=provider_name,
                model=model_name,
                messages=[msg.dict() for msg in request.messages],
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cost=response.router_metadata.estimated_cost,
                response_time_ms=response.router_metadata.response_time_ms,
                routing_reason=routing_reason,
                success=True,
            )

            return response

    except ProviderError as e:
        # Log failed request
        await db.log_request(
            user_id=user_id,
            provider=provider_name if "provider_name" in locals() else "unknown",
            model=model_name if "model_name" in locals() else "unknown",
            messages=[msg.dict() for msg in request.messages],
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            response_time_ms=(time.time() - start_time) * 1000,
            routing_reason=routing_reason if "routing_reason" in locals() else "error",
            success=False,
            error_message=str(e),
        )

        if isinstance(e, AuthenticationError):
            status_code = 401
            error_type = "authentication_error"
        elif isinstance(e, RateLimitError):
            status_code = 429
            error_type = "rate_limit_error"
        else:
            status_code = 502
            error_type = "provider_error"

        raise HTTPException(
            status_code=status_code,
            detail=ErrorResponse.create(
                message=f"Provider error: {str(e)}", error_type=error_type, code="provider_error"
            ).dict(),
        ) from e

    except Exception as e:
        # Log unexpected error
        await db.log_request(
            user_id=user_id,
            provider="unknown",
            model="unknown",
            messages=[msg.dict() for msg in request.messages],
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            response_time_ms=(time.time() - start_time) * 1000,
            routing_reason="error",
            success=False,
            error_message=str(e),
        )

        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.create(
                message="Internal server error",
                error_type="internal_error",
                code="internal_server_error",
            ).dict(),
        ) from e


async def stream_chat_completion(
    provider, request, model_name, routing_reason, user_id, start_time
):
    """Handle streaming chat completion."""
    try:
        async for chunk in provider.stream_chat_completion(
            messages=request.messages,
            model=model_name,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            **{
                k: v
                for k, v in request.dict().items()
                if k not in ["messages", "model", "max_tokens", "temperature", "stream"]
                and v is not None
            },
        ):
            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

        # Log streaming request (with estimated tokens)
        response_time_ms = (time.time() - start_time) * 1000
        estimated_tokens = cost_tracker.count_tokens(
            request.messages, provider.provider_name, model_name
        )
        estimated_cost = cost_tracker.calculate_cost(
            provider.provider_name, model_name, estimated_tokens, 100
        )

        await db.log_request(
            user_id=user_id,
            provider=provider.provider_name,
            model=model_name,
            messages=[msg.dict() for msg in request.messages],
            input_tokens=estimated_tokens,
            output_tokens=100,  # Estimate for streaming
            cost=estimated_cost,
            response_time_ms=response_time_ms,
            routing_reason=routing_reason,
            success=True,
        )

    except Exception as e:
        error_chunk = {"error": {"message": str(e), "type": "provider_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"


@app.get("/health", response_model=HealthResponse)
async def health_check() -> None:
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0", timestamp=datetime.now())


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(user_info: dict[str, Any] = Depends(get_authenticated_user)):
    """Get system metrics and usage statistics."""
    metrics = await db.get_system_metrics()

    return MetricsResponse(
        total_requests=metrics["total_requests"],
        total_cost=metrics["total_cost"],
        active_users=metrics["active_users"],
        provider_usage=metrics["provider_usage"],
        model_usage=metrics["model_usage"],
        average_response_time=metrics["average_response_time"],
    )


@app.get("/user/stats", response_model=UserStats)
async def get_user_stats(user_info: dict[str, Any] = Depends(get_authenticated_user)):
    """Get user-specific usage statistics."""
    user_id = user_info["user_id"]
    stats = await db.get_user_stats(user_id)

    return UserStats(**stats)


@app.get("/providers")
@app.get("/v1/providers")
async def get_providers(user_info: dict[str, Any] = Depends(get_authenticated_user)):
    """Get available providers and their models."""
    provider_info = {}

    for name, provider in providers.items():
        provider_info[name] = {
            "name": name,
            "models": provider.get_supported_models(),
            "status": "available",
        }

    return {"providers": provider_info}


@app.get("/v1/models")
async def list_models(user_info: dict[str, Any] = Depends(get_authenticated_user)):
    """List all available models across providers."""
    models = []

    # Get pricing data
    pricing = settings.get_provider_pricing()

    for provider, provider_models in pricing.items():
        for model, costs in provider_models.items():
            models.append(
                {
                    "id": f"{provider}/{model}",
                    "object": "model",
                    "provider": provider,
                    "model": model,
                    "pricing": {
                        "input": costs["input"],
                        "output": costs["output"],
                        "unit": "per_million_tokens",
                    },
                }
            )

    return {"object": "list", "data": models}


@app.get("/v1/analytics/costs")
async def get_cost_analytics(
    user_info: dict[str, Any] = Depends(get_authenticated_user),
    days: int = 30,
    provider: str | None = None,
    model: str | None = None,
):
    """Get cost analytics for the user."""
    user_id = user_info["user_id"]

    # In basic mode, return simple analytics
    if not model_muxer.enhanced_mode:
        return {
            "message": "Enhanced analytics available in enhanced mode only",
            "basic_stats": {
                "total_requests": 0,
                "total_cost": 0.0,
                "period_days": days,
            },
        }

    # Enhanced mode would have detailed analytics
    return {
        "user_id": user_id,
        "period_days": days,
        "total_cost": 0.0,
        "total_requests": 0,
        "cost_by_provider": {},
        "cost_by_model": {},
        "daily_breakdown": [],
    }


@app.get("/v1/analytics/budgets")
async def get_budget_status(
    user_info: dict[str, Any] = Depends(get_authenticated_user),
    budget_type: str | None = None,
):
    """Get budget status and alerts for the user."""
    user_id = user_info["user_id"]

    # Budget management requires enhanced mode
    if not model_muxer.enhanced_mode:
        return JSONResponse(
            status_code=501,
            content={
                "error": {
                    "message": "Budget management requires enhanced mode",
                    "type": "feature_not_available",
                    "code": "enhanced_mode_required",
                }
            },
        )

    try:
        # Get budget status from advanced cost tracker
        if hasattr(model_muxer, "advanced_cost_tracker") and model_muxer.advanced_cost_tracker:
            budget_statuses = await model_muxer.advanced_cost_tracker.get_budget_status(
                user_id, budget_type
            )

            # Convert to response format
            response_budgets = []
            for status in budget_statuses:
                alerts = [
                    {
                        "type": alert["type"],
                        "message": alert["message"],
                        "threshold": alert["threshold"],
                        "current_usage": alert["current_usage"],
                    }
                    for alert in status["alerts"]
                ]

                response_budgets.append(
                    {
                        "budget_type": status["budget_type"],
                        "budget_limit": status["budget_limit"],
                        "current_usage": status["current_usage"],
                        "usage_percentage": status["usage_percentage"],
                        "remaining_budget": status["remaining_budget"],
                        "provider": status["provider"],
                        "model": status["model"],
                        "alerts": alerts,
                        "period_start": status["period_start"],
                        "period_end": status["period_end"],
                    }
                )

            return {
                "message": "Budget status retrieved successfully",
                "budgets": response_budgets,
                "total_budgets": len(response_budgets),
            }
        else:
            return {
                "message": "Budget tracking not initialized",
                "budgets": [],
                "total_budgets": 0,
            }

    except Exception as e:
        logger.error("budget_status_error", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve budget status") from e


@app.post("/v1/analytics/budgets")
async def set_budget(
    request: dict[str, Any],
    user_info: dict[str, Any] = Depends(get_authenticated_user),
):
    """Set budget limits and alert thresholds for the user."""
    user_id = user_info["user_id"]

    # Budget management requires enhanced mode
    if not model_muxer.enhanced_mode:
        return JSONResponse(
            status_code=501,
            content={
                "error": {
                    "message": "Budget management requires enhanced mode",
                    "type": "feature_not_available",
                    "code": "enhanced_mode_required",
                }
            },
        )

    try:
        # Validate request
        budget_type = request.get("budget_type")
        budget_limit = request.get("budget_limit")

        if not budget_type or budget_limit is None:
            raise HTTPException(status_code=400, detail="budget_type and budget_limit are required")

        if budget_type not in ["daily", "weekly", "monthly", "yearly"]:
            raise HTTPException(
                status_code=400, detail="budget_type must be one of: daily, weekly, monthly, yearly"
            )

        if not isinstance(budget_limit, int | float) or budget_limit <= 0:
            raise HTTPException(status_code=400, detail="budget_limit must be a positive number")

        provider = request.get("provider")
        model = request.get("model")
        alert_thresholds = request.get("alert_thresholds", [50.0, 80.0, 95.0])

        # Set budget using advanced cost tracker
        if hasattr(model_muxer, "advanced_cost_tracker") and model_muxer.advanced_cost_tracker:
            await model_muxer.advanced_cost_tracker.set_budget(
                user_id=user_id,
                budget_type=budget_type,
                budget_limit=float(budget_limit),
                provider=provider,
                model=model,
                alert_thresholds=alert_thresholds,
            )

            return {
                "message": "Budget set successfully",
                "budget": {
                    "budget_type": budget_type,
                    "budget_limit": budget_limit,
                    "provider": provider,
                    "model": model,
                    "alert_thresholds": alert_thresholds,
                },
            }
        else:
            raise HTTPException(status_code=500, detail="Budget tracking not initialized")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("set_budget_error", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail="Failed to set budget") from e


@app.post("/v1/chat/completions/enhanced")
async def enhanced_chat_completions(
    request: ChatCompletionRequest,
    user_info: dict[str, Any] = Depends(get_authenticated_user),
):
    """Enhanced chat completions with advanced routing and features."""
    if not model_muxer.enhanced_mode:
        return JSONResponse(
            status_code=501,
            content={
                "error": {
                    "message": "Enhanced chat completions require enhanced mode",
                    "type": "feature_not_available",
                    "code": "enhanced_mode_required",
                }
            },
        )

    # In enhanced mode, this would use advanced routing, caching, etc.
    # For now, fall back to regular chat completions
    return await chat_completions(request, user_info)


def cli():
    """CLI entry point for ModelMuxer."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="ModelMuxer LLM Router")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")  # nosec B104
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument(
        "--mode",
        choices=["basic", "enhanced", "production"],
        default="basic",
        help="Deployment mode",
    )

    args = parser.parse_args()

    # Set mode environment variable
    os.environ["MODELMUXER_MODE"] = args.mode

    print(f"🚀 Starting ModelMuxer in {args.mode} mode...")

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

    # Use enhanced mode for main entry point
    os.environ.setdefault("MODELMUXER_MODE", "enhanced")

    config = model_muxer.config
    uvicorn.run(
        "app.main:app",
        host=config.host if hasattr(config, "host") else "0.0.0.0",  # nosec B104
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

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.debug)
