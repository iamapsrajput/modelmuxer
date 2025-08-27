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

# Prometheus not used in basic mode - only in enhanced mode
# Core imports
from .auth import SecurityHeaders, auth, sanitize_user_input, validate_request_size
from .config import settings as legacy_settings  # kept for enhanced mode compatibility
from .cost_tracker import cost_tracker
from .settings import settings as app_settings
from .database import db
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    UserStats,
    Choice,
    Usage,
    RouterMetadata,
    ChatMessage,
)
from .providers import AnthropicProvider, LiteLLMProvider, MistralProvider, OpenAIProvider
from .providers.base import AuthenticationError, ProviderError, RateLimitError
from .router import router
from app.policy.rules import enforce_policies  # policy integration
from app.telemetry.tracing import init_tracing, start_span, get_trace_id
from app.telemetry.metrics import HTTP_REQUESTS, HTTP_LATENCY, ROUTER_INTENT_TOTAL
from app.core.intent import classify_intent
from app.telemetry.logging import configure_json_logging

try:  # optional dependency in test/basic envs
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
except Exception:  # pragma: no cover
    CONTENT_TYPE_LATEST = None  # type: ignore[assignment]
    generate_latest = None  # type: ignore[assignment]

# Enhanced imports (optional - loaded based on availability and configuration)
try:
    # Core enhanced config (always required for enhanced mode)
    from .config import enhanced_config

    # Core enhanced features available
    CORE_ENHANCED_AVAILABLE = True
    logger = structlog.get_logger(__name__)

    # Optional cache imports (can fail without breaking enhanced mode)
    try:
        from .cache.memory_cache import MemoryCache
        from .cache.redis_cache import RedisCache

        CACHE_FEATURES_AVAILABLE = True
    except ImportError:
        MemoryCache = None
        RedisCache = None
        CACHE_FEATURES_AVAILABLE = False

    # Optional ML imports (can fail without breaking enhanced mode)
    try:
        from .classification.embeddings import EmbeddingManager
        from .classification.prompt_classifier import PromptClassifier

        ML_FEATURES_AVAILABLE = True
    except ImportError:
        EmbeddingManager = None
        PromptClassifier = None
        ML_FEATURES_AVAILABLE = False

    # Optional middleware imports (can fail without breaking enhanced mode)
    try:
        from .middleware.auth_middleware import AuthMiddleware
        from .middleware.logging_middleware import LoggingMiddleware
        from .middleware.rate_limit_middleware import RateLimitMiddleware

        MIDDLEWARE_FEATURES_AVAILABLE = True
    except ImportError:
        AuthMiddleware = None
        LoggingMiddleware = None
        RateLimitMiddleware = None
        MIDDLEWARE_FEATURES_AVAILABLE = False

    # Optional monitoring imports (can fail without breaking enhanced mode)
    try:
        from .monitoring.health_checker import HealthChecker
        from .monitoring.metrics_collector import MetricsCollector

        MONITORING_FEATURES_AVAILABLE = True
    except ImportError:
        HealthChecker = None
        MetricsCollector = None
        MONITORING_FEATURES_AVAILABLE = False

    # Optional routing imports (can fail without breaking enhanced mode)
    try:
        from .routing.cascade_router import CascadeRouter
        from .routing.heuristic_router import HeuristicRouter
        from .routing.hybrid_router import HybridRouter
        from .routing.semantic_router import SemanticRouter

        ROUTING_FEATURES_AVAILABLE = True
    except ImportError:
        CascadeRouter = None
        HeuristicRouter = None
        HybridRouter = None
        SemanticRouter = None
        ROUTING_FEATURES_AVAILABLE = False

    ENHANCED_FEATURES_AVAILABLE = CORE_ENHANCED_AVAILABLE

except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    CORE_ENHANCED_AVAILABLE = False
    CACHE_FEATURES_AVAILABLE = False
    ML_FEATURES_AVAILABLE = False
    MIDDLEWARE_FEATURES_AVAILABLE = False
    MONITORING_FEATURES_AVAILABLE = False
    ROUTING_FEATURES_AVAILABLE = False
    # Create a fallback logger to avoid AttributeError
    logger = structlog.get_logger(__name__)
    enhanced_config = None
    MemoryCache = None
    RedisCache = None
    # Enhanced features not available, running in basic mode

# Determine if we should run in enhanced mode (via centralized settings)
ENHANCED_MODE = (
    ENHANCED_FEATURES_AVAILABLE
    and app_settings.features.mode in ["enhanced", "production"]
    and enhanced_config is not None
)

# Provider instances
providers: dict[str, Any] = {}


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
        self.config = enhanced_config if enhanced_mode and enhanced_config else app_settings

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
        # Basic ModelMuxer initialized (logged via structlog if available)

    def _initialize_enhanced_components(self) -> None:
        """Initialize all enhanced components."""
        if not ENHANCED_FEATURES_AVAILABLE:
            # Enhanced features requested but not available, falling back to basic mode
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
            # Enhanced ModelMuxer initialized (logged via structlog above)

        except Exception:
            # Enhanced initialization failed, falling back to basic mode
            # Error details logged via structlog if available
            self._initialize_basic_components()

    def _initialize_cache(self) -> None:
        """Initialize caching system."""
        if not hasattr(self.config, "cache") or not self.config.cache.enabled:
            return

        # Skip if cache features are not available
        if not CACHE_FEATURES_AVAILABLE or MemoryCache is None or RedisCache is None:
            if logger:
                logger.info("cache_skipped", reason="Cache dependencies not available")
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

    def _initialize_classification(self) -> None:
        """Initialize ML classification system."""
        if not hasattr(self.config, "classification") or not self.config.classification.enabled:
            return

        # Skip if ML features are not available
        if not ML_FEATURES_AVAILABLE or EmbeddingManager is None or PromptClassifier is None:
            if logger:
                logger.info("classification_skipped", reason="ML dependencies not available")
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

            # Use Docker-compatible Redis URL
            redis_url = "redis://host.docker.internal:6379/0"
            if hasattr(self.config, "cache") and hasattr(self.config.cache, "redis_url"):
                redis_url = self.config.cache.redis_url
            elif hasattr(self.config, "cache"):
                redis_url = f"redis://{self.config.cache.redis_host}:{self.config.cache.redis_port}/{self.config.cache.redis_db}"

            self.advanced_cost_tracker = create_advanced_cost_tracker(
                db_path="cost_tracker_enhanced.db", redis_url=redis_url
            )
            self.cost_tracker = self.advanced_cost_tracker  # For backward compatibility
            if logger:
                logger.info("enhanced_cost_tracker_initialized", redis_url=redis_url)
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
                prometheus_enabled=False,  # Basic mode doesn't support Prometheus
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

            if hasattr(self.config.middleware, "rate_limit") and self.config.middleware.rate_limit.enabled:
                self.rate_limit_middleware = RateLimitMiddleware(self.config.middleware.rate_limit)

            if hasattr(self.config.middleware, "logging") and self.config.middleware.logging.enabled:
                self.logging_middleware = LoggingMiddleware(self.config.middleware.logging)

            if logger:
                logger.info("middleware_initialized")
        except Exception as e:
            if logger:
                logger.warning("middleware_init_failed", error=str(e))


# Global model muxer instance
model_muxer = ModelMuxer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    # Starting ModelMuxer LLM Router (logged via structlog if available)

    # Initialize database
    await db.init_database()
    # Database initialized (logged via structlog if available)

    # Initialize providers with API keys from centralized settings
    # OpenAI
    openai_key = app_settings.api.openai_api_key
    if openai_key and not openai_key.startswith("your-") and not openai_key.endswith("-here"):
        try:
            providers["openai"] = OpenAIProvider(api_key=openai_key)
            # OpenAI provider initialized (logged via structlog if available)
        except Exception as e:  # nosec B110
            logger.error("OpenAI provider failed to initialize", error=str(e))
            # OpenAI provider failed to initialize (logged via structlog)

    # Anthropic
    anthropic_key = app_settings.api.anthropic_api_key
    if anthropic_key and not anthropic_key.startswith("your-") and not anthropic_key.endswith("-here"):
        try:
            providers["anthropic"] = AnthropicProvider(api_key=anthropic_key)
            # Anthropic provider initialized (logged via structlog if available)
        except Exception:  # nosec B110
            # Anthropic provider failed to initialize (logged via structlog if available)
            pass

    # Mistral
    mistral_key = app_settings.api.mistral_api_key
    if mistral_key and not mistral_key.startswith("your-") and not mistral_key.endswith("-here"):
        try:
            providers["mistral"] = MistralProvider(api_key=mistral_key)
            # Mistral provider initialized (logged via structlog if available)
        except Exception:  # nosec B110
            # Mistral provider failed to initialize (logged via structlog if available)
            pass

    # LiteLLM Proxy
    litellm_base_url = str(app_settings.api.litellm_base_url) if app_settings.api.litellm_base_url else None
    litellm_api_key = app_settings.api.litellm_api_key
    if litellm_base_url:
        try:
            # LiteLLM can work with or without API key depending on proxy configuration
            providers["litellm"] = LiteLLMProvider(
                base_url=litellm_base_url,
                api_key=litellm_api_key,
                custom_models={},  # Can be configured via environment or config file
            )
            # LiteLLM provider initialized (logged via structlog if available)
            if logger:
                logger.info("litellm_provider_initialized", base_url=litellm_base_url)
        except Exception as e:  # nosec B110
            if logger:
                logger.error("LiteLLM provider failed to initialize", error=str(e))
            # LiteLLM provider failed to initialize (logged via structlog if available)

    if not providers:
        # No providers initialized! Check your API keys.
        # Make sure you have valid API keys in your .env file:
        # - OPENAI_API_KEY=sk-...
        # - ANTHROPIC_API_KEY=sk-ant-...
        # - MISTRAL_API_KEY=...
        # (Keys should not contain placeholder text like 'your-key-here')
        pass

    # Router ready with providers (logged via structlog if available)

    yield

    # Shutdown
    # Shutting down ModelMuxer (logged via structlog if available)
    for provider in providers.values():
        if hasattr(provider, "client"):
            await provider.client.aclose()
    # Cleanup complete (logged via structlog if available)


# Initialize tracing and logging
try:
    if app_settings.observability.enable_tracing:
        init_tracing(
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
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
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
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
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


@app.middleware("http")
async def request_observability_middleware(request: Request, call_next):
    route = request.scope.get("route").path if request.scope.get("route") else request.url.path
    method = request.method
    start_ts = time.perf_counter()
    with start_span("http.request", route=route, method=method):
        try:
            response = await call_next(request)
            status = str(response.status_code)
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


if app_settings.observability.enable_metrics and generate_latest and CONTENT_TYPE_LATEST:

    @app.get(app_settings.observability.prom_metrics_path, include_in_schema=False)
    async def prometheus_metrics():
        """Expose Prometheus metrics endpoint (unauthenticated, read-only)."""
        return JSONResponse(content=generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


async def get_authenticated_user(request: Request, authorization: str | None = Header(None)) -> dict[str, Any]:
    """Dependency to authenticate requests."""
    return await auth.authenticate_request(request, authorization)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, user_info: dict[str, Any] = Depends(get_authenticated_user)):
    """
    Create a chat completion using the optimal LLM provider.

    This endpoint is compatible with OpenAI's chat completions API.
    """
    start_time = time.time()
    user_id = user_info["user_id"]

    try:
        # DEBUG: Check providers at request time
        print(f"DEBUG: Global providers dict at request time: {list(providers.keys())}")
        print(f"DEBUG: ModelMuxer providers dict at request time: {list(model_muxer.providers.keys())}")

        # Sanitize input messages
        for message in request.messages:
            message.content = sanitize_user_input(message.content)

        # Enforce policies early (before routing, budget, or provider calls)
        tenant_id = user_id or "anonymous"
        policy_result = enforce_policies(request, tenant_id)
        if policy_result.blocked:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": {
                        "message": "Request blocked by policy",
                        "type": "policy_violation",
                        "reasons": policy_result.reasons,
                    }
                },
            )
        # Replace last message content with sanitized prompt
        if request.messages:
            request.messages[-1].content = policy_result.sanitized_prompt

        # Classify intent (feature-flagged)
        intent = {"label": "unknown", "confidence": 0.0, "signals": {}, "method": "disabled"}
        try:
            intent = await classify_intent(request.messages)
            with start_span(
                "router.intent",
                **{
                    "route.intent.label": intent.get("label"),
                    "route.intent.confidence": intent.get("confidence"),
                    "route.intent.method": intent.get("method"),
                },
            ):
                try:
                    ROUTER_INTENT_TOTAL.labels(intent.get("label", "unknown")).inc()
                except Exception:
                    pass
        except Exception:
            pass

        # Route the request to the best provider/model
        print(f"DEBUG: About to call router.select_model with router type: {type(router)}")
        try:
            provider_name, model_name, routing_reason = router.select_model(messages=request.messages, user_id=user_id)
            print(f"DEBUG: Router selected - provider: {provider_name}, model: {model_name}, reason: {routing_reason}")
        except Exception as e:
            print(f"DEBUG: Router selection failed: {e}")
            raise

        # Check if provider is available
        if provider_name not in providers:
            print(f"DEBUG: Provider {provider_name} not found in providers dict!")
            raise HTTPException(
                status_code=503,
                detail=ErrorResponse.create(
                    message=f"Provider {provider_name} is not available",
                    error_type="service_unavailable",
                    code="provider_unavailable",
                ).dict(),
            )

        provider = providers[provider_name]
        print(f"DEBUG: Successfully got provider: {type(provider).__name__}")

        # Estimate cost and check budget using ModelMuxer's cost tracker
        active_cost_tracker = model_muxer.cost_tracker if model_muxer.enhanced_mode else cost_tracker
        print(f"DEBUG: About to estimate cost with cost_tracker: {type(active_cost_tracker)}")
        try:
            cost_estimate = active_cost_tracker.estimate_request_cost(
                messages=request.messages,
                provider=provider_name,
                model=model_name,
                max_tokens=request.max_tokens,
            )
            print(f"DEBUG: Cost estimate successful: {cost_estimate}")
        except Exception as cost_exc:
            print(f"DEBUG: Cost estimation failed: {cost_exc}")
            print(f"DEBUG: Cost exception type: {type(cost_exc).__name__}")
            raise

        print(f"DEBUG: About to check budget for user: {user_id}")
        try:
            budget_check = await db.check_budget(user_id, cost_estimate["estimated_cost"])
            print(f"DEBUG: Budget check result: {budget_check}")
        except Exception as budget_exc:
            print(f"DEBUG: Budget check failed: {budget_exc}")
            print(f"DEBUG: Budget exception type: {type(budget_exc).__name__}")
            raise

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
                stream_chat_completion(provider, request, model_name, routing_reason, user_id, start_time),
                media_type="text/event-stream",  # Correct media type for SSE
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            # Make the request to the provider
            print(f"DEBUG: About to call provider.chat_completion with provider: {type(provider).__name__}")
            print(f"DEBUG: Model name: {model_name}")
            print(f"DEBUG: Number of messages: {len(request.messages)}")
            print(f"DEBUG: Max tokens: {request.max_tokens}")
            print(f"DEBUG: Temperature: {request.temperature}")

            try:
                if app_settings.features.provider_adapters_enabled:
                    # Use unified adapter path
                    from app.providers.registry import PROVIDERS

                    # Simple prompt join for adapters
                    prompt_text = "\n".join([msg.content for msg in request.messages if msg.content])
                    adapter = PROVIDERS.get(provider_name)
                    if adapter is None:
                        raise RuntimeError(f"No adapter registered for provider: {provider_name}")
                    adapter_resp = await adapter.invoke(
                        model=model_name,
                        prompt=prompt_text,
                        temperature=request.temperature or app_settings.router.temperature_default,
                        max_tokens=request.max_tokens or app_settings.router.max_tokens_default,
                    )
                    if adapter_resp.error:
                        raise ProviderError(adapter_resp.error)
                    # Build ChatCompletionResponse
                    estimated_cost = cost_tracker.calculate_cost(
                        provider_name,
                        model_name,
                        adapter_resp.tokens_in,
                        adapter_resp.tokens_out,
                    )
                    response = ChatCompletionResponse(
                        id=f"chatcmpl-adapter-{int(time.time() * 1000)}",
                        object="chat.completion",
                        created=int(datetime.now().timestamp()),
                        model=model_name,
                        choices=[
                            Choice(
                                index=0,
                                message=ChatMessage(role="assistant", content=adapter_resp.output_text, name=None),
                                finish_reason="stop",
                            )
                        ],
                        usage=Usage(
                            prompt_tokens=adapter_resp.tokens_in,
                            completion_tokens=adapter_resp.tokens_out,
                            total_tokens=adapter_resp.tokens_in + adapter_resp.tokens_out,
                        ),
                        router_metadata=RouterMetadata(
                            selected_provider=provider_name,
                            selected_model=model_name,
                            routing_reason=routing_reason,
                            estimated_cost=estimated_cost,
                            response_time_ms=adapter_resp.latency_ms,
                            intent_label=intent.get("label"),
                            intent_confidence=float(intent.get("confidence", 0.0)),
                            intent_signals=intent.get("signals"),
                        ),
                    )
                else:
                    response = await provider.chat_completion(
                        messages=request.messages,
                        model=model_name,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        **{
                            k: v
                            for k, v in request.dict().items()
                            if k not in ["messages", "model", "max_tokens", "temperature", "stream"] and v is not None
                        },
                    )
                print(f"DEBUG: Successfully got response from provider: {type(response)}")
            except Exception as provider_exc:
                print(f"DEBUG: Provider call failed: {provider_exc}")
                print(f"DEBUG: Provider exception type: {type(provider_exc).__name__}")
                raise

            # Update routing reason in metadata and attach intent
            response.router_metadata.routing_reason = routing_reason
            try:
                response.router_metadata.intent_label = intent.get("label")
                response.router_metadata.intent_confidence = float(intent.get("confidence", 0.0))
                response.router_metadata.intent_signals = intent.get("signals")
            except Exception:
                pass

            # Log the request using enhanced cost tracker if available
            if (
                model_muxer.enhanced_mode
                and hasattr(model_muxer, "advanced_cost_tracker")
                and model_muxer.advanced_cost_tracker
            ):
                await model_muxer.advanced_cost_tracker.log_simple_request(
                    user_id=user_id,
                    session_id=user_id,  # Use user_id as session_id for now
                    provider=provider_name,
                    model=model_name,
                    cost=response.router_metadata.estimated_cost,
                    success=True,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                )
            else:
                # Fallback to basic database logging
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


async def stream_chat_completion(provider, request, model_name, routing_reason, user_id, start_time):
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
                if k not in ["messages", "model", "max_tokens", "temperature", "stream"] and v is not None
            },
        ):
            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

        # Log streaming request (with estimated tokens)
        response_time_ms = (time.time() - start_time) * 1000
        estimated_tokens = cost_tracker.count_tokens(request.messages, provider.provider_name, model_name)
        estimated_cost = cost_tracker.calculate_cost(provider.provider_name, model_name, estimated_tokens, 100)

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

    except Exception:
        structlog.get_logger().exception("Exception in stream_chat_completion", exc_info=True)
        error_chunk = {"error": {"message": "An internal error occurred.", "type": "provider_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
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
            budget_statuses = await model_muxer.advanced_cost_tracker.get_budget_status(user_id, budget_type)

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
            raise HTTPException(status_code=400, detail="budget_type must be one of: daily, weekly, monthly, yearly")

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


# =============================================================================
# ANTHROPIC API COMPATIBILITY
# =============================================================================


@app.post("/v1/messages")
@app.post("/messages")
async def anthropic_messages(
    request: Request,
    beta: bool = False,  # Support for beta parameter
    user_info: dict[str, Any] = Depends(get_authenticated_user),
):
    """Anthropic Messages API compatibility endpoint with beta support."""
    try:
        # Parse Anthropic format request
        body = await request.json()

        # Convert Anthropic format to OpenAI format
        messages = []
        if "system" in body:
            messages.append({"role": "system", "content": body["system"]})

        if "messages" in body:
            messages.extend(body["messages"])

        # Import message model
        from .models import ChatMessage

        # Convert messages to proper format
        converted_messages = []
        for msg in messages:
            if isinstance(msg.get("content"), str):
                converted_messages.append(ChatMessage(role=msg["role"], content=msg["content"]))
            elif isinstance(msg.get("content"), list):
                # Handle multi-part content
                content_text = ""
                for part in msg["content"]:
                    if part.get("type") == "text":
                        content_text += part.get("text", "")

                converted_messages.append(ChatMessage(role=msg["role"], content=content_text))

        # Create OpenAI format request
        openai_request = ChatCompletionRequest(
            messages=converted_messages,
            model=body.get("model", "claude-3-5-sonnet-20241022"),
            max_tokens=body.get("max_tokens", 1000),
            temperature=body.get("temperature", 0.7),
            stream=body.get("stream", False),
        )

        # Route through normal chat completions
        response = await chat_completions(openai_request, user_info)

        # Handle different response types
        if isinstance(response, JSONResponse):
            # Error response - pass through
            return response
        elif isinstance(response, StreamingResponse):
            # For streaming responses, return as-is (already in correct SSE format)
            # Claude Dev expects Server-Sent Events, which our streaming already provides
            return response

        # Convert non-streaming OpenAI response to Anthropic format
        anthropic_response = {
            "id": response.id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": response.choices[0].message.content}],
            "model": response.router_metadata.selected_model,
            "stop_reason": (
                "end_turn" if response.choices[0].finish_reason == "stop" else response.choices[0].finish_reason
            ),
            "stop_sequence": None,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
        }

        return anthropic_response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error("anthropic_api_error", error=str(e), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {"type": "internal_server_error", "message": str(e)},
            },
        )


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
        choices=["basic", "enhanced", "production"],
        default="basic",
        help="Deployment mode",
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

    # Use enhanced mode for main entry point
    os.environ.setdefault("MODELMUXER_MODE", "enhanced")

    config = model_muxer.config
    uvicorn.run(
        "app.main:app",
        host=config.host if hasattr(config, "host") else "127.0.0.1",  # safer default
        port=config.port if hasattr(config, "port") else 8000,
        reload=config.debug if hasattr(config, "debug") else False,
        log_level=(
            config.logging.level.lower() if hasattr(config, "logging") and hasattr(config.logging, "level") else "info"
        ),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.debug)
