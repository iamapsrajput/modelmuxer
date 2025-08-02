# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Enhanced ModelMuxer main application with all advanced features.

This module provides the complete ModelMuxer application with advanced routing,
multiple providers, ML-based classification, comprehensive monitoring, and more.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Request, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import structlog

# Core imports
from .models import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    BudgetRequest,
    CascadeConfig,
    EnhancedChatCompletionRequest,
    RoutingMetadata,
    EnhancedChatCompletionResponse,
    BudgetPeriodEnum,
)
from .config.enhanced_config import enhanced_config

# Provider imports
from .providers import (
    OpenAIProvider,
    AnthropicProvider,
    MistralProvider,
    GoogleProvider,
    CohereProvider,
    GroqProvider,
    TogetherProvider,
    LiteLLMProvider,
)

# Routing imports
from .routing.heuristic_router import HeuristicRouter
from .routing.semantic_router import SemanticRouter
from .routing.cascade_router import CascadeRouter
from .routing.hybrid_router import HybridRouter

# Enhanced cost tracking
from .cost_tracker_enhanced import AdvancedCostTracker, BudgetPeriod

# Classification imports
from .classification.embeddings import EmbeddingManager
from .classification.prompt_classifier import PromptClassifier

# Middleware imports
from .middleware.auth_middleware import AuthMiddleware
from .middleware.rate_limit_middleware import RateLimitMiddleware
from .middleware.logging_middleware import LoggingMiddleware

# Monitoring imports
from .monitoring.metrics import MetricsCollector, HealthChecker

# Cache imports
from .cache.memory_cache import MemoryCache
from .cache.redis_cache import RedisCache

# Core exceptions
from .core.exceptions import (
    ModelMuxerError,
    ProviderError,
    RoutingError,
    AuthenticationError,
    RateLimitError,
    CacheError,
    ClassificationError,
)

logger = structlog.get_logger(__name__)


class EnhancedModelMuxer:
    """
    Enhanced ModelMuxer with all advanced features.

    Provides intelligent LLM routing with multiple providers, ML-based classification,
    comprehensive monitoring, caching, authentication, and rate limiting.
    """

    def __init__(self):
        self.config = enhanced_config

        # Initialize components
        self.providers: Dict[str, Any] = {}
        self.routers: Dict[str, Any] = {}
        self.cache = None
        self.embedding_manager = None
        self.classifier = None
        self.metrics_collector = None
        self.health_checker = None

        # Enhanced cost tracking
        self.cost_tracker = None

        # Middleware
        self.auth_middleware = None
        self.rate_limit_middleware = None
        self.logging_middleware = None

        # Initialize all components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all ModelMuxer components."""
        logger.info("initializing_enhanced_modelmuxer")

        # Initialize providers
        self._initialize_providers()

        # Initialize cache
        self._initialize_cache()

        # Initialize enhanced cost tracking
        self._initialize_cost_tracking()

        # Initialize classification
        self._initialize_classification()

        # Initialize routing
        self._initialize_routing()

        # Initialize monitoring
        self._initialize_monitoring()

        # Initialize middleware
        self._initialize_middleware()

        logger.info("enhanced_modelmuxer_initialized")

    def _initialize_providers(self):
        """Initialize LLM providers."""
        provider_config = self.config.providers

        # OpenAI
        if provider_config.openai_api_key:
            self.providers["openai"] = OpenAIProvider(
                api_key=provider_config.openai_api_key, base_url=provider_config.openai_base_url
            )
            logger.info("openai_provider_initialized")

        # Anthropic
        if provider_config.anthropic_api_key:
            self.providers["anthropic"] = AnthropicProvider(api_key=provider_config.anthropic_api_key)
            logger.info("anthropic_provider_initialized")

        # Mistral
        if provider_config.mistral_api_key:
            self.providers["mistral"] = MistralProvider(api_key=provider_config.mistral_api_key)
            logger.info("mistral_provider_initialized")

        # Google
        if provider_config.google_api_key:
            self.providers["google"] = GoogleProvider(api_key=provider_config.google_api_key)
            logger.info("google_provider_initialized")

        # Cohere
        if provider_config.cohere_api_key:
            self.providers["cohere"] = CohereProvider(api_key=provider_config.cohere_api_key)
            logger.info("cohere_provider_initialized")

        # Groq
        if provider_config.groq_api_key:
            self.providers["groq"] = GroqProvider(api_key=provider_config.groq_api_key)
            logger.info("groq_provider_initialized")

        # Together AI
        if provider_config.together_api_key:
            self.providers["together"] = TogetherProvider(api_key=provider_config.together_api_key)
            logger.info("together_provider_initialized")

        # LiteLLM Proxy
        if provider_config.litellm_base_url:
            self.providers["litellm"] = LiteLLMProvider(
                base_url=provider_config.litellm_base_url, api_key=provider_config.litellm_api_key
            )
            logger.info("litellm_provider_initialized")

        if not self.providers:
            logger.warning("no_providers_configured", message="No API keys provided. ModelMuxer will run in test mode.")
        else:
            logger.info("providers_initialized", count=len(self.providers))

    def _initialize_cache(self):
        """Initialize caching system."""
        cache_config = self.config.cache

        if not cache_config.enabled:
            logger.info("caching_disabled")
            return

        if cache_config.backend == "redis":
            try:
                self.cache = RedisCache(
                    redis_url=cache_config.redis_url,
                    db=cache_config.redis_db,
                    key_prefix=cache_config.redis_key_prefix,
                    default_ttl=cache_config.default_ttl,
                    compression_enabled=cache_config.redis_compression,
                )
                logger.info("redis_cache_initialized")
            except Exception as e:
                logger.warning("redis_cache_init_failed", error=str(e))
                # Fallback to memory cache
                self.cache = MemoryCache(
                    max_size=cache_config.memory_max_size,
                    default_ttl=cache_config.default_ttl,
                    max_memory_mb=cache_config.memory_max_memory_mb,
                )
                logger.info("fallback_to_memory_cache")
        else:
            self.cache = MemoryCache(
                max_size=cache_config.memory_max_size,
                default_ttl=cache_config.default_ttl,
                max_memory_mb=cache_config.memory_max_memory_mb,
            )
            logger.info("memory_cache_initialized")

    def _initialize_cost_tracking(self):
        """Initialize enhanced cost tracking system."""
        try:
            # Use Redis URL from cache config if available
            redis_url = "redis://localhost:6379/0"
            if hasattr(self.config.cache, "redis_url"):
                redis_url = self.config.cache.redis_url

            self.cost_tracker = AdvancedCostTracker(db_path="cost_tracker.db", redis_url=redis_url)
            logger.info("enhanced_cost_tracker_initialized")
        except Exception as e:
            logger.warning("cost_tracker_init_failed", error=str(e))
            # Create a basic cost tracker without Redis
            self.cost_tracker = AdvancedCostTracker(
                db_path="cost_tracker.db",
                redis_url="redis://localhost:6379/0",  # Will use MockRedisClient on failure
            )
            logger.info("cost_tracker_initialized_with_fallback")

    def _initialize_classification(self):
        """Initialize ML-based classification."""
        classification_config = self.config.classification

        if not classification_config.enabled:
            logger.info("classification_disabled")
            return

        try:
            # Initialize embedding manager
            self.embedding_manager = EmbeddingManager(
                model_name=classification_config.embedding_model,
                cache_dir=classification_config.embedding_cache_dir,
                enable_cache=classification_config.embedding_cache_enabled,
            )

            # Initialize classifier
            self.classifier = PromptClassifier(
                embedding_manager=self.embedding_manager,
                config={
                    "confidence_threshold": classification_config.confidence_threshold,
                    "max_history_size": classification_config.max_history_size,
                },
            )

            logger.info("classification_initialized")

        except Exception as e:
            logger.warning("classification_init_failed", error=str(e))
            self.embedding_manager = None
            self.classifier = None

    def _initialize_routing(self):
        """Initialize routing strategies."""
        routing_config = self.config.routing

        # Initialize individual routers
        if routing_config.heuristic_enabled:
            self.routers["heuristic"] = HeuristicRouter()
            logger.info("heuristic_router_initialized")

        if routing_config.semantic_enabled and self.embedding_manager:
            try:
                self.routers["semantic"] = SemanticRouter(
                    {
                        "model_name": routing_config.semantic_model,
                        "similarity_threshold": routing_config.semantic_threshold,
                    }
                )
                logger.info("semantic_router_initialized")
            except Exception as e:
                logger.warning("semantic_router_init_failed", error=str(e))

        if routing_config.cascade_enabled:
            self.routers["cascade"] = CascadeRouter(
                cost_tracker=self.cost_tracker,
                config={
                    "max_cascade_levels": routing_config.cascade_max_levels,
                    "quality_threshold": routing_config.cascade_quality_threshold,
                },
            )
            logger.info("cascade_router_initialized")

        # Initialize hybrid router if multiple strategies are available
        if len(self.routers) > 1:
            try:
                self.routers["hybrid"] = HybridRouter(
                    {
                        "strategy_weights": routing_config.get_strategy_weights_dict(),
                        "consensus_threshold": routing_config.hybrid_consensus_threshold,
                        "heuristic": {},
                        "semantic": {"model_name": routing_config.semantic_model},
                        "cascade": {"max_cascade_levels": routing_config.cascade_max_levels},
                    }
                )
                logger.info("hybrid_router_initialized")
            except Exception as e:
                logger.warning("hybrid_router_init_failed", error=str(e))

        # Set default router
        default_strategy = routing_config.default_strategy
        if default_strategy not in self.routers:
            # Fallback to first available router
            default_strategy = list(self.routers.keys())[0]
            logger.warning("default_router_fallback", fallback=default_strategy)

        self.default_router = self.routers[default_strategy]
        logger.info("routing_initialized", default_strategy=default_strategy)

    def _initialize_monitoring(self):
        """Initialize monitoring and metrics."""
        monitoring_config = self.config.monitoring

        if not monitoring_config.enabled:
            logger.info("monitoring_disabled")
            return

        # Initialize metrics collector
        self.metrics_collector = MetricsCollector()

        # Initialize health checker
        self.health_checker = HealthChecker(self.metrics_collector)

        # Set system info
        self.metrics_collector.set_system_info(
            {
                "version": "1.0.0",
                "providers": ",".join(self.providers.keys()),
                "routers": ",".join(self.routers.keys()),
                "cache_backend": self.config.cache.backend if self.config.cache.enabled else "none",
            }
        )

        logger.info("monitoring_initialized")

    def _initialize_middleware(self):
        """Initialize middleware components."""
        # Authentication middleware
        if self.config.auth.enabled:
            self.auth_middleware = AuthMiddleware(
                {
                    "auth_methods": self.config.auth.get_methods_list(),
                    "api_keys": self.config.auth.get_api_keys_list(),
                    "jwt_secret": self.config.auth.jwt_secret,
                    "jwt_algorithm": self.config.auth.jwt_algorithm,
                    "require_https": self.config.auth.require_https,
                    "allowed_origins": self.config.auth.get_allowed_origins_list(),
                }
            )
            logger.info("auth_middleware_initialized")

        # Rate limiting middleware
        if self.config.rate_limit.enabled:
            self.rate_limit_middleware = RateLimitMiddleware(
                {
                    "algorithm": self.config.rate_limit.algorithm,
                    "default_limits": {
                        "requests_per_second": self.config.rate_limit.requests_per_second,
                        "requests_per_minute": self.config.rate_limit.requests_per_minute,
                        "requests_per_hour": self.config.rate_limit.requests_per_hour,
                        "burst_size": self.config.rate_limit.burst_size,
                    },
                    "global_limits": {
                        "requests_per_second": self.config.rate_limit.global_requests_per_second,
                        "requests_per_minute": self.config.rate_limit.global_requests_per_minute,
                    },
                    "enable_global_limits": self.config.rate_limit.global_enabled,
                    "enable_adaptive_limits": self.config.rate_limit.adaptive_enabled,
                    "system_load_threshold": self.config.rate_limit.system_load_threshold,
                }
            )
            logger.info("rate_limit_middleware_initialized")

        # Logging middleware
        self.logging_middleware = LoggingMiddleware(
            {
                "log_requests": self.config.logging.log_requests,
                "log_responses": self.config.logging.log_responses,
                "log_request_body": self.config.logging.log_request_body,
                "log_response_body": self.config.logging.log_response_body,
                "log_headers": self.config.logging.log_headers,
                "sanitize_sensitive_data": self.config.logging.sanitize_sensitive_data,
                "enable_audit_log": self.config.logging.audit_enabled,
                "track_performance": self.config.monitoring.track_performance,
                "slow_request_threshold": self.config.monitoring.slow_request_threshold,
            }
        )
        logger.info("logging_middleware_initialized")

    async def route_request(
        self,
        messages: List[ChatMessage],
        user_id: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        routing_strategy: Optional[str] = None,
    ) -> tuple:
        """Route a request to the optimal provider and model."""
        # Select router
        router = self.routers.get(routing_strategy) or self.default_router

        # Route the request
        provider_name, model, reasoning, confidence = await router.select_provider_and_model(
            messages, user_id, constraints
        )

        # Record routing decision
        if self.metrics_collector:
            self.metrics_collector.record_routing_decision(
                strategy=router.name, selected_provider=provider_name, selected_model=model, confidence=confidence
            )

        return provider_name, model, reasoning, confidence

    async def chat_completion(
        self, request: ChatCompletionRequest, user_info: Dict[str, Any]
    ) -> ChatCompletionResponse:
        """Process a chat completion request."""
        start_time = time.time()

        try:
            # Route the request
            provider_name, model, reasoning, confidence = await self.route_request(
                messages=request.messages,
                user_id=user_info.get("user_id"),
                constraints=getattr(request, "constraints", None),
                routing_strategy=getattr(request, "routing_strategy", None),
            )

            # Get the provider
            if provider_name not in self.providers:
                raise ProviderError(f"Provider {provider_name} not available")

            provider = self.providers[provider_name]

            # Make the request
            response = await provider.chat_completion(
                messages=request.messages,
                model=model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=request.stream,
            )

            # Update response with routing metadata
            response.router_metadata.routing_strategy = reasoning
            response.router_metadata.confidence_score = confidence

            # Record metrics
            duration = time.time() - start_time
            if self.metrics_collector:
                self.metrics_collector.record_provider_request(
                    provider=provider_name,
                    model=model,
                    status="success",
                    duration=duration,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    cost=response.usage.total_cost,
                )

            return response

        except Exception as e:
            duration = time.time() - start_time

            # Record error metrics
            if self.metrics_collector:
                self.metrics_collector.record_error(
                    error_type=type(e).__name__,
                    endpoint="/v1/chat/completions",
                    provider=provider_name if "provider_name" in locals() else None,
                )

            raise

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        if not self.health_checker:
            return {"status": "healthy", "message": "Health checking disabled"}

        # Check provider health
        provider_health = {}
        for name, provider in self.providers.items():
            provider_health[name] = await self.health_checker.check_provider_health(name, provider)

        # Check cache health
        cache_health = True
        if self.cache:
            cache_health = await self.health_checker.check_cache_health(self.cache)

        # Check system resources
        system_resources = self.health_checker.check_system_resources()

        # Get overall health
        overall_health = self.health_checker.get_overall_health()

        return {
            **overall_health,
            "providers": provider_health,
            "cache": {"status": "healthy" if cache_health else "unhealthy"},
            "system_resources": system_resources,
        }


# Global ModelMuxer instance
model_muxer = EnhancedModelMuxer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("enhanced_modelmuxer_starting")

    # Perform initial health checks
    if model_muxer.health_checker:
        await model_muxer.health_check()

    logger.info("enhanced_modelmuxer_started")

    yield

    # Shutdown
    logger.info("enhanced_modelmuxer_shutting_down")

    # Close cache connections
    if model_muxer.cache:
        if hasattr(model_muxer.cache, "close"):
            await model_muxer.cache.close()

    logger.info("enhanced_modelmuxer_shutdown_complete")


# Create FastAPI application
app = FastAPI(
    title="ModelMuxer Enhanced",
    description="Advanced LLM routing system with intelligent provider selection",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=model_muxer.config.auth.get_allowed_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency for authentication
async def get_current_user(request: Request, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Get current authenticated user."""
    if not model_muxer.auth_middleware:
        # Authentication disabled
        return {"user_id": "anonymous", "role": "user", "auth_method": "none"}

    return await model_muxer.auth_middleware.authenticate_request(request, authorization)


# Middleware for request/response logging
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Request/response logging middleware."""
    if not model_muxer.logging_middleware:
        return await call_next(request)

    # Log request
    request_context = await model_muxer.logging_middleware.log_request(request)

    # Process request
    error = None
    response = None

    try:
        response = await call_next(request)

        # Record request metrics
        if model_muxer.metrics_collector:
            duration = time.time() - request_context["start_time"]
            model_muxer.metrics_collector.record_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code,
                duration=duration,
                user_id=request_context.get("user_info", {}).get("user_id"),
            )

    except Exception as e:
        error = e
        response = JSONResponse(
            status_code=500,
            content={
                "error": {"message": "Internal server error", "type": "internal_error", "code": "internal_server_error"}
            },
        )

    # Log response
    await model_muxer.logging_middleware.log_response(response, request_context, error)

    if error and not isinstance(error, HTTPException):
        raise error

    return response


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, user_info: Dict[str, Any] = Depends(get_current_user)):
    """Enhanced chat completions endpoint with advanced routing."""
    try:
        # Check rate limits
        if model_muxer.rate_limit_middleware:
            await model_muxer.rate_limit_middleware.check_rate_limit(
                request=request,  # This would need to be the FastAPI Request object
                user_id=user_info["user_id"],
                user_limits=user_info.get("rate_limits"),
            )

        # Classify the prompt if classifier is available
        if model_muxer.classifier:
            full_text = " ".join([msg.content for msg in request.messages if msg.content])
            classification = await model_muxer.classifier.classify(full_text)

            # Record classification metrics
            if model_muxer.metrics_collector:
                model_muxer.metrics_collector.record_classification(
                    category=classification["category"],
                    method=classification["method"],
                    confidence=classification["confidence"],
                )

        # Process the request
        response = await model_muxer.chat_completion(request, user_info)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("chat_completion_error", error=str(e), user_id=user_info.get("user_id"))

        if isinstance(e, ModelMuxerError):
            raise HTTPException(status_code=400, detail=e.to_dict())
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": "Internal server error",
                        "type": "internal_error",
                        "code": "internal_server_error",
                    }
                },
            )


@app.get("/health")
async def health_endpoint():
    """Health check endpoint."""
    try:
        health_status = await model_muxer.health_check()

        if health_status["status"] == "healthy":
            return health_status
        else:
            return JSONResponse(status_code=503, content=health_status)
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    if not model_muxer.metrics_collector:
        raise HTTPException(status_code=404, detail="Metrics not enabled")

    metrics_data = generate_latest(model_muxer.metrics_collector.registry)
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/models")
async def list_models(user_info: Dict[str, Any] = Depends(get_current_user)):
    """List available models from all providers."""
    models = []

    for provider_name, provider in model_muxer.providers.items():
        try:
            provider_models = provider.get_supported_models()
            for model in provider_models:
                models.append(
                    {
                        "id": f"{provider_name}/{model}",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": provider_name,
                        "provider": provider_name,
                        "model": model,
                    }
                )
        except Exception as e:
            logger.warning("failed_to_get_models", provider=provider_name, error=str(e))

    return {"object": "list", "data": models}


@app.get("/v1/providers")
async def list_providers(user_info: Dict[str, Any] = Depends(get_current_user)):
    """List available providers and their status."""
    providers = []

    for provider_name, provider in model_muxer.providers.items():
        try:
            is_healthy = await provider.health_check()
            supported_models = provider.get_supported_models()

            providers.append(
                {
                    "name": provider_name,
                    "status": "healthy" if is_healthy else "unhealthy",
                    "models": supported_models,
                    "model_count": len(supported_models),
                }
            )
        except Exception as e:
            providers.append(
                {"name": provider_name, "status": "error", "error": str(e), "models": [], "model_count": 0}
            )

    return {"providers": providers}


# Enhanced API Endpoints for Part 2: Cost-Aware Cascading & Analytics


@app.get("/v1/analytics/costs")
async def get_cost_analytics(
    user_info: Dict[str, Any] = Depends(get_current_user),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    group_by: str = Query("day", pattern="^(hour|day|provider|model)$"),
):
    """Get detailed cost analytics"""
    user_id = user_info.get("user_id", "anonymous")

    analytics = await model_muxer.cost_tracker.get_cost_analytics(
        user_id=user_id, start_date=start_date, end_date=end_date, group_by=group_by
    )
    return analytics


@app.get("/v1/analytics/budgets")
async def get_budget_status(user_info: Dict[str, Any] = Depends(get_current_user)):
    """Get current budget status and alerts"""
    user_id = user_info.get("user_id", "anonymous")
    budget_status = await model_muxer.cost_tracker.get_budget_status(user_id)
    return budget_status


@app.post("/v1/analytics/budgets")
async def set_budget(budget_request: BudgetRequest, user_info: Dict[str, Any] = Depends(get_current_user)):
    """Set or update user budget"""
    user_id = user_info.get("user_id", "anonymous")

    await model_muxer.cost_tracker.set_budget(
        user_id=user_id,
        budget_type=BudgetPeriod(budget_request.budget_type),
        budget_limit=budget_request.budget_limit,
        provider=budget_request.provider,
        model=budget_request.model,
        alert_thresholds=budget_request.alert_thresholds,
    )
    return {"status": "success", "message": "Budget updated successfully"}


@app.post("/v1/chat/completions/enhanced")
async def enhanced_chat_completions(
    request: EnhancedChatCompletionRequest,
    user_info: Dict[str, Any] = Depends(get_current_user),
    routing_strategy: str = Header("balanced", alias="X-Routing-Strategy"),
    max_budget: float = Header(0.1, alias="X-Max-Budget"),
    enable_cascade: bool = Header(True, alias="X-Enable-Cascade"),
):
    """Enhanced chat completions with cascade routing"""
    user_id = user_info.get("user_id", "anonymous")

    try:
        # Check budget before processing
        current_budget_status = await model_muxer.cost_tracker.get_budget_status(user_id)

        # Check if any daily budgets are exceeded
        daily_budgets = [b for b in current_budget_status["budgets"] if b["type"] == "daily"]
        if any(b["utilization_percent"] >= 100 for b in daily_budgets):
            raise HTTPException(
                status_code=429, detail="Daily budget exceeded. Please increase your budget or try again tomorrow."
            )

        session_id = request.session_id or f"session_{int(datetime.now().timestamp())}"

        if enable_cascade and "cascade" in model_muxer.routers:
            # Use cascade routing
            cascade_router = model_muxer.routers["cascade"]
            response, routing_metadata = await cascade_router.route_with_cascade(
                messages=request.messages,
                cascade_type=routing_strategy,
                max_budget=max_budget,
                user_id=user_id,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            # Log cascade request
            await model_muxer.cost_tracker.log_request_with_cascade(
                user_id=user_id, session_id=session_id, cascade_metadata=routing_metadata, success=True
            )

            # Add routing metadata to response
            response["routing_metadata"] = routing_metadata

        else:
            # Use single-model routing (existing logic)
            if "semantic" in model_muxer.routers:
                semantic_router = model_muxer.routers["semantic"]
                route_result = semantic_router.route(request.messages)
                selected_model = route_result["model"]
                provider_name = route_result["provider"]
            else:
                # Fallback to first available provider
                provider_name = list(model_muxer.providers.keys())[0]
                provider = model_muxer.providers[provider_name]
                selected_model = provider.get_supported_models()[0]

            provider = model_muxer.providers[provider_name]
            response = await provider.chat_completion(
                messages=request.messages,
                model=selected_model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            # Calculate and log cost
            usage = response.get("usage", {})
            cost = provider.calculate_cost(
                usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), selected_model
            )

            await model_muxer.cost_tracker.log_simple_request(
                user_id=user_id,
                session_id=session_id,
                provider=provider_name,
                model=selected_model,
                cost=cost,
                success=True,
            )

        return response

    except Exception as e:
        # Log failed request
        await model_muxer.cost_tracker.log_request_with_cascade(
            user_id=user_id,
            session_id=session_id or f"session_{int(datetime.now().timestamp())}",
            cascade_metadata={"error": str(e)},
            success=False,
            error_message=str(e),
        )

        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main_enhanced:app",
        host=model_muxer.config.host,
        port=model_muxer.config.port,
        reload=model_muxer.config.debug,
        log_level=model_muxer.config.logging.level.lower(),
    )
