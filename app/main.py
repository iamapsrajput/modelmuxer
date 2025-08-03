# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Main FastAPI application for the LLM Router.
"""

import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

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

# Provider instances
providers = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """Application lifespan manager."""
    # Startup
    print("🚀 Starting ModelMuxer LLM Router...")

    # Initialize database
    await db.init_database()
    print("✅ Database initialized")

    # Initialize providers
    try:
        providers["openai"] = OpenAIProvider()
        print("✅ OpenAI provider initialized")
    except Exception as e:
        print(f"⚠️  OpenAI provider failed to initialize: {e}")

    try:
        providers["anthropic"] = AnthropicProvider()
        print("✅ Anthropic provider initialized")
    except Exception as e:
        print(f"⚠️  Anthropic provider failed to initialize: {e}")

    try:
        providers["mistral"] = MistralProvider()
        print("✅ Mistral provider initialized")
    except Exception as e:
        print(f"⚠️  Mistral provider failed to initialize: {e}")

    if not providers:
        print("❌ No providers initialized! Check your API keys.")

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
allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080,https://modelmuxer.com").split(
    ","
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Secure: specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Specific methods only
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],  # Specific headers only
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
        # Sanitize input messages
        for message in request.messages:
            message.content = sanitize_user_input(message.content)

        # Route the request to the best provider/model
        provider_name, model_name, routing_reason = router.select_model(messages=request.messages, user_id=user_id)

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
                stream_chat_completion(provider, request, model_name, routing_reason, user_id, start_time),
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
                    if k not in ["messages", "model", "max_tokens", "temperature", "stream"] and v is not None
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.debug)
