# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""Chat completion endpoints (OpenAI-compatible plus Anthropic compatibility)."""

import json
import time
from datetime import datetime
from typing import Any, cast

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Shared singletons (router, db, model_muxer, settings, ...) are resolved
# through app.main at call time so tests can keep patching app.main.<name>.
import app.main as app_main
from app.core.exceptions import BudgetExceededError
from app.core.validation_helpers import validate_model_format
from app.main import get_authenticated_user
from app.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ErrorResponse,
    RouterMetadata,
    Usage,
)
from app.providers.base import AuthenticationError, ProviderError, RateLimitError
from app.telemetry.metrics import LLM_ROUTER_BUDGET_EXCEEDED_TOTAL

router = APIRouter()


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest, user_info: dict[str, Any] = Depends(get_authenticated_user)
):
    """
    Create a chat completion using the optimal LLM provider.

    This endpoint is compatible with OpenAI's chat completions API.
    """
    start_time = time.time()
    user_id = user_info["user_id"]
    app_settings = app_main.app_settings
    logger = app_main.logger

    try:
        # Sanitize input messages early
        for message in request.messages:
            message.content = app_main.sanitize_user_input(message.content)

        # Enforce policies early so metrics/redactions run even if providers are unavailable later
        tenant_id = user_id or "anonymous"
        policy_result = app_main.enforce_policies(request, tenant_id)
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

        # Validate model name format only in production (tests use provider-native IDs like Together with '/')
        if app_settings.features.mode == "production" and request.model:
            validate_model_format(request.model)

        # Check providers at request time
        current_registry = app_main.providers_registry.get_provider_registry()
        if app_settings.server.debug:
            logger.debug("Provider registry at request time: %s", list(current_registry.keys()))

        if not current_registry:
            raise HTTPException(
                status_code=503,
                detail=ErrorResponse.create(
                    message="Service unavailable: No providers available",
                    error_type="service_unavailable",
                    code="no_providers_available",
                ).dict(),
            )

        # Intent classification is now handled in the router core

        # Route the request to the best provider/model
        if app_settings.server.debug:
            logger.debug(
                "About to call router.select_model with router type: %s", type(app_main.router)
            )
        # In non-production/testing, re-instantiate router so patched HeuristicRouter is used
        if app_settings.features.mode != "production":
            try:
                _new_router = app_main.HeuristicRouter(
                    provider_registry_fn=app_main.providers_registry.get_provider_registry
                )
                # use the new instance for this request
                _active_router = _new_router
            except Exception:
                _active_router = app_main.router
        else:
            _active_router = app_main.router
        try:
            (
                provider_name,
                model_name,
                routing_reason,
                intent_metadata,
                estimate_metadata,
            ) = await _active_router.select_model(
                messages=request.messages,
                user_id=user_id,
                max_tokens=request.max_tokens,
                budget_constraint=getattr(request, "max_budget", None),
            )
            if app_settings.server.debug:
                logger.debug(
                    "Router selected - provider: %s, model: %s, reason: %s",
                    provider_name,
                    model_name,
                    routing_reason,
                )
                logger.debug(
                    "Cost estimate: $%.4f, ETA: %dms",
                    estimate_metadata["usd"],
                    estimate_metadata["eta_ms"],
                )
        except BudgetExceededError as e:
            if app_settings.server.debug:
                logger.debug("Budget exceeded: %s", e)
            # Record budget exceeded metric
            LLM_ROUTER_BUDGET_EXCEEDED_TOTAL.labels("chat", "budget_exceeded").inc()

            # Special case for no_pricing reason
            if e.reason == "no_pricing":
                raise HTTPException(
                    status_code=402,
                    detail=ErrorResponse.create(
                        message=f"Budget exceeded: {e.message}",
                        error_type="budget_exceeded",
                        code="no_pricing",
                        details={"limit": e.limit, "estimate": None},
                    ).dict(),
                ) from e
            else:
                raise HTTPException(
                    status_code=402,
                    detail=ErrorResponse.create(
                        message=f"Budget exceeded: {e.message}",
                        error_type="budget_exceeded",
                        code="budget_exceeded",  # Standardize on budget_exceeded
                        details={
                            "limit": e.limit,
                            "estimate": e.estimates[0][1] if e.estimates else 0.0,
                        },
                    ).dict(),
                ) from e
        except Exception as e:
            if app_settings.server.debug:
                logger.debug("Router selection failed: %s", e)
            raise

        # Get provider adapter from registry
        provider_registry = app_main.providers_registry.get_provider_registry()
        if provider_name not in provider_registry:
            if app_settings.server.debug:
                logger.debug("Provider %s not found in provider registry!", provider_name)
            raise HTTPException(
                status_code=503,
                detail=ErrorResponse.create(
                    message=f"Provider {provider_name} is not available",
                    error_type="service_unavailable",
                    code="provider_unavailable",
                ).dict(),
            )

        provider = provider_registry[provider_name]
        if app_settings.server.debug:
            logger.debug("Successfully got provider adapter for: %s", provider_name)

        # Use router's cost estimate from estimate_metadata when available
        # Skip redundant cost estimation since router already provides accurate estimates
        router_cost_estimate = estimate_metadata.get("usd")
        if app_settings.server.debug:
            logger.debug("Router cost estimate: $%.4f", router_cost_estimate)

        # Only fall back to cost tracker if router estimate is not available
        cost_estimate = router_cost_estimate
        if cost_estimate is None:
            active_cost_tracker = app_main.model_muxer.cost_tracker
            if app_settings.server.debug:
                logger.debug(
                    "Router estimate not available, falling back to cost_tracker: %s",
                    type(active_cost_tracker),
                )
            try:
                cost_estimate = active_cost_tracker.estimate_request_cost(
                    messages=request.messages,
                    provider=provider_name,
                    model=model_name,
                    max_tokens=request.max_tokens,
                )
                if app_settings.server.debug:
                    logger.debug("Cost tracker estimate: $%.4f", cost_estimate)
            except Exception as cost_exc:
                if app_settings.server.debug:
                    logger.debug("Cost estimation failed: %s", cost_exc)
                    logger.debug("Cost exception type: %s", type(cost_exc).__name__)
                # Continue without cost tracking if it fails
                cost_estimate = 0.0

        # Note: Latency priors will be updated after actual provider invocation
        # to use measured round-trip time instead of estimate

        # Handle streaming vs non-streaming
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(
                    provider_name, request, model_name, routing_reason, user_id, start_time
                ),
                media_type="text/event-stream",  # Correct media type for SSE
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            # Make the request to the provider
            if app_settings.server.debug:
                logger.debug(
                    "About to call router.invoke_via_adapter with provider: %s", provider_name
                )
                logger.debug("Model name: %s", model_name)
                logger.debug("Number of messages: %d", len(request.messages))
                logger.debug("Max tokens: %s", request.max_tokens)
                logger.debug("Temperature: %s", request.temperature)

            try:
                if app_settings.features.provider_adapters_enabled:
                    # Use router's unified adapter interface for consistency
                    prompt_text = "\n".join(
                        [msg.content for msg in request.messages if msg.content]
                    )
                    adapter_resp = await _active_router.invoke_via_adapter(
                        provider=provider_name,
                        model=model_name,
                        prompt=prompt_text,
                        temperature=request.temperature or app_settings.router.temperature_default,
                        max_tokens=request.max_tokens,
                        user_id=user_id,
                        metadata={
                            k: v
                            for k, v in request.dict().items()
                            if k not in ["messages", "model", "max_tokens", "temperature", "stream"]
                            and v is not None
                        },
                    )
                    if adapter_resp.error:
                        raise ProviderError(adapter_resp.error)
                    # Update latency priors with the measured provider latency
                    try:
                        latency_key = f"{provider_name}:{model_name}"
                        _active_router.record_latency(
                            latency_key, int(getattr(adapter_resp, "latency_ms", 0) or 0)
                        )
                    except Exception as e:
                        if app_settings.server.debug:
                            logger.debug("Failed to update latency priors: %s", e)
                    # Build ChatCompletionResponse
                    # Use router's cost estimate when available, otherwise calculate from actual tokens
                    if cost_estimate is not None:
                        estimated_cost = cost_estimate
                    else:
                        estimated_cost = app_main.cost_tracker.calculate_cost(
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
                                message=ChatMessage(
                                    role="assistant", content=adapter_resp.output_text, name=None
                                ),
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
                            estimated_cost=float(estimate_metadata.get("usd") or 0.0),
                            response_time_ms=float(getattr(adapter_resp, "latency_ms", 0) or 0),
                            intent_label=(
                                str(intent_metadata.get("label")) if intent_metadata else None
                            ),
                            intent_confidence=(
                                float(intent_metadata.get("confidence", 0.0))
                                if intent_metadata
                                else None
                            ),
                            intent_signals=(
                                cast(dict[str, object], intent_metadata.get("signals", {}))
                                if intent_metadata
                                else None
                            ),
                            estimated_cost_usd=float(estimate_metadata.get("usd") or 0.0),
                            estimated_eta_ms=int(estimate_metadata.get("eta_ms") or 0),
                            estimated_tokens_in=int(estimate_metadata.get("tokens_in") or 0),
                            estimated_tokens_out=int(estimate_metadata.get("tokens_out") or 0),
                            direct_providers_only=True,
                        ),
                    )
                    # Add flat keys expected by tests
                    resp_dict = response.dict()
                    resp_dict["router_metadata"]["provider"] = provider_name
                    resp_dict["router_metadata"]["model"] = model_name
                    return JSONResponse(content=resp_dict)
                else:
                    # Use router's adapter interface for consistent invocation
                    prompt_text = "\n".join(
                        [msg.content for msg in request.messages if msg.content]
                    )

                    # Measure actual provider latency
                    provider_start_time = time.perf_counter()
                    response = await app_main.router.invoke_via_adapter(
                        provider=provider_name,
                        model=model_name,
                        prompt=prompt_text,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                    )
                    provider_latency_ms = int((time.perf_counter() - provider_start_time) * 1000)

                    # Update latency priors with actual measured latency (only on success)
                    try:
                        latency_key = f"{provider_name}:{model_name}"
                        _active_router.record_latency(latency_key, provider_latency_ms)
                    except Exception as e:
                        if app_settings.server.debug:
                            logger.debug("Failed to update latency priors: %s", e)

                if app_settings.server.debug:
                    logger.debug("Successfully got response from provider: %s", type(response))
            except Exception as provider_exc:
                if app_settings.server.debug:
                    logger.debug("Provider call failed: %s", provider_exc)
                    logger.debug("Provider exception type: %s", type(provider_exc).__name__)
                raise

            # Update routing reason in metadata and attach intent
            response.router_metadata.routing_reason = routing_reason
            response.router_metadata.direct_providers_only = True
            try:
                response.router_metadata.intent_label = intent_metadata.get("label")
                response.router_metadata.intent_confidence = float(
                    intent_metadata.get("confidence", 0.0)
                )
                response.router_metadata.intent_signals = intent_metadata.get("signals")
                response.router_metadata.estimated_cost_usd = estimate_metadata.get("usd")
                response.router_metadata.estimated_eta_ms = estimate_metadata.get("eta_ms")
                response.router_metadata.estimated_tokens_in = estimate_metadata.get("tokens_in")
                response.router_metadata.estimated_tokens_out = estimate_metadata.get("tokens_out")
            except Exception:
                pass

            # Log the request using the advanced cost tracker if available
            if app_main.model_muxer.advanced_cost_tracker:
                await app_main.model_muxer.advanced_cost_tracker.log_simple_request(
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
                await app_main.db.log_request(
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

            # Convert response to JSONResponse to add X-Route-Decision header
            response_dict = response.dict()
            json_response = JSONResponse(content=response_dict)

            # Add X-Route-Decision header with routing information
            routing_decision = f"{provider_name}:{model_name}"
            json_response.headers["X-Route-Decision"] = routing_decision

            # Add cost estimate header when debug mode is enabled
            if app_settings.server.debug and estimate_metadata.get("usd") is not None:
                json_response.headers["X-Route-Estimate-USD"] = f"{estimate_metadata['usd']:.6f}"

            return json_response

    except ProviderError as e:
        # Log failed request
        await app_main.db.log_request(
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

    except HTTPException:
        # Let FastAPI handle mapped HTTP errors (e.g., 402 budget exceeded)
        raise
    except Exception as e:
        # Log unexpected error
        try:
            structlog.get_logger().exception("Unhandled error in chat_completions", error=str(e))
        except Exception:
            pass
        await app_main.db.log_request(
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
    provider_name, request, model_name, routing_reason, user_id, start_time
):
    """Handle streaming chat completion."""
    app_settings = app_main.app_settings
    logger = app_main.logger
    try:
        # Get provider from registry
        provider_registry = app_main.providers_registry.get_provider_registry()
        provider = provider_registry[provider_name]

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

        # Update latency priors with actual measured latency (only on success)
        response_time_ms = (time.time() - start_time) * 1000
        try:
            # Normalize key to match estimator convention
            latency_key = f"{provider_name}:{model_name}"
            app_main.router.record_latency(latency_key, int(response_time_ms))
        except Exception as e:
            if app_settings.server.debug:
                logger.debug("Failed to update latency priors: %s", e)
            # Continue without latency tracking if it fails

        # Log streaming request (with estimated tokens)
        estimated_tokens = app_main.cost_tracker.count_tokens(
            request.messages, provider_name, model_name
        )
        estimated_cost = app_main.cost_tracker.calculate_cost(
            provider_name, model_name, estimated_tokens, 100
        )

        await app_main.db.log_request(
            user_id=user_id,
            provider=provider_name,
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
        error_chunk = {
            "error": {"message": "An internal error occurred.", "type": "provider_error"}
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


# =============================================================================
# ANTHROPIC API COMPATIBILITY
# =============================================================================


@router.post("/v1/messages")
@router.post("/messages")
async def anthropic_messages(
    request: Request,
    beta: bool = False,  # Support for beta parameter
    user_info: dict[str, Any] = Depends(get_authenticated_user),
):
    """Anthropic Messages API compatibility endpoint with beta support."""
    logger = app_main.logger
    try:
        # Parse Anthropic format request
        body = await request.json()

        # Convert Anthropic format to OpenAI format
        messages = []
        if "system" in body:
            messages.append({"role": "system", "content": body["system"]})

        if "messages" in body:
            messages.extend(body["messages"])

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

        model_name = body.get("model", "claude-3-5-sonnet-20241022")

        # Validate model name format (reject proxy-style prefixes)
        validate_model_format(model_name)

        # Create OpenAI format request
        openai_request = ChatCompletionRequest(
            messages=converted_messages,
            model=model_name,
            max_tokens=body.get("max_tokens", 1000),
            temperature=body.get("temperature", 0.7),
            stream=body.get("stream", False),
        )

        # Route through normal chat completions (resolved via app.main so
        # tests patching app.main.chat_completions keep working)
        response = await app_main.chat_completions(openai_request, user_info)

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
                "end_turn"
                if response.choices[0].finish_reason == "stop"
                else response.choices[0].finish_reason
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
        # Check if it's a validation error (typically from Pydantic)
        error_msg = str(e)
        if "validation error" in error_msg.lower() or "literal_error" in error_msg.lower():
            logger.warning("anthropic_validation_error", error=error_msg)
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": error_msg},
                },
            )

        logger.error("anthropic_api_error", error=error_msg, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {"type": "internal_server_error", "message": error_msg},
            },
        )
