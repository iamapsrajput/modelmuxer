# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Centralized response mock configurations.

This module provides standardized API response mocks that match
OpenAI and Anthropic API formats for consistent testing.
"""

import time
from typing import Any, Dict, List

from app.models import (
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    RouterMetadata,
    Usage,
)


def create_openai_chat_response(
    content: str = "Test response",
    model: str = "gpt-3.5-turbo",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    provider: str = "openai",
    routing_reason: str = "test",
) -> ChatCompletionResponse:
    """Create a standardized OpenAI-compatible chat completion response."""
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        router_metadata=RouterMetadata(
            selected_provider=provider,
            selected_model=model,
            routing_reason=routing_reason,
            estimated_cost=0.01,
            response_time_ms=100.0,
        ),
    )


def create_anthropic_message_response(
    content: str = "Test response",
    model: str = "claude-3-haiku-20240307",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> Dict[str, Any]:
    """Create a standardized Anthropic Messages API response."""
    return {
        "id": f"msg_{int(time.time())}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def create_streaming_chunk(
    delta_content: str = "chunk",
    finish_reason: str = None,
    index: int = 0,
) -> Dict[str, Any]:
    """Create a streaming response chunk."""
    chunk = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": index,
                "delta": {"content": delta_content} if delta_content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return chunk


def create_error_response(
    message: str = "Test error",
    error_type: str = "test_error",
    code: str = "test_code",
    status_code: int = 500,
) -> Dict[str, Any]:
    """Create a standardized error response."""
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    }


def create_budget_exceeded_response(
    limit: float = 100.0,
    estimate: float = 150.0,
) -> Dict[str, Any]:
    """Create a budget exceeded error response."""
    return {
        "error": {
            "message": f"Budget exceeded: estimated cost ${estimate} exceeds limit ${limit}",
            "type": "budget_exceeded",
            "code": "insufficient_budget",
            "details": {
                "limit": limit,
                "estimate": estimate,
            },
        }
    }


def create_provider_unavailable_response(
    provider: str = "openai",
) -> Dict[str, Any]:
    """Create a provider unavailable error response."""
    return {
        "error": {
            "message": f"Provider {provider} is not available",
            "type": "service_unavailable",
            "code": "provider_unavailable",
        }
    }


# Standard response configurations
STANDARD_CHAT_RESPONSE = create_openai_chat_response()
STANDARD_ANTHROPIC_RESPONSE = create_anthropic_message_response()
STANDARD_ERROR_RESPONSE = create_error_response()
BUDGET_EXCEEDED_RESPONSE = create_budget_exceeded_response()
PROVIDER_ERROR_RESPONSE = create_provider_unavailable_response()
