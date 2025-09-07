# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Centralized sample response data for testing.

This module provides standardized response data that matches
real API responses for consistent mocking.
"""

import time
from typing import Dict, Any, List


# Standard OpenAI chat completion responses
OPENAI_CHAT_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": int(time.time()),
    "model": "gpt-3.5-turbo",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! I'm doing well, thank you for asking. How can I help you today?",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
}

# Anthropic Messages API responses
ANTHROPIC_MESSAGE_RESPONSE = {
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "Hello! I'm Claude, an AI assistant. I'm doing well and ready to help you with any questions or tasks you might have.",
        }
    ],
    "model": "claude-3-haiku-20240307",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 8, "output_tokens": 25},
}

# Streaming response chunks
STREAMING_CHUNKS = [
    {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "gpt-3.5-turbo",
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "gpt-3.5-turbo",
        "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
    },
    {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "gpt-3.5-turbo",
        "choices": [{"index": 0, "delta": {"content": "!"}, "finish_reason": "stop"}],
    },
]

# Error responses
BUDGET_EXCEEDED_ERROR = {
    "error": {
        "message": "Budget exceeded: estimated cost $0.15 exceeds limit $0.08",
        "type": "budget_exceeded",
        "code": "insufficient_budget",
        "details": {"limit": 0.08, "estimate": 0.15},
    }
}

PROVIDER_ERROR_RESPONSE = {
    "error": {
        "message": "Provider openai is not available",
        "type": "service_unavailable",
        "code": "provider_unavailable",
    }
}

VALIDATION_ERROR_RESPONSE = {
    "error": {
        "message": "Invalid request: messages cannot be empty",
        "type": "validation_error",
        "code": "invalid_request",
    }
}

# Health and metrics responses
HEALTH_RESPONSE = {
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": int(time.time()),
    "providers": {"openai": "available", "anthropic": "available"},
}

METRICS_RESPONSE = {
    "requests_total": 1000,
    "requests_success": 950,
    "requests_failed": 50,
    "avg_response_time_ms": 250.5,
    "total_cost_usd": 45.67,
    "providers": {
        "openai": {"requests": 600, "success_rate": 0.95, "avg_latency_ms": 200},
        "anthropic": {"requests": 400, "success_rate": 0.97, "avg_latency_ms": 300},
    },
}

# Provider and models listing
PROVIDERS_RESPONSE = {
    "providers": [
        {"name": "openai", "status": "available", "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]},
        {
            "name": "anthropic",
            "status": "available",
            "models": ["claude-3-haiku-20240307", "claude-3-sonnet-20240229"],
        },
    ]
}

MODELS_RESPONSE = {
    "models": [
        {
            "id": "gpt-3.5-turbo",
            "provider": "openai",
            "context_length": 4096,
            "cost_per_token": 0.000002,
        },
        {
            "id": "claude-3-haiku-20240307",
            "provider": "anthropic",
            "context_length": 200000,
            "cost_per_token": 0.00000025,
        },
    ]
}
