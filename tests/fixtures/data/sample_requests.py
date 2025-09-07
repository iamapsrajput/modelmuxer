# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Centralized sample request data for testing.

This module provides standardized request objects and data
for consistent testing across all test files.
"""

from app.models import ChatCompletionRequest, ChatMessage


# Standard chat completion requests
SIMPLE_CHAT_REQUEST = ChatCompletionRequest(
    messages=[ChatMessage(role="user", content="Hello, how are you?")],
    model="gpt-3.5-turbo",
    max_tokens=100,
    temperature=0.7,
)

COMPLEX_CHAT_REQUEST = ChatCompletionRequest(
    messages=[
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Explain quantum computing in detail."),
    ],
    model="gpt-4",
    max_tokens=500,
    temperature=0.3,
    top_p=0.9,
    presence_penalty=0.1,
    frequency_penalty=0.1,
)

STREAMING_CHAT_REQUEST = ChatCompletionRequest(
    messages=[ChatMessage(role="user", content="Tell me a story")],
    model="gpt-3.5-turbo",
    max_tokens=200,
    stream=True,
)

ANTHROPIC_MESSAGE_REQUEST = {
    "model": "claude-3-haiku-20240307",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello, Claude!"}],
}

ANTHROPIC_SYSTEM_MESSAGE_REQUEST = {
    "model": "claude-3-haiku-20240307",
    "max_tokens": 100,
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Hello, Claude!"}],
}

ANTHROPIC_MULTIPART_REQUEST = {
    "model": "claude-3-haiku-20240307",
    "max_tokens": 150,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {"type": "text", "text": "Please describe it in detail."},
            ],
        }
    ],
}

# Empty and invalid requests for error testing
EMPTY_MESSAGES_REQUEST = ChatCompletionRequest(
    messages=[],
    model="gpt-3.5-turbo",
)

INVALID_MODEL_REQUEST = ChatCompletionRequest(
    messages=[ChatMessage(role="user", content="Hello")],
    model="invalid-model-name",
)

# High-cost request for budget testing
HIGH_COST_REQUEST = ChatCompletionRequest(
    messages=[
        ChatMessage(
            role="user",
            content="Write a very long detailed essay about artificial intelligence, machine learning, deep learning, neural networks, and their applications in modern technology. Include examples, case studies, and future predictions. Make it at least 2000 words.",
        )
    ],
    model="gpt-4",
    max_tokens=4000,
    temperature=0.7,
)
