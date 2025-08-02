# API Reference

ModelMuxer provides a comprehensive REST API for intelligent LLM routing and management.

## Base URL

```
https://api.modelmuxer.com/api/v1
```

For local development:
```
http://localhost:8000/api/v1
```

## Authentication

ModelMuxer supports multiple authentication methods:

### JWT Bearer Token

```bash
curl -H "Authorization: Bearer <jwt_token>" \
  https://api.modelmuxer.com/api/v1/chat/completions
```

### API Key

```bash
curl -H "X-API-Key: <api_key>" \
  https://api.modelmuxer.com/api/v1/chat/completions
```

## Core Endpoints

### Chat Completions

#### POST /chat/completions

Create a chat completion using intelligent routing.

**Request Body:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ],
  "routing_strategy": "hybrid",
  "budget_constraint": 0.01,
  "quality_threshold": 0.8,
  "max_tokens": 1000,
  "temperature": 0.7,
  "stream": false
}
```

**Response:**

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-4o-mini",
  "provider": "openai",
  "routing_info": {
    "strategy_used": "hybrid",
    "reason": "Cost-optimized selection for simple query",
    "alternatives_considered": 3,
    "estimated_cost": 0.0001
  },
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 7,
    "total_tokens": 20
  },
  "cost": {
    "input_cost": 0.0000195,
    "output_cost": 0.0000042,
    "total_cost": 0.0000237
  }
}
```

#### POST /chat/completions/stream

Stream chat completions with server-sent events.

**Request:** Same as above with `"stream": true`

**Response:** Server-sent events stream

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":" capital"},"finish_reason":null}]}

data: [DONE]
```

### Model Management

#### GET /models

List available models across all providers.

**Response:**

```json
{
  "data": [
    {
      "id": "gpt-4o",
      "object": "model",
      "provider": "openai",
      "capabilities": ["chat", "completion"],
      "max_tokens": 4096,
      "pricing": {
        "input": 0.005,
        "output": 0.015
      }
    }
  ]
}
```

#### GET /models/{provider}

List models for a specific provider.

#### GET /models/{provider}/{model}

Get details for a specific model.

### Routing

#### POST /routing/analyze

Analyze a prompt and get routing recommendations.

**Request:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Write a Python function to sort a list"
    }
  ],
  "budget_constraint": 0.01
}
```

**Response:**

```json
{
  "analysis": {
    "task_type": "code",
    "complexity": "medium",
    "estimated_tokens": 150,
    "code_confidence": 0.85,
    "complexity_confidence": 0.6
  },
  "recommendations": [
    {
      "provider": "openai",
      "model": "gpt-4o",
      "confidence": 0.9,
      "estimated_cost": 0.0075,
      "reason": "Optimal for code generation tasks"
    },
    {
      "provider": "anthropic",
      "model": "claude-3-5-sonnet-20241022",
      "confidence": 0.85,
      "estimated_cost": 0.0045,
      "reason": "Good balance of quality and cost"
    }
  ]
}
```

#### PUT /routing/strategy

Update the default routing strategy.

**Request:**

```json
{
  "strategy": "cascade",
  "parameters": {
    "quality_threshold": 0.8,
    "max_budget": 0.05
  }
}
```

### Cost Tracking

#### GET /usage

Get usage statistics for the authenticated user.

**Response:**

```json
{
  "user_id": "user_123",
  "period": "current_month",
  "total_requests": 1250,
  "total_tokens": 125000,
  "total_cost": 12.50,
  "breakdown": {
    "by_provider": {
      "openai": {"requests": 800, "cost": 8.00},
      "anthropic": {"requests": 450, "cost": 4.50}
    },
    "by_model": {
      "gpt-4o-mini": {"requests": 600, "cost": 3.00},
      "gpt-4o": {"requests": 200, "cost": 5.00}
    }
  }
}
```

#### GET /usage/budget

Get budget status and alerts.

**Response:**

```json
{
  "daily_budget": 10.00,
  "daily_used": 2.50,
  "daily_remaining": 7.50,
  "monthly_budget": 300.00,
  "monthly_used": 75.00,
  "monthly_remaining": 225.00,
  "alerts": [
    {
      "type": "warning",
      "message": "25% of daily budget used",
      "threshold": 0.25
    }
  ]
}
```

#### POST /usage/budget

Set budget limits.

**Request:**

```json
{
  "daily_limit": 15.00,
  "monthly_limit": 400.00,
  "alert_thresholds": [0.5, 0.8, 0.95]
}
```

### Health & Status

#### GET /health

Basic health check.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

#### GET /health/detailed

Detailed health check with component status.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "database": {"status": "healthy", "response_time": "5ms"},
    "cache": {"status": "healthy", "hit_rate": 0.85},
    "providers": {
      "openai": {"status": "healthy", "latency": "250ms"},
      "anthropic": {"status": "healthy", "latency": "180ms"}
    }
  }
}
```

#### GET /metrics

Prometheus metrics endpoint.

## Error Handling

ModelMuxer uses standard HTTP status codes and provides detailed error messages.

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request is invalid",
    "details": {
      "field": "messages",
      "issue": "messages array cannot be empty"
    },
    "request_id": "req_123456789"
  }
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `502 Bad Gateway`: Provider error
- `503 Service Unavailable`: Service temporarily unavailable

## Rate Limits

Default rate limits per user:

- **Requests per minute**: 60
- **Requests per hour**: 1000
- **Requests per day**: 10000

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1677652348
```

## SDKs and Libraries

### Python SDK

```bash
pip install modelmuxer-python
```

```python
from modelmuxer import ModelMuxer

client = ModelMuxer(api_key="your_api_key")

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    routing_strategy="hybrid"
)
```

### JavaScript SDK

```bash
npm install modelmuxer-js
```

```javascript
import { ModelMuxer } from 'modelmuxer-js';

const client = new ModelMuxer({ apiKey: 'your_api_key' });

const response = await client.chat.completions.create({
  messages: [{ role: 'user', content: 'Hello!' }],
  routing_strategy: 'hybrid'
});
```

## OpenAPI Specification

The complete OpenAPI specification is available at:

- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## Next Steps

- [Configuration Guide](configuration.md)
- [Deployment Guide](deployment.md)
- [SDK Documentation](sdks.md)
