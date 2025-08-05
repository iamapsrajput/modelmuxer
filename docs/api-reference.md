# API Reference

ModelMuxer provides a comprehensive REST API for intelligent LLM routing and management.

## Base URL

```text
https://api.modelmuxer.com/api/v1
```

For local development:

```text
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

```text
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
  "total_cost": 12.5,
  "breakdown": {
    "by_provider": {
      "openai": { "requests": 800, "cost": 8.0 },
      "anthropic": { "requests": 450, "cost": 4.5 }
    },
    "by_model": {
      "gpt-4o-mini": { "requests": 600, "cost": 3.0 },
      "gpt-4o": { "requests": 200, "cost": 5.0 }
    }
  }
}
```

#### GET /v1/analytics/costs

Get detailed cost analytics and usage statistics.

**Response:**

```json
{
  "user_id": "user123",
  "period_days": 30,
  "total_cost": 15.75,
  "total_requests": 150,
  "cost_by_provider": {
    "openai": 10.5,
    "anthropic": 5.25
  },
  "cost_by_model": {
    "gpt-4o": 8.0,
    "claude-3-sonnet": 7.75
  },
  "daily_breakdown": [
    {
      "date": "2024-01-15",
      "cost": 0.85,
      "requests": 12
    }
  ]
}
```

### Budget Management

#### GET /v1/analytics/budgets

Get budget status and alerts for the authenticated user.

**Query Parameters:**

- `budget_type` (optional): Filter by budget type (daily, weekly, monthly, yearly)

**Response:**

```json
{
  "message": "Budget status retrieved successfully",
  "budgets": [
    {
      "budget_type": "daily",
      "budget_limit": 10.0,
      "current_usage": 2.5,
      "usage_percentage": 25.0,
      "remaining_budget": 7.5,
      "provider": null,
      "model": null,
      "alerts": [
        {
          "type": "warning",
          "message": "Budget usage at 25.0% (threshold: 50%)",
          "threshold": 50.0,
          "current_usage": 25.0
        }
      ],
      "period_start": "2024-01-15",
      "period_end": "2024-01-15"
    }
  ],
  "total_budgets": 1
}
```

#### POST /v1/analytics/budgets

Set budget limits and alert thresholds for the authenticated user.

**Request:**

```json
{
  "budget_type": "daily",
  "budget_limit": 15.0,
  "provider": null,
  "model": null,
  "alert_thresholds": [50.0, 80.0, 95.0]
}
```

**Response:**

```json
{
  "message": "Budget set successfully",
  "budget": {
    "budget_type": "daily",
    "budget_limit": 15.0,
    "provider": null,
    "model": null,
    "alert_thresholds": [50.0, 80.0, 95.0]
  }
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

```text
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
import { ModelMuxer } from "modelmuxer-js";

const client = new ModelMuxer({ apiKey: "your_api_key" });

const response = await client.chat.completions.create({
  messages: [{ role: "user", content: "Hello!" }],
  routing_strategy: "hybrid",
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
