# ModelMuxer - Intelligent LLM Router

ModelMuxer is an intelligent LLM routing service that optimizes cost and quality
by automatically selecting the best provider and model for each request. It uses
direct provider connections for optimal performance and reliability, providing
advanced features like cost tracking, caching, and intelligent routing.

## Features

- **Multi-Provider Support**: Direct connections to OpenAI, Anthropic, Mistral,
  Google, Groq, Cohere, Together AI
- **Intelligent Routing**: Automatic provider/model selection based on request
  characteristics
- **Cost Tracking**: Real-time cost monitoring and budget management
- **Caching**: Response caching for improved performance and cost savings
- **Enterprise Features**: Multi-tenancy, policy enforcement, and compliance
- **Observability**: Comprehensive metrics, tracing, and monitoring

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/modelmuxer.git
cd modelmuxer

# Install dependencies
poetry install

# Set up environment with direct provider API keys
cp .env.example .env
# Edit .env with your direct provider API keys:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# MISTRAL_API_KEY=...
# GOOGLE_API_KEY=...

# Run the server
poetry run python -m app.main
```

### Basic Usage

```bash
# Start the server
poetry run python -m app.main --mode basic

# Make a request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "model": "gpt-3.5-turbo"
  }'
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Direct Provider API Keys (Primary)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MISTRAL_API_KEY=...
GOOGLE_API_KEY=...
GROQ_API_KEY=gsk_...
COHERE_API_KEY=...
TOGETHER_API_KEY=...

# Provider Configuration (Direct connections only)
PROVIDER_ADAPTERS_ENABLED=true


# Intent Classifier (Phase 1)
ROUTER_INTENT_CLASSIFIER_ENABLED=true
INTENT_LOW_CONFIDENCE=0.4
INTENT_MIN_CONF_FOR_DIRECT=0.7

# Test Mode
TEST_MODE=false

# Pricing and Cost Estimation
PRICE_TABLE_PATH=./scripts/data/prices.json
LATENCY_PRIORS_WINDOW_S=1800
ESTIMATOR_DEFAULT_TOKENS_IN=400
ESTIMATOR_DEFAULT_TOKENS_OUT=300

# Budget Thresholds
MAX_ESTIMATED_USD_PER_REQUEST=0.08
```

### Deployment Modes

- **Basic Mode**: Direct provider routing with cost tracking
- **Enhanced Mode**: Advanced features with ML classification and caching
- **Production Mode**: Full enterprise features with monitoring and advanced
  routing

## Architecture: Direct Providers Only

ModelMuxer uses direct provider connections exclusively, offering:

- **Lower Latency**: Direct API calls without proxy overhead
- **Better Error Handling**: Provider-specific error handling and retry logic
- **Enhanced Control**: Fine-grained configuration per provider
- **Improved Observability**: Detailed telemetry and circuit breaker patterns

### Provider Requirements

**At least one provider must be configured for ModelMuxer to function:**

- **OpenAI**: Set `OPENAI_API_KEY=sk-...` for GPT models
- **Anthropic**: Set `ANTHROPIC_API_KEY=sk-ant-...` for Claude models
- **Mistral**: Set `MISTRAL_API_KEY=...` for Mistral models
- **Google**: Set `GOOGLE_API_KEY=...` for Gemini models
- **Groq**: Set `GROQ_API_KEY=gsk_...` for Groq models
- **Together AI**: Set `TOGETHER_API_KEY=...` for Together AI models
- **Cohere**: Set `COHERE_API_KEY=...` for Cohere models

If no providers are configured:

- The service will log a warning at startup
- Requests will fail with a 503 error
- Check your API key configuration and ensure keys are valid

## API Reference

### Chat Completions

```http
POST /v1/chat/completions
```

Compatible with OpenAI's chat completions API. ModelMuxer will automatically
route to the optimal provider.

### Health Check

```http
GET /health
```

Returns service health status.

### Metrics

```http
GET /metrics/prometheus
```

Prometheus metrics endpoint.

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test categories
poetry run pytest tests/test_intent_classifier.py
poetry run pytest tests/test_routing.py
```

### Code Quality

```bash
# Format code
poetry run black .

# Lint code
poetry run ruff check .

# Type checking
poetry run mypy .
```

## Cost Estimation & Budget Management

ModelMuxer includes a comprehensive cost estimation and budget management system
that helps control spending and optimize model selection based on cost
constraints.

### Price Table

The system uses a centralized price table (`scripts/data/prices.json`)
containing current market rates for all supported providers and models. The
price table format is:

```json
{
  "provider:model": {
    "input_per_1k_usd": 2.5,
    "output_per_1k_usd": 10.0
  }
}
```

Prices are in USD per 1k tokens and use the mtoks = tokens/1000 formula. The
system automatically loads and validates this price table on startup.

### Latency Priors

The system maintains latency priors for each model using a ring buffer of recent
measurements. This provides p95 and p99 percentile estimates for ETA
calculation, helping with both cost and performance optimization.

### Budget Gate

The budget gate enforces cost constraints before routing decisions:

- **Pre-request Estimation**: Estimates cost using token heuristics and current
  prices
- **Budget Enforcement**: Blocks requests that exceed
  `MAX_ESTIMATED_USD_PER_REQUEST`
- **Down-routing**: Automatically selects cheaper models when budget allows
- **Structured Errors**: Returns HTTP 402 with detailed cost information when
  budget exceeded

### Configuration

Configure budget constraints and estimation parameters:

```bash
# Budget threshold (typical values: 0.05 conservative, 0.08 balanced, 0.15 permissive)
MAX_ESTIMATED_USD_PER_REQUEST=0.08

# Latency measurement window (30 minutes default)
LATENCY_PRIORS_WINDOW_S=1800

# Default token estimates when not provided
ESTIMATOR_DEFAULT_TOKENS_IN=400
ESTIMATOR_DEFAULT_TOKENS_OUT=300
```

### Error Response Format

ModelMuxer uses a standardized error response format for all API errors. All
error responses include an `error` object with consistent structure:

```json
{
  "error": {
    "message": "Human-readable error description",
    "type": "error_category",
    "code": "specific_error_code",
    "details": {
      // Additional error-specific information
    }
  }
}
```

#### Budget Exceeded Errors (HTTP 402)

```json
{
  "error": {
    "message": "Budget exceeded: No models within budget limit of $0.08",
    "type": "budget_exceeded",
    "code": "insufficient_budget",
    "details": {
      "limit": 0.08,
      "estimate": 0.12
    }
  }
}
```

#### Validation Errors (HTTP 400)

```json

```

#### Authentication Errors (HTTP 401)

```json
{
  "error": {
    "message": "Invalid API key provided.",
    "type": "authentication_error",
    "code": "invalid_api_key",
    "details": {}
  }
}
```

#### Rate Limiting Errors (HTTP 429)

```json
{
  "error": {
    "message": "Rate limit exceeded: 100/100 requests per minute",
    "type": "rate_limit_exceeded",
    "code": "security_rate_limit",
    "details": {
      "current": 100,
      "limit": 100,
      "window": "minute"
    }
  }
}
```

#### Provider Errors (HTTP 502)

```json
{
  "error": {
    "message": "Provider error: OpenAI API returned 429",
    "type": "provider_error",
    "code": "provider_error",
    "details": {}
  }
}
```

#### Service Unavailable Errors (HTTP 503)

```json
{
  "error": {
    "message": "Provider openai is not available",
    "type": "service_unavailable",
    "code": "provider_unavailable",
    "details": {}
  }
}
```

### Integration with Existing Systems

The new cost estimation system works alongside the existing cost tracking
system:

- **Pre-request Estimation**: New system estimates costs before routing
- **Post-request Tracking**: Existing system tracks actual costs after
  completion
- **Telemetry Integration**: Both systems contribute to monitoring and metrics
- **Backward Compatibility**: Existing cost tracking continues to work unchanged

### Response Headers

When debug mode is enabled (`SERVER_DEBUG=true`), the API includes additional
headers for observability:

- **`X-Route-Decision`**: Shows the selected provider and model (e.g.,
  `openai:gpt-4o-mini`)
- **`X-Route-Estimate-USD`**: Shows the estimated cost in USD (e.g., `0.000150`)

**Note**: These headers are non-contractual and may change without notice. They
are intended for debugging and monitoring purposes only.

## Phase 1: Intent Classifier

The Routing Mind intent classifier is the first building block for intelligent
routing. It analyzes each request and tags it with a task label and confidence
score before routing decisions are made.

### Features

- **Lightweight Classification**: Uses heuristics by default, with optional
  cheap LLM integration
- **Deterministic Results**: Test mode ensures reproducible behavior
- **Feature Extraction**: Extracts lexical and structural signals from prompts
- **Telemetry Integration**: OpenTelemetry spans and Prometheus metrics

### Intent Labels

The classifier supports 7 intent labels:

- `chat_lite`: Simple conversation and basic questions
- `deep_reason`: Complex analysis, explanations, and reasoning
- `code_gen`: Code generation and programming tasks
- `json_extract`: JSON parsing and structured data extraction
- `translation`: Language translation tasks
- `vision`: Image analysis and OCR tasks
- `safety_risk`: Potentially harmful content detection

### Configuration

```bash
# Enable/disable the classifier
ROUTER_INTENT_CLASSIFIER_ENABLED=true

# Confidence thresholds
INTENT_LOW_CONFIDENCE=0.4
INTENT_MIN_CONF_FOR_DIRECT=0.7
```

### Usage

The classifier runs automatically on each request and attaches intent metadata
to the response:

```json
{
  "router_metadata": {
    "selected_provider": "openai",
    "selected_model": "gpt-3.5-turbo",
    "routing_reason": "Simple query detected",
    "intent_label": "chat_lite",
    "intent_confidence": 0.85,
    "intent_signals": {
      "token_length_est": 45.2,
      "has_code_fence": false,
      "has_programming_keywords": false,
      "signals": {
        "code": false,
        "translation": false,
        "vision": false,
        "safety": false
      }
    }
  }
}
```

### Testing

The classifier includes comprehensive tests with a dataset of 60 labeled
examples:

```bash
# Run intent classifier tests
poetry run pytest tests/test_intent_classifier.py -v
```

The test suite validates:

- Accuracy ≥ 80% on the labeled dataset
- Deterministic results in test mode
- Proper handling of disabled feature flag
- Confidence score validation
- Feature signal extraction

### Architecture

The intent classifier consists of:

1. **Feature Extraction** (`app/core/features.py`): Extracts lexical and
   structural signals
2. **Intent Classification** (`app/core/intent.py`): Heuristic classification
   with LLM fallback
3. **Integration** (`app/main.py`): Wired into request flow with telemetry
4. **Telemetry** (`app/telemetry/metrics.py`): Prometheus counter and
   OpenTelemetry spans

### Future Enhancements

- Cheap LLM integration for improved accuracy
- Dynamic confidence thresholds based on model performance
- Intent-aware routing decisions
- A/B testing framework for intent strategies

## Monitoring and Metrics

ModelMuxer provides comprehensive Prometheus metrics for monitoring routing
decisions, cost estimation, and system performance.

### Key Metrics

- **`modelmuxer_router_cost_estimate_usd_sum`**: Total estimated costs by route,
  model, and budget status
  - Labels: `route`, `model`, `within_budget` (true/false)
  - The `within_budget` label helps analyze budget gating effectiveness
- **`modelmuxer_router_budget_exceeded_total`**: Budget exceeded events by route
  and reason
- **`modelmuxer_router_decision_latency_ms`**: Router decision latency
  distribution
- **`modelmuxer_provider_latency_seconds`**: Provider response latency by
  provider and model

### Budget Monitoring

The `within_budget` label in cost estimation metrics provides visibility into:

- How often models exceed budget thresholds
- Which models are most frequently down-routed due to cost
- Budget gating effectiveness across different request types

### Grafana Dashboard

A pre-configured Grafana dashboard is available in
`grafana/dashboard_modelmuxer.json` for visualizing these metrics.

## Known Limitations

### Latency Priors (In-Memory Only)

The current latency tracking system (`LatencyPriors`) is implemented as an
in-memory ring buffer that resets on application restart. This means:

- **Limitation**: All latency measurements are lost when the service restarts
- **Impact**: ETA estimates will fall back to defaults until new measurements
  are collected
- **Workaround**: For production deployments, consider implementing a
  Redis-backed version that persists measurements across restarts
- **Future Enhancement**: The interface is designed to be easily replaceable
  with a persistent backend

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of
conduct and the process for submitting pull requests.

## License

This project is licensed under the Business Source License 1.1 - see the
[LICENSE](LICENSE) file for details.

## Commercial Licensing

For commercial licensing and enterprise support, contact:

- Email: licensing@modelmuxer.com

## Support

For support and questions:

- Create an issue on GitHub
- Check the [documentation](docs/)
- Review [troubleshooting guide](docs/troubleshooting.md)
