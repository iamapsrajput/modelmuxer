# ModelMuxer - Intelligent LLM Router

ModelMuxer is an intelligent LLM routing service that optimizes cost and quality
by automatically selecting the best provider and model for each request. It
supports multiple providers (OpenAI, Anthropic, Mistral, LiteLLM) and provides
advanced features like cost tracking, caching, and intelligent routing.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, Mistral, LiteLLM proxy
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

# Set up environment
cp .env.example .env
# Edit .env with your API keys

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
# Provider API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MISTRAL_API_KEY=...

# LiteLLM Proxy (optional)
LITELLM_BASE_URL=http://localhost:4000
LITELLM_API_KEY=...

# Intent Classifier (Phase 1)
ROUTER_INTENT_CLASSIFIER_ENABLED=true
INTENT_LOW_CONFIDENCE=0.4
INTENT_MIN_CONF_FOR_DIRECT=0.7

# Test Mode
TEST_MODE=false
```

### Deployment Modes

- **Basic Mode**: Simple routing with cost tracking
- **Enhanced Mode**: Advanced features including ML classification and caching
- **Production Mode**: Full enterprise features with monitoring

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

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of
conduct and the process for submitting pull requests.

## License

This project is licensed under the Business Source License 1.1 - see the
[LICENSE](LICENSE) file for details.

## Support

For support and questions:

- Create an issue on GitHub
- Check the [documentation](docs/)
- Review [troubleshooting guide](docs/troubleshooting.md)
