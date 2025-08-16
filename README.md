# ModelMuxer™

## The Enterprise-Grade Intelligent LLM Routing Engine

[![CI](https://github.com/ajayrajput/modelmuxer/workflows/CI/badge.svg)](https://github.com/ajayrajput/modelmuxer/actions/workflows/ci.yml)
[![CodeQL](https://github.com/ajayrajput/modelmuxer/workflows/CodeQL/badge.svg)](https://github.com/ajayrajput/modelmuxer/actions/workflows/codeql.yml)
[![Coverage](https://codecov.io/gh/ajayrajput/modelmuxer/branch/main/graph/badge.svg)](https://codecov.io/gh/ajayrajput/modelmuxer)
[![Ruff](https://img.shields.io/badge/ruff-checked-brightgreen)](https://github.com/astral-sh/ruff)
[![Mypy](https://img.shields.io/badge/mypy-checked-brightgreen)](https://mypy-lang.org/)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-108%20Total-green.svg)](tests/)
[![Production Ready](https://img.shields.io/badge/Production-Ready-green.svg)](docs/deployment.md)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)

ModelMuxer is a production-ready, enterprise-grade LLM routing platform that
intelligently routes requests to the optimal AI model based on cost, quality,
and performance requirements. Built for scale, security, and efficiency.

## ✨ Key Features

- **🧠 Intelligent Routing**: Cascade, semantic, heuristic, and hybrid routing strategies
- **💰 Cost Optimization**: Real-time budget management and cost-aware model selection
- **🌐 Multi-Provider**: OpenAI, Anthropic, Google, Mistral, Groq, LiteLLM, and more
- **🔐 Enterprise Security**: JWT authentication, RBAC, and audit logging
- **📊 Observability**: Comprehensive metrics, tracing, and monitoring
- **☸️ Production Ready**: Kubernetes-native with high availability

## 🏗️ Architecture

ModelMuxer is built with a microservices architecture designed for enterprise scale:

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│  ModelMuxer API │────│   Provider APIs │
│    (NGINX)      │    │   (FastAPI)     │    │ (OpenAI, etc.)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌────────┴────────┐
                       │                 │
                ┌──────▼──────┐   ┌──────▼──────┐
                │ PostgreSQL  │   │    Redis    │
                │  (Primary)  │   │  (Cluster)  │
                └─────────────┘   └─────────────┘
```

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/iamapsrajput/ModelMuxer.git
cd ModelMuxer
poetry install && poetry shell

# Configure and run
cp .env.example .env  # Add your API keys
MODELMUXER_MODE=enhanced poetry run uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

📖 **[Complete Installation Guide](docs/installation.md)** | 🚀 **[Deployment Guide](docs/deployment.md)**

## 📚 API Documentation

### Core Endpoints

| Endpoint                        | Method   | Description                    |
| ------------------------------- | -------- | ------------------------------ |
| `/v1/chat/completions`          | POST     | Standard chat completion       |
| `/v1/chat/completions/enhanced` | POST     | Enhanced completion            |
| `/v1/analytics/costs`           | GET      | Detailed cost analytics        |
| `/v1/analytics/budgets`         | GET/POST | Budget management              |
| `/v1/providers`                 | GET      | Available providers and models |
| `/v1/models`                    | GET      | List all available models      |
| `/health`                       | GET      | System health check            |
| `/metrics`                      | GET      | Prometheus metrics             |

### Example Usage

#### Basic Chat Completion

```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions",
    headers={"Authorization": "Bearer your-jwt-token"},
    json={
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
)
```

#### Enhanced Cascade Routing

```python
response = requests.post("http://localhost:8000/v1/chat/completions/enhanced",
    headers={
        "Authorization": "Bearer your-jwt-token",
        "X-Routing-Strategy": "cost_optimized",
        "X-Max-Budget": "0.05"
    },
    json={
        "messages": [
            {"role": "user", "content": "Explain quantum computing"}
        ],
        "cascade_config": {
            "cascade_type": "quality_focused",
            "max_budget": 0.05,
            "quality_threshold": 0.8
        }
    }
)
```

#### Cost Analytics

```python
response = requests.get("http://localhost:8000/v1/analytics/costs",
    headers={"Authorization": "Bearer your-jwt-token"},
    params={
        "start_date": "2025-01-01",
        "end_date": "2025-01-31",
        "group_by": "provider"
    }
)
```

For complete API documentation, see our [OpenAPI Specification](docs/openapi/openapi.yaml).

## Setup Instructions

### Prerequisites

- Python 3.11 or higher
- API keys for at least one LLM provider (OpenAI, Anthropic, Mistral, or LiteLLM proxy)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ModelMuxer
   ```

2. **Install dependencies**

   ```bash
   poetry install --with dev,ml
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the server**

   ```bash
   # Enhanced mode (recommended for production)
   # Includes: ML routing, advanced cost tracking, monitoring, enterprise features
   MODELMUXER_MODE=enhanced uvicorn app.main:app --reload

   # Basic mode for development/testing
   # Includes: Basic routing, simple cost tracking, minimal features
   MODELMUXER_MODE=basic uvicorn app.main:app --reload
   ```

5. **Test the installation**

   ```bash
   python test_requests.py
   ```

### Environment Variables

Copy `.env.example` to `.env` and configure the following:

```bash
# Required: LLM Provider API Keys
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
MISTRAL_API_KEY=your-mistral-key-here

# Optional: LiteLLM Proxy Configuration
LITELLM_BASE_URL=http://localhost:4000
LITELLM_API_KEY=your-litellm-api-key-here

# Optional: Router Configuration
DEFAULT_MODEL=gpt-3.5-turbo
MAX_TOKENS_DEFAULT=1000
TEMPERATURE_DEFAULT=0.7

# Optional: Security
ALLOWED_API_KEYS=sk-test-key-1,sk-test-key-2

# Optional: Cost Limits (USD)
DEFAULT_DAILY_BUDGET=10.0
DEFAULT_MONTHLY_BUDGET=100.0
```

## Containerization

ModelMuxer supports multiple containerization platforms with auto-detection:

### **Quick Start (Auto-Detection)**

```bash
# Automatically detects and uses the best available containerization system
./scripts/container-auto.sh run
```

### **Apple Container (macOS 15+ Beta)**

```bash
# Check system compatibility
./scripts/apple-container-commands.sh check

# Build and run with Apple Container
./scripts/apple-container-commands.sh run

# Using Apple Container Compose
container compose -f container-compose.yaml up -d
```

### **Docker (Cross-Platform)**

```bash
# Using Docker Compose
docker-compose up -d

# Or build and run manually
docker build -t modelmuxer:latest .
docker run -d --name modelmuxer --env-file .env -p 8000:8000 modelmuxer:latest
```

For detailed containerization instructions and troubleshooting, see [docs/containerization-guide.md](docs/containerization-guide.md).

## API Usage

### Chat Completion API

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer sk-test-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

### Streaming Response

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer sk-test-key-1" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Count from 1 to 10"}
    ],
    "stream": true
  }'
```

### Python Client Example

```python
import httpx
import asyncio

async def chat_with_router():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/chat/completions",
            headers={"Authorization": "Bearer sk-test-key-1"},
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a Python function to calculate fibonacci"
                    }
                ],
                "max_tokens": 500
            }
        )

        data = response.json()
        print(f"Response: {data['choices'][0]['message']['content']}")
        print(f"Provider: {data['router_metadata']['selected_provider']}")
        print(f"Model: {data['router_metadata']['selected_model']}")
        print(f"Cost: ${data['router_metadata']['estimated_cost']:.6f}")

asyncio.run(chat_with_router())
```

## Docker Deployment

### Using Docker Compose

1. **Create environment file**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Build and run**

   ```bash
   docker-compose up --build
   ```

3. **Test the deployment**

   ```bash
   curl http://localhost:8000/health
   ```

### Manual Docker Build

```bash
# Build the image
docker build -t modelmuxer .

# Run the container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -e ANTHROPIC_API_KEY=your-key \
  -e MISTRAL_API_KEY=your-key \
  modelmuxer
```

## Monitoring and Metrics

### System Metrics

```bash
curl -H "Authorization: Bearer sk-test-key-1" \
  http://localhost:8000/metrics
```

### User Statistics

```bash
curl -H "Authorization: Bearer sk-test-key-1" \
  http://localhost:8000/user/stats
```

### Available Providers

```bash
curl -H "Authorization: Bearer sk-test-key-1" \
  http://localhost:8000/providers
```

## 🔗 LiteLLM Integration

ModelMuxer supports [LiteLLM](https://docs.litellm.ai/) as a unified proxy for accessing multiple LLM providers through a single endpoint. This is particularly useful for:

- **Unified API**: Access 100+ LLM models through one consistent interface
- **Cost Optimization**: Leverage LiteLLM's built-in cost tracking and optimization
- **Provider Abstraction**: Switch between providers without changing your code
- **Advanced Features**: Load balancing, fallbacks, and custom model configurations

### LiteLLM Setup

1. **Install and run LiteLLM proxy**:

   ```bash
   pip install litellm[proxy]
   litellm --config config.yaml
   ```

2. **Configure ModelMuxer**:

   ```bash
   # In your .env file
   LITELLM_BASE_URL=http://localhost:4000
   LITELLM_API_KEY=your-litellm-api-key-here  # Optional
   ```

3. **Example LiteLLM config.yaml**:
   ```yaml
   model_list:
     - model_name: gpt-3.5-turbo
       litellm_params:
         model: openai/gpt-3.5-turbo
         api_key: sk-your-openai-key
     - model_name: claude-3-haiku
       litellm_params:
         model: anthropic/claude-3-haiku-20240307
         api_key: sk-ant-your-anthropic-key
   ```

### LiteLLM Benefits in ModelMuxer

- **Seamless Integration**: Works with existing routing strategies
- **Cost Tracking**: Combines LiteLLM and ModelMuxer cost analytics
- **Fallback Support**: Automatic failover when LiteLLM proxy is unavailable
- **Custom Models**: Support for custom model configurations and pricing

## Routing Logic Details

The router uses heuristic analysis to select the optimal provider and model:

### Code Detection

- **Triggers**: Code blocks (```), inline code (`), programming keywords
- **Route to**: GPT-4o or Claude-Sonnet for high-quality code generation
- **Example**: "Write a Python function to sort a list"

### Complexity Analysis

- **Triggers**: Keywords like "analyze", "explain", "debug", "reasoning"
- **Route to**: Premium models (GPT-4o, Claude-Sonnet)
- **Example**: "Analyze the time complexity of merge sort"

### Simple Queries

- **Triggers**: Short prompts (<100 chars), simple question patterns
- **Route to**: Cost-effective models (Mistral-small, Claude-Haiku)
- **Example**: "What is 2+2?"

### General Queries

- **Default**: Balanced cost/quality models (GPT-3.5-turbo)
- **Example**: "Tell me about climate change"

## Cost Optimization

### Budget Management

- Daily and monthly budget limits per user
- Automatic cost estimation before requests
- Budget exceeded protection

### Cost Tracking

- Real-time cost calculation
- Usage analytics per user
- Provider cost comparison

### Model Pricing (per million tokens)

| Provider  | Model           | Input | Output |
| --------- | --------------- | ----- | ------ |
| OpenAI    | GPT-4o          | $5.00 | $15.00 |
| OpenAI    | GPT-3.5-turbo   | $0.50 | $1.50  |
| Anthropic | Claude-3-Sonnet | $3.00 | $15.00 |
| Anthropic | Claude-3-Haiku  | $0.25 | $1.25  |
| Mistral   | Mistral-Small   | $0.20 | $0.60  |

## Testing

### Run All Tests

```bash
# Integration tests
python test_requests.py

# Unit tests
python test_router.py
```

### Test Specific Scenarios

```bash
# Test with custom URL and API key
python test_requests.py --url http://localhost:8000 --api-key your-key
```

## Troubleshooting

### Common Issues

1. **"Provider not available" error**

   - Check that API keys are correctly set in `.env`
   - Verify API keys are valid and have sufficient credits
   - Check network connectivity to provider APIs

2. **"Budget exceeded" error**

   - Check daily/monthly budget limits in settings
   - Review usage with `/user/stats` endpoint
   - Adjust budget limits in `.env` file

3. **Authentication errors**

   - Verify API key format (should start with `sk-` for test keys)
   - Check `ALLOWED_API_KEYS` in configuration
   - Ensure Authorization header is properly formatted

4. **High response times**
   - Check provider API status
   - Consider using faster models for simple queries
   - Monitor system resources

### Debug Mode

Run with debug logging:

```bash
DEBUG=true uvicorn app.main:app --reload --log-level debug
```

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check provider availability
curl -H "Authorization: Bearer sk-test-key-1" \
  http://localhost:8000/providers
```

## Architecture

### Components

- **FastAPI Application**: Main API server with OpenAI-compatible endpoints
- **Heuristic Router**: Intelligent routing logic based on prompt analysis
- **Provider Adapters**: Unified interface for OpenAI, Anthropic, and Mistral
- **Cost Tracker**: Real-time cost calculation and budget management
- **Database Layer**: SQLite for request logging and usage tracking
- **Authentication**: API key-based authentication with rate limiting

### Data Flow

1. **Request Authentication**: Validate API key and check rate limits
2. **Prompt Analysis**: Analyze prompt characteristics (code, complexity, length)
3. **Model Selection**: Route to optimal provider/model based on analysis
4. **Budget Check**: Verify user has sufficient budget for estimated cost
5. **Provider Call**: Execute request with selected provider
6. **Response Processing**: Standardize response format and add metadata
7. **Logging**: Record request details, costs, and performance metrics

## Contributing

### Development Setup

1. Fork the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `poetry install --with dev,ml`
5. Run tests: `python test_router.py && python test_requests.py`

### Adding New Providers

1. Create new provider class in `app/providers/`
2. Inherit from `LLMProvider` base class
3. Implement required methods: `chat_completion`, `stream_chat_completion`, `calculate_cost`
4. Add provider to `app/providers/__init__.py`
5. Update configuration and pricing in `app/config.py`
6. Add tests for the new provider

## 📊 Performance & Benchmarks

ModelMuxer delivers exceptional performance with intelligent routing:

- **Cost Savings**: Up to 70% cost reduction through cascade routing
- **Response Time**: <200ms average routing decision time
- **Throughput**: 10,000+ requests/minute per instance
- **Availability**: 99.9% uptime with proper deployment
- **Quality**: Maintains 95%+ response quality with cost optimization

## 🔒 Security & Compliance

- **🔐 Enterprise Security**: JWT authentication, RBAC, audit logging
- **🛡️ PII Protection**: Automatic detection and redaction of sensitive data
- **📋 Compliance Ready**: GDPR, CCPA, SOC 2 compliance features
- **🔍 Security Scanning**: Automated vulnerability scanning in CI/CD
- **🏰 Network Security**: Kubernetes network policies and TLS encryption

## 📈 Monitoring & Observability

- **📊 Grafana Dashboards**: Pre-built dashboards for cost, performance, and health
- **🚨 Alerting**: Comprehensive alerting for budgets, errors, and performance
- **📝 Structured Logging**: JSON logs with correlation IDs and tracing
- **🔍 Distributed Tracing**: End-to-end request tracing with OpenTelemetry
- **📈 Custom Metrics**: Business metrics for cost optimization and quality

## 📖 Documentation

- **[Production Deployment Guide](docs/deployment/production-guide.md)**:
  Complete production setup
- **[API Documentation](docs/openapi/openapi.yaml)**: OpenAPI specification
- **[Security Guide](docs/security.md)**: Security configuration and practices
- **[Monitoring Guide](docs/monitoring.md)**: Observability setup and configuration
- **[Production Checklist](docs/deployment/production-checklist.md)**:
  Pre-deployment checklist

## 📄 License

ModelMuxer is licensed under the [Business Source License 1.1](LICENSE).

### 📋 License Summary

- ✅ **Non-commercial use**: Free for personal, academic, and research use
- ✅ **Evaluation**: Free to test and evaluate the software
- ✅ **Contributions**: Community contributions welcome under the same license
- ❌ **Commercial use**: Requires separate commercial license until January 1, 2027
- 🔄 **Future**: Automatically becomes Apache 2.0 licensed on January 1, 2027

### 💼 Commercial Licensing

For commercial use, enterprise licenses, or questions about licensing,
please open a GitHub Issue with the "licensing" label.

### 🏛️ Academic and Research Use

ModelMuxer is free for academic research, educational use, and non-commercial
research projects. Please cite the project in academic publications.

### 📜 Legal Documents

- [LICENSE](LICENSE) - Complete Business Source License 1.1 terms
- [COPYRIGHT](COPYRIGHT) - Copyright and ownership information
- [NOTICE](NOTICE) - Distribution and attribution requirements
- [TRADEMARKS.md](TRADEMARKS.md) - Trademark usage guidelines
- [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) - Third-party dependency licenses

## 🏢 Enterprise Support

**ModelMuxer Enterprise** offers additional features and support:

- 🎯 **Priority Support**: 24/7 support with SLA guarantees
- 🏗️ **Custom Deployment**: On-premises and private cloud deployment
- 🔧 **Custom Features**: Tailored routing strategies and integrations
- 📊 **Advanced Analytics**: Enhanced reporting and business intelligence
- 🔒 **Enhanced Security**: Additional compliance and security features

Please open a GitHub Issue with the "enterprise" label for more information.

## 🙏 Acknowledgments

- **FrugalGPT**: Inspiration for cascade routing strategies
- **OpenAI**: API compatibility and excellent models
- **Anthropic**: High-quality Claude models and safety research
- **FastAPI**: Excellent Python web framework
- **Kubernetes**: Container orchestration platform
- **Prometheus & Grafana**: Monitoring and observability stack

---

## Built with ❤️ by the ModelMuxer team

For questions, support, or feedback:

- 💬 Discord: [Join our community](https://discord.gg/modelmuxer)
- 🐛 Issues: [GitHub Issues](https://github.com/iamapsrajput/modelmuxer/issues)
- 📧 Support: Open a GitHub Issue with the "support" label
- 📖 Docs: [docs.modelmuxer.com](https://docs.modelmuxer.com)

## Policy Module

ModelMuxer includes a comprehensive policy enforcement system that provides compliance and safety features for LLM interactions.

### Features

- **PII Redaction**: Automatically detects and redacts sensitive information including:

  - Email addresses
  - Phone numbers
  - Credit card numbers
  - Social Security Numbers
  - IPv4/IPv6 addresses
  - IBAN codes
  - JWT-like tokens
  - Physical addresses
  - National IDs
  - Custom patterns via regex

- **Jailbreak Detection**: Identifies and blocks attempts to bypass safety measures using pattern matching

- **Per-tenant Model/Region Control**: Fine-grained access control for different tenants:
  - Allow/deny specific models per tenant
  - Allow/deny specific regions per tenant

### Configuration

Add these settings to your `.env` file:

```bash
# Enable/disable policy features
FEATURES__REDACT_PII=true
FEATURES__ENABLE_PII_NER=false

# Jailbreak detection
POLICY__ENABLE_JAILBREAK_DETECTION=true
POLICY__JAILBREAK_PATTERNS_PATH=app/policy/patterns/jailbreak.txt

# Per-tenant model control (JSON format)
POLICY__MODEL_ALLOW='{"tenant_a":["gpt-4o","claude-3-opus"],"tenant_b":["gpt-3.5-turbo"]}'
POLICY__MODEL_DENY='{"tenant_a":["gpt-4o-mini"],"tenant_b":["gpt-4o"]}'

# Per-tenant region control (JSON format)
POLICY__REGION_ALLOW='{"tenant_a":["us","eu"],"tenant_b":["us"]}'
POLICY__REGION_DENY='{"tenant_a":["cn","ru"],"tenant_b":["cn"]}'

# Custom PII patterns (JSON array of regex strings)
POLICY__EXTRA_PII_REGEX='["customsecret\\d+","internal_id_\\w+"]'
```

### Adding Jailbreak Patterns

Edit `app/policy/patterns/jailbreak.txt` to add new patterns (one per line):

```txt
ignore previous instructions
dan mode
simulate developer mode
system prompt reveal
bypass safety
ignore safety guidelines
```

### Safety Notes

- **No Raw PII Logging**: All PII is redacted before logging to prevent data leaks
- **Pattern Caching**: Jailbreak patterns are cached with a 15-second TTL for performance
- **Label Whitelisting**: PII types in metrics are whitelisted to prevent cardinality explosion
- **Error Handling**: Policy violations return structured 403 errors with detailed reasons

### Testing

Run policy tests:

```bash
# Unit tests for policy logic
pytest tests/policy/test_rules.py

# Integration tests with metrics
pytest tests/routing/test_router_policy_integration.py
```

### Observability

The policy module provides comprehensive observability:

- **Prometheus Metrics**:

  - `policy_redactions_total{pii_type}` - Count of PII redactions by type
  - `policy_violations_total{type}` - Count of policy violations by type

- **OpenTelemetry Spans**:
  - `policy.enforce` span with attributes:
    - `tenant_id` - The tenant being enforced
    - `blocked` - Whether the request was blocked
    - `reasons` - Comma-separated list of violation reasons
    - `num_redactions` - Total number of redactions performed
    - `pii_types_redacted` - Types of PII that were redacted

### API Response

When a request is blocked by policy, the API returns a 403 error:

```json
{
  "error": {
    "message": "Request blocked by policy",
    "type": "policy_violation",
    "reasons": ["jailbreak_detected", "model_denied"]
  }
}
```

The `reasons` array contains specific violation types that caused the block.

## 📊 Observability

ModelMuxer provides comprehensive observability through OpenTelemetry tracing, Prometheus metrics, and structured logging.

### Features

- **OpenTelemetry Tracing**: Distributed tracing across request → routing → provider calls
- **Prometheus Metrics**: HTTP, router, adapter, and policy metrics with proper labeling
- **Structured Logging**: JSON logs with trace/span IDs and no PII exposure
- **Grafana Dashboard**: Pre-built dashboard for monitoring and alerting

### Configuration

Add these settings to your `.env` file:

```bash
# Enable observability features
OBSERVABILITY__ENABLE_TRACING=true
OBSERVABILITY__ENABLE_METRICS=true
OBSERVABILITY__LOG_LEVEL=info

# OpenTelemetry configuration
OBSERVABILITY__SAMPLING_RATIO=1.0
OBSERVABILITY__OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Prometheus metrics endpoint
OBSERVABILITY__PROM_METRICS_PATH=/metrics/prometheus

# CORS origins for monitoring
OBSERVABILITY__CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

### Metrics Endpoints

| Endpoint              | Description        | Authentication |
| --------------------- | ------------------ | -------------- |
| `/metrics/prometheus` | Prometheus metrics | None (public)  |
| `/health`             | Health check       | None (public)  |

### Available Metrics

#### HTTP Metrics

- `http_requests_total{route,method,status}` - Request count by route/method/status
- `http_request_latency_ms{route,method}` - Request latency histogram

#### Router Metrics

- `llm_router_requests_total{route}` - Router decision count
- `llm_router_decision_latency_ms{route}` - Router decision latency
- `llm_router_fallbacks_total{route,reason}` - Fallback count by reason

#### Adapter Metrics

- `llm_requests_total{provider,model,outcome}` - Provider request count
- `llm_request_latency_ms{provider,model}` - Provider latency
- `llm_request_tokens_in_total{provider,model}` - Input token count
- `llm_request_tokens_out_total{provider,model}` - Output token count

#### Policy Metrics

- `policy_violations_total{type}` - Policy violation count
- `policy_redactions_total{pii_type}` - PII redaction count

### Tracing

OpenTelemetry spans are created for:

- **HTTP Requests**: `http.request` span with route/method attributes
- **Router Decisions**: `router.decide` span with task type and confidence scores
- **Provider Calls**: `provider.invoke` span (from adapters) with provider/model/tokens

Trace IDs are propagated in response headers as `x-trace-id`.

### Grafana Dashboard

Import the included dashboard:

1. **Download**: `grafana/dashboard_modelmuxer.json`
2. **Import**: In Grafana, go to Dashboards → Import
3. **Configure**: Set your Prometheus data source as `${DS_PROMETHEUS}`
4. **View**: Dashboard includes panels for:
   - HTTP latency p95/p99 by route
   - Router decision latency
   - Adapter latency by provider/model
   - Router fallbacks by reason
   - Policy violations and redactions
   - Token usage rates
   - Error rates

### Logging

Structured JSON logging includes:

- **Trace Context**: `trace_id` and `span_id` when available
- **No PII**: Raw prompts and PII are never logged
- **Configurable Level**: Set via `OBSERVABILITY__LOG_LEVEL`

Example log entry:

```json
{
  "level": "INFO",
  "logger": "app.main",
  "message": "Request processed successfully",
  "time": "2025-01-16T10:30:00+00:00",
  "trace_id": "1234567890abcdef1234567890abcdef",
  "span_id": "abcdef1234567890"
}
```

### Testing

Run observability tests:

```bash
# Test metrics endpoint
pytest tests/observability/test_metrics_endpoint.py

# Test tracing spans
pytest tests/observability/test_tracing_spans.py
```

### Production Setup

For production deployment:

1. **Prometheus**: Configure scraping from `/metrics/prometheus`
2. **Jaeger/Zipkin**: Set `OTEL_EXPORTER_OTLP_ENDPOINT` to your collector
3. **Grafana**: Import dashboard and configure alerts
4. **Log Aggregation**: Send JSON logs to your log aggregation system

## 🔧 CI/CD Pipeline

ModelMuxer uses a comprehensive CI/CD pipeline to ensure code quality, security, and reliability.

### Local Development

Run the same checks locally that are used in CI:

```bash
# Install development dependencies
poetry install --with dev

# Run linting and formatting
make lint

# Run type checking
make typecheck

# Run tests with coverage
make test-cov

# Run security scans
make security

# Run all checks
make lint && make typecheck && make test-cov && make security
```

### CI Workflows

#### Main CI Pipeline (`.github/workflows/ci.yml`)

- **Lint**: Ruff and Black formatting checks across Python 3.10-3.12
- **Type Check**: MyPy static type checking
- **Test**: Pytest with coverage (≥70% required) and Redis/PostgreSQL services
- **Security**: Bandit, Semgrep, and Trivy vulnerability scanning

#### Container Security (`.github/workflows/container-security.yml`)

- **Image Scanning**: Trivy vulnerability scanning of Docker images
- **SBOM Generation**: Software Bill of Materials for supply chain security
- **Image Signing**: Cosign-based container signing for releases

#### CodeQL Analysis (`.github/workflows/codeql.yml`)

- **Static Analysis**: GitHub's CodeQL for advanced security analysis
- **Scheduled Scans**: Weekly automated security scanning

### Quality Gates

All PRs must pass:

- ✅ **Linting**: No Ruff or Black violations
- ✅ **Type Checking**: No MyPy errors
- ✅ **Tests**: ≥70% code coverage
- ✅ **Security**: No critical/high severity vulnerabilities
- ✅ **Documentation**: Updated README and docstrings

### Pre-commit Hooks

Install pre-commit hooks for automatic code quality checks:

```bash
# Install pre-commit
poetry run pre-commit install

# Run all hooks
poetry run pre-commit run --all-files
```

### Security Notes

- **Metrics Endpoint**: Public read-only access (no authentication required)
- **No PII in Spans**: Only route/method/status and confidence scores are recorded
- **Sampling Control**: Adjust `OBSERVABILITY__SAMPLING_RATIO` for high-volume deployments
