# ModelMuxer - LLM Router API

An intelligent LLM routing service that optimizes cost and quality by routing requests to the most appropriate language model provider based on prompt characteristics.

## Quick Start

1. Clone the repository
2. Copy `.env.example` to `.env` and add your API keys
3. Install dependencies: `pip install -r requirements.txt`
4. Run the server: `uvicorn app.main:app --reload`
5. Test with the provided test script: `python test_requests.py`

## Features

- **Intelligent Routing**: Automatically selects the best LLM based on prompt analysis
- **Cost Optimization**: Routes simple queries to cheaper models, complex ones to premium models
- **OpenAI Compatible**: Drop-in replacement for OpenAI API endpoints
- **Cost Tracking**: Built-in usage and cost monitoring
- **Multiple Providers**: Supports OpenAI, Anthropic, and Mistral

## API Endpoints

- `POST /v1/chat/completions` - Main chat completion endpoint
- `GET /health` - Health check
- `GET /metrics` - Usage and cost metrics

## Routing Logic

The system uses heuristic analysis to route requests:

- **Code-related prompts** → GPT-4o or Claude-Sonnet (high-quality reasoning)
- **Simple queries** → Mistral-small (cost-effective)
- **Complex analysis** → GPT-4o (premium reasoning)
- **Default** → GPT-3.5-turbo (balanced cost/quality)

## Setup Instructions

### Prerequisites

- Python 3.11 or higher
- API keys for at least one LLM provider (OpenAI, Anthropic, or Mistral)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ModelMuxer
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the server**

   ```bash
   uvicorn app.main:app --reload
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

## API Usage

### Basic Chat Completion

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
                    {"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}
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
4. Install dependencies: `pip install -r requirements.txt`
5. Run tests: `python test_router.py && python test_requests.py`

### Adding New Providers

1. Create new provider class in `app/providers/`
2. Inherit from `LLMProvider` base class
3. Implement required methods: `chat_completion`, `stream_chat_completion`, `calculate_cost`
4. Add provider to `app/providers/__init__.py`
5. Update configuration and pricing in `app/config.py`
6. Add tests for the new provider

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:

- Create an issue on GitHub
- Check the troubleshooting section above
- Review the test scripts for usage examples
