# LiteLLM Provider Integration

## Overview

The LiteLLM provider enables ModelMuxer to integrate with [LiteLLM](https://docs.litellm.ai/), a unified proxy that provides access to 100+ LLM models through a single, consistent API. This integration allows you to leverage LiteLLM's powerful features while benefiting from ModelMuxer's intelligent routing and cost optimization.

## Key Benefits

### 🌐 Unified Access

- Access 100+ models from 20+ providers through one endpoint
- Consistent API interface regardless of underlying provider
- Simplified model management and configuration

### 💰 Cost Optimization

- Combine LiteLLM's cost tracking with ModelMuxer's budget management
- Real-time cost calculation and optimization
- Support for custom pricing configurations

### 🔄 Advanced Features

- Load balancing across multiple models
- Automatic fallbacks and retries
- Custom model configurations and aliases

### 🛡️ Production Ready

- Built-in rate limiting and error handling
- Comprehensive logging and monitoring
- Health checks and availability monitoring

## Setup and Configuration

### Prerequisites

1. **Install LiteLLM**:

   ```bash
   pip install litellm[proxy]
   ```

2. **Create LiteLLM configuration** (`config.yaml`):

   ```yaml
   model_list:
     # OpenAI Models
     - model_name: gpt-4
       litellm_params:
         model: openai/gpt-4
         api_key: os.environ/OPENAI_API_KEY

     - model_name: gpt-3.5-turbo
       litellm_params:
         model: openai/gpt-3.5-turbo
         api_key: os.environ/OPENAI_API_KEY

     # Anthropic Models
     - model_name: claude-3-sonnet
       litellm_params:
         model: anthropic/claude-3-sonnet-20240229
         api_key: os.environ/ANTHROPIC_API_KEY

     - model_name: claude-3-haiku
       litellm_params:
         model: anthropic/claude-3-haiku-20240307
         api_key: os.environ/ANTHROPIC_API_KEY

     # Google Models
     - model_name: gemini-pro
       litellm_params:
         model: google/gemini-pro
         api_key: os.environ/GOOGLE_API_KEY

     # Groq Models
     - model_name: llama2-70b-chat
       litellm_params:
         model: groq/llama2-70b-4096
         api_key: os.environ/GROQ_API_KEY

   # Optional: General settings
   general_settings:
     master_key: your-master-key-here
     database_url: postgresql://user:pass@localhost/litellm
   ```

3. **Start LiteLLM proxy**:
   ```bash
   litellm --config config.yaml --port 4000
   ```

### ModelMuxer Configuration

1. **Environment Variables** (`.env`):

   ```bash
   # LiteLLM Configuration
   LITELLM_BASE_URL=http://localhost:4000
   LITELLM_API_KEY=your-master-key-here  # Optional, if using master_key
   ```

2. **Custom Model Configuration** (optional):

   ```python
   # In your application startup
   from app.providers import LiteLLMProvider

   # Initialize with custom models
   custom_models = {
       "gpt-4": {
           "pricing": {"input": 0.03, "output": 0.06},
           "rate_limits": {"requests_per_minute": 20, "tokens_per_minute": 20000},
           "metadata": {"context_window": 8192, "provider": "openai"}
       },
       "claude-3-haiku": {
           "pricing": {"input": 0.00025, "output": 0.00125},
           "rate_limits": {"requests_per_minute": 100, "tokens_per_minute": 100000},
           "metadata": {"context_window": 200000, "provider": "anthropic"}
       }
   }

   provider = LiteLLMProvider(
       base_url="http://localhost:4000",
       api_key="your-master-key",
       custom_models=custom_models
   )
   ```

## Usage Examples

### Basic Chat Completion

```python
import asyncio
import httpx
from app.providers.litellm_provider import LiteLLMProvider
from app.models import ChatMessage

async def basic_example():
    provider = LiteLLMProvider(
        base_url="http://localhost:4000",
        api_key="your-api-key"
    )

    messages = [
        ChatMessage(role="user", content="What is the capital of France?")
    ]

    response = await provider.chat_completion(
        messages=messages,
        model="gpt-3.5-turbo",
        max_tokens=100,
        temperature=0.7
    )

    print(f"Response: {response.choices[0].message.content}")
    print(f"Cost: ${provider.calculate_cost(response.usage.prompt_tokens, response.usage.completion_tokens, 'gpt-3.5-turbo'):.6f}")

asyncio.run(basic_example())
```

### Streaming Chat Completion

```python
async def streaming_example():
    provider = LiteLLMProvider(base_url="http://localhost:4000")

    messages = [
        ChatMessage(role="user", content="Write a short story about AI")
    ]

    print("Streaming response:")
    async for chunk in provider.stream_chat_completion(
        messages=messages,
        model="claude-3-haiku",
        max_tokens=500
    ):
        if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
            print(chunk["choices"][0]["delta"]["content"], end="", flush=True)

    print("\n\nStreaming complete!")

asyncio.run(streaming_example())
```

### Cost Optimization

```python
async def cost_optimization_example():
    provider = LiteLLMProvider(base_url="http://localhost:4000")

    # Get available models and their costs
    models = provider.get_supported_models()

    prompt_tokens = 1000
    completion_tokens = 500

    print("Cost comparison for 1000 input + 500 output tokens:")
    for model in models:
        cost = provider.calculate_cost(prompt_tokens, completion_tokens, model)
        print(f"{model}: ${cost:.6f}")

    # Find the most cost-effective model
    costs = {model: provider.calculate_cost(prompt_tokens, completion_tokens, model)
             for model in models}
    cheapest_model = min(costs, key=costs.get)

    print(f"\nMost cost-effective model: {cheapest_model} (${costs[cheapest_model]:.6f})")

asyncio.run(cost_optimization_example())
```

## Advanced Configuration

### Custom Model Management

```python
# Add custom models dynamically
provider.add_custom_model(
    model_name="custom-llama-7b",
    pricing={"input": 0.0001, "output": 0.0002},
    rate_limits={"requests_per_minute": 200, "tokens_per_minute": 200000},
    metadata={
        "provider": "ollama",
        "context_window": 4096,
        "task_types": ["general", "code"]
    }
)

# Get model information
model_info = provider.get_model_info("custom-llama-7b")
print(f"Model info: {model_info}")
```

### Health Monitoring

```python
async def health_check_example():
    provider = LiteLLMProvider(base_url="http://localhost:4000")

    # Check if LiteLLM proxy is healthy
    is_healthy = await provider.health_check()
    print(f"LiteLLM proxy health: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")

    # Get available models from proxy
    try:
        available_models = await provider.get_available_models()
        print(f"Available models: {[model['id'] for model in available_models]}")
    except Exception as e:
        print(f"Failed to get models: {e}")

asyncio.run(health_check_example())
```

## Integration with ModelMuxer Routing

The LiteLLM provider seamlessly integrates with ModelMuxer's routing strategies:

### Heuristic Routing

```python
# The router will automatically consider LiteLLM models
# when making routing decisions based on:
# - Cost optimization
# - Model capabilities
# - Rate limits
# - Provider availability

from app.router import HeuristicRouter

router = HeuristicRouter()
providers = {"litellm": provider}

# Router will select optimal model from LiteLLM proxy
selected_provider, selected_model, reason = router.select_model(
    messages=[ChatMessage(role="user", content="Complex coding task")],
    providers=providers
)

print(f"Selected: {selected_provider}/{selected_model}")
print(f"Reason: {reason}")
```

### Cost-Aware Routing

```python
# ModelMuxer will consider LiteLLM pricing when routing
# for cost optimization scenarios

messages = [ChatMessage(role="user", content="Simple question")]

# Router considers cost when selecting from LiteLLM models
response = await router.route_request(
    messages=messages,
    providers=providers,
    strategy="cost_optimized",
    max_cost=0.01  # Maximum cost constraint
)
```

## Troubleshooting

### Common Issues

1. **Connection Refused**

   ```
   Error: Connection refused to http://localhost:4000
   ```

   - Ensure LiteLLM proxy is running: `litellm --config config.yaml`
   - Check the port and URL configuration
   - Verify firewall settings

2. **Authentication Errors**

   ```
   Error: 401 Unauthorized
   ```

   - Check if `LITELLM_API_KEY` matches the master key in LiteLLM config
   - Verify API key format and validity
   - Ensure the key has proper permissions

3. **Model Not Found**

   ```
   Error: Model 'xyz' not found
   ```

   - Check LiteLLM config.yaml for model definitions
   - Verify model names match exactly
   - Restart LiteLLM proxy after config changes

4. **Rate Limiting**
   ```
   Error: 429 Too Many Requests
   ```
   - Check LiteLLM proxy rate limits
   - Verify underlying provider rate limits
   - Consider implementing backoff strategies

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed HTTP requests/responses
provider = LiteLLMProvider(
    base_url="http://localhost:4000",
    api_key="your-key"
)
```

### Health Checks

Regular health monitoring:

```python
async def monitor_litellm():
    provider = LiteLLMProvider(base_url="http://localhost:4000")

    while True:
        try:
            is_healthy = await provider.health_check()
            if not is_healthy:
                print("⚠️ LiteLLM proxy is unhealthy")
                # Implement alerting logic here

            await asyncio.sleep(30)  # Check every 30 seconds
        except Exception as e:
            print(f"Health check failed: {e}")
            await asyncio.sleep(60)  # Wait longer on errors
```

## LiteLLM Docker Setup

For a complete Docker-based setup with LiteLLM proxy:

### Quick Setup

1. **Configure API Keys**
   - Update `config/litellm-config.yaml` with your provider API keys
   - Update `.env` with LiteLLM configuration

2. **Start Services**
   ```bash
   # Basic setup
   docker-compose -f infra/docker-compose.litellm.yaml up -d

   # With monitoring (Prometheus + Grafana)
   docker-compose -f infra/docker-compose.litellm.yaml --profile monitoring up -d
   ```

3. **Access Services**
   - ModelMuxer API: http://localhost:8000
   - LiteLLM Proxy: http://localhost:4000
   - Grafana (with monitoring): http://localhost:3000 (admin/admin123)

### Configuration Files

**`config/litellm-config.yaml`**:
```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: your-openai-key-here
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: your-anthropic-key-here
```

**Environment Variables**:
```env
LITELLM_BASE_URL=http://localhost:4000
LITELLM_API_KEY=your-litellm-key
DEFAULT_ROUTING_STRATEGY=hybrid
CASCADE_ROUTING_ENABLED=true
```

## Best Practices

1. **Configuration Management**

   - Use environment variables for sensitive data
   - Version control your LiteLLM config files
   - Implement configuration validation

2. **Error Handling**

   - Implement proper retry logic
   - Handle provider-specific errors gracefully
   - Monitor error rates and patterns

3. **Performance Optimization**

   - Use connection pooling for high-throughput scenarios
   - Implement caching for model metadata
   - Monitor response times and optimize accordingly

4. **Security**

   - Secure your LiteLLM proxy endpoint
   - Use HTTPS in production
   - Implement proper API key rotation

5. **Monitoring**
   - Track usage patterns and costs
   - Monitor model performance and availability
   - Set up alerting for critical issues

For more information, see the [LiteLLM documentation](https://docs.litellm.ai/) and [ModelMuxer architecture guide](../architecture.md).
