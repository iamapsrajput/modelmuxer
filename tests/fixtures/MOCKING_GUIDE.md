# ModelMuxer Testing Mock Guide

This guide explains how to use the realistic mocking framework for integration testing in ModelMuxer.

## Overview

The realistic mocking framework provides high-fidelity mocks that simulate:
- Network latency
- Error rates
- Realistic token counting and costs
- Provider-specific behaviors
- Intelligent routing decisions
- Streaming responses

## Basic Usage

### 1. Simple Provider Mock

```python
from tests.fixtures.realistic_mocks import RealisticMockProvider

# Create a provider with 100ms latency and 1% error rate
provider = RealisticMockProvider("openai", latency_ms=100, error_rate=0.01)

# Use it like a real provider
response = await provider.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4o-mini",
    max_tokens=50
)
```

### 2. Provider Registry

```python
from tests.fixtures.realistic_mocks import create_realistic_provider_registry

# Create multiple providers at once
registry = create_realistic_provider_registry(
    providers=["openai", "anthropic", "groq"],
    latency_ms=100,
    error_rate=0.05
)

# Use in tests with patching
with patch("app.providers.registry.get_provider_registry", return_value=registry):
    # Your test code here
    pass
```

### 3. Intelligent Router Mock

```python
from tests.fixtures.realistic_mocks import RealisticMockRouter

router = RealisticMockRouter()

# Router makes decisions based on query complexity
provider, model, reason, _, metadata = await router.select_model(
    messages=[{"role": "user", "content": "Complex analysis..."}],
    constraints={"max_cost": 0.1}
)
```

## Integration Test Example

```python
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from tests.fixtures.realistic_mocks import (
    create_realistic_provider_registry,
    RealisticMockRouter
)

@pytest.fixture
def realistic_client():
    """Create test client with realistic mocks."""
    mock_providers = create_realistic_provider_registry()
    mock_router = RealisticMockRouter()
    
    with patch("app.providers.registry.get_provider_registry", return_value=mock_providers):
        with patch("app.router.HeuristicRouter", return_value=mock_router):
            from app.main import app
            with TestClient(app) as client:
                yield client

def test_api_with_realistic_behavior(realistic_client):
    """Test API with realistic provider behavior."""
    response = realistic_client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}]
        },
        headers={"Authorization": "Bearer test-key"}
    )
    
    assert response.status_code == 200
    # Response will have realistic latency, token counts, etc.
```

## Testing Different Scenarios

### 1. High Error Rate Testing

```python
# Test resilience with 20% error rate
provider = RealisticMockProvider("openai", error_rate=0.2)
```

### 2. Latency Testing

```python
# Test with slow provider (500ms latency)
provider = RealisticMockProvider("anthropic", latency_ms=500)
```

### 3. Cost-Aware Testing

```python
# Router will select cheaper models for budget constraints
result = await router.select_model(
    messages=[{"role": "user", "content": "Simple query"}],
    constraints={"max_cost": 0.0001}
)
# Will route to groq/llama for cost efficiency
```

### 4. Streaming Response Testing

```python
stream = create_realistic_mock_response(
    model="gpt-4",
    messages=messages,
    stream=True
)

async for chunk in stream:
    # Process streaming chunks
    pass
```

## Best Practices

1. **Match Production Behavior**: Configure latency and error rates to match real provider behavior
2. **Test Edge Cases**: Use high error rates to test retry logic
3. **Concurrent Testing**: Test multiple providers simultaneously to verify concurrency handling
4. **Cost Validation**: Verify cost calculations match production pricing
5. **Routing Logic**: Test that routing decisions make sense for different query types

## Advanced Features

### Custom Provider Behavior

```python
class CustomMockProvider(RealisticMockProvider):
    async def chat_completion(self, messages, model, **kwargs):
        # Add custom behavior
        if "test" in str(messages):
            raise Exception("Custom test error")
        return await super().chat_completion(messages, model, **kwargs)
```

### Routing Statistics

```python
# Get insights into routing decisions
stats = router.get_routing_stats()
print(f"Total routes: {stats['total_routes']}")
print(f"Provider distribution: {stats['provider_distribution']}")
```

## Migration from Simple Mocks

Replace simple mocks:
```python
# Old
mock_provider = Mock()
mock_provider.chat_completion = AsyncMock(return_value={...})

# New
mock_provider = RealisticMockProvider("openai")
```

The realistic mocks provide much better test coverage by simulating real-world conditions.