# Testing Guide

## Overview

ModelMuxer uses a comprehensive testing strategy to ensure reliability, performance, and correctness.

## Testing Framework

### Test Structure

- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint and database testing
- **End-to-End Tests**: Full system workflow testing
- **Performance Tests**: Load and stress testing

### Testing Tools

- **pytest**: Primary testing framework
- **pytest-asyncio**: Async test support
- **pytest-cov**: Coverage reporting
- **httpx**: HTTP client for API testing
- **factory-boy**: Test data generation

## Running Tests

### Basic Test Commands

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test file
poetry run pytest tests/test_router.py -v

# Run tests by marker
poetry run pytest -m unit
poetry run pytest -m integration
```

### Test Configuration

```bash
# Environment variables for testing
TEST_DATABASE_URL=sqlite:///test.db
TEST_OPENAI_API_KEY=test-key
TEST_ANTHROPIC_API_KEY=test-key
```

## CI/CD Testing Pipeline

### Automated Testing

The CI/CD pipeline includes comprehensive automated testing:

#### Test Matrix

- **Python versions**: 3.11, 3.12
- **Operating systems**: Ubuntu, macOS, Windows
- **Deployment modes**: basic, enhanced, production

#### Quality Gates

- **100% test success rate**: All 116+ tests must pass
- **Code coverage**: Minimum 80% coverage requirement
- **Performance benchmarks**: Automated performance regression detection
- **Security scanning**: No high-severity vulnerabilities allowed

#### Test Optimization

- **Parallel execution**: Tests run in parallel for faster feedback
- **Smart caching**: Dependencies cached between runs
- **Conditional testing**: Performance tests run only when relevant
- **Error handling**: Graceful fallbacks for optional dependencies

## Test Categories

### Unit Tests

- Router logic testing
- Provider client testing
- Cost calculation testing
- Utility function testing

### Integration Tests

- API endpoint testing
- Database integration testing
- Cache integration testing
- Provider API mocking

### Performance Tests

- Load testing with realistic traffic
- Stress testing for breaking points
- Memory usage profiling
- Response time benchmarking

## Test Examples

### Unit Test Example

```python
import pytest
from app.routing.hybrid_router import HybridRouter

class TestHybridRouter:
    def test_route_simple_query(self):
        router = HybridRouter()
        result = router.select_model(
            query="What is 2+2?",
            budget_constraint=0.01
        )
        assert result.provider == "openai"
        assert result.model == "gpt-4o-mini"
```

### Integration Test Example

```python
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_chat_completion_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "routing_strategy": "hybrid"
            },
            headers={"Authorization": "Bearer test-token"}
        )
    assert response.status_code == 200
    assert "choices" in response.json()
```

## Test Data Management

### Fixtures

```python
@pytest.fixture
def sample_chat_request():
    return {
        "messages": [{"role": "user", "content": "Test message"}],
        "routing_strategy": "hybrid",
        "max_tokens": 100
    }

@pytest.fixture
async def test_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
```

### Factory Classes

```python
import factory
from app.models import User, Request

class UserFactory(factory.Factory):
    class Meta:
        model = User

    id = factory.Sequence(lambda n: f"user_{n}")
    email = factory.LazyAttribute(lambda obj: f"{obj.id}@example.com")
```

## Mocking External Services

### Provider API Mocking

```python
import pytest
from unittest.mock import patch, AsyncMock

@pytest.fixture
def mock_openai_client():
    with patch('app.providers.openai_client.OpenAIClient') as mock:
        mock.return_value.chat_completion = AsyncMock(
            return_value={"choices": [{"message": {"content": "Test response"}}]}
        )
        yield mock
```

## Coverage Requirements

### Coverage Targets

- **Overall**: >85% code coverage
- **Critical paths**: >95% coverage
- **New features**: 100% coverage required

### Coverage Commands

```bash
# Generate HTML coverage report
poetry run pytest --cov=app --cov-report=html

# View coverage in terminal
poetry run pytest --cov=app --cov-report=term-missing

# Fail if coverage below threshold
poetry run pytest --cov=app --cov-fail-under=85
```

## Continuous Integration

### GitHub Actions Testing

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: |
          poetry install --with dev
          poetry run pytest --cov=app
```

### Pre-commit Testing

```bash
# Install pre-commit hooks
pre-commit install

# Run tests before each commit
# Configured in .pre-commit-config.yaml
```

## Performance Testing

### Load Testing Setup

```python
# tests/performance/test_load.py
import pytest
import asyncio
from httpx import AsyncClient

@pytest.mark.performance
async def test_concurrent_requests():
    async def make_request(client):
        return await client.post("/api/v1/chat/completions", json=test_data)

    async with AsyncClient(app=app) as client:
        tasks = [make_request(client) for _ in range(100)]
        responses = await asyncio.gather(*tasks)
        assert all(r.status_code == 200 for r in responses)
```

## Test Best Practices

### Writing Good Tests

1. **Test one thing at a time**
2. **Use descriptive test names**
3. **Keep tests independent**
4. **Use appropriate assertions**
5. **Mock external dependencies**

### Test Organization

```
tests/
├── unit/
│   ├── test_router.py
│   ├── test_providers.py
│   └── test_utils.py
├── integration/
│   ├── test_api.py
│   └── test_database.py
├── performance/
│   └── test_load.py
└── conftest.py
```

## Debugging Tests

### Common Issues

- **Async test failures**: Use `pytest-asyncio`
- **Database conflicts**: Use test database isolation
- **Flaky tests**: Implement proper cleanup and mocking

### Debugging Commands

```bash
# Run with detailed output
poetry run pytest -v -s

# Run specific test with debugging
poetry run pytest tests/test_router.py::test_function -v -s --pdb

# Show test durations
poetry run pytest --durations=10
```

## Summary

Created all missing documentation files to fix broken links in CI:

- ✅ **FAQ.md**: Frequently asked questions
- ✅ **docs/troubleshooting.md**: Common issues and solutions
- ✅ **docs/monitoring.md**: Observability and metrics
- ✅ **docs/security.md**: Security best practices
- ✅ **docs/performance.md**: Optimization strategies
- ✅ **docs/sdks.md**: SDK documentation (skipped detailed content as requested)
- ✅ **docs/architecture.md**: System architecture overview
- ✅ **docs/testing.md**: Testing framework and practices

Fixed ruff formatting issues. All linting and link checks should now pass in CI/CD.
