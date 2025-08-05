# Performance Guide

## Overview

This guide covers performance optimization strategies, benchmarking, and monitoring for ModelMuxer.

## Performance Characteristics

### Response Time Targets
- Simple queries: < 500ms (p95)
- Complex queries: < 2s (p95)
- Streaming responses: < 200ms to first token

### Throughput Capacity
- Standard deployment: 1000 requests/minute
- Optimized deployment: 5000+ requests/minute
- Auto-scaling: Up to 50,000 requests/minute

## Optimization Strategies

### Caching

#### Response Caching
```python
# Enable response caching
ENABLE_CACHE = True
CACHE_BACKEND = "redis"
CACHE_TTL = 3600  # 1 hour

# Cache configuration
REDIS_URL = "redis://localhost:6379/0"
CACHE_KEY_PREFIX = "modelmuxer:"
```

#### Model Selection Caching
```python
# Cache routing decisions
ENABLE_ROUTING_CACHE = True
ROUTING_CACHE_TTL = 300  # 5 minutes
```

### Connection Pooling

#### HTTP Client Optimization
```python
# Provider connection pooling
HTTP_POOL_CONNECTIONS = 20
HTTP_POOL_MAXSIZE = 100
HTTP_POOL_BLOCK = False
```

#### Database Connection Pooling
```python
# SQLAlchemy pool settings
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 30
DATABASE_POOL_RECYCLE = 3600
```

### Async Processing

#### Concurrent Request Handling
```python
# FastAPI async configuration
MAX_WORKERS = 4
WORKER_CONNECTIONS = 1000
```

#### Background Task Processing
```python
# Celery for background tasks
CELERY_BROKER_URL = "redis://localhost:6379/1"
CELERY_RESULT_BACKEND = "redis://localhost:6379/2"
```

## Routing Optimization

### Smart Model Selection
- Use fastest models for simple queries
- Cache routing decisions
- Implement fallback strategies
- Monitor provider latency

### Load Balancing
```python
# Provider load balancing
ENABLE_LOAD_BALANCING = True
LOAD_BALANCE_STRATEGY = "round_robin"  # or "least_connections"
```

### Request Batching
```python
# Batch similar requests
ENABLE_REQUEST_BATCHING = True
BATCH_SIZE = 10
BATCH_TIMEOUT = 100  # milliseconds
```

## Database Performance

### Query Optimization
- Use appropriate indexes
- Optimize frequent queries
- Implement query result caching
- Use read replicas for analytics

### Schema Design
```sql
-- Optimized indexes for common queries
CREATE INDEX idx_requests_user_timestamp ON requests(user_id, created_at);
CREATE INDEX idx_costs_provider_model ON costs(provider, model);
```

### Connection Management
```python
# Database optimization settings
SQLALCHEMY_ENGINE_OPTIONS = {
    "pool_pre_ping": True,
    "pool_recycle": 300,
    "echo": False  # Disable in production
}
```

## Memory Management

### Memory Usage Optimization
- Configure appropriate worker memory limits
- Implement request size limits
- Use streaming for large responses
- Monitor memory leaks

### Garbage Collection
```python
# Python GC optimization
import gc
gc.set_threshold(700, 10, 10)
```

## Network Optimization

### Compression
```python
# Enable response compression
ENABLE_GZIP = True
GZIP_MINIMUM_SIZE = 1000
```

### CDN Integration
- Use CDN for static assets
- Cache API responses at edge locations
- Implement geographic routing

## Monitoring and Profiling

### Performance Metrics
```python
# Key metrics to monitor
- Request rate (req/sec)
- Response time percentiles (p50, p95, p99)
- Error rate percentage
- Memory usage
- CPU utilization
- Database connection pool usage
```

### Application Performance Monitoring (APM)
```python
# APM integration examples
# New Relic
import newrelic.agent
newrelic.agent.initialize('newrelic.ini')

# DataDog
from ddtrace import patch_all
patch_all()
```

### Profiling Tools
```bash
# Memory profiling
poetry run python -m memory_profiler app/main.py

# CPU profiling
poetry run python -m cProfile -o profile.prof app/main.py

# Line profiler
poetry run kernprof -l -v app/main.py
```

## Load Testing

### Performance Testing Setup
```bash
# Install testing tools
pip install locust

# Run load tests
locust -f tests/performance/locustfile.py --host=http://localhost:8000
```

### Test Scenarios
```python
# Example load test
from locust import HttpUser, task, between

class ModelMuxerUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def chat_completion(self):
        self.client.post("/api/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello!"}],
            "routing_strategy": "hybrid"
        })
```

### Benchmarking Results
- Baseline performance metrics
- Performance regression detection
- Capacity planning data

## Scaling Strategies

### Horizontal Scaling
```yaml
# Kubernetes scaling configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: modelmuxer-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: modelmuxer
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Vertical Scaling
- Memory scaling based on usage patterns
- CPU scaling for compute-intensive operations
- Storage scaling for data growth

### Auto-scaling Configuration
```python
# Auto-scaling triggers
CPU_SCALE_UP_THRESHOLD = 70
MEMORY_SCALE_UP_THRESHOLD = 80
RESPONSE_TIME_THRESHOLD = 2000  # milliseconds
```

## Provider-Specific Optimizations

### OpenAI
- Use connection pooling for API calls
- Implement exponential backoff for rate limits
- Cache model availability checks

### Anthropic
- Optimize batch request handling
- Use streaming for long responses
- Monitor Claude-specific rate limits

### Mistral
- Leverage regional endpoints
- Optimize for European latency
- Use appropriate model variants

## Performance Best Practices

### Code Optimization
1. **Use async/await** for I/O operations
2. **Implement caching** at multiple levels
3. **Optimize database queries** with proper indexing
4. **Use connection pooling** for external services
5. **Monitor memory usage** and prevent leaks

### Configuration Tuning
1. **Adjust worker processes** based on CPU cores
2. **Configure appropriate timeouts** for providers
3. **Set proper cache TTL values** based on use case
4. **Use compression** for large responses
5. **Enable request/response logging** only when needed

### Infrastructure Optimization
1. **Use SSD storage** for databases
2. **Configure load balancers** properly
3. **Implement health checks** for auto-scaling
4. **Use monitoring** for performance insights
5. **Regular performance testing** in CI/CD

## Troubleshooting Performance Issues

### Common Performance Problems
1. **High latency**: Check provider response times, network issues
2. **Memory leaks**: Monitor memory usage trends
3. **Database bottlenecks**: Analyze slow queries
4. **Cache misses**: Review cache hit rates and TTL settings

### Performance Debugging
```bash
# Monitor system resources
htop
iotop
nethogs

# Application-specific monitoring
poetry run python -m py-spy top --pid <process_id>
```

## Performance Checklist

### Pre-deployment
- [ ] Load testing completed
- [ ] Performance benchmarks established
- [ ] Caching strategy implemented
- [ ] Database queries optimized
- [ ] Connection pooling configured

### Production
- [ ] Monitoring and alerting active
- [ ] Auto-scaling configured
- [ ] Performance metrics tracked
- [ ] Regular performance reviews
- [ ] Capacity planning updated

## Next Steps

1. **Establish baseline metrics** for your deployment
2. **Implement monitoring** and alerting
3. **Run regular load tests** to identify bottlenecks
4. **Optimize based on usage patterns** and metrics
5. **Plan for growth** with appropriate scaling strategies
