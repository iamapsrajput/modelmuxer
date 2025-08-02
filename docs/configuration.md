# Configuration Guide

ModelMuxer provides extensive configuration options through environment variables, configuration files, and runtime settings.

## Environment Variables

### Core Configuration

```env
# Server Settings
HOST=0.0.0.0                    # Server host
PORT=8000                       # Server port
DEBUG=false                     # Debug mode
WORKERS=4                       # Number of worker processes

# Application Settings
APP_NAME=ModelMuxer
APP_VERSION=1.0.0
ENVIRONMENT=production          # development, staging, production
```

### Provider API Keys

```env
# Required: At least one provider API key
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MISTRAL_API_KEY=...
GOOGLE_API_KEY=...
GROQ_API_KEY=gsk_...
TOGETHER_API_KEY=...
COHERE_API_KEY=...
```

### Routing Configuration

```env
# Routing Strategy
DEFAULT_ROUTING_STRATEGY=hybrid  # hybrid, cascade, semantic, heuristic

# Strategy Toggles
CASCADE_ROUTING_ENABLED=true
SEMANTIC_ROUTING_ENABLED=true
HEURISTIC_ROUTING_ENABLED=true

# Cascade Routing
CASCADE_QUALITY_THRESHOLD=0.7
CASCADE_CONFIDENCE_THRESHOLD=0.7
CASCADE_MAX_BUDGET=0.1

# Heuristic Routing
CODE_DETECTION_THRESHOLD=0.2
COMPLEXITY_THRESHOLD=0.2
SIMPLE_QUERY_THRESHOLD=0.3
SIMPLE_QUERY_MAX_LENGTH=100

# Semantic Routing
CLASSIFICATION_ENABLED=true
CLASSIFICATION_MODEL=all-MiniLM-L6-v2
SIMILARITY_THRESHOLD=0.7
CACHE_EMBEDDINGS=true
```

### Authentication & Security

```env
# JWT Configuration
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=RS256
JWT_EXPIRATION_HOURS=24

# API Key Authentication
AUTH_ENABLED=true
REQUIRE_EMAIL_VERIFICATION=true

# Security Headers
CORS_ORIGINS=["http://localhost:3000", "https://yourdomain.com"]
CORS_METHODS=["GET", "POST", "PUT", "DELETE"]
CORS_HEADERS=["*"]
```

### Database Configuration

```env
# PostgreSQL (Primary Database)
DATABASE_URL=postgresql://user:password@localhost:5432/modelmuxer
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_POOL_TIMEOUT=30

# SQLite (Development)
DATABASE_URL=sqlite:///./modelmuxer.db
```

### Caching Configuration

```env
# Cache Settings
CACHE_ENABLED=true
CACHE_BACKEND=redis             # memory, redis
CACHE_DEFAULT_TTL=3600
CACHE_MAX_SIZE=10000

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=
REDIS_SSL=false
REDIS_POOL_SIZE=10
```

### Rate Limiting

```env
# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_REQUESTS_PER_HOUR=1000
RATE_LIMIT_REQUESTS_PER_DAY=10000
RATE_LIMIT_BURST_SIZE=10
```

### Monitoring & Observability

```env
# Monitoring
MONITORING_ENABLED=true
METRICS_ENABLED=true
TRACING_ENABLED=true
HEALTH_CHECK_ENABLED=true

# Prometheus
PROMETHEUS_PORT=9090
PROMETHEUS_PATH=/metrics

# Logging
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json                 # json, text
ENABLE_CORRELATION_ID=true
ENABLE_REQUEST_LOGGING=true
LOG_FILE_PATH=/var/log/modelmuxer.log
```

## Configuration Files

### config.yaml

Create a `config.yaml` file for more complex configurations:

```yaml
# config.yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  debug: false

providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    timeout: 30
    max_retries: 3
    models:
      - gpt-4o
      - gpt-4o-mini
      - gpt-3.5-turbo
  
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    timeout: 30
    max_retries: 3
    models:
      - claude-3-5-sonnet-20241022
      - claude-3-haiku-20240307

routing:
  default_strategy: "hybrid"
  strategies:
    cascade:
      enabled: true
      quality_threshold: 0.7
      confidence_threshold: 0.7
      max_budget: 0.1
    
    semantic:
      enabled: true
      model: "all-MiniLM-L6-v2"
      similarity_threshold: 0.7
      cache_embeddings: true
    
    heuristic:
      enabled: true
      code_detection_threshold: 0.2
      complexity_threshold: 0.2
      simple_query_threshold: 0.3

database:
  url: "${DATABASE_URL}"
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30

cache:
  enabled: true
  backend: "redis"
  default_ttl: 3600
  redis:
    url: "${REDIS_URL}"
    pool_size: 10

auth:
  enabled: true
  jwt:
    secret_key: "${JWT_SECRET_KEY}"
    algorithm: "RS256"
    expiration_hours: 24

monitoring:
  enabled: true
  metrics:
    enabled: true
    prometheus_port: 9090
  tracing:
    enabled: true
  logging:
    level: "INFO"
    format: "json"
    correlation_id: true
```

### Model Configurations

Define custom model configurations in `models.yaml`:

```yaml
# models.yaml
models:
  openai:
    gpt-4o:
      max_tokens: 4096
      temperature: 0.7
      pricing:
        input: 0.005
        output: 0.015
    
    gpt-4o-mini:
      max_tokens: 16384
      temperature: 0.7
      pricing:
        input: 0.00015
        output: 0.0006

  anthropic:
    claude-3-5-sonnet-20241022:
      max_tokens: 4096
      temperature: 0.7
      pricing:
        input: 0.003
        output: 0.015

routing_preferences:
  code:
    - provider: "openai"
      model: "gpt-4o"
    - provider: "anthropic"
      model: "claude-3-5-sonnet-20241022"
  
  simple:
    - provider: "openai"
      model: "gpt-4o-mini"
    - provider: "anthropic"
      model: "claude-3-haiku-20240307"
  
  complex:
    - provider: "openai"
      model: "gpt-4o"
    - provider: "anthropic"
      model: "claude-3-5-sonnet-20241022"
```

## Runtime Configuration

### API Configuration Endpoints

Update configuration at runtime using the API:

```bash
# Update routing strategy
curl -X PUT http://localhost:8000/api/v1/config/routing \
  -H "Content-Type: application/json" \
  -d '{"default_strategy": "cascade"}'

# Update rate limits
curl -X PUT http://localhost:8000/api/v1/config/rate-limits \
  -H "Content-Type: application/json" \
  -d '{"requests_per_minute": 120}'

# Update model preferences
curl -X PUT http://localhost:8000/api/v1/config/models \
  -H "Content-Type: application/json" \
  -d '{"preferences": {"code": [{"provider": "openai", "model": "gpt-4o"}]}}'
```

### Configuration Validation

ModelMuxer validates all configuration on startup:

```bash
# Validate configuration
poetry run python -m app.config.validate

# Check configuration status
curl http://localhost:8000/api/v1/config/status
```

## Environment-Specific Configurations

### Development

```env
DEBUG=true
LOG_LEVEL=DEBUG
CACHE_BACKEND=memory
DATABASE_URL=sqlite:///./dev.db
RATE_LIMIT_ENABLED=false
```

### Staging

```env
DEBUG=false
LOG_LEVEL=INFO
ENVIRONMENT=staging
DATABASE_URL=postgresql://user:pass@staging-db:5432/modelmuxer
REDIS_URL=redis://staging-redis:6379/0
```

### Production

```env
DEBUG=false
LOG_LEVEL=WARNING
ENVIRONMENT=production
WORKERS=8
DATABASE_URL=postgresql://user:pass@prod-db:5432/modelmuxer
REDIS_URL=redis://prod-redis:6379/0
MONITORING_ENABLED=true
RATE_LIMIT_ENABLED=true
```

## Configuration Best Practices

1. **Use Environment Variables**: Store sensitive data in environment variables
2. **Validate on Startup**: Always validate configuration before starting
3. **Use Defaults**: Provide sensible defaults for all configuration options
4. **Document Changes**: Document any configuration changes in your deployment
5. **Test Configurations**: Test configuration changes in staging before production
6. **Monitor Configuration**: Monitor configuration changes and their impact

## Next Steps

- [API Reference](api-reference.md)
- [Deployment Guide](deployment.md)
- [Monitoring Guide](monitoring.md)
