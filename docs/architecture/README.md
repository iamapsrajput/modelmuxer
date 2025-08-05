# ModelMuxer Architecture

## Overview

ModelMuxer is an enterprise-grade LLM routing platform designed for production scale with intelligent request routing, cost optimization, and comprehensive monitoring.

## System Architecture

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │    │   Load Balancer │    │   API Gateway   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   ModelMuxer    │
                    │   FastAPI App   │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Routing       │    │   Providers     │    │   Monitoring    │
│   Engine        │    │   (OpenAI,      │    │   & Metrics     │
│                 │    │   Anthropic,    │    │                 │
│                 │    │   Mistral, etc) │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Data Layer    │
                    │   (SQLite/      │
                    │   PostgreSQL +  │
                    │   Redis Cache)  │
                    └─────────────────┘
```

## Core Components

### 1. Application Layers

#### **Unified Application (`app/main.py`) - CONSOLIDATED**

- **Unified application** with both basic and advanced features
- Automatic mode detection based on environment and available dependencies
- Advanced routing with ML-based classification (when enabled)
- Comprehensive monitoring and metrics (when enabled)
- Enterprise features and security (when enabled)
- Enhanced cost tracking with budget management (when enabled)
- **Used in**: All deployments - automatically adapts to environment
- **Suitable for**: Development, testing, and production deployments

- Minimal feature set for development and testing
- Basic routing and provider integration
- Simple cost tracking
- Maintained for backward compatibility and testing
- **Used in**: Development testing, basic Docker builds
- **Suitable for**: Development, testing, lightweight deployments

### 2. Routing Engine (`app/routing/`)

#### **Base Router (`base_router.py`)**

- Abstract base class for all routing strategies
- Defines common interface and utilities

#### **Heuristic Router (`heuristic_router.py`)**

- Rule-based routing using prompt analysis
- Fast, deterministic routing decisions
- No ML dependencies required

#### **Semantic Router (`semantic_router.py`)**

- ML-powered routing using sentence transformers
- Context-aware model selection
- Requires ML dependencies

#### **Cascade Router (`cascade_router.py`)**

- Multi-tier routing with fallback strategies
- Cost optimization through provider cascading
- Intelligent retry logic

#### **Hybrid Router (`hybrid_router.py`)**

- Combines multiple routing strategies
- Adaptive routing based on request characteristics
- Production-recommended approach

### 3. Provider Integration (`app/providers/`)

#### **Supported Providers**

- **OpenAI**: GPT-3.5, GPT-4, GPT-4o models
- **Anthropic**: Claude 3 family (Haiku, Sonnet, Opus)
- **Mistral**: Mistral 7B, Mixtral models
- **Google**: Gemini models
- **Groq**: High-speed inference
- **Together AI**: Open source models
- **Cohere**: Command and Embed models

#### **Provider Features**

- Unified API interface
- Automatic retry and error handling
- Rate limiting and quota management
- Cost tracking and optimization

### 4. Cost Management

#### **Unified Cost Tracker (`app/cost_tracker.py`) - CONSOLIDATED**

- **Production-ready** advanced budget management
- Multi-period budget tracking (daily/weekly/monthly/yearly)
- Redis-backed real-time monitoring
- Cost optimization recommendations
- Budget alerts and notifications
- **Used by**: main.py in enhanced mode for production deployments

#### **Basic Cost Tracker (`app/cost_tracker.py`) - COMPATIBILITY**

- Token counting and cost calculation
- Simple pricing models
- Real-time cost estimation
- **Used by**: main.py and test files for basic functionality

### 5. Security & Authentication (`app/security/`)

#### **Authentication**

- API key-based authentication
- JWT token support
- Role-based access control (RBAC)

#### **Security Features**

- PII detection and protection
- Input sanitization and validation
- Rate limiting and DDoS protection
- Security headers and CORS

#### **Compliance**

- SOC 2 Type II compliance ready
- GDPR compliance features
- Audit logging and monitoring

### 6. Caching Layer (`app/cache/`)

#### **Memory Cache (`memory_cache.py`)**

- In-memory caching for development
- Fast access, limited scalability

#### **Redis Cache (`redis_cache.py`)**

- Distributed caching for production
- Scalable, persistent caching
- Session and response caching

### 7. Monitoring & Observability (`app/monitoring/`)

#### **Metrics Collection**

- Prometheus metrics export
- Custom business metrics
- Performance monitoring

#### **Logging**

- Structured logging with structlog
- Request/response logging
- Error tracking and alerting

#### **Health Checks**

- Application health endpoints
- Provider availability monitoring
- Database connectivity checks

## Data Flow

### Request Processing Flow

1. **Request Ingestion**

   - Client sends request to FastAPI application
   - Authentication and authorization validation
   - Input sanitization and validation

2. **Routing Decision**

   - Routing engine analyzes request
   - Selects optimal provider and model
   - Considers cost constraints and availability

3. **Provider Execution**

   - Request forwarded to selected provider
   - Response processing and formatting
   - Error handling and retry logic

4. **Response Processing**

   - Cost calculation and tracking
   - Metrics collection and logging
   - Response caching (if enabled)

5. **Response Delivery**
   - Formatted response sent to client
   - Monitoring and analytics updates

### Data Storage

#### **SQLite Database (Development)**

- Request/response logs
- Cost tracking data
- User statistics
- Configuration settings

#### **PostgreSQL Database (Production)**

- Scalable data storage
- Advanced querying capabilities
- Backup and recovery support

#### **Redis Cache**

- Session management
- Response caching
- Real-time metrics
- Rate limiting counters

## Deployment Architectures

### Development Deployment

```text
┌─────────────────┐
│   Developer     │
│   Machine       │
│                 │
│ ┌─────────────┐ │
│ │ ModelMuxer  │ │
│ │ (Basic)     │ │
│ │ + SQLite    │ │
│ └─────────────┘ │
└─────────────────┘
```

### Production Deployment

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   ModelMuxer    │    │   Monitoring    │
│   (nginx/ALB)   │    │   Cluster       │    │   Stack         │
│                 │    │                 │    │                 │
│                 │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│                 │────│ │ ModelMuxer  │ │    │ │ Prometheus  │ │
│                 │    │ │ Instance 1  │ │    │ │ Grafana     │ │
│                 │    │ └─────────────┘ │    │ │ AlertMgr    │ │
│                 │    │ ┌─────────────┐ │    │ └─────────────┘ │
│                 │    │ │ ModelMuxer  │ │    └─────────────────┘
│                 │    │ │ Instance N  │ │
│                 │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────────────────┼───────────────────────┐
                                 │                       │
                    ┌─────────────────┐    ┌─────────────────┐
                    │   PostgreSQL    │    │   Database      │
                    │   Database      │    │   Cache         │
                    └─────────────────┘    └─────────────────┘
```

## Configuration Management

### Environment-Based Configuration

- Development: `.env` files
- Staging: Environment variables + secrets
- Production: Kubernetes secrets + ConfigMaps

### Configuration Hierarchy

1. Environment variables (highest priority)
2. Configuration files
3. Default values (lowest priority)

## Security Considerations

### Network Security

- TLS/SSL encryption in transit
- VPC/network isolation
- Firewall rules and security groups

### Application Security

- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CSRF protection

### Data Security

- Encryption at rest
- PII detection and masking
- Audit logging
- Access controls

## Performance Characteristics

### Throughput

- **Development**: ~100 requests/second
- **Production**: ~1000+ requests/second (with proper scaling)

### Latency

- **Routing Decision**: <10ms
- **Provider Response**: 100ms-5s (depends on model)
- **Total Response Time**: Provider latency + ~50ms overhead

### Scalability

- Horizontal scaling through load balancing
- Stateless application design
- Distributed caching with Redis
- Database connection pooling

## Monitoring and Alerting

### Key Metrics

- Request rate and response times
- Error rates by provider
- Cost per request and budget utilization
- Provider availability and performance

### Alerting Rules

- High error rates (>5%)
- Slow response times (>10s)
- Budget threshold breaches
- Provider outages

### Dashboards

- Real-time operational dashboard
- Cost analysis and optimization
- Provider performance comparison
- Security and compliance monitoring
