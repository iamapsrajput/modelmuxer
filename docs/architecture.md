# ModelMuxer Architecture

## Overview

ModelMuxer is an enterprise-grade LLM routing platform designed for production
scale with intelligent request routing, cost optimization, and comprehensive
monitoring.

## Routing Architecture: Direct Providers First

ModelMuxer uses a direct-provider-only architecture that provides direct API
connections for maximum reliability, performance, and control.

### Primary Routing: Direct Providers

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │───▶│  ModelMuxer │───▶│   Direct    │───▶│   Provider  │
│  Request    │    │   Router    │    │  Provider   │    │     API     │
│             │    │             │    │  Adapter    │    │  (OpenAI,   │
└─────────────┘    └─────────────┘    └─────────────┘    │ Anthropic,  │
                                                         │   etc.)     │
                                                         └─────────────┘
         │                │                │                │
         ▼                ▼                ▼                ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Intent     │    │  Cost       │    │  Circuit    │    │  Provider-  │
│Classifier   │    │Estimation   │    │  Breaker    │    │  Specific   │
│             │    │             │    │             │    │  Error      │
└─────────────┘    └─────────────┘    └─────────────┘    │  Handling   │
                                                         └─────────────┘
```

**Benefits:**

- **Lower latency**: No proxy overhead, direct API connections
- **Provider-specific error handling**: Tailored retry logic and circuit
  breakers
- **Enhanced observability**: Detailed telemetry and metrics per provider
- **Cost optimization**: Direct pricing and real-time cost tracking
- **Better control**: Fine-grained configuration and monitoring
- **Reduced complexity**: Fewer moving parts in the critical path

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
                    │   optional      │
                    │   Redis)        │
                    └─────────────────┘
```

## Core Components

### 1. Application Layers

#### **Application Entry Point (`app/main.py`)**

- App factory and lifespan management (database, provider registry, router
  initialization)
- CORS, security-headers, request-size, and observability middleware
- Exception handlers and the `get_authenticated_user` dependency
- CLI entry points (`--mode basic|production`; production enables strict
  startup validation)
- **Suitable for**: Development, testing, and production deployments

#### **HTTP Route Modules (`app/api/routes/`)**

- `chat.py`: `POST /v1/chat/completions`, streaming helper, `POST /v1/messages`
  and `/messages` (Anthropic compatibility)
- `system.py`: `GET /health`, `GET /metrics`, `GET /metrics/prometheus`
- `analytics.py`: `GET /v1/analytics/costs`, `GET/POST /v1/analytics/budgets`,
  `GET /user/stats`
- `providers.py`: `GET /providers`, `GET /v1/providers`, `GET /v1/models`

### 2. Routing Engine (`app/router.py`)

#### **Heuristic Router (`HeuristicRouter`)**

- The single routing strategy in ModelMuxer
- Rule-based routing using prompt analysis; fast, deterministic decisions
- Intent classification via `app/core/intent.py` (heuristics with optional
  cheap-LLM assist)
- Cost estimation and budget gating via `app/core/costing.py` with automatic
  down-routing to cheaper models
- In-memory latency priors (p95/p99) for ETA estimates
- Invokes providers through the adapter registry
  (`HeuristicRouter.invoke_via_adapter`)

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

## Data Flow

### Request Processing Pipeline

```
Client Request → Authentication (APIKeyAuth) → Policy Enforcement →
HeuristicRouter.select_model (intent + cost estimate + budget gate) →
Provider Adapter (via registry) → Response Processing →
Cost Tracking + Telemetry Metrics → Response to Client
```

### Analytics Pipeline

```
Request Metrics → Real-time Aggregation → Time-series Storage →
Dashboard Updates → Alert Processing → Reporting
```

## Scalability Design

### Horizontal Scaling

- Stateless application design
- Load balancer distribution
- Auto-scaling based on metrics
- Database read replicas

### Vertical Scaling

- Async request handling
- Connection pooling
- Memory optimization
- CPU-efficient algorithms

## Security Architecture

### Authentication & Authorization

- API key authentication (`APIKeyAuth` in `app/auth.py`)
- Rate limiting per user

### Data Protection

- Encryption at rest and in transit
- PII detection and redaction
- Audit logging
- Secure configuration management

## Deployment Architecture

### Production Setup

- Multi-zone deployment for high availability
- Database clustering with failover
- Redis cluster for caching
- Load balancers with health checks

### Monitoring & Observability

- Application performance monitoring (APM)
- Distributed tracing
- Error tracking and alerting
- Business metrics dashboards
