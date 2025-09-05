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

## Data Flow

### Request Processing Pipeline

```
Client Request → Authentication → Rate Limiting → Query Analysis →
Model Selection → Provider API Call → Response Processing →
Cost Calculation → Caching → Response to Client
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

- JWT token validation
- API key management
- Role-based access control
- Rate limiting per user/organization

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
