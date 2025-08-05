# Deployment Guide

This guide covers deploying ModelMuxer in various environments from development to production.

## Quick Deployment Options

### 1. Docker Compose (Recommended for Development)

> **Note**: On macOS, you can use Apple's built-in containerization instead of Docker Desktop. The commands below work with any Docker-compatible container runtime.

```bash
# Clone and setup
git clone https://github.com/iamapsrajput/modelmuxer.git
cd ModelMuxer

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Deploy with all services
docker-compose up -d

# Check status
docker-compose ps
```

### 2. Kubernetes with Helm (Recommended for Production)

```bash
# Add Helm repository
helm repo add modelmuxer https://charts.modelmuxer.com
helm repo update

# Install with custom values
helm install modelmuxer modelmuxer/modelmuxer \
  --set config.openai.apiKey=your_key \
  --set config.anthropic.apiKey=your_key \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=api.yourdomain.com
```

### 3. Cloud Platforms

#### AWS ECS

```bash
# Deploy using AWS CLI
aws ecs create-service \
  --cluster modelmuxer-cluster \
  --service-name modelmuxer \
  --task-definition modelmuxer:1 \
  --desired-count 3
```

#### Google Cloud Run

```bash
# Deploy to Cloud Run
gcloud run deploy modelmuxer \
  --image gcr.io/your-project/modelmuxer \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Production Deployment

### Architecture Overview

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│  ModelMuxer API │────│   Provider APIs │
│   (nginx/ALB)   │    │   (3+ replicas) │    │ (OpenAI, etc.)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │   PostgreSQL    │              │
         │              │   (Primary DB)  │              │
         │              └─────────────────┘              │
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────│      Redis      │──────────────┘
                        │   (Cache/Queue) │
                        └─────────────────┘
```

### Prerequisites

- Kubernetes cluster (1.20+) or Docker Swarm
- PostgreSQL database (12+)
- Redis instance (6+)
- SSL certificates
- Domain name and DNS configuration

### Step 1: Database Setup

#### PostgreSQL

```sql
-- Create database and user
CREATE DATABASE modelmuxer;
CREATE USER modelmuxer_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE modelmuxer TO modelmuxer_user;

-- Enable required extensions
\c modelmuxer;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
```

#### Redis Configuration

```redis
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### Step 2: Environment Configuration

Create production environment file:

```env
# Production .env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=8

# Database
DATABASE_URL=postgresql://modelmuxer_user:secure_password@db:5432/modelmuxer
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis
REDIS_URL=redis://redis:6379/0
CACHE_ENABLED=true
CACHE_DEFAULT_TTL=3600

# Security
JWT_SECRET_KEY=your-production-secret-key
CORS_ORIGINS=["https://yourdomain.com"]

# Monitoring
MONITORING_ENABLED=true
METRICS_ENABLED=true
PROMETHEUS_PORT=9090

# Provider APIs
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
# ... other provider keys
```

### Step 3: Kubernetes Deployment

#### Namespace and ConfigMap

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: modelmuxer

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: modelmuxer-config
  namespace: modelmuxer
data:
  DATABASE_URL: "postgresql://modelmuxer_user:password@postgres:5432/modelmuxer"
  REDIS_URL: "redis://redis:6379/0"
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
```

#### Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: modelmuxer-secrets
  namespace: modelmuxer
type: Opaque
stringData:
  OPENAI_API_KEY: "your_openai_key"
  ANTHROPIC_API_KEY: "your_anthropic_key"
  JWT_SECRET_KEY: "your_jwt_secret"
```

#### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: modelmuxer
  namespace: modelmuxer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: modelmuxer
  template:
    metadata:
      labels:
        app: modelmuxer
    spec:
      containers:
        - name: modelmuxer
          image: modelmuxer:latest
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: modelmuxer-config
            - secretRef:
                name: modelmuxer-secrets
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
```

#### Service and Ingress

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: modelmuxer-service
  namespace: modelmuxer
spec:
  selector:
    app: modelmuxer
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: modelmuxer-ingress
  namespace: modelmuxer
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
    - hosts:
        - api.yourdomain.com
      secretName: modelmuxer-tls
  rules:
    - host: api.yourdomain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: modelmuxer-service
                port:
                  number: 80
```

### Step 4: Monitoring Setup

#### Prometheus Configuration

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'modelmuxer'
      static_configs:
      - targets: ['modelmuxer-service:9090']
```

#### Grafana Dashboard

Import the ModelMuxer dashboard from `monitoring/grafana/dashboard.json`.

### Step 5: SSL and Security

#### Let's Encrypt with cert-manager

```bash
# Install cert-manager
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.8.0/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@domain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### Step 6: Backup and Recovery

#### Database Backup

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h postgres -U modelmuxer_user modelmuxer > $BACKUP_DIR/modelmuxer_$DATE.sql
```

#### Redis Backup

```bash
# Redis backup
redis-cli --rdb /backups/redis_backup_$(date +%Y%m%d).rdb
```

## Scaling and Performance

### Horizontal Scaling

```bash
# Scale deployment
kubectl scale deployment modelmuxer --replicas=6

# Auto-scaling
kubectl autoscale deployment modelmuxer --cpu-percent=70 --min=3 --max=10
```

### Performance Tuning

#### Database Optimization

```sql
-- PostgreSQL tuning
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
```

#### Redis Optimization

```redis
# redis.conf
tcp-keepalive 300
timeout 0
tcp-backlog 511
maxclients 10000
```

## Monitoring and Alerting

### Key Metrics to Monitor

- Request rate and latency
- Error rates by provider
- Cost per request
- Cache hit rates
- Database connection pool usage
- Memory and CPU usage

### Alerting Rules

```yaml
# alerts.yaml
groups:
  - name: modelmuxer
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        annotations:
          summary: High error rate detected

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        annotations:
          summary: High latency detected
```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**

   ```bash
   kubectl logs -f deployment/modelmuxer | grep database
   ```

2. **Provider API Failures**

   ```bash
   kubectl exec -it deployment/modelmuxer -- curl http://localhost:8000/health/detailed
   ```

3. **Memory Issues**

   ```bash
   kubectl top pods -n modelmuxer
   ```

### Health Checks

```bash
# Check all components
curl https://api.yourdomain.com/health/detailed

# Check specific provider
curl https://api.yourdomain.com/api/v1/providers/openai/health
```

## Next Steps

- [Monitoring Guide](monitoring.md)
- [Security Guide](security.md)
- [Performance Tuning](performance.md)
