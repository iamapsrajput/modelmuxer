# ModelMuxer Production Deployment Guide

This guide provides comprehensive instructions for deploying ModelMuxer in a production environment with high availability, security, and scalability.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Security Setup](#security-setup)
4. [Database Configuration](#database-configuration)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Monitoring Setup](#monitoring-setup)
7. [SSL/TLS Configuration](#ssltls-configuration)
8. [Backup and Recovery](#backup-and-recovery)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools
- **Kubernetes cluster** (v1.25+) with at least 3 nodes
- **Helm** (v3.12+) for package management
- **kubectl** configured for your cluster
- **Docker** for image building (if customizing)
- **OpenSSL** for certificate generation

### Required Services
- **PostgreSQL** (v15+) with read replicas
- **Redis** (v7+) cluster for caching
- **Load Balancer** (AWS ALB, GCP Load Balancer, or NGINX)
- **DNS** management for domain configuration
- **Certificate Authority** for SSL/TLS certificates

### Minimum Resource Requirements

| Component | CPU | Memory | Storage | Replicas |
|-----------|-----|--------|---------|----------|
| ModelMuxer API | 2 cores | 4GB | - | 3+ |
| PostgreSQL Primary | 4 cores | 8GB | 100GB SSD | 1 |
| PostgreSQL Replica | 2 cores | 4GB | 100GB SSD | 2+ |
| Redis Cluster | 1 core | 2GB | 10GB SSD | 6 |
| Monitoring Stack | 4 cores | 8GB | 50GB SSD | - |

## Infrastructure Requirements

### Cloud Provider Setup

#### AWS
```bash
# Create EKS cluster
eksctl create cluster \
  --name modelmuxer-prod \
  --version 1.28 \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed

# Configure storage classes
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
EOF
```

#### Google Cloud Platform
```bash
# Create GKE cluster
gcloud container clusters create modelmuxer-prod \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10 \
  --enable-autorepair \
  --enable-autoupgrade
```

### Network Configuration

#### Security Groups / Firewall Rules
```yaml
# Allow HTTPS traffic (443)
# Allow HTTP traffic (80) - redirect to HTTPS
# Allow PostgreSQL (5432) - internal only
# Allow Redis (6379) - internal only
# Allow Kubernetes API (6443) - restricted IPs
# Allow monitoring (9090, 3000) - internal only
```

## Security Setup

### 1. Create Kubernetes Namespace and RBAC

```bash
# Create production namespace
kubectl create namespace modelmuxer-production

# Apply RBAC configuration
kubectl apply -f k8s/namespace.yaml
```

### 2. Generate and Store Secrets

```bash
# Generate encryption keys
ENCRYPTION_KEY=$(openssl rand -base64 32)
JWT_SECRET_KEY=$(openssl rand -base64 64)
PII_ENCRYPTION_KEY=$(openssl rand -base64 32)

# Create secrets
kubectl create secret generic modelmuxer-secrets \
  --namespace=modelmuxer-production \
  --from-literal=DATABASE_URL="postgresql://user:password@postgresql-primary:5432/modelmuxer" \
  --from-literal=REDIS_URL="redis://:password@redis-cluster:6379/0" \
  --from-literal=JWT_SECRET_KEY="$JWT_SECRET_KEY" \
  --from-literal=ENCRYPTION_KEY="$ENCRYPTION_KEY" \
  --from-literal=PII_ENCRYPTION_KEY="$PII_ENCRYPTION_KEY" \
  --from-literal=OPENAI_API_KEY="your-openai-key" \
  --from-literal=ANTHROPIC_API_KEY="your-anthropic-key" \
  --from-literal=GOOGLE_API_KEY="your-google-key" \
  --from-literal=MISTRAL_API_KEY="your-mistral-key" \
  --from-literal=GROQ_API_KEY="your-groq-key"
```

### 3. SSL/TLS Certificate Setup

```bash
# Using cert-manager with Let's Encrypt
helm repo add jetstack https://charts.jetstack.io
helm repo update

helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.13.0 \
  --set installCRDs=true

# Create ClusterIssuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@yourdomain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

## Database Configuration

### 1. Deploy PostgreSQL with High Availability

```bash
# Add Bitnami Helm repository
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Deploy PostgreSQL with read replicas
helm install postgresql-production bitnami/postgresql \
  --namespace modelmuxer-production \
  --set auth.postgresPassword="$POSTGRES_PASSWORD" \
  --set auth.database=modelmuxer \
  --set primary.persistence.size=100Gi \
  --set primary.persistence.storageClass=fast-ssd \
  --set readReplicas.replicaCount=2 \
  --set readReplicas.persistence.size=100Gi \
  --set readReplicas.persistence.storageClass=fast-ssd \
  --set metrics.enabled=true \
  --set metrics.serviceMonitor.enabled=true
```

### 2. Configure Database Backups

```bash
# Create backup CronJob
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgresql-backup
  namespace: modelmuxer-production
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: postgres-backup
            image: postgres:15
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgresql-production
                  key: postgres-password
            command:
            - /bin/bash
            - -c
            - |
              BACKUP_FILE="modelmuxer-backup-$(date +%Y%m%d-%H%M%S).sql"
              pg_dump -h postgresql-production -U postgres -d modelmuxer > /backup/$BACKUP_FILE
              # Upload to S3 or your backup storage
              aws s3 cp /backup/$BACKUP_FILE s3://your-backup-bucket/postgresql/
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            emptyDir: {}
          restartPolicy: OnFailure
EOF
```

### 3. Run Database Migrations

```bash
# Run initial migrations
kubectl run migration-job \
  --namespace=modelmuxer-production \
  --image=ghcr.io/your-org/modelmuxer:latest \
  --restart=Never \
  --env="DATABASE_URL=postgresql://postgres:$POSTGRES_PASSWORD@postgresql-production:5432/modelmuxer" \
  --command -- python -m alembic upgrade head

# Wait for completion
kubectl wait --for=condition=complete --timeout=300s job/migration-job -n modelmuxer-production
```

## Kubernetes Deployment

### 1. Deploy Redis Cluster

```bash
# Deploy Redis cluster
helm install redis-production bitnami/redis \
  --namespace modelmuxer-production \
  --set architecture=replication \
  --set auth.password="$REDIS_PASSWORD" \
  --set master.persistence.size=10Gi \
  --set master.persistence.storageClass=fast-ssd \
  --set replica.replicaCount=3 \
  --set replica.persistence.size=10Gi \
  --set replica.persistence.storageClass=fast-ssd \
  --set sentinel.enabled=true \
  --set metrics.enabled=true \
  --set metrics.serviceMonitor.enabled=true
```

### 2. Deploy ModelMuxer Application

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml  # Update with your values first
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/pdb.yaml

# Verify deployment
kubectl get pods -n modelmuxer-production
kubectl get services -n modelmuxer-production
kubectl get ingress -n modelmuxer-production
```

### 3. Configure Horizontal Pod Autoscaling

```bash
# Verify HPA is working
kubectl get hpa -n modelmuxer-production

# Check metrics
kubectl top pods -n modelmuxer-production
```

## Monitoring Setup

### 1. Deploy Prometheus Stack

```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Deploy kube-prometheus-stack
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
  --set grafana.adminPassword="$GRAFANA_ADMIN_PASSWORD" \
  --set grafana.persistence.enabled=true \
  --set grafana.persistence.size=10Gi \
  --set alertmanager.alertmanagerSpec.storage.volumeClaimTemplate.spec.resources.requests.storage=10Gi
```

### 2. Configure Grafana Dashboards

```bash
# Import ModelMuxer dashboards
kubectl create configmap modelmuxer-dashboards \
  --namespace monitoring \
  --from-file=monitoring/grafana/dashboards/

# Label for automatic discovery
kubectl label configmap modelmuxer-dashboards \
  --namespace monitoring \
  grafana_dashboard=1
```

### 3. Set Up Alerting

```bash
# Apply alerting rules
kubectl apply -f monitoring/prometheus/rules/modelmuxer-alerts.yml
```

## Performance Tuning

### 1. Application Configuration

```yaml
# Update ConfigMap for production settings
apiVersion: v1
kind: ConfigMap
metadata:
  name: modelmuxer-config
  namespace: modelmuxer-production
data:
  # Database connection pooling
  DATABASE_POOL_SIZE: "20"
  DATABASE_MAX_OVERFLOW: "30"
  
  # Redis configuration
  REDIS_MAX_CONNECTIONS: "100"
  
  # Cache settings
  CACHE_DEFAULT_TTL: "3600"
  CACHE_MAX_SIZE: "10000"
  
  # Rate limiting
  RATE_LIMIT_ENABLED: "true"
  
  # Monitoring
  METRICS_ENABLED: "true"
  HEALTH_CHECK_ENABLED: "true"
```

### 2. Resource Optimization

```yaml
# Update deployment resources
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi
```

### 3. Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX CONCURRENTLY idx_requests_user_timestamp ON requests(user_id, timestamp);
CREATE INDEX CONCURRENTLY idx_requests_provider_model ON requests(provider, model);
CREATE INDEX CONCURRENTLY idx_audit_logs_org_timestamp ON audit_logs(organization_id, timestamp);
CREATE INDEX CONCURRENTLY idx_usage_metrics_org_period ON usage_metrics(organization_id, period_start);

-- Update PostgreSQL configuration
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET work_mem = '32MB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
ALTER SYSTEM SET max_connections = 200;
SELECT pg_reload_conf();
```

## Backup and Recovery

### 1. Database Backup Strategy

```bash
# Full backup script
#!/bin/bash
BACKUP_DIR="/backups/postgresql"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="modelmuxer_full_backup_$TIMESTAMP.sql"

# Create backup
pg_dump -h postgresql-production -U postgres -d modelmuxer \
  --verbose --format=custom --compress=9 \
  --file="$BACKUP_DIR/$BACKUP_FILE"

# Upload to cloud storage
aws s3 cp "$BACKUP_DIR/$BACKUP_FILE" s3://your-backup-bucket/postgresql/

# Cleanup old backups (keep last 30 days)
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
```

### 2. Application State Backup

```bash
# Backup Redis data
kubectl exec -n modelmuxer-production redis-production-master-0 -- \
  redis-cli --rdb /data/dump.rdb

# Backup configuration
kubectl get configmaps -n modelmuxer-production -o yaml > config-backup.yaml
kubectl get secrets -n modelmuxer-production -o yaml > secrets-backup.yaml
```

### 3. Disaster Recovery Plan

1. **Database Recovery**:
   ```bash
   # Restore from backup
   pg_restore -h postgresql-production -U postgres -d modelmuxer \
     --verbose --clean --if-exists backup_file.sql
   ```

2. **Application Recovery**:
   ```bash
   # Redeploy application
   kubectl apply -f k8s/
   
   # Verify health
   kubectl get pods -n modelmuxer-production
   ```

3. **Data Validation**:
   ```bash
   # Run health checks
   curl -f https://api.modelmuxer.com/health
   
   # Verify database connectivity
   kubectl exec -n modelmuxer-production deployment/modelmuxer -- \
     python -c "from app.database import engine; print(engine.execute('SELECT 1').scalar())"
   ```

## Troubleshooting

### Common Issues

#### 1. Pod Startup Issues
```bash
# Check pod logs
kubectl logs -n modelmuxer-production deployment/modelmuxer

# Check events
kubectl get events -n modelmuxer-production --sort-by='.lastTimestamp'

# Check resource usage
kubectl top pods -n modelmuxer-production
```

#### 2. Database Connection Issues
```bash
# Test database connectivity
kubectl run db-test --rm -i --tty \
  --image=postgres:15 \
  --namespace=modelmuxer-production \
  -- psql -h postgresql-production -U postgres -d modelmuxer

# Check database logs
kubectl logs -n modelmuxer-production postgresql-production-0
```

#### 3. Performance Issues
```bash
# Check metrics
kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80

# Analyze slow queries
kubectl exec -n modelmuxer-production postgresql-production-0 -- \
  psql -U postgres -d modelmuxer -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"
```

### Health Check Endpoints

- **Application Health**: `https://api.modelmuxer.com/health`
- **Readiness**: `https://api.modelmuxer.com/health/ready`
- **Liveness**: `https://api.modelmuxer.com/health/live`
- **Metrics**: `https://api.modelmuxer.com/metrics`

### Support Contacts

- **Technical Support**: support@modelmuxer.com
- **Emergency**: emergency@modelmuxer.com
- **Documentation**: https://docs.modelmuxer.com
- **Status Page**: https://status.modelmuxer.com

---

For additional support or questions about production deployment, please contact our support team or refer to the comprehensive documentation at https://docs.modelmuxer.com.
