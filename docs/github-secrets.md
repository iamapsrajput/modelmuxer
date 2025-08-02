# GitHub Repository Secrets Configuration

This document outlines all the GitHub repository secrets that need to be configured for CI/CD pipelines and automated deployments.

## Required Secrets for CI/CD

### 1. LLM Provider API Keys

These are required for integration tests and deployment:

```
OPENAI_API_KEY
ANTHROPIC_API_KEY
MISTRAL_API_KEY
GOOGLE_API_KEY
GROQ_API_KEY
TOGETHER_API_KEY
COHERE_API_KEY
```

**Note**: At least one provider API key must be configured for tests to pass.

### 2. Authentication & Security

```
JWT_SECRET_KEY          # JWT signing key (generate with: python -c 'import secrets; print(secrets.token_urlsafe(32))')
ENCRYPTION_KEY          # General encryption key
PII_ENCRYPTION_KEY      # PII-specific encryption key
```

### 3. Database Configuration

```
DATABASE_URL            # Production database connection string
DATABASE_USER           # Database username
DATABASE_PASSWORD       # Database password
```

### 4. Cache Configuration

```
REDIS_URL              # Redis connection string
REDIS_PASSWORD         # Redis password (if required)
```

### 5. Container Registry

```
DOCKER_USERNAME        # Docker Hub username
DOCKER_PASSWORD        # Docker Hub password or access token
GHCR_TOKEN            # GitHub Container Registry token
```

### 6. Cloud Provider Credentials

#### AWS (if deploying to AWS)
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
```

#### Google Cloud (if deploying to GCP)
```
GCP_SERVICE_ACCOUNT_KEY    # Base64 encoded service account JSON
GCP_PROJECT_ID
```

#### Azure (if deploying to Azure)
```
AZURE_CLIENT_ID
AZURE_CLIENT_SECRET
AZURE_TENANT_ID
AZURE_SUBSCRIPTION_ID
```

### 7. Kubernetes Deployment

```
KUBE_CONFIG            # Base64 encoded kubeconfig file
KUBE_NAMESPACE         # Target namespace
```

### 8. Monitoring & Observability

```
PROMETHEUS_AUTH_TOKEN  # Prometheus authentication token
GRAFANA_API_KEY       # Grafana API key for dashboard management
SENTRY_DSN            # Sentry error tracking DSN
```

### 9. Notification Services

```
SLACK_WEBHOOK_URL     # Slack webhook for deployment notifications
DISCORD_WEBHOOK_URL   # Discord webhook for notifications
```

## Setting Up GitHub Secrets

### Via GitHub Web Interface

1. Go to your repository on GitHub
2. Click on **Settings** tab
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Add each secret with its name and value

### Via GitHub CLI

```bash
# Install GitHub CLI if not already installed
# brew install gh  # macOS
# sudo apt install gh  # Ubuntu

# Authenticate
gh auth login

# Set secrets
gh secret set OPENAI_API_KEY --body "your-openai-api-key"
gh secret set ANTHROPIC_API_KEY --body "your-anthropic-api-key"
gh secret set JWT_SECRET_KEY --body "$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

# Set from file
gh secret set KUBE_CONFIG < path/to/kubeconfig.yaml
```

## Environment-Specific Secrets

### Development Environment
- Use `.env` file locally (never commit this)
- Minimal set of API keys for testing

### Staging Environment
```
STAGING_DATABASE_URL
STAGING_REDIS_URL
STAGING_JWT_SECRET_KEY
```

### Production Environment
```
PROD_DATABASE_URL
PROD_REDIS_URL
PROD_JWT_SECRET_KEY
PROD_ENCRYPTION_KEY
```

## Security Best Practices

### 1. Secret Rotation
- Rotate secrets regularly (every 90 days recommended)
- Use GitHub's secret scanning to detect exposed secrets
- Monitor for unauthorized access

### 2. Least Privilege Access
- Only grant necessary permissions to service accounts
- Use separate credentials for different environments
- Implement proper RBAC in Kubernetes

### 3. Secret Management
- Use strong, randomly generated secrets
- Never log or expose secrets in application code
- Use secret management services (AWS Secrets Manager, Azure Key Vault, etc.)

### 4. Validation
- Validate secrets format and permissions during CI/CD
- Test secret rotation procedures
- Monitor secret usage and access patterns

## CI/CD Pipeline Configuration

### GitHub Actions Workflow Example

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure environment
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        JWT_SECRET_KEY: ${{ secrets.JWT_SECRET_KEY }}
        DATABASE_URL: ${{ secrets.PROD_DATABASE_URL }}
        REDIS_URL: ${{ secrets.PROD_REDIS_URL }}
      run: |
        echo "Environment configured"
        
    - name: Deploy to Kubernetes
      env:
        KUBE_CONFIG: ${{ secrets.KUBE_CONFIG }}
      run: |
        echo "$KUBE_CONFIG" | base64 -d > kubeconfig
        kubectl --kubeconfig=kubeconfig apply -f k8s/
```

## Troubleshooting

### Common Issues

1. **Secret Not Found**
   - Verify secret name matches exactly (case-sensitive)
   - Check if secret is set at repository or organization level
   - Ensure workflow has access to the secret

2. **Invalid Secret Format**
   - Validate secret format (base64 encoding, JSON structure, etc.)
   - Check for trailing whitespace or newlines
   - Verify special characters are properly escaped

3. **Permission Denied**
   - Check service account permissions
   - Verify API key scopes and permissions
   - Ensure proper RBAC configuration

### Validation Scripts

```bash
# Validate required secrets are set
#!/bin/bash
REQUIRED_SECRETS=(
    "OPENAI_API_KEY"
    "JWT_SECRET_KEY"
    "DATABASE_URL"
)

for secret in "${REQUIRED_SECRETS[@]}"; do
    if [ -z "${!secret}" ]; then
        echo "❌ Missing required secret: $secret"
        exit 1
    else
        echo "✅ Secret $secret is configured"
    fi
done
```

## Migration from Other CI/CD Systems

### From Jenkins
- Export environment variables as GitHub secrets
- Update Jenkinsfile syntax to GitHub Actions workflow
- Migrate credential stores to GitHub secrets

### From GitLab CI
- Export GitLab CI/CD variables as GitHub secrets
- Update `.gitlab-ci.yml` to `.github/workflows/*.yml`
- Migrate GitLab Container Registry to GitHub Container Registry

## Monitoring and Auditing

### Secret Usage Monitoring
- Enable GitHub audit logs
- Monitor secret access patterns
- Set up alerts for unusual secret usage

### Compliance
- Document secret access and rotation procedures
- Implement approval workflows for secret changes
- Regular security audits and penetration testing

## Support and Resources

- [GitHub Secrets Documentation](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [GitHub CLI Secrets Commands](https://cli.github.com/manual/gh_secret)
- [Security Best Practices](https://docs.github.com/en/actions/security-guides)

For questions or issues with secret configuration, please:
1. Check this documentation first
2. Review GitHub Actions logs for specific error messages
3. Contact the DevOps team or create an issue in the repository
