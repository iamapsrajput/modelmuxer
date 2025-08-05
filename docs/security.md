# Security Guide

## Overview

ModelMuxer implements multiple layers of security to protect your data, API keys, and infrastructure.

## Authentication and Authorization

### API Key Management
- Store API keys securely in environment variables
- Never commit API keys to version control
- Rotate keys regularly
- Use different keys for different environments

### JWT Token Authentication
```python
# Configure JWT settings
JWT_SECRET_KEY = "your-secure-secret-key"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
```

### Role-Based Access Control (RBAC)
- Admin: Full system access
- User: Standard API access with quotas
- Readonly: Monitoring and analytics only

## Data Protection

### Encryption at Rest
- Database encryption for sensitive data
- Encrypted configuration files
- Secure key storage using HashiCorp Vault or AWS KMS

### Encryption in Transit
- TLS 1.3 for all API communications
- Certificate pinning for provider connections
- End-to-end encryption for sensitive requests

### Data Retention
- Configurable log retention periods
- Automatic data purging
- GDPR compliance features

## Network Security

### API Rate Limiting
```python
# Rate limiting configuration
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_STORAGE = "redis://localhost:6379"
```

### IP Whitelisting
```python
# Allow specific IP ranges
ALLOWED_IPS = [
    "192.168.1.0/24",
    "10.0.0.0/8"
]
```

### CORS Configuration
```python
# Secure CORS settings
CORS_ALLOWED_ORIGINS = [
    "https://yourdomain.com",
    "https://api.yourdomain.com"
]
CORS_ALLOW_CREDENTIALS = True
```

## Input Validation and Sanitization

### Request Validation
- Pydantic models for all API inputs
- Schema validation for all parameters
- SQL injection prevention
- XSS protection

### Content Filtering
- Prompt sanitization
- Malicious content detection
- PII detection and redaction

## Provider Security

### API Key Isolation
- Separate credentials per provider
- Key rotation without service interruption
- Provider-specific security policies

### Request Signing
- HMAC signing for critical requests
- Timestamp validation
- Replay attack prevention

## Audit and Compliance

### Audit Logging
```python
# Security events logged
- Authentication attempts
- API key usage
- Configuration changes
- Suspicious activities
- Data access patterns
```

### Compliance Features
- SOC 2 Type II compatible logging
- GDPR data handling
- HIPAA compliance options
- PCI DSS security controls

## Vulnerability Management

### Security Scanning
- Automated dependency scanning
- Container image vulnerability scanning
- Static code analysis (SAST)
- Dynamic application security testing (DAST)

### Security Updates
```bash
# Regular security updates
poetry update
poetry audit
```

## Incident Response

### Security Incident Handling
1. **Detection**: Automated monitoring and alerting
2. **Containment**: Immediate threat isolation
3. **Investigation**: Root cause analysis
4. **Recovery**: Service restoration
5. **Lessons Learned**: Process improvement

### Emergency Procedures
- API key revocation process
- Service isolation protocols
- Communication templates
- Recovery checklists

## Development Security

### Secure Coding Practices
- Input validation on all user inputs
- Parameterized queries to prevent SQL injection
- Secure random number generation
- Proper error handling without information disclosure

### Security Testing
```bash
# Security testing commands
poetry run bandit -r app/
poetry run safety check
poetry run semgrep --config=auto app/
```

### Pre-commit Security Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
  - repo: https://github.com/gitguardian/ggshield
    rev: v1.18.0
    hooks:
      - id: ggshield
        language: python
        stages: [commit]
```

## Deployment Security

### Container Security
```dockerfile
# Security-hardened Dockerfile
FROM python:3.11-slim-bullseye
RUN useradd --create-home --shell /bin/bash modelmuxer
USER modelmuxer
COPY --chown=modelmuxer:modelmuxer . /app
```

### Environment Security
- Separate environments (dev/staging/prod)
- Environment-specific configurations
- Secrets management with external vaults
- Network segmentation

### Production Hardening
- Remove debug endpoints in production
- Disable verbose error messages
- Implement proper logging without sensitive data
- Use secure session management

## Monitoring and Detection

### Security Monitoring
- Failed authentication tracking
- Unusual access patterns
- API abuse detection
- Anomaly detection

### Alerting
```python
# Security alerts
- Multiple failed login attempts
- Unusual API usage patterns
- Quota exceeded warnings
- Suspicious geographic access
```

## Configuration Security

### Secure Defaults
- Strong password policies
- Secure communication protocols
- Minimal permissions principle
- Regular security updates

### Environment Variables
```bash
# Secure configuration management
export OPENAI_API_KEY="sk-..."
export JWT_SECRET_KEY="$(openssl rand -base64 32)"
export DATABASE_URL="postgresql://..."
```

## Best Practices

1. **Never hardcode secrets** in source code
2. **Use HTTPS everywhere** for API communications
3. **Implement proper logging** without exposing sensitive data
4. **Regular security audits** and penetration testing
5. **Keep dependencies updated** and scan for vulnerabilities
6. **Use least privilege principle** for all access controls
7. **Implement defense in depth** with multiple security layers

## Security Checklist

### Pre-deployment
- [ ] All secrets in environment variables
- [ ] HTTPS configured with valid certificates
- [ ] Rate limiting enabled
- [ ] Input validation implemented
- [ ] Audit logging configured
- [ ] Security headers set
- [ ] Dependencies scanned for vulnerabilities

### Production
- [ ] Regular security updates
- [ ] Monitoring and alerting active
- [ ] Backup and recovery tested
- [ ] Incident response plan documented
- [ ] Security training completed
- [ ] Compliance requirements met

## Contact

For security issues or questions:
- Email: security@modelmuxer.com
- Security advisory: Report privately via GitHub Security Advisories

**Note**: Please do not report security vulnerabilities in public issues.
