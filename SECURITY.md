# Security Policy

## Supported Versions

We actively support and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it to [security@modelmuxer.com](mailto:security@modelmuxer.com).

**Please do not report security vulnerabilities through public GitHub issues.**

### What to Include

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Any suggested fixes (if available)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution**: Varies based on severity and complexity

## Known Security Considerations

### PyTorch Vulnerability (GHSA-887c-mr87-cxwp)

**Status**: Acknowledged - Mitigated
**Affected Component**: PyTorch 2.7.1 (`torch.nn.functional.ctc_loss`)
**Severity**: Medium
**Impact**: Limited to optional ML routing features

**Mitigation Strategy**:

1. **Isolation**: PyTorch is in optional ML dependency group
2. **Deployment**: Core routing works without ML dependencies
3. **Monitoring**: CI pipeline monitors for updates
4. **Alternative**: Fallback routing available when ML disabled

**Deployment Recommendations**:

- For security-critical environments: Deploy without `--with ml` flag
- For ML-enabled deployments: Monitor PyTorch security advisories
- Use semantic router fallback mode in production

### Container Security

**Base Images**:

- Production: `python:3.11-slim` (Debian-based, regular security updates)
- Alpine: `python:3.12-alpine3.20` (Minimal attack surface)

**Security Measures**:

- Multi-stage builds to minimize final image size
- Non-root user execution
- Minimal package installation
- Regular base image updates

### Authentication & Authorization

**JWT Security**:

- Strong secret key generation required
- Token expiration enforced
- Role-based access control (RBAC)

**API Security**:

- Rate limiting implemented
- Input validation on all endpoints
- PII detection and redaction
- Audit logging for all requests

## Security Best Practices

### Deployment

1. **Environment Variables**: Never commit secrets to version control
2. **Network Security**: Use HTTPS/TLS in production
3. **Database Security**: Use encrypted connections and strong passwords
4. **Monitoring**: Enable comprehensive audit logging
5. **Updates**: Regularly update dependencies and base images

### Development

1. **Code Review**: All changes require review
2. **Static Analysis**: Automated security scanning in CI/CD
3. **Dependency Scanning**: Regular vulnerability assessments
4. **Testing**: Security-focused test cases

## Compliance

ModelMuxer implements security controls to support:

- SOC 2 Type II compliance
- GDPR data protection requirements
- Industry-standard security frameworks

## Contact

For security-related questions or concerns:

- **Security Team**: [security@modelmuxer.com](mailto:security@modelmuxer.com)
- **General Support**: [support@modelmuxer.com](mailto:support@modelmuxer.com)
