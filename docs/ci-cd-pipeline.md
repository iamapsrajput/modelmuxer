# CI/CD Pipeline Documentation

## Overview

ModelMuxer implements a comprehensive CI/CD pipeline using GitHub Actions, providing automated testing, security scanning, quality assurance, and deployment capabilities.

## Pipeline Architecture

### Workflow Structure

The CI/CD pipeline consists of several specialized workflows:

1. **Test Workflow** (`.github/workflows/test.yaml`)
2. **Build Workflow** (`.github/workflows/build.yaml`)
3. **Security Workflow** (`.github/workflows/security.yaml`)
4. **License Compliance** (`.github/workflows/license-compliance.yaml`)

## Test Workflow

### Multi-Environment Testing

- **Python versions**: 3.11, 3.12
- **Operating systems**: Ubuntu Latest, macOS Latest, Windows Latest
- **Test modes**: Basic, Enhanced, Production

### Test Execution Strategy

```yaml
# Test matrix configuration
strategy:
  matrix:
    python-version: ["3.11", "3.12"]
    os: [ubuntu-latest, macos-latest, windows-latest]
    mode: [basic, enhanced, production]
```

### Quality Gates

- **116+ unit and integration tests** must pass
- **Code coverage** reporting with HTML output
- **Ruff formatting compliance** - all code must pass `ruff format --check`
- **Ruff linting standards** - zero linting violations allowed
- **MyPy type checking** - strict type safety enforcement
- **Performance benchmarking** with custom PR comments and artifact storage
- **Dependency vulnerability scanning**

## Build Workflow

### Multi-Platform Docker Builds

#### Platform Strategy

- **Pull Requests**: Single platform (`linux/amd64`) for faster iteration
- **Production Builds**: Multi-platform (`linux/amd64`, `linux/arm64`)

#### Build Features

- **Conditional attestations**: SBOM and provenance only for production
- **GitHub Actions caching**: Optimized build performance
- **Vulnerability scanning**: Integrated Trivy security scanning
- **Smart tagging**: Dynamic tag generation based on branch and event

### Build Configuration

```yaml
# Dynamic platform selection
platforms: ${{ github.event_name == 'pull_request' && 'linux/amd64' || 'linux/amd64,linux/arm64' }}
push: ${{ github.event_name != 'pull_request' }}
load: ${{ github.event_name == 'pull_request' }}
provenance: ${{ github.event_name != 'pull_request' }}
sbom: ${{ github.event_name != 'pull_request' }}
```

## Security Workflow

### Comprehensive Security Scanning

#### Secret Detection

- **Gitleaks integration**: Advanced secret scanning with custom configuration
- **False positive filtering**: Smart exclusions for documentation and test files
- **Pattern recognition**: Detects various credential formats and API keys

#### Vulnerability Assessment

- **Trivy container scanning**: Multi-layer vulnerability detection
- **Dependency scanning**: Third-party package vulnerability monitoring
- **SARIF reporting**: Standardized security findings format

#### Code Quality Security

- **Semgrep static analysis**: Advanced security pattern detection with SARIF reporting
- **Bandit static analysis**: Python-specific security issue detection
- **License compliance**: Automated license compatibility checking
- **Supply chain security**: SBOM generation and provenance tracking

### Gitleaks Configuration

Custom `.gitleaks.toml` configuration provides:

- **Path-based exclusions**: Automatic filtering of documentation and test files
- **Pattern matching**: Recognition of legitimate placeholders and examples
- **Stop word filtering**: Semantic analysis to identify non-secrets
- **Comprehensive coverage**: Supports multiple credential formats

## License Compliance Workflow

### Automated License Checking

- **Header validation**: Ensures all source files include proper license headers
- **Dependency licensing**: Validates third-party package licenses
- **Compliance reporting**: Generates compliance status reports

## Workflow Permissions

### Job-Level Permissions

- **Performance Testing**: `contents: read`, `pull-requests: write`, `issues: write` for benchmark commenting
- **Security Scanning**: `contents: read`, `security-events: write` for SARIF uploads
- **Build Jobs**: `contents: read`, `packages: write` for container registry access

### Permission Best Practices

- **Minimal permissions**: Each job only gets the permissions it needs
- **Explicit declarations**: All permissions explicitly declared for transparency
- **Security-first approach**: No blanket permissions that could compromise security

## Performance Optimizations

### Caching Strategy

- **Dependency caching**: Poetry and pip dependencies cached between runs
- **Docker layer caching**: GitHub Actions cache for Docker builds
- **Test result caching**: Optimized test execution with smart caching

### Parallel Execution

- **Matrix builds**: Parallel execution across multiple environments
- **Job parallelization**: Independent workflow jobs run concurrently
- **Resource optimization**: Efficient resource utilization across runners

## Deployment Integration

### Environment-Specific Deployments

- **Development**: Automatic deployment to staging environment
- **Production**: Manual approval required for production deployments
- **Feature branches**: Temporary preview deployments

### Deployment Validation

- **Health checks**: Automated post-deployment health verification
- **Smoke tests**: Basic functionality validation after deployment
- **Rollback capability**: Automated rollback on deployment failures

## Monitoring and Observability

### Pipeline Monitoring

- **Build status tracking**: Real-time build status monitoring
- **Performance metrics**: Build time and resource usage tracking
- **Failure analysis**: Automated failure detection and reporting

### Quality Metrics

- **Test coverage trends**: Historical test coverage tracking
- **Security posture**: Vulnerability trend analysis
- **Performance benchmarks**: Automated performance regression detection

## Troubleshooting

### Common Issues

#### Docker Build Failures

- **Manifest list errors**: Resolved with conditional platform builds
- **Attestation conflicts**: Fixed with conditional SBOM/provenance generation

#### Security Scan False Positives

- **Gitleaks configuration**: Custom rules prevent documentation false positives
- **Pattern exclusions**: Smart filtering for legitimate examples
- **Semgrep nosec handling**: Respects `# nosec` comments for intentional security exceptions

#### Test Failures

- **Environment isolation**: Each test run uses clean environment
- **Dependency management**: Consistent dependency versions across environments

#### Benchmark Action Issues

- **gh-pages branch dependency**: Replaced benchmark action with custom GitHub Script solution
- **Direct comment generation**: Custom script parses benchmark.json and creates detailed PR comments
- **No external dependencies**: Eliminates all GitHub Pages and branch-related issues
- **Robust error handling**: Built-in fallback for parsing errors or missing data

#### Semgrep Configuration Issues

- **Deprecated generateSarif parameter**: Updated to use modern semgrep/semgrep-action@v1
- **Unsupported auditOn parameter**: Removed invalid parameter to eliminate configuration warnings
- **SARIF output handling**: Automatic SARIF generation without deprecated parameters
- **Enhanced error handling**: Improved result processing and summary reporting

#### GitHub Script Permissions Issues

- **PR commenting permissions**: Added `pull-requests: write` and `issues: write` permissions
- **Resource access errors**: Resolved "Resource not accessible by integration" errors
- **Enhanced error handling**: Improved fallback mechanisms for API failures

#### Security Scanning Artifact Issues

- **Bandit report generation**: Added `--exit-zero` flag to ensure JSON reports are always created
- **Safety check artifacts**: Implemented fallback JSON generation when vulnerabilities are found
- **pip-audit report reliability**: Added fallback mechanisms for consistent artifact upload
- **Report file consistency**: All security tools now guarantee report file generation

### Debug Strategies

- **Verbose logging**: Enable detailed logging for troubleshooting
- **Matrix debugging**: Test specific combinations independently
- **Local reproduction**: Run CI commands locally for debugging

## Best Practices

### Local Development Validation

Before committing code, developers should run the complete validation suite:

```bash
# 1. Format code
poetry run ruff format app/ tests/

# 2. Check formatting compliance
poetry run ruff format --check app/ tests/

# 3. Run linting
poetry run ruff check app/ tests/

# 4. Type checking
poetry run mypy app/ --ignore-missing-imports

# 5. Security scanning
poetry run bandit -r app/ --exit-zero

# 6. Run tests
poetry run pytest
```

### Code Quality

- **Automated formatting**: Ruff formatting enforced in CI
- **Type checking**: MyPy validation for type safety
- **Security scanning**: Multiple security tools for comprehensive coverage

### Performance

- **Efficient caching**: Minimize redundant operations
- **Parallel execution**: Maximize concurrency where possible
- **Resource optimization**: Efficient use of GitHub Actions minutes

### Security

- **Secret management**: Secure handling of API keys and credentials
- **Vulnerability monitoring**: Continuous security posture assessment
- **Compliance validation**: Automated license and security compliance

## Future Enhancements

### Planned Improvements

- **Advanced deployment strategies**: Blue-green and canary deployments
- **Enhanced monitoring**: Integration with external monitoring systems
- **Performance optimization**: Further build time improvements
- **Security enhancements**: Additional security scanning tools
