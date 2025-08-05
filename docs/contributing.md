# Contributing Guide

Thank you for your interest in contributing to ModelMuxer! This guide will help you get started with contributing to the project.

## Code of Conduct

By participating in this project, you agree to maintain professional and respectful behavior in all interactions.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Poetry for dependency management
- Git for version control
- Docker (optional, for testing)

### Development Setup

1. **Fork and Clone**

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/modelmuxer.git
cd ModelMuxer

# Add upstream remote
git remote add upstream https://github.com/iamapsrajput/ModelMuxer.git
```

2. **Install Dependencies**

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies (including ML dependencies for full development)
poetry install --with dev,ml

# Activate virtual environment
poetry shell
```

3. **Environment Setup**

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys (at least one provider required for testing)
```

4. **Install Pre-commit Hooks**

```bash
pre-commit install
```

5. **Verify Setup**

```bash
# Run tests
poetry run pytest

# Start development server
poetry run uvicorn app.main:app --reload
```

## Development Workflow

### Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical production fixes

### Making Changes

1. **Create a Branch**

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

2. **Make Your Changes**

- Write clean, readable code
- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation as needed

3. **Test Your Changes**

```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific tests
poetry run pytest tests/test_router.py -v

# Run linting
poetry run flake8 app/ tests/
poetry run mypy app/

# Format code
poetry run black app/ tests/
poetry run isort app/ tests/
```

4. **Commit Your Changes**

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add cascade routing strategy

- Implement FrugalGPT-inspired cascade routing
- Add quality threshold configuration
- Include cost optimization logic
- Add comprehensive tests

Closes #123"
```

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(router): add semantic routing strategy
fix(auth): resolve JWT token validation issue
docs: update API documentation
test(cache): add Redis cache integration tests
```

## Pull Request Process

### Before Submitting

1. **Ensure Quality**

```bash
# Run full test suite
poetry run pytest

# Check code coverage (aim for >80%)
poetry run pytest --cov=app --cov-report=term

# Lint and format
poetry run flake8 app/ tests/
poetry run black app/ tests/
poetry run isort app/ tests/
poetry run mypy app/
```

2. **Update Documentation**

- Update relevant documentation files
- Add docstrings to new functions/classes
- Update API documentation if needed

3. **Test Edge Cases**

- Test with different provider configurations
- Test error handling scenarios
- Test with various input types

### Submitting the PR

1. **Push Your Branch**

```bash
git push origin feature/your-feature-name
```

2. **Create Pull Request**

- Go to GitHub and create a pull request
- Use the PR template
- Provide clear description of changes
- Link related issues

3. **PR Template**

```markdown
## Description

Brief description of changes made.

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process

1. **Automated Checks**

   - All tests must pass
   - Code coverage must meet threshold
   - Linting checks must pass

2. **Code Review**

   - At least one maintainer review required
   - Address all feedback
   - Update PR as needed

3. **Merge**
   - Squash and merge for feature branches
   - Merge commit for release branches

## Contribution Areas

### High Priority

- **New Routing Strategies**: Implement additional intelligent routing algorithms
- **Provider Integrations**: Add support for new LLM providers
- **Performance Optimizations**: Improve latency and throughput
- **Security Enhancements**: Strengthen authentication and authorization

### Medium Priority

- **Monitoring & Observability**: Enhanced metrics and dashboards
- **Documentation**: Improve guides and examples
- **Testing**: Increase test coverage and add integration tests
- **Developer Experience**: Better tooling and debugging

### Good First Issues

Look for issues labeled `good-first-issue` or `help-wanted`:

- Documentation improvements
- Small bug fixes
- Test additions
- Code cleanup

## Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use type hints for all functions
- Write descriptive variable and function names
- Keep functions small and focused

### Testing

- Write tests for all new functionality
- Aim for >80% code coverage
- Use pytest for testing framework
- Mock external dependencies

### Documentation

- Add docstrings to all public functions/classes
- Use Google-style docstrings
- Update relevant documentation files
- Include examples in docstrings

### Performance

- Consider performance implications
- Use async/await for I/O operations
- Implement proper caching strategies
- Profile code for bottlenecks

## Architecture Guidelines

### Adding New Providers

1. Create provider class in `app/providers/`
2. Inherit from `BaseProvider`
3. Implement required methods
4. Add provider configuration
5. Add comprehensive tests
6. Update documentation

### Adding New Routing Strategies

1. Create router class in `app/routing/`
2. Inherit from `BaseRouter`
3. Implement routing logic
4. Add configuration options
5. Add performance tests
6. Update documentation

### Database Changes

1. Create migration scripts
2. Update models in `app/models.py`
3. Test migration up and down
4. Document schema changes

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord**: Real-time chat with maintainers
- **Email**: <security@modelmuxer.com> for security issues

### Resources

- [Architecture Documentation](architecture.md)
- [API Reference](api-reference.md)
- [Development Setup](installation.md)
- [Testing Guide](testing.md)

## Recognition

Contributors are recognized in:

- `CONTRIBUTORS.md` file
- Release notes
- Project documentation
- Annual contributor highlights

## License

By contributing to ModelMuxer, you agree that your contributions will be licensed under the same license as the project (Business Source License 1.1, converting to Apache 2.0).

## Questions?

Don't hesitate to ask questions! We're here to help:

- Open a GitHub Discussion
- Join our Discord server
- Email the maintainers

Thank you for contributing to ModelMuxer! 🚀
