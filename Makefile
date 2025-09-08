# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

.PHONY: help install install-dev test test-unit test-integration lint format clean run docker-build docker-run license-check license-headers license-compliance security-scan validate-direct-providers test-direct validate-architecture benchmark-direct

# Default target
help:
	@echo "ModelMuxer Development Commands"
	@echo "==============================="
	@echo ""
	@echo "Development:"
	@echo "  install         - Install production dependencies"
	@echo "  install-dev     - Install development dependencies"
	@echo "  test            - Run all tests"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  validate-direct-providers - Run comprehensive direct provider validation"
	@echo "  test-direct     - Run direct provider tests only"
	@echo "  validate-architecture - Validate architecture compliance"
	@echo "  benchmark-direct - Run performance benchmarks"
	@echo "  lint            - Run linting and type checking"
	@echo "  format          - Format code with black and isort"
	@echo "  security-scan   - Run security scans"
	@echo "  clean           - Clean cache files and artifacts"
	@echo "  run             - Run development server"
	@echo "  docker-build    - Build Docker image"
	@echo "  docker-run      - Run Docker container"
	@echo ""
	@echo "License Management:"
	@echo "  license-check      - Run license compliance check"
	@echo "  license-headers    - Add license headers to all source files"
	@echo "  license-compliance - Full license compliance validation"
	@echo ""

# Run license compliance check
license-check:
	@echo "Running license compliance check..."
	python scripts/check_license_compliance.py

# Add license headers to all source files
license-headers:
	@echo "Adding license headers to source files..."
	python scripts/add_license_headers.py

# Full license compliance validation
license-compliance: license-headers license-check
	@echo "✅ License compliance validation complete"

# Install production dependencies
install:
	poetry install --only=main

# Install development dependencies
install-dev:
	poetry install --with dev

# Run all tests
test:
	DEBUG=false poetry run pytest tests/ -v

# Run unit tests only
test-unit:
	DEBUG=false poetry run pytest tests/ -v -m "not integration and not performance"

# Run integration tests only
test-integration:
	DEBUG=false poetry run pytest tests/ -v -m "integration"

# Comprehensive direct provider validation
validate-direct-providers:
	@echo "🔍 Running comprehensive direct provider validation..."
	poetry run python scripts/validate_direct_provider_architecture.py
	poetry run pytest tests/test_comprehensive_direct_provider_validation.py -v
	@echo "✅ Direct provider validation complete"

# Quick direct provider tests
test-direct:
	@echo "🧪 Running direct provider tests..."
	poetry run pytest tests/direct/ -v -m direct
	@echo "✅ Direct provider tests complete"

# Architecture validation only
validate-architecture:
	@echo "🏗️ Validating architecture..."
	poetry run pytest tests/test_comprehensive_direct_provider_validation.py::TestArchitectureValidation -v
	@echo "✅ Architecture validation complete"

# Performance benchmarking
benchmark-direct:
	@echo "⚡ Running performance benchmarks..."
	poetry run pytest tests/test_comprehensive_direct_provider_validation.py::TestPerformanceAndMetrics -v
	@echo "✅ Performance benchmarks complete"

# Run linting and type checking
lint:
	poetry run ruff check .
	poetry run black --check .

typecheck:
	poetry run mypy app/

test-quick:
	DEBUG=false poetry run pytest -q

test-cov:
	DEBUG=false poetry run pytest --cov=app --cov-report=term-missing --cov-report=xml --cov-report=html --cov-fail-under=70

security:
	poetry run bandit -q -r app || true
	poetry run semgrep --error --config p/python || true
	poetry run trivy fs --exit-code 1 --severity HIGH,CRITICAL . || true

# Format code
format:
	poetry run black app/ tests/
	poetry run isort app/ tests/

# Run security scans
security-scan:
	poetry run bandit -r app/
	poetry run safety check

# Run development server
run:
	poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Build Docker image
docker-build:
	docker build -f infra/docker/Dockerfile.production -t modelmuxer:latest .

# Run Docker container
docker-run:
	docker run -p 8000:8000 --env-file .env modelmuxer:latest

# Clean up generated files and cache
clean:
	@echo "Cleaning up compliance reports and cache..."
	@rm -f compliance_report.json
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	@echo "✅ Cleanup complete"
