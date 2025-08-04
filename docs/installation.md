# Installation Guide

## Prerequisites

- Python 3.11 or higher
- Poetry (recommended) or pip
- Redis (optional, for caching)
- PostgreSQL (optional, for persistent storage)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/iamapsrajput/ModelMuxer.git
cd ModelMuxer
```

### 2. Install Dependencies

#### Using Poetry (Recommended)

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

#### Using pip (Alternative)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies using Poetry
poetry install --with dev,ml
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit the `.env` file with your configuration:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Database (Optional)
DATABASE_URL=postgresql://user:password@localhost:5432/modelmuxer

# Redis (Optional)
REDIS_URL=redis://localhost:6379/0

# Authentication
JWT_SECRET_KEY=your_jwt_secret_key_here
JWT_ALGORITHM=RS256

# Routing Configuration
DEFAULT_ROUTING_STRATEGY=hybrid
CASCADE_ROUTING_ENABLED=true
SEMANTIC_ROUTING_ENABLED=true
HEURISTIC_ROUTING_ENABLED=true
```

### 4. Run the Application

#### Development Mode

```bash
# Using Poetry
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Using pip
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Production Mode

```bash
# Using Poetry
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Using pip
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Verify Installation

Visit `http://localhost:8000/docs` to access the interactive API documentation.

Test the health endpoint:

```bash
curl http://localhost:8000/health
```

## Docker Installation

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/iamapsrajput/ModelMuxer.git
cd ModelMuxer

# Copy environment file
cp .env.example .env
# Edit .env with your configuration

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

### Using Docker

```bash
# Build the image
docker build -t modelmuxer .

# Run the container
docker run -d \
  --name modelmuxer \
  -p 8000:8000 \
  --env-file .env \
  modelmuxer
```

## Kubernetes Installation

### Using Helm (Recommended)

```bash
# Add the ModelMuxer Helm repository
helm repo add modelmuxer https://charts.modelmuxer.com
helm repo update

# Install ModelMuxer
helm install modelmuxer modelmuxer/modelmuxer \
  --set config.openai.apiKey=your_openai_api_key \
  --set config.anthropic.apiKey=your_anthropic_api_key
```

### Using kubectl

```bash
# Apply the Kubernetes manifests
kubectl apply -f k8s/

# Check the deployment
kubectl get pods -l app=modelmuxer
```

## Development Setup

### 1. Install Development Dependencies

```bash
poetry install --with dev
```

### 2. Install Pre-commit Hooks

```bash
pre-commit install
```

### 3. Run Tests

```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test file
poetry run pytest tests/test_router.py -v
```

### 4. Code Quality Checks

```bash
# Format code
poetry run black app/ tests/

# Sort imports
poetry run isort app/ tests/

# Type checking
poetry run mypy app/

# Linting
poetry run flake8 app/ tests/
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

If you encounter import errors, ensure you're in the correct virtual environment:

```bash
# Using Poetry
poetry shell

# Using pip
source venv/bin/activate
```

#### 2. API Key Issues

Ensure your API keys are correctly set in the `.env` file and have sufficient permissions.

#### 3. Port Already in Use

If port 8000 is already in use, change the port in your `.env` file:

```env
PORT=8001
```

#### 4. Database Connection Issues

If using PostgreSQL, ensure the database is running and accessible:

```bash
# Check PostgreSQL status
pg_ctl status

# Create database
createdb modelmuxer
```

#### 5. Redis Connection Issues

If using Redis for caching, ensure Redis is running:

```bash
# Start Redis
redis-server

# Check Redis status
redis-cli ping
```

### Getting Help

- Check the [FAQ](../faq.md)
- Review the [troubleshooting guide](../troubleshooting.md)
- Open an issue on [GitHub](https://github.com/iamapsrajput/ModelMuxer/issues)
- Join our [Discord community](https://discord.gg/modelmuxer)

## Next Steps

- [Configuration Guide](configuration.md)
- [API Reference](api-reference.md)
- [Deployment Guide](deployment.md)
- [Contributing Guide](contributing.md)
