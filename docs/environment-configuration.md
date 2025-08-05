# Environment Configuration Guide

This guide explains the relationship between different environment files in the ModelMuxer project and how to properly configure them.

## Environment Files Overview

### 1. `.env` (Primary Configuration)

**Purpose**: Main environment configuration for local development and production
**Usage**: Contains all core application settings including API keys, database URLs, and feature flags

### 2. `.env.security` (Security Overrides)

**Purpose**: Security-focused configuration overrides
**Usage**: Contains security-specific settings that override defaults for production environments
**Note**: This file supplements `.env`, it doesn't replace it

### 3. `.env.secure` (Minimal Security Profile)

**Purpose**: Minimal configuration for security-conscious deployments
**Usage**: Disables ML dependencies and uses simpler routing strategies

### 4. `.env.example` (Template)

**Purpose**: Template file showing all available configuration options
**Usage**: Copy to `.env` and customize with your actual values

## Configuration Priority

1. Environment variables (highest priority)
2. `.env` file
3. `.env.security` (if loaded)
4. Default values in code (lowest priority)

## Required Environment Variables

### Core Application

```bash
# Required for JWT authentication
JWT_SECRET_KEY=your-32-character-secret-key

# At least one provider API key is required
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
MISTRAL_API_KEY=your-mistral-key
```

### Optional but Recommended

```bash
# Database (defaults to SQLite)
DATABASE_URL=sqlite:///./modelmuxer.db

# Server configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Authentication
AUTH_ENABLED=true
API_KEYS=sk-test-key-1,sk-test-key-2
```

## Common Issues and Solutions

### Issue 1: "No providers initialized"

**Cause**: API keys are missing or contain placeholder text
**Solution**:

- Check that API keys don't contain "your-" or "-here"
- Ensure keys are valid and not expired
- Verify environment variables are loaded correctly

### Issue 2: JWT Secret validation errors

**Cause**: JWT_SECRET_KEY is missing or too short
**Solution**:

```bash
# Generate a secure JWT secret
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Issue 3: Configuration conflicts

**Cause**: Multiple environment files with conflicting values
**Solution**: Use only `.env` for local development, add security overrides only when needed

## Best Practices

1. **Use `.env` for all local development**
2. **Never commit real API keys to version control**
3. **Use environment variables in production, not files**
4. **Generate strong JWT secrets (32+ characters)**
5. **Validate configuration on startup**

## Podman/Container Configuration

When using containers, environment variables can be passed via:

- `env_file` directive in compose files
- Individual `environment` entries
- Host environment variables

Example:

```yaml
services:
  modelmuxer:
    env_file: .env
    environment:
      - TESTING=false
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
```
