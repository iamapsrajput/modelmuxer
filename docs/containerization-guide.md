# ModelMuxer Containerization Guide

This guide covers containerization options for ModelMuxer, including Apple's native containerization and traditional Docker support.

## Overview

ModelMuxer supports multiple containerization approaches:

1. **Apple Container** (macOS 15+ Beta) - Apple's native containerization
2. **Docker** - Traditional containerization (cross-platform compatibility)

## Apple Container (Native macOS Containerization)

### Prerequisites

- macOS 15 Beta or later
- Apple Silicon Mac (M1/M2/M3)
- Xcode Command Line Tools

### Installation

Apple Container is built into macOS 15 Beta. No additional installation required.

### Basic Commands

```bash
# Build image
container build -t modelmuxer:latest .

# Run container
container run -d --name modelmuxer -p 8000:8000 modelmuxer:latest

# List containers
container ps

# View logs
container logs modelmuxer

# Stop container
container stop modelmuxer

# Remove container
container rm modelmuxer
```

### Apple Container Compose

Apple Container supports a compose-like syntax with `container-compose.yaml`:

```yaml
version: "1.0"
services:
  modelmuxer:
    build: .
    ports:
      - "8000:8000"
    environment:
      - TESTING=false
      - DATABASE_URL=sqlite:///data/modelmuxer.db
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
```

Run with:

```bash
container compose up -d
```

## Docker (Cross-Platform)

### Installation

- **macOS**: Docker Desktop or OrbStack
- **Linux**: Docker Engine
- **Windows**: Docker Desktop

### Usage

```bash
# Build and run
docker build -t modelmuxer:latest .
docker run -d --name modelmuxer -p 8000:8000 modelmuxer:latest

# Using compose
docker-compose up -d
```

## Feature Comparison

| Feature                 | Apple Container | Docker |
| ----------------------- | --------------- | ------ |
| Native macOS            | ✅              | ❌     |
| Apple Silicon Optimized | ✅              | ✅     |
| Compose Support         | ✅ (Limited)    | ✅     |
| Cross-Platform          | ❌              | ✅     |
| Rootless                | ✅              | ❌     |
| OCI Compatible          | ✅              | ✅     |

## Recommendations

- **macOS 15+ Beta**: Use Apple Container for best native performance
- **Cross-Platform Development**: Use Docker for consistency
- **CI/CD**: Use Docker for maximum compatibility

## Troubleshooting

### Apple Container Issues

1. **Command not found**: Ensure macOS 15 Beta is installed
2. **Permission denied**: Apple Container runs rootless by default
3. **Build failures**: Check Dockerfile compatibility with Apple's runtime

### Docker Issues

1. **Docker daemon not running**: Start Docker Desktop
2. **Permission denied**: Add user to docker group (Linux)
3. **Resource limits**: Increase memory/CPU in Docker settings

## Performance Optimization

### Apple Container

- Uses native macOS virtualization framework
- Optimized for Apple Silicon
- Lower memory overhead than Docker Desktop

### Docker

- Mature ecosystem
- Extensive tooling support
- Consistent across platforms
