# Claude Integration with ModelMuxer

This guide covers how to integrate various Claude-based tools and CLIs with ModelMuxer for intelligent routing, cost optimization, and enhanced features.

## Overview

ModelMuxer provides seamless integration with Claude-based tools by supporting:

- **Anthropic Messages API compatibility** - Full support for `/v1/messages` endpoint
- **Claude CLI integration** - Official Anthropic CLI routing through ModelMuxer
- **Claude Dev tools** - VS Code extensions and development tools
- **Intelligent routing** - Automatic model selection based on request complexity
- **Cost management** - Real-time budget tracking and optimization
- **Streaming responses** - Live token-by-token output

## Quick Setup

### 1. Environment Configuration

Configure your Claude tools to use ModelMuxer:

```bash
# For Claude CLI and tools that use Anthropic API format
export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_API_KEY="sk-test-claude-dev"

# Alternative environment variable names (tool-dependent)
export ANTHROPIC_API_URL="http://localhost:8000" 
export ANTHROPIC_AUTH_TOKEN="sk-test-claude-dev"
```

**Important**: Use `http://localhost:8000` (NOT `/v1`) as most Claude tools add the `/v1/messages` path automatically.

### 2. Verify ModelMuxer is Running

```bash
# Check health status
curl http://localhost:8000/health

# Verify API key works
curl -H "Authorization: Bearer sk-test-claude-dev" \
  http://localhost:8000/v1/models
```

## Supported Integrations

### 1. Official Claude CLI

The official Anthropic Claude CLI can be configured to route through ModelMuxer:

```powershell
# PowerShell configuration
$env:ANTHROPIC_BASE_URL = "http://localhost:8000"
$env:ANTHROPIC_API_KEY = "sk-test-claude-dev"

# Test the integration
claude "Write a Python function to reverse a string"
```

**Benefits:**
- Official Claude CLI experience with intelligent routing
- Automatic model selection based on task complexity
- Cost tracking and budget management
- Access to multiple models through single interface

### 2. VS Code Claude Dev Extensions

Configure Claude Dev extensions in VS Code:

```json
// VS Code settings.json
{
  "claude-dev.apiProvider": "anthropic",
  "claude-dev.anthropicBaseUrl": "http://localhost:8000",
  "claude-dev.anthropicApiKey": "sk-test-claude-dev",
  "claude-dev.enableStreaming": true,
  "claude-dev.maxTokens": 4000
}
```

**Alternative configuration names:**
```json
{
  "claude-dev.provider": "anthropic",
  "claude-dev.baseUrl": "http://localhost:8000",
  "claude-dev.apiKey": "sk-test-claude-dev",
  "claude-dev.streaming": true
}
```

### 3. Custom Claude CLI (Included)

ModelMuxer includes a custom Claude CLI in `scripts/claude-cli.py`:

```bash
# Interactive mode
python scripts/claude-cli.py

# Single command
python scripts/claude-cli.py -m "Explain Python decorators"

# With specific model
python scripts/claude-cli.py -m "Debug this code" --model claude-3-5-sonnet-20241022
```

## Intelligent Routing Examples

ModelMuxer automatically analyzes requests and routes to optimal models:

### Code Tasks → Claude Sonnet
```bash
# Request: "Write a Python function to calculate fibonacci"
# Routes to: claude-3-5-sonnet-latest
# Reason: Code detected (confidence: 0.85), optimal for programming
```

### Simple Questions → Cost-Effective Models
```bash
# Request: "What is the capital of France?"
# Routes to: gpt-3.5-turbo or claude-haiku
# Reason: Simple query (32 chars), cost-optimized selection
```

### Complex Analysis → Premium Models
```bash
# Request: "Analyze the performance implications of microservices..."
# Routes to: claude-3-5-sonnet-latest or gpt-4
# Reason: Complex analysis required (confidence: 0.90)
```

## API Endpoints

ModelMuxer supports these Claude-compatible endpoints:

### Messages API
```bash
# Primary endpoint
POST http://localhost:8000/v1/messages

# Alternative endpoint
POST http://localhost:8000/messages
```

### Request Format
```json
{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1000,
    "messages": [
        {
            "role": "user",
            "content": "Hello! How can you help me with code?"
        }
    ],
    "system": "You are a helpful coding assistant.",
    "stream": true
}
```

### Response Format
```json
{
    "id": "msg_abc123",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "I'd be happy to help you with coding tasks..."
        }
    ],
    "model": "gpt-4",
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 15,
        "output_tokens": 25
    }
}
```

## Budget Management

Set up cost controls for Claude usage:

```bash
# Set daily budget
curl -X POST "http://localhost:8000/v1/analytics/budgets" \
  -H "Authorization: Bearer sk-test-claude-dev" \
  -H "Content-Type: application/json" \
  -d '{
    "budget_type": "daily",
    "budget_limit": 5.0,
    "alert_thresholds": [50, 80, 95]
  }'

# Check budget status
curl -H "Authorization: Bearer sk-test-claude-dev" \
  "http://localhost:8000/v1/analytics/budgets"

# View cost analytics
curl -H "Authorization: Bearer sk-test-claude-dev" \
  "http://localhost:8000/v1/analytics/costs?days=7"
```

## Streaming Setup

Enable real-time streaming responses:

### For Claude CLI Tools
Most Claude tools support streaming automatically when configured with ModelMuxer.

### For Custom Applications
```python
import httpx

async def stream_claude_response():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/messages",
            headers={"Authorization": "Bearer sk-test-claude-dev"},
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Explain async programming"}],
                "stream": true,
                "max_tokens": 1000
            }
        )
        
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                # Process streaming data
                print(line[6:])
```

## Configuration Scripts

ModelMuxer includes helpful configuration scripts:

### PowerShell Configuration Scripts
- `scripts/configure_claude_cli.ps1` - Configure Claude CLI for current user
- `scripts/activate_claude_modelmuxer.ps1` - Activate for current session
- `scripts/test_claude_integration.ps1` - Test the integration

### Usage Examples
```powershell
# Make permanent for user profile
.\scripts\configure_claude_cli.ps1 -UserProfile

# Activate for current session
. .\scripts\activate_claude_modelmuxer.ps1

# Test the setup
.\scripts\test_claude_integration.ps1
```

## Benefits of Integration

### Cost Optimization
- **Automatic model selection** - Cheap models for simple tasks, premium for complex
- **Budget controls** - Daily/monthly limits with alerts
- **Usage analytics** - Detailed cost breakdown and trends
- **60-80% cost savings** vs always using premium models

### Enhanced Capabilities  
- **Multi-provider access** - Claude, GPT-4, Gemini, Mistral through single interface
- **Intelligent fallbacks** - Automatic failover if primary model unavailable
- **Advanced caching** - Redis-backed response caching
- **Request logging** - Complete audit trails

### Enterprise Features
- **PII protection** - Automatic detection and redaction
- **Rate limiting** - Configurable per-user/API key limits
- **Security headers** - Enhanced request validation
- **Monitoring** - Prometheus metrics and health checks

## Troubleshooting

### Common Issues

#### Connection Refused

- Verify ModelMuxer is running: `curl http://localhost:8000/health`
- Check port availability: `netstat -ano | findstr :8000`

#### Authentication Failed

- Verify API key: Check `.env` file has `API_KEYS=sk-test-claude-dev`
- Check Authorization header: `Bearer sk-test-claude-dev`

#### No Streaming

- Enable streaming in tool settings: `"stream": true`
- Check logs: `docker logs modelmuxer-app -f`

#### Budget Exceeded

- Check current usage: `curl -H "Authorization: Bearer sk-test-claude-dev" http://localhost:8000/v1/analytics/budgets`
- Increase limits or wait for reset

### Debug Mode
```bash
# Enable debug logging
export MODELMUXER_DEBUG=true

# View detailed logs
docker logs modelmuxer-app --tail 50 -f
```

## Example Workflows

### Development Workflow
1. Start ModelMuxer in enhanced mode
2. Configure Claude Dev in VS Code
3. Use intelligent routing for code tasks
4. Monitor costs via budget dashboard

### Team Workflow
1. Set up individual API keys per team member
2. Configure team budgets and alerts
3. Monitor usage patterns and optimize
4. Use analytics for cost optimization

This integration provides the official Claude experience enhanced with intelligent routing, cost management, and enterprise features!
