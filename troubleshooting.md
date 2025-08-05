# Troubleshooting Guide

## Common Issues

### Installation Problems

#### Poetry Installation Issues
```bash
# If Poetry is not found, install it:
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
export PATH="$HOME/.local/bin:$PATH"
```

#### Dependency Conflicts
```bash
# Clear Poetry cache and reinstall
poetry cache clear --all .
poetry install --with dev
```

### Configuration Issues

#### Missing API Keys
**Error**: `Configuration error: Missing required API key`

**Solution**:
1. Copy `.env.example` to `.env`
2. Add your API keys to the `.env` file
3. Ensure at least one provider API key is configured

#### Database Connection Issues
**Error**: `Database connection failed`

**Solution**:
```bash
# For SQLite (default)
touch router_data.db

# For PostgreSQL, verify connection string in .env
DATABASE_URL="postgresql://user:pass@localhost/dbname"
```

### Runtime Issues

#### Model Not Found
**Error**: `Model 'gpt-4' not available`

**Solution**: Check that the model name is correct and supported by the provider.

#### Rate Limiting
**Error**: `Rate limit exceeded`

**Solution**:
- Check your provider API quotas
- Implement retry logic with exponential backoff
- Consider upgrading your provider plan

#### High Latency
**Symptoms**: Slow response times

**Solutions**:
- Enable caching in configuration
- Choose geographically closer provider regions
- Use faster models for simple queries

### Provider-Specific Issues

#### OpenAI
- Verify API key format: `sk-...`
- Check organization ID if using team accounts
- Ensure sufficient credits in your account

#### Anthropic
- Verify API key format: `sk-ant-...`
- Check model availability in your region
- Review usage limits

#### Mistral
- Verify API key format
- Check model availability
- Review rate limits

## Logging and Debugging

### Enable Debug Logging
```bash
# Set environment variable
export DEBUG=true

# Or in .env file
DEBUG=true
```

### Check Application Logs
```bash
# View recent logs
tail -f logs/application.log

# Search for errors
grep -i error logs/application.log
```

### API Response Debugging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your ModelMuxer code here
```

## Performance Issues

### Memory Usage
- Monitor memory consumption with large requests
- Implement request size limits
- Use streaming for large responses

### Response Time
- Enable response caching
- Use appropriate routing strategies
- Monitor provider latency

## Getting Help

If issues persist:

1. **Check the logs** for detailed error messages
2. **Search existing issues** on GitHub
3. **Create a new issue** with:
   - Error messages
   - Configuration details (without API keys)
   - Steps to reproduce
   - Environment information

## Contact

- GitHub Issues: [Create an issue](https://github.com/iamapsrajput/ModelMuxer/issues)
- Discussions: [Join the discussion](https://github.com/iamapsrajput/ModelMuxer/discussions)
