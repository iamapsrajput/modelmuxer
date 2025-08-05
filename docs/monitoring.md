# Monitoring and Observability

## Overview

ModelMuxer provides comprehensive monitoring capabilities to track performance, usage, and system health.

## Metrics Collection

### Built-in Metrics

ModelMuxer automatically collects:

- Request count and rate
- Response times and latency percentiles
- Error rates by provider and model
- Token usage and costs
- Cache hit/miss rates

### Prometheus Integration

```python
# Enable Prometheus metrics
ENABLE_METRICS = true
METRICS_PORT = 9090
```

Access metrics at: `http://localhost:9090/metrics`

### Custom Metrics

```python
from app.monitoring import metrics

# Custom counter
metrics.request_counter.inc({"provider": "openai", "model": "gpt-4"})

# Custom histogram
metrics.response_time.observe(0.5, {"endpoint": "/chat/completions"})
```

## Health Checks

### Basic Health Check

```
GET /health
```

Returns service status and version information.

## Logging

### Log Levels

- `DEBUG`: Detailed debugging information
- `INFO`: General operational messages
- `WARNING`: Important events that may need attention
- `ERROR`: Error conditions
- `CRITICAL`: Critical system failures

### Log Configuration

```bash
# Environment variables
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/modelmuxer/app.log
```

### Structured Logging

```python
import structlog

logger = structlog.get_logger()
logger.info("Request processed",
           user_id="user123",
           model="gpt-4",
           tokens=150)
```

## Alerting

### Threshold-based Alerts

- High error rates (>5%)
- Slow response times (>2s p95)
- Budget exceeded warnings
- Provider API failures

### Integration Options

- Email notifications
- Slack webhooks
- PagerDuty integration
- Custom webhook endpoints

## Dashboards

### Grafana Integration

Import the provided dashboard templates:

- System overview
- Provider performance
- Cost analysis
- User activity

### Key Performance Indicators (KPIs)

- Requests per second
- Average response time
- Error rate percentage
- Cost per request
- Provider uptime

## Performance Monitoring

### Response Time Tracking

- P50, P95, P99 latency percentiles
- Provider-specific response times
- Geographic latency distribution

### Resource Utilization

- CPU usage
- Memory consumption
- Network I/O
- Disk space

### Capacity Planning

- Traffic growth trends
- Resource utilization forecasts
- Scaling recommendations

## Cost Monitoring

### Usage Tracking

- Token consumption by user/organization
- Cost breakdown by provider and model
- Budget utilization alerts

### Cost Optimization

- Identify expensive queries
- Monitor routing efficiency
- Track cost savings from intelligent routing

## Security Monitoring

### Audit Logging

- Authentication events
- API key usage
- Configuration changes
- Suspicious activity patterns

### Security Metrics

- Failed authentication attempts
- Rate limiting triggers
- Unusual access patterns

## Troubleshooting

### Common Monitoring Issues

1. **Missing metrics**: Check Prometheus configuration
2. **High memory usage**: Review log retention settings
3. **Alert fatigue**: Adjust threshold values

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
DEBUG=true poetry run uvicorn app.main:app
```

## Best Practices

1. **Set up alerts early** to catch issues before they impact users
2. **Monitor key business metrics** like cost per request and user satisfaction
3. **Use dashboards** for real-time visibility into system health
4. **Regular review** of metrics and thresholds
5. **Document runbooks** for common alert scenarios

## Integration Examples

### Datadog

```python
from datadog import initialize, statsd

initialize(api_key='your-api-key')
statsd.increment('modelmuxer.requests', tags=['provider:openai'])
```

### New Relic

```python
import newrelic.agent

@newrelic.agent.function_trace()
def process_request():
    # Your code here
    pass
```

## Next Steps

- Set up monitoring infrastructure
- Configure alerts for critical metrics
- Create custom dashboards
- Implement automated scaling based on metrics
