# SDK Documentation

## Overview

ModelMuxer provides official SDKs for popular programming languages to make integration seamless and developer-friendly.

## Python SDK

### Installation
```bash
pip install modelmuxer-python
```

### Quick Start
```python
from modelmuxer import ModelMuxer

# Initialize client
client = ModelMuxer(
    api_key="your_api_key",
    base_url="https://api.modelmuxer.com/api/v1"
)

# Simple chat completion
response = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    routing_strategy="hybrid"
)

print(response.choices[0].message.content)
```

### Advanced Usage
```python
# Streaming responses
stream = client.chat.completions.create(
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
    routing_strategy="cascade",
    budget_constraint=0.01
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Custom routing parameters
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Complex reasoning task"}],
    routing_strategy="quality_first",
    quality_threshold=0.9,
    max_tokens=2000,
    temperature=0.3
)

# Budget tracking
usage = client.usage.get_current()
print(f"Total cost this month: ${usage.monthly_cost}")
```

### Configuration
```python
# Advanced client configuration
client = ModelMuxer(
    api_key="your_api_key",
    base_url="https://api.modelmuxer.com/api/v1",
    timeout=30,
    max_retries=3,
    default_routing_strategy="hybrid"
)

# Environment-based configuration
import os
client = ModelMuxer.from_env()  # Uses MODELMUXER_API_KEY
```

## JavaScript/TypeScript SDK

### Installation
```bash
npm install modelmuxer-js
# or
yarn add modelmuxer-js
```

### Quick Start
```javascript
import { ModelMuxer } from "modelmuxer-js";

const client = new ModelMuxer({
  apiKey: "your_api_key",
  baseURL: "https://api.modelmuxer.com/api/v1",
});

// Chat completion
const response = await client.chat.completions.create({
  messages: [{ role: "user", content: "Hello, world!" }],
  routing_strategy: "hybrid",
});

console.log(response.choices[0].message.content);
```

### TypeScript Support
```typescript
import { ModelMuxer, ChatCompletionRequest } from "modelmuxer-js";

const client = new ModelMuxer({
  apiKey: process.env.MODELMUXER_API_KEY!,
});

const request: ChatCompletionRequest = {
  messages: [{ role: "user", content: "Explain TypeScript" }],
  routing_strategy: "quality_first",
  max_tokens: 1000,
};

const response = await client.chat.completions.create(request);
```

### Streaming
```javascript
// Streaming responses
const stream = await client.chat.completions.create({
  messages: [{ role: "user", content: "Write a poem" }],
  stream: true,
});

for await (const chunk of stream) {
  if (chunk.choices[0]?.delta?.content) {
    process.stdout.write(chunk.choices[0].delta.content);
  }
}
```

## Go SDK

### Installation
```bash
go get github.com/modelmuxer/modelmuxer-go
```

### Quick Start
```go
package main

import (
    "context"
    "fmt"
    "log"
    
    "github.com/modelmuxer/modelmuxer-go"
)

func main() {
    client := modelmuxer.NewClient("your_api_key")
    
    resp, err := client.Chat.Completions.Create(context.Background(), &modelmuxer.ChatCompletionRequest{
        Messages: []modelmuxer.Message{
            {Role: "user", Content: "Hello from Go!"},
        },
        RoutingStrategy: "hybrid",
    })
    
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println(resp.Choices[0].Message.Content)
}
```

### Advanced Usage
```go
// Custom configuration
config := &modelmuxer.Config{
    APIKey:  "your_api_key",
    BaseURL: "https://api.modelmuxer.com/api/v1",
    Timeout: 30 * time.Second,
}
client := modelmuxer.NewClientWithConfig(config)

// Streaming
stream, err := client.Chat.Completions.CreateStream(ctx, &modelmuxer.ChatCompletionRequest{
    Messages: []modelmuxer.Message{
        {Role: "user", Content: "Stream this response"},
    },
    Stream: true,
})

for {
    chunk, err := stream.Recv()
    if err == io.EOF {
        break
    }
    if err != nil {
        log.Fatal(err)
    }
    
    if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
        fmt.Print(chunk.Choices[0].Delta.Content)
    }
}
```

## Java SDK

### Installation
```xml
<dependency>
    <groupId>com.modelmuxer</groupId>
    <artifactId>modelmuxer-java</artifactId>
    <version>1.0.0</version>
</dependency>
```

### Quick Start
```java
import com.modelmuxer.ModelMuxer;
import com.modelmuxer.model.ChatCompletionRequest;
import com.modelmuxer.model.Message;

public class Example {
    public static void main(String[] args) {
        ModelMuxer client = ModelMuxer.builder()
            .apiKey("your_api_key")
            .build();
        
        ChatCompletionRequest request = ChatCompletionRequest.builder()
            .addMessage(Message.user("Hello from Java!"))
            .routingStrategy("hybrid")
            .build();
        
        var response = client.chat().completions().create(request);
        System.out.println(response.getChoices().get(0).getMessage().getContent());
    }
}
```

## C# SDK

### Installation
```bash
dotnet add package ModelMuxer.Net
```

### Quick Start
```csharp
using ModelMuxer;

var client = new ModelMuxerClient("your_api_key");

var response = await client.Chat.Completions.CreateAsync(new ChatCompletionRequest
{
    Messages = new[] 
    {
        new Message { Role = "user", Content = "Hello from C#!" }
    },
    RoutingStrategy = "hybrid"
});

Console.WriteLine(response.Choices[0].Message.Content);
```

## Ruby SDK

### Installation
```bash
gem install modelmuxer-ruby
```

### Quick Start
```ruby
require 'modelmuxer'

client = ModelMuxer::Client.new(api_key: 'your_api_key')

response = client.chat.completions.create(
  messages: [{ role: 'user', content: 'Hello from Ruby!' }],
  routing_strategy: 'hybrid'
)

puts response.choices.first.message.content
```

## PHP SDK

### Installation
```bash
composer require modelmuxer/modelmuxer-php
```

### Quick Start
```php
<?php
require_once 'vendor/autoload.php';

use ModelMuxer\Client;

$client = new Client('your_api_key');

$response = $client->chat->completions->create([
    'messages' => [
        ['role' => 'user', 'content' => 'Hello from PHP!']
    ],
    'routing_strategy' => 'hybrid'
]);

echo $response['choices'][0]['message']['content'];
```

## REST API

### Direct HTTP Calls
```bash
# cURL example
curl -X POST https://api.modelmuxer.com/api/v1/chat/completions \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello via REST API!"}
    ],
    "routing_strategy": "hybrid"
  }'
```

### HTTP Client Libraries
```python
# Using requests library
import requests

response = requests.post(
    "https://api.modelmuxer.com/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer your_api_key",
        "Content-Type": "application/json"
    },
    json={
        "messages": [{"role": "user", "content": "Hello!"}],
        "routing_strategy": "hybrid"
    }
)

result = response.json()
print(result['choices'][0]['message']['content'])
```

## Error Handling

### Python
```python
from modelmuxer import ModelMuxer, ModelMuxerError, RateLimitError

try:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello!"}]
    )
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except ModelMuxerError as e:
    print(f"API error: {e}")
```

### JavaScript
```javascript
try {
  const response = await client.chat.completions.create({
    messages: [{ role: "user", content: "Hello!" }],
  });
} catch (error) {
  if (error.status === 429) {
    console.log("Rate limit exceeded");
  } else {
    console.log(`API error: ${error.message}`);
  }
}
```

## Configuration Options

### Common Configuration
```python
# Environment variables
MODELMUXER_API_KEY=your_api_key
MODELMUXER_BASE_URL=https://api.modelmuxer.com/api/v1
MODELMUXER_TIMEOUT=30
MODELMUXER_MAX_RETRIES=3

# Client configuration
client = ModelMuxer(
    api_key="your_api_key",
    base_url="https://api.modelmuxer.com/api/v1",
    timeout=30,
    max_retries=3,
    default_headers={"User-Agent": "MyApp/1.0"},
    proxy="http://proxy.company.com:8080"
)
```

## Best Practices

### Security
1. **Never hardcode API keys** in source code
2. **Use environment variables** for configuration
3. **Implement proper error handling**
4. **Use HTTPS** for all requests
5. **Rotate API keys** regularly

### Performance
1. **Reuse client instances** instead of creating new ones
2. **Implement request timeouts** appropriate for your use case
3. **Use streaming** for large responses
4. **Implement exponential backoff** for retries
5. **Monitor rate limits** and usage

### Development
1. **Use TypeScript/type hints** when available
2. **Implement comprehensive logging**
3. **Test with different routing strategies**
4. **Monitor costs and usage**
5. **Use SDK-specific debugging features**

## Migration Guides

### From OpenAI SDK
```python
# Before (OpenAI)
from openai import OpenAI
client = OpenAI(api_key="sk-...")

# After (ModelMuxer)
from modelmuxer import ModelMuxer
client = ModelMuxer(api_key="your_key")

# The API is compatible, just add routing parameters
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    routing_strategy="hybrid"  # New parameter
)
```

### From Anthropic SDK
```python
# Before (Anthropic)
from anthropic import Anthropic
client = Anthropic(api_key="sk-ant-...")

# After (ModelMuxer)
from modelmuxer import ModelMuxer
client = ModelMuxer(api_key="your_key")

# Use the same message format
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    routing_strategy="quality_first"
)
```

## Examples and Tutorials

### Complete Examples
- [Chatbot Integration](examples/chatbot.md)
- [Document Analysis](examples/document-analysis.md)
- [Content Generation](examples/content-generation.md)
- [Code Assistant](examples/code-assistant.md)

### Video Tutorials
- Getting Started with ModelMuxer
- Advanced Routing Strategies
- Cost Optimization Techniques
- Production Deployment Guide

## Support and Community

### Getting Help
- **Documentation**: [docs.modelmuxer.com](https://docs.modelmuxer.com)
- **GitHub Issues**: Report bugs and request features
- **Discord**: Join our developer community
- **Stack Overflow**: Tag questions with `modelmuxer`

### Contributing
- SDK contributions welcome on GitHub
- Follow language-specific contribution guidelines
- Ensure tests pass before submitting PRs
- Update documentation for new features

## Changelog

### Latest Updates
- **v1.2.0**: Added streaming support across all SDKs
- **v1.1.0**: Improved error handling and retry logic
- **v1.0.0**: Initial stable release

See individual SDK repositories for detailed changelogs.
