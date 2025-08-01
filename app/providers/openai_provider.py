"""
OpenAI provider implementation.
"""

import json
import time
from typing import List, Optional, Dict, Any, AsyncGenerator
import httpx

from .base import LLMProvider, ProviderError
from ..models import ChatMessage, ChatCompletionResponse
from ..config import settings


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, api_key: str = None):
        super().__init__(
            api_key=api_key or settings.openai_api_key,
            base_url="https://api.openai.com/v1",
            provider_name="openai"
        )
        
        # Pricing per million tokens (updated as of 2024)
        self.pricing = {
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.6},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "gpt-3.5-turbo-0125": {"input": 0.5, "output": 1.5}
        }
        
        self.supported_models = list(self.pricing.keys())
    
    def _create_headers(self) -> Dict[str, str]:
        """Create headers for OpenAI API requests."""
        headers = super()._create_headers()
        headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported OpenAI models."""
        return self.supported_models
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost for OpenAI request."""
        if model not in self.pricing:
            return 0.0
        
        model_pricing = self.pricing[model]
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        
        return input_cost + output_cost
    
    def _prepare_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        """Convert ChatMessage objects to OpenAI format."""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                **({"name": msg.name} if msg.name else {})
            }
            for msg in messages
        ]
    
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> ChatCompletionResponse:
        """Generate a chat completion using OpenAI API."""
        start_time = time.time()
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": self._prepare_messages(messages),
            "stream": stream
        }
        
        # Add optional parameters
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ["messages", "model", "stream"] and value is not None:
                payload[key] = value
        
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=self._create_headers(),
                json=payload
            )
            
            self._handle_http_error(response)
            response_data = response.json()
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Extract usage information
            usage = response_data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            
            # Extract content
            choices = response_data.get("choices", [])
            if not choices:
                raise ProviderError("No choices returned from OpenAI", provider=self.provider_name)
            
            content = choices[0].get("message", {}).get("content", "")
            finish_reason = choices[0].get("finish_reason", "stop")
            
            # Create standardized response
            return self._create_standard_response(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                routing_reason="Selected OpenAI for request",
                response_time_ms=response_time_ms,
                finish_reason=finish_reason
            )
            
        except httpx.RequestError as e:
            raise ProviderError(f"OpenAI request failed: {str(e)}", provider=self.provider_name)
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(f"OpenAI unexpected error: {str(e)}", provider=self.provider_name)
    
    async def stream_chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a chat completion using OpenAI API."""
        # Prepare request payload
        payload = {
            "model": model,
            "messages": self._prepare_messages(messages),
            "stream": True
        }
        
        # Add optional parameters
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ["messages", "model", "stream"] and value is not None:
                payload[key] = value
        
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._create_headers(),
                json=payload
            ) as response:
                self._handle_http_error(response)
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        
                        if data == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data)
                            yield chunk
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.RequestError as e:
            raise ProviderError(f"OpenAI streaming request failed: {str(e)}", provider=self.provider_name)
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(f"OpenAI streaming unexpected error: {str(e)}", provider=self.provider_name)
