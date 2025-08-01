"""
Anthropic provider implementation.
"""

import json
import time
from typing import List, Optional, Dict, Any, AsyncGenerator
import httpx

from .base import LLMProvider, ProviderError
from ..models import ChatMessage, ChatCompletionResponse
from ..config import settings


class AnthropicProvider(LLMProvider):
    """Anthropic API provider implementation."""
    
    def __init__(self, api_key: str = None):
        super().__init__(
            api_key=api_key or settings.anthropic_api_key,
            base_url="https://api.anthropic.com/v1",
            provider_name="anthropic"
        )
        
        # Pricing per million tokens (updated as of 2024)
        self.pricing = {
            "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
            "claude-3-opus-20240229": {"input": 15.0, "output": 75.0}
        }
        
        self.supported_models = list(self.pricing.keys())
    
    def _create_headers(self) -> Dict[str, str]:
        """Create headers for Anthropic API requests."""
        headers = super()._create_headers()
        headers["x-api-key"] = self.api_key
        headers["anthropic-version"] = "2023-06-01"
        return headers
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported Anthropic models."""
        return self.supported_models
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost for Anthropic request."""
        if model not in self.pricing:
            return 0.0
        
        model_pricing = self.pricing[model]
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        
        return input_cost + output_cost
    
    def _prepare_messages(self, messages: List[ChatMessage]) -> tuple[str, List[Dict[str, str]]]:
        """Convert ChatMessage objects to Anthropic format."""
        system_message = ""
        conversation_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return system_message, conversation_messages
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation for Anthropic (similar to OpenAI)."""
        # Anthropic uses roughly 4 characters per token
        return len(text) // 4
    
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> ChatCompletionResponse:
        """Generate a chat completion using Anthropic API."""
        start_time = time.time()
        
        # Prepare messages
        system_message, conversation_messages = self._prepare_messages(messages)
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": conversation_messages,
            "max_tokens": max_tokens or settings.max_tokens_default,
            "stream": stream
        }
        
        # Add system message if present
        if system_message:
            payload["system"] = system_message
        
        # Add optional parameters
        if temperature is not None:
            payload["temperature"] = temperature
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ["messages", "model", "stream", "max_tokens"] and value is not None:
                payload[key] = value
        
        try:
            response = await self.client.post(
                f"{self.base_url}/messages",
                headers=self._create_headers(),
                json=payload
            )
            
            self._handle_http_error(response)
            response_data = response.json()
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Extract usage information
            usage = response_data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            
            # If usage not provided, estimate
            if input_tokens == 0:
                input_text = " ".join([msg.content for msg in messages])
                input_tokens = self._estimate_tokens(input_text)
            
            # Extract content
            content_blocks = response_data.get("content", [])
            if not content_blocks:
                raise ProviderError("No content returned from Anthropic", provider=self.provider_name)
            
            # Combine all text content blocks
            content = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    content += block.get("text", "")
            
            if output_tokens == 0:
                output_tokens = self._estimate_tokens(content)
            
            finish_reason = response_data.get("stop_reason", "stop")
            # Map Anthropic stop reasons to OpenAI format
            if finish_reason == "end_turn":
                finish_reason = "stop"
            elif finish_reason == "max_tokens":
                finish_reason = "length"
            
            # Create standardized response
            return self._create_standard_response(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                routing_reason="Selected Anthropic for request",
                response_time_ms=response_time_ms,
                finish_reason=finish_reason
            )
            
        except httpx.RequestError as e:
            raise ProviderError(f"Anthropic request failed: {str(e)}", provider=self.provider_name)
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(f"Anthropic unexpected error: {str(e)}", provider=self.provider_name)
    
    async def stream_chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a chat completion using Anthropic API."""
        # Prepare messages
        system_message, conversation_messages = self._prepare_messages(messages)
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": conversation_messages,
            "max_tokens": max_tokens or settings.max_tokens_default,
            "stream": True
        }
        
        # Add system message if present
        if system_message:
            payload["system"] = system_message
        
        # Add optional parameters
        if temperature is not None:
            payload["temperature"] = temperature
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ["messages", "model", "stream", "max_tokens"] and value is not None:
                payload[key] = value
        
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/messages",
                headers=self._create_headers(),
                json=payload
            ) as response:
                self._handle_http_error(response)
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        
                        try:
                            chunk = json.loads(data)
                            # Convert Anthropic streaming format to OpenAI format
                            if chunk.get("type") == "content_block_delta":
                                delta = chunk.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    # Convert to OpenAI streaming format
                                    openai_chunk = {
                                        "id": f"chatcmpl-anthropic-{int(time.time())}",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {
                                                "content": delta.get("text", "")
                                            },
                                            "finish_reason": None
                                        }]
                                    }
                                    yield openai_chunk
                            elif chunk.get("type") == "message_stop":
                                # Send final chunk
                                final_chunk = {
                                    "id": f"chatcmpl-anthropic-{int(time.time())}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "stop"
                                    }]
                                }
                                yield final_chunk
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.RequestError as e:
            raise ProviderError(f"Anthropic streaming request failed: {str(e)}", provider=self.provider_name)
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(f"Anthropic streaming unexpected error: {str(e)}", provider=self.provider_name)
