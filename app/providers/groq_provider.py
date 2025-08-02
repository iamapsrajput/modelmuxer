# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Groq provider implementation.

This module provides integration with Groq's API for the ModelMuxer
LLM routing system. Groq provides fast inference for open-source models.
"""

import json
import time
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
import httpx
import structlog

from .base import LLMProvider, ProviderError, RateLimitError, AuthenticationError
from ..models import ChatMessage, ChatCompletionResponse, Choice, Usage, RouterMetadata
from ..core.utils import estimate_tokens

logger = structlog.get_logger(__name__)


class GroqProvider(LLMProvider):
    """Groq API provider implementation."""
    
    def __init__(self, api_key: str = None):
        if not api_key:
            raise AuthenticationError("Groq API key is required")
        
        super().__init__(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            provider_name="groq"
        )
        
        # Pricing per million tokens (as of 2024) - Groq is very cost-effective
        self.pricing = {
            "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
            "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
            "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
            "gemma-7b-it": {"input": 0.07, "output": 0.07},
            "gemma2-9b-it": {"input": 0.20, "output": 0.20}
        }
        
        self.supported_models = list(self.pricing.keys())
        
        # Rate limits (requests per minute) - Groq has generous limits
        self.rate_limits = {
            "llama-3.1-70b-versatile": 30,
            "llama-3.1-8b-instant": 30,
            "mixtral-8x7b-32768": 30,
            "gemma-7b-it": 30,
            "gemma2-9b-it": 30
        }
    
    def _create_headers(self) -> Dict[str, str]:
        """Create headers for Groq API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"ModelMuxer/1.0.0 (Groq)"
        }
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported Groq models."""
        return self.supported_models
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost for Groq request."""
        if model not in self.pricing:
            model = "llama-3.1-8b-instant"  # Default fallback
        
        model_pricing = self.pricing[model]
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        
        return input_cost + output_cost
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get rate limit information."""
        return {
            "requests_per_minute": self.rate_limits,
            "tokens_per_minute": {
                "llama-3.1-70b-versatile": 6000,
                "llama-3.1-8b-instant": 30000,
                "mixtral-8x7b-32768": 5000,
                "gemma-7b-it": 15000,
                "gemma2-9b-it": 15000
            }
        }
    
    def _prepare_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        """Convert ChatMessage objects to Groq format (OpenAI-compatible)."""
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
        model: str = "llama-3.1-8b-instant",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> ChatCompletionResponse:
        """Generate a chat completion using Groq API."""
        start_time = time.time()
        
        # Prepare request payload (OpenAI-compatible)
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
                json=payload,
                headers=self._create_headers(),
                timeout=60.0
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
                raise ProviderError("No choices returned from Groq", provider=self.provider_name)
            
            content = choices[0].get("message", {}).get("content", "")
            finish_reason = choices[0].get("finish_reason", "stop")
            
            # Create standardized response
            return self._create_standard_response(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                routing_reason="Selected Groq for fast inference",
                response_time_ms=response_time_ms,
                finish_reason=finish_reason
            )
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(
                    f"Groq API rate limit exceeded",
                    provider=self.provider_name
                )
            elif e.response.status_code == 401:
                raise AuthenticationError(
                    f"Groq API authentication failed",
                    provider=self.provider_name
                )
            else:
                error_detail = ""
                try:
                    error_data = e.response.json()
                    error_detail = error_data.get("error", {}).get("message", "")
                except:
                    pass
                
                raise ProviderError(
                    f"Groq API error: {e.response.status_code} {error_detail}",
                    provider=self.provider_name,
                    status_code=e.response.status_code
                )
        except httpx.RequestError as e:
            raise ProviderError(f"Groq request failed: {str(e)}", provider=self.provider_name)
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(f"Groq unexpected error: {str(e)}", provider=self.provider_name)
    
    async def stream_chat_completion(
        self,
        messages: List[ChatMessage],
        model: str = "llama-3.1-8b-instant",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a chat completion using Groq API."""
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
                json=payload,
                headers=self._create_headers(),
                timeout=120.0
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
            raise ProviderError(f"Groq streaming request failed: {str(e)}", provider=self.provider_name)
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(f"Groq streaming unexpected error: {str(e)}", provider=self.provider_name)
    
    async def health_check(self) -> bool:
        """Check if Groq API is accessible."""
        try:
            test_messages = [ChatMessage(role="user", content="Hi")]
            await self.chat_completion(
                messages=test_messages,
                model="llama-3.1-8b-instant",
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.warning("groq_health_check_failed", error=str(e))
            return False
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get detailed information about a Groq model."""
        model_info = {
            "llama-3.1-70b-versatile": {
                "description": "Meta's Llama 3.1 70B model, versatile for many tasks",
                "context_length": 131072,
                "strengths": ["reasoning", "code", "math"],
                "speed": "medium"
            },
            "llama-3.1-8b-instant": {
                "description": "Meta's Llama 3.1 8B model, optimized for speed",
                "context_length": 131072,
                "strengths": ["speed", "general tasks"],
                "speed": "very fast"
            },
            "mixtral-8x7b-32768": {
                "description": "Mistral's Mixtral 8x7B mixture of experts model",
                "context_length": 32768,
                "strengths": ["multilingual", "code", "reasoning"],
                "speed": "fast"
            },
            "gemma-7b-it": {
                "description": "Google's Gemma 7B instruction-tuned model",
                "context_length": 8192,
                "strengths": ["instruction following", "safety"],
                "speed": "fast"
            },
            "gemma2-9b-it": {
                "description": "Google's Gemma 2 9B instruction-tuned model",
                "context_length": 8192,
                "strengths": ["improved performance", "efficiency"],
                "speed": "fast"
            }
        }
        
        return model_info.get(model, {
            "description": "Unknown model",
            "context_length": 4096,
            "strengths": [],
            "speed": "unknown"
        })
