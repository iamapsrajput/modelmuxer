# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Google Gemini provider implementation.

This module provides integration with Google's Gemini API for the ModelMuxer
LLM routing system.
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


class GoogleProvider(LLMProvider):
    """Google Gemini API provider implementation."""
    
    def __init__(self, api_key: str = None):
        if not api_key:
            raise AuthenticationError("Google API key is required")
        
        super().__init__(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta",
            provider_name="google"
        )
        
        # Pricing per million tokens (as of 2024)
        self.pricing = {
            "gemini-1.5-pro": {"input": 3.5, "output": 10.5},
            "gemini-1.5-flash": {"input": 0.075, "output": 0.3},
            "gemini-1.0-pro": {"input": 0.5, "output": 1.5}
        }
        
        self.supported_models = list(self.pricing.keys())
        
        # Rate limits (requests per minute)
        self.rate_limits = {
            "gemini-1.5-pro": 60,
            "gemini-1.5-flash": 1000,
            "gemini-1.0-pro": 60
        }
    
    def _create_headers(self) -> Dict[str, str]:
        """Create headers for Google API requests."""
        return {
            "Content-Type": "application/json",
            "User-Agent": f"ModelMuxer/1.0.0 (Google Gemini)"
        }
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported Google models."""
        return self.supported_models
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost for Google request."""
        if model not in self.pricing:
            model = "gemini-1.5-flash"  # Default fallback
        
        model_pricing = self.pricing[model]
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        
        return input_cost + output_cost
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get rate limit information."""
        return {
            "requests_per_minute": self.rate_limits,
            "tokens_per_minute": {
                "gemini-1.5-pro": 32000,
                "gemini-1.5-flash": 1000000,
                "gemini-1.0-pro": 32000
            }
        }
    
    def _convert_messages_to_google_format(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """Convert OpenAI format messages to Google Gemini format."""
        contents = []
        system_instruction = None
        
        for msg in messages:
            if msg.role == "system":
                # Google uses systemInstruction separately
                system_instruction = msg.content
            elif msg.role == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": msg.content}]
                })
            elif msg.role == "assistant":
                contents.append({
                    "role": "model",  # Google uses "model" instead of "assistant"
                    "parts": [{"text": msg.content}]
                })
        
        result = {"contents": contents}
        if system_instruction:
            result["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
        
        return result
    
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str = "gemini-1.5-flash",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> ChatCompletionResponse:
        """Generate a chat completion using Google Gemini API."""
        start_time = time.time()
        
        # Convert messages to Google format
        google_messages = self._convert_messages_to_google_format(messages)
        
        # Prepare request payload
        payload = google_messages.copy()
        
        # Add generation config
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        
        if generation_config:
            payload["generationConfig"] = generation_config
        
        # Add safety settings (optional)
        payload["safetySettings"] = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        try:
            url = f"{self.base_url}/models/{model}:generateContent"
            
            response = await self.client.post(
                url,
                params={"key": self.api_key},
                json=payload,
                headers=self._create_headers(),
                timeout=60.0
            )
            
            self._handle_http_error(response)
            response_data = response.json()
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Extract content from Google response
            content = ""
            finish_reason = "stop"
            
            if "candidates" in response_data and response_data["candidates"]:
                candidate = response_data["candidates"][0]
                
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    content = "".join([part.get("text", "") for part in parts])
                
                # Map Google finish reasons to OpenAI format
                google_finish_reason = candidate.get("finishReason", "STOP")
                finish_reason_mapping = {
                    "STOP": "stop",
                    "MAX_TOKENS": "length",
                    "SAFETY": "content_filter",
                    "RECITATION": "content_filter",
                    "OTHER": "stop"
                }
                finish_reason = finish_reason_mapping.get(google_finish_reason, "stop")
            
            # Estimate token usage (Google doesn't always provide exact counts)
            input_tokens = sum(estimate_tokens(msg.content, model) for msg in messages)
            output_tokens = estimate_tokens(content, model)
            
            # Check if Google provided usage metadata
            if "usageMetadata" in response_data:
                usage_metadata = response_data["usageMetadata"]
                input_tokens = usage_metadata.get("promptTokenCount", input_tokens)
                output_tokens = usage_metadata.get("candidatesTokenCount", output_tokens)
            
            # Create standardized response
            return self._create_standard_response(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                routing_reason="Selected Google Gemini for request",
                response_time_ms=response_time_ms,
                finish_reason=finish_reason
            )
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(
                    f"Google API rate limit exceeded",
                    provider=self.provider_name
                )
            elif e.response.status_code == 401:
                raise AuthenticationError(
                    f"Google API authentication failed",
                    provider=self.provider_name
                )
            else:
                raise ProviderError(
                    f"Google API error: {e.response.status_code}",
                    provider=self.provider_name,
                    status_code=e.response.status_code
                )
        except httpx.RequestError as e:
            raise ProviderError(f"Google request failed: {str(e)}", provider=self.provider_name)
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(f"Google unexpected error: {str(e)}", provider=self.provider_name)
    
    async def stream_chat_completion(
        self,
        messages: List[ChatMessage],
        model: str = "gemini-1.5-flash",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a chat completion using Google Gemini API."""
        # Convert messages to Google format
        google_messages = self._convert_messages_to_google_format(messages)
        
        # Prepare request payload
        payload = google_messages.copy()
        
        # Add generation config
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        
        if generation_config:
            payload["generationConfig"] = generation_config
        
        try:
            url = f"{self.base_url}/models/{model}:streamGenerateContent"
            
            async with self.client.stream(
                "POST",
                url,
                params={"key": self.api_key},
                json=payload,
                headers=self._create_headers(),
                timeout=120.0
            ) as response:
                self._handle_http_error(response)
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            # Google returns JSON objects separated by newlines
                            chunk_data = json.loads(line)
                            
                            # Convert to OpenAI streaming format
                            if "candidates" in chunk_data and chunk_data["candidates"]:
                                candidate = chunk_data["candidates"][0]
                                
                                if "content" in candidate and "parts" in candidate["content"]:
                                    parts = candidate["content"]["parts"]
                                    text = "".join([part.get("text", "") for part in parts])
                                    
                                    if text:
                                        openai_chunk = {
                                            "id": f"chatcmpl-google-{int(time.time())}",
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"content": text},
                                                "finish_reason": None
                                            }]
                                        }
                                        yield openai_chunk
                                
                                # Check for finish reason
                                if "finishReason" in candidate:
                                    final_chunk = {
                                        "id": f"chatcmpl-google-{int(time.time())}",
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
                                    break
                        
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.RequestError as e:
            raise ProviderError(f"Google streaming request failed: {str(e)}", provider=self.provider_name)
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(f"Google streaming unexpected error: {str(e)}", provider=self.provider_name)
    
    async def health_check(self) -> bool:
        """Check if Google Gemini API is accessible."""
        try:
            test_messages = [ChatMessage(role="user", content="Hi")]
            await self.chat_completion(
                messages=test_messages,
                model="gemini-1.5-flash",
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.warning("google_health_check_failed", error=str(e))
            return False
