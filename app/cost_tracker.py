# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Cost calculation and tracking utilities.
"""

import tiktoken
from typing import List, Dict, Any, Optional
from .config import settings
from .models import ChatMessage


class CostTracker:
    """Handles cost calculation and token counting for different providers."""
    
    def __init__(self):
        self.pricing = settings.get_provider_pricing()
        # Initialize tokenizers for different providers
        self._tokenizers = {}
    
    def get_tokenizer(self, provider: str, model: str):
        """Get or create tokenizer for a specific model."""
        key = f"{provider}:{model}"
        if key not in self._tokenizers:
            try:
                if provider == "openai":
                    if "gpt-4" in model:
                        self._tokenizers[key] = tiktoken.encoding_for_model("gpt-4")
                    else:
                        self._tokenizers[key] = tiktoken.encoding_for_model("gpt-3.5-turbo")
                elif provider == "anthropic":
                    # Anthropic uses a similar tokenizer to GPT-3.5
                    self._tokenizers[key] = tiktoken.encoding_for_model("gpt-3.5-turbo")
                elif provider == "mistral":
                    # Mistral uses a similar tokenizer to GPT-3.5
                    self._tokenizers[key] = tiktoken.encoding_for_model("gpt-3.5-turbo")
                else:
                    # Default fallback
                    self._tokenizers[key] = tiktoken.encoding_for_model("gpt-3.5-turbo")
            except Exception:
                # Fallback to cl100k_base encoding
                self._tokenizers[key] = tiktoken.get_encoding("cl100k_base")
        
        return self._tokenizers[key]
    
    def count_tokens(self, messages: List[ChatMessage], provider: str, model: str) -> int:
        """Count tokens in a list of messages."""
        tokenizer = self.get_tokenizer(provider, model)
        
        total_tokens = 0
        for message in messages:
            # Add tokens for role and content
            total_tokens += len(tokenizer.encode(message.role))
            total_tokens += len(tokenizer.encode(message.content))
            # Add overhead tokens (varies by provider, using OpenAI's format)
            total_tokens += 4  # Overhead per message
        
        # Add overhead for the conversation
        total_tokens += 2
        
        return total_tokens
    
    def estimate_output_tokens(self, max_tokens: Optional[int] = None) -> int:
        """Estimate output tokens based on max_tokens parameter."""
        if max_tokens:
            return min(max_tokens, 1000)  # Cap at reasonable default
        return settings.max_tokens_default // 2  # Estimate half of max as typical output
    
    def calculate_cost(
        self, 
        provider: str, 
        model: str, 
        input_tokens: int, 
        output_tokens: int
    ) -> float:
        """Calculate cost for a request."""
        if provider not in self.pricing:
            return 0.0
        
        if model not in self.pricing[provider]:
            return 0.0
        
        model_pricing = self.pricing[provider][model]
        
        # Calculate cost per million tokens
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        
        return input_cost + output_cost
    
    def estimate_request_cost(
        self, 
        messages: List[ChatMessage], 
        provider: str, 
        model: str,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Estimate the cost of a request before making it."""
        input_tokens = self.count_tokens(messages, provider, model)
        estimated_output_tokens = self.estimate_output_tokens(max_tokens)
        estimated_cost = self.calculate_cost(provider, model, input_tokens, estimated_output_tokens)
        
        return {
            "input_tokens": input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_total_tokens": input_tokens + estimated_output_tokens,
            "estimated_cost": estimated_cost,
            "provider": provider,
            "model": model
        }
    
    def get_cheapest_model_for_task(self, task_type: str = "general") -> Dict[str, str]:
        """Get the cheapest model for a given task type."""
        # Define model preferences by task type
        task_models = {
            "simple": [
                ("mistral", "mistral-small-latest"),
                ("anthropic", "claude-3-haiku-20240307"),
                ("openai", "gpt-3.5-turbo")
            ],
            "code": [
                ("openai", "gpt-4o"),
                ("anthropic", "claude-3-sonnet-20240229"),
                ("openai", "gpt-3.5-turbo")
            ],
            "complex": [
                ("openai", "gpt-4o"),
                ("anthropic", "claude-3-sonnet-20240229"),
                ("openai", "gpt-3.5-turbo")
            ],
            "general": [
                ("openai", "gpt-3.5-turbo"),
                ("anthropic", "claude-3-haiku-20240307"),
                ("mistral", "mistral-small-latest")
            ]
        }
        
        models = task_models.get(task_type, task_models["general"])
        
        # Return the first available model (they're ordered by preference)
        for provider, model in models:
            if provider in self.pricing and model in self.pricing[provider]:
                return {"provider": provider, "model": model}
        
        # Fallback
        return {"provider": "openai", "model": "gpt-3.5-turbo"}
    
    def compare_model_costs(
        self, 
        messages: List[ChatMessage], 
        models: List[Dict[str, str]],
        max_tokens: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Compare costs across multiple models for the same request."""
        comparisons = []
        
        for model_info in models:
            provider = model_info["provider"]
            model = model_info["model"]
            
            estimate = self.estimate_request_cost(messages, provider, model, max_tokens)
            estimate.update(model_info)
            comparisons.append(estimate)
        
        # Sort by estimated cost
        return sorted(comparisons, key=lambda x: x["estimated_cost"])
    
    def get_model_info(self, provider: str, model: str) -> Dict[str, Any]:
        """Get detailed information about a model including pricing."""
        if provider not in self.pricing or model not in self.pricing[provider]:
            return {}
        
        pricing = self.pricing[provider][model]
        return {
            "provider": provider,
            "model": model,
            "input_price_per_million": pricing["input"],
            "output_price_per_million": pricing["output"],
            "total_price_per_million": pricing["input"] + pricing["output"]
        }


# Global cost tracker instance
cost_tracker = CostTracker()
