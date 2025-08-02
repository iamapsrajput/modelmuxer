# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Abstract interfaces for ModelMuxer components.

This module defines the core interfaces that all components must implement
to ensure consistency and interoperability across the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
from ..models import ChatMessage, ChatCompletionResponse


class RouterInterface(ABC):
    """Abstract interface for routing strategies."""
    
    @abstractmethod
    async def select_provider_and_model(
        self,
        messages: List[ChatMessage],
        user_id: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, str, float]:
        """
        Select the optimal provider and model for the given request.
        
        Args:
            messages: List of chat messages
            user_id: Optional user identifier
            constraints: Optional routing constraints (budget, latency, etc.)
            
        Returns:
            Tuple of (provider_name, model_name, reasoning, confidence_score)
        """
        pass
    
    @abstractmethod
    async def analyze_prompt(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """
        Analyze prompt characteristics for routing decisions.
        
        Args:
            messages: List of chat messages
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    @abstractmethod
    def get_supported_strategies(self) -> List[str]:
        """Get list of supported routing strategies."""
        pass


class ProviderInterface(ABC):
    """Abstract interface for LLM providers."""
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        **kwargs
    ) -> ChatCompletionResponse:
        """Generate a chat completion."""
        pass
    
    @abstractmethod
    async def stream_chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a chat completion."""
        pass
    
    @abstractmethod
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate the cost for a request."""
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Get list of supported models."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        pass
    
    @abstractmethod
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get rate limit information."""
        pass


class CacheInterface(ABC):
    """Abstract interface for caching implementations."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache with optional TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class ClassifierInterface(ABC):
    """Abstract interface for prompt classifiers."""
    
    @abstractmethod
    async def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify a text prompt.
        
        Args:
            text: Input text to classify
            
        Returns:
            Classification results with confidence scores
        """
        pass
    
    @abstractmethod
    async def train(self, training_data: List[Dict[str, Any]]) -> bool:
        """
        Train the classifier with new data.
        
        Args:
            training_data: List of training examples
            
        Returns:
            True if training was successful
        """
        pass
    
    @abstractmethod
    def get_categories(self) -> List[str]:
        """Get list of supported classification categories."""
        pass


class MetricsInterface(ABC):
    """Abstract interface for metrics collection."""
    
    @abstractmethod
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        pass
    
    @abstractmethod
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value."""
        pass
    
    @abstractmethod
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric value."""
        pass
    
    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        pass
