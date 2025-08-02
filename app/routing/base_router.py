# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Base router implementation for ModelMuxer.

This module provides the abstract base class that all routing strategies
must inherit from, ensuring consistent interface and behavior.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import time
import structlog

from ..core.interfaces import RouterInterface
from ..core.exceptions import RoutingError
from ..models import ChatMessage

logger = structlog.get_logger(__name__)


class BaseRouter(RouterInterface):
    """
    Abstract base class for all routing strategies.
    
    This class provides common functionality and enforces the interface
    that all routing implementations must follow.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.metrics = {
            "total_requests": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "average_response_time": 0.0
        }
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize router-specific configuration."""
        pass
    
    async def select_provider_and_model(
        self,
        messages: List[ChatMessage],
        user_id: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, str, float]:
        """
        Select the optimal provider and model for the given request.
        
        This method implements the common routing flow and delegates
        the actual selection logic to the concrete implementation.
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            # Validate inputs
            if not messages:
                raise RoutingError("No messages provided for routing")
            
            # Analyze the prompt
            analysis = await self.analyze_prompt(messages)
            
            # Apply constraints if provided
            if constraints:
                analysis = self._apply_constraints(analysis, constraints)
            
            # Perform the actual routing
            provider, model, reasoning, confidence = await self._route_request(
                messages, analysis, user_id, constraints
            )
            
            # Update metrics
            response_time = time.time() - start_time
            self._update_metrics(response_time, success=True)
            
            # Log the routing decision
            logger.info(
                "routing_decision",
                router=self.name,
                provider=provider,
                model=model,
                reasoning=reasoning,
                confidence=confidence,
                response_time=response_time,
                user_id=user_id
            )
            
            return provider, model, reasoning, confidence
            
        except Exception as e:
            self.metrics["failed_routes"] += 1
            response_time = time.time() - start_time
            self._update_metrics(response_time, success=False)
            
            logger.error(
                "routing_error",
                router=self.name,
                error=str(e),
                response_time=response_time,
                user_id=user_id
            )
            
            if isinstance(e, RoutingError):
                raise
            else:
                raise RoutingError(f"Routing failed: {str(e)}", routing_strategy=self.name)
    
    @abstractmethod
    async def _route_request(
        self,
        messages: List[ChatMessage],
        analysis: Dict[str, Any],
        user_id: Optional[str],
        constraints: Optional[Dict[str, Any]]
    ) -> Tuple[str, str, str, float]:
        """
        Perform the actual routing logic.
        
        This method must be implemented by concrete router classes.
        
        Returns:
            Tuple of (provider, model, reasoning, confidence)
        """
        pass
    
    def _apply_constraints(
        self,
        analysis: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply routing constraints to the analysis."""
        # Budget constraints
        if "max_cost" in constraints:
            analysis["max_cost"] = constraints["max_cost"]
        
        # Latency constraints
        if "max_latency" in constraints:
            analysis["max_latency"] = constraints["max_latency"]
        
        # Provider preferences
        if "preferred_providers" in constraints:
            analysis["preferred_providers"] = constraints["preferred_providers"]
        
        # Model preferences
        if "excluded_models" in constraints:
            analysis["excluded_models"] = constraints["excluded_models"]
        
        return analysis
    
    def _update_metrics(self, response_time: float, success: bool) -> None:
        """Update router metrics."""
        if success:
            self.metrics["successful_routes"] += 1
        
        # Update average response time
        total_requests = self.metrics["total_requests"]
        current_avg = self.metrics["average_response_time"]
        self.metrics["average_response_time"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get router performance metrics."""
        return {
            "router_name": self.name,
            "metrics": self.metrics.copy(),
            "success_rate": (
                self.metrics["successful_routes"] / max(self.metrics["total_requests"], 1)
            )
        }
    
    def get_supported_strategies(self) -> List[str]:
        """Get list of supported routing strategies."""
        return [self.name]
    
    def reset_metrics(self) -> None:
        """Reset router metrics."""
        self.metrics = {
            "total_requests": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "average_response_time": 0.0
        }
