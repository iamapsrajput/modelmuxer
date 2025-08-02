# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Pydantic models for request/response schemas and data validation.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime, date
from enum import Enum


class ChatMessage(BaseModel):
    """Individual chat message in a conversation."""

    role: Literal["system", "user", "assistant"] = Field(..., description="The role of the message author")
    content: str = Field(..., description="The content of the message")
    name: Optional[str] = Field(None, description="Optional name of the message author")


class ChatCompletionRequest(BaseModel):
    """Request schema for chat completions endpoint."""

    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    model: Optional[str] = Field(None, description="Model to use (will be overridden by router)")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, description="Sampling temperature")
    top_p: Optional[float] = Field(None, description="Nucleus sampling parameter")
    n: Optional[int] = Field(1, description="Number of completions to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(None, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(None, description="Frequency penalty")
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Logit bias")
    user: Optional[str] = Field(None, description="User identifier")


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens")


class Choice(BaseModel):
    """Individual choice in a chat completion response."""

    index: int = Field(..., description="Index of the choice")
    message: ChatMessage = Field(..., description="The generated message")
    finish_reason: Optional[str] = Field(None, description="Reason the generation finished")


class RouterMetadata(BaseModel):
    """Additional metadata about the routing decision."""

    selected_provider: str = Field(..., description="Provider that was selected")
    selected_model: str = Field(..., description="Model that was selected")
    routing_reason: str = Field(..., description="Reason for the routing decision")
    estimated_cost: float = Field(..., description="Estimated cost in USD")
    response_time_ms: float = Field(..., description="Response time in milliseconds")


class ChatResponse(BaseModel):
    """Basic chat response model for caching."""
    
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for the completion")
    choices: List[Dict[str, Any]] = Field(..., description="List of completion choices")
    usage: Dict[str, int] = Field(..., description="Token usage information")


class ChatCompletionResponse(BaseModel):
    """Response schema for chat completions endpoint."""

    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for the completion")
    choices: List[Choice] = Field(..., description="List of completion choices")
    usage: Usage = Field(..., description="Token usage information")
    router_metadata: RouterMetadata = Field(..., description="Router-specific metadata")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current timestamp")


class UserStats(BaseModel):
    """User usage statistics."""

    user_id: str = Field(..., description="User identifier")
    total_requests: int = Field(..., description="Total number of requests")
    total_cost: float = Field(..., description="Total cost in USD")
    daily_cost: float = Field(..., description="Cost for today in USD")
    monthly_cost: float = Field(..., description="Cost for this month in USD")
    daily_budget: float = Field(..., description="Daily budget limit in USD")
    monthly_budget: float = Field(..., description="Monthly budget limit in USD")
    favorite_model: Optional[str] = Field(None, description="Most used model")


class MetricsResponse(BaseModel):
    """System metrics response."""

    total_requests: int = Field(..., description="Total requests processed")
    total_cost: float = Field(..., description="Total cost across all users")
    active_users: int = Field(..., description="Number of active users")
    provider_usage: Dict[str, int] = Field(..., description="Usage count by provider")
    model_usage: Dict[str, int] = Field(..., description="Usage count by model")
    average_response_time: float = Field(..., description="Average response time in ms")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: Dict[str, Any] = Field(..., description="Error details")

    @classmethod
    def create(cls, message: str, error_type: str = "invalid_request_error", code: Optional[str] = None):
        """Create a standardized error response."""
        error_data = {"message": message, "type": error_type}
        if code:
            error_data["code"] = code
        return cls(error=error_data)


# Enhanced models for Part 2: Cost-Aware Cascading & Analytics


class BudgetPeriodEnum(str, Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
    yearly = "yearly"


class BudgetRequest(BaseModel):
    budget_type: BudgetPeriodEnum
    budget_limit: float = Field(..., gt=0, description="Budget limit in USD")
    provider: Optional[str] = Field(None, description="Specific provider (optional)")
    model: Optional[str] = Field(None, description="Specific model (optional)")
    alert_thresholds: Optional[List[float]] = Field([50.0, 80.0, 95.0], description="Alert thresholds as percentages")

    @validator("alert_thresholds")
    def validate_thresholds(cls, v):
        if v:
            for threshold in v:
                if not 0 <= threshold <= 100:
                    raise ValueError("Alert thresholds must be between 0 and 100")
        return v


class CascadeConfig(BaseModel):
    cascade_type: str = Field("balanced", pattern="^(cost_optimized|quality_focused|balanced)$")
    max_budget: float = Field(0.1, gt=0, le=10.0)
    quality_threshold: float = Field(0.7, ge=0.0, le=1.0)
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)


class EnhancedChatCompletionRequest(ChatCompletionRequest):
    session_id: Optional[str] = None
    cascade_config: Optional[CascadeConfig] = None
    enable_analytics: bool = True
    routing_preference: Optional[str] = Field(None, pattern="^(cost|quality|balanced|speed)$")


class RoutingMetadata(BaseModel):
    strategy_used: str
    total_cost: float
    cascade_steps: Optional[int] = None
    quality_score: Optional[float] = None
    confidence_score: Optional[float] = None
    provider_chain: List[str] = []
    escalation_reasons: List[str] = []
    response_time_ms: float


class EnhancedChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    routing_metadata: Optional[RoutingMetadata] = None
