# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Pydantic models for request/response schemas and data validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, validator


class ChatMessage(BaseModel):
    """Individual chat message in a conversation."""

    role: Literal["system", "user", "assistant"] = Field(..., description="The role of the message author")
    content: str = Field(..., description="The content of the message")
    name: str | None = Field(None, description="Optional name of the message author")


class ChatCompletionRequest(BaseModel):
    """Request schema for chat completions endpoint."""

    messages: list[ChatMessage] = Field(..., description="List of messages in the conversation")
    model: str | None = Field(None, description="Model to use (will be overridden by router)")
    max_tokens: int | None = Field(None, description="Maximum tokens to generate")
    temperature: float | None = Field(None, description="Sampling temperature")
    top_p: float | None = Field(None, description="Nucleus sampling parameter")
    n: int | None = Field(1, description="Number of completions to generate")
    stream: bool | None = Field(False, description="Whether to stream the response")
    stop: str | list[str] | None = Field(None, description="Stop sequences")
    presence_penalty: float | None = Field(None, description="Presence penalty")
    frequency_penalty: float | None = Field(None, description="Frequency penalty")
    logit_bias: dict[str, float] | None = Field(None, description="Logit bias")
    user: str | None = Field(None, description="User identifier")


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens")


class Choice(BaseModel):
    """Individual choice in a chat completion response."""

    index: int = Field(..., description="Index of the choice")
    message: ChatMessage = Field(..., description="The generated message")
    finish_reason: str | None = Field(None, description="Reason the generation finished")


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
    choices: list[dict[str, Any]] = Field(..., description="List of completion choices")
    usage: dict[str, int] = Field(..., description="Token usage information")


class ChatCompletionResponse(BaseModel):
    """Response schema for chat completions endpoint."""

    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for the completion")
    choices: list[Choice] = Field(..., description="List of completion choices")
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
    favorite_model: str | None = Field(None, description="Most used model")


class MetricsResponse(BaseModel):
    """System metrics response."""

    total_requests: int = Field(..., description="Total requests processed")
    total_cost: float = Field(..., description="Total cost across all users")
    active_users: int = Field(..., description="Number of active users")
    provider_usage: dict[str, int] = Field(..., description="Usage count by provider")
    model_usage: dict[str, int] = Field(..., description="Usage count by model")
    average_response_time: float = Field(..., description="Average response time in ms")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: dict[str, Any] = Field(..., description="Error details")

    @classmethod
    def create(
        cls, message: str, error_type: str = "invalid_request_error", code: str | None = None
    ) -> "ErrorResponse":
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
    provider: str | None = Field(None, description="Specific provider (optional)")
    model: str | None = Field(None, description="Specific model (optional)")
    alert_thresholds: list[float] | None = Field([50.0, 80.0, 95.0], description="Alert thresholds as percentages")

    @validator("alert_thresholds")
    def validate_thresholds(cls, v: list[float] | None) -> list[float] | None:
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
    session_id: str | None = None
    cascade_config: CascadeConfig | None = None
    enable_analytics: bool = True
    routing_preference: str | None = Field(None, pattern="^(cost|quality|balanced|speed)$")


class RoutingMetadata(BaseModel):
    strategy_used: str
    total_cost: float
    cascade_steps: int | None = None
    quality_score: float | None = None
    confidence_score: float | None = None
    provider_chain: list[str] = []
    escalation_reasons: list[str] = []
    response_time_ms: float


class EnhancedChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]
    routing_metadata: RoutingMetadata | None = None


class BudgetAlert(BaseModel):
    """Budget alert information."""

    type: str = Field(..., description="Alert type (warning, critical)")
    message: str = Field(..., description="Alert message")
    threshold: float = Field(..., description="Threshold percentage that triggered alert")
    current_usage: float = Field(..., description="Current usage percentage")


class BudgetStatus(BaseModel):
    """Budget status response."""

    budget_type: BudgetPeriodEnum
    budget_limit: float = Field(..., description="Budget limit in USD")
    current_usage: float = Field(..., description="Current usage in USD")
    usage_percentage: float = Field(..., description="Usage as percentage of budget")
    remaining_budget: float = Field(..., description="Remaining budget in USD")
    provider: str | None = Field(None, description="Specific provider (if applicable)")
    model: str | None = Field(None, description="Specific model (if applicable)")
    alerts: list[BudgetAlert] = Field(default_factory=list, description="Active budget alerts")
    period_start: str = Field(..., description="Budget period start date")
    period_end: str = Field(..., description="Budget period end date")


class BudgetResponse(BaseModel):
    """Budget management response."""

    message: str = Field(..., description="Response message")
    budgets: list[BudgetStatus] = Field(default_factory=list, description="Budget statuses")
    total_budgets: int = Field(..., description="Total number of budgets configured")
