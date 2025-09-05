# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""Validation helper functions for ModelMuxer."""

from fastapi import HTTPException

from app.models import ErrorResponse


def validate_model_format(model: str) -> None:
    """
    Validate that a model name does not contain proxy-style separators.

    Args:
        model: The model name to validate

    Raises:
        HTTPException: If the model name contains ':' or '/' separators
    """
    if model and (":" in model or "/" in model):
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse.create(
                message="Invalid model name format. Use direct provider model names (e.g., 'gpt-4o', 'claude-3-5-sonnet-latest').",
                error_type="invalid_request",
                code="invalid_model_format",
                details={"provided_model": model},
            ).dict(),
        )


def reject_proxy_style_model(model: str) -> None:
    """
    Reject proxy-style model names (alias for validate_model_format for clarity).

    Args:
        model: The model name to validate

    Raises:
        HTTPException: If the model name contains ':' or '/' separators
    """
    validate_model_format(model)
