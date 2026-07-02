# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Declarative routing preference rules loaded from JSON configuration.

Rules match on intent label, task type, model prefix, and optional max cost,
then supply an ordered list of provider/model preferences for the router.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.exceptions import RouterConfigurationError

logger = logging.getLogger(__name__)

# Aliases used in routing rule files (e.g. intent: code)
INTENT_ALIASES: dict[str, str] = {
    "code": "code_gen",
    "complex": "deep_reason",
    "simple": "chat_lite",
    "general": "chat_lite",
    "chat": "chat_lite",
    "reasoning": "deep_reason",
    "json": "json_extract",
    "translate": "translation",
    "safety": "safety_risk",
}


def _normalize_intent(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    return INTENT_ALIASES.get(lowered, lowered)


def _parse_prefer_entry(entry: str) -> tuple[str, str]:
    """Parse 'provider:model' or 'provider/model' into (provider, model)."""
    normalized = entry.strip()
    if ":" in normalized:
        provider, model = normalized.split(":", 1)
    elif "/" in normalized:
        provider, model = normalized.split("/", 1)
    else:
        raise ValueError(f"Invalid prefer entry (expected provider:model): {entry!r}")
    if not provider or not model:
        raise ValueError(f"Invalid prefer entry (empty provider or model): {entry!r}")
    return provider, model


@dataclass
class RoutingRule:
    """Single declarative routing preference rule."""

    name: str
    prefer: list[tuple[str, str]] = field(default_factory=list)
    intent: str | None = None
    task_type: str | None = None
    model_prefix: str | None = None
    max_cost: float | None = None

    def matches(
        self,
        *,
        intent_label: str,
        task_type: str,
        requested_model: str | None,
        budget: float,
    ) -> bool:
        if self.intent is not None:
            normalized_rule = _normalize_intent(self.intent)
            normalized_label = _normalize_intent(intent_label)
            if normalized_rule != normalized_label:
                return False
        if self.task_type is not None and self.task_type != task_type:
            return False
        if self.model_prefix is not None:
            if not requested_model or not requested_model.startswith(self.model_prefix):
                return False
        if self.max_cost is not None and budget > self.max_cost:
            return False
        return True


@dataclass
class RoutingRulesConfig:
    """Loaded routing rules with validation metadata."""

    rules: list[RoutingRule] = field(default_factory=list)
    source_path: str | None = None

    def resolve_preferences(
        self,
        *,
        intent_label: str,
        task_type: str,
        requested_model: str | None,
        budget: float,
    ) -> tuple[list[tuple[str, str]], str | None]:
        """Return the first matching rule's preferences and rule name."""
        for rule in self.rules:
            if rule.matches(
                intent_label=intent_label,
                task_type=task_type,
                requested_model=requested_model,
                budget=budget,
            ):
                return rule.prefer, rule.name
        return [], None


def _parse_rule(raw: dict[str, Any], index: int) -> RoutingRule:
    name = str(raw.get("name") or f"rule_{index}")
    prefer_raw = raw.get("prefer") or []
    if not isinstance(prefer_raw, list) or not prefer_raw:
        raise ValueError(f"Rule {name!r} must include a non-empty 'prefer' list")
    prefer = [_parse_prefer_entry(str(item)) for item in prefer_raw]
    intent = raw.get("intent")
    task_type = raw.get("task_type") or raw.get("task")
    model_prefix = raw.get("model_prefix")
    max_cost = raw.get("max_cost")
    if max_cost is not None:
        max_cost = float(max_cost)
    return RoutingRule(
        name=name,
        prefer=prefer,
        intent=str(intent) if intent is not None else None,
        task_type=str(task_type) if task_type is not None else None,
        model_prefix=str(model_prefix) if model_prefix is not None else None,
        max_cost=max_cost,
    )


def load_routing_rules_from_data(data: Any, *, source: str = "<inline>") -> RoutingRulesConfig:
    """Parse routing rules from a dict or list structure."""
    rules_raw: list[Any]
    if isinstance(data, dict):
        rules_raw = data.get("rules") or data.get("routing_rules") or []
    elif isinstance(data, list):
        rules_raw = data
    else:
        raise TypeError("Routing rules must be a JSON object with 'rules' or a list")

    if not isinstance(rules_raw, list):
        raise TypeError("'rules' must be a list")

    rules: list[RoutingRule] = []
    for index, item in enumerate(rules_raw):
        if not isinstance(item, dict):
            raise TypeError(f"Rule at index {index} must be an object")
        rules.append(_parse_rule(item, index))
    return RoutingRulesConfig(rules=rules, source_path=source)


def load_routing_rules(path: str | None) -> RoutingRulesConfig:
    """Load routing rules from a JSON file. Returns empty config when path is unset."""
    if not path:
        return RoutingRulesConfig()
    file_path = Path(path)
    if not file_path.exists():
        logger.warning("Routing rules file not found: %s", path)
        return RoutingRulesConfig(source_path=path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    config = load_routing_rules_from_data(data, source=str(file_path))
    config.source_path = str(file_path)
    logger.info("Loaded %d routing rules from %s", len(config.rules), file_path)
    return config


def validate_routing_rules(
    config: RoutingRulesConfig,
    *,
    mode: str,
    price_table_keys: set[str],
    available_providers: set[str],
) -> None:
    """Validate rule targets against price table and provider registry."""
    if not config.rules:
        return

    missing_pricing: set[str] = set()
    missing_providers: set[str] = set()
    for rule in config.rules:
        for provider, model in rule.prefer:
            model_key = f"{provider}:{model}"
            if model_key not in price_table_keys and provider != "ollama":
                missing_pricing.add(model_key)
            if provider not in available_providers:
                missing_providers.add(provider)

    messages: list[str] = []
    if missing_pricing:
        messages.append(
            "Routing rules reference models missing from price table: " f"{sorted(missing_pricing)}"
        )
    if missing_providers:
        messages.append(
            "Routing rules reference unavailable providers: "
            f"{sorted(missing_providers)} (available: {sorted(available_providers)})"
        )
    if not messages:
        return

    error_msg = "; ".join(messages)
    if mode == "production":
        logger.error(error_msg)
        raise RouterConfigurationError(error_msg)
    logger.warning(error_msg)


__all__ = [
    "RoutingRule",
    "RoutingRulesConfig",
    "load_routing_rules",
    "load_routing_rules_from_data",
    "validate_routing_rules",
]
