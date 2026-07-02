#!/usr/bin/env python3
# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.exceptions import RouterConfigurationError
from app.core.routing_rules import (
    RoutingRulesConfig,
    load_routing_rules,
    load_routing_rules_from_data,
    validate_routing_rules,
)


def test_load_routing_rules_from_data_parses_prefer_entries():
    data = {
        "rules": [
            {
                "name": "code_rule",
                "intent": "code",
                "prefer": ["openai:gpt-4o-mini", "anthropic/claude-3-5-haiku-20241022"],
            }
        ]
    }
    config = load_routing_rules_from_data(data)
    assert len(config.rules) == 1
    rule = config.rules[0]
    assert rule.name == "code_rule"
    assert rule.intent == "code"
    assert rule.prefer == [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-haiku-20241022"),
    ]


def test_rule_matches_intent_alias_and_task_type():
    config = load_routing_rules_from_data(
        {
            "rules": [
                {"name": "by_intent", "intent": "code", "prefer": ["openai:gpt-4o-mini"]},
                {"name": "by_task", "task_type": "simple", "prefer": ["openai:gpt-3.5-turbo"]},
            ]
        }
    )
    prefs, name = config.resolve_preferences(
        intent_label="code_gen",
        task_type="code",
        requested_model=None,
        budget=0.08,
    )
    assert name == "by_intent"
    assert prefs == [("openai", "gpt-4o-mini")]

    prefs, name = config.resolve_preferences(
        intent_label="chat_lite",
        task_type="simple",
        requested_model=None,
        budget=0.08,
    )
    assert name == "by_task"
    assert prefs == [("openai", "gpt-3.5-turbo")]


def test_rule_max_cost_and_model_prefix():
    config = load_routing_rules_from_data(
        {
            "rules": [
                {
                    "name": "budget_rule",
                    "intent": "code_gen",
                    "max_cost": 0.05,
                    "prefer": ["openai:gpt-4o-mini"],
                },
                {
                    "name": "local_prefix",
                    "model_prefix": "ollama/",
                    "prefer": ["ollama:llama3.2"],
                },
            ]
        }
    )
    _, name = config.resolve_preferences(
        intent_label="code_gen",
        task_type="code",
        requested_model=None,
        budget=0.10,
    )
    assert name is None

    _, name = config.resolve_preferences(
        intent_label="code_gen",
        task_type="code",
        requested_model=None,
        budget=0.04,
    )
    assert name == "budget_rule"

    _, name = config.resolve_preferences(
        intent_label="chat_lite",
        task_type="general",
        requested_model="ollama/llama3.2",
        budget=0.08,
    )
    assert name == "local_prefix"


def test_load_routing_rules_from_file(tmp_path: Path):
    rules_file = tmp_path / "routing_rules.json"
    rules_file.write_text(
        json.dumps(
            {
                "rules": [
                    {"name": "sample", "intent": "deep_reason", "prefer": ["openai:gpt-4o-mini"]}
                ]
            }
        ),
        encoding="utf-8",
    )
    config = load_routing_rules(str(rules_file))
    assert config.source_path == str(rules_file)
    assert config.rules[0].name == "sample"


def test_validate_routing_rules_warns_in_basic_mode():
    config = load_routing_rules_from_data(
        {"rules": [{"name": "bad", "intent": "code", "prefer": ["missing:model"]}]}
    )
    validate_routing_rules(
        config,
        mode="basic",
        price_table_keys={"openai:gpt-4o-mini"},
        available_providers={"openai"},
    )


def test_validate_routing_rules_fails_in_production():
    config = load_routing_rules_from_data(
        {"rules": [{"name": "bad", "intent": "code", "prefer": ["missing:model"]}]}
    )
    with pytest.raises(RouterConfigurationError):
        validate_routing_rules(
            config,
            mode="production",
            price_table_keys={"openai:gpt-4o-mini"},
            available_providers={"openai"},
        )


def test_empty_path_returns_empty_config():
    config = load_routing_rules(None)
    assert isinstance(config, RoutingRulesConfig)
    assert config.rules == []
