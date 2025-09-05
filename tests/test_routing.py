#!/usr/bin/env python3
import json

import requests

jwt_token = "sk-test-claude-dev"
base_headers = {"Content-Type": "application/json", "Authorization": f"Bearer {jwt_token}"}

# Test different types of queries
test_cases = [
    {"name": "Simple Query", "messages": [{"role": "user", "content": "What is Python?"}]},
    {
        "name": "Code Query",
        "messages": [
            {
                "role": "user",
                "content": "Write a Python function to calculate fibonacci numbers using recursion",
            }
        ],
    },
    {
        "name": "Complex Analysis",
        "messages": [
            {
                "role": "user",
                "content": "Analyze the trade-offs between different sorting algorithms and explain their performance characteristics in detail",
            }
        ],
    },
    {
        "name": "General Query",
        "messages": [
            {
                "role": "user",
                "content": "How can I improve my productivity while working from home?",
            }
        ],
    },
]

for case in test_cases:
    print()
    print(f'=== {case["name"]} ===')
    print(f'Query: {case["messages"][0]["content"][:50]}...')

    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        headers=base_headers,
        json={"model": "gpt-3.5-turbo", "messages": case["messages"], "max_tokens": 50},
    )

    if response.status_code == 200:
        data = response.json()
        metadata = data.get("router_metadata", {})
        print(f'Selected Provider: {metadata.get("selected_provider", "N/A")}')
        print(f'Selected Model: {metadata.get("selected_model", "N/A")}')
        print(f'Routing Reason: {metadata.get("routing_reason", "N/A")}')
        print(f'Cost: ${metadata.get("estimated_cost", 0):.6f}')
        print(f'Tokens: {data.get("usage", {}).get("total_tokens", "N/A")}')
    else:
        print(f"Error: {response.status_code} - {response.text}")
