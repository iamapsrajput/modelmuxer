#!/usr/bin/env python3
# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

import requests
import json
import time

headers = {"Content-Type": "application/json", "Authorization": "Bearer sk-test-claude-dev"}

print("=== Testing Enhanced Budget Management ===")
print()

# Make a few API calls to accumulate usage
print("Making test API calls to accumulate usage...")
for i in range(3):
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        headers=headers,
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": f"Hello world {i + 1}"}],
            "max_tokens": 30,
        },
    )
    print(f"Request {i + 1}: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        cost = data.get("router_metadata", {}).get("estimated_cost", 0)
        tokens = data.get("usage", {}).get("total_tokens", 0)
        print(f"  Cost: ${cost:.6f}, Tokens: {tokens}")
    time.sleep(0.5)

print()
print("=== Updated Budget Status ===")
response = requests.get("http://localhost:8000/v1/analytics/budgets", headers=headers)

if response.status_code == 200:
    data = response.json()
    for budget in data["budgets"]:
        print(f"Budget Type: {budget['budget_type']}")
        print(f"Budget Limit: ${budget['budget_limit']:.2f}")
        print(f"Current Usage: ${budget['current_usage']:.6f}")
        print(f"Usage Percentage: {budget['usage_percentage']:.1f}%")
        print(f"Remaining Budget: ${budget['remaining_budget']:.6f}")
        print(f"Alerts: {len(budget['alerts'])}")
else:
    print(f"Error: {response.status_code} - {response.text}")

print()
print("=== Setting Provider-Specific Budget ===")
# Set a low budget for OpenAI provider specifically
provider_budget = {
    "budget_type": "daily",
    "budget_limit": 0.01,  # Very low budget
    "provider": "openai",
    "alert_thresholds": [25.0, 50.0, 75.0, 90.0],
}

response = requests.post("http://localhost:8000/v1/analytics/budgets", headers=headers, json=provider_budget)

if response.status_code == 200:
    print("Provider-specific budget set successfully!")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Error: {response.status_code} - {response.text}")

print()
print("=== Updated Budget Status with Provider Budget ===")
response = requests.get("http://localhost:8000/v1/analytics/budgets", headers=headers)

if response.status_code == 200:
    data = response.json()
    print(f"Total budgets: {data['total_budgets']}")
    for i, budget in enumerate(data["budgets"]):
        provider_info = f" (Provider: {budget['provider']})" if budget["provider"] else " (All Providers)"
        print(f"Budget {i + 1}: {budget['budget_type']}{provider_info}")
        print(f"  Limit: ${budget['budget_limit']:.2f}")
        print(f"  Usage: ${budget['current_usage']:.6f} ({budget['usage_percentage']:.1f}%)")
        print(f"  Alerts: {len(budget['alerts'])}")
        if budget["alerts"]:
            for alert in budget["alerts"]:
                print(f"    - {alert['type'].upper()}: {alert['message']}")
else:
    print(f"Error: {response.status_code} - {response.text}")

print()
print("=== System Metrics ===")
response = requests.get("http://localhost:8000/metrics", headers=headers)

if response.status_code == 200:
    data = response.json()
    print(f"Total Requests: {data['total_requests']}")
    print(f"Total Cost: ${data['total_cost']:.6f}")
    print(f"Active Users: {data['active_users']}")
    print(f"Provider Usage: {data['provider_usage']}")
    print(f"Average Response Time: {data['average_response_time']:.2f}ms")
else:
    print(f"Error: {response.status_code} - {response.text}")
