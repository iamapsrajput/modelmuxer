#!/usr/bin/env python3
# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
import json

import requests


def test_chat_completion():
    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer sk-test-claude-dev"}

    payload = {
        "model": "anthropic/claude-3-5-sonnet-latest",
        "messages": [
            {
                "role": "user",
                "content": "Hello! This is a test message. Please respond with a simple greeting.",
            }
        ],
        "max_tokens": 100,
    }

    try:
        print("Testing ModelMuxer API...")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("SUCCESS! Response received:")
            print(f"Model: {result.get('model')}")
            print(f"Content: {result['choices'][0]['message']['content']}")
            print(f"Total tokens: {result['usage']['total_tokens']}")
        else:
            print(f"ERROR: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    test_chat_completion()
