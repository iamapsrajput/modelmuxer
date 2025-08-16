#!/usr/bin/env python3
"""
Test script to verify ModelMuxer routing with updated LiteLLM models.
Tests different query types to ensure correct model selection.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"
API_KEY = "sk-test-claude-dev"  # Your API key


def test_request(query_type, content):
    """Test a request and return routing information."""
    print(f"\n{'='*60}")
    print(f"Testing {query_type.upper()} Query")
    print(f"Content: {content[:100]}...")
    print(f"{'='*60}")

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    payload = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 50,
        "stream": False,
    }

    start_time = time.time()

    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions", headers=headers, json=payload, timeout=30
        )

        end_time = time.time()
        response_time = end_time - start_time

        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {response_time:.2f}s")

        if response.status_code == 200:
            data = response.json()

            # Extract model information from response
            model = data.get("model", "Unknown")
            usage = data.get("usage", {})

            print("✅ SUCCESS")
            print(f"Selected Model: {model}")
            print(f"Token Usage: {usage}")

            # Check for routing information in headers or response
            if "x-routing-info" in response.headers:
                print(f"Routing Info: {response.headers['x-routing-info']}")

            # Print first part of response
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"][:200]
                print(f"Response Preview: {content}...")

            return True

        else:
            print("❌ FAILED")
            print(f"Error: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ REQUEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return False


def main():
    print("🚀 Testing ModelMuxer with Updated LiteLLM Models")
    print(f"Base URL: {BASE_URL}")
    print(f"API Key: {API_KEY}")

    # Test cases for different query types
    test_cases = [
        {
            "type": "code",
            "content": "Write a Python function to calculate the factorial of a number using recursion. Include error handling for negative numbers.",
        },
        {
            "type": "complex",
            "content": "Analyze the trade-offs between microservices and monolithic architecture. Provide a detailed comparison covering scalability, performance, deployment complexity, and team organization aspects.",
        },
        {"type": "simple", "content": "What is the capital of France?"},
        {
            "type": "general",
            "content": "Explain the concept of artificial intelligence and its applications in modern technology.",
        },
    ]

    results = []

    for test_case in test_cases:
        success = test_request(test_case["type"], test_case["content"])
        results.append((test_case["type"], success))

        # Small delay between requests
        time.sleep(1)

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    successful = sum(1 for _, success in results if success)
    total = len(results)

    for query_type, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{query_type.upper():10} | {status}")

    print(f"\nOverall: {successful}/{total} tests passed")

    if successful == total:
        print("🎉 All tests passed! ModelMuxer is routing correctly with LiteLLM models.")
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")


if __name__ == "__main__":
    main()
