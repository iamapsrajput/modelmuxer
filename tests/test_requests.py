#!/usr/bin/env python3
# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Test script for the ModelMuxer LLM Router API.

This script tests various scenarios to ensure the router works correctly
and routes requests to appropriate providers based on prompt characteristics.
"""

import asyncio
import json
import time
from typing import Any

import httpx


class RouterTester:
    """Test suite for the LLM Router API."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "sk-test-key-1"):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.client = httpx.AsyncClient(timeout=60.0)

    async def __aenter__(self) -> 'RouterTester':
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.client.aclose()

    def create_test_request(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Create a test request payload."""
        return {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 150),
            "temperature": kwargs.get("temperature", 0.7),
            **kwargs,
        }

    async def test_health_check(self) -> bool:
        """Test the health check endpoint."""
        print("🔍 Testing health check...")
        try:
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data['status']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False

    async def test_authentication(self) -> bool:
        """Test authentication with valid and invalid keys."""
        print("🔍 Testing authentication...")

        # Test with valid key
        try:
            response = await self.client.get(f"{self.base_url}/providers", headers=self.headers)
            if response.status_code == 200:
                print("✅ Valid API key accepted")
            else:
                print(f"❌ Valid API key rejected: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Authentication test error: {e}")
            return False

        # Test with invalid key
        try:
            invalid_headers = {"Authorization": "Bearer invalid-key"}
            response = await self.client.get(f"{self.base_url}/providers", headers=invalid_headers)
            if response.status_code == 401:
                print("✅ Invalid API key rejected")
                return True
            else:
                print(f"❌ Invalid API key accepted: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Invalid key test error: {e}")
            return False

    async def test_routing_scenarios(self) -> bool:
        """Test different routing scenarios."""
        print("🔍 Testing routing scenarios...")

        test_cases = [
            {
                "name": "Simple Question",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "expected_provider": "mistral",  # Should route to cheapest for simple queries
                "description": "Short, simple math question",
            },
            {
                "name": "Code Generation",
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a Python function to sort a list using quicksort algorithm",
                    }
                ],
                "expected_provider": "openai",  # Should route to GPT-4o for code
                "description": "Code generation request",
            },
            {
                "name": "Complex Analysis",
                "messages": [
                    {
                        "role": "user",
                        "content": "Analyze the time complexity of merge sort algorithm and explain the trade-offs compared to quicksort. Provide a detailed comparison.",
                    }
                ],
                "expected_provider": "openai",  # Should route to premium model for analysis
                "description": "Complex algorithmic analysis",
            },
            {
                "name": "Code Review",
                "messages": [
                    {
                        "role": "user",
                        "content": "Review this Python code and suggest improvements:\n\n```python\ndef bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n```",
                    }
                ],
                "expected_provider": "openai",  # Code-related, should use premium model
                "description": "Code review with code block",
            },
            {
                "name": "General Conversation",
                "messages": [
                    {
                        "role": "user",
                        "content": "Tell me about the weather patterns in tropical regions.",
                    }
                ],
                "expected_provider": "openai",  # General knowledge, balanced model
                "description": "General knowledge question",
            },
        ]

        success_count = 0
        for test_case in test_cases:
            print(f"\n📝 Testing: {test_case['name']}")
            print(f"   Description: {test_case['description']}")

            try:
                request_payload = self.create_test_request(test_case["messages"])

                start_time = time.time()
                response = await self.client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self.headers,
                    json=request_payload,
                )
                response_time = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    selected_provider = data.get("router_metadata", {}).get(
                        "selected_provider", "unknown"
                    )
                    selected_model = data.get("router_metadata", {}).get(
                        "selected_model", "unknown"
                    )
                    routing_reason = data.get("router_metadata", {}).get(
                        "routing_reason", "unknown"
                    )
                    estimated_cost = data.get("router_metadata", {}).get("estimated_cost", 0)

                    print(f"   ✅ Response received ({response_time:.0f}ms)")
                    print(f"   🎯 Routed to: {selected_provider}/{selected_model}")
                    print(f"   💰 Estimated cost: ${estimated_cost:.6f}")
                    print(f"   🧠 Reasoning: {routing_reason}")

                    # Check if routing matches expectation (flexible check)
                    if selected_provider == test_case["expected_provider"]:
                        print("   ✅ Routing as expected")
                    else:
                        print(
                            f"   ⚠️  Routing differs from expected ({test_case['expected_provider']})"
                        )

                    success_count += 1
                else:
                    print(f"   ❌ Request failed: {response.status_code}")
                    print(f"   Error: {response.text}")

            except Exception as e:
                print(f"   ❌ Test error: {e}")

        print(f"\n📊 Routing tests: {success_count}/{len(test_cases)} passed")
        return success_count == len(test_cases)

    async def test_streaming(self) -> bool:
        """Test streaming responses."""
        print("🔍 Testing streaming...")

        try:
            request_payload = self.create_test_request(
                [{"role": "user", "content": "Count from 1 to 5"}], stream=True
            )

            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=request_payload,
            ) as response:
                if response.status_code == 200:
                    chunks_received = 0
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                json.loads(data)  # Validate JSON format
                                chunks_received += 1
                            except json.JSONDecodeError:
                                continue

                    print(f"✅ Streaming test passed ({chunks_received} chunks received)")
                    return True
                else:
                    print(f"❌ Streaming test failed: {response.status_code}")
                    return False

        except Exception as e:
            print(f"❌ Streaming test error: {e}")
            return False

    async def test_metrics_endpoints(self) -> bool:
        """Test metrics and stats endpoints."""
        print("🔍 Testing metrics endpoints...")

        try:
            # Test system metrics
            response = await self.client.get(f"{self.base_url}/metrics", headers=self.headers)
            if response.status_code == 200:
                metrics = response.json()
                print(
                    f"✅ System metrics: {metrics['total_requests']} requests, ${metrics['total_cost']:.6f} total cost"
                )
            else:
                print(f"❌ Metrics endpoint failed: {response.status_code}")
                return False

            # Test user stats
            response = await self.client.get(f"{self.base_url}/user/stats", headers=self.headers)
            if response.status_code == 200:
                stats = response.json()
                print(
                    f"✅ User stats: {stats['total_requests']} requests, ${stats['total_cost']:.6f} spent"
                )
                return True
            else:
                print(f"❌ User stats endpoint failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Metrics test error: {e}")
            return False

    async def run_all_tests(self) -> bool:
        """Run all tests."""
        print("🚀 Starting ModelMuxer API Tests\n")

        tests = [
            ("Health Check", self.test_health_check),
            ("Authentication", self.test_authentication),
            ("Routing Scenarios", self.test_routing_scenarios),
            ("Streaming", self.test_streaming),
            ("Metrics Endpoints", self.test_metrics_endpoints),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            print(f"\n{'='*50}")
            print(f"🧪 {test_name}")
            print("=" * 50)

            try:
                if await test_func():
                    passed += 1
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")
            except Exception as e:
                print(f"❌ {test_name} ERROR: {e}")

        print(f"\n{'='*50}")
        print("📊 TEST SUMMARY")
        print("=" * 50)
        print(f"Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")

        if passed == total:
            print("🎉 All tests passed!")
            return True
        else:
            print("⚠️  Some tests failed. Check the logs above.")
            return False


async def main() -> int:
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test the ModelMuxer LLM Router API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--api-key", default="sk-test-key-1", help="API key to use for testing")

    args = parser.parse_args()

    async with RouterTester(args.url, args.api_key) as tester:
        success = await tester.run_all_tests()
        return 0 if success else 1


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
