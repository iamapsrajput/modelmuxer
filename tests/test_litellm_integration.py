#!/usr/bin/env python3
"""
ModelMuxer + LiteLLM Integration Test Script
Tests the complete integration between ModelMuxer and LiteLLM proxy.
"""

import asyncio
import json
import sys
from typing import Dict, Any
import httpx
import time

# Test configuration
MODELMUXER_BASE_URL = "http://localhost:8000"
LITELLM_BASE_URL = "https://litellm.int.thomsonreuters.com"
TEST_API_KEY = "sk-test-claude-dev"  # From .env file


class IntegrationTester:
    def __init__(self):
        self.modelmuxer_url = MODELMUXER_BASE_URL
        self.litellm_url = LITELLM_BASE_URL
        self.api_key = TEST_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def test_health_checks(self) -> Dict[str, bool]:
        """Test health endpoints for both services."""
        print("🏥 Testing health checks...")
        results = {}

        async with httpx.AsyncClient() as client:
            # Test ModelMuxer health
            try:
                response = await client.get(f"{self.modelmuxer_url}/health", timeout=10)
                results["modelmuxer_health"] = response.status_code == 200
                print(f"   ✅ ModelMuxer health: {response.status_code}")
            except Exception as e:
                results["modelmuxer_health"] = False
                print(f"   ❌ ModelMuxer health failed: {e}")

            # Test LiteLLM health
            try:
                response = await client.get(f"{self.litellm_url}/health", timeout=10)
                results["litellm_health"] = response.status_code == 200
                print(f"   ✅ LiteLLM health: {response.status_code}")
            except Exception as e:
                results["litellm_health"] = False
                print(f"   ❌ LiteLLM health failed: {e}")

        return results

    async def test_available_models(self) -> Dict[str, Any]:
        """Test available models through ModelMuxer."""
        print("🎯 Testing available models...")

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.modelmuxer_url}/v1/models", headers=self.headers, timeout=15
                )

                if response.status_code == 200:
                    models = response.json()
                    print(f"   ✅ Found {len(models.get('data', []))} available models")
                    for model in models.get("data", [])[:5]:  # Show first 5
                        print(f"      - {model.get('id', 'Unknown')}")
                    return {"success": True, "models": models}
                else:
                    print(f"   ❌ Models endpoint failed: {response.status_code}")
                    print(f"      Response: {response.text}")
                    return {"success": False, "error": response.text}

            except Exception as e:
                print(f"   ❌ Models test failed: {e}")
                return {"success": False, "error": str(e)}

    async def test_simple_completion(self) -> Dict[str, Any]:
        """Test simple chat completion."""
        print("💬 Testing simple chat completion...")

        payload = {
            "messages": [{"role": "user", "content": "Hello! Just say 'Hi' back."}],
            "max_tokens": 50,
            "temperature": 0.7,
        }

        async with httpx.AsyncClient() as client:
            try:
                start_time = time.time()
                response = await client.post(
                    f"{self.modelmuxer_url}/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30,
                )
                end_time = time.time()

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    metadata = result.get("router_metadata", {})

                    print(f"   ✅ Completion successful ({end_time - start_time:.2f}s)")
                    print(f"      Model: {metadata.get('selected_model', 'Unknown')}")
                    print(f"      Provider: {metadata.get('selected_provider', 'Unknown')}")
                    print(f"      Response: {content[:100]}...")
                    print(f"      Cost: ${metadata.get('estimated_cost', 0):.6f}")

                    return {"success": True, "response": result}
                else:
                    print(f"   ❌ Completion failed: {response.status_code}")
                    print(f"      Response: {response.text}")
                    return {"success": False, "error": response.text}

            except Exception as e:
                print(f"   ❌ Completion test failed: {e}")
                return {"success": False, "error": str(e)}

    async def test_code_completion(self) -> Dict[str, Any]:
        """Test code completion (should prefer Claude)."""
        print("👨‍💻 Testing code completion...")

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Write a simple Python function to calculate fibonacci numbers",
                }
            ],
            "max_tokens": 200,
            "temperature": 0.1,
        }

        async with httpx.AsyncClient() as client:
            try:
                start_time = time.time()
                response = await client.post(
                    f"{self.modelmuxer_url}/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30,
                )
                end_time = time.time()

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    metadata = result.get("router_metadata", {})

                    print(f"   ✅ Code completion successful ({end_time - start_time:.2f}s)")
                    print(f"      Model: {metadata.get('selected_model', 'Unknown')}")
                    print(f"      Provider: {metadata.get('selected_provider', 'Unknown')}")
                    print(f"      Routing strategy: {metadata.get('routing_strategy', 'Unknown')}")
                    print(f"      Cost: ${metadata.get('estimated_cost', 0):.6f}")

                    # Check if Claude was preferred for code
                    used_claude = "claude" in metadata.get("selected_model", "").lower()
                    if used_claude:
                        print("      ✅ Claude was correctly chosen for code task")
                    else:
                        print("      ⚠️ Claude was not chosen (may be due to routing logic)")

                    return {"success": True, "response": result, "used_claude": used_claude}
                else:
                    print(f"   ❌ Code completion failed: {response.status_code}")
                    return {"success": False, "error": response.text}

            except Exception as e:
                print(f"   ❌ Code completion test failed: {e}")
                return {"success": False, "error": str(e)}

    async def test_enhanced_features(self) -> Dict[str, Any]:
        """Test enhanced ModelMuxer features."""
        print("🚀 Testing enhanced features...")

        results = {}

        async with httpx.AsyncClient() as client:
            # Test analytics endpoint
            try:
                response = await client.get(
                    f"{self.modelmuxer_url}/v1/analytics/costs", headers=self.headers, timeout=10
                )
                results["analytics"] = response.status_code == 200
                print(f"   ✅ Analytics endpoint: {response.status_code}")
            except Exception as e:
                results["analytics"] = False
                print(f"   ❌ Analytics failed: {e}")

            # Test metrics endpoint
            try:
                response = await client.get(
                    f"{self.modelmuxer_url}/metrics", headers=self.headers, timeout=10
                )
                results["metrics"] = response.status_code == 200
                print(f"   ✅ Metrics endpoint: {response.status_code}")
            except Exception as e:
                results["metrics"] = False
                print(f"   ❌ Metrics failed: {e}")

        return results

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        print("🧪 Starting ModelMuxer + LiteLLM Integration Tests\n")

        all_results = {
            "timestamp": time.time(),
            "health_checks": await self.test_health_checks(),
            "models": await self.test_available_models(),
            "simple_completion": await self.test_simple_completion(),
            "code_completion": await self.test_code_completion(),
            "enhanced_features": await self.test_enhanced_features(),
        }

        # Summary
        print("\n📊 Test Summary:")
        total_tests = 0
        passed_tests = 0

        for category, results in all_results.items():
            if category == "timestamp":
                continue

            if isinstance(results, dict):
                for test_name, passed in results.items():
                    total_tests += 1
                    if passed:
                        passed_tests += 1
                        print(f"   ✅ {category}.{test_name}")
                    else:
                        print(f"   ❌ {category}.{test_name}")

        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        print(f"\n🎯 Overall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")

        return all_results


async def main():
    """Main test runner."""
    tester = IntegrationTester()
    results = await tester.run_all_tests()

    # Save results to file
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n📄 Results saved to test_results.json")

    # Exit with appropriate code
    health_ok = all(results["health_checks"].values())
    if not health_ok:
        print("\n⚠️  Services are not healthy. Please check your Docker containers.")
        sys.exit(1)

    print("\n🎉 Integration tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
