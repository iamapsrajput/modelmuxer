#!/usr/bin/env python3
# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Comprehensive test script for Enhanced ModelMuxer.

This script tests all the advanced features including routing strategies,
multiple providers, ML-based classification, caching, and monitoring.
"""

import asyncio
import os
import sys
import time
from typing import Any, Dict, List

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

# Test imports
try:
    from app.config.enhanced_config import enhanced_config
    from app.core.exceptions import ModelMuxerError
    from app.main_enhanced import model_muxer
    from app.models import ChatCompletionRequest, ChatMessage

    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class EnhancedModelMuxerTester:
    """Comprehensive tester for Enhanced ModelMuxer."""

    def __init__(self):
        self.test_results = []
        self.failed_tests = []

    def log_test(self, test_name: str, success: bool, message: str = ""):
        """Log test result."""
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")

        self.test_results.append({"test": test_name, "success": success, "message": message})

        if not success:
            self.failed_tests.append(test_name)

    async def test_configuration(self):
        """Test configuration loading."""
        try:
            config = enhanced_config

            # Test basic config
            assert hasattr(config, "host")
            assert hasattr(config, "port")
            assert hasattr(config, "providers")
            assert hasattr(config, "routing")
            assert hasattr(config, "cache")
            assert hasattr(config, "auth")

            self.log_test(
                "Configuration Loading", True, f"Host: {config.host}, Port: {config.port}"
            )

        except Exception as e:
            self.log_test("Configuration Loading", False, str(e))

    async def test_providers(self):
        """Test provider initialization."""
        try:
            providers = model_muxer.providers

            if not providers:
                self.log_test("Provider Initialization", False, "No providers configured")
                return

            provider_names = list(providers.keys())
            self.log_test(
                "Provider Initialization", True, f"Providers: {', '.join(provider_names)}"
            )

            # Test provider health checks
            for name, provider in providers.items():
                try:
                    # Skip health check for now as it requires API keys
                    supported_models = provider.get_supported_models()
                    self.log_test(
                        f"Provider {name} Models", True, f"{len(supported_models)} models"
                    )
                except Exception as e:
                    self.log_test(f"Provider {name} Models", False, str(e))

        except Exception as e:
            self.log_test("Provider Initialization", False, str(e))

    async def test_routing(self):
        """Test routing strategies."""
        try:
            routers = model_muxer.routers

            if not routers:
                self.log_test("Routing Initialization", False, "No routers configured")
                return

            router_names = list(routers.keys())
            self.log_test("Routing Initialization", True, f"Routers: {', '.join(router_names)}")

            # Test routing with sample messages
            test_messages = [
                ChatMessage(role="user", content="Write a Python function to sort a list"),
                ChatMessage(role="user", content="What is the capital of France?"),
                ChatMessage(role="user", content="Explain machine learning in simple terms"),
            ]

            for router_name, router in routers.items():
                try:
                    provider, model, reasoning, confidence = await router.select_provider_and_model(
                        test_messages
                    )
                    self.log_test(
                        f"Router {router_name}",
                        True,
                        f"Selected {provider}/{model} (confidence: {confidence:.2f})",
                    )
                except Exception as e:
                    self.log_test(f"Router {router_name}", False, str(e))

        except Exception as e:
            self.log_test("Routing Initialization", False, str(e))

    async def test_classification(self):
        """Test ML-based classification."""
        try:
            if not model_muxer.classifier:
                self.log_test("Classification", False, "Classifier not initialized")
                return

            test_prompts = [
                "Write a Python function to calculate fibonacci numbers",
                "What is the weather like today?",
                "Analyze the pros and cons of renewable energy",
                "Create a marketing email for our new product",
                "Debug this JavaScript code that's not working",
            ]

            for i, prompt in enumerate(test_prompts):
                try:
                    result = await model_muxer.classifier.classify(prompt)
                    self.log_test(
                        f"Classification Test {i+1}",
                        True,
                        f"Category: {result['category']} (confidence: {result['confidence']:.2f})",
                    )
                except Exception as e:
                    self.log_test(f"Classification Test {i+1}", False, str(e))

        except Exception as e:
            self.log_test("Classification", False, str(e))

    async def test_caching(self):
        """Test caching system."""
        try:
            if not model_muxer.cache:
                self.log_test("Caching", False, "Cache not initialized")
                return

            # Test basic cache operations
            test_key = "test_key"
            test_value = {"message": "Hello, World!", "timestamp": time.time()}

            # Set value
            set_result = await model_muxer.cache.set(test_key, test_value, ttl=60)
            self.log_test("Cache Set", set_result, "Successfully set test value")

            # Get value
            retrieved_value = await model_muxer.cache.get(test_key)
            get_success = retrieved_value == test_value
            self.log_test("Cache Get", get_success, "Successfully retrieved test value")

            # Check existence
            exists = await model_muxer.cache.exists(test_key)
            self.log_test("Cache Exists", exists, "Key exists in cache")

            # Delete value
            delete_result = await model_muxer.cache.delete(test_key)
            self.log_test("Cache Delete", delete_result, "Successfully deleted test value")

            # Verify deletion
            after_delete = await model_muxer.cache.get(test_key)
            delete_verify = after_delete is None
            self.log_test("Cache Delete Verify", delete_verify, "Value no longer exists")

            # Test cache info
            cache_info = await model_muxer.cache.get_cache_info()
            self.log_test("Cache Info", True, f"Status: {cache_info.get('status', 'unknown')}")

        except Exception as e:
            self.log_test("Caching", False, str(e))

    async def test_monitoring(self):
        """Test monitoring and metrics."""
        try:
            if not model_muxer.metrics_collector:
                self.log_test("Monitoring", False, "Metrics collector not initialized")
                return

            # Test metrics recording
            model_muxer.metrics_collector.record_request(
                method="POST",
                endpoint="/v1/chat/completions",
                status_code=200,
                duration=1.5,
                user_id="test_user",
            )

            model_muxer.metrics_collector.record_provider_request(
                provider="test_provider",
                model="test_model",
                status="success",
                duration=1.2,
                input_tokens=100,
                output_tokens=50,
                cost=0.001,
            )

            # Get metrics summary
            stats = model_muxer.metrics_collector.get_summary_stats()
            self.log_test("Metrics Recording", True, f"Total requests: {stats['total_requests']}")

            # Test health checker
            if model_muxer.health_checker:
                health_status = await model_muxer.health_check()
                self.log_test("Health Check", True, f"Status: {health_status['status']}")

        except Exception as e:
            self.log_test("Monitoring", False, str(e))

    async def test_middleware(self):
        """Test middleware components."""
        try:
            # Test authentication middleware
            if model_muxer.auth_middleware:
                self.log_test("Auth Middleware", True, "Authentication middleware initialized")
            else:
                self.log_test("Auth Middleware", False, "Authentication middleware not initialized")

            # Test rate limiting middleware
            if model_muxer.rate_limit_middleware:
                stats = model_muxer.rate_limit_middleware.get_rate_limit_stats()
                self.log_test(
                    "Rate Limit Middleware", True, f"Active users: {stats['active_users']}"
                )
            else:
                self.log_test(
                    "Rate Limit Middleware", False, "Rate limiting middleware not initialized"
                )

            # Test logging middleware
            if model_muxer.logging_middleware:
                metrics = model_muxer.logging_middleware.get_performance_metrics()
                self.log_test(
                    "Logging Middleware", True, f"Total requests: {metrics['total_requests']}"
                )
            else:
                self.log_test("Logging Middleware", False, "Logging middleware not initialized")

        except Exception as e:
            self.log_test("Middleware", False, str(e))

    async def test_integration(self):
        """Test end-to-end integration."""
        try:
            # Create a test request
            test_request = ChatCompletionRequest(
                messages=[
                    ChatMessage(
                        role="user",
                        content="Hello, this is a test message for integration testing.",
                    )
                ],
                max_tokens=10,
                temperature=0.7,
            )

            test_user = {"user_id": "test_user", "role": "user", "auth_method": "test"}

            # Test routing without actual API call
            try:
                provider_name, model, reasoning, confidence = await model_muxer.route_request(
                    messages=test_request.messages, user_id=test_user["user_id"]
                )

                self.log_test(
                    "Integration Routing",
                    True,
                    f"Routed to {provider_name}/{model} (confidence: {confidence:.2f})",
                )

            except Exception as e:
                self.log_test("Integration Routing", False, str(e))

        except Exception as e:
            self.log_test("Integration", False, str(e))

    async def run_all_tests(self):
        """Run all tests."""
        print("🚀 Starting Enhanced ModelMuxer Tests\n")

        await self.test_configuration()
        await self.test_providers()
        await self.test_routing()
        await self.test_classification()
        await self.test_caching()
        await self.test_monitoring()
        await self.test_middleware()
        await self.test_integration()

        # Print summary
        print(f"\n📊 Test Summary:")
        print(f"Total tests: {len(self.test_results)}")
        print(f"Passed: {len(self.test_results) - len(self.failed_tests)}")
        print(f"Failed: {len(self.failed_tests)}")

        if self.failed_tests:
            print(f"\n❌ Failed tests: {', '.join(self.failed_tests)}")
            return False
        else:
            print("\n🎉 All tests passed!")
            return True


async def main():
    """Main test function."""
    tester = EnhancedModelMuxerTester()
    success = await tester.run_all_tests()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
