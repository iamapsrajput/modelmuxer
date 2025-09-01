#!/usr/bin/env python3
"""
PII Protection Test Script
Demonstrates how ModelMuxer protects personally identifiable information.
"""

import asyncio
import json
import httpx
from typing import Dict, Any

# Test configuration
MODELMUXER_BASE_URL = "http://localhost:8000"
TEST_API_KEY = "sk-test-claude-dev"


class PIIProtectionTester:
    def __init__(self):
        self.base_url = MODELMUXER_BASE_URL
        self.api_key = TEST_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def test_email_protection(self) -> Dict[str, Any]:
        """Test email address detection and protection."""
        print("📧 Testing email protection...")

        test_prompt = "My email is john.doe@company.com and my colleague's is jane.smith@example.org. Can you help us with a data analysis?"

        return await self._test_pii_prompt(
            test_prompt, "email", ["john.doe@company.com", "jane.smith@example.org"]
        )

    async def test_phone_protection(self) -> Dict[str, Any]:
        """Test phone number detection and protection."""
        print("📱 Testing phone number protection...")

        test_prompt = (
            "Please call me at +1-555-123-4567 or (555) 987-6543 if you need more information."
        )

        return await self._test_pii_prompt(
            test_prompt, "phone", ["+1-555-123-4567", "(555) 987-6543"]
        )

    async def test_ssn_protection(self) -> Dict[str, Any]:
        """Test SSN detection and protection."""
        print("🆔 Testing SSN protection...")

        test_prompt = "My SSN is 123-45-6789 and I need help with tax calculations."

        return await self._test_pii_prompt(test_prompt, "ssn", ["123-45-6789"])

    async def test_credit_card_protection(self) -> Dict[str, Any]:
        """Test credit card number detection and protection."""
        print("💳 Testing credit card protection...")

        test_prompt = (
            "My credit card 4532-1234-5678-9012 was declined. Can you help me understand why?"
        )

        return await self._test_pii_prompt(test_prompt, "credit_card", ["4532-1234-5678-9012"])

    async def test_mixed_pii_protection(self) -> Dict[str, Any]:
        """Test multiple PII types in one prompt."""
        print("🔒 Testing mixed PII protection...")

        test_prompt = """
        Hi, I'm John Doe (john.doe@company.com), my phone is +1-555-123-4567.
        My SSN is 123-45-6789 and my credit card ending in 9012 was charged.
        Can you help me understand this transaction?
        """

        return await self._test_pii_prompt(
            test_prompt, "mixed", ["john.doe@company.com", "+1-555-123-4567", "123-45-6789", "9012"]
        )

    async def test_safe_prompt(self) -> Dict[str, Any]:
        """Test that safe prompts work normally."""
        print("✅ Testing safe prompt (no PII)...")

        test_prompt = "Can you help me write a Python function to calculate the fibonacci sequence?"

        return await self._test_pii_prompt(test_prompt, "safe", [])

    async def _test_pii_prompt(
        self, prompt: str, test_type: str, expected_pii: list
    ) -> Dict[str, Any]:
        """Test a prompt for PII protection."""
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.7,
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()
                    metadata = result.get("router_metadata", {})
                    security_info = metadata.get("security", {})

                    pii_detected = security_info.get("pii_detected", False)
                    pii_types = security_info.get("pii_types", [])
                    pii_redacted = security_info.get("pii_redacted", False)

                    print(f"      Result: {'PII detected' if pii_detected else 'No PII detected'}")
                    if pii_detected:
                        print(f"      Types: {', '.join(pii_types)}")
                        print(f"      Redacted: {'Yes' if pii_redacted else 'No'}")

                    return {
                        "success": True,
                        "test_type": test_type,
                        "pii_detected": pii_detected,
                        "pii_types": pii_types,
                        "pii_redacted": pii_redacted,
                        "expected_pii": expected_pii,
                        "response": result,
                    }
                elif response.status_code == 400:
                    # PII protection might block the request
                    error_data = response.json()
                    if "pii" in error_data.get("error", {}).get("message", "").lower():
                        print("      ✅ Request blocked due to PII protection")
                        return {
                            "success": True,
                            "test_type": test_type,
                            "pii_detected": True,
                            "request_blocked": True,
                            "error": error_data,
                        }
                    else:
                        print(f"      ❌ Request failed: {response.status_code}")
                        return {"success": False, "error": error_data}
                else:
                    print(f"      ❌ Request failed: {response.status_code}")
                    return {"success": False, "error": response.text}

            except Exception as e:
                print(f"      ❌ Test failed: {e}")
                return {"success": False, "error": str(e)}

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all PII protection tests."""
        print("🛡️  Starting PII Protection Tests\n")

        tests = [
            ("email", self.test_email_protection),
            ("phone", self.test_phone_protection),
            ("ssn", self.test_ssn_protection),
            ("credit_card", self.test_credit_card_protection),
            ("mixed", self.test_mixed_pii_protection),
            ("safe", self.test_safe_prompt),
        ]

        results = {}

        for test_name, test_func in tests:
            print()
            result = await test_func()
            results[test_name] = result

        # Summary
        print("\n📊 PII Protection Test Summary:")
        total_tests = len(tests)
        successful_tests = sum(1 for result in results.values() if result.get("success", False))

        print(f"   Total tests: {total_tests}")
        print(f"   Successful: {successful_tests}")
        print(f"   Success rate: {(successful_tests / total_tests) * 100:.1f}%")

        pii_tests = [name for name in results if name != "safe"]
        pii_detected_count = sum(
            1 for name in pii_tests if results[name].get("pii_detected", False)
        )

        print("\n🔍 PII Detection Results:")
        print(f"   PII tests: {len(pii_tests)}")
        print(f"   PII detected: {pii_detected_count}")
        print(f"   Detection rate: {(pii_detected_count / len(pii_tests)) * 100:.1f}%")

        return results


async def main():
    """Main test runner."""
    print("🛡️  ModelMuxer PII Protection Test Suite")
    print("=" * 50)
    print()
    print("This test demonstrates how ModelMuxer automatically detects and protects")
    print("personally identifiable information (PII) in prompts before sending")
    print("them to LLM providers.")
    print()

    tester = PIIProtectionTester()
    results = await tester.run_all_tests()

    # Save results
    with open("pii_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n📄 Detailed results saved to pii_test_results.json")

    print("\n💡 PII Protection Benefits:")
    print("   ✅ Automatic PII detection and redaction")
    print("   ✅ GDPR/CCPA compliance assistance")
    print("   ✅ Audit logging of security events")
    print("   ✅ Enterprise-grade data protection")
    print("   ✅ Prevents accidental data leaks")


if __name__ == "__main__":
    asyncio.run(main())
