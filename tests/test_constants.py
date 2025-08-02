# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""
Test constants for ModelMuxer tests.

These are clearly test values and not production credentials.
"""

import os

# Test API keys - clearly for testing only
TEST_API_KEY_1 = os.getenv("TEST_API_KEY_1", "test-api-key-001")
TEST_API_KEY_2 = os.getenv("TEST_API_KEY_2", "test-api-key-002")
TEST_API_KEY_SAMPLE = os.getenv("TEST_API_KEY_SAMPLE", "test-sample-key-123")
TEST_API_KEY_INVALID = "invalid-test-key"


def get_test_api_keys() -> list[str]:
    """Get list of test API keys."""
    return [TEST_API_KEY_1, TEST_API_KEY_2]


# Test configuration values
TEST_JWT_SECRET = "test-jwt-secret-for-testing-only-not-for-production"
TEST_DATABASE_URL = "sqlite:///test.db"
TEST_REDIS_URL = "redis://localhost:6379/15"  # Use DB 15 for tests

# Test user data
TEST_USER_ID = "test-user-123"
TEST_ORG_ID = "test-org-456"
