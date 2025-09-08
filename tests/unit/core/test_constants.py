# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""
Test constants for ModelMuxer tests.

⚠️  SECURITY NOTE: These are clearly test values and not production credentials.
⚠️  All values are safe for testing environments and do not contain real secrets.
⚠️  SNYK: This file contains intentional test data, not production credentials.
"""

import os

# Test API keys - clearly for testing only
# SNYK: False positive - these are test constants with obvious test patterns
TEST_API_KEY_1 = os.getenv("TEST_API_KEY_1", "test-api-key-001-NOT-REAL")
TEST_API_KEY_2 = os.getenv("TEST_API_KEY_2", "test-api-key-002-NOT-REAL")
TEST_API_KEY_SAMPLE = os.getenv("TEST_API_KEY_SAMPLE", "test-sample-key-123-FAKE")
TEST_API_KEY_INVALID = "invalid-test-key-OBVIOUSLY-FAKE"


def get_test_api_keys() -> list[str]:
    """Get list of test API keys."""
    return [TEST_API_KEY_1, TEST_API_KEY_2]


# Test configuration values
# SNYK: False positive - JWT secret is clearly marked as test-only with obvious pattern
TEST_JWT_SECRET = "test-jwt-secret-FOR-TESTING-ONLY-not-for-production-FAKE"
TEST_DATABASE_URL = "sqlite:///test_db_NOT_REAL.db"
TEST_REDIS_URL = "redis://localhost:6379/15"  # Use DB 15 for tests

# Test user data
TEST_USER_ID = "test-user-123-FAKE-ID"
TEST_ORG_ID = "test-org-456"
