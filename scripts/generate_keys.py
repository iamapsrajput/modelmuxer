#!/usr/bin/env python3
"""
ModelMuxer Security Key Generator

Generates cryptographically secure keys for ModelMuxer configuration.
Use this script to generate JWT secrets, encryption keys, and API keys.

Usage:
    python scripts/generate_keys.py

Outputs environment variables that can be added to your .env file.
"""

import secrets
import string


def generate_secure_key(length=32):
    """Generate a cryptographically secure random key."""
    return secrets.token_urlsafe(length)


def generate_hex_key(length=32):
    """Generate a cryptographically secure hex key."""
    return secrets.token_hex(length)


def generate_alphanumeric_key(length=32):
    """Generate a cryptographically secure alphanumeric key."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def main():
    print("🔐 ModelMuxer Security Key Generator")
    print("=" * 50)
    print()

    # Generate JWT Secret Key
    jwt_key = generate_secure_key(32)
    print(f"JWT_SECRET_KEY={jwt_key}")
    print()

    # Generate Encryption Key
    encryption_key = generate_secure_key(32)
    print(f"PII_ENCRYPTION_KEY={encryption_key}")
    print()

    # Generate additional keys that might be useful
    print("Additional keys (if needed):")
    print("-" * 30)
    session_key = generate_secure_key(24)
    print(f"SESSION_SECRET={session_key}")

    api_key = generate_alphanumeric_key(40)
    print(f"INTERNAL_API_KEY=sk-{api_key}")

    webhook_secret = generate_hex_key(16)
    print(f"WEBHOOK_SECRET={webhook_secret}")

    print()
    print("⚠️  Security Notes:")
    print("- Keep these keys secure and never commit them to version control")
    print("- Use different keys for different environments (dev/staging/prod)")
    print("- Rotate keys regularly in production environments")
    print("- Store production keys in secure key management systems")


if __name__ == "__main__":
    main()
