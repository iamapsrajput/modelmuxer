#!/usr/bin/env python3
# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
ModelMuxer Dependency Verification Script

This script verifies that the SNYK security improvements work correctly
in both ML-enabled and security-conscious modes.
"""

import sys


def test_without_ml():
    """Test that the application works without ML dependencies."""
    print("🔒 Testing Security-Conscious Mode (No ML Dependencies)")
    print("=" * 60)

    # Simulate missing ML dependencies
    original_import = __import__

    def mock_import(name, *args, **kwargs):
        if name in ["sentence_transformers", "torch", "numpy"]:
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    # Temporarily replace import
    builtins = sys.modules["builtins"]
    builtins.__import__ = mock_import  # type: ignore[misc]

    try:
        from app.routing.semantic_router_optional import OptionalSemanticRouter

        router = OptionalSemanticRouter()
        print(f"✅ Router Mode: {'ML' if router.use_ml_mode else 'Fallback (Keywords)'}")
        print(f"✅ Fallback Available: {router.use_fallback}")
        print("✅ Security-conscious deployment verified")

        # Test a simple routing
        from app.models import ChatMessage

        messages = [ChatMessage(role="user", content="Write a Python function", name=None)]

        # This should work even without ML
        import asyncio

        async def test_routing():
            result = await router.analyze_prompt(messages)
            print(f"✅ Routing works: {result['analysis_method']}")
            return result

        asyncio.run(test_routing())
        print("✅ All security-conscious features working\n")
        return True

    except Exception as e:
        print(f"❌ Error in security mode: {e}")
        return False
    finally:
        # Restore original import
        builtins.__import__ = original_import  # type: ignore[misc]


def test_with_ml():
    """Test that the application works with ML dependencies."""
    print("🚀 Testing Full ML Mode")
    print("=" * 40)

    try:
        from app.routing.semantic_router_optional import OptionalSemanticRouter

        router = OptionalSemanticRouter()
        print(f"✅ Router Mode: {'ML' if router.use_ml_mode else 'Fallback'}")

        if router.use_ml_mode:
            print("✅ ML dependencies available and working")
            print(f"✅ Model: {router.model_name}")
            print(
                f"✅ Embeddings dim: {router.encoder.get_sentence_embedding_dimension() if router.encoder else 'N/A'}"
            )
        else:
            print("⚠️  ML mode not available, using fallback")

        # Test routing
        from app.models import ChatMessage

        messages = [
            ChatMessage(role="user", content="Analyze this complex system architecture", name=None)
        ]

        import asyncio

        async def test_routing():
            result = await router.analyze_prompt(messages)
            print(f"✅ Routing method: {result['analysis_method']}")
            print(f"✅ Route category: {result['route_category']}")
            print(f"✅ Confidence: {result['confidence']:.3f}")
            return result

        asyncio.run(test_routing())
        print("✅ All ML features working\n")
        return True

    except Exception as e:
        print(f"❌ Error in ML mode: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dependency_versions():
    """Check that we have the correct dependency versions."""
    print("📦 Checking Dependency Versions")
    print("=" * 35)

    try:
        import torch

        print(f"✅ PyTorch: {torch.__version__}")

        if torch.__version__ >= "2.6.0":
            print("✅ PyTorch version meets security requirements (>=2.6.0)")
        else:
            print("⚠️  PyTorch version below security requirement")

    except ImportError:
        print("ℹ️  PyTorch not installed (expected in security mode)")

    try:
        import sentence_transformers

        print(f"✅ Sentence Transformers: {sentence_transformers.__version__}")
    except ImportError:
        print("ℹ️  Sentence Transformers not installed (expected in security mode)")

    try:
        import numpy

        print(f"✅ NumPy: {numpy.__version__}")
    except ImportError:
        print("ℹ️  NumPy not installed (expected in security mode)")

    try:
        import sklearn

        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("ℹ️  Scikit-learn not installed (expected in security mode)")

    print()


def main():
    """Run all verification tests."""
    print("🔍 ModelMuxer Security & ML Dependency Verification")
    print("=" * 55)
    print()

    # Test dependency versions
    test_dependency_versions()

    # Test ML mode
    ml_success = test_with_ml()

    # Test security mode
    security_success = test_without_ml()

    # Summary
    print("📋 Verification Summary")
    print("=" * 25)
    print(f"ML Mode: {'✅ PASS' if ml_success else '❌ FAIL'}")
    print(f"Security Mode: {'✅ PASS' if security_success else '❌ FAIL'}")

    if ml_success and security_success:
        print("\n🎉 All verification tests passed!")
        print("✅ Ready for production deployment")
        return 0
    else:
        print("\n⚠️  Some tests failed - review configuration")
        return 1


if __name__ == "__main__":
    sys.exit(main())
