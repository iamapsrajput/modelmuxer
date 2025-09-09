# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Tests for core.serialization module to improve coverage.
"""

import json
import math
from datetime import UTC, date, datetime, timezone
from unittest.mock import Mock, patch

import pytest

from app.core.serialization import (
    SecureSerializer,
    deserialize_securely,
    secure_serializer,
    serialize_securely,
)


class TestSecureSerializer:
    """Test SecureSerializer class."""

    def test_init(self):
        """Test SecureSerializer initialization."""
        serializer = SecureSerializer(compression_enabled=True, compression_threshold=1024)
        assert serializer.compression_enabled is True
        assert serializer.compression_threshold == 1024

        # Test with compression disabled
        serializer = SecureSerializer(compression_enabled=False)
        assert serializer.compression_enabled is False

    def test_prepare_for_json_primitives(self):
        """Test preparing primitive types for JSON."""
        serializer = SecureSerializer()

        # Test None
        assert serializer._prepare_for_json(None) is None

        # Test string
        assert serializer._prepare_for_json("test") == "test"

        # Test int
        assert serializer._prepare_for_json(42) == 42

        # Test float
        assert serializer._prepare_for_json(math.pi) == math.pi

        # Test bool
        assert serializer._prepare_for_json(True) is True

    def test_prepare_for_json_datetime(self):
        """Test preparing datetime objects for JSON."""
        serializer = SecureSerializer()

        # Test datetime
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = serializer._prepare_for_json(dt)
        assert result["__type__"] == "datetime"
        assert "2024-01-01" in result["__value__"]

        # Test date
        d = date(2024, 1, 1)
        result = serializer._prepare_for_json(d)
        assert result["__type__"] == "datetime"
        assert "2024-01-01" in result["__value__"]

    def test_prepare_for_json_dict(self):
        """Test preparing dictionaries for JSON."""
        serializer = SecureSerializer()

        data = {"string": "test", "number": 42, "nested": {"key": "value"}}

        result = serializer._prepare_for_json(data)
        assert result["string"] == "test"
        assert result["number"] == 42
        assert result["nested"]["key"] == "value"

    def test_prepare_for_json_list(self):
        """Test preparing lists for JSON."""
        serializer = SecureSerializer()

        data = [1, "test", {"key": "value"}]
        result = serializer._prepare_for_json(data)

        assert result[0] == 1
        assert result[1] == "test"
        assert result[2]["key"] == "value"

    def test_prepare_for_json_object(self):
        """Test preparing objects with __dict__ for JSON."""
        serializer = SecureSerializer()

        class TestObject:
            def __init__(self):
                self.name = "test"
                self.value = 42

        obj = TestObject()
        result = serializer._prepare_for_json(obj)

        assert result["__type__"] == "object"
        assert result["__class__"] == "TestObject"
        assert result["__value__"]["name"] == "test"
        assert result["__value__"]["value"] == 42

    def test_prepare_for_json_unsupported(self):
        """Test preparing unsupported types for JSON."""
        serializer = SecureSerializer()

        # Create an object without __dict__
        obj = object()

        with patch("app.core.serialization.logger") as mock_logger:
            result = serializer._prepare_for_json(obj)
            assert result["__type__"] == "string_repr"
            assert isinstance(result["__value__"], str)
            mock_logger.warning.assert_called_once()

    def test_restore_from_json_primitives(self):
        """Test restoring primitive types from JSON."""
        serializer = SecureSerializer()

        assert serializer._restore_from_json(None) is None
        assert serializer._restore_from_json("test") == "test"
        assert serializer._restore_from_json(42) == 42
        assert serializer._restore_from_json(math.pi) == math.pi
        assert serializer._restore_from_json(True) is True

    def test_restore_from_json_datetime(self):
        """Test restoring datetime from JSON."""
        serializer = SecureSerializer()

        data = {"__type__": "datetime", "__value__": "2024-01-01T12:00:00"}

        result = serializer._restore_from_json(data)
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1

    def test_restore_from_json_object(self):
        """Test restoring object from JSON."""
        serializer = SecureSerializer()

        data = {
            "__type__": "object",
            "__class__": "TestClass",
            "__value__": {"name": "test", "value": 42},
        }

        result = serializer._restore_from_json(data)
        assert result["_restored_object"] is True
        assert result["_class"] == "TestClass"
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_restore_from_json_string_repr(self):
        """Test restoring string representation from JSON."""
        serializer = SecureSerializer()

        data = {"__type__": "string_repr", "__value__": "object representation"}

        result = serializer._restore_from_json(data)
        assert result == "object representation"

    def test_restore_from_json_dict(self):
        """Test restoring regular dict from JSON."""
        serializer = SecureSerializer()

        data = {"key1": "value1", "key2": 42, "nested": {"inner": "value"}}

        result = serializer._restore_from_json(data)
        assert result["key1"] == "value1"
        assert result["key2"] == 42
        assert result["nested"]["inner"] == "value"

    def test_serialize_simple(self):
        """Test serializing simple data."""
        serializer = SecureSerializer(compression_enabled=False)

        data = {"message": "hello", "count": 5}
        result = serializer.serialize(data)

        assert isinstance(result, bytes)
        assert result.startswith(b"raw:")

        # Verify we can decode it
        json_str = result[4:].decode("utf-8")
        parsed = json.loads(json_str)
        assert parsed["message"] == "hello"
        assert parsed["count"] == 5

    def test_serialize_with_compression(self):
        """Test serializing with compression."""
        serializer = SecureSerializer(compression_enabled=True, compression_threshold=10)

        # Create data larger than threshold
        data = {"message": "x" * 100}
        result = serializer.serialize(data)

        assert isinstance(result, bytes)
        assert result.startswith(b"compressed:")

    def test_serialize_error(self):
        """Test serialization error handling."""
        serializer = SecureSerializer()

        # Create an object that can't be serialized
        class BadObject:  # noqa: B903
            def __init__(self):
                self.circular = self

        obj = BadObject()

        # Mock _prepare_for_json to raise an exception
        with patch.object(serializer, "_prepare_for_json", side_effect=Exception("Test error")):
            with pytest.raises(ValueError, match="Failed to serialize"):
                serializer.serialize(obj)

    def test_deserialize_raw(self):
        """Test deserializing raw data."""
        serializer = SecureSerializer()

        data = {"test": "value"}
        json_bytes = json.dumps(data).encode("utf-8")
        raw_data = b"raw:" + json_bytes

        result = serializer.deserialize(raw_data)
        assert result["test"] == "value"

    def test_deserialize_compressed(self):
        """Test deserializing compressed data."""
        import gzip

        serializer = SecureSerializer()

        data = {"test": "value"}
        json_bytes = json.dumps(data).encode("utf-8")
        compressed = gzip.compress(json_bytes)
        compressed_data = b"compressed:" + compressed

        result = serializer.deserialize(compressed_data)
        assert result["test"] == "value"

    def test_deserialize_legacy(self):
        """Test deserializing legacy format (no prefix)."""
        serializer = SecureSerializer()

        data = {"test": "value"}
        json_bytes = json.dumps(data).encode("utf-8")

        result = serializer.deserialize(json_bytes)
        assert result["test"] == "value"

    def test_deserialize_error(self):
        """Test deserialization error handling."""
        serializer = SecureSerializer()

        bad_data = b"raw:invalid json"

        with pytest.raises(ValueError, match="Failed to deserialize"):
            serializer.deserialize(bad_data)

    def test_roundtrip_serialization(self):
        """Test roundtrip serialization/deserialization."""
        serializer = SecureSerializer()

        original_data = {
            "string": "test",
            "number": 42,
            "datetime": datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }

        # Serialize then deserialize
        serialized = serializer.serialize(original_data)
        restored = serializer.deserialize(serialized)

        assert restored["string"] == "test"
        assert restored["number"] == 42
        assert isinstance(restored["datetime"], datetime)
        assert restored["list"] == [1, 2, 3]
        assert restored["nested"]["key"] == "value"


class TestGlobalFunctions:
    """Test global serialization functions."""

    def test_serialize_securely(self):
        """Test serialize_securely function."""
        data = {"test": "value", "number": 42}

        result = serialize_securely(data)
        assert isinstance(result, bytes)
        assert b"test" in result or result.startswith(b"compressed:")

    def test_deserialize_securely(self):
        """Test deserialize_securely function."""
        data = {"test": "value", "number": 42}

        # First serialize
        serialized = serialize_securely(data)

        # Then deserialize
        result = deserialize_securely(serialized)
        assert result["test"] == "value"
        assert result["number"] == 42

    def test_global_serializer_instance(self):
        """Test that global serializer instance exists."""
        assert secure_serializer is not None
        assert isinstance(secure_serializer, SecureSerializer)


class TestNumpySupport:
    """Test numpy array serialization support."""

    @pytest.mark.skipif(not pytest.importorskip("numpy"), reason="numpy not available")
    def test_prepare_numpy_array(self):
        """Test preparing numpy array for JSON."""
        import numpy as np

        serializer = SecureSerializer()

        arr = np.array([1, 2, 3, 4])
        result = serializer._prepare_for_json(arr)

        assert result["__type__"] == "numpy_array"
        assert result["__value__"] == [1, 2, 3, 4]
        assert "__dtype__" in result
        assert "__shape__" in result

    @pytest.mark.skipif(not pytest.importorskip("numpy"), reason="numpy not available")
    def test_restore_numpy_array(self):
        """Test restoring numpy array from JSON."""
        import numpy as np

        serializer = SecureSerializer()

        data = {
            "__type__": "numpy_array",
            "__value__": [[1, 2], [3, 4]],
            "__dtype__": "int64",
            "__shape__": (2, 2),
        }

        result = serializer._restore_from_json(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        assert result[0, 0] == 1
        assert result[1, 1] == 4

    def test_restore_numpy_array_without_numpy(self):
        """Test restoring numpy array when numpy is not available."""
        serializer = SecureSerializer()

        data = {
            "__type__": "numpy_array",
            "__value__": [1, 2, 3],
            "__dtype__": "float64",
            "__shape__": (3,),
        }

        # Mock NUMPY_AVAILABLE to be False
        with patch("app.core.serialization.NUMPY_AVAILABLE", False):
            with patch("app.core.serialization.logger") as mock_logger:
                result = serializer._restore_from_json(data)
                assert result == [1, 2, 3]  # Returns raw list
                mock_logger.warning.assert_called_once()
