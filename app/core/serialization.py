# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Secure serialization utilities for ModelMuxer.

This module provides secure alternatives to pickle for serializing and deserializing
data in cache and storage systems. It uses JSON with type hints and validation
to ensure data integrity and security.
"""

import gzip
import json
from datetime import date, datetime
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Supported types for secure serialization
SerializableType = str | int | float | bool | None | dict[str, Any] | list[Any]


class SecureSerializer:
    """Secure serializer that replaces pickle with JSON-based serialization."""

    def __init__(self, compression_enabled: bool = True, compression_threshold: int = 1024):
        """Initialize the secure serializer.

        Args:
            compression_enabled: Whether to enable compression for large data
            compression_threshold: Minimum size in bytes to trigger compression
        """
        self.compression_enabled = compression_enabled
        self.compression_threshold = compression_threshold

    def _prepare_for_json(self, obj: Any) -> SerializableType:
        """Prepare an object for JSON serialization by converting complex types."""
        if obj is None or isinstance(obj, str | int | float | bool):
            return obj
        elif isinstance(obj, datetime | date):
            return {"__type__": "datetime", "__value__": obj.isoformat()}
        elif isinstance(obj, np.ndarray):
            return {
                "__type__": "numpy_array",
                "__value__": obj.tolist(),
                "__dtype__": str(obj.dtype),
                "__shape__": obj.shape,
            }
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list | tuple):
            return [self._prepare_for_json(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            # For simple objects, serialize their __dict__
            return {
                "__type__": "object",
                "__class__": obj.__class__.__name__,
                "__value__": self._prepare_for_json(obj.__dict__),
            }
        else:
            # For unsupported types, convert to string representation
            logger.warning("unsupported_type_serialization", type=type(obj).__name__)
            return {"__type__": "string_repr", "__value__": str(obj)}

    def _restore_from_json(self, obj: Any) -> Any:
        """Restore an object from JSON serialization."""
        if not isinstance(obj, dict):
            return obj

        if "__type__" in obj:
            obj_type = obj["__type__"]
            value = obj["__value__"]

            if obj_type == "datetime":
                return datetime.fromisoformat(value)
            elif obj_type == "numpy_array":
                array = np.array(value, dtype=obj["__dtype__"])
                return array.reshape(obj["__shape__"])
            elif obj_type == "object":
                # For simple objects, we can't fully restore them
                # Return as a dict with metadata
                return {
                    "_restored_object": True,
                    "_class": obj["__class__"],
                    **self._restore_from_json(value),
                }
            elif obj_type == "string_repr":
                return value

        # Regular dict - recursively restore values
        if isinstance(obj, dict):
            return {k: self._restore_from_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._restore_from_json(item) for item in obj]

        return obj

    def serialize(self, obj: Any) -> bytes:
        """Serialize an object to bytes using secure JSON serialization.

        Args:
            obj: Object to serialize

        Returns:
            Serialized bytes

        Raises:
            ValueError: If object cannot be serialized
        """
        try:
            # Prepare object for JSON serialization
            prepared_obj = self._prepare_for_json(obj)

            # Serialize to JSON
            json_str = json.dumps(prepared_obj, separators=(",", ":"), ensure_ascii=False)
            json_bytes = json_str.encode("utf-8")

            # Compress if enabled and data is large enough
            if self.compression_enabled and len(json_bytes) > self.compression_threshold:
                compressed = gzip.compress(json_bytes)
                return b"compressed:" + compressed
            else:
                return b"raw:" + json_bytes

        except Exception as e:
            raise ValueError(f"Failed to serialize object: {e}") from e

    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes back to an object using secure JSON deserialization.

        Args:
            data: Serialized bytes

        Returns:
            Deserialized object

        Raises:
            ValueError: If data cannot be deserialized
        """
        try:
            if data.startswith(b"compressed:"):
                # Decompress and deserialize
                compressed_data = data[11:]  # Remove "compressed:" prefix
                json_bytes = gzip.decompress(compressed_data)
            elif data.startswith(b"raw:"):
                # Just deserialize
                json_bytes = data[4:]  # Remove "raw:" prefix
            else:
                # Legacy format - assume raw JSON
                json_bytes = data

            # Deserialize from JSON
            json_str = json_bytes.decode("utf-8")
            prepared_obj = json.loads(json_str)

            # Restore object from JSON representation
            return self._restore_from_json(prepared_obj)

        except Exception as e:
            raise ValueError(f"Failed to deserialize data: {e}") from e


# Global serializer instance
secure_serializer = SecureSerializer()


def serialize_securely(obj: Any) -> bytes:
    """Serialize an object securely using JSON-based serialization."""
    return secure_serializer.serialize(obj)


def deserialize_securely(data: bytes) -> Any:
    """Deserialize data securely using JSON-based deserialization."""
    return secure_serializer.deserialize(data)
