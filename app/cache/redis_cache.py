# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Redis-based caching implementation for ModelMuxer.

This module provides Redis-based caching with advanced features like
TTL management, compression, and intelligent cache warming.
"""

import gzip
import pickle
from typing import Any

import redis.asyncio as redis
import structlog
from redis.asyncio import Redis

from ..core.exceptions import CacheError
from ..core.interfaces import CacheInterface

# Removed unused import: hash_prompt

logger = structlog.get_logger(__name__)


class RedisCache(CacheInterface):
    """
    Redis-based cache implementation with advanced features.

    Provides high-performance caching with compression, TTL management,
    and intelligent cache warming strategies.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        db: int = 0,
        key_prefix: str = "modelmuxer:",
        default_ttl: int = 3600,
        compression_enabled: bool = True,
        compression_threshold: int = 1024,
        max_connections: int = 10,
    ):
        self.redis_url = redis_url
        self.db = db
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self.compression_enabled = compression_enabled
        self.compression_threshold = compression_threshold

        # Redis connection
        self.redis: Redis | None = None
        self.connection_pool = None

        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
            "compression_saves": 0,
        }

        # Initialize connection
        self._initialize_connection(max_connections)

        logger.info("redis_cache_initialized", redis_url=redis_url, db=db, compression=compression_enabled)

    def _initialize_connection(self, max_connections: int) -> None:
        """Initialize Redis connection pool."""
        try:
            self.connection_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                db=self.db,
                max_connections=max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
            )

            self.redis = Redis(connection_pool=self.connection_pool)

        except Exception as e:
            logger.error("redis_connection_failed", error=str(e))
            raise CacheError(f"Failed to initialize Redis connection: {e}") from e

    def _make_key(self, key: str) -> str:
        """Create a prefixed cache key."""
        return f"{self.key_prefix}{key}"

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize and optionally compress a value."""
        try:
            # Serialize using pickle for Python objects
            serialized = pickle.dumps(value)

            # Compress if enabled and value is large enough
            if self.compression_enabled and len(serialized) > self.compression_threshold:
                compressed = gzip.compress(serialized)

                # Only use compression if it actually saves space
                if len(compressed) < len(serialized):
                    self.stats["compression_saves"] += 1
                    return b"compressed:" + compressed

            return b"raw:" + serialized

        except Exception as e:
            raise CacheError(f"Failed to serialize value: {e}") from e

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize and optionally decompress a value."""
        try:
            if data.startswith(b"compressed:"):
                # Decompress and deserialize
                compressed_data = data[11:]  # Remove "compressed:" prefix
                decompressed = gzip.decompress(compressed_data)
                return pickle.loads(decompressed)
            elif data.startswith(b"raw:"):
                # Just deserialize
                raw_data = data[4:]  # Remove "raw:" prefix
                return pickle.loads(raw_data)
            else:
                # Legacy format - assume raw pickle
                return pickle.loads(data)

        except Exception as e:
            raise CacheError(f"Failed to deserialize value: {e}") from e

    async def get(self, key: str) -> Any | None:
        """Get a value from cache."""
        if not self.redis:
            return None

        try:
            cache_key = self._make_key(key)
            data = await self.redis.get(cache_key)

            if data is None:
                self.stats["misses"] += 1
                return None

            value = self._deserialize_value(data)
            self.stats["hits"] += 1

            logger.debug("cache_hit", key=key)
            return value

        except Exception as e:
            self.stats["errors"] += 1
            logger.error("cache_get_error", key=key, error=str(e))
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set a value in cache with optional TTL."""
        if not self.redis:
            return False

        try:
            cache_key = self._make_key(key)
            serialized_value = self._serialize_value(value)

            # Use provided TTL or default
            expiry = ttl if ttl is not None else self.default_ttl

            if expiry > 0:
                await self.redis.setex(cache_key, expiry, serialized_value)
            else:
                await self.redis.set(cache_key, serialized_value)

            self.stats["sets"] += 1
            logger.debug("cache_set", key=key, ttl=expiry)
            return True

        except Exception as e:
            self.stats["errors"] += 1
            logger.error("cache_set_error", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        if not self.redis:
            return False

        try:
            cache_key = self._make_key(key)
            result = await self.redis.delete(cache_key)

            self.stats["deletes"] += 1
            logger.debug("cache_delete", key=key, existed=bool(result))
            return bool(result)

        except Exception as e:
            self.stats["errors"] += 1
            logger.error("cache_delete_error", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        if not self.redis:
            return False

        try:
            cache_key = self._make_key(key)
            result = await self.redis.exists(cache_key)
            return bool(result)

        except Exception as e:
            logger.error("cache_exists_error", key=key, error=str(e))
            return False

    async def get_ttl(self, key: str) -> int | None:
        """Get the TTL of a key."""
        if not self.redis:
            return None

        try:
            cache_key = self._make_key(key)
            ttl = await self.redis.ttl(cache_key)

            if ttl == -2:  # Key doesn't exist
                return None
            elif ttl == -1:  # Key exists but has no expiry
                return -1
            else:
                return ttl

        except Exception as e:
            logger.error("cache_ttl_error", key=key, error=str(e))
            return None

    async def extend_ttl(self, key: str, additional_seconds: int) -> bool:
        """Extend the TTL of a key."""
        if not self.redis:
            return False

        try:
            cache_key = self._make_key(key)
            current_ttl = await self.redis.ttl(cache_key)

            if current_ttl > 0:
                new_ttl = current_ttl + additional_seconds
                await self.redis.expire(cache_key, new_ttl)
                return True

            return False

        except Exception as e:
            logger.error("cache_extend_ttl_error", key=key, error=str(e))
            return False

    async def get_multiple(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from cache."""
        if not self.redis or not keys:
            return {}

        try:
            cache_keys = [self._make_key(key) for key in keys]
            values = await self.redis.mget(cache_keys)

            result = {}
            for _i, (original_key, data) in enumerate(zip(keys, values, strict=False)):
                if data is not None:
                    try:
                        result[original_key] = self._deserialize_value(data)
                        self.stats["hits"] += 1
                    except Exception as e:
                        logger.warning("cache_deserialize_error", key=original_key, error=str(e))
                        self.stats["errors"] += 1
                else:
                    self.stats["misses"] += 1

            return result

        except Exception as e:
            self.stats["errors"] += 1
            logger.error("cache_get_multiple_error", keys=keys, error=str(e))
            return {}

    async def set_multiple(self, items: dict[str, Any], ttl: int | None = None) -> bool:
        """Set multiple values in cache."""
        if not self.redis or not items:
            return True

        try:
            # Prepare data for mset
            cache_data = {}
            for key, value in items.items():
                cache_key = self._make_key(key)
                serialized_value = self._serialize_value(value)
                cache_data[cache_key] = serialized_value

            # Set all values
            await self.redis.mset(cache_data)

            # Set TTL if specified
            expiry = ttl if ttl is not None else self.default_ttl
            if expiry > 0:
                for cache_key in cache_data.keys():
                    await self.redis.expire(cache_key, expiry)

            self.stats["sets"] += len(items)
            return True

        except Exception as e:
            self.stats["errors"] += 1
            logger.error("cache_set_multiple_error", items_count=len(items), error=str(e))
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern."""
        if not self.redis:
            return 0

        try:
            cache_pattern = self._make_key(pattern)
            keys = []

            # Scan for matching keys
            async for key in self.redis.scan_iter(match=cache_pattern):
                keys.append(key)

            if keys:
                deleted_count = await self.redis.delete(*keys)
                self.stats["deletes"] += deleted_count
                logger.info("cache_pattern_cleared", pattern=pattern, deleted=deleted_count)
                return deleted_count

            return 0

        except Exception as e:
            self.stats["errors"] += 1
            logger.error("cache_clear_pattern_error", pattern=pattern, error=str(e))
            return 0

    async def get_cache_info(self) -> dict[str, Any]:
        """Get cache information and statistics."""
        if not self.redis:
            return {"status": "disconnected"}

        try:
            info = await self.redis.info()

            # Calculate hit rate
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / max(total_requests, 1)

            return {
                "status": "connected",
                "redis_info": {
                    "version": info.get("redis_version"),
                    "used_memory": info.get("used_memory"),
                    "used_memory_human": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "total_commands_processed": info.get("total_commands_processed"),
                },
                "cache_stats": self.stats.copy(),
                "hit_rate": hit_rate,
                "compression_enabled": self.compression_enabled,
            }

        except Exception as e:
            logger.error("cache_info_error", error=str(e))
            return {"status": "error", "error": str(e)}

    async def health_check(self) -> bool:
        """Check if Redis is healthy."""
        if not self.redis:
            return False

        try:
            # Try a simple ping
            pong = await self.redis.ping()
            return pong == b"PONG" or pong == "PONG"

        except Exception as e:
            logger.error("redis_health_check_failed", error=str(e))
            return False

    async def clear(self) -> bool:
        """Clear all cache entries."""
        if not self.redis:
            return False

        try:
            # Clear all keys with our prefix
            pattern = self._make_key("*")
            keys = []

            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted_count = await self.redis.delete(*keys)
                self.stats["deletes"] += deleted_count
                logger.info("redis_cache_cleared", deleted_count=deleted_count)
                return True

            return True

        except Exception as e:
            logger.error("redis_cache_clear_failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            logger.info("redis_connection_closed")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / max(total_requests, 1)

        return {**self.stats, "total_requests": total_requests, "hit_rate": hit_rate}
