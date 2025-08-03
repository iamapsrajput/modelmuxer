# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
In-memory caching implementation for ModelMuxer.

This module provides high-performance in-memory caching with LRU eviction,
TTL support, and memory management.
"""

import threading
import time
from collections import OrderedDict
from typing import Any, NamedTuple

import structlog

# Removed unused import: CacheError
from ..core.interfaces import CacheInterface

logger = structlog.get_logger(__name__)


class CacheEntry(NamedTuple):
    """Cache entry with value and metadata."""

    value: Any
    created_at: float
    expires_at: float | None
    access_count: int
    last_accessed: float


class MemoryCache(CacheInterface):
    """
    High-performance in-memory cache with LRU eviction and TTL support.

    Provides fast caching for frequently accessed data with automatic
    memory management and configurable eviction policies.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,
        cleanup_interval: int = 300,
        max_memory_mb: int | None = None,
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.max_memory_bytes = max_memory_mb * 1024 * 1024 if max_memory_mb else None

        # Thread-safe storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
            "expired_cleanups": 0,
            "memory_cleanups": 0,
        }

        # Background cleanup
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_thread()

        logger.info(
            "memory_cache_initialized",
            max_size=max_size,
            default_ttl=default_ttl,
            max_memory_mb=max_memory_mb,
        )

    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        if self.cleanup_interval > 0:
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_worker, daemon=True, name="MemoryCache-Cleanup"
            )
            self._cleanup_thread.start()

    def _cleanup_worker(self) -> None:
        """Background worker for cleaning up expired entries."""
        while not self._stop_cleanup.wait(self.cleanup_interval):
            try:
                self._cleanup_expired()
                self._cleanup_memory_if_needed()
            except Exception as e:
                logger.error("cache_cleanup_error", error=str(e))

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []

        with self._lock:
            for key, entry in self._cache.items():
                if entry.expires_at and entry.expires_at <= current_time:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                self.stats["expired_cleanups"] += 1

        if expired_keys:
            logger.debug("expired_entries_cleaned", count=len(expired_keys))

    def _cleanup_memory_if_needed(self) -> None:
        """Clean up memory if usage is too high."""
        if not self.max_memory_bytes:
            return

        try:
            import sys

            current_memory = sys.getsizeof(self._cache)

            # Rough estimation of memory usage
            for entry in self._cache.values():
                current_memory += sys.getsizeof(entry.value)

            if current_memory > self.max_memory_bytes:
                # Remove least recently used items
                items_to_remove = max(1, len(self._cache) // 10)  # Remove 10%

                with self._lock:
                    for _ in range(items_to_remove):
                        if self._cache:
                            self._cache.popitem(last=False)  # Remove oldest
                            self.stats["memory_cleanups"] += 1

                logger.info(
                    "memory_cleanup_performed",
                    removed_items=items_to_remove,
                    memory_usage_mb=current_memory / (1024 * 1024),
                )

        except Exception as e:
            logger.warning("memory_cleanup_failed", error=str(e))

    def _evict_if_needed(self) -> None:
        """Evict entries if cache is at capacity."""
        while len(self._cache) >= self.max_size:
            # Remove least recently used item
            self._cache.popitem(last=False)
            self.stats["evictions"] += 1

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if an entry is expired."""
        if entry.expires_at is None:
            return False
        return time.time() >= entry.expires_at

    async def get(self, key: str) -> Any | None:
        """Get a value from cache."""
        current_time = time.time()

        with self._lock:
            if key not in self._cache:
                self.stats["misses"] += 1
                return None

            entry = self._cache[key]

            # Check if expired
            if self._is_expired(entry):
                del self._cache[key]
                self.stats["misses"] += 1
                self.stats["expired_cleanups"] += 1
                return None

            # Update access information and move to end (most recently used)
            updated_entry = entry._replace(access_count=entry.access_count + 1, last_accessed=current_time)

            # Move to end (most recently used)
            del self._cache[key]
            self._cache[key] = updated_entry

            self.stats["hits"] += 1
            return entry.value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set a value in cache with optional TTL."""
        current_time = time.time()

        # Calculate expiration time
        expires_at = None
        if ttl is not None:
            if ttl > 0:
                expires_at = current_time + ttl
        elif self.default_ttl > 0:
            expires_at = current_time + self.default_ttl

        # Create cache entry
        entry = CacheEntry(
            value=value,
            created_at=current_time,
            expires_at=expires_at,
            access_count=0,
            last_accessed=current_time,
        )

        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]

            # Evict if needed
            self._evict_if_needed()

            # Add new entry
            self._cache[key] = entry
            self.stats["sets"] += 1

        return True

    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self.stats["deletes"] += 1
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        with self._lock:
            if key not in self._cache:
                return False

            entry = self._cache[key]
            if self._is_expired(entry):
                del self._cache[key]
                self.stats["expired_cleanups"] += 1
                return False

            return True

    async def get_ttl(self, key: str) -> int | None:
        """Get the TTL of a key."""
        current_time = time.time()

        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            if entry.expires_at is None:
                return -1  # No expiration

            if entry.expires_at <= current_time:
                del self._cache[key]
                self.stats["expired_cleanups"] += 1
                return None  # Expired

            return int(entry.expires_at - current_time)

    async def extend_ttl(self, key: str, additional_seconds: int) -> bool:
        """Extend the TTL of a key."""
        current_time = time.time()

        with self._lock:
            if key not in self._cache:
                return False

            entry = self._cache[key]

            if self._is_expired(entry):
                del self._cache[key]
                self.stats["expired_cleanups"] += 1
                return False

            # Calculate new expiration time
            if entry.expires_at is None:
                new_expires_at = current_time + additional_seconds
            else:
                new_expires_at = entry.expires_at + additional_seconds

            # Update entry
            updated_entry = entry._replace(expires_at=new_expires_at)
            self._cache[key] = updated_entry

            return True

    async def get_multiple(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from cache."""
        result = {}

        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value

        return result

    async def set_multiple(self, items: dict[str, Any], ttl: int | None = None) -> bool:
        """Set multiple values in cache."""
        for key, value in items.items():
            await self.set(key, value, ttl)

        return True

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern (simple wildcard support)."""
        import fnmatch

        matching_keys = []

        with self._lock:
            for key in self._cache.keys():
                if fnmatch.fnmatch(key, pattern):
                    matching_keys.append(key)

            for key in matching_keys:
                del self._cache[key]
                self.stats["deletes"] += 1

        return len(matching_keys)

    def clear_all(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            deleted_count = len(self._cache)
            self._cache.clear()
            self.stats["deletes"] += deleted_count

        logger.info("cache_cleared", deleted_count=deleted_count)

    async def get_cache_info(self) -> dict[str, Any]:
        """Get cache information and statistics."""
        time.time()

        with self._lock:
            cache_size = len(self._cache)

            # Calculate memory usage estimation
            try:
                import sys

                memory_usage = sys.getsizeof(self._cache)
                for entry in self._cache.values():
                    memory_usage += sys.getsizeof(entry.value)
                memory_usage_mb = memory_usage / (1024 * 1024)
            except Exception:
                memory_usage_mb = 0

            # Count expired entries
            expired_count = 0
            for entry in self._cache.values():
                if self._is_expired(entry):
                    expired_count += 1

            # Calculate hit rate
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / max(total_requests, 1)

        return {
            "status": "active",
            "cache_size": cache_size,
            "max_size": self.max_size,
            "memory_usage_mb": round(memory_usage_mb, 2),
            "expired_entries": expired_count,
            "cache_stats": self.stats.copy(),
            "hit_rate": hit_rate,
            "cleanup_thread_active": self._cleanup_thread and self._cleanup_thread.is_alive(),
        }

    async def health_check(self) -> bool:
        """Check if cache is healthy."""
        try:
            # Try a simple operation
            test_key = "health_check_test"
            await self.set(test_key, "test_value", ttl=1)
            value = await self.get(test_key)
            await self.delete(test_key)

            return value == "test_value"

        except Exception as e:
            logger.error("memory_cache_health_check_failed", error=str(e))
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / max(total_requests, 1)

        with self._lock:
            cache_size = len(self._cache)

        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "current_size": cache_size,
            "max_size": self.max_size,
        }

    async def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            with self._lock:
                deleted_count = len(self._cache)
                self._cache.clear()
                self.stats["deletes"] += deleted_count

            logger.info("cache_cleared", deleted_count=deleted_count)
            return True
        except Exception as e:
            logger.error("cache_clear_failed", error=str(e))
            return False

    def close(self) -> None:
        """Close the cache and cleanup resources."""
        self._stop_cleanup.set()

        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)

        with self._lock:
            self._cache.clear()

        logger.info("memory_cache_closed")
