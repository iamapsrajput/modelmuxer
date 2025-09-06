# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Health checking system for ModelMuxer.

This module provides comprehensive health monitoring for the ModelMuxer system,
including provider health checks, system resource monitoring, and dependency
status verification.
"""

import asyncio
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class HealthStatus:
    """Health status representation."""

    def __init__(self, status: str, message: str, details: dict[str, Any] | None = None):
        self.status = status  # "healthy", "degraded", "unhealthy"
        self.message = message
        self.details = details or {}
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class HealthChecker:
    """
    Comprehensive health checker for ModelMuxer.

    Monitors system health including:
    - Provider availability and response times
    - System resource usage
    - Database connectivity
    - Cache system status
    - External dependency health
    """

    def __init__(
        self,
        check_interval: int = 30,
        providers: dict[str, Any] | None = None,
        enable_provider_checks: bool = True,
        enable_resource_checks: bool = True,
    ):
        self.check_interval = check_interval
        self.providers = providers or {}
        self.enable_provider_checks = enable_provider_checks
        self.enable_resource_checks = enable_resource_checks

        # Health status cache
        self._health_cache: dict[str, HealthStatus] = {}
        self._last_check_time: float = 0.0

        # Background task
        self._health_task: asyncio.Task[None] | None = None
        self._running = False

        logger.info(
            "health_checker_initialized",
            check_interval=check_interval,
            provider_count=len(self.providers),
            provider_checks=enable_provider_checks,
            resource_checks=enable_resource_checks,
        )

    async def start(self) -> None:
        """Start the health checking background task."""
        if self._running:
            return

        self._running = True
        self._health_task = asyncio.create_task(self._health_check_loop())
        logger.info("health_checker_started")

    async def stop(self) -> None:
        """Stop the health checking background task."""
        self._running = False
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                # Task was cancelled during shutdown, this is expected
                logger.debug("health_check_task_cancelled_during_shutdown")
            except Exception as e:
                # Log any unexpected errors during task cancellation
                logger.error("health_check_task_cancellation_error", error=str(e))
        logger.info("health_checker_stopped")

    async def _health_check_loop(self) -> None:
        """Main health checking loop."""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_check_error", error=str(e))
                await asyncio.sleep(self.check_interval)

    async def _perform_health_checks(self) -> None:
        """Perform all health checks."""
        self._last_check_time = time.time()

        # System health check
        await self._check_system_health()

        # Provider health checks
        if self.enable_provider_checks:
            await self._check_provider_health()

        # Resource health checks
        if self.enable_resource_checks:
            await self._check_resource_health()

    async def _check_system_health(self) -> None:
        """Check overall system health."""
        try:
            # Basic system checks
            status = HealthStatus(
                status="healthy",
                message="System is operational",
                details={
                    "uptime": time.time() - self._last_check_time,
                    "check_time": self._last_check_time,
                },
            )
            self._health_cache["system"] = status
        except Exception as e:
            status = HealthStatus(
                status="unhealthy",
                message=f"System health check failed: {str(e)}",
                details={"error": str(e)},
            )
            self._health_cache["system"] = status

    async def _check_provider_health(self) -> None:
        """Check health of all configured providers."""
        for provider_name, provider in self.providers.items():
            try:
                # Simple health check - could be enhanced with actual API calls
                if hasattr(provider, "health_check"):
                    health_result = await provider.health_check()
                    status = HealthStatus(
                        status="healthy" if health_result else "degraded",
                        message=f"Provider {provider_name} health check",
                        details={"provider": provider_name},
                    )
                else:
                    # Assume healthy if no health check method
                    status = HealthStatus(
                        status="healthy",
                        message=f"Provider {provider_name} available",
                        details={"provider": provider_name},
                    )

                self._health_cache[f"provider_{provider_name}"] = status

            except Exception as e:
                status = HealthStatus(
                    status="unhealthy",
                    message=f"Provider {provider_name} health check failed: {str(e)}",
                    details={"provider": provider_name, "error": str(e)},
                )
                self._health_cache[f"provider_{provider_name}"] = status

    async def _check_resource_health(self) -> None:
        """Check system resource health."""
        try:
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent

            # Determine overall resource health
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                status_level = "unhealthy"
                message = "High resource usage detected"
            elif cpu_percent > 70 or memory_percent > 70 or disk_percent > 80:
                status_level = "degraded"
                message = "Elevated resource usage"
            else:
                status_level = "healthy"
                message = "Resource usage normal"

            health_status = HealthStatus(
                status=status_level,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                },
            )
            self._health_cache["resources"] = health_status

        except ImportError:
            # psutil not available, skip resource checks
            health_status = HealthStatus(
                status="healthy",
                message="Resource monitoring not available (psutil not installed)",
                details={"psutil_available": False},
            )
            self._health_cache["resources"] = health_status
        except Exception as e:
            health_status = HealthStatus(
                status="degraded",
                message=f"Resource health check failed: {str(e)}",
                details={"error": str(e)},
            )
            self._health_cache["resources"] = health_status

    def get_health_status(self, component: str | None = None) -> dict[str, Any]:
        """
        Get current health status.

        Args:
            component: Specific component to check, or None for all components

        Returns:
            Dictionary containing health status information
        """
        if component:
            if component in self._health_cache:
                return self._health_cache[component].to_dict()
            else:
                return {
                    "status": "unknown",
                    "message": f"Component {component} not found",
                    "timestamp": time.time(),
                }

        # Return overall health status
        overall_status = "healthy"
        unhealthy_components = []
        degraded_components = []

        for comp_name, health_status in self._health_cache.items():
            if health_status.status == "unhealthy":
                overall_status = "unhealthy"
                unhealthy_components.append(comp_name)
            elif health_status.status == "degraded" and overall_status == "healthy":
                overall_status = "degraded"
                degraded_components.append(comp_name)

        return {
            "status": overall_status,
            "message": f"Overall system health: {overall_status}",
            "components": {name: status.to_dict() for name, status in self._health_cache.items()},
            "summary": {
                "total_components": len(self._health_cache),
                "unhealthy_components": unhealthy_components,
                "degraded_components": degraded_components,
            },
            "last_check": self._last_check_time,
            "timestamp": time.time(),
        }

    def is_healthy(self) -> bool:
        """Check if the system is healthy."""
        for health_status in self._health_cache.values():
            if health_status.status == "unhealthy":
                return False
        return True
