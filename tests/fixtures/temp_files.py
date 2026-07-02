# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Centralized temporary file management for tests.

This module provides utilities to manage temporary files and databases
in a consistent location, avoiding duplication and cleanup issues.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


# Global temporary directory for all test files
_TEST_TEMP_DIR: Optional[Path] = None


def get_test_temp_dir() -> Path:
    """Get or create the global test temporary directory."""
    global _TEST_TEMP_DIR
    if _TEST_TEMP_DIR is None or not _TEST_TEMP_DIR.exists():
        _TEST_TEMP_DIR = Path(tempfile.mkdtemp(prefix="modelmuxer_test_"))
    return _TEST_TEMP_DIR


def cleanup_test_temp_dir() -> None:
    """Clean up the global test temporary directory."""
    global _TEST_TEMP_DIR
    if _TEST_TEMP_DIR and _TEST_TEMP_DIR.exists():
        shutil.rmtree(_TEST_TEMP_DIR, ignore_errors=True)
        _TEST_TEMP_DIR = None


@contextmanager
def temp_database(suffix: str = ".db"):
    """Create a temporary database file."""
    temp_dir = get_test_temp_dir()
    temp_file = temp_dir / f"test_db_{os.getpid()}_{suffix}"
    try:
        yield str(temp_file)
    finally:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)


@contextmanager
def temp_json_file(data: dict, suffix: str = ".json"):
    """Create a temporary JSON file with data."""
    import json

    temp_dir = get_test_temp_dir()
    temp_file = temp_dir / f"test_data_{os.getpid()}_{suffix}"
    try:
        with temp_file.open("w") as f:
            json.dump(data, f, indent=2)
        yield str(temp_file)
    finally:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)


def get_temp_compliance_report_path() -> str:
    """Get standardized path for compliance report."""
    temp_dir = get_test_temp_dir()
    return str(temp_dir / "compliance_report.json")


def get_temp_db_path(name: str = "modelmuxer") -> str:
    """Get standardized path for test database."""
    temp_dir = get_test_temp_dir()
    return str(temp_dir / f"{name}.db")


# Pytest fixtures for easy use
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_temp_dir():
    """Set up and tear down test temporary directory for the session."""
    yield
    cleanup_test_temp_dir()


@pytest.fixture
def temp_db():
    """Provide a temporary database path."""
    with temp_database() as db_path:
        yield db_path


@pytest.fixture
def temp_json():
    """Provide a temporary JSON file creator."""

    def _create_temp_json(data: dict):
        with temp_json_file(data) as json_path:
            return json_path

    return _create_temp_json
