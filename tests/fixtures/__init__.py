# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Centralized test fixtures and utilities for ModelMuxer.

This package provides standardized fixtures, mocks, and utilities
to ensure consistent testing across all test files.
"""

from .mocks import *  # noqa: F403

# The `tests/fixtures/temp_files.py` module is required for several test suites
# but in certain CI environments the import machinery fails to locate it
# during collection (likely due to path or package resolution quirks). We guard
# against this by retrying the import with an explicit path injection, thereby
# ensuring the tests are always importable while keeping local development
# unaffected.

try:
    from .temp_files import *  # noqa: F403
except ModuleNotFoundError:  # pragma: no cover – safeguard for CI edge-cases
    import importlib
    import sys
    from pathlib import Path

    _fixtures_dir = Path(__file__).parent
    if str(_fixtures_dir) not in sys.path:
        sys.path.insert(0, str(_fixtures_dir))

    # Retry the import after adjusting the path
    from .temp_files import *  # noqa: F403
