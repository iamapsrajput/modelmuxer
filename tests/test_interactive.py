#!/usr/bin/env python3
# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Test script to demonstrate the interactive Claude CLI
This simulates user input for testing purposes
"""

import subprocess  # noqa: S404
import sys
import time


def test_interactive_mode():
    """Test the interactive mode with simulated input"""

    # Commands to test
    test_commands = [
        "/status",
        "/models",
        "Hello, can you explain what machine learning is?",
        "/help",
        "exit",
    ]

    # Create input for the interactive session
    input_text = "\n".join(test_commands)

    # Run the CLI in interactive mode
    # Using trusted input from constants, no user input involved
    # ruff: noqa: S603
    process = subprocess.Popen(
        [sys.executable, "claude-cli.py", "--api-key", "sk-test-claude-dev"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Send the test commands
    stdout, stderr = process.communicate(input=input_text)

    print("=== Interactive Session Test ===")
    print("STDOUT:")
    print(stdout)

    if stderr:
        print("\nSTDERR:")
        print(stderr)

    print(f"\nReturn code: {process.returncode}")


if __name__ == "__main__":
    test_interactive_mode()
