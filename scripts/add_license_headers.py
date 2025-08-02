#!/usr/bin/env python3
# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Script to add license headers to all source code files in the ModelMuxer project.
"""

from pathlib import Path

# License headers for different file types
HEADERS = {
    "python": [
        "# ModelMuxer (c) 2025 Ajay Rajput",
        "# Licensed under Business Source License 1.1 – see LICENSE for details.",
        "",
    ],
    "javascript": [
        "// ModelMuxer (c) 2025 Ajay Rajput",
        "// Licensed under Business Source License 1.1 – see LICENSE for details.",
        "",
    ],
    "yaml": [
        "# ModelMuxer (c) 2025 Ajay Rajput",
        "# Licensed under Business Source License 1.1 – see LICENSE for details.",
        "",
    ],
    "sql": [
        "-- ModelMuxer (c) 2025 Ajay Rajput",
        "-- Licensed under Business Source License 1.1 – see LICENSE for details.",
        "",
    ],
    "dockerfile": [
        "# ModelMuxer (c) 2025 Ajay Rajput",
        "# Licensed under Business Source License 1.1 – see LICENSE for details.",
        "",
    ],
}

# File extensions to header type mapping
EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "javascript",
    ".jsx": "javascript",
    ".tsx": "javascript",
    ".vue": "javascript",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sql": "sql",
}

# Special files
SPECIAL_FILES = {
    "Dockerfile": "dockerfile",
    "docker-compose.yml": "yaml",
    "docker-compose.yaml": "yaml",
}

# Files and directories to skip
SKIP_PATTERNS = [
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    "cache",
    ".cache",
    "LICENSE",
    "COPYRIGHT",
    "NOTICE",
    "README.md",
    "THIRD_PARTY_LICENSES.md",
    "TRADEMARKS.md",
]


def should_skip(path: Path) -> bool:
    """Check if a file or directory should be skipped."""
    for pattern in SKIP_PATTERNS:
        if pattern in str(path):
            return True
    return False


def get_header_type(file_path: Path) -> str | None:
    """Get the appropriate header type for a file."""
    # Check special files first
    if file_path.name in SPECIAL_FILES:
        return SPECIAL_FILES[file_path.name]

    # Check extension
    suffix = file_path.suffix.lower()
    return EXTENSION_MAP.get(suffix)


def has_license_header(content: str, header_type: str) -> bool:
    """Check if the file already has a license header."""
    header_lines = HEADERS[header_type]
    header_lines[0].strip()

    # Look for copyright notice in first few lines
    lines = content.split('\n')[:10]
    for line in lines:
        if "ModelMuxer (c) 2025 Ajay Rajput" in line:
            return True
    return False


def add_header_to_file(file_path: Path, header_type: str) -> bool:
    """Add license header to a file. Returns True if file was modified."""
    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Skipping binary file: {file_path}")
        return False
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

    if has_license_header(content, header_type):
        return False

    header_lines = HEADERS[header_type]
    header_text = '\n'.join(header_lines)

    # Handle special cases for different file types
    if header_type == "python":
        # Handle shebang lines
        if content.startswith("#!"):
            lines = content.split('\n')
            shebang = lines[0]
            rest = '\n'.join(lines[1:])
            new_content = shebang + '\n' + header_text + rest
        else:
            new_content = header_text + content
    else:
        new_content = header_text + content

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")
        return False


def process_directory(directory: Path) -> dict[str, list[str]]:
    """Process all files in a directory recursively."""
    results = {
        "modified": [],
        "skipped": [],
        "errors": [],
    }

    for file_path in directory.rglob("*"):
        if file_path.is_file() and not should_skip(file_path):
            header_type = get_header_type(file_path)

            if header_type:
                try:
                    if add_header_to_file(file_path, header_type):
                        results["modified"].append(str(file_path))
                        print(f"Added header to: {file_path}")
                    else:
                        results["skipped"].append(str(file_path))
                except Exception as e:
                    results["errors"].append(f"{file_path}: {e}")
                    print(f"Error processing {file_path}: {e}")

    return results


def main():
    """Main function to add license headers to all applicable files."""
    project_root = Path(__file__).parent.parent

    print("Adding license headers to ModelMuxer source files...")
    print(f"Project root: {project_root}")

    results = process_directory(project_root)

    print("\nSummary:")
    print(f"Files modified: {len(results['modified'])}")
    print(f"Files skipped (already have headers): {len(results['skipped'])}")
    print(f"Errors: {len(results['errors'])}")

    if results["modified"]:
        print("\nModified files:")
        for file_path in results["modified"]:
            print(f"  - {file_path}")

    if results["errors"]:
        print("\nErrors:")
        for error in results["errors"]:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
