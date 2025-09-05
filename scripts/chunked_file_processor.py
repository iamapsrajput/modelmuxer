#!/usr/bin/env python3
"""
Chunked file processor for handling large files to avoid 413 Request Entity Too Large errors.
Reads files in chunks and processes them incrementally.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple


class ChunkedFileProcessor:
    """Process large files in manageable chunks."""

    def __init__(self, chunk_size: int = 1024 * 1024):  # 1MB default chunk size
        self.chunk_size = chunk_size
        self.processed_files: List[str] = []

    def get_file_size(self, filepath: Path) -> int:
        """Get file size in bytes."""
        return Path(filepath).stat().st_size

    def should_chunk_file(self, filepath: Path) -> bool:
        """Determine if file should be processed in chunks."""
        return self.get_file_size(filepath) > self.chunk_size

    def read_file_chunks(self, filepath: Path) -> Iterator[Tuple[int, str, int, int]]:
        """
        Read file in chunks.

        Returns:
            Iterator of (chunk_number, chunk_content, start_line, end_line)
        """
        file_size = self.get_file_size(filepath)

        if not self.should_chunk_file(filepath):
            # File is small enough to read entirely
            # ruff: noqa: FURB101 - Need file handle for chunked processing consistency
            with open(filepath, encoding="utf-8") as f:
                content = f.read()
                line_count = content.count("\n") + 1
                yield (1, content, 1, line_count)
            return

        print(f"📁 Processing large file: {filepath}")
        print(f"📊 File size: {file_size:,} bytes ({file_size / (1024 * 1024):.2f} MB)")

        chunk_count = (file_size + self.chunk_size - 1) // self.chunk_size
        print(f"🔢 Will process in {chunk_count} chunks")

        with open(filepath, encoding="utf-8") as f:
            chunk_num = 1
            current_line = 1

            while True:
                chunk_content = f.read(self.chunk_size)
                if not chunk_content:
                    break

                # Count lines in this chunk
                lines_in_chunk = chunk_content.count("\n")
                if not chunk_content.endswith("\n") and lines_in_chunk > 0:
                    lines_in_chunk += 1

                end_line = current_line + lines_in_chunk - 1

                print(
                    f"📋 Chunk {chunk_num}/{chunk_count}: lines {current_line}-{end_line} ({len(chunk_content):,} chars)"
                )

                yield (chunk_num, chunk_content, current_line, end_line)

                current_line = end_line + 1
                chunk_num += 1

    def analyze_file_structure(self, filepath: Path) -> Dict[str, Any]:
        """Analyze file structure and provide summary."""
        file_size = self.get_file_size(filepath)

        analysis: Dict[str, Any] = {
            "path": str(filepath),
            "size_bytes": file_size,
            "size_mb": file_size / (1024 * 1024),
            "requires_chunking": self.should_chunk_file(filepath),
            "chunks_needed": 0,
            "total_lines": 0,
            "chunk_summaries": [],
        }

        for chunk_num, chunk_content, start_line, end_line in self.read_file_chunks(filepath):
            analysis["chunks_needed"] += 1
            analysis["total_lines"] = end_line

            # Analyze chunk content
            chunk_summary = {
                "chunk_number": chunk_num,
                "start_line": start_line,
                "end_line": end_line,
                "line_count": end_line - start_line + 1,
                "char_count": len(chunk_content),
                "functions_detected": chunk_content.count("def "),
                "classes_detected": chunk_content.count("class "),
                "imports_detected": chunk_content.count("import "),
            }
            analysis["chunk_summaries"].append(chunk_summary)

        return analysis

    def find_large_files(self, directory: Path, extensions: List[str] = None) -> List[Path]:
        """Find files that need chunked processing."""
        if extensions is None:
            extensions = [".py", ".md", ".txt", ".json", ".yaml", ".yml"]

        large_files = []

        for ext in extensions:
            pattern = f"**/*{ext}"
            for filepath in directory.glob(pattern):
                if filepath.is_file() and not any(
                    skip in str(filepath) for skip in [".venv", "__pycache__", ".git"]
                ):
                    if self.get_file_size(filepath) > 100 * 1024:  # Files > 100KB
                        large_files.append(filepath)

        return sorted(large_files, key=self.get_file_size, reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Chunked file processor for large files")
    parser.add_argument("--file", type=Path, help="Specific file to analyze")
    parser.add_argument(
        "--directory", type=Path, default=Path(), help="Directory to scan for large files"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1024 * 1024, help="Chunk size in bytes (default: 1MB)"
    )
    parser.add_argument("--list-large", action="store_true", help="List large files only")

    args = parser.parse_args()

    processor = ChunkedFileProcessor(chunk_size=args.chunk_size)

    if args.list_large:
        print("🔍 Scanning for large files...")
        large_files = processor.find_large_files(args.directory)

        print(f"\n📋 Found {len(large_files)} large files (>100KB):")
        for filepath in large_files:
            size = processor.get_file_size(filepath)
            needs_chunking = "🔄 CHUNKED" if processor.should_chunk_file(filepath) else "📄 DIRECT"
            print(f"{needs_chunking} {filepath} ({size:,} bytes, {size / (1024 * 1024):.2f} MB)")

        return

    if args.file:
        if not args.file.exists():
            print(f"❌ File not found: {args.file}")
            sys.exit(1)

        print(f"🔍 Analyzing file: {args.file}")
        analysis = processor.analyze_file_structure(args.file)

        print("\n📊 File Analysis Results:")
        print(f"📁 Path: {analysis['path']}")
        print(f"📏 Size: {analysis['size_bytes']:,} bytes ({analysis['size_mb']:.2f} MB)")
        print(f"📝 Total lines: {analysis['total_lines']:,}")
        print(f"🔄 Requires chunking: {'Yes' if analysis['requires_chunking'] else 'No'}")
        print(f"🔢 Chunks needed: {analysis['chunks_needed']}")

        print("\n📋 Chunk Details:")
        for chunk in analysis["chunk_summaries"]:
            print(
                f"  Chunk {chunk['chunk_number']}: Lines {chunk['start_line']}-{chunk['end_line']} "
                f"({chunk['line_count']} lines, {chunk['char_count']:,} chars)"
            )
            if chunk["functions_detected"] > 0:
                print(f"    🔧 Functions: {chunk['functions_detected']}")
            if chunk["classes_detected"] > 0:
                print(f"    🏛️  Classes: {chunk['classes_detected']}")
            if chunk["imports_detected"] > 0:
                print(f"    📦 Imports: {chunk['imports_detected']}")


if __name__ == "__main__":
    main()
