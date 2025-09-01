#!/usr/bin/env python3
# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
License compliance checker for ModelMuxer project.
Validates that all required license files exist and all source files have proper headers.
"""

import json
import sys
from pathlib import Path


class LicenseComplianceChecker:
    """Check license compliance across the ModelMuxer project."""

    REQUIRED_FILES = ["LICENSE", "COPYRIGHT", "NOTICE", "TRADEMARKS.md", "THIRD_PARTY_LICENSES.md"]

    REQUIRED_HEADERS = {
        ".py": "# ModelMuxer (c) 2025 Ajay Rajput",
        ".js": "// ModelMuxer (c) 2025 Ajay Rajput",
        ".ts": "// ModelMuxer (c) 2025 Ajay Rajput",
        ".jsx": "// ModelMuxer (c) 2025 Ajay Rajput",
        ".tsx": "// ModelMuxer (c) 2025 Ajay Rajput",
        ".yaml": "# ModelMuxer (c) 2025 Ajay Rajput",
        ".yml": "# ModelMuxer (c) 2025 Ajay Rajput",
    }

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
    ]

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.errors = []
        self.warnings = []

    def should_skip(self, path: Path) -> bool:
        """Check if a file or directory should be skipped."""
        for pattern in self.SKIP_PATTERNS:
            if pattern in str(path):
                return True
        return False

    def check_required_files(self) -> None:
        """Check that all required license files exist."""
        for file_name in self.REQUIRED_FILES:
            file_path = self.project_root / file_name
            if not file_path.exists():
                self.errors.append(f"Missing required license file: {file_name}")
            elif file_path.stat().st_size == 0:
                self.errors.append(f"Required license file is empty: {file_name}")

    def check_file_header(self, file_path: Path) -> bool:
        """Check if a file has the required license header."""
        suffix = file_path.suffix.lower()
        if suffix not in self.REQUIRED_HEADERS:
            return True  # Not a file type we check

        self.REQUIRED_HEADERS[suffix]

        try:
            content = Path(file_path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return True  # Skip binary files
        except Exception:
            return False

        # Check first 10 lines for the copyright notice
        lines = content.split("\n")[:10]
        for line in lines:
            if "ModelMuxer (c) 2025 Ajay Rajput" in line:
                return True

        return False

    def check_source_file_headers(self) -> None:
        """Check all source files have proper license headers."""
        missing_headers = []

        for file_path in self.project_root.rglob("*"):
            if (
                file_path.is_file()
                and not self.should_skip(file_path)
                and file_path.suffix.lower() in self.REQUIRED_HEADERS
            ):
                if not self.check_file_header(file_path):
                    missing_headers.append(str(file_path.relative_to(self.project_root)))

        if missing_headers:
            self.errors.append(f"Files missing license headers: {len(missing_headers)}")
            for file_path in missing_headers[:10]:  # Show first 10
                self.errors.append(f"  - {file_path}")
            if len(missing_headers) > 10:
                self.errors.append(f"  ... and {len(missing_headers) - 10} more")

    def check_license_file_content(self) -> None:
        """Check that license files have expected content."""
        license_file = self.project_root / "LICENSE"
        if license_file.exists():
            try:
                content = license_file.read_text(encoding="utf-8")
                if "Business Source License 1.1" not in content:
                    self.errors.append("LICENSE file does not contain 'Business Source License 1.1'")
                if "Ajay Rajput" not in content:
                    self.errors.append("LICENSE file does not contain 'Ajay Rajput'")
                if "January 1, 2027" not in content:
                    self.errors.append("LICENSE file does not contain correct change date")
            except Exception as e:
                self.errors.append(f"Could not read LICENSE file: {e}")

    def check_pyproject_license(self) -> None:
        """Check that pyproject.toml has correct license information."""
        pyproject_file = self.project_root / "pyproject.toml"
        if pyproject_file.exists():
            try:
                content = pyproject_file.read_text(encoding="utf-8")
                if 'license = { text = "Business Source License 1.1" }' not in content:
                    self.warnings.append("pyproject.toml license field should be 'Business Source License 1.1'")
            except Exception as e:
                self.warnings.append(f"Could not read pyproject.toml: {e}")

    def check_readme_license_section(self) -> None:
        """Check that README.md has proper license section."""
        readme_file = self.project_root / "README.md"
        if readme_file.exists():
            try:
                content = readme_file.read_text(encoding="utf-8")
                if "Business Source License" not in content:
                    self.warnings.append("README.md should mention Business Source License")
                if "licensing@modelmuxer.com" not in content:
                    self.warnings.append("README.md should include commercial licensing contact")
            except Exception as e:
                self.warnings.append(f"Could not read README.md: {e}")

    def run_all_checks(self) -> tuple[list[str], list[str]]:
        """Run all compliance checks."""
        print("Running license compliance checks...")

        self.check_required_files()
        self.check_license_file_content()
        self.check_source_file_headers()
        self.check_pyproject_license()
        self.check_readme_license_section()

        return self.errors, self.warnings

    def generate_report(self) -> dict:
        """Generate a detailed compliance report."""
        errors, warnings = self.run_all_checks()

        report = {
            "compliance_status": "PASS" if not errors else "FAIL",
            "timestamp": "2025-08-02",  # Current date
            "project_root": str(self.project_root),
            "errors": errors,
            "warnings": warnings,
            "summary": {
                "total_errors": len(errors),
                "total_warnings": len(warnings),
                "required_files_check": ("PASS" if not any("Missing required" in e for e in errors) else "FAIL"),
                "header_check": ("PASS" if not any("missing license headers" in e for e in errors) else "FAIL"),
                "content_check": "PASS" if not any("LICENSE file" in e for e in errors) else "FAIL",
            },
        }

        return report


def main():
    """Main function to run license compliance checks."""
    project_root = Path(__file__).parent.parent
    checker = LicenseComplianceChecker(project_root)

    print(f"Checking license compliance for ModelMuxer at: {project_root}")
    print("=" * 60)

    # Generate and display report
    report = checker.generate_report()

    print(f"Compliance Status: {report['compliance_status']}")
    print(f"Errors: {report['summary']['total_errors']}")
    print(f"Warnings: {report['summary']['total_warnings']}")
    print()

    if report["errors"]:
        print("❌ ERRORS:")
        for error in report["errors"]:
            print(f"  {error}")
        print()

    if report["warnings"]:
        print("⚠️  WARNINGS:")
        for warning in report["warnings"]:
            print(f"  {warning}")
        print()

    # Save detailed report
    report_file = project_root / "compliance_report.json"
    report_file.write_text(json.dumps(report, indent=2))
    print(f"Detailed report saved to: {report_file}")

    # Exit with error code if compliance failed
    if report["compliance_status"] == "FAIL":
        print("\n❌ License compliance check FAILED")
        sys.exit(1)
    else:
        print("\n✅ License compliance check PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
