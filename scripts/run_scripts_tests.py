#!/usr/bin/env python3
"""Wrapper to run the scripts test suite via pytest.

Used by the CI Gate Registry (Issue #4254).
"""
import subprocess
import sys

if __name__ == "__main__":
    # Run pytest on the scripts directory
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "scripts/", "-q"],
        cwd=".",
    )
    sys.exit(result.returncode)
