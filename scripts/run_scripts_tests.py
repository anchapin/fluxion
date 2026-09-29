#!/usr/bin/env python3
"""Wrapper to run the scripts test suite via pytest.

Used by the CI Gate Registry (Issue #4254).

Ensures pytest is available before running (Issue #4261): the gate
runner environment does not pre-install it.
"""
import subprocess
import sys


def ensure_pytest():
    """Install test dependencies from scripts/requirements-test.txt if missing."""
    # The requirements file lists pytest, its ini-required plugins
    # (pytest-cov, pytest-asyncio), and the third-party imports used by
    # the scripts under test (pyyaml, numpy, boto3, ...).
    req_file = "scripts/requirements-test.txt"
    try:
        import pytest  # noqa: F401
        import yaml  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        print(f"Installing test deps from {req_file}...", flush=True)
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "--quiet",
                "--break-system-packages",
                "-r", req_file,
            ],
        )
        if result.returncode != 0:
            print("ERROR: failed to install test dependencies", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    ensure_pytest()
    # Run pytest on the scripts directory.
    # Note: the gate passes --quick, which is a wrapper-level flag (not a
    # pytest option), so it is not forwarded.
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "scripts/", "-q"],
        cwd=".",
    )
    sys.exit(result.returncode)
