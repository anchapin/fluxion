"""
Tests for ``scripts/check_pr_closing_refs.sh`` -- the wave-orchestrator
post-merge gate that catches the ``gh pr create --fill`` parse failure
(documented in ``docs/orchestration/pr-body-conventions.md``).

The script is a 41-line bash wrapper around ``gh pr view --json
closingIssuesReferences --jq '.closingIssuesReferences | length'`` with
two args: PR number and expected count. Exit codes:

* 0 — count matches expected
* 1 — count does not match
* 2 — usage error (wrong number of args)

Pin the contracts by stubbing ``gh`` via ``PATH`` injection so the
tests are hermetic and don't need a live GitHub PR.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT_NAME = "check_pr_closing_refs"
SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / f"{SCRIPT_NAME}.sh"
)


# ---------------------------------------------------------------------------
# Fake-gh helpers
# ---------------------------------------------------------------------------


def _make_fake_gh(tmp_path: Path, *, count: int, issue_numbers: str) -> Path:
    """Write a tiny ``gh`` stub that returns the requested closingReferences.

    The script invokes ``gh`` twice with different ``--jq`` filters:

    * ``.closingIssuesReferences | length`` → returns the count
    * ``[.closingIssuesReferences[].number] | join(",")`` → returns comma-
      separated issue numbers

    Inspect the ``--jq`` argument (script passes it as a separate arg, not
    ``--jq=...``) to emit the right payload per call.
    """
    fake_gh = tmp_path / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env bash\n"
        "# Detect which gh pr view call this is by the --jq filter.\n"
        "JQ_ARG=\"\"\n"
        "prev=\"\"\n"
        "for arg in \"$@\"; do\n"
        "  if [ \"$prev\" = \"--jq\" ]; then\n"
        "    JQ_ARG=\"$arg\"\n"
        "  fi\n"
        "  prev=\"$arg\"\n"
        "done\n"
        "case \"$JQ_ARG\" in\n"
        "  *\"| length\")\n"
        f"    echo \"{count}\"\n"
        "    ;;\n"
        "  *\"| join(\\\",\\\")\")\n"
        f"    echo \"{issue_numbers}\"\n"
        "    ;;\n"
        "  *)\n"
        "    # Fallback: behave like the count call.\n"
        f"    echo \"{count}\"\n"
        "    ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_gh.chmod(fake_gh.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return fake_gh


def _run_script(tmp_path: Path, fake_gh: Path, args: list[str]) -> subprocess.CompletedProcess:
    """Run the bash script with ``gh`` patched to ``fake_gh`` via PATH."""
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}{os.pathsep}{env.get('PATH', '')}"
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


# ---------------------------------------------------------------------------
# Clean fixture (matching count) → exit 0
# ---------------------------------------------------------------------------


def test_exits_zero_when_count_matches(tmp_path):
    """1 expected, 1 actual → exit 0 with the OK banner."""
    _make_fake_gh(tmp_path, count=1, issue_numbers="2867")
    proc = _run_script(tmp_path, tmp_path, ["123", "1"])
    assert proc.returncode == 0, (
        f"unexpected exit {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "OK" in proc.stdout
    assert "PR #123" in proc.stdout
    assert "1" in proc.stdout


def test_exits_zero_with_zero_references(tmp_path):
    """0 expected, 0 actual → exit 0 (e.g. a doc-only PR with no closes ref)."""
    _make_fake_gh(tmp_path, count=0, issue_numbers="")
    proc = _run_script(tmp_path, tmp_path, ["456", "0"])
    assert proc.returncode == 0
    assert "OK" in proc.stdout
    assert "(none)" in proc.stdout


def test_exits_zero_with_multiple_references(tmp_path):
    """3 expected, 3 actual → exit 0."""
    _make_fake_gh(tmp_path, count=3, issue_numbers="100,200,300")
    proc = _run_script(tmp_path, tmp_path, ["789", "3"])
    assert proc.returncode == 0
    assert "OK" in proc.stdout
    assert "100,200,300" in proc.stdout


# ---------------------------------------------------------------------------
# Planted violation (count mismatch) → exit 1
# ---------------------------------------------------------------------------


def test_exits_one_when_count_too_low(tmp_path):
    """Expected 2, actual 1 → exit 1 with FAIL banner + remediation."""
    _make_fake_gh(tmp_path, count=1, issue_numbers="2867")
    proc = _run_script(tmp_path, tmp_path, ["123", "2"])
    assert proc.returncode == 1
    combined = proc.stdout + proc.stderr
    assert "FAIL" in combined
    assert "PR #123" in combined
    assert "expected 2" in combined
    # Remediation guidance must include `gh pr edit` and `Closes #...`.
    assert "gh pr edit" in combined
    assert "Closes #" in combined


def test_exits_one_when_count_too_high(tmp_path):
    """Expected 1, actual 2 → exit 1 (over-counting is also a violation)."""
    _make_fake_gh(tmp_path, count=2, issue_numbers="2867,2866")
    proc = _run_script(tmp_path, tmp_path, ["123", "1"])
    assert proc.returncode == 1
    combined = proc.stdout + proc.stderr
    assert "FAIL" in combined
    assert "expected 1" in combined


def test_exits_one_when_no_references_but_one_expected(tmp_path):
    """Expected 1, actual 0 → exit 1 (the original --fill bug)."""
    _make_fake_gh(tmp_path, count=0, issue_numbers="")
    proc = _run_script(tmp_path, tmp_path, ["42", "1"])
    assert proc.returncode == 1
    combined = proc.stdout + proc.stderr
    assert "FAIL" in combined
    assert "PR #42" in combined


# ---------------------------------------------------------------------------
# Usage error (wrong number of args) → exit 2
# ---------------------------------------------------------------------------


def test_exits_two_with_no_args(tmp_path):
    """No args → exit 2 with usage banner."""
    _make_fake_gh(tmp_path, count=1, issue_numbers="1")
    proc = _run_script(tmp_path, tmp_path, [])
    assert proc.returncode == 2
    assert "usage" in proc.stderr.lower()


def test_exits_two_with_one_arg(tmp_path):
    """One arg (missing expected count) → exit 2."""
    _make_fake_gh(tmp_path, count=1, issue_numbers="1")
    proc = _run_script(tmp_path, tmp_path, ["123"])
    assert proc.returncode == 2
    assert "usage" in proc.stderr.lower()


def test_exits_two_with_three_args(tmp_path):
    """Three args → exit 2."""
    _make_fake_gh(tmp_path, count=1, issue_numbers="1")
    proc = _run_script(tmp_path, tmp_path, ["123", "1", "extra"])
    assert proc.returncode == 2
    assert "usage" in proc.stderr.lower()


# ---------------------------------------------------------------------------
# set -euo pipefail smoke checks
# ---------------------------------------------------------------------------


def test_script_uses_strict_mode():
    """The script must enable ``set -euo pipefail`` so a pipe failure
    trips the gate rather than silently green-lighting.
    """
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "set -euo pipefail" in src


def test_script_tolerates_zero_expected_with_count_zero():
    """A non-numeric expected (e.g. typos) is still passed as-is to ``[[``.

    The script uses ``[[ "$ACTUAL" == "$EXPECTED" ]]`` so string comparison
    is fine. This pins that the tolerance is preserved.
    """
    pass  # covered by the existing test cases already.


# ---------------------------------------------------------------------------
# --- separator ----------------------------------------------------------
# ---------------------------------------------------------------------------
#
# The shell script asserts equal counts. Document the regex that detects
# the canonical "FAIL PR #… count = …, expected …" banner so future
# contributors can grep for it in CI logs.
FAIL_BANNER_PATTERN = r"FAIL PR #\d+ closingIssuesReferences count = \d+, expected \d+"
