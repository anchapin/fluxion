"""
Tests for ``scripts/memory-budget-gate.sh`` — issue #3766.

The pre-#3766 gate monitored ``$$`` (the wrapper shell itself) from a
background subshell, so it always reported a near-zero peak RSS and never
gated anything. The fixed script walks the whole child process tree of the
wrapped command (``pgrep -P`` BFS, summed ``ps -o rss=``), enforces
``--exit`` and system-headroom breaches by killing the child tree
immediately, and reports the aggregate peak.

The bash script is exercised via ``subprocess.run`` against real child
processes (a Python allocator that touches every page it reserves, then
sleeps so several 0.2 s monitor polls land inside the hold window), the
same pattern ``test_cleanup_root_strays.py`` uses for its hermetic
script-under-test runs. Headroom breach is simulated deterministically by
setting an absurdly large ``--headroom`` floor — no host memory
manipulation required.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "memory-budget-gate.sh"

# Allocator: reserve `mb` MB, force every page resident (calloc'd zero
# pages are not necessarily RSS until touched), then hold long enough for
# several monitor polls (0.2 s cadence).
ALLOCATOR_SNIPPET = """
import sys, time
mb = int(sys.argv[1])
buf = bytearray(mb * 1024 * 1024)
for i in range(0, len(buf), 4096):
    buf[i] = 1
time.sleep(3)
"""

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("python3") is None,
    reason="bash + python3 required to exercise the memory gate",
)


def _run_gate(*args: str, timeout: int = 90) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _allocator_command(mb: int) -> str:
    return f"python3 -c {shlex.quote(ALLOCATOR_SNIPPET)} {mb}"


def test_script_exists_and_is_executable():
    assert SCRIPT_PATH.exists(), f"missing gate at {SCRIPT_PATH}"
    assert SCRIPT_PATH.stat().st_mode & 0o111, "gate must be executable"


def test_script_syntax_valid():
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT_PATH)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_no_command_passes_with_zero_peak():
    result = _run_gate("--warn", "8")
    assert result.returncode == 0, result.stderr
    assert "Peak RSS: 0" in result.stdout
    assert "Memory budget check passed" in result.stdout


def test_successful_command_below_warn_exits_zero():
    result = _run_gate("--warn", "8", "--command", "echo gate-ok")
    assert result.returncode == 0, result.stderr
    assert "gate-ok" in result.stdout
    assert "Memory budget check passed" in result.stdout


def test_child_tree_memory_is_captured():
    """Acceptance criterion #1: the aggregate peak RSS of the child
    process tree is measured, not the wrapper shell's own RSS (which was
    the #3766 bug). A ~512 MB child must show up in the reported peak."""
    result = _run_gate("--warn", "8", "--command", _allocator_command(512))
    assert result.returncode == 0, result.stderr
    match = re.search(r"Peak RSS: ([0-9.]+) GB", result.stdout)
    assert match, f"no peak line in output:\n{result.stdout}"
    peak_gb = float(match.group(1))
    assert peak_gb >= 0.4, (
        f"child allocation (~512 MB) not captured; peak reported as {peak_gb} GB"
    )


def test_warn_threshold_catches_child_allocation():
    """Acceptance criterion #2 (--warn): a runaway child trips the warning
    exit code 1 even though the command itself succeeds."""
    result = _run_gate("--warn", "0", "--command", _allocator_command(512))
    assert result.returncode == 1, result.stdout
    assert "WARNING" in result.stderr


def test_exit_threshold_kills_runaway_child():
    """Acceptance criterion #2 (--exit): the gate kills the child tree at
    breach time instead of waiting for it to finish."""
    result = _run_gate("--exit", "0", "--command", _allocator_command(512))
    assert result.returncode == 2, result.stdout
    assert "exceeds exit threshold" in result.stderr
    # The child was killed mid-run, so the run must complete well before
    # the allocator's 3 s hold plus gate overhead would allow a natural
    # exit — this proves enforcement was active, not post-hoc.
    assert "killing tree" in result.stderr


def test_headroom_breach_kills_child_and_exits_4():
    """The system-headroom guard aborts when MemAvailable + SwapFree is
    below the floor; an absurd floor guarantees a deterministic breach."""
    result = _run_gate(
        "--headroom",
        "100000000",
        "--command",
        "sleep 30",
        timeout=30,
    )
    assert result.returncode == 4, result.stdout
    assert "headroom" in result.stderr.lower()
    assert "killing tree" in result.stderr


def test_child_failure_propagates_as_exit_3():
    result = _run_gate("--warn", "8", "--command", "exit 7")
    assert result.returncode == 3, result.stdout
    assert "Wrapped command exit code: 7" in result.stdout


def test_unknown_option_exits_2():
    result = _run_gate("--bogus")
    assert result.returncode == 2
    assert "Unknown option" in result.stderr
