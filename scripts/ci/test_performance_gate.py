"""
Tests for ``scripts/performance_gate.py`` -- the PR-time benchmark
regression gate (>10% perf regression fails).

The script's load-bearing pieces are pure:

* ``check_perf_regression(current, baseline)`` -- arithmetic against the
  10% threshold, returns a list of regression descriptors.
* ``BENCHMARK_THRESHOLD`` -- the constant 0.10.

The impure / subprocess-bound paths (``run_command``, ``get_main_branch_baseline``,
``run_benchmarks``, ``save_baseline``, ``main``) are exercised via the
``run_command`` injection stub so the tests are hermetic.

Pattern mirrors the other ``scripts/ci/test_*.py`` files: load the script
as a fresh module via the shared ``load_script`` fixture, then drive
``check_perf_regression`` and the ``--check`` CLI through a synthetic
``tmp_path`` repo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPT_NAME = "performance_gate"


@pytest.fixture
def gate(load_script):
    """Freshly-loaded copy of the performance gate."""
    return load_script(SCRIPT_NAME)


# ---------------------------------------------------------------------------
# BENCHMARK_THRESHOLD constant pin
# ---------------------------------------------------------------------------


def test_benchmark_threshold_is_ten_percent(gate):
    """The 10% threshold is the binding contract.

    A regression that bumped this to 0.20 (or 0.50) would silently extend
    the gate's tolerance. The constant is at module scope so it is
    directly assertable.
    """
    assert gate.BENCHMARK_THRESHOLD == 0.10


# ---------------------------------------------------------------------------
# check_perf_regression — pure threshold arithmetic
# ---------------------------------------------------------------------------


def test_check_perf_regression_returns_empty_when_within_threshold(gate):
    """Current +9% (under 10% threshold) → empty list."""
    current = {"bench_a": 1.09}
    baseline = {"bench_a": 1.0}
    assert gate.check_perf_regression(current, baseline) == []


def test_check_perf_regression_flags_eleven_percent_regression(gate):
    """Current +11% (over 10% threshold) → single regression."""
    current = {"bench_a": 1.11}
    baseline = {"bench_a": 1.0}
    regressions = gate.check_perf_regression(current, baseline)
    assert len(regressions) == 1
    assert "bench_a" in regressions[0]
    assert "+11.0%" in regressions[0]


def test_check_perf_regression_exactly_at_threshold(gate):
    """Current +9.99% → empty list (under 10% threshold).

    The script uses ``> BENCHMARK_THRESHOLD`` (strict). We deliberately
    pick a value that is unambiguously below 10% to avoid floating-point
    artefacts (`1.10 / 1.0 - 1` evaluates to ~0.09999...).
    """
    current = {"bench_a": 1.0999}
    baseline = {"bench_a": 1.0}
    assert gate.check_perf_regression(current, baseline) == []


def test_check_perf_regression_just_over_threshold(gate):
    """Current +10.5% → single regression (just over 10% threshold)."""
    current = {"bench_a": 1.105}
    baseline = {"bench_a": 1.0}
    regressions = gate.check_perf_regression(current, baseline)
    assert len(regressions) == 1
    assert "bench_a" in regressions[0]


def test_check_perf_regression_ignored_for_improvements(gate):
    """Performance improvements (negative pct) → empty list."""
    current = {"bench_a": 0.5}
    baseline = {"bench_a": 1.0}
    assert gate.check_perf_regression(current, baseline) == []


def test_check_perf_regression_skips_new_benchmarks(gate):
    """Benchmarks not in the baseline are skipped (no false positives)."""
    current = {"bench_new": 10.0, "bench_a": 1.0}
    baseline = {"bench_a": 1.0}
    assert gate.check_perf_regression(current, baseline) == []


def test_check_perf_regression_skips_zero_baseline(gate):
    """Baseline == 0 → skipped (avoids division-by-zero)."""
    current = {"bench_a": 5.0}
    baseline = {"bench_a": 0.0}
    assert gate.check_perf_regression(current, baseline) == []


def test_check_perf_regression_returns_multiple(gate):
    """Multiple regressions are reported in a single list."""
    current = {"bench_a": 1.5, "bench_b": 1.2, "bench_c": 1.0}
    baseline = {"bench_a": 1.0, "bench_b": 1.0, "bench_c": 1.0}
    regressions = gate.check_perf_regression(current, baseline)
    assert len(regressions) == 2
    names = {r.split(":")[0].strip() for r in regressions}
    assert names == {"bench_a", "bench_b"}


def test_check_perf_regression_message_format(gate):
    """Per-regression message includes baseline→current + percentage."""
    current = {"bench_a": 1.25}
    baseline = {"bench_a": 1.0}
    msg = gate.check_perf_regression(current, baseline)[0]
    assert "bench_a" in msg
    assert "1.0000s" in msg
    assert "1.2500s" in msg
    assert "25.0%" in msg


# ---------------------------------------------------------------------------
# run_command — pure subprocess wrapper
# ---------------------------------------------------------------------------


def test_run_command_returns_stdout_and_zero_on_success(gate):
    """A successful subprocess is returned with rc=0."""
    stdout, rc = gate.run_command(["true"], timeout=5)
    assert rc == 0
    assert isinstance(stdout, str)


def test_run_command_returns_nonzero_on_failure(gate):
    """A failing subprocess is returned with rc != 0."""
    stdout, rc = gate.run_command(["false"], timeout=5)
    assert rc != 0


def test_run_command_returns_124_on_timeout(gate):
    """Subprocess timeout is surfaced as return code 124 (timeout(1) convention)."""
    # Use a guaranteed-slow command: sleep 10 with a 1s timeout.
    stdout, rc = gate.run_command(["sleep", "10"], timeout=1)
    assert rc == 124
    assert stdout == ""


# ---------------------------------------------------------------------------
# get_main_branch_baseline — pure file-loader logic
# ---------------------------------------------------------------------------


def test_get_main_branch_baseline_returns_empty_when_no_file(gate, tmp_path, monkeypatch):
    """``BASELINE_FILE`` does not exist → empty dict (no error)."""
    fake_baseline = tmp_path / "no-such-baseline.json"
    monkeypatch.setattr(gate, "BASELINE_FILE", fake_baseline)
    # Suppress the git checkout logic by recording that the inner
    # subprocess.run is never called.
    calls = {"git": 0}

    def fake_run(cmd, *args, **kwargs):
        calls["git"] += 1
        return ("", 0)

    monkeypatch.setattr(gate, "run_command", fake_run)
    result = gate.get_main_branch_baseline()
    assert result == {}
    # The git stash / checkout branch must NOT be entered when the file
    # was missing on disk.
    assert calls["git"] == 0


def test_get_main_branch_baseline_reads_existing_file(gate, tmp_path, monkeypatch):
    """Existing ``.perf_baseline.json`` is loaded into the dict."""
    fake_baseline = tmp_path / ".perf_baseline.json"
    fake_baseline.write_text(json.dumps({"bench_a": 1.0, "bench_b": 2.0}))
    monkeypatch.setattr(gate, "BASELINE_FILE", fake_baseline)
    # Stub the git stash / checkout flow so the test never sees the real
    # repo's working tree.
    def fake_run(cmd, *args, **kwargs):
        return ("", 0)

    monkeypatch.setattr(gate, "run_command", fake_run)
    result = gate.get_main_branch_baseline()
    assert result == {"bench_a": 1.0, "bench_b": 2.0}


# ---------------------------------------------------------------------------
# save_baseline — JSON-write helper
# ---------------------------------------------------------------------------


def test_save_baseline_writes_json(gate, tmp_path, monkeypatch):
    """``save_baseline`` writes a JSON file at ``BASELINE_FILE``."""
    fake_baseline = tmp_path / ".perf_baseline.json"
    monkeypatch.setattr(gate, "BASELINE_FILE", fake_baseline)
    gate.save_baseline({"bench_a": 1.5})
    assert json.loads(fake_baseline.read_text()) == {"bench_a": 1.5}


# ---------------------------------------------------------------------------
# main() — CLI driven through stubbed interfaces
# ---------------------------------------------------------------------------


def _stub_git(gate, monkeypatch, *, branch: str = "feature") -> None:
    """Replace ``run_command`` and ``get_git_branch`` for a hermetic CLI run."""

    def fake_run(cmd, *args, **kwargs):
        # Honour the "real" semantics: any git command returns 0.
        if cmd and cmd[0] == "git":
            return ("", 0)
        # `cargo bench` returns no benchmarks → orchestrator exits 1.
        return ("no benchmarks parsed", 0)

    monkeypatch.setattr(gate, "run_command", fake_run)
    monkeypatch.setattr(gate, "get_git_branch", lambda: branch)


def test_main_returns_zero_when_no_regressions(gate, tmp_path, monkeypatch, capsys):
    """``--check`` with synthetic current == baseline → exit 0."""
    # Plant an empty baseline so the script does not exit 0 for "no baseline".
    fake_baseline = tmp_path / ".perf_baseline.json"
    fake_baseline.write_text(json.dumps({"bench_a": 1.0}))
    monkeypatch.setattr(gate, "BASELINE_FILE", fake_baseline)

    # Inject a synthetic current via run_benchmarks.
    def fake_run_benchmarks():
        return {"bench_a": 1.05}  # 5% improvement, well under threshold

    _stub_git(gate, monkeypatch, branch="feature")
    monkeypatch.setattr(gate, "run_benchmarks", fake_run_benchmarks)

    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME, "--check"]
    try:
        rc = gate.main()
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 0
    finally:
        sys.argv[:] = saved

    out = capsys.readouterr().out
    assert rc == 0, f"expected exit 0, got {rc}\noutput:\n{out}"
    assert "within threshold" in out or "PASS" in out


def test_main_returns_one_when_regression_detected(gate, tmp_path, monkeypatch, capsys):
    """``--check`` with a +50% regression → exit 1."""
    fake_baseline = tmp_path / ".perf_baseline.json"
    fake_baseline.write_text(json.dumps({"bench_a": 1.0}))
    monkeypatch.setattr(gate, "BASELINE_FILE", fake_baseline)

    def fake_run_benchmarks():
        return {"bench_a": 1.5}  # 50% regression

    _stub_git(gate, monkeypatch, branch="feature")
    monkeypatch.setattr(gate, "run_benchmarks", fake_run_benchmarks)

    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME, "--check"]
    try:
        rc = gate.main()
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 0
    finally:
        sys.argv[:] = saved

    out = capsys.readouterr().out
    assert rc == 1, f"expected exit 1, got {rc}\noutput:\n{out}"
    assert "REGRESSION" in out.upper() or "regression" in out.lower()
    assert "bench_a" in out


def test_main_returns_zero_when_no_baseline_on_pr(gate, tmp_path, monkeypatch, capsys):
    """``--check`` on a PR with no baseline → exit 0 (graceful skip)."""
    fake_baseline = tmp_path / "missing-baseline.json"
    monkeypatch.setattr(gate, "BASELINE_FILE", fake_baseline)

    def fake_run_benchmarks():
        return {"bench_a": 1.0}

    _stub_git(gate, monkeypatch, branch="feature")
    monkeypatch.setattr(gate, "run_benchmarks", fake_run_benchmarks)

    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME, "--check"]
    try:
        rc = gate.main()
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 0
    finally:
        sys.argv[:] = saved

    out = capsys.readouterr().out
    assert rc == 0
    assert "No baseline" in out or "baseline" in out.lower()


# ---------------------------------------------------------------------------
# PROJECT_ROOT pinning
# ---------------------------------------------------------------------------


def test_project_root_is_repo_parent(gate, repo_root):
    """``PROJECT_ROOT`` must be the parent of the scripts directory."""
    assert gate.PROJECT_ROOT == repo_root
