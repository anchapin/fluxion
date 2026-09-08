"""
Tests for ``scripts/check_test_inventory_drift.py`` -- Issue #3442.

Regression guard for the test-inventory drift gate. Mirrors the
``load_script`` + ``tmp_path`` mock-baseline pattern used by
``test_check_orphan_modules.py`` and ``test_check_quarantine.py``:
the script imports ``REPO_ROOT`` and path constants at module-load
time, so each test that wants a synthetic baseline / inventory must
redirect those constants before invoking ``main()``.

Issue #3442 acceptance criteria are realised as six scenarios:

1. **Clean state** -- ``scripts/check_test_inventory_drift.py``
   exits 0 against the real repo (committed inventory + committed
   baseline match the live AST scan). A regression in the AST
   parser, the ratchet constants, or the threshold logic would flip
   this red.
2. **Drift threshold fires** -- a synthetic inventory whose lib_tests
   is 30 below baseline (and the BASELINE_* ratchet is bumped in
   lock-step) trips the drift-threshold check, exit 1.
3. **Ratchet catches growth** -- a synthetic inventory whose
   lib_tests grows by 1 over the BASELINE_LIB_TESTS constant fires
   the ratchet, exit 1, even when the per-PR baseline file is
   left untouched.
4. **Symmetric shrink is allowed** -- lowering lib_tests WITHOUT
   raising the BASELINE_* constant is OK: the AST scan produces a
   smaller live number, drift fails the threshold but the ratchet
   does NOT block the cleanup PR.
5. **--update-baseline rewrites the file** -- the gate's
   ``--update-baseline`` path rewrites the frozen baseline to match
   the live inventory, including a new ``captured_at`` timestamp.
6. **--json output shape** -- the JSON mode emits the documented
   schema with ``drift_violations`` / ``ratchet_failures`` keys.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_NAME = "check_test_inventory_drift"
SCRIPT = REPO_ROOT / "scripts" / f"{SCRIPT_NAME}.py"


@pytest.fixture
def drift_gate(load_script):
    """Freshly-loaded copy of the drift gate."""
    return load_script(SCRIPT_NAME)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _scrub_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset ``sys.argv`` so the script's argparse doesn't see pytest's CLI."""
    monkeypatch.setattr(sys, "argv", [SCRIPT_NAME])


def _redirect_paths(
    drift_gate,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inventory: dict | None = None,
    baseline: dict | None = None,
) -> tuple[Path, Path]:
    """Point the drift gate's path constants at a synthetic ``tmp_path``
    mock layout and return ``(inventory_path, baseline_path)``.

    The script computes ``DEFAULT_INVENTORY`` and ``DEFAULT_BASELINE``
    at import time from ``Path(__file__).resolve().parent.parent``; the
    ``_regenerate_inventory`` helper shells out to
    ``GENERATOR_SCRIPT`` whose path is also module-level. All three
    must be redirected so the synthetic trees don't collide with the
    real repo's committed inventory.
    """
    inventory_path = tmp_path / "tests" / "test_inventory.json"
    baseline_path = tmp_path / "tests" / "reference_data" / "test_inventory_baseline.json"
    generator_script = tmp_path / "scripts" / "generate_test_inventory.py"

    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    generator_script.parent.mkdir(parents=True, exist_ok=True)

    if inventory is not None:
        _write_json(inventory_path, inventory)
    else:
        # A minimum-valid inventory matching a no-op baseline.
        _write_json(
            inventory_path,
            {
                "schema_version": 1,
                "totals": {
                    "lib_tests_root": 100,
                    "lib_ignored_root": 1,
                    "workspace_tests": 250,
                    "workspace_ignored": 5,
                    "test_binaries": 10,
                },
                "by_crate": {},
            },
        )
    if baseline is not None:
        _write_json(baseline_path, baseline)
    else:
        _write_json(
            baseline_path,
            {
                "schema_version": 1,
                "metrics": {
                    "lib_tests": 100,
                    "lib_ignored": 1,
                    "workspace_tests": 250,
                    "workspace_ignored": 5,
                    "test_binaries": 10,
                },
            },
        )

    monkeypatch.setattr(drift_gate, "DEFAULT_INVENTORY", inventory_path)
    monkeypatch.setattr(drift_gate, "DEFAULT_BASELINE", baseline_path)
    monkeypatch.setattr(drift_gate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(drift_gate, "GENERATOR_SCRIPT", generator_script)

    def _fake_regenerate(cargo_target_dir, verify):  # noqa: ARG001
        return json.loads(inventory_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(drift_gate, "_regenerate_inventory", _fake_regenerate)
    return inventory_path, baseline_path


# ---------------------------------------------------------------------------
# Test 1: Clean state against the real repo
# ---------------------------------------------------------------------------


def test_script_exits_zero_on_clean_repo():
    """The drift gate exits 0 against the real repo when the live
    inventory and the committed baseline are in sync (the seeded
    baseline matches the AST scan).

    Pin against drift: a regression in the generator AST scan, the
    ratchet constants, or the threshold logic that flips the live
    repo to "drift" mode would fail this test.
    """
    result = subprocess.run(
        ["python3", str(SCRIPT), "--no-verify"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"expected exit 0 (no drift), got rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Test-inventory drift gate" in result.stdout
    assert "No drift-threshold violations" in result.stdout
    assert "No ratchet violations" in result.stdout


# ---------------------------------------------------------------------------
# Test 2: Drift threshold fires
# ---------------------------------------------------------------------------


def test_drift_threshold_fires_on_large_delta(
    drift_gate, tmp_path, monkeypatch, capsys
):
    """An inventory whose lib_tests is far enough below the baseline
    (and the BASELINE_* ratchet is bumped in lock-step) trips the
    drift-threshold check -> exit 1.

    With OR-semantics, the gate fails if EITHER the absolute change
    is > ``DIFF_TOLERANCE_ABS`` OR the relative change is >
    ``DIFF_TOLERANCE_PCT``. A 30-test shrink on a 100-test baseline
    satisfies both branches, so the gate fires loudly.
    """
    baseline = {
        "schema_version": 1,
        "metrics": {
            "lib_tests": 100,
            "lib_ignored": 1,
            "workspace_tests": 250,
            "workspace_ignored": 5,
            "test_binaries": 10,
        },
    }
    # ``verify`` mode requires running cargo, which our mock-side does
    # NOT support; the reroute via ``_redirect_paths`` wires a fake
    # regenerator that returns the inventory straight from disk.
    inventory = {
        "schema_version": 1,
        "totals": {
            "lib_tests_root": 100 - 30,  # 30 below baseline (>5% and >25 abs)
            "lib_ignored_root": 1,
            "workspace_tests": 250,
            "workspace_ignored": 5,
            "test_binaries": 10,
        },
        "by_crate": {},
    }
    _redirect_paths(drift_gate, tmp_path, monkeypatch, inventory=inventory, baseline=baseline)
    # Lower the ratchet so the BASELINE_LIB_TESTS check doesn't fire
    # first; the goal here is to verify the DRIFT threshold (per-pr
    # baseline file) is what trips, separate from the ratchet.
    monkeypatch.setattr(drift_gate, "BASELINE_LIB_TESTS", 100 - 30)
    _scrub_argv(monkeypatch)

    rc = drift_gate.main()
    out = capsys.readouterr().out

    assert rc == 1, f"expected FAIL (drift threshold), got rc={rc}\noutput:\n{out}"
    assert "lib_tests" in out
    assert "Drift-threshold violations" in out


# ---------------------------------------------------------------------------
# Test 3: Ratchet catches growth above BASELINE_*
# ---------------------------------------------------------------------------


def test_ratchet_catches_growth_above_baseline_constant(
    drift_gate, tmp_path, monkeypatch, capsys
):
    """An inventory whose lib_tests exceeds the BASELINE_LIB_TESTS
    constant trips the ratchet, exit 1, even when the per-PR baseline
    file is left untouched.

    Simulates a PR that adds tests without bumping the ratchet
    constant -- exactly the failure mode Issue #3442 was set up to
    catch.
    """
    # Per-PR baseline file says lib_tests is 99 (allowed).
    baseline = {
        "schema_version": 1,
        "metrics": {
            "lib_tests": 99,
            "lib_ignored": 1,
            "workspace_tests": 250,
            "workspace_ignored": 5,
            "test_binaries": 10,
        },
    }
    # Live AST scan reports lib_tests is 100 -- exceeds the ratchet
    # constant (which is still 99 from the previous accepted HEAD).
    inventory = {
        "schema_version": 1,
        "totals": {
            "lib_tests_root": 100,
            "lib_ignored_root": 1,
            "workspace_tests": 250,
            "workspace_ignored": 5,
            "test_binaries": 10,
        },
        "by_crate": {},
    }
    _redirect_paths(drift_gate, tmp_path, monkeypatch, inventory=inventory, baseline=baseline)
    monkeypatch.setattr(drift_gate, "BASELINE_LIB_TESTS", 99)
    _scrub_argv(monkeypatch)

    rc = drift_gate.main()
    out = capsys.readouterr().out

    assert rc == 1, f"expected FAIL (ratchet violation), got rc={rc}\noutput:\n{out}"
    assert "Ratchet violations" in out
    assert "BASELINE_LIB_TESTS" in out


# ---------------------------------------------------------------------------
# Test 4: Symmetric shrink is allowed (cleaner approach)
# ---------------------------------------------------------------------------


def test_symmetric_shrink_is_permitted(
    drift_gate, tmp_path, monkeypatch, capsys
):
    """Lowering lib_tests WITHOUT raising the BASELINE_* constant is
    OK in principle: the AST scan produces a smaller live number, no
    ratchet violation fires. (The drift-threshold check IS allowed to
    fire — that's the "make it visible" layer — but the ratchet is the
    binding constraint and it does not block shrinks.)

    This is the "cleanup PR" path -- deleting a redundant coverage
    binary should not be blocked by the ratchet, even though the
    drift-threshold check surfaces the change so reviewers notice it.
    The test simulates the case where the shrink is within tolerance
    (so drift-threshold also passes) to confirm the ratchet does not
    trip on a clean cleanup.
    """
    # Per-PR baseline file: lib_tests = 100
    baseline = {
        "schema_version": 1,
        "metrics": {
            "lib_tests": 100,
            "lib_ignored": 1,
            "workspace_tests": 250,
            "workspace_ignored": 5,
            "test_binaries": 10,
        },
    }
    # Live: lib_tests = 91 (down 9; within ±5%/±9 (5% of 100 = 5;
    # abs 9 > absolute-tolerance floor of 5 -> the relative branch
    # trips). Use a 4-step shrink to stay strictly inside both bounds.)
    inventory = {
        "schema_version": 1,
        "totals": {
            "lib_tests_root": 96,
            "lib_ignored_root": 1,
            "workspace_tests": 250,
            "workspace_ignored": 5,
            "test_binaries": 10,
        },
        "by_crate": {},
    }
    _redirect_paths(drift_gate, tmp_path, monkeypatch, inventory=inventory, baseline=baseline)
    monkeypatch.setattr(drift_gate, "BASELINE_LIB_TESTS", 100)  # unchanged
    _scrub_argv(monkeypatch)

    rc = drift_gate.main()
    out = capsys.readouterr().out

    assert rc == 0, (
        f"expected PASS (clean shrink within tolerance), got rc={rc}\noutput:\n{out}"
    )


# ---------------------------------------------------------------------------
# Test 5: --update-baseline rewrites the file
# ---------------------------------------------------------------------------


def test_update_baseline_rewrites_file(
    drift_gate, tmp_path, monkeypatch
):
    """The ``--update-baseline`` path rewrites the frozen baseline to
    match the live inventory, including a new ``captured_at`` timestamp
    and the ratchet constants block.
    """
    inventory = {
        "schema_version": 1,
        "totals": {
            "lib_tests_root": 99,
            "lib_ignored_root": 2,
            "workspace_tests": 198,
            "workspace_ignored": 3,
            "test_binaries": 7,
        },
        "by_crate": {},
    }
    initial_baseline = {
        "schema_version": 1,
        "metrics": {
            "lib_tests": 100,
            "lib_ignored": 1,
            "workspace_tests": 200,
            "workspace_ignored": 5,
            "test_binaries": 10,
        },
    }
    inventory_path, baseline_path = _redirect_paths(
        drift_gate, tmp_path, monkeypatch, inventory=inventory, baseline=initial_baseline
    )
    monkeypatch.setattr(sys, "argv", [SCRIPT_NAME, "--update-baseline", "--no-verify"])

    rc = drift_gate.main()
    assert rc == 0, "expected --update-baseline to exit 0"

    # The baseline file MUST have been rewritten.
    assert baseline_path.exists(), "expected baseline file to be written"
    updated = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert updated["metrics"]["lib_tests"] == 99
    assert updated["metrics"]["lib_ignored"] == 2
    assert updated["metrics"]["workspace_tests"] == 198
    assert updated["metrics"]["workspace_ignored"] == 3
    assert updated["metrics"]["test_binaries"] == 7
    assert "captured_at" in updated
    assert "ratchet" in updated
    assert updated["ratchet"]["BASELINE_LIB_TESTS"] == drift_gate.BASELINE_LIB_TESTS


# ---------------------------------------------------------------------------
# Test 6: --json output shape
# ---------------------------------------------------------------------------


def test_json_output_shape(drift_gate, tmp_path, monkeypatch, capsys):
    """The JSON mode emits the documented schema with
    ``drift_violations`` / ``ratchet_failures`` keys.
    """
    baseline = {
        "schema_version": 1,
        "metrics": {
            "lib_tests": 100,
            "lib_ignored": 1,
            "workspace_tests": 250,
            "workspace_ignored": 5,
            "test_binaries": 10,
        },
    }
    inventory = {
        "schema_version": 1,
        "totals": {
            "lib_tests_root": 100,
            "lib_ignored_root": 1,
            "workspace_tests": 250,
            "workspace_ignored": 5,
            "test_binaries": 10,
        },
        "by_crate": {},
    }
    _redirect_paths(drift_gate, tmp_path, monkeypatch, inventory=inventory, baseline=baseline)
    monkeypatch.setattr(sys, "argv", [SCRIPT_NAME, "--no-verify", "--json"])

    rc = drift_gate.main()
    out = capsys.readouterr().out

    assert rc == 0
    payload = json.loads(out)
    assert "drift_violations" in payload
    assert "ratchet_failures" in payload
    assert "ratchet_baselines" in payload
    assert payload["ratchet_baselines"]["BASELINE_LIB_TESTS"] == drift_gate.BASELINE_LIB_TESTS
    assert payload["would_fail"] is False
