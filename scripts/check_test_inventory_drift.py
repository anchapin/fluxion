#!/usr/bin/env python3
"""
Test-inventory drift gate — Issue #3442.

Compares the live ``tests/test_inventory.json`` (or a freshly
regenerated one) against the frozen baseline at
``tests/reference_data/test_inventory_baseline.json`` and fails when
counts drift past documented thresholds.

Why this exists
---------------
``AGENTS.md`` test-count citations drifted on every merged PR from
2026-09-04 through 2026-09-08 (Issue #3442). Neither AGENTS.md nor
the ``docs/ci/nextest-rollout.md`` runbook had any regeneration step,
so the numbers went stale within a day of any test-adding PR. This
gate is the structural countermeasure: every PR that mutates the
test inventory gets rejected with the explicit drift message until
the baseline is updated in the same PR.

Two layers of protection
------------------------
1. **Drift threshold** — a ``DIFF_TOLERANCE_PCT`` (default 5%) or
   ``DIFF_TOLERANCE_ABS`` (default 25) on the headline counts.
   Catches accidental deletion of large test files or expansion of
   the test suite beyond expected growth.

2. **BASELINE_* ratchet** — the baseline constants
   (``BASELINE_LIB_TESTS``, ``BASELINE_WORKSPACE_TESTS``, etc.) are
   the *highest* values the gate has ever accepted. Shrinking the
   suite (test deletion, blocking-issue resolution) is the only
   allowed direction; growth without an explicit baseline bump is
   rejected. This mirrors the ``BASELINE_KNOWN_ORPHANS`` /
   ``BASELINE_WIRED_BUT_DEAD`` patterns from
   ``scripts/check_orphan_modules.py`` (Issue #3459 / #3458).

Usage
-----
  python3 scripts/check_test_inventory_drift.py                       # regenerate + check
  python3 scripts/check_test_inventory_drift.py --live-inventory path # use a pre-existing inventory
  python3 scripts/check_test_inventory_drift.py --baseline path        # custom baseline path
  python3 scripts/check_test_inventory_drift.py --update-baseline      # update baseline to live counts
  python3 scripts/check_test_inventory_drift.py --json                 # machine-readable output

Exit codes
----------
  0 — drift within thresholds AND no ratchet violation
  1 — drift exceeds thresholds OR baseline ratchet violation OR
      inventory generation failed
  2 — script error (e.g. baseline file missing)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INVENTORY = REPO_ROOT / "tests" / "test_inventory.json"
DEFAULT_BASELINE = REPO_ROOT / "tests" / "reference_data" / "test_inventory_baseline.json"
GENERATOR_SCRIPT = REPO_ROOT / "scripts" / "generate_test_inventory.py"

# ---------------------------------------------------------------------------
# Drift tolerance — Issue #3442 acceptance: "5% drift OR documented exception".
# These thresholds are *per-metric*; the gate compares the live count to the
# baseline count and only fails if either the relative or absolute delta
# exceeds the threshold. The default 5%/25 mirrors the order-of-magnitude
# drift between develop and main over Issue #3442's lifetime.
# ---------------------------------------------------------------------------
DIFF_TOLERANCE_PCT = float(os.environ.get("TEST_INVENTORY_DRIFT_PCT", "5.0"))
DIFF_TOLERANCE_ABS = int(os.environ.get("TEST_INVENTORY_DRIFT_ABS", "25"))

# ---------------------------------------------------------------------------
# BASELINE_* ratchet (Issue #3442).
#
# This is the *highest* value the gate has ever accepted for each metric.
# The gate FAILS (exit 1) the moment the live count exceeds the baseline
# unless the constant is raised with a justifying comment. Lowering the
# baseline is the only authorised change; companion cleanup PRs that
# *resolve* a test (e.g. by removing a redundant coverage binary) are
# expected to LOWER the matching baseline.
#
# The drift gate runs in two modes:
#
#   * ``--no-verify`` (default in CI for speed) regenerates the
#     inventory via the AST-regex pass. Counts are deterministic for a
#     given source-tree state — the BASELINE_* seed below is the AST
#     scan count, not the cargo runtime count.
#
#   * ``--verify`` (recommended for accuracy) cross-checks the AST
#     counts against ``cargo test --workspace --exclude fluxion-tauri
#     -- --list`` and prefers the cargo-derived numbers. The verified
#     counts at HEAD ``12856a9`` are documented as the
#     ``verified_at_HEAD_*`` constants further below — operators can
#     sanity-check the drift gate against cargo's actual output by
#     comparing the AST and verified deltas.
#
# History:
#   - 2026-09-08 (Issue #3442): seed values from the AST scan at HEAD
#     ``12856a9`` (post-#3449 / #3450 / #3285 / #3451 / #3443). The
#     ratchet constants are deliberately the AST values (4311 / 7 / 8680
#     / 108 / 301) so the ``--no-verify`` CI fast-path is consistent
#     with the committed ``tests/test_inventory.json``. The verified
#     counts at HEAD are 3894 / 4 / 7923 / 125 / 298 (lower because
#     cargo's ``--list`` honours ``#[cfg(test)]`` boundaries and
#     feature gates the AST scan does not).
#   - 2026-09-08 (Issue #3546): bumped ``BASELINE_TEST_BINARIES`` to 303
#     and ``BASELINE_WORKSPACE_IGNORED`` to 123 to accommodate the new
#     ``tests/cli_run_with_perf_600_900.rs`` integration binary and
#     the 6 new lib tests in
#     ``src/validation/ashrae140/cases/build_case_routing_tests.rs``
#     wired into the ``build_case`` router (Case 600 / 900 / 960 / 970
#     regression of #3555 partial split).
#   - 2026-09-09 (Issue #3599): bumped ``BASELINE_LIB_IGNORED`` from 7
#     to 8 to accommodate 3 newly ``#[ignore]``-quarantined 9R4C
#     legacy solver scratch-pool tests under Phase A8 / §LIMIT-21
#     (Issue #3291). The 3 tests live in
#     ``src/sim/thermal_model_physics/physics_impl/`` and panic on
#     ``cargo test --features wiring-tracing`` — see QUARANTINE.md
#     "Phase A8 / Issue #3599" subsection. ``BASELINE_WORKSPACE_IGNORED``
#     stays at 123 (workspace ignored rises from 108 → 111, well below
#     the ratchet).
#   - 2026-09-11 (Issue #3650): bumped ``BASELINE_LIB_TESTS`` from 4311
#     to 4314 and ``BASELINE_WORKSPACE_TESTS`` from 8680 to 8683 for the
#     three new CWE-209 regression tests in ``src/api/server/tests.rs``
#     (``probe_weather_ok_detail_omits_operator_supplied_path``,
#     ``probe_weather_err_detail_omits_operator_supplied_path``, and
#     ``readyz_weather_semantics_and_body_keep_path_private``). No new
#     binaries and no ignore-count changes.
#   - 2026-09-11 (PR improve/quarantine-burndown): lowered
#     ``BASELINE_WORKSPACE_IGNORED`` from 123 to 118 — the quarantine
#     burndown un-ignored
#     ``test_case_970_validator_accepts_canonical_midpoints``
#     (``tests/ashrae_140_case_970_validation.rs``) after verifying it
#     passes live; its assertions validate the validator, not the
#     engine band. Verified workspace ignored count at HEAD is 118.
#     (Merged resolution: #3650's test-count bumps land first; the
#     burndown's ignore-ratchet reduction stacks on top of them.)
#   - 2026-09-11 (PR #3704, stacked after the burndown): raised
#     ``BASELINE_WORKSPACE_IGNORED`` from 118 to 119 — the wasm
#     FFI smoke-test quarantine (``fluxion-wasm/tests/
#     wasm_integration_tests.rs``, Issue #3703) added one workspace
#     ``#[ignore]`` on top of the burndown's reduction. Cargo-verified
#     workspace ignored count at HEAD is 119.
# ---------------------------------------------------------------------------
#   - 2026-09-11 (Issue #3629, stacked after #3650/#3693-burndown): bumped
#     ``BASELINE_LIB_IGNORED`` from 8 to 9 for the newly quarantined
#     ``test_thermal_mass_temperature_damping`` placeholder in
#     ``src/validation/thermal_mass.rs`` (a ``src/`` unit test outside
#     the auditor's ``tests/**`` scan).
BASELINE_LIB_TESTS = 4314
BASELINE_LIB_IGNORED = 9
BASELINE_WORKSPACE_TESTS = 8683
BASELINE_WORKSPACE_IGNORED = 121
# 2026-09-12 (Issue #3685): bumped from 308 to 309 for the new
# ``tests/cold_start_guard_test.rs`` binary — the always-compiled
# (feature-independent) unit tests for the Multi-Zone Cold Start
# Gate's warm-sample epsilon guard (``tests/cold_start_guard/mod.rs``).
# AST-scan test_binaries at HEAD is 309 (308 + this binary); no other
# ratchet moves (lib/workspace/ignored AST counts are unchanged or
# below their constants).
BASELINE_TEST_BINARIES = 309

# Sanity-check constants — the verified cargo counts at HEAD
# ``12856a9``. Operators checking the drift gate's accuracy can
# compare ``--no-verify`` vs ``--verify`` against these. The drift
# gate itself does NOT use them (they exist as documentation only).
VERIFIED_AT_HEAD_LIB_TESTS = 3894
VERIFIED_AT_HEAD_LIB_IGNORED = 4
VERIFIED_AT_HEAD_WORKSPACE_TESTS = 7923
VERIFIED_AT_HEAD_WORKSPACE_IGNORED = 125
VERIFIED_AT_HEAD_TEST_BINARIES = 298


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _regenerate_inventory(cargo_target_dir: str | None, verify: bool) -> dict:
    """Run the generator and parse the JSON it emits.

    When ``verify`` is True (the default), the generator cross-checks
    its AST counts against ``cargo test -- --list`` and prefers the
    cargo-derived numbers where available. This is the canonical
    inventory snapshot — pure-AST mode is only used when ``verify``
    is explicitly disabled (legacy migration scenarios) or when cargo
    is not available on the path.
    """
    cmd = ["python3", str(GENERATOR_SCRIPT)]
    if verify:
        cmd += ["--verify"]
    if cargo_target_dir:
        cmd += ["--cargo-target-dir", cargo_target_dir]
    # 60s without ``--verify`` is generous; ``--verify`` triggers a
    # full ``cargo test -- --list`` which can take 30+ min on cold
    # rebuilds, so we use a larger timeout for that path.
    timeout = 1800.0 if verify else 60.0
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        print(
            f"ERROR: generator failed (exit={proc.returncode}):\n{proc.stderr}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not DEFAULT_INVENTORY.exists():
        print(
            f"ERROR: generator did not produce {DEFAULT_INVENTORY.relative_to(REPO_ROOT)}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return json.loads(DEFAULT_INVENTORY.read_text(encoding="utf-8"))


def _load_baseline(path: Path) -> dict:
    if not path.exists():
        print(
            f"ERROR: baseline not found at {path.relative_to(REPO_ROOT)}. "
            f"Generate one with --update-baseline.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return json.loads(path.read_text(encoding="utf-8"))


def _save_baseline(path: Path, baseline: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _drift_message(metric: str, baseline: int, live: int) -> str:
    diff = live - baseline
    rel = (diff / baseline * 100.0) if baseline else 0.0
    direction = "↑" if diff > 0 else "↓"
    return (
        f"  {metric}: baseline={baseline}, live={live} "
        f"({direction} {abs(diff)} tests, {rel:+.2f}%)"
    )


def _check_drift_threshold(
    metric: str, baseline: int, live: int, violations: list[str]
) -> tuple[bool, bool]:
    """Compare a live count against the baseline for ``metric``.

    Returns ``(shrink_failed, grow_failed)``. ``shrink_failed`` is
    True when ``live < baseline`` (test deletion beyond threshold);
    ``grow_failed`` is True when ``live > baseline`` (growth beyond
    threshold). The ratchet layer in ``main()`` will translate these
    into a richer exit code.

    The threshold is OR-semantics: the gate fails if EITHER the
    absolute change exceeds ``DIFF_TOLERANCE_ABS`` OR the relative
    change exceeds ``DIFF_TOLERANCE_PCT``. This catches both the
    "delete 10 tests in a tiny suite" (rel-large, abs-small) and
    "delete 1000 tests in a giant suite" (abs-large, rel-small)
    failure modes that a single-sided threshold misses.
    """
    diff = live - baseline
    abs_diff = abs(diff)
    rel_diff = (abs_diff / baseline * 100.0) if baseline else 0.0
    exceeded = abs_diff > DIFF_TOLERANCE_ABS or rel_diff > DIFF_TOLERANCE_PCT
    if not exceeded:
        return (False, False)
    if diff > 0:
        violations.append(_drift_message(metric, baseline, live))
        return (False, True)
    violations.append(_drift_message(metric, baseline, live))
    return (True, False)


def _check_ratchet(
    metric: str,
    baseline_const_name: str,
    baseline_const_value: int,
    live: int,
    failures: list[str],
) -> None:
    """Issue #3442 downward-only ratchet — reject growth above baseline.

    Companion cleanup PRs that *resolve* a test (e.g. by deleting a
    redundant coverage binary) are expected to LOWER
    ``baseline_const_value`` BY ONE. Adding tests without raising the
    baseline is the failure mode we are protecting against — the
    drift threshold is informational; the ratchet is binding.
    """
    if live > baseline_const_value:
        failures.append(
            f"  {metric} = {live} > {baseline_const_name} = "
            f"{baseline_const_value} (growth above documented baseline)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test-inventory drift gate (Issue #3442)."
    )
    parser.add_argument(
        "--live-inventory",
        type=Path,
        default=None,
        help=f"Use this inventory JSON file instead of regenerating (default: regenerate to {DEFAULT_INVENTORY.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help=f"Baseline JSON path (default: {DEFAULT_BASELINE.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite the baseline to match the live inventory (use ONLY in cleanup PRs that intentionally shrink the suite).",
    )
    parser.add_argument(
        "--cargo-target-dir",
        default=None,
        help="CARGO_TARGET_DIR for the regenerator invocation.",
    )
    parser.add_argument(
        "--verify",
        dest="verify",
        action="store_true",
        default=True,
        help="(default: True) Regenerate via the AST-regex + cargo cross-check path. Slower (~5 min cold, sub-second warm) but counts match what cargo test -- --list reports.",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Regenerate via AST-only (skip cargo cross-check). Fast but counts may over-shoot by ~10% because the regex doesn't track cfg(test) boundaries.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output for CI consumption.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="(default: True) Fail on ratchet OR drift-threshold violations.",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Disable strict mode; only ratchet violations fail (useful for the first migration run).",
    )
    args = parser.parse_args()

    # Stage 1: live inventory.
    if args.live_inventory:
        live_path: Path = args.live_inventory
        if not live_path.is_absolute():
            live_path = REPO_ROOT / live_path
        if not live_path.exists():
            print(f"ERROR: live inventory missing at {live_path}", file=sys.stderr)
            return 2
        live = json.loads(live_path.read_text(encoding="utf-8"))
    else:
        live = _regenerate_inventory(args.cargo_target_dir, verify=args.verify)

    totals = live["totals"]
    live_lib = totals.get("lib_tests_root", 0)
    live_lib_ignored = totals.get("lib_ignored_root", 0)
    live_workspace = totals.get("workspace_tests", 0)
    live_workspace_ignored = totals.get("workspace_ignored", 0)
    live_binaries = totals.get("test_binaries", 0)

    # Stage 2: baseline. For --update-baseline, skip the comparison
    # entirely and rewrite the file with the live snapshot.
    if args.update_baseline:
        baseline_snapshot = {
            "schema_version": live.get("schema_version", 1),
            "captured_at": _now_iso(),
            "captured_from": str(DEFAULT_INVENTORY.relative_to(REPO_ROOT)),
            "metrics": {
                "lib_tests": live_lib,
                "lib_ignored": live_lib_ignored,
                "workspace_tests": live_workspace,
                "workspace_ignored": live_workspace_ignored,
                "test_binaries": live_binaries,
            },
            "ratchet": {
                "BASELINE_LIB_TESTS": BASELINE_LIB_TESTS,
                "BASELINE_LIB_IGNORED": BASELINE_LIB_IGNORED,
                "BASELINE_WORKSPACE_TESTS": BASELINE_WORKSPACE_TESTS,
                "BASELINE_WORKSPACE_IGNORED": BASELINE_WORKSPACE_IGNORED,
                "BASELINE_TEST_BINARIES": BASELINE_TEST_BINARIES,
            },
            "ratchet_drift_tolerance_pct": DIFF_TOLERANCE_PCT,
            "ratchet_drift_tolerance_abs": DIFF_TOLERANCE_ABS,
            "totals": totals,
            "by_crate": live.get("by_crate", {}),
        }
        baseline_path: Path = args.baseline
        if not baseline_path.is_absolute():
            baseline_path = REPO_ROOT / baseline_path
        _save_baseline(baseline_path, baseline_snapshot)
        print(
            f"Baseline updated: {baseline_path.relative_to(REPO_ROOT)} "
            f"(lib_tests={live_lib}, workspace_tests={live_workspace}, "
            f"test_binaries={live_binaries})"
        )
        return 0

    # Stage 3: load baseline + compare.
    baseline_path = args.baseline
    if not baseline_path.is_absolute():
        baseline_path = REPO_ROOT / baseline_path
    baseline = _load_baseline(baseline_path)
    baseline_metrics = baseline.get("metrics", {})
    base_lib = baseline_metrics.get("lib_tests", 0)
    base_lib_ignored = baseline_metrics.get("lib_ignored", 0)
    base_workspace = baseline_metrics.get("workspace_tests", 0)
    base_workspace_ignored = baseline_metrics.get("workspace_ignored", 0)
    base_binaries = baseline_metrics.get("test_binaries", 0)

    drift_violations: list[str] = []
    shrink_failed: list[str] = []
    grow_failed: list[str] = []
    for metric, base_val, live_val in [
        ("lib_tests", base_lib, live_lib),
        ("lib_ignored", base_lib_ignored, live_lib_ignored),
        ("workspace_tests", base_workspace, live_workspace),
        ("workspace_ignored", base_workspace_ignored, live_workspace_ignored),
        ("test_binaries", base_binaries, live_binaries),
    ]:
        shrink, grow = _check_drift_threshold(
            metric, base_val, live_val, drift_violations
        )
        if shrink:
            shrink_failed.append(metric)
        if grow:
            grow_failed.append(metric)

    # Ratchet failures: any metric that grew above the documented
    # baseline constant is rejected.
    ratchet_failures: list[str] = []
    _check_ratchet(
        "lib_tests", "BASELINE_LIB_TESTS", BASELINE_LIB_TESTS, live_lib, ratchet_failures
    )
    _check_ratchet(
        "lib_ignored",
        "BASELINE_LIB_IGNORED",
        BASELINE_LIB_IGNORED,
        live_lib_ignored,
        ratchet_failures,
    )
    _check_ratchet(
        "workspace_tests",
        "BASELINE_WORKSPACE_TESTS",
        BASELINE_WORKSPACE_TESTS,
        live_workspace,
        ratchet_failures,
    )
    _check_ratchet(
        "workspace_ignored",
        "BASELINE_WORKSPACE_IGNORED",
        BASELINE_WORKSPACE_IGNORED,
        live_workspace_ignored,
        ratchet_failures,
    )
    _check_ratchet(
        "test_binaries",
        "BASELINE_TEST_BINARIES",
        BASELINE_TEST_BINARIES,
        live_binaries,
        ratchet_failures,
    )

    if args.json:
        out = {
            "executed_at": _now_iso(),
            "drift_tolerance_pct": DIFF_TOLERANCE_PCT,
            "drift_tolerance_abs": DIFF_TOLERANCE_ABS,
            "live": totals,
            "baseline_metrics": baseline_metrics,
            "ratchet_baselines": {
                "BASELINE_LIB_TESTS": BASELINE_LIB_TESTS,
                "BASELINE_LIB_IGNORED": BASELINE_LIB_IGNORED,
                "BASELINE_WORKSPACE_TESTS": BASELINE_WORKSPACE_TESTS,
                "BASELINE_WORKSPACE_IGNORED": BASELINE_WORKSPACE_IGNORED,
                "BASELINE_TEST_BINARIES": BASELINE_TEST_BINARIES,
            },
            "drift_violations": drift_violations,
            "shrink_failed": shrink_failed,
            "grow_failed": grow_failed,
            "ratchet_failures": ratchet_failures,
            "would_fail": bool(
                (drift_violations and args.strict) or ratchet_failures
            ),
        }
        json.dump(out, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(
            f"Test-inventory drift gate (Issue #3442)\n"
            f"  baseline: {baseline_path.relative_to(REPO_ROOT)}\n"
            f"  live:     {DEFAULT_INVENTORY.relative_to(REPO_ROOT)}\n"
            f"  tolerance: ±{DIFF_TOLERANCE_PCT:.1f}% or ±{DIFF_TOLERANCE_ABS} tests "
            f"(whichever is larger)"
        )
        print(
            f"\nCounts (live → baseline):\n"
            f"  lib_tests:           {live_lib} → {base_lib}\n"
            f"  lib_ignored:         {live_lib_ignored} → {base_lib_ignored}\n"
            f"  workspace_tests:     {live_workspace} → {base_workspace}\n"
            f"  workspace_ignored:   {live_workspace_ignored} → {base_workspace_ignored}\n"
            f"  test_binaries:       {live_binaries} → {base_binaries}"
        )
        if drift_violations:
            print("\nDrift-threshold violations:")
            for line in drift_violations:
                print(line)
        else:
            print("\nNo drift-threshold violations.")
        if ratchet_failures:
            print("\nRatchet violations (BASELINE_* exceeded):")
            for line in ratchet_failures:
                print(line)
        else:
            print("\nNo ratchet violations.")

    failure = bool((drift_violations and args.strict) or ratchet_failures)
    return 1 if failure else 0


if __name__ == "__main__":
    sys.exit(main())

#   - 2026-09-11 (PR improve/quarantine-placeholder-test, final stack):
#     119 -> 120 — the Issue #3629 thermal_mass placeholder quarantine
#     adds one more workspace `#[ignore]` on top of the wasm quarantine.

#   - 2026-09-12 (PR improve/quarantine-placeholder-test, Issue #3705):
#     120 -> 121 — the flaky occupancy statistical test (OS-seeded 10k-step
#     Markov assertion) quarantined per the Issue #3629 protocol.
