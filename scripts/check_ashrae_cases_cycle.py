#!/usr/bin/env python3
"""
Cycle Regression Guard for the `sim <-> validation` cycle (#1441 + #2495).

Verifies the `sim <-> validation` cycle documented in `ARCHITECTURE.md`
§"Cycle break (#1441 — ASHRAE-140 leaf types → `fluxion-core`)" cannot
grow. Issue #1441 only *partially* broke the cycle — it moved 13 pure-data
leaf types (Orientation, WindowArea, ...) into `fluxion_core::ashrae_cases`
and the original version of this guard forbid a single import shape
(`use crate::validation::ashrae_140_cases::Orientation`). The composite
types that actually drive the cycle (`CaseSpec`, `CaseBuilder`,
`ASHRAE140Case`, `CommonWall`, `ConstructionSpec`, ...) were never moved
because they carry upward deps to `crate::sim::*` / `crate::physics::*`,
so the cycle persisted undetected (issue #2495).

This guard enforces four invariants:

1. `fluxion-core` must NOT import from `fluxion` (no `crate::sim::*`,
   `crate::physics::*`, `crate::ai::*`, `crate::validation::*`, etc.) —
   the leaf crate stays acyclic w.r.t. the main crate.
2. `fluxion_core::ashrae_cases` MUST exist and contain all 13 moved leaf
   types (structural verification of the #1441 move).
3. `src/sim/**` → `src/validation/**` edge count is at or below the
   documented baseline (any `crate::validation::*` reference — `use`
   imports AND fully-qualified usage like match arms on `CaseSpec`).
4. `src/validation/**` → `src/{sim,physics,weather}/**` edge counts are
   each at or below their documented baseline.

Baseline/regression semantics (mirrors `scripts/check_physics_sim_cycle.py`):
the documented baseline counts below snapshot the cycle as of the commit
that introduced this guard (#2495). The script PASSES when every count is
at or below its baseline and FAILS (exit 1) the moment a count INCREASES.
This enforces "no new cycle edges" without requiring the (large, deferred)
removal of the ~72 sim->validation + ~143 validation->{sim,physics,weather}
edges that exist today. The companion cycle-removal work is the only
change authorised to *lower* a baseline; this guard rejects growth.

Usage:
  python3 scripts/check_ashrae_cases_cycle.py

Exit codes:
  0 — no cycle regression (every count at or below its baseline)
  1 — cycle regression detected (a count rose above its baseline)
  2 — script error
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FLUXION_CORE_SRC = REPO_ROOT / "fluxion-core" / "src"
FLUXION_SRC = REPO_ROOT / "src"
SIM_DIR = FLUXION_SRC / "sim"
VALIDATION_DIR = FLUXION_SRC / "validation"
ASHRAE_CASES_FILE = FLUXION_CORE_SRC / "ashrae_cases.rs"

# Leaf types that issue #1441 moved into fluxion-core.
MOVED_LEAF_TYPES = {
    "Orientation",
    "WindowArea",
    "ConstructionType",
    "ShadingType",
    "ShadingDevice",
    "GlassType",
    "WindowSpec",
    "InternalLoads",
    "HvacSchedule",
    "NightVentilation",
    "BuildingType",
    "GeometrySpec",
    "ConductanceReferences",
}

# ---------------------------------------------------------------------------
# Documented baseline edge counts (issue #2495).
#
# Each baseline is the count of `crate::<dir>::` references the scan below
# reports against the tree at the commit that introduced this guard. The
# guard PASSES at-or-below baseline and FAILS when a count grows above it
# (a new cycle edge appeared). Lowering a baseline is only authorised by
# the companion cycle-removal work; see ARCHITECTURE.md
# §"Cycle break (#1441 — ASHRAE-140 leaf types → `fluxion-core`)".
#
# Why a broad scan (not just `use` imports): the cycle is driven equally by
# `use crate::validation::ashrae_140_cases::CaseSpec` imports and by
# fully-qualified usages such as `crate::validation::ashrae_140_cases::
# CaseSpec` in `impl`/`fn`/match arms. Counting every reference (after
# stripping `//` and `*` comment lines) catches both forms and matches the
# technique already used by `scan_fluxion_core_for_upward_deps` below and
# by `check_physics_sim_cycle.py`'s physics->sim direction.
# ---------------------------------------------------------------------------
BASELINE_SIM_TO_VALIDATION = 99  # src/sim/**    -> crate::validation::* (was 72; +22 prod for #3291, +5 test-module)
# Issue #3291 (umbrella: GaugeSolver production-path wiring + default
# flip, PR2 commit `13648c3`): +22 sim->validation edges from
# `src/sim/thermal_model_core.rs` and `src/sim/thermal_model_physics/
# step_dispatcher.rs`. The new edges are intentional: the gauge-path
# integration needs `crate::validation::ashrae_140_cases::{CaseSpec,
# WindowSpec, ConstructionType, Orientation, CommonWall, GeometrySpec}`
# to populate the gauge backend (windows for solar, construction type
# for the auto-promote, common walls for the multi-zone coupling,
# surface orientations for the t_sol_air/t_i Crank-Nicolson mass-state
# proxy). Lowering this baseline is still reserved for the companion
# cycle-removal work; raising (as here) accommodates a one-shot
# feature-driven growth with rationale.
BASELINE_VALIDATION_TO_SIM = 65  # src/validation/** -> crate::sim::* (was 58; +7 for #3291)
# Issue #3291: +7 validation->sim edges. Validation now imports
# `crate::sim::thermal_selector::{ThermalSelector, ZoneSolverKind,
# ConductionSolverKind}` to drive the per-case selector in
# `from_spec_with_selector` (which is the API that consumes the
# selector). The cycle-removal work (issue #1441) already hoisted
# the ASHRAE-140 leaf types to fluxion-core; further baseline
# reduction requires moving composite validation types
# (CaseSpec, etc.), which is out of scope for the gauge integration
# tracked by #3291.
# Issue #2980: +3 physics / +2 weather to run the real 8760-step Case 970
# physics simulation in `src/validation/ashrae_140_multi_zone.rs`
# (`run_real_case_970_energy`). The new edges are intentional: the function
# instantiates `ThermalModel<VectorField>` and loads EPW weather so the
# validator consumes engine output rather than the pre-#2980 hardcoded MWh
# placeholders. The cycle-removal work (issue #1441) already hoisted the
# ASHRAE-140 leaf types to fluxion-core; further baseline reduction requires
# moving composite validation types (CaseSpec, etc.), which is out of scope
# for the placeholder-completion work tracked by #2980. Lowering this
# baseline is still reserved for the companion cycle-removal work; raising
# (as here) accommodates a one-shot feature-driven growth with rationale.
BASELINE_VALIDATION_TO_PHYSICS = 65  # src/validation/** -> crate::physics::* (was 62; +3 for #2980)
BASELINE_VALIDATION_TO_WEATHER = 25  # src/validation/** -> crate::weather::* (was 23; +2 for #2980)


def _scan_dir_for_prefixes(directory: Path, prefixes: tuple[str, ...]) -> list[str]:
    """Walk `directory`/**/*.rs and report every non-comment line containing
    any of the given `crate::` prefixes.

    Uses the `prefix in stripped` technique (no `use` anchor) so we catch
    every form that would create the upward edge: `use`, `pub use`, glob
    imports, and fully-qualified `crate::dir::Type` paths in expressions,
    match arms, and signatures. Comment lines (`//`, `///`, `//!`, and `*`
    block-comment continuations) are skipped, mirroring
    `scan_fluxion_core_for_upward_deps`.
    """
    offenders: list[str] = []
    if not directory.exists():
        return offenders
    for rs_file in directory.rglob("*.rs"):
        rel = rs_file.relative_to(REPO_ROOT)
        for lineno, line in enumerate(
            rs_file.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            for prefix in prefixes:
                if prefix in stripped:
                    offenders.append(f"{rel}:{lineno}: {stripped}")
                    break
    return offenders


def scan_fluxion_core_for_upward_deps() -> list[str]:
    """Walk fluxion-core/src and forbid any reference to crate::sim|physics|ai|validation."""
    upward_prefixes = (
        "crate::sim::",
        "crate::physics::",
        "crate::ai::",
        "crate::validation::",
        "crate::interop::",
        "crate::analysis::",
        "crate::python::",
        "crate::cli::",
        "crate::napi::",
        "crate::api::",
        "crate::performance::",
        "crate::orchestration::",
        "crate::thermal::",
        "crate::testing::",
    )
    return _scan_dir_for_prefixes(FLUXION_CORE_SRC, upward_prefixes)


def scan_sim_for_validation_deps() -> list[str]:
    """src/sim/** -> crate::validation::* (composite types + re-exports).

    Issue #2495: the old guard only forbid `use ...::Orientation` (a leaf
    type that #1441 moved). The cycle is actually driven by composite types
    — `CaseSpec`, `CaseBuilder`, `ASHRAE140Case`, `CommonWall`,
    `ConstructionSpec`, plus the `validation::diagnostics` /
    `validation::config` imports — so we now count *every*
    `crate::validation::*` reference from `src/sim/**`.
    """
    return _scan_dir_for_prefixes(SIM_DIR, ("crate::validation::",))


def scan_validation_for_sim_deps() -> list[str]:
    """src/validation/** -> crate::sim::* (the validation->sim direction)."""
    return _scan_dir_for_prefixes(VALIDATION_DIR, ("crate::sim::",))


def scan_validation_for_physics_deps() -> list[str]:
    """src/validation/** -> crate::physics::* (validation->physics edges)."""
    return _scan_dir_for_prefixes(VALIDATION_DIR, ("crate::physics::",))


def scan_validation_for_weather_deps() -> list[str]:
    """src/validation/** -> crate::weather::* (validation->weather edges)."""
    return _scan_dir_for_prefixes(VALIDATION_DIR, ("crate::weather::",))


def verify_ashrae_cases_module() -> list[str]:
    """Confirm fluxion_core::ashrae_cases contains all moved leaf types."""
    offenders: list[str] = []
    if not ASHRAE_CASES_FILE.exists():
        offenders.append(f"{ASHRAE_CASES_FILE.relative_to(REPO_ROOT)}: file missing")
        return offenders
    content = ASHRAE_CASES_FILE.read_text(encoding="utf-8", errors="replace")
    for type_name in MOVED_LEAF_TYPES:
        # Match `pub enum TypeName` or `pub struct TypeName`
        if not re.search(rf"\bpub\s+(enum|struct)\s+{type_name}\b", content):
            offenders.append(
                f"{ASHRAE_CASES_FILE.relative_to(REPO_ROOT)}: missing pub {type_name}"
            )
    return offenders


def _check_baseline(
    name: str, found: list[str], baseline: int, failures: list[str]
) -> None:
    """Append a failure if `len(found)` exceeds `baseline`; print either way.

    Mirrors the messaging shape of `check_physics_sim_cycle.py`: report the
    count, the baseline, and (on regression) how many edges are NEW above
    baseline. At-or-below baseline is OK.
    """
    count = len(found)
    if count > baseline:
        new_edges = count - baseline
        failures.append(
            f"{name}: {count} edge(s) (baseline: {baseline}; "
            f"{new_edges} NEW edge(s) above baseline):"
        )
        failures.extend(f"    {o}" for o in found)
        print(f"    FAIL: {count} edge(s) ({new_edges} above baseline {baseline})")
    elif count:
        print(f"    OK: {count} edge(s) (at baseline {baseline})")
    else:
        print(f"    OK: 0 edges (below baseline {baseline})")


def main() -> int:
    print(
        f"Checking sim<->validation cycle guards for issues #1441 + #2495 "
        f"(repo: {REPO_ROOT})"
    )
    print()

    failures: list[str] = []

    print(
        "[1/6] fluxion-core must have no upward deps to sim/physics/ai/validation ..."
    )
    upward = scan_fluxion_core_for_upward_deps()
    if upward:
        failures.append(f"fluxion-core has {len(upward)} upward dep(s):")
        failures.extend(f"    {o}" for o in upward)
        print(f"    FAIL: {len(upward)} offender(s)")
    else:
        print("    OK: 0 upward deps")

    print("[2/6] fluxion_core::ashrae_cases must contain all moved leaf types ...")
    missing = verify_ashrae_cases_module()
    if missing:
        failures.append("fluxion_core::ashrae_cases is incomplete:")
        failures.extend(f"    {m}" for m in missing)
        print(f"    FAIL: {len(missing)} missing type(s)")
    else:
        print(f"    OK: all {len(MOVED_LEAF_TYPES)} leaf types present")

    sim_to_validation = scan_sim_for_validation_deps()
    validation_to_sim = scan_validation_for_sim_deps()
    validation_to_physics = scan_validation_for_physics_deps()
    validation_to_weather = scan_validation_for_weather_deps()

    print("[3/6] src/sim/** -> crate::validation::* (composite + leaf types) ...")
    _check_baseline(
        "sim->validation", sim_to_validation, BASELINE_SIM_TO_VALIDATION, failures
    )

    print("[4/6] src/validation/** -> crate::sim::* ...")
    _check_baseline(
        "validation->sim", validation_to_sim, BASELINE_VALIDATION_TO_SIM, failures
    )

    print("[5/6] src/validation/** -> crate::physics::* ...")
    _check_baseline(
        "validation->physics",
        validation_to_physics,
        BASELINE_VALIDATION_TO_PHYSICS,
        failures,
    )

    print("[6/6] src/validation/** -> crate::weather::* ...")
    _check_baseline(
        "validation->weather",
        validation_to_weather,
        BASELINE_VALIDATION_TO_WEATHER,
        failures,
    )

    total = (
        len(sim_to_validation)
        + len(validation_to_sim)
        + len(validation_to_physics)
        + len(validation_to_weather)
    )
    baseline_total = (
        BASELINE_SIM_TO_VALIDATION
        + BASELINE_VALIDATION_TO_SIM
        + BASELINE_VALIDATION_TO_PHYSICS
        + BASELINE_VALIDATION_TO_WEATHER
    )
    print()
    print(f"Total sim<->validation cycle edges: {total}")
    print(
        f"(Documented baseline: {baseline_total} = "
        f"{BASELINE_SIM_TO_VALIDATION} sim->validation + "
        f"{BASELINE_VALIDATION_TO_SIM} validation->sim + "
        f"{BASELINE_VALIDATION_TO_PHYSICS} validation->physics + "
        f"{BASELINE_VALIDATION_TO_WEATHER} validation->weather)"
    )
    print(
        "The companion cycle-removal work drives each baseline toward 0; "
        "this guard rejects growth."
    )

    print()
    if failures:
        print("CYCLE REGRESSION DETECTED:")
        for f in failures:
            print(f"  {f}")
        print()
        print("A new `crate::validation::*` reference under `src/sim/**`, or a new")
        print("`crate::{sim,physics,weather}::*` reference under `src/validation/**`,")
        print("has appeared that exceeds the documented baseline. The companion")
        print("cycle-removal issue is the only work authorised to lower a baseline;")
        print('this guard rejects growth. See ARCHITECTURE.md §"Cycle break (#1441)".')
        return 1
    print(
        f"No cycle regression. sim<->validation cycle stays at {total} edge(s) "
        f"(at or below documented baseline of {baseline_total})."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
