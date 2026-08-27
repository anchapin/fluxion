#!/usr/bin/env python3
"""
Cycle Regression Guard for the `physics <-> sim` cycle (Issue #2463).

Mirrors `scripts/check_ashrae_cases_cycle.py` (Issue #1441) and verifies
the `physics <-> sim` cycle closed by Issue #2462 stays closed:

1. `src/physics/**` must NOT import from `src/sim/**` (no `use crate::sim::`
   upward deps). Issue #2462 hoisted `ConstructionLayer`, `Construction`,
   `MassClass`, `Materials`, `Assemblies`, `SurfaceType` to
   `fluxion_core::construction`; `SurfaceKind`, `MassNode`, `SurfaceNode`,
   `PerSurfaceConductionSolver` to `fluxion_core::per_surface_conduction`;
   and `STEFAN_BOLTZMANN` to `fluxion_core::physics_constants` — driving the
   physics->sim edge count from 5 to 0.
2. **All** of `src/sim/**/*.rs` must NOT grow its `use crate::physics::`
   import count above the documented baseline (Issue #2766). The original
   Phase 2 (Issue #2463) guarded only the two files that previously hosted
   shared domain types — `src/sim/construction.rs` and
   `src/sim/per_surface_conduction.rs` — leaving ~83 ``use crate::physics::``
   imports across 26 other sim files (``thermal_model.rs``, ``engine.rs``,
   ``ventilation.rs``, ...) completely unguarded. Issue #2766 extended
   coverage to ALL of ``src/sim/**`` and snapshotted the 83 pre-existing
   edges as the initial baseline; PR #3020 (issue #2896) lowered the
   baseline to 83 after deleting doc-only stubs, and PR #3024 (issue #2891)
   raised it to 85 to admit two legitimate ``use crate::physics::exterior_convection::{...}``
   edges that implement wind-velocity-dependent exterior convection
   (ASHRAE 140 §5.2.6) in `src/sim/thermal_model_core.rs` (line 243)
   and `src/sim/thermal_model_physics/physics_impl.rs` (line 322).
   Any NEW edge beyond these 85 fails the guard.
3. Summary: report the total cycle-edge count. As of #2462 + #2766 +
   #2896 + #2891 + #2878 + #3214 the documented baseline is 0 physics->sim + 68 sim->physics
   edges.

Usage:
  python3 scripts/check_physics_sim_cycle.py

Exit codes:
  0 — no cycle regression (offender count is at or below the documented
      baseline)
  1 — cycle regression detected (a NEW edge appeared, exceeding the
      documented baseline)
  2 — script error

The script reports ``BASELINE_PHYSICS_TO_SIM = 0`` and
``BASELINE_SIM_TO_PHYSICS = 68`` documented edges as the *current state*.
A future PR that adds a *new* ``use crate::sim::`` import under
``src/physics/**`` (or a *new* ``use crate::physics::`` import under any
``src/sim/**/*.rs`` file) — pushing the count *above* the documented
baseline — is flagged immediately as a regression. See ARCHITECTURE.md
§"Regression guard (Issue #2463, closed by #2462)" and
§"Regression guard (Issue #2766, extends #2463)" for the source-of-truth
numbers.

See ARCHITECTURE.md §"Cycle break (#2462 — physics ↔ sim shared domain
types → `fluxion-core`)" and docs/mutation_testing_crate_split.md
§"Phase 2 — break the physics <-> sim cycle (extract shared domain
types)" for context.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FLUXION_SRC = REPO_ROOT / "src"
PHYSICS_DIR = FLUXION_SRC / "physics"
SIM_DIR = FLUXION_SRC / "sim"

# ``src/sim/`` files exempted from the Phase 2 scan. Carry-forward of the
# original ``PROTECTED_SIM_FILES`` allowlist mechanism (Issue #2463): add a
# POSIX-relative path (relative to ``SIM_DIR``) here ONLY when a documented
# thin re-export shim (see ARCHITECTURE.md §"Re-export shims") legitimately
# needs to re-export a physics type. Currently empty — every
# ``src/sim/**/*.rs`` file, including the four documented shims
# (``assembly.rs``, ``multi_node_thermal.rs``, ``construction.rs``,
# ``per_surface_conduction.rs``), is scanned. All four shims are presently
# clean (0 ``use crate::physics::`` imports, per #2462), so they pass the
# scan naturally without needing an exemption.
SIM_SHIM_EXCEPTIONS: frozenset[str] = frozenset()

# Documented baseline edge counts.
#
# Phase 1 (physics->sim): Issue #2462 drove this direction to 0 edges by
# hoisting the shared domain types into ``fluxion_core::*``. Any PR that
# pushes this above 0 is a regression.
#
# Phase 2 (sim->physics): Issue #2766 extended this phase from the 2
# originally-guarded files (``construction.rs`` +
# ``per_surface_conduction.rs``) to ALL ``src/sim/**/*.rs`` files. The
# extension surfaced 84 pre-existing ``use crate::physics::`` imports
# across 26 sim files that the original guard never saw. These 83 edges
# were snapshotted as the initial baseline; PR #3020 (issue #2896) lowered
# the baseline to 83 after deleting doc-only stub
# ``src/sim/thermal_model_network.rs`` and its single physics edge; PR #3024
# (issue #2891) then raised the baseline to 85 to admit two new
# ``use crate::physics::exterior_convection::{...}`` edges that implement
# ASHRAE 140 §5.2.6 wind-velocity-dependent exterior convection in the 5R1C
# path (see ``src/sim/thermal_model_core.rs:243`` and
# ``src/sim/thermal_model_physics/physics_impl.rs:322``); PR #3034
# (issue #2878) then lowered the baseline to 79 by deleting the legacy
# ``src/sim/thermal_model_data.rs`` god-struct (8 ``use crate::physics::``
# imports at lines 6-14) and replacing it with a per-domain split in
# ``src/sim/thermal_model_data/`` that consolidates physics imports into
# a single ``pub use crate::physics::{...}`` block in the new
# ``mod.rs`` (2 ``pub use`` lines: the consolidated block + a cfg-gated
# re-export of ``gauge_zone_solver::GaugeZoneSolver``). Net effect:
# -6 sim->physics edges (8 removed by god-struct deletion, 2 added by
# consolidated re-exports). The guard PASSES at-or-below 68 and FAILS
# when a NEW edge pushes the count to 69+. Lowering this baseline is
# authorised only by companion cycle-removal work; see ARCHITECTURE.md
# §"Regression guard (Issue #2766, extends #2463)".
#
# See ARCHITECTURE.md §"Regression guard (Issue #2463, closed by #2462)"
# for the source-of-truth numbers.
BASELINE_PHYSICS_TO_SIM = 0
BASELINE_SIM_TO_PHYSICS = 68

# Regex for Phase 2: match `use` or `pub use` against `crate::physics::`.
# Mirrors `scan_sim_for_orientation_cycle` in check_ashrae_cases_cycle.py
# which uses `(pub\s+)?use\s+crate::validation::...::Orientation\b`.
_PHYSICS_IMPORT_RE = re.compile(r"^\s*(pub\s+)?use\s+crate::physics::")


def scan_physics_for_sim_deps() -> list[str]:
    r"""Walk src/physics/** and forbid any `use crate::sim::` upward dep.

    Mirrors `scan_fluxion_core_for_upward_deps` in check_ashrae_cases_cycle.py
    but scoped to `src/physics/**`. We use the same `prefix in stripped` scan
    technique (without the `pub use / use` anchor) so we catch every form
    of import that would create the upward edge — `use`, `pub use`,
    `use crate::sim::foo::*`, fully-qualified `crate::sim::foo::bar()` paths,
    etc.
    """
    offenders: list[str] = []
    sim_prefix = "crate::sim::"
    if not PHYSICS_DIR.exists():
        return offenders
    for rs_file in PHYSICS_DIR.rglob("*.rs"):
        rel = rs_file.relative_to(REPO_ROOT)
        for lineno, line in enumerate(
            rs_file.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            if sim_prefix in stripped:
                offenders.append(f"{rel}:{lineno}: {stripped}")
    return offenders


def scan_sim_for_physics_deps() -> list[str]:
    """Walk ``src/sim/**/*.rs`` and report every ``use crate::physics::`` import.

    Issue #2766: the original Phase 2 (Issue #2463) scanned only the two
    files in the old ``PROTECTED_SIM_FILES`` tuple — ``construction.rs``
    and ``per_surface_conduction.rs`` — leaving ~83 ``use crate::physics::``
    imports across 26 other sim files completely unguarded. This function
    extends coverage to ALL of ``src/sim/**`` (minus the files in
    ``SIM_SHIM_EXCEPTIONS``) so any new sim->physics edge in any sim file
    is caught.

    Mirrors ``scan_sim_for_validation_deps`` in check_ashrae_cases_cycle.py
    (directory ``rglob`` walk) but anchors on the existing
    ``(pub\\s+)?use\\s+crate::physics::`` regex — the established Phase 2
    detection logic is unchanged; only the set of files scanned grew.
    Reports every match as ``file:line: text`` so cycle-removal work can
    verify progress file-by-file.
    """
    offenders: list[str] = []
    if not SIM_DIR.exists():
        return offenders
    for rs_file in sorted(SIM_DIR.rglob("*.rs")):
        rel_posix = rs_file.relative_to(SIM_DIR).as_posix()
        if rel_posix in SIM_SHIM_EXCEPTIONS:
            continue
        rel = rs_file.relative_to(REPO_ROOT)
        for lineno, line in enumerate(
            rs_file.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if _PHYSICS_IMPORT_RE.match(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    return offenders


# Backward-compat alias. The pre-#2766 name is kept so downstream consumers
# (notably ``scripts/check_cycle_downward_trend.py``'s
# ``collect_current_edges``, Issue #2768) keep working without modification.
# New code should call ``scan_sim_for_physics_deps`` directly, paralleling
# ``scan_sim_for_validation_deps`` in check_ashrae_cases_cycle.py.
scan_protected_sim_files_for_physics_deps = scan_sim_for_physics_deps


def main() -> int:
    print(f"Checking physics<->sim cycle guards for issue #2463 (repo: {REPO_ROOT})")
    print()

    failures: list[str] = []

    print("[1/3] src/physics/** must not `use crate::sim::` ...")
    physics_to_sim = scan_physics_for_sim_deps()
    if len(physics_to_sim) > BASELINE_PHYSICS_TO_SIM:
        new_edges = len(physics_to_sim) - BASELINE_PHYSICS_TO_SIM
        failures.append(
            f"src/physics has {len(physics_to_sim)} upward dep(s) to src/sim "
            f"(baseline: {BASELINE_PHYSICS_TO_SIM}; {new_edges} NEW edge(s) above baseline):"
        )
        failures.extend(f"    {o}" for o in physics_to_sim)
        print(
            f"    FAIL: {len(physics_to_sim)} offender(s) "
            f"({new_edges} above baseline {BASELINE_PHYSICS_TO_SIM})"
        )
    else:
        if physics_to_sim:
            print(
                f"    OK: {len(physics_to_sim)} offender(s) "
                f"(at baseline {BASELINE_PHYSICS_TO_SIM})"
            )
        else:
            print(f"    OK: 0 upward deps (below baseline {BASELINE_PHYSICS_TO_SIM})")

    print(
        "[2/3] src/sim/**/*.rs must not `use crate::physics::` "
        "(all sim files; issue #2766) ..."
    )
    sim_to_physics = scan_sim_for_physics_deps()
    if len(sim_to_physics) > BASELINE_SIM_TO_PHYSICS:
        new_edges = len(sim_to_physics) - BASELINE_SIM_TO_PHYSICS
        failures.append(
            f"src/sim has {len(sim_to_physics)} upward dep(s) to src/physics "
            f"(baseline: {BASELINE_SIM_TO_PHYSICS}; {new_edges} NEW edge(s) above baseline):"
        )
        failures.extend(f"    {o}" for o in sim_to_physics)
        print(
            f"    FAIL: {len(sim_to_physics)} offender(s) "
            f"({new_edges} above baseline {BASELINE_SIM_TO_PHYSICS})"
        )
    else:
        if sim_to_physics:
            print(
                f"    OK: {len(sim_to_physics)} offender(s) "
                f"(at baseline {BASELINE_SIM_TO_PHYSICS})"
            )
        else:
            print(
                f"    OK: no cycle markers (below baseline {BASELINE_SIM_TO_PHYSICS})"
            )

    print("[3/3] summary ...")
    total = len(physics_to_sim) + len(sim_to_physics)
    baseline_total = BASELINE_PHYSICS_TO_SIM + BASELINE_SIM_TO_PHYSICS
    print(f"    Total cycle edges: {total}")
    print(
        f"    (Documented baseline: {baseline_total} "
        f"({BASELINE_PHYSICS_TO_SIM} physics->sim + "
        f"{BASELINE_SIM_TO_PHYSICS} sim->physics);"
    )
    print("     companion cycle-break work drives this to 0.)")

    print()
    if failures:
        print("CYCLE REGRESSION DETECTED:")
        for f in failures:
            print(f"  {f}")
        print()
        print("A new `use crate::sim::*` import under `src/physics/**` (or a")
        print("`use crate::physics::*` import under any `src/sim/**/*.rs` file)")
        print(
            f"has appeared that exceeds the documented baseline of "
            f"{baseline_total} edges. The companion cycle-break issue is the"
        )
        print("work authorised to reduce the count; this guard rejects growth.")
        return 1
    print(
        f"No cycle regression. Issue #2463 cycle stays at {total} edge(s) "
        f"(at or below documented baseline of {baseline_total})."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
