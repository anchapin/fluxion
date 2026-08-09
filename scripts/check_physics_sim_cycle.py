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
2. `src/sim/construction.rs` and `src/sim/per_surface_conduction.rs`
   (the two `sim` files that previously hosted shared domain types) must
   NOT import from `src/physics/**` (no `use crate::physics::` upward
   deps). Both files are now thin re-export shims over the leaf modules
   above (no physics imports remain).
3. Summary: report the total cycle-edge count. As of #2462 the documented
   baseline is 0+0 edges.

Usage:
  python3 scripts/check_physics_sim_cycle.py

Exit codes:
  0 — no cycle regression (offender count is at or below the documented
      baseline of 0)
  1 — cycle regression detected (a NEW edge appeared, exceeding the
      documented baseline)
  2 — script error

The script reports `BASELINE_PHYSICS_TO_SIM = 0` and
`BASELINE_SIM_TO_PHYSICS = 0` documented edges as the *current state*.
A future PR that adds a *new* `use crate::sim::` import under
`src/physics/**` (or `use crate::physics::` under the two protected
`src/sim/**` files) — pushing the count *above* zero — is flagged
immediately as a regression. See ARCHITECTURE.md
§"Regression guard (Issue #2463, closed by #2462)" for the
source-of-truth numbers.

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

# `sim` files that host shared domain types and must not import physics internals.
# Per ARCHITECTURE.md §"Remaining cycles" and the companion cycle-break plan,
# these are the two sides of the `physics <-> sim` seam. They currently carry
# leaf re-exports from `physics::constants`; the cycle-break work moves those.
PROTECTED_SIM_FILES = (
    FLUXION_SRC / "sim" / "construction.rs",
    FLUXION_SRC / "sim" / "per_surface_conduction.rs",
)

# Documented baseline edge counts. Issue #2462 drove the cycle to 0+0
# edges by hoisting the shared domain types into `fluxion_core::*` (see
# ARCHITECTURE.md §"Cycle break (#2462 — physics ↔ sim shared domain types
# → `fluxion-core`)"). Any PR that pushes these *up* is a regression and is
# flagged by this guard. See ARCHITECTURE.md §"Regression guard (Issue
# #2463, closed by #2462)" for the source-of-truth numbers.
BASELINE_PHYSICS_TO_SIM = 0
BASELINE_SIM_TO_PHYSICS = 0

# Regex for Phase 2: match `use` or `pub use` against `crate::physics::`.
# Mirrors `scan_sim_for_orientation_cycle` in check_ashrae_cases_cycle.py
# which uses `(pub\s+)?use\s+crate::validation::...::Orientation\b`.
_PHYSICS_IMPORT_RE = re.compile(
    r"^\s*(pub\s+)?use\s+crate::physics::"
)


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


def scan_protected_sim_files_for_physics_deps() -> list[str]:
    """Walk the two protected `sim` files and forbid `use crate::physics::`.

    Mirrors `scan_sim_for_orientation_cycle` in check_ashrae_cases_cycle.py:
    anchors on `(pub\\s+)?use\\s+crate::physics::` and reports every match
    (file:line + the offending line) so the cycle-break PR can verify
    progress.
    """
    offenders: list[str] = []
    for rs_file in PROTECTED_SIM_FILES:
        if not rs_file.exists():
            offenders.append(f"{rs_file.relative_to(REPO_ROOT)}: file missing")
            continue
        rel = rs_file.relative_to(REPO_ROOT)
        for lineno, line in enumerate(
            rs_file.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if _PHYSICS_IMPORT_RE.match(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    return offenders


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
        print(f"    FAIL: {len(physics_to_sim)} offender(s) "
              f"({new_edges} above baseline {BASELINE_PHYSICS_TO_SIM})")
    else:
        if physics_to_sim:
            print(f"    OK: {len(physics_to_sim)} offender(s) "
                  f"(at baseline {BASELINE_PHYSICS_TO_SIM})")
        else:
            print(f"    OK: 0 upward deps (below baseline {BASELINE_PHYSICS_TO_SIM})")

    print("[2/3] src/sim/construction.rs + src/sim/per_surface_conduction.rs "
          "must not `use crate::physics::` ...")
    sim_to_physics = scan_protected_sim_files_for_physics_deps()
    if len(sim_to_physics) > BASELINE_SIM_TO_PHYSICS:
        new_edges = len(sim_to_physics) - BASELINE_SIM_TO_PHYSICS
        failures.append(
            f"protected sim files have {len(sim_to_physics)} upward dep(s) to src/physics "
            f"(baseline: {BASELINE_SIM_TO_PHYSICS}; {new_edges} NEW edge(s) above baseline):"
        )
        failures.extend(f"    {o}" for o in sim_to_physics)
        print(f"    FAIL: {len(sim_to_physics)} offender(s) "
              f"({new_edges} above baseline {BASELINE_SIM_TO_PHYSICS})")
    else:
        if sim_to_physics:
            print(f"    OK: {len(sim_to_physics)} offender(s) "
                  f"(at baseline {BASELINE_SIM_TO_PHYSICS})")
        else:
            print(f"    OK: no cycle markers (below baseline {BASELINE_SIM_TO_PHYSICS})")

    print("[3/3] summary ...")
    total = len(physics_to_sim) + len(sim_to_physics)
    baseline_total = BASELINE_PHYSICS_TO_SIM + BASELINE_SIM_TO_PHYSICS
    print(f"    Total cycle edges: {total}")
    print(f"    (Documented baseline: {baseline_total} "
          f"({BASELINE_PHYSICS_TO_SIM} physics->sim + "
          f"{BASELINE_SIM_TO_PHYSICS} sim->physics);")
    print(f"     companion cycle-break work drives this to 0.)")

    print()
    if failures:
        print("CYCLE REGRESSION DETECTED:")
        for f in failures:
            print(f"  {f}")
        print()
        print(f"A new `use crate::sim::*` import under `src/physics/**` (or a")
        print(f"`use crate::physics::*` import under the two protected sim files)")
        print(f"has appeared that exceeds the documented baseline of "
              f"{baseline_total} edges. The companion cycle-break issue is the")
        print(f"work authorised to reduce the count; this guard rejects growth.")
        return 1
    print(f"No cycle regression. Issue #2463 cycle stays at {total} edge(s) "
          f"(at or below documented baseline of {baseline_total}).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
