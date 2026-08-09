#!/usr/bin/env python3
"""
Cycle Regression Guard for the `physics <-> sim` cycle (Issue #2463).

Mirrors `scripts/check_ashrae_cases_cycle.py` (Issue #1441) and verifies
the `physics <-> sim` cycle documented in `ARCHITECTURE.md` §"Remaining
cycles" stays in its current target state (5 known physics->sim edges;
the companion cycle-break issue is the work that drives the count to 0):

1. `src/physics/**` must NOT import from `src/sim/**` (no `use crate::sim::`
   upward deps). The 5 currently-known offenders are:

   - `src/physics/thermal_mass/construction.rs -> sim::construction::ConstructionLayer`
   - `src/physics/thermal_mass/diagnostics.rs   -> sim::construction::ConstructionLayer`
   - `src/physics/multi_node_solver.rs          -> sim::per_surface_conduction::{...}`
   - `src/physics/multi_node_solver.rs          -> sim::sky_radiation::STEFAN_BOLTZMANN`
   - `src/physics/multi_node_solver.rs          -> sim::sky_radiation::SkyRadiationExchange`

2. `src/sim/construction.rs` and `src/sim/per_surface_conduction.rs`
   (the two `sim` files that host shared domain types) must NOT import
   from `src/physics/**` (no `use crate::physics::` upward deps). Currently
   `sim::construction` re-exports a few leaf constants from
   `physics::constants`; the companion cycle-break work moves them.
3. Summary: report the total cycle-edge count so branch protection can
   track the cycle-break PR's progress to 0.

Usage:
  python3 scripts/check_physics_sim_cycle.py

Exit codes:
  0 — no cycle regression (or only the documented baseline edges remain)
  1 — cycle regression detected (a NEW edge appeared, beyond the baseline)
  2 — script error

The script deliberately reports the current 5+ baseline edges as failures
so that any future PR that adds a *new* `use crate::sim::` import under
`src/physics/**` (or `use crate::physics::` under the two protected
`src/sim/**` files) is flagged immediately. The companion cycle-break
issue is the work that drives the count to 0.

See ARCHITECTURE.md §"Remaining cycles (deferred to follow-up issues)"
and docs/mutation_testing_crate_split.md §"Phase 2 — break the physics
<-> sim cycle (extract shared domain types)" for context.
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
    if physics_to_sim:
        failures.append(f"src/physics has {len(physics_to_sim)} upward dep(s) to src/sim:")
        failures.extend(f"    {o}" for o in physics_to_sim)
        print(f"    FAIL: {len(physics_to_sim)} offender(s)")
    else:
        print("    OK: 0 upward deps")

    print("[2/3] src/sim/construction.rs + src/sim/per_surface_conduction.rs "
          "must not `use crate::physics::` ...")
    sim_to_physics = scan_protected_sim_files_for_physics_deps()
    if sim_to_physics:
        failures.append(
            f"protected sim files have {len(sim_to_physics)} upward dep(s) to src/physics:"
        )
        failures.extend(f"    {o}" for o in sim_to_physics)
        print(f"    FAIL: {len(sim_to_physics)} offender(s)")
    else:
        print("    OK: no cycle markers")

    print("[3/3] summary ...")
    total = len(physics_to_sim) + len(sim_to_physics)
    print(f"    Total cycle edges: {total}")
    print(f"    (Baseline: 5 physics->sim edges from the [Architecture] issue;")
    print(f"     companion cycle-break work drives this to 0.)")

    print()
    if failures:
        print("CYCLE REGRESSION DETECTED:")
        for f in failures:
            print(f"  {f}")
        print()
        print("These cycle markers were already documented in issue #2463. The")
        print("guard script enforces the *target* invariant (zero cycle edges);")
        print("the companion cycle-break issue is the work that closes the gap.")
        return 1
    print("No cycle regression. Issue #2463 cycle stays at 0 edges.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
