#!/usr/bin/env python3
"""
Cycle Regression Guard for the ASHRAE-140 leaf-types cycle (#1441).

Verifies the `sim ↔ validation` cycle documented in `ARCHITECTURE.md`
§"Remaining cycles" stays closed:

1. `fluxion-core` must NOT import from `fluxion` (i.e. no `crate::sim::*`,
   `crate::physics::*`, `crate::ai::*`, `crate::validation::*`, etc.).
2. The `sim` source tree must NOT carry a `use crate::validation::ashrae_140_cases::Orientation`
   import (the canonical cycle marker from issue #1441).
3. `fluxion_core::ashrae_cases` MUST exist and contain the moved leaf types.

Usage:
  python3 scripts/check_ashrae_cases_cycle.py

Exit codes:
  0 — no cycle regression
  1 — cycle regression detected
  2 — script error
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FLUXION_CORE_SRC = REPO_ROOT / "fluxion-core" / "src"
FLUXION_SRC = REPO_ROOT / "src"
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


def scan_fluxion_core_for_upward_deps() -> list[str]:
    """Walk fluxion-core/src and forbid any reference to crate::sim|physics|ai|validation."""
    offenders: list[str] = []
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
    for rs_file in FLUXION_CORE_SRC.rglob("*.rs"):
        rel = rs_file.relative_to(REPO_ROOT)
        for lineno, line in enumerate(rs_file.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            for prefix in upward_prefixes:
                if prefix in stripped:
                    offenders.append(f"{rel}:{lineno}: {stripped}")
    return offenders


def scan_sim_for_orientation_cycle() -> list[str]:
    """Walk src/sim and forbid `use crate::validation::ashrae_140_cases::Orientation`."""
    offenders: list[str] = []
    pattern = re.compile(r"^\s*(pub\s+)?use\s+crate::validation::ashrae_140_cases::Orientation\b")
    sim_dir = FLUXION_SRC / "sim"
    if not sim_dir.exists():
        return offenders
    for rs_file in sim_dir.rglob("*.rs"):
        rel = rs_file.relative_to(REPO_ROOT)
        for lineno, line in enumerate(rs_file.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if pattern.match(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    return offenders


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
            offenders.append(f"{ASHRAE_CASES_FILE.relative_to(REPO_ROOT)}: missing pub {type_name}")
    return offenders


def main() -> int:
    print(f"Checking cycle guards for issue #1441 (repo: {REPO_ROOT})")
    print()

    failures: list[str] = []

    print("[1/3] fluxion-core must have no upward deps to sim/physics/ai/validation ...")
    upward = scan_fluxion_core_for_upward_deps()
    if upward:
        failures.append(f"fluxion-core has {len(upward)} upward dep(s):")
        failures.extend(f"    {o}" for o in upward)
        print(f"    FAIL: {len(upward)} offender(s)")
    else:
        print("    OK: 0 upward deps")

    print("[2/3] src/sim must not `use crate::validation::ashrae_140_cases::Orientation` ...")
    cycle_markers = scan_sim_for_orientation_cycle()
    if cycle_markers:
        failures.append(f"src/sim has {len(cycle_markers)} cycle marker(s):")
        failures.extend(f"    {m}" for m in cycle_markers)
        print(f"    FAIL: {len(cycle_markers)} marker(s)")
    else:
        print("    OK: no cycle markers")

    print("[3/3] fluxion_core::ashrae_cases must contain all moved leaf types ...")
    missing = verify_ashrae_cases_module()
    if missing:
        failures.append("fluxion_core::ashrae_cases is incomplete:")
        failures.extend(f"    {m}" for m in missing)
        print(f"    FAIL: {len(missing)} missing type(s)")
    else:
        print(f"    OK: all {len(MOVED_LEAF_TYPES)} leaf types present")

    print()
    if failures:
        print("CYCLE REGRESSION DETECTED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("No cycle regression. Issue #1441 cycle stays broken.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)