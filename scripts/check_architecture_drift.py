#!/usr/bin/env python3
"""
Architecture Drift Detection for Fluxion.

Parses the actual Rust source code and compares against ARCHITECTURE.md.
Fails if:
  1. A new Rust trait appears that isn't documented in ARCHITECTURE.md
  2. A documented module file no longer exists
  3. A documented trait no longer exists in code

Usage:
  python3 scripts/check_architecture_drift.py

Exit codes:
  0 — No drift detected
  1 — Drift detected (print details to stdout)
  2 — Script error
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCH_FILE = REPO_ROOT / "ARCHITECTURE.md"
# Scan source directories from all workspace members
SRC_DIRS = [REPO_ROOT / "src", REPO_ROOT / "fluxion-core" / "src"]


def find_rust_traits(src_dirs: list[Path]) -> dict[str, str]:
    """Find all pub trait definitions and their source files."""
    traits = {}
    for src_dir in src_dirs:
        for rs_file in src_dir.rglob("*.rs"):
            content = rs_file.read_text(encoding="utf-8", errors="replace")
            for match in re.finditer(r"pub\s+trait\s+(\w+)", content):
                trait_name = match.group(1)
                rel_path = rs_file.relative_to(REPO_ROOT)
                traits[trait_name] = str(rel_path)
    return traits


def find_rust_structs(src_dirs: list[Path]) -> dict[str, str]:
    """Find all pub struct definitions."""
    structs = {}
    for src_dir in src_dirs:
        for rs_file in src_dir.rglob("*.rs"):
            content = rs_file.read_text(encoding="utf-8", errors="replace")
            for match in re.finditer(r"pub\s+struct\s+(\w+)", content):
                struct_name = match.group(1)
                rel_path = rs_file.relative_to(REPO_ROOT)
                structs[struct_name] = str(rel_path)
    return structs


def find_trait_implementations(src_dirs: list[Path]) -> list[str]:
    """Find all `impl Trait for Struct` relationships."""
    impls = []
    for src_dir in src_dirs:
        for rs_file in src_dir.rglob("*.rs"):
            content = rs_file.read_text(encoding="utf-8", errors="replace")
            for match in re.finditer(r"impl\s+(\w+)\s+for\s+(\w+)", content):
                impls.append(f"{match.group(1)} -> {match.group(2)}")
    return impls


def extract_documented_traits(arch_content: str) -> set[str]:
    """Extract trait names mentioned in ARCHITECTURE.md."""
    traits = set()
    # Match backticked trait names with common suffixes
    for pattern in [
        r"`(\w+Trait)`",
        r"`(\w+Solver)`",
        r"`(\w+Schedule)`",
        r"`(\w+Source)`",
        r"`(\w+Calculations)`",
        r"`(\w+Layer)`",
        r"`(\w+Equipment)`",
        r"`(\w+Temperature)`",
    ]:
        for match in re.finditer(pattern, arch_content):
            traits.add(match.group(1))
    # Traits in code blocks within ARCHITECTURE.md
    for match in re.finditer(r"pub\s+trait\s+(\w+)", arch_content):
        traits.add(match.group(1))
    # Traits in the supporting traits table (format: | `TraitName` | path | purpose |)
    # Only look in the section titled "Supporting Traits"
    supporting_section = arch_content.split("### Supporting Traits")
    if len(supporting_section) > 1:
        table_text = supporting_section[1].split("## ")[0]  # Stop at next ## heading
        for match in re.finditer(r"\|\s*`(\w+)`\s*\|.*\|", table_text):
            name = match.group(1)
            if name[0].isupper():
                traits.add(name)
    return traits


def extract_documented_files(arch_content: str) -> set[str]:
    """Extract file paths mentioned in ARCHITECTURE.md."""
    files = set()
    for match in re.finditer(r"`(src/[\w/]+\.rs)`", arch_content):
        files.add(match.group(1))
    for match in re.finditer(r"\((src/[\w/]+\.rs)\)", arch_content):
        files.add(match.group(1))
    # From the Key Files table
    for match in re.finditer(r"`(src/[\w/]+\.rs)`", arch_content):
        files.add(match.group(1))
    return files


def check_drift() -> list[str]:
    """Run all drift checks. Returns list of drift findings."""
    findings = []

    if not ARCH_FILE.exists():
        return ["CRITICAL: ARCHITECTURE.md does not exist"]

    arch_content = ARCH_FILE.read_text(encoding="utf-8")

    # --- Check 1: Traits in code but not documented ---
    code_traits = find_rust_traits(SRC_DIRS)
    documented_traits = extract_documented_traits(arch_content)

    # Filter out internal/auxiliary traits that don't need documentation
    skip_traits = {
        "ContinuousTensor",  # internal CTA trait
        "ContinuousField",  # internal CTA trait
        "CrossValidationAdapter",  # validation infra, not physics
        "FromF64",  # internal unit conversion trait
        "ToF64",  # internal unit conversion trait
        "BatchOrchestrator",  # perf infra (rayon chunks), not a physics trait
    }

    # These are structs mentioned in ARCHITECTURE.md that get false-positived
    # because they're backticked but are NOT traits
    documented_but_not_traits = {
        "FiveR1CSolver",  # struct implementing HeatConductionSolver
        "SolAirTemperature",  # struct in sky_radiation.rs
        "CTFSolverWrapper",  # struct implementing HeatConductionSolver
        "FDSolverWrapper",  # struct implementing HeatConductionSolver
        "MultiNodeSolver",  # struct in physics/multi_node_solver.rs
    }

    # Traits documented as planned-but-not-yet-implemented in the multi-phase
    # gauge-theory migration (#1461, #1462). Remove an entry here once its
    # code lands so the drift check re-asserts documentation/code agreement.
    planned_traits = {
        "GaugeSolver",  # Phase 1b (#1462) — added to ARCHITECTURE.md in #1474
    }

    for trait_name, source_file in sorted(code_traits.items()):
        if trait_name in skip_traits:
            continue
        if trait_name not in documented_traits:
            findings.append(
                f"DRIFT: Trait `{trait_name}` exists in {source_file} "
                f"but is not documented in ARCHITECTURE.md"
            )

    # --- Check 2: Documented files that no longer exist ---
    documented_files = extract_documented_files(arch_content)
    for doc_file in sorted(documented_files):
        if not (REPO_ROOT / doc_file).exists():
            findings.append(
                f"DRIFT: File `{doc_file}` referenced in ARCHITECTURE.md no longer exists"
            )

    # --- Check 3: Documented traits that no longer exist in code ---
    for trait_name in sorted(documented_traits):
        if trait_name in documented_but_not_traits:
            continue
        if trait_name in planned_traits:
            continue
        if trait_name not in code_traits:
            findings.append(
                f"DRIFT: Trait `{trait_name}` documented in ARCHITECTURE.md "
                f"no longer exists in source code"
            )

    # --- Check 4: Key modules existence ---
    key_modules = [
        "src/physics/solver_trait.rs",
        "src/sim/thermal_model.rs",
        "src/sim/solar.rs",
        "src/sim/ventilation.rs",
        "fluxion-core/src/weather/epw.rs",
        "src/sim/sky_radiation.rs",
        "src/sim/solar_gain_distribution.rs",
    ]
    for mod_path in key_modules:
        if not (REPO_ROOT / mod_path).exists():
            findings.append(f"CRITICAL: Key module `{mod_path}` is missing")

    return findings


def main():
    print("=== Fluxion Architecture Drift Detection ===\n")

    findings = check_drift()

    if not findings:
        print("PASS: No architecture drift detected.")
        print(
            "\nDocumented traits, files, and modules are consistent with source code."
        )
        sys.exit(0)

    print(f"FAIL: {len(findings)} drift finding(s) detected:\n")
    for finding in findings:
        severity = "CRITICAL" if finding.startswith("CRITICAL") else "WARNING"
        print(f"  [{severity}] {finding}")

    print("\n--- Remediation ---")
    print("Either:")
    print("  1. Update ARCHITECTURE.md to reflect the new code structure, OR")
    print("  2. Fix the code to match the documented architecture")
    print("  3. Add false-positive trait names to `skip_traits` in this script")

    sys.exit(1)


if __name__ == "__main__":
    main()
