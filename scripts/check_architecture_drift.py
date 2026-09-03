#!/usr/bin/env python3
"""
Architecture Drift Detection for Fluxion.

Parses the actual Rust source code and compares against ARCHITECTURE.md.
Fails if:
  1. A new Rust trait appears that isn't documented in ARCHITECTURE.md
  2. A documented module file no longer exists
  3. A documented trait no longer exists in code
  4. Trait contract invariants are violated (method signatures)

Usage:
  python3 scripts/check_architecture_drift.py

Exit codes:
  0 — No drift detected
  1 — Drift detected (print details to stdout)
  2 — Script error
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCH_FILE = REPO_ROOT / "ARCHITECTURE.md"
BASELINE_FILE = REPO_ROOT / "scripts" / "trait_contract_baseline.json"
# Scan source directories from all workspace members
SRC_DIRS = [
    REPO_ROOT / "src",
    REPO_ROOT / "fluxion-core" / "src",
    REPO_ROOT / "fluxion-grid" / "src",
]

# Key trait source files for contract verification
KEY_TRAIT_FILES = {
    "HeatConductionSolver": REPO_ROOT / "src" / "physics" / "solver_trait.rs",
    "VentilationSchedule": REPO_ROOT / "src" / "sim" / "ventilation.rs",
    "ThermalModelTrait": REPO_ROOT / "src" / "sim" / "thermal_model.rs",
}


@dataclass
class MethodSignature:
    name: str
    receiver: str  # "&self", "&mut self", or "" for static
    params: list[str]
    return_type: str


@dataclass
class TraitContract:
    trait_name: str
    source_file: str
    methods: dict[str, MethodSignature]


def parse_trait_methods(content: str, trait_name: str) -> dict[str, MethodSignature]:
    """Parse all method signatures from a trait definition in Rust source."""
    methods = {}

    # Find the trait block
    trait_pattern = rf"pub\s+trait\s+{trait_name}\s*[:\{{][^{{]*\{{"
    trait_match = re.search(trait_pattern, content, re.DOTALL)
    if not trait_match:
        # Try simpler pattern
        trait_pattern = rf"pub\s+trait\s+{trait_name}\s*\{{"
        trait_match = re.search(trait_pattern, content, re.DOTALL)

    if not trait_match:
        return methods

    # Extract trait body (everything between { and matching })
    start = trait_match.end() - 1  # Position of opening {
    depth = 1
    pos = start + 1
    while pos < len(content) and depth > 0:
        if content[pos] == "{":
            depth += 1
        elif content[pos] == "}":
            depth -= 1
        pos += 1
    trait_body = content[start:pos]

    # Parse method signatures using a state machine that correctly handles
    # both trait declarations (ending in `;`) and methods with bodies (ending in `{`).
    # This avoids the regex bug where [^{{]+ greedily captures too much.
    fn_lines = trait_body.split("\n")
    for i, line in enumerate(fn_lines):
        stripped = line.lstrip()
        if not stripped.startswith("fn ") or stripped.startswith("///"):
            continue

        # Extract fn name and params
        fn_match = re.match(r"fn\s+(\w+)\s*\(([^)]*)\)", stripped)
        if not fn_match:
            continue
        fn_name = fn_match.group(1)
        params_str = fn_match.group(2)

        # Find return type: scan from end of line backwards
        # For declarations: `-> Type;`  For bodies: `-> Type {`
        return_type = ""
        arrow_pos = stripped.find("->")
        if arrow_pos != -1:
            ret_part = stripped[arrow_pos + 2 :].strip()
            # Find the end of the return type
            if "{" in ret_part:
                return_type = "-> " + ret_part[: ret_part.find("{")].strip()
            elif ";" in ret_part:
                return_type = "-> " + ret_part[: ret_part.find(";")].strip()
            else:
                return_type = "-> " + ret_part.strip()

        # Parse receiver
        receiver = ""
        if "&mut self" in params_str:
            receiver = "&mut self"
        elif "&self" in params_str:
            receiver = "&self"

        # Parse parameters (strip self variants)
        params = []
        inner_params = params_str.strip("()")
        if inner_params:
            for param in inner_params.split(","):
                param = param.strip()
                if (
                    param
                    and not param.startswith("&mut self")
                    and not param.startswith("&self")
                ):
                    params.append(param)

        methods[fn_name] = MethodSignature(
            name=fn_name,
            receiver=receiver,
            params=params,
            return_type=return_type.strip(),
        )

    return methods


def extract_trait_contracts() -> dict[str, TraitContract]:
    """Extract trait contracts from all key source files."""
    contracts = {}

    for trait_name, source_path in KEY_TRAIT_FILES.items():
        if not source_path.exists():
            continue

        content = source_path.read_text(encoding="utf-8", errors="replace")
        methods = parse_trait_methods(content, trait_name)

        if methods:
            contracts[trait_name] = TraitContract(
                trait_name=trait_name,
                source_file=str(source_path.relative_to(REPO_ROOT)),
                methods=methods,
            )

    return contracts


def check_trait_invariants(contracts: dict[str, TraitContract]) -> list[str]:
    """Check trait contract invariants. Returns list of violations."""
    violations = []

    # HeatConductionSolver invariants
    if "HeatConductionSolver" in contracts:
        contract = contracts["HeatConductionSolver"]

        # step() must be &mut self
        if "step" in contract.methods:
            step_sig = contract.methods["step"]
            if step_sig.receiver != "&mut self":
                violations.append(
                    f"INVARIANT VIOLATION: HeatConductionSolver::step must be `&mut self`, "
                    f"found `{step_sig.receiver}` in {contract.source_file}"
                )

        # steady_state_flux() must be &self (pure query method)
        if "steady_state_flux" in contract.methods:
            ssf_sig = contract.methods["steady_state_flux"]
            if ssf_sig.receiver != "&self":
                violations.append(
                    f"INVARIANT VIOLATION: HeatConductionSolver::steady_state_flux must be `&self` "
                    f"(pure query), found `{ssf_sig.receiver}` in {contract.source_file}"
                )

        # energy_storage_rate() must be &self
        if "energy_storage_rate" in contract.methods:
            esr_sig = contract.methods["energy_storage_rate"]
            if esr_sig.receiver != "&self":
                violations.append(
                    f"INVARIANT VIOLATION: HeatConductionSolver::energy_storage_rate must be `&self`, "
                    f"found `{esr_sig.receiver}` in {contract.source_file}"
                )

    # VentilationSchedule invariants
    if "VentilationSchedule" in contracts:
        contract = contracts["VentilationSchedule"]

        # get_ach() must be &self
        if "get_ach" in contract.methods:
            ach_sig = contract.methods["get_ach"]
            if ach_sig.receiver != "&self":
                violations.append(
                    f"INVARIANT VIOLATION: VentilationSchedule::get_ach must be `&self`, "
                    f"found `{ach_sig.receiver}` in {contract.source_file}"
                )

    return violations


def serialize_contract(contract: TraitContract) -> dict:
    """Serialize a TraitContract to a dict for JSON serialization."""
    return {
        "trait_name": contract.trait_name,
        "source_file": contract.source_file,
        "methods": {
            name: {
                "name": sig.name,
                "receiver": sig.receiver,
                "params": sig.params,
                "return_type": sig.return_type,
            }
            for name, sig in contract.methods.items()
        },
    }


def deserialize_contract(data: dict) -> TraitContract:
    """Deserialize a dict to a TraitContract."""
    methods = {
        name: MethodSignature(
            name=sig["name"],
            receiver=sig["receiver"],
            params=sig["params"],
            return_type=sig["return_type"],
        )
        for name, sig in data["methods"].items()
    }
    return TraitContract(
        trait_name=data["trait_name"],
        source_file=data["source_file"],
        methods=methods,
    )


def check_contract_drift(
    current: dict[str, TraitContract], baseline: dict[str, TraitContract]
) -> list[str]:
    """Check for drift between current contracts and baseline. Returns list of violations."""
    violations = []

    for trait_name, current_contract in current.items():
        if trait_name not in baseline:
            continue

        baseline_contract = baseline[trait_name]

        # Check methods match
        for method_name, current_sig in current_contract.methods.items():
            if method_name not in baseline_contract.methods:
                violations.append(
                    f"CONTRACT DRIFT: New method `{method_name}` added to "
                    f"{trait_name} in {current_contract.source_file} — baseline must be updated"
                )
            else:
                baseline_sig = baseline_contract.methods[method_name]
                if current_sig.receiver != baseline_sig.receiver:
                    violations.append(
                        f"CONTRACT DRIFT: `{trait_name}::{method_name}` receiver changed "
                        f"from `{baseline_sig.receiver}` to `{current_sig.receiver}` "
                        f"in {current_contract.source_file}"
                    )
                if current_sig.return_type != baseline_sig.return_type:
                    violations.append(
                        f"CONTRACT DRIFT: `{trait_name}::{method_name}` return type changed "
                        f"from `{baseline_sig.return_type}` to `{current_sig.return_type}` "
                        f"in {current_contract.source_file}"
                    )

        # Check for removed methods
        for method_name in baseline_contract.methods:
            if method_name not in current_contract.methods:
                violations.append(
                    f"CONTRACT DRIFT: Method `{method_name}` removed from "
                    f"{trait_name} — baseline must be updated"
                )

    return violations


def load_or_create_baseline() -> tuple[dict[str, TraitContract], bool]:
    """Load baseline from file or create from current state. Returns (baseline, created_new)."""
    contracts = extract_trait_contracts()

    if BASELINE_FILE.exists():
        with open(BASELINE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        baseline = {name: deserialize_contract(cd) for name, cd in data.items()}
        return baseline, False
    else:
        # Create baseline from current state
        BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {name: serialize_contract(c) for name, c in contracts.items()},
                f,
                indent=2,
            )
        return contracts, True


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


def check_drift() -> tuple[list[str], bool]:
    """Run all drift checks. Returns (findings, baseline_was_created)."""
    findings = []
    baseline_created = False

    if not ARCH_FILE.exists():
        return ["CRITICAL: ARCHITECTURE.md does not exist"], False

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
        "ZoneEquipment",  # zone-level HVAC equipment trait (src/sim/hvac/zone_equipment.rs)
        "DaeSystem",  # BDF ODE system trait in bdf_engine.rs (#2074)
        "ResidualFunction",  # BDF residual trait in bdf_engine.rs (#2074)
        "DecoupledLoopEquipment",  # ECS/rayon parallel loop evaluator (#1991)
        "PhysicsEquipment",  # pre-existing drift on develop
        "FfdSolver",  # FFD solver trait in src/sim/loose_coupling.rs (new in #2420)
        "Sealed",  # private sealed-trait marker used by AlgebraicFloat in src/physics/fp_algebraic.rs (#3322); not a public extension point
    }

    # These are structs mentioned in ARCHITECTURE.md that get false-positived
    # because they're backticked but are NOT traits
    documented_but_not_traits = {
        "FiveR1CSolver",  # struct implementing HeatConductionSolver
        "SolAirTemperature",  # struct in sky_radiation.rs
        "CTFSolverWrapper",  # struct implementing HeatConductionSolver
        "FDSolverWrapper",  # struct implementing HeatConductionSolver
        "MultiNodeSolver",  # struct in physics/multi_node_solver.rs
        "JointConvergenceSolver",  # struct in fluxion-grid/src/lib.rs
        "UrbanRadiationSolver",  # struct in fluxion-city/src/lib.rs (sparse module)
        "ConstructionLayer",  # struct in fluxion-core/src/construction.rs (#2462)
        "PerSurfaceConductionSolver",  # struct in fluxion-core/src/per_surface_conduction.rs (#2462)
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

    # --- Check 5: Trait contract invariants and baseline comparison ---
    baseline, was_created = load_or_create_baseline()
    if was_created:
        baseline_created = True
    else:
        current_contracts = extract_trait_contracts()

        # Check invariants first (these are always errors)
        invariant_violations = check_trait_invariants(current_contracts)
        findings.extend(invariant_violations)

        # Check baseline drift (only if no invariant violations)
        if not invariant_violations:
            contract_drift = check_contract_drift(current_contracts, baseline)
            findings.extend(contract_drift)

    return findings, baseline_created


def main():
    print("=== Fluxion Architecture Drift Detection ===\n")

    findings, baseline_created = check_drift()

    if baseline_created:
        print(
            "INFO: Baseline trait contract file created at "
            f"{BASELINE_FILE.relative_to(REPO_ROOT)}"
        )
        print(
            "      This baseline must be committed alongside any trait signature changes."
        )
        print()

    if not findings:
        print("PASS: No architecture drift detected.")
        print(
            "\nDocumented traits, files, and modules are consistent with source code."
        )
        if baseline_created:
            print(
                "\nNOTE: A new baseline was created. Commit it with:\n"
                f"  git add {BASELINE_FILE.relative_to(REPO_ROOT)}\n"
                "  git commit -m 'chore: update trait contract baseline'"
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
    print("  4. For trait contract drift, update the baseline:")
    print("     python3 scripts/check_architecture_drift.py --update-baseline")

    sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fluxion Architecture Drift Detection")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Force update of the trait contract baseline",
    )
    args = parser.parse_args()

    if args.update_baseline:
        contracts = extract_trait_contracts()
        BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {name: serialize_contract(c) for name, c in contracts.items()},
                f,
                indent=2,
            )
        print(f"Baseline updated: {BASELINE_FILE.relative_to(REPO_ROOT)}")
        sys.exit(0)

    main()
