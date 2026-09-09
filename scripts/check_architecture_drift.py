#!/usr/bin/env python3
"""
Architecture Drift Detection for Fluxion.

Parses the actual Rust source code and compares against ARCHITECTURE.md.
Fails if:
  1. A new Rust trait appears that isn't documented in ARCHITECTURE.md
  2. A documented module file no longer exists
  3. A documented trait no longer exists in code
  4. Trait contract invariants are violated (method signatures)
  5. Documented cycle-edge counts diverge from the cycle-guard baseline
     constants (issue #3460)

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
# Cycle-guard scripts whose BASELINE_* constants are the source of truth
# for the cycle-edge counts documented in ARCHITECTURE.md (issue #3460).
ASHRAE_CYCLE_GUARD_FILE = REPO_ROOT / "scripts" / "check_ashrae_cases_cycle.py"
PHYSICS_SIM_CYCLE_GUARD_FILE = (
    REPO_ROOT / "scripts" / "check_physics_sim_cycle.py"
)
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

    # Parse method signatures by accumulating multi-line `fn` declarations
    # (issue #3575). The four Goal #5 swap-point traits — HeatConductionSolver,
    # VentilationSchedule, ThermalModelTrait — use multi-line signatures with
    # typed-unit parameters that span 6–8 lines. The previous single-line
    # regex silently dropped every multi-line `fn`, so the baseline JSON
    # only captured the pre-#1392 surface.
    #
    # Algorithm: walk lines, collect everything from `fn ... (` until the
    # parameter list closes (paren_depth == 0), then continue until the
    # signature terminator (`;` for required methods, `{` for default
    # methods). After accumulation, the rest of the parsing logic operates
    # on the joined single-line signature as before.
    lines = trait_body.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if not stripped.startswith("fn ") or stripped.startswith("///"):
            i += 1
            continue

        # Start accumulating this fn signature.
        accumulated = stripped
        paren_depth = accumulated.count("(") - accumulated.count(")")
        i += 1

        # Continue until the parameter list closes (paren_depth == 0).
        while i < len(lines) and paren_depth > 0:
            accumulated += " " + lines[i].strip()
            paren_depth += lines[i].count("(") - lines[i].count(")")
            i += 1

        if paren_depth > 0:
            # Unbalanced parens — skip this signature and resync.
            continue

        # If the closing line hasn't already hit the terminator, keep
        # reading until `;` (required method) or `{` (default method body)
        # at the end of a line.
        if not (
            accumulated.rstrip().endswith(";") or accumulated.rstrip().endswith("{")
        ):
            while i < len(lines):
                tail = lines[i].rstrip()
                accumulated += " " + lines[i].strip()
                i += 1
                if tail.endswith(";") or tail.endswith("{"):
                    break

        # Extract fn name and params from the joined signature.
        fn_match = re.match(r"fn\s+(\w+)\s*\(([^)]*)\)", accumulated)
        if not fn_match:
            continue
        fn_name = fn_match.group(1)
        params_str = fn_match.group(2)

        # Find return type: scan from the closing paren forward.
        # For declarations: `-> Type;`  For bodies: `-> Type {`
        return_type = ""
        arrow_pos = accumulated.find("->")
        if arrow_pos != -1:
            ret_part = accumulated[arrow_pos + 2 :].strip()
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
        inner_params = params_str.strip()
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


def parse_guard_constant(script_path: Path, constant: str) -> int | None:
    """Regex-parse ``<CONSTANT> = <int>`` from a cycle-guard script.

    Deliberately regex-based rather than an import: the text parse reads
    exactly the literal a human reviewer reads when re-syncing docs, with
    no import-time side effects. Returns ``None`` when the script or the
    constant is missing so callers can fail loudly instead of guessing.
    """
    if not script_path.exists():
        return None
    match = re.search(
        rf"^{constant}\s*=\s*(\d+)",
        script_path.read_text(encoding="utf-8", errors="replace"),
        re.MULTILINE,
    )
    return int(match.group(1)) if match else None


def check_cycle_edge_count_drift(arch_content: str) -> list[str]:
    """Compare ARCHITECTURE.md's documented cycle-edge counts with the
    cycle guards' BASELINE_* constants (issue #3460).

    ARCHITECTURE.md narrates the sim<->validation and physics<->sim cycle
    magnitudes in prose; those numbers drifted from the guard constants
    once already (issue #3460: "~220 directional edges" documented vs a
    measured 254). This check re-parses the specific claims and fails on
    mismatch *or* removal, so docs and guards cannot diverge silently.

    All prose regexes run against whitespace-normalised text so markdown
    line-wrapping cannot hide a drifted literal.
    """
    findings: list[str] = []

    ashrae = {
        "sim→validation": parse_guard_constant(
            ASHRAE_CYCLE_GUARD_FILE, "BASELINE_SIM_TO_VALIDATION"
        ),
        "validation→sim": parse_guard_constant(
            ASHRAE_CYCLE_GUARD_FILE, "BASELINE_VALIDATION_TO_SIM"
        ),
        "validation→physics": parse_guard_constant(
            ASHRAE_CYCLE_GUARD_FILE, "BASELINE_VALIDATION_TO_PHYSICS"
        ),
        "validation→weather": parse_guard_constant(
            ASHRAE_CYCLE_GUARD_FILE, "BASELINE_VALIDATION_TO_WEATHER"
        ),
    }
    physics_sim = {
        "physics→sim": parse_guard_constant(
            PHYSICS_SIM_CYCLE_GUARD_FILE, "BASELINE_PHYSICS_TO_SIM"
        ),
        "sim→physics": parse_guard_constant(
            PHYSICS_SIM_CYCLE_GUARD_FILE, "BASELINE_SIM_TO_PHYSICS"
        ),
    }

    prose = re.sub(r"\s+", " ", arch_content)

    # (label, regex over normalised prose, guard value, constant name)
    scalar_claims = [
        (
            "sim→validation baseline",
            r"documented baseline \(currently (\d+)\)",
            ashrae["sim→validation"],
            "BASELINE_SIM_TO_VALIDATION",
        ),
        (
            "validation→sim baseline",
            r"`crate::sim::\*` \(baseline (\d+)\)",
            ashrae["validation→sim"],
            "BASELINE_VALIDATION_TO_SIM",
        ),
        (
            "validation→physics baseline",
            r"`crate::physics::\*` \(baseline (\d+)\)",
            ashrae["validation→physics"],
            "BASELINE_VALIDATION_TO_PHYSICS",
        ),
        (
            "validation→weather baseline",
            r"`crate::weather::\*` \(baseline (\d+)\)",
            ashrae["validation→weather"],
            "BASELINE_VALIDATION_TO_WEATHER",
        ),
        (
            "sim→physics baseline",
            r"`BASELINE_SIM_TO_PHYSICS = (\d+)`",
            physics_sim["sim→physics"],
            "BASELINE_SIM_TO_PHYSICS",
        ),
    ]
    for label, pattern, guard_value, const_name in scalar_claims:
        if guard_value is None:
            findings.append(
                f"DRIFT: {const_name} not parseable from its cycle-guard "
                f"script — cannot verify the ARCHITECTURE.md {label} claim"
            )
            continue
        match = re.search(pattern, prose)
        if not match:
            findings.append(
                f"DRIFT: ARCHITECTURE.md no longer documents the {label} "
                f"claim (issue #3460 sync guard cannot verify it)"
            )
        elif int(match.group(1)) != guard_value:
            findings.append(
                f"DRIFT: ARCHITECTURE.md documents the {label} as "
                f"{match.group(1)} but the cycle guard defines "
                f"{const_name} = {guard_value}"
            )

    # Composite claim: "~N directional edges remain (a sim→validation +
    # b validation→sim + c validation→physics + d validation→weather)".
    breakdown = re.search(
        r"~(\d+) directional edges remain \((\d+) sim→validation \+ (\d+) "
        r"validation→sim \+ (\d+) validation→physics \+ (\d+) "
        r"validation→weather\)",
        prose,
    )
    if not breakdown:
        findings.append(
            "DRIFT: ARCHITECTURE.md no longer documents the directional-edge "
            "breakdown sentence (issue #3460 sync guard)"
        )
    else:
        parts = {
            "sim→validation": int(breakdown.group(2)),
            "validation→sim": int(breakdown.group(3)),
            "validation→physics": int(breakdown.group(4)),
            "validation→weather": int(breakdown.group(5)),
        }
        for label, doc_value in parts.items():
            guard_value = ashrae[label]
            if guard_value is not None and doc_value != guard_value:
                findings.append(
                    f"DRIFT: ARCHITECTURE.md breakdown documents {label} as "
                    f"{doc_value} but the cycle guard baseline is {guard_value}"
                )
        documented_total = int(breakdown.group(1))
        if documented_total != sum(parts.values()):
            findings.append(
                f"DRIFT: ARCHITECTURE.md claims ~{documented_total} "
                f"directional edges but its own breakdown sums to "
                f"{sum(parts.values())}"
            )

    # Composite claim: "**0+N edges** (0 physics→sim + M sim→physics)".
    # Both captures are the sim->physics count; the physics->sim direction
    # is pinned by the literal 0s (if it ever grows, the sentence shape
    # changes and the "no longer documents" finding below fires instead).
    total_claim = re.search(
        r"\*\*0\+(\d+) edges\*\* \(0 physics→sim \+ (\d+) sim→physics\)",
        prose,
    )
    if not total_claim:
        findings.append(
            "DRIFT: ARCHITECTURE.md no longer documents the physics<->sim "
            "0+N baseline sentence (issue #3460 sync guard)"
        )
    else:
        if (
            physics_sim["sim→physics"] is not None
            and int(total_claim.group(1)) != physics_sim["sim→physics"]
        ):
            findings.append(
                f"DRIFT: ARCHITECTURE.md 0+N baseline headline documents "
                f"sim→physics as {total_claim.group(1)} but "
                f"BASELINE_SIM_TO_PHYSICS = {physics_sim['sim→physics']}"
            )
        if (
            physics_sim["sim→physics"] is not None
            and int(total_claim.group(2)) != physics_sim["sim→physics"]
        ):
            findings.append(
                f"DRIFT: ARCHITECTURE.md 0+N baseline parenthetical "
                f"documents sim→physics as {total_claim.group(2)} but "
                f"BASELINE_SIM_TO_PHYSICS = {physics_sim['sim→physics']}"
            )

    return findings


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
        "Circuit",  # BDF benchmark-circuit trait in bdf_benchmarks.rs (#3339); benchmark-fixture surface, not a physics swap-point
        "DynCircuit",  # BDF dyn-dispatch wrapper for Circuit + DaeSystem in bdf_benchmarks.rs (#3339); internal boxed-driver surface
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
        # Issue #3555 burn-down: `src/sim/solar_gain_distribution.rs` was
        # a wired-but-dead sibling of `src/sim/solar.rs` and has been
        # deleted (per-surface distribution now lives in
        # `src/sim/thermal_model_data/incident_solar_accumulator.rs`).
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

    # --- Check 6: cycle-edge count claims vs guard baselines (#3460) ---
    findings.extend(check_cycle_edge_count_drift(arch_content))

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
