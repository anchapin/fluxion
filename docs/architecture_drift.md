# Architecture Drift Check

Trait contract verification for Fluxion's core physics traits.

## Overview

The `scripts/check_architecture_drift.py` script verifies that the codebase
remains consistent with the documented architecture. In addition to file-level
and trait-level checks, it now includes **trait contract verification** for
the three core physics swap-point traits:

- `HeatConductionSolver` (`src/physics/solver_trait.rs`)
- `VentilationSchedule` (`src/sim/ventilation.rs`)
- `ThermalModelTrait` (`src/sim/thermal_model.rs`)

## What is Checked

### 1. Trait Existence
- All traits in source code must be documented in `ARCHITECTURE.md`
- All documented traits must exist in source code

### 2. Key Module Existence
Critical modules must exist at their documented paths:
- `src/physics/solver_trait.rs`
- `src/sim/thermal_model.rs`
- `src/sim/ventilation.rs`
- `src/sim/solar.rs`
- `fluxion-core/src/weather/epw.rs`

### 3. Trait Contract Invariants

Certain methods have required receivers that are enforced as invariant checks:

| Trait | Method | Required Receiver | Rationale |
|-------|--------|------------------|----------|
| `HeatConductionSolver` | `step` | `&mut self` | Mutates internal thermal-mass state |
| `HeatConductionSolver` | `steady_state_flux` | `&self` | Pure query; no side effects |
| `HeatConductionSolver` | `energy_storage_rate` | `&self` | Read-only accessor |
| `VentilationSchedule` | `get_ach` | `&self` | Read-only query |

### 4. Trait Contract Baseline

A committed baseline at `scripts/trait_contract_baseline.json` records the
expected method signatures for key traits. Changes to signatures (receiver type,
return type) are detected as **contract drift**.

The baseline is authoritative: any drift from it fails CI until the baseline
is explicitly updated.

## Usage

```bash
# Run all architecture drift checks
python3 scripts/check_architecture_drift.py

# Update the trait contract baseline
python3 scripts/check_architecture_drift.py --update-baseline
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | No drift detected |
| 1 | Drift detected (details printed to stdout) |
| 2 | Script error |

## Baseline Update Workflow

When a trait signature change is intentional and approved:

1. Make the code change
2. Run `python3 scripts/check_architecture_drift.py --update-baseline`
3. Commit both the code change and the updated baseline

```bash
git add src/physics/solver_trait.rs scripts/trait_contract_baseline.json
git commit -m "feat(physics): update HeatConductionSolver::step signature"
```

## CI Integration

The script runs as part of the `Architecture Drift Detection` CI job.
Failures block the PR with a clear error message indicating which contract
was violated.

## Adding New Invariants

To add a new invariant check, edit `check_trait_invariants()` in
`scripts/check_architecture_drift.py`:

```python
if "TraitName" in contracts:
    contract = contracts["TraitName"]
    if "method_name" in contract.methods:
        sig = contract.methods["method_name"]
        if sig.receiver != "&self":
            violations.append(
                f"INVARIANT VIOLATION: TraitName::method_name must be `&self`"
            )
```

## References

- Issue #2377: Architecture Drift Check Improvements for Trait Contracts
- ARCHITECTURE.md §Trait Contracts
- ADR-002: Multi-node thermal mass (9R4C) promotion
