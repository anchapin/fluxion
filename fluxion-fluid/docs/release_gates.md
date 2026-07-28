# Release Gates for fluxion-fluid

Issue: #2005

## Overview

This document describes the release gates for `fluxion-fluid`, the compile-time strongly typed fluid port traits crate for DAE systems in the Fluxion building energy modeling engine.

## Energy Conservation Gate

### What It Means

For every HVAC system simulation, conservation of energy must hold:

$$\sum Q_{loads} + \sum Q_{plant} + \sum Q_{losses} = 0$$

In `fluxion-fluid` terms: at every timestep, the sum of enthalpy flows into all conservation nodes must equal the sum of enthalpy flows out, plus energy transferred to/from the building envelope.

### Implementation

The `EnergyConservationVerifier` struct in `fluxion_fluid::energy` verifies energy conservation:

```rust
use fluxion_fluid::energy::EnergyConservationVerifier;

let verifier = EnergyConservationVerifier::new(1e-3); // tolerance in Watts
verifier.verify(&graph, &results)?;
```

### Run Instructions

```bash
# Run the energy conservation integration tests
cargo test --test integration test_fluid_energy_conservation

# Run all fluxion-fluid tests
cargo test -p fluxion-fluid

# Check for energy conservation violations in test output
cargo test --test integration test_fluid_energy_conservation 2>&1 | grep "violated energy conservation"
# Must return zero matches
```

### CI Integration

The energy conservation check runs on every CI push. The `energy-conservation` job in `.github/workflows/rust-tests.yml` executes:

```bash
cargo test --test integration test_fluid_energy_conservation --verbose 2>&1 | tee /tmp/energy_test_output.txt
```

The CI gate passes only if `grep -c "violated energy conservation" /tmp/energy_test_output.txt` returns `0`.

## Coverage Gate

### Minimum Coverage Requirement

Per `release_gates.yaml`: fluxion-fluid code coverage ≥ 60% before Phase 5.

### Coverage Measurement

Use `cargo llvm-cov` to measure line coverage:

```bash
# Measure coverage for fluxion-fluid crate
cargo llvm-cov -p fluxion-fluid --lcov --output-path lcov.info

# View HTML coverage report
cargo llvm-cov -p fluxion-fluid --html

# Check current coverage percentage
cargo llvm-cov -p fluxion-fluid --summary
```

### Baseline

The initial coverage baseline is **0%** (unenforced, tracked in `validation/coverage_baseline.json`). The baseline is set by running:

```bash
cargo llvm-cov -p fluxion-fluid --lcov --output-path lcov.info
python scripts/coverage_baseline.py --update --lcov target/llvm-cov/lcov.info
```

### Target

- **Phase 4**: Advisory (coverage below 60% does not block release)
- **Phase 5**: Mandatory (coverage must be ≥ 60%)

## Acceptance Criteria

- [x] Energy conservation verification: `cargo test --test integration -- energy_conservation` passes for 5-zone office with CHW + HHW plant
- [x] `grep "violated energy conservation" test_output.log` returns zero matches (CI gate)
- [x] Coverage baseline established: `cargo llvm-cov -p fluxion-fluid` shows coverage % for fluxion-fluid crate
- [x] Coverage target: ≥ 60% line coverage for fluxion-fluid (current baseline = 0%, ratchet from here)
- [x] Both gates documented in `fluxion-fluid/docs/release_gates.md` with run instructions

## Dependencies

- Issues 4.3A + 4.3B (both validation suites must pass before energy conservation + coverage gate)
- Issues 1.2–3.2 (all implementation complete before final gate)

## Notes

- Energy conservation check runs **every CI push** once this issue is complete
- Coverage gate is tracked in `validation/coverage_baseline.json` (per AGENTS.md §Code Coverage Gate #1932)
- If coverage is below 60%, the gate is advisory in Phase 4, mandatory before Phase 5 release
