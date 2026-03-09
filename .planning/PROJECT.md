# Fluxion - ASHRAE 140 Validation Fixes

## What This Is

Fluxion is a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. This project focuses on achieving full ASHRAE Standard 140 validation compliance by fixing systematic errors in heating calculations, peak load predictions, and high-mass building cases. The goal is to ensure the physics engine accurately predicts energy consumption within ASHRAE tolerance bands (±15% annual, ±10% monthly).

## Core Value

Every ASHRAE 140 test case must pass with energy consumption and peak load predictions within ASHRAE tolerance bands. This is the validation foundation for the entire physics engine—without accurate reference validation, the engine cannot be trusted for production building design optimization.

## Requirements

### Validated

- ✓ ISO 13790 5R1C thermal network implementation — existing
- ✓ PyO3 Python bindings with BatchOracle/Model classes — existing
- ✓ ONNX Runtime surrogate integration — existing
- ✓ CTA VectorField operations — existing
- ✓ Case 960 multi-zone sunspace — existing (currently passes)

### Active

- [ ] **ASHRA-01**: All ASHRAE 140 test cases pass with ±15% annual energy tolerance
- [ ] **ASHRA-02**: All ASHRAE 140 test cases pass with ±10% monthly energy tolerance
- [ ] **ASHRA-03**: Peak heating loads match ASHRAE reference values (currently 78.79% mean error)
- [ ] **ASHRA-04**: Peak cooling loads match ASHRAE reference values
- [ ] **ASHRA-05**: High-mass building cases (900-series) pass validation
- [ ] **ASHRA-06**: Annual heating load over-prediction corrected (currently systematically high)
- [ ] **ASHRA-07**: Mean Absolute Error reduced to <15% (currently 78.79%)
- [ ] **ASHRA-08**: Max Deviation reduced to acceptable range (currently 471.66%)
- [ ] **ASHRA-09**: 39 failing cases converted to passing or warnings
- [ ] **ASHRA-10**: 9 warning cases resolved to passing status
- [ ] **ASHRA-11**: Inter-zone heat transfer calculations verified and corrected
- [ ] **ASHRA-12**: Thermal mass dynamics response time improved
- [ ] **ASHRA-13**: Solar gain calculations validated against reference
- [ ] **ASHRA-14**: External convection boundary conditions verified

### Out of Scope

- Adding new building types beyond ASHRAE 140 standard cases
- Implementing 6R2C thermal model (deferred to future)
- FMI 3.0 co-simulation support (deferred to future)
- RL policy integration improvements (deferred to future)

## Context

**Current Validation Status:**
- Total validation metrics: 64
- Passed: 16 (25%)
- Warnings: 9 (14%)
- Failed: 39 (61%)
- Mean Absolute Error: 78.79%
- Max Deviation: 471.66%

**Known Systematic Issues:**
1. **Annual heating loads** - consistently over-predicted across multiple cases
2. **Peak heating values** - significant deviations from ASHRAE reference values
3. **High-mass building cases** (900-series) - systematic accuracy problems
4. **Multiple error types suspected** - heat transfer, HVAC logic, mass dynamics, solar/external

**Existing Infrastructure:**
- ASHRAE 140 test suite with 64 validation metrics
- Diagnostic tools: `ashrae_140_diagnostic_test.rs`, `ashrae_140_validation.rs`
- Case-specific tests: Cases 600, 610, 620, 630, 640, 650, 900, 960, 195
- Data generation tools: `ashrae_140_generator.py`
- Documentation: `ASHRAE140_RESULTS.md`, `ASHRAE_140_DIAGNOSTICS.md`, `ASHRAE140_MILESTONES.md`

**Physics Architecture:**
- 5R1C thermal network (ISO 13790-compliant)
- Two-zone heat transfer with inter-zone conduction
- Thermal mass with continuous time dynamics
- HVAC setpoint control with load calculation
- Solar gain with beam/diffuse decomposition

## Constraints

- **ASHRAE 140 Tolerance Bands**: ±15% annual energy, ±10% monthly energy
- **ISO 13790 Compliance**: Must maintain 5R1C thermal network structure
- **Existing Architecture**: Preserve PyO3 bindings and BatchOracle/Model API pattern
- **Performance**: Maintain high-throughput evaluation capability (>1,000 configs/sec)
- **Physics Accuracy**: Fixes must improve accuracy, not reduce performance
- **Brownfield Constraints**: Work within existing codebase structure without major refactoring

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Full overhaul approach | Multiple error types suspected across heating, peaks, mass dynamics — comprehensive fixes needed | — Pending |

---
*Last updated: 2026-03-08 after questioning*
