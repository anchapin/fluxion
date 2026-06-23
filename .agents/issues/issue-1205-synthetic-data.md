## Issue Description

67% of conduction reference data in `tests/reference_data/conduction/` is **synthetic** — not from EnergyPlus. This creates false validation confidence.

| File | Status | Source |
|------|--------|--------|
| `step_response_200mm_concrete.csv` | Real | EnergyPlus 25.2.0 |
| `step_response_fixed_zone_20c.csv` | Real | EnergyPlus 25.2.0 |
| `step_response_lightweight.csv` | SYNTHETIC | Generated 2026-06-12 |
| `step_response_roof.csv` | SYNTHETIC | Generated 2026-06-12 |
| `step_response_floor.csv` | SYNTHETIC | Generated 2026-06-12 |
| `step_response_composite.csv` | SYNTHETIC | Generated 2026-06-12 |

The CSV headers themselves say: `# Generated: 2026-06-12 (synthetic for testing)`

## Impact

- `tests/conduction_5r1c_isolation.rs` passes steady-state tests against synthetic data
- 15 steady-state tests validate against non-EnergyPlus values
- Violates ARCHITECTURE.md Phase 1 strategy: "Each module must be unit-tested in isolation against EnergyPlus reference data"

## Fix Options

1. **Option A**: Regenerate 4 files from E+ 25.2.0 using IDF models in `tests/reference_data/energyplus_models/`
2. **Option B**: Relabel as "analytical test fixtures" with clear documentation that they're not E+ reference

## Files Affected

- `tests/reference_data/conduction/` (4 synthetic files)
- `tests/conduction_5r1c_isolation.rs`

## Acceptance Criteria

- [ ] All 6 conduction reference CSVs trace back to EnergyPlus or are clearly labeled as synthetic fixtures
- [ ] Test docstrings explicitly state data source
- [ ] No synthetic data passes as EnergyPlus reference

## References

- ARCHITECTURE.md Phase 1 validation strategy (lines 477-489)