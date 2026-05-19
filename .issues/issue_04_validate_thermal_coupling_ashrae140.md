## Problem

After implementing furniture factor-based C_me and h_tr_me values, need to validate that the thermal model produces results consistent with ASHRAE 140 reference values for Case 600 and Case 900.

## Research Baseline

ASHRAE 140-2023 does NOT explicitly specify internal mass values. The standard implies:
- Case 600 (lightweight): Thermal mass from envelope only, minimal furniture impact
- Case 900 (heavyweight): High mass from thick concrete, but internal mass (furniture) still contributes

Expected behavior:
- Internal mass should respond faster than envelope mass (τ_me ~3.4 hours vs τ_em ~8 hours)
- Combined thermal buffer effect should produce realistic damping

## Validation Targets

### Case 900 (High Mass)
| Metric | Target Range | Notes |
|--------|--------------|-------|
| Peak heating load | ASHRAE reference ±10% | |
| Peak cooling load | ASHRAE reference ±10% | |
| Internal mass temp | Should lag envelope temp | Furniture responds faster |
| Time constant τ_me | ~3.4 hours | C_me/h_tr_me |

### Case 600 (Low Mass)
| Metric | Target Range | Notes |
|--------|--------------|-------|
| Peak heating load | ASHRAE reference ±10% | |
| Peak cooling load | ASHRAE reference ±10% | |
| Time constant τ_me | ~3.4 hours | Same physics |

## Tasks

- [ ] Run full ASHRAE 140 test suite after Issues 1-3 are resolved
- [ ] Compare Case 900 heating/cooling loads against reference data
- [ ] Compare Case 600 heating/cooling loads against reference data
- [ ] Verify internal mass temperature waveform shows expected faster response
- [ ] Check that envelope and internal masses respond together (strong coupling)
- [ ] Document any deviations and root-cause if outside ±10%

## Files to Check/Run

- `tests/test_ashrae_140_cases.rs`
- ASHRAE 140 reference data in `refdata/` or test fixtures

## Reference

- research_internal_mass_capacitance.md Section 3 (ASHRAE 140 Requirements)
- research_internal_mass_capacitance.md Section 5 (Time Constant Analysis)
