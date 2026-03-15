# Phase 8 Context: Critical Issue Resolution

## Phase Overview

**Phase 8: Critical Issue Resolution** is the first phase of v0.3 maintenance release. Its sole purpose is to fix the blocking issue with ASHRAE 140 Case 960 (sunspace) where annual cooling energy exceeds the reference range, preventing full validation suite from passing.

## Requirements

- **CASE960-01**: Investigate root cause of Case 960 annual cooling failure
- **CASE960-02**: Implement fix without breaking other validated cases
- **CASE960-03**: Validate with full ASHRAE 140 suite

## Current State (2026-03-11)

### Validation Results

| Metric | Value | Reference Range | Status |
|--------|-------|-----------------|--------|
| Annual Heating | 5.78 MWh | 5.0 - 15.0 MWh | ✅ PASS |
| Annual Cooling | 4.53 MWh | 1.0 - 3.5 MWh | ❌ FAIL |
| Peak Heating | 2.10 kW | 2.0 - 8.0 kW | ✅ PASS |
| Peak Cooling | 3.79 kW | 0.0 - 4.0 kW | ⚠️ BORDERLINE |

### Key Observations

1. **Heating passes** but cooling is ~30% above maximum allowed.
2. **Inter-zone temperature gradient is reversed**: Sunspace (Zone 1) averages 18.02°C, back-zone (Zone 0) averages 22.82°C. The sunspace is **colder** than the back-zone.
3. Expected behavior: Sunspace should be **warmer** than back-zone during summer (solar gains buffer), providing heat to back-zone and REDUCING cooling load.
4. Actual behavior: Cold sunspace draws heat from back-zone, **increasing** cooling load (back-zone HVAC must compensate for heat loss to sunspace).
5. Debug logs show `solar_gain_watts=0` for both zones in winter, suggesting potential solar calculation issues.

## Historical Context

### Issue #273 History

- **Original Problem**: HVAC was applied to all zones including sunspace (should be free-floating). This caused cooling ~23x too high.
- **Fix**: Made HVAC zone-specific with `hvac_enabled` flags. Sunspace now free-floating.
- **Side Effect**: After HVAC fix, cooling dropped from 64 MWh to 0.02 MWh (too low), then to current 4.53 MWh (still too high but in opposite direction).
- **Current Status**: HVAC logic correct, but inter-zone thermal dynamics still not matching reference.

### MULTI-01 in KNOWN_ISSUES.md

> "Inter-zone heat transfer via radiation, conduction, and stack effect appears to transfer excessive heat from sunspace to conditioned back-zone during cooling season."

This description is now outdated: the sunspace is actually too cold, meaning inter-zone heat flow is reversed (back to sunspace), which could also cause excessive cooling if the back-zone loses heat to the sunspace and must reheat/cool more.

## Dependencies

Phase 8 depends on:
- **Phase 7 completion**: Phase 7 (advanced analysis/visualization) must be fully complete. Specifically:
  - Plan 07-10 (MREF-03 remote reference tests)
  - Plan 07-11 (Sensitivity BatchOracle refactor)
- Both Phase 7 plans show SUMMARY files indicating completion, but code changes may need verification.

## Success Criteria

1. Case 960 annual cooling within 1.0-3.5 MWh (benchmark calibrated range)
2. No regression in Cases 600-950 after fix (pass rates maintained)
3. Root cause documented with before/after metrics
4. Full ASHRAE 140 validation passes (18/18 cases instantiate, target pass rate improves)

## Out of Scope for Phase 8

- Performance optimization (Phase 9)
- Test coverage expansion (Phase 10)
- API polish (Phase 11)
- 6R2C model exploration (Phase 12)
- Documentation updates beyond root cause analysis (Phase 13)

## Key Files

| File | Purpose |
|------|---------|
| `src/sim/engine.rs` | Core physics: `calculate_zone_solar_gain`, `step_physics_5r1c`, inter-zone heat transfer |
| `src/validation/ashrae_140_cases.rs` | Case 960 spec builder (`case_960_sunspace`) |
| `tests/ashrae_140_case_960_sunspace.rs` | Existing validation test |
| `tests/debug_960_summer.rs` | New diagnostic test (to be created) |
| `docs/CASE_960_ROOT_CAUSE.md` | Documentation of findings and fix (to be created) |
| `KNOWN_ISSUES.md` | Track issue status updates |

## References

- `CASE_960_ANALYSIS.md` - Original case specification and analysis
- `ISSUE_273_FINAL_SUMMARY.md` - Prior fix summary
- `KNOWN_ISSUES.md` - MULTI-01 entry
- `src/validation/benchmark.rs` - Reference ranges (lines 330-352)
- ASHRAE Standard 140-2023, Test Case 960: Sunspace
