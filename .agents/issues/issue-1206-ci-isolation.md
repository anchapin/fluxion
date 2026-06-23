## Issue Description

The CI validation gate only runs `ashrae_140_validation`. **None of the 5 module isolation tests are in the CI gate**:

| Module | Isolation Test | In CI? |
|--------|---------------|--------|
| Weather | `weather_isolation.rs` | No |
| Solar | `solar_isolation.rs` | No |
| Conduction | `conduction_5r1c_isolation.rs` | No |
| Ventilation | `ventilation_infiltration_vs_energyplus.rs` | No |
| Zone Balance | `zone_balance_eplus_isolation.rs` | No |

## Impact

- PRs can pass CI with broken module isolation tests
- Phase 1 completion (module isolation) cannot be verified via CI
- False sense of validation — ASHRAE 140 tests can pass even when modules are broken

## Fix

Add isolation tests to CI workflow test matrix.

## Files Affected

- `.github/workflows/ci.yml`
- `.github/workflows/ashrae_validation.yml`

## Acceptance Criteria

- [ ] All 5 module isolation tests run in CI on every PR
- [ ] Isolation test failures block merge
- [ ] Test results visible in PR status checks