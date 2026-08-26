# Test Quarantine Registry

Machine-readable registry of all `#[ignore]`-quarantined tests, mapped to their blocking
issues, un-ignore criteria, and current status.

**Purpose**: Provide a single source of truth for tracking when quarantined tests can be
un-ignored. Without this registry, tests accumulate in quarantine indefinitely and actual
testing coverage is opaque (Issue #3211).

**Protocol**: When the un-ignore criteria for a test are met, the test owner (listed in
the `Owner` column) removes the `#[ignore]` attribute and updates the `Status` to
`closed`. The `Closed By` column records the PR that un-ignores the test.

**Categories**:
- `diagnostic` — No assertions; run manually with `--ignored --nocapture`
- `structural` — BLOCKED by a known physics/architecture gap (LIMIT-* in KNOWN_ISSUES.md)
- `performance` — Memory/performance profiling; too slow for unit-test CI
- `hardware` — Requires special hardware (GPU) to run
- `calibration` — Awaiting external data or calibration verification
- `ci-broken` — CI infrastructure issue; test itself may be valid

---

## Category: Diagnostic Tests (per #2536)

These tests are `#[ignore]`-quarantined because they have no assertions and are run
manually for investigation. They are NOT part of CI gates.

| Test File | Test Name | Blocking Issue | Un-Ignore Criteria | Status |
|-----------|-----------|----------------|-------------------|--------|
| `tests/diagnostics/diag_917_energy.rs` | `diag_917_energy` | #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_917_solar.rs` | `diag_917_solar` | #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_917_v2.rs` | `diag_917_v2` | #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_check.rs` | `diag_check` | #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_mass_traj.rs` | `diag_mass_traj` | #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_phim.rs` | `diag_phim` | #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_solar_hr.rs` | `diag_solar_hr` | #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_solfields.rs` | `diag_solfields` | #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/case_920_orientation_attribution.rs` | `case_920_orientation_attribution` | #2454, #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/case_940_setback_diagnostic.rs` | `case_940_setback_*` (4 tests) | #2452, #3062 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/case_195_weather_source_diagnostic.rs` | `case_195_weather_source_*` | #3060 (LIMIT-15) | Re-derive reference from E+ TMY3; add assertions | `pending` |

---

## Category: Structural Gaps (LIMIT-*, KNOWN_ISSUES.md)

These tests are `#[ignore]` because they fail due to known physics/architecture gaps
documented in `docs/KNOWN_ISSUES.md`. They are tracked by LIMIT-* entries.

### LIMIT-05 / LIMIT-12 / LIMIT-14 / LIMIT-16 / LIMIT-17 / LIMIT-18 / LIMIT-19 / LIMIT-20

| Test File | Test Name | Blocking Issue(s) | Un-Ignore Criteria | Status |
|-----------|-----------|------------------|-------------------|--------|
| `tests/ashrae_140_case_920.rs` | `test_ashrae_140_case_920_strict_band` | #2427, #2454, LIMIT-05 | GaugeSolver (#1465/#1462) ships and closes peak cooling gap | `pending` |
| `tests/ashrae_140_case_920.rs` | `test_case_920_per_month_attribution` | #2454, LIMIT-05 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/ashrae_140_case_920.rs` | `test_case_920_reference_vs_engine_comparison` | #2454, LIMIT-05 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/limit_05_inversion_regression.rs` | `test_limit_05_*` (4 tests) | #1280, LIMIT-05 | GaugeSolver (#1465/#1462) ships; direction confirmed corrected | `pending` |
| `tests/case_900_annual_energy_attribution.rs` | `test_case_900_series_strict_cooling_bands` | #2448, LIMIT-05 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/case_900_series_seasonal_attribution.rs` | `test_case_900_series_seasonal_attribution` | #2453, LIMIT-05 | GaugeSolver (#1465/#1462) ships; bidirectional gap closed | `pending` |
| `tests/case_900_multinode_validation.rs` | `test_case_900_peak_cooling_*` | #1356, LIMIT-05 | CTF transient wall modeling lands; peak cooling in band | `pending` |
| `tests/zone_balance_eplus_isolation.rs` | `test_case_600_annual_cooling_within_ashrae140_band` | #2506, LIMIT-05 | GaugeSolver (#1465/#1462) ships; annual cooling in band | `pending` |
| `tests/zone_balance_eplus_isolation.rs` | `test_case_900_annual_cooling_within_ashrae140_band` | #2506, LIMIT-05 | GaugeSolver (#1465/#1462) ships; annual cooling in band | `pending` |
| `tests/known_issues_regression.rs` | `test_limit_05_*` (multiple) | LIMIT-05 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/known_issues_regression.rs` | `test_solar_02_*` | #275, SOLAR-02 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/known_issues_regression.rs` | `test_solar_03_*` | #276, SOLAR-03 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/known_issues_regression.rs` | `test_solar_04_*` | #276, SOLAR-04 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/known_issues_regression.rs` | `test_free_01_*` | FREE-01 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/known_issues_regression.rs` | `test_free_03_*` | FREE-03 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/known_issues_regression.rs` | `test_adr_0003_*` | ADR-0003 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/known_issues_regression.rs` | `test_issue_532_*` | #532 | Resolved or closed | `pending` |
| `tests/known_issues_regression.rs` | `test_issue_533_*` | #533 | Resolved or closed | `pending` |
| `tests/issue_1860_5r1c_time_constant_aware.rs` | `test_issue_1860_*` (4 tests) | #1860, LIMIT-05 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/invariant_checker_test.rs` | `test_one_watt_artificial_gain_increases_imbalance` | #3103, LIMIT-19 | EnergyBalanceValidator (#1344) investigation resolves algebraic invariant confusion | `pending` |
| `tests/validation/hvac_bestest/runner.rs` | `test_hvac_bestest_case_*_comparative` | LIMIT-05, SOLAR-02 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/ffd_cosimulation_validation.rs` | `ffd_cosimulation_validation` | #2612, FFD-02 | Real coupled BES↔FFD solver ships | `pending` |

---

## Category: Performance / Memory Profiling (dhat tests)

These tests are `#[ignore]` because dhat backtrace capture makes them too slow for
unit-test CI. They are run manually for memory profiling.

| Test File | Test Name | Blocking Issue | Un-Ignore Criteria | Status |
|-----------|-----------|----------------|-------------------|--------|
| `tests/dhat_alloc_budget.rs` | `test_dhat_*` | Performance | CI profile budget defined; run in perf CI | `pending` |
| `tests/dhat_batched_surrogate_zero_growth.rs` | `test_dhat_*` (2 tests) | Performance | CI profile budget defined; run in perf CI | `pending` |
| `tests/dhat_evaluate_population_numpy_zero_copy.rs` | `test_dhat_*` | Performance | CI profile budget defined; run in perf CI | `pending` |
| `tests/dhat_hybrid_zero_alloc.rs` | `test_dhat_*` | Performance | CI profile budget defined; run in perf CI | `pending` |
| `tests/dhat_step_physics_zero_alloc.rs` | `test_dhat_*` | Performance | CI profile budget defined; run in perf CI | `pending` |
| `tests/dhat_zone_solar_gain_zero_alloc.rs` | `test_dhat_*` | Performance | CI profile budget defined; run in perf CI | `pending` |

---

## Category: Hardware-Dependent Tests

These tests require special hardware and are `#[ignore]` on machines without that hardware.

| Test File | Test Name | Blocking Issue | Un-Ignore Criteria | Status |
|-----------|-----------|----------------|-------------------|--------|
| `tests/surrogate_backend_parity.rs` | `test_cpu_vs_cuda_parity` | Hardware (GPU) | Run on GPU hardware-in-loop CI with `--include-ignored` | `pending` |

---

## Category: Calibration / Pending Data

These tests are `#[ignore]` because they await external calibration data or verification.

| Test File | Test Name | Blocking Issue | Un-Ignore Criteria | Status |
|-----------|-----------|----------------|-------------------|--------|
| `tests/solar_peak_cooling_tdd.rs` | `test_solar_peak_cooling_*` (2 tests) | Calibration | Expected values verified against ASHRAE 140 reference | `pending` |
| `tests/thermal_comfort_prediction_validation.rs` | `test_thermal_comfort_*` | Data | EnergyPlus thermal comfort benchmark data available | `pending` |
| `tests/test_statistical_validation.rs` | `test_statistical_*` | Environment | Compiled `fluxion` binary at `target/release/fluxion` | `pending` |
| `tests/surface_flux_parity.rs` | `test_surface_flux_parity_post_1323` | #1323 | Post-#1323 roof-solar physics fix lands | `pending` |
| `tests/gauge_validation_case_900.rs` | `test_gauge_solver_steady_state_option_a` | #1669 | GaugeSolver thermal mass implementation | `pending` |

---

## Category: CI Infrastructure

These tests are `#[ignore]` because CI is broken, not because the test logic is wrong.

| Test File | Test Name | Blocking Issue | Un-Ignore Criteria | Status |
|-----------|-----------|----------------|-------------------|--------|
| `tests/idf_ashrae_140_acceptance.rs` | `test_idf_case_600_acceptance` | #1577 | CI fixed; develop CI can run tests to verify | `pending` |

---

## Category: Manual Baseline Regeneration

These tests are `#[ignore]` because they regenerate baselines and should only be run
manually after legitimate changes.

| Test File | Test Name | Blocking Issue | Un-Ignore Criteria | Status |
|-----------|-----------|----------------|-------------------|--------|
| `tests/surrogate_drift_fallback_regression.rs` | `test_surrogate_drift_fallback_baseline_regeneration` | Manual | Run manually after surrogate change; not in CI | `pending` |
| `tests/surrogate_cold_start_test.rs` | `test_surrogate_cold_start_baseline_regeneration` | Manual | Run manually after ort version bump; not in CI | `pending` |

---

## Category: Other / Unclassified

| Test File | Test Name | Blocking Issue | Un-Ignore Criteria | Status |
|-----------|-----------|----------------|-------------------|--------|
| `tests/lib_batch_oracle.rs` | `test_batch_oracle_*` (5 tests) | Slow | Full-year simulation; run in integration CI | `pending` |
| `tests/bdf_solver_tests.rs` | `test_bdf_*` (2 tests) | Unknown | Investigate; determine un-ignore criteria | `pending` |
| `tests/weather_vs_energyplus.rs` | `test_derived_humidity_ratio_parity` | #2673 | Formula generator embedded; issue #2673 resolves | `pending` |
| `tests/weather_vs_energyplus.rs` | `test_wet_bulb_vs_dry_bulb_scaling` | #2673 | Formula generator embedded; issue #2673 resolves | `pending` |
| `tests/energyplus_comparison_tests.rs` | `test_energyplus_*` | Long-running | Run explicitly when needed; not in CI | `pending` |

---

## Summary

| Category | Count | Status |
|----------|-------|--------|
| Diagnostic tests (#2536) | 12 | `pending` |
| Structural gaps (LIMIT-*) | ~40 | `pending` |
| Performance/memory (dhat) | 9 | `pending` |
| Hardware-dependent (GPU) | 1 | `pending` |
| Calibration/pending data | 5 | `pending` |
| CI infrastructure | 1 | `pending` |
| Manual baseline regen | 2 | `pending` |
| Other/unclassified | 8 | `pending` |
| **Total** | **~78** | |

---

## Un-Ignore Checklist

When a blocking issue is resolved, the test owner should:

1. Remove the `#[ignore]` attribute from the test
2. Verify the test passes on CI
3. Update this registry:
   - Set `Status` to `closed`
   - Set `Closed By` to the PR number that un-ignores the test
4. If the test was moved from `diagnostic` to CI gate, update the `Un-Ignore Criteria` to "CI gate"

---

*Generated by `scripts/generate_quarantine_registry.py` (Issue #3211)*
*Last Updated: 2026-08-26*
