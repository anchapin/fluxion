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
- `manual-baseline` — Manual baseline regeneration; not part of CI
- `pending-data` — Awaiting pending reference CSV / data delivery
- `other` — Other / unclassified (slow tests, env-required, etc.)

**Audit invariant (Issue #3443)**: every `#[ignore]` attribute under `tests/**/*.rs`
MUST have a corresponding row in this registry. The CI gate
(`scripts/generate_quarantine_registry.py --strict`) enforces this with a downward-only
ratchet mirroring the `BASELINE_KNOWN_ORPHANS` pattern from
`scripts/check_orphan_modules.py` (Issue #3459).

---

## Category: Diagnostic Tests (per #2536)

These tests are `#[ignore]`-quarantined because they have no assertions and are run
manually for investigation. They are NOT part of CI gates.

| Test File | Test Name | Blocking Issue | Un-Ignore Criteria | Status |
|-----------|-----------|----------------|-------------------|--------|
| `tests/diagnostics/diag_917_energy.rs` | `diag_energy_balance_600ff` | #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_917_solar.rs` | `diag_solar_gains_600ff` | #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_917_v2.rs` | `diagnostic` | #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_check.rs` | `check_temps` | #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_mass_traj.rs` | `diag_mass_trajectory` | #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_phim.rs` | `phi_m_diagnostic` | #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_solar_hr.rs` | `solar_diagnostic` | #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_solfields.rs` | `solar_fields` | #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/case_920_orientation_attribution.rs` | `test_case_920_per_orientation_solar_decomposition` | #2454, #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/case_940_setback_diagnostic.rs` | `test_case_940_setback_diagnostic` | #2452, #3062 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/case_940_setback_diagnostic.rs` | `test_case_940_setback_controller_mode_trace` | #2452, #3062 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/case_940_setback_diagnostic.rs` | `test_case_940_ctf_path_comparison` | #2452, #3062 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/case_940_setback_diagnostic.rs` | `test_case_940_blind_vs_ctf_ratio_pinned` | #2452, #3062 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/case_940_setback_diagnostic.rs` | `test_case_940_setback_recovery_window_diagnostic` | #2452, #3062 | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/case_195_weather_source_diagnostic.rs` | `test_case_195_weather_source_comparison` | #3060 (LIMIT-15) | Re-derive reference from E+ TMY3; add assertions | `pending` |
| `tests/ashrae_140_case_920.rs` | `test_case_920_per_month_attribution` | #2454, #2536 | Add assertions; convert to CI gate | `pending` |
| `tests/ashrae_140_case_920.rs` | `test_case_920_engine_vs_reference_per_month` | #2454, #2536 | Add assertions; convert to CI gate | `pending` |

---

## Category: Structural Gaps (LIMIT-*, KNOWN_ISSUES.md)

These tests are `#[ignore]` because they fail due to known physics/architecture gaps
documented in `docs/KNOWN_ISSUES.md`. They are tracked by LIMIT-* entries.

### LIMIT-05 / LIMIT-12 / LIMIT-14 / LIMIT-16 / LIMIT-17 / LIMIT-18 / LIMIT-19 / LIMIT-20

| Test File | Test Name | Blocking Issue(s) | Un-Ignore Criteria | Status |
|-----------|-----------|------------------|-------------------|--------|
| `tests/ashrae_140_case_920.rs` | `test_case_920_strict_annual_energy_within_band` | #2427, #2454, LIMIT-05 | GaugeSolver (#1465/#1462) ships and closes peak cooling gap | `pending` |
| `tests/ashrae_140_case_920.rs` | `test_case_920_per_month_attribution` | #2454, LIMIT-05 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/ashrae_140_case_920.rs` | `test_case_920_engine_vs_reference_per_month` | #2454, LIMIT-05 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/limit_05_inversion_regression.rs` | `test_limit_05_inversion_case_900_peak_cooling` | #1280, LIMIT-05 | GaugeSolver (#1465/#1462) ships; direction confirmed corrected | `pending` |
| `tests/limit_05_inversion_regression.rs` | `test_limit_05_inversion_case_950_peak_cooling` | #1280, LIMIT-05 | GaugeSolver (#1465/#1462) ships; direction confirmed corrected | `pending` |
| `tests/limit_05_inversion_regression.rs` | `test_limit_05_inversion_case_960_peak_cooling` | #1280, LIMIT-05 | GaugeSolver (#1465/#1462) ships; direction confirmed corrected | `pending` |
| `tests/limit_05_inversion_regression.rs` | `test_limit_05_inversion_summary` | #1280, LIMIT-05 | GaugeSolver (#1465/#1462) ships; direction confirmed corrected | `pending` |
| `tests/case_900_annual_energy_attribution.rs` | `test_issue_2448_case_910_shading_attribution` | #2448, LIMIT-05 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/case_900_series_seasonal_attribution.rs` | `test_case_900_series_seasonal_attribution` | #2453, LIMIT-05 | GaugeSolver (#1465/#1462) ships; bidirectional gap closed | `pending` |
| `tests/case_900_multinode_validation.rs` | `test_case_900_peak_cooling_*` | #1356, LIMIT-05 | CTF transient wall modeling lands; peak cooling in band | `pending` |
| `tests/zone_balance_eplus_isolation.rs` | `test_case_600_annual_energy_ashrae140_tolerance` | #2506, LIMIT-05 | GaugeSolver (#1465/#1462) ships; annual cooling in band | `pending` |
| `tests/zone_balance_eplus_isolation.rs` | `test_case_900_annual_energy_ashrae140_tolerance` | #2506, LIMIT-05 | GaugeSolver (#1465/#1462) ships; annual cooling in band | `pending` |
| `tests/known_issues_regression.rs` | `test_solar01_high_mass_peak_cooling_regression` | LIMIT-05 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/known_issues_regression.rs` | `test_solar02_high_mass_annual_cooling_regression` | #275, SOLAR-02 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/known_issues_regression.rs` | `test_solar03_shading_sensitivity_regression` | #276, SOLAR-03 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/known_issues_regression.rs` | `test_solar04_night_ventilation_regression` | #276, SOLAR-04 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/known_issues_regression.rs` | `test_free01_low_mass_max_temp_regression` | FREE-01 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/known_issues_regression.rs` | `test_free02_high_mass_min_temp_regression` | ADR-0003 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/known_issues_regression.rs` | `test_free03_temperature_swing_regression` | FREE-03 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/known_issues_regression.rs` | `test_limit05_high_mass_peak_cooling_model_limitation` | LIMIT-05 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/known_issues_regression.rs` | `test_issue532_case195_energy_regression` | #532 | Resolved or closed | `pending` |
| `tests/known_issues_regression.rs` | `test_issue533_600_series_peak_load_regression` | #533 | Resolved or closed | `pending` |
| `tests/issue_1860_5r1c_time_constant_aware.rs` | `test_case_600_annual_cooling_within_ashrae140_band` | #1860, LIMIT-05 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/issue_1860_5r1c_time_constant_aware.rs` | `test_case_600_annual_heating_within_ashrae140_band` | #1860, LIMIT-05 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/issue_1860_5r1c_time_constant_aware.rs` | `test_case_650_annual_cooling_within_ashrae140_band` | #1860, LIMIT-05 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/issue_1860_5r1c_time_constant_aware.rs` | `test_case_950_annual_cooling_within_ashrae140_band` | #1860, LIMIT-05 | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/invariant_checker_test.rs` | `test_one_watt_artificial_gain_increases_imbalance` | #3103, LIMIT-19 | EnergyBalanceValidator (#1344) investigation resolves algebraic invariant confusion | `pending` |
| `tests/validation/hvac_bestest/runner.rs` | `comparative_e200_cooling_vs_iea_task22_ensemble` | LIMIT-05, SOLAR-02 | GaugeSolver (#1465/#1462) ships; Case-600-class cooling closes | `pending` |
| `tests/ffd_cosimulation_validation.rs` | `test_peak_cooling_load_tolerance` | #2612, FFD-02 | Real coupled BES↔FFD solver ships; stub `BuoyancyDrivenFfdSolver` replaced | `pending` |

### Case 920 / 950 / 960 blind-mode cohort (Issue #1323 / #1213 / #3071 / #1422)

| Test File | Test Name | Blocking Issue(s) | Un-Ignore Criteria | Status |
|-----------|-----------|------------------|-------------------|--------|
| `tests/ashrae_140_blind_validation.rs` | `test_blind_mode_case_960_infrastructure` | LIMIT-18, #1465/#1462 | GaugeSolver structural 5R1C multi-lumped-mass lands; Case 960 blind heating closes | `pending` |
| `tests/ashrae_140_blind_validation.rs` | `test_blind_mode_case_920_annual_energy_within_band` | #1213, #1323, #1346 AC | Roof-solar / high-mass cooling physics fix (#1323) lands; Case 920 annual heating closes | `pending` |
| `tests/ashrae_140_blind_validation.rs` | `test_blind_mode_case_950_annual_energy_within_band` | #1323, #1347 AC2 | Roof-solar / high-mass cooling physics fix (#1323) lands; Case 950 strict band closes | `pending` |
| `tests/ashrae_140_blind_validation.rs` | `test_case_950_5r1c_free_float_uses_night_vent_overrides_issue_1422` | #3071, #1422, #1465/#1462 | GaugeSolver mass trajectory matches legacy night-flush pre-cool | `pending` |

### Ashrae 140 Case 900 / 920 paired-comparison cohort (Issue #2490 / LIMIT-05)

| Test File | Test Name | Blocking Issue(s) | Un-Ignore Criteria | Status |
|-----------|-----------|------------------|-------------------|--------|
| `tests/ashrae_140_case_900.rs` | `test_case_900_annual_cooling_within_reference_range` | LIMIT-05, #2490, #1465/#1462 | GaugeSolver ships; high-mass 9R4C over-damping closes | `pending` |
| `tests/ashrae_140_case_900.rs` | `test_case_900_peak_cooling_within_reference_range` | LIMIT-05, #2490, #1465/#1462 | GaugeSolver ships; instantaneous peak cooling closes | `pending` |
| `tests/ashrae_140_case_900.rs` | `test_case_900ff_max_temperature_within_reference_range` | LIMIT-05, #2490, #1465/#1462 | GaugeSolver ships; free-float max-temp closes | `pending` |
| `tests/ashrae_140_case_900.rs` | `test_case_900_annual_cooling_energy_with_correction` | LIMIT-05, #2490, #1465/#1462 | GaugeSolver ships | `pending` |
| `tests/ashrae_140_case_900.rs` | `test_case_600ff_vs_900ff_paired_comparison` | LIMIT-05, #2490, #1465/#1462 | GaugeSolver ships; 900FF paired-comparison closes | `pending` |
| `tests/ashrae_140_case_900.rs` | `test_900_series_regression` | Test-pollution (superceded by individual case tests) | Investigate; either un-ignore after pollution fix or delete | `pending` |
| `tests/ashrae_140_integration.rs` | `test_case_600_full_reference_tolerance` | #2683, SOLAR-02, LIMIT-05, #1465/#1462 | All four Case 600 metrics re-enter reference bands | `pending` |
| `tests/ashrae_140_integration.rs` | `test_case_610_shading` | #62 | Issue #62 merges; shading test wired into strict gate | `pending` |

### Case 195 / solid conduction cohort (Issue #3064 / LIMIT-20 / #3218)

| Test File | Test Name | Blocking Issue(s) | Un-Ignore Criteria | Status |
|-----------|-----------|------------------|-------------------|--------|
| `tests/ashrae_140_solid_conduction_variants.rs` | `test_case_195_high_mass_walls` | #3064, LIMIT-11, #1465/#1462 | GaugeSolver ships; zero-energy assertion closes | `pending` |
| `tests/ashrae_140_solid_conduction_variants.rs` | `test_solid_conduction_variants_integration` | LIMIT-20, #3218, LIMIT-11, #3064, #1465/#1462 | GaugeSolver ships; HighMass variant integration closes | `pending` |
| `tests/gauge_validation_case_900.rs` | `test_case_900_gauge_fiver1c_diurnal_parity` | #1669 | GaugeSolver thermal mass implementation lands (Option A) | `pending` |

### LIMIT-22 (gauge-build-only, `cfg_attr(feature = "gauge-solver", ignore)`, Issue #3297)

Feature-gated quarantines: these tests FAIL only under `--features gauge-solver`
(the exact Crank-Nicolson mass-state proxy of `fd7ef13` exposes that each was
passing for a physically-wrong reason). Default-build assertions are fully live
and pass. See KNOWN_ISSUES.md §LIMIT-22 for the root-cause analysis. (The
§LIMIT-21 pre-existing gauge air-trajectory cohort is deliberately NOT
quarantined — it is the #3286 β-soak gate signal.)

| Test File | Test Name | Blocking Issue(s) | Un-Ignore Criteria | Status |
|-----------|-----------|------------------|-------------------|--------|
| `tests/ashrae_140_blind_validation.rs` | `test_case_950_mass_temperature_precooled_issue_1422` | #3297, LIMIT-22 | Gauge air trajectory matches legacy night-flush pre-cool (τ_mass ≈ 61 h CN node swings +1.09 °C vs legacy +2.41 °C), or #1422 re-derives the band with maintainer sign-off | `pending` |
| `tests/ashrae_140_case_960_sunspace.rs` | `test_case_960_inter_zone_heat_transfer_analysis` | #3297, LIMIT-22 | Gauge multi-zone integration stability lands for Case 960-class configs (±140 °C step ΔT spikes around the fail-closed guard) | `pending` |
| `tests/ashrae_140_case_960_sunspace.rs` | `test_case_960_comprehensive_energy_validation` | #3392, LIMIT-22, #273 (inter-zone radiation / condensation 20× over reference) | Inter-zone radiation / condensation coupling that drives the 20× cooling gap is resolved (tracked under the GaugeSolver cohort #1465/#1462, #3297). When the underlying 5R1C/9R4C trajectory holds for the sunspace config, the comprehensive test can assert all 4 metrics without the documented cooling acceptance. | `pending` |
| `tests/invariant_checker_test.rs` | `test_different_zones_respond_differently_to_targeted_gain` | #3297, LIMIT-22 (§LIMIT-19/#3103 sibling) | §LIMIT-19 / #1344 investigation resolves the checker's zero-leverage artificial-gain formula (gain enters the 5R1C residual only through φm·m_air_frac) | `pending` |

### Phase A8 / Issue #3599 — 9R4C legacy solver scratch pool (Issue #3291 / §LIMIT-21)

These `#[ignore]`-quarantined unit tests live in `src/` (not `tests/`) and so
fall outside the `tests/**` audit-invariant of `tests/QUARANTINE.md` (Issue
#3443). They are tracked here for completeness because they block the Code
Coverage Gate (Issue #1932) on every PR off `develop` until the 9R4C legacy
dispatch path is removed post-Phase A8 (Issue #3291). GaugeSolver is the
unconditional default zone solver (`src/sim/thermal_selector.rs`), and the
`9R4C` scratch pool is the LEGACY solver's scratch buffer — once the legacy
dispatch is removed, these tests can be deleted.

| Test File | Test Name | Blocking Issue(s) | Un-Ignore Criteria | Status |
|-----------|-----------|------------------|-------------------|--------|
| `src/sim/thermal_model_physics/physics_impl/mod.rs` | `scratch_pool_9r4c_*` | #3599, #3291, LIMIT-21 | Legacy 9R4C dispatch removed; GaugeSolver-only path verified | `pending` |
| `src/sim/thermal_model_physics/physics_impl/mod.rs` | `scratch_pool_9r4c_*` (restored_on_free_float) | #3599, #3291, LIMIT-21 | Legacy 9R4C dispatch removed; GaugeSolver-only path verified | `pending` |
| `src/sim/thermal_model_physics/physics_impl/step_9r4c.rs` | `scratch_pool_9r4c_*` | #3599, #3291, LIMIT-21 | Legacy 9R4C dispatch removed; GaugeSolver-only path verified | `pending` |

> `src/` unit-test quarantines (like the Phase A8 rows above) fall outside the
> `tests/**` audit-invariant of this registry (Issue #3443) — the auditor scans
> `tests/**/*.rs` only, so exact-name rows for `src/` files are structural
> ghosts by construction. Wildcard names use the audit's documented wildcard
> convention (matched against the union, never counted as ghosts).

---

## Category: Performance / Memory Profiling (dhat tests)

These tests are `#[ignore]` because dhat backtrace capture makes them too slow for
unit-test CI. They are run manually for memory profiling.

| Test File | Test Name | Blocking Issue | Un-Ignore Criteria | Status |
|-----------|-----------|----------------|-------------------|--------|
| `tests/dhat_alloc_budget.rs` | `batch_oracle_hot_loop_alloc_budget` | Performance | CI profile budget defined; run in perf CI | `pending` |
| `tests/dhat_batched_surrogate_zero_growth.rs` | `predict_loads_batched_into_zero_steady_state_growth` | Performance | CI profile budget defined; run in perf CI | `pending` |
| `tests/dhat_batched_surrogate_zero_growth.rs` | `submit_with_sender_pingpong_steady_state_floor` | Performance | CI profile budget defined; run in perf CI | `pending` |
| `tests/dhat_evaluate_population_numpy_zero_copy.rs` | `evaluate_population_from_slice_zero_steady_state_growth` | Performance | CI profile budget defined; run in perf CI | `pending` |
| `tests/dhat_hybrid_zero_alloc.rs` | `hybrid_solve_timesteps_surrogate_load_branch_zero_steady_state_growth` | Performance | CI profile budget defined; run in perf CI | `pending` |
| `tests/dhat_step_physics_zero_alloc.rs` | `step_physics_day_mode_steady_state_alloc_budget` | Performance | CI profile budget defined; run in perf CI | `pending` |
| `tests/dhat_zone_solar_gain_zero_alloc.rs` | `zone_solar_gain_zero_steady_state_alloc` | Performance | CI profile budget defined; run in perf CI | `pending` |

### BDF / batch-oracle benchmarks (slow, manual)

| Test File | Test Name | Blocking Issue | Un-Ignore Criteria | Status |
|-----------|-----------|----------------|-------------------|--------|
| `tests/bdf_solver_tests.rs` | `benchmark_bdf_stiff_network_100` | Performance (manual benchmark) | Run in perf CI under `--release` with `--nocapture` | `pending` |
| `tests/bdf_solver_tests.rs` | `benchmark_bdf_stiff_network_100_throughput` | Performance (manual benchmark) | Run in perf CI under `--release` with `--nocapture` | `pending` |
| `tests/lib_batch_oracle.rs` | `test_batch_oracle_*` (5 tests) | Slow (full-year simulation) | Integration CI profile; run on perf runner | `pending` |

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
| `tests/solar_peak_cooling_tdd.rs` | `test_case_600_peak_cooling_red` | Calibration | Expected values verified against ASHRAE 140 reference | `pending` |
| `tests/solar_peak_cooling_tdd.rs` | `test_case_900_peak_cooling_red` | Calibration | Expected values verified against ASHRAE 140 reference | `pending` |
| `tests/thermal_comfort_prediction_validation.rs` | `test_eplus_thermal_comfort_reference_pending` | Data | EnergyPlus thermal comfort benchmark data available | `pending` |
| `tests/thermal_comfort_prediction_validation.rs` | `test_eplus_thermal_comfort_reference_pending` | Data | EnergyPlus thermal comfort benchmark data available | `pending` |
| `tests/test_statistical_validation.rs` | `test_cli_statistical_flag` | Environment | Compiled `fluxion` binary at `target/release/fluxion` | `pending` |
| `tests/surface_flux_parity.rs` | `test_parity_roof_zero_followup_1323` | #1323 | Post-#1323 roof-solar physics fix lands | `pending` |
| `tests/gauge_validation_case_900.rs` | `test_case_900_gauge_fiver1c_diurnal_parity` | #1669 | GaugeSolver thermal mass implementation | `pending` |

### Pending reference CSVs (Issue #1331 / #1168 / #1166)

| Test File | Test Name | Blocking Issue | Un-Ignore Criteria | Status |
|-----------|-----------|----------------|-------------------|--------|
| `tests/ashrae_140_blind_validation.rs` | `test_blind_mode_case_800_annual_energy_within_band` | #1331, #1168 | `case_800_energy_reference.csv` regenerated from EnergyPlus | `pending` |
| `tests/ashrae_140_blind_validation.rs` | `test_blind_mode_case_810_annual_energy_within_band` | #1331, #1168 | `case_810_energy_reference.csv` regenerated from EnergyPlus | `pending` |
| `tests/ashrae_140_blind_validation.rs` | `test_blind_mode_case_960_annual_energy_within_band` | #1331, #1168 | `case_960_energy_reference.csv` regenerated from EnergyPlus | `pending` |

---

## Category: CI Infrastructure

These tests are `#[ignore]` because CI is broken, not because the test logic is wrong.

| Test File | Test Name | Blocking Issue | Un-Ignore Criteria | Status |
|-----------|-----------|----------------|-------------------|--------|
| `tests/idf_ashrae_140_acceptance.rs` | `idf_case_600_annual_heating_within_15_percent_strict` | #1577 | CI fixed; develop CI can run tests to verify | `pending` |

---

## Category: Manual Baseline Regeneration

These tests are `#[ignore]` because they regenerate baselines and should only be run
manually after legitimate changes.

| Test File | Test Name | Blocking Issue | Un-Ignore Criteria | Status |
|-----------|-----------|----------------|-------------------|--------|
| `tests/surrogate_drift_fallback_regression.rs` | `fallback_annual_hvac_diagnostic` | Manual | Run manually after surrogate change; not in CI | `pending` |
| `tests/surrogate_cold_start_test.rs` | `diagnostic_print_cold_warm_cycles` | Manual | Run manually after ort version bump; not in CI | `pending` |

---

## Category: Other / Unclassified

| Test File | Test Name | Blocking Issue | Un-Ignore Criteria | Status |
|-----------|-----------|----------------|-------------------|--------|
| `tests/lib_batch_oracle.rs` | `test_evaluate_population_u_value_impact` | Slow (full-year simulation) | Integration CI profile | `pending` |
| `tests/lib_batch_oracle.rs` | `test_evaluate_population_setpoint_impact` | Slow (full-year simulation) | Integration CI profile | `pending` |
| `tests/lib_batch_oracle.rs` | `test_evaluate_population_with_large_population` | Slow (full-year simulation) | Integration CI profile | `pending` |
| `tests/lib_batch_oracle.rs` | `test_evaluate_population_with_surrogates_no_model` | Slow (hangs when surrogates=true without model loaded) | Investigate; either fix or delete | `pending` |
| `tests/lib_batch_oracle.rs` | `test_evaluate_population_parallel_execution` | Slow (full-year simulation) | Integration CI profile | `pending` |
| `tests/bdf_solver_tests.rs` | `test_bdf_*` (2 tests) | Unknown | Investigate; determine un-ignore criteria | `pending` |
| `tests/weather_vs_energyplus.rs` | `test_humidity_ratio_psychrometrics_vs_energyplus` | #2673 | Formula generator embedded; issue #2673 resolves | `pending` |
| `tests/weather_vs_energyplus.rs` | `test_synthetic_miami_tmy_matches_reference` | #2673 | Formula generator embedded; issue #2673 resolves | `pending` |
| `tests/energyplus_comparison_tests.rs` | `test_900_series_comprehensive_comparison` | Long-running | Run explicitly when needed; not in CI | `pending` |
| `tests/energyplus_comparison_tests.rs` | `test_900_series_comprehensive_comparison` | Long-running | Run explicitly when needed; not in CI | `pending` |

## Category: Strict-Energy-Gate & Structural Diagnostics (Issue #3443 reconciliation)

> Reconciliation section: these 10 `#[ignore]` attributes landed on `develop`
> (via #3572 / #3585 Wave-5/8 and the #3551/#3552 diagnostic placeholders)
> without registry rows, tripping the Issue #3443 downward-only ratchet. This
> section back-fills the missing rows and raises `BASELINE_ORPHANED_IGNORES`
> to 10 with the freeze-set entries in `scripts/generate_quarantine_registry.py`.

| Test File | Test Name | Blocking Issue | Un-Ignore Criteria | Status |
|-----------|-----------|----------------|-------------------|--------|
| `tests/zone_balance_eplus_isolation.rs` | `test_case_800_annual_energy_ashrae140_tolerance` | #3572 | Engine re-enters ±15% ASHRAE 140-2023 Annex B band; lower in-file baseline | `pending` |
| `tests/zone_balance_eplus_isolation.rs` | `test_case_810_annual_energy_ashrae140_tolerance` | #3572 | Engine re-enters ±15% band; lower in-file baseline | `pending` |
| `tests/zone_balance_eplus_isolation.rs` | `test_case_920_annual_energy_ashrae140_tolerance` | #3572 | Engine re-enters ±15% band; lower in-file baseline | `pending` |
| `tests/zone_balance_eplus_isolation.rs` | `test_case_950_annual_energy_ashrae140_tolerance` | #3572 | Engine re-enters ±15% band; lower in-file baseline | `pending` |
| `tests/zone_balance_eplus_isolation.rs` | `test_case_960_annual_energy_ashrae140_tolerance` | #3572 | Engine re-enters ±15% band; lower in-file baseline | `pending` |
| `tests/zone_balance_eplus_isolation.rs` | `test_case_970_annual_energy_ashrae140_tolerance` | #3572 | Engine re-enters ±15% band; lower in-file baseline | `pending` |
| `tests/ashrae_140_case_970_validation.rs` | `test_case_970_annual_energy_band` | #3585 / §LIMIT-23 (#3552) | Multi-zone air-mass distribution gap closed (GaugeSolver #1465/#1462 rework) | `pending` |
| `tests/ashrae_140_case_970_validation.rs` | `test_case_970_validator_accepts_canonical_midpoints` | #3585 / §LIMIT-23 (#3552) | Multi-zone air-mass distribution gap closed (GaugeSolver #1465/#1462 rework) | `pending` |
| `tests/diagnostics/case_950_hvac_mode_seasonal_attribution.rs` | `test_case_950_hvac_mode_seasonal_attribution` | #3551 / §LIMIT-24 | Implement per-month attribution walk (follow-up PR, GaugeSolver #1465/#1462) | `pending` |
| `tests/diagnostics/case_970_multi_zone_seasonal_attribution.rs` | `case_970_per_zone_seasonal_attribution_placeholder` | #3552 / §LIMIT-23 | Implement per-month per-zone attribution | `pending` |
| `fluxion-wasm/tests/wasm_integration_tests.rs` | `wasm_run_full_annual_*` | #3703 (wasm step() toy model; #3595 smoke test) | wasm `FluidSimulation::step()` wired to the real engine (or test re-pointed at an engine-backed surface); then restore the published ±15% band assertion | `pending` |

---

## Summary

| Category | Count | Status |
|----------|-------|--------|
| Diagnostic tests (#2536) | 13 | `pending` |
| Structural gaps (LIMIT-*) | ~49 (3 gauge-build-only, Issue #3297; 3 9R4C legacy pool, Issue #3599) | `pending` |
| Performance/memory (dhat + BDF + batch) | 17 | `pending` |
| Hardware-dependent (GPU) | 1 | `pending` |
| Calibration/pending data | 8 | `pending` |
| Pending reference CSVs (#1331/#1168) | 3 | `pending` |
| CI infrastructure | 2 | `pending` |
| Manual baseline regen | 4 | `pending` |
| Other/unclassified | 12 | `pending` |
| Strict-energy-gate & structural diagnostics (Issue #3443 reconciliation) | 10 | `pending` |
| **Total** | **~119** | |

(82 orphan entries were triaged into this registry by Issue #3443; the totals
above include the 23 pre-existing entries and the 82 newly-added ones. The 3
gauge-build-only `cfg_attr(...)` ignores live in the structural-cohort section
above; the audit scanner counts only unconditional `#[ignore]` attributes.)

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

*Generated by `scripts/generate_quarantine_registry.py` (Issue #3211, #3393, #3443)*
*Last Updated: 2026-09-11 (Issue #3443 reconciliation: back-filled 10 missing rows for the #3572/#3585/#3551/#3552 ignores; wildcarded the 3 src/ 9R4C scratch_pool rows per the audit's wildcard convention)*
