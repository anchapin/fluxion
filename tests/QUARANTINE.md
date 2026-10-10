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
- `calibration` — Awaiting external data or calibration verification (includes pending reference CSVs)
- `ci-broken` — CI infrastructure issue; test itself may be valid
- `manual-baseline` — Manual baseline regeneration; not part of CI
- `other` — Other / unclassified (slow tests, env-required, etc.)

**Audit invariant (Issues #3443 / #4178)**: every `#[ignore]` attribute under
`tests/**/*.rs`, `src/**/*.rs`, and every workspace member's `src/**/*.rs` /
`tests/**/*.rs` MUST have a corresponding row in this registry. The CI gate
(`scripts/generate_quarantine_registry.py --strict`) enforces this with a
key-membership ratchet (Issue #4179): any orphan `#[ignore]` absent from the
freeze snapshot fails the gate, regardless of the total count — adding a
registry row is the only way back to green. The same gate enforces the row
schema (Issue #4179): every row must carry non-empty `Category`,
`Blocking Issue`, `Owner`, `Un-Ignore Criteria` and `Status` cells, and
`Category` must be one of the values listed below.

---

## Category: Diagnostic Tests (per #2536)

These tests are `#[ignore]`-quarantined because they have no assertions and are run
manually for investigation. They are NOT part of CI gates.

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/all_tests/fd_time_integration_accuracy.rs` | `fd_time_integration_sweep_harness` | `diagnostic` | #3980 | `unassigned` | Prints the accuracy-vs-cost table for docs/validation/conduction_time_integration.md; convert to a CI gate only if a sweep-drift gate is ever warranted | `pending` |
| `tests/diagnostics/diag_917_energy.rs` | `diag_energy_balance_600ff` | `diagnostic` | #2536 | `unassigned` | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_917_solar.rs` | `diag_solar_gains_600ff` | `diagnostic` | #2536 | `unassigned` | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_917_v2.rs` | `diagnostic` | `diagnostic` | #2536 | `unassigned` | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_check.rs` | `check_temps` | `diagnostic` | #2536 | `unassigned` | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_mass_traj.rs` | `diag_mass_trajectory` | `diagnostic` | #2536 | `unassigned` | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_phim.rs` | `phi_m_diagnostic` | `diagnostic` | #2536 | `unassigned` | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_solar_hr.rs` | `solar_diagnostic` | `diagnostic` | #2536 | `unassigned` | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_solfields.rs` | `solar_fields` | `diagnostic` | #2536 | `unassigned` | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/case_920_orientation_attribution.rs` | `test_case_920_per_orientation_solar_decomposition` | `diagnostic` | #2454, #2536 | `unassigned` | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/case_940_setback_diagnostic.rs` | `test_case_940_setback_diagnostic` | `diagnostic` | #2452, #3062 | `unassigned` | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/case_940_setback_diagnostic.rs` | `test_case_940_setback_controller_mode_trace` | `diagnostic` | #2452, #3062 | `unassigned` | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/case_940_setback_diagnostic.rs` | `test_case_940_ctf_path_comparison` | `diagnostic` | #2452, #3062 | `unassigned` | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/case_940_setback_diagnostic.rs` | `test_case_940_blind_vs_ctf_ratio_pinned` | `diagnostic` | #2452, #3062 | `unassigned` | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/case_940_setback_diagnostic.rs` | `test_case_940_setback_recovery_window_diagnostic` | `diagnostic` | #2452, #3062 | `unassigned` | Add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/case_195_weather_source_diagnostic.rs` | `test_case_195_weather_source_comparison` | `diagnostic` | #3060 (LIMIT-15) | `unassigned` | Re-derive reference from E+ TMY3; add assertions | `pending` |
| `tests/diagnostics/case_950_hvac_mode_seasonal_attribution.rs` | `test_case_950_hvac_mode_seasonal_attribution` | `diagnostic` | #3551, #2536 | `unassigned` | Implement diagnostic per §LIMIT-24 table; add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/case_970_multi_zone_seasonal_attribution.rs` | `case_970_per_zone_seasonal_attribution_placeholder` | `diagnostic` | #3552, #2536 | `unassigned` | Implement per-month per-zone attribution; add assertions; convert to CI gate | `pending` |
| `tests/diagnostics/diag_air_node_equilibration.rs` | `diag_air_node_equilibration` | `diagnostic` | #2536 | `unassigned` | Add assertions; convert to CI gate | `pending` |
| `tests/all_tests/diag_air_node_equilibration.rs` | `diag_air_node_equilibration` | `diagnostic` | #2536 | `unassigned` | Consolidated runner re-export of `tests/diagnostics/diag_air_node_equilibration.rs::diag_air_node_equilibration`; tracked at the canonical source row above | `pending` |
| `tests/all_tests/ashrae_140_case_920.rs` | `test_case_920_per_month_attribution` | `diagnostic` | #2454, #2536 | `unassigned` | Add assertions; convert to CI gate | `pending` |
| `tests/all_tests/ashrae_140_case_920.rs` | `test_case_920_engine_vs_reference_per_month` | `diagnostic` | #2454, #2536 | `unassigned` | Add assertions; convert to CI gate | `pending` |
| `fluxion-core/tests/schedule_conformance.rs` | `diagnostic_case_640_ramp_profile` | `diagnostic` | #4196 | `unassigned` | Prints the Case 640 ramp profile (hours 6-10) for investigation; convert to a CI gate only if ramp-characterization drift gating is warranted | `pending` |
| `fluxion-core/tests/schedule_conformance.rs` | `diagnostic_case_940_ramp_profile` | `diagnostic` | #4196 | `unassigned` | Prints the Case 940 ramp profile (hours 6-10) for investigation; convert to a CI gate only if ramp-characterization drift gating is warranted | `pending` |

---

## Category: Structural Gaps (LIMIT-*, KNOWN_ISSUES.md)

These tests are `#[ignore]` because they fail due to known physics/architecture gaps
documented in `docs/KNOWN_ISSUES.md`. They are tracked by LIMIT-* entries.

### Documented deviation: 2-hour setback ramp (Issues #4196 / #4226)

These tests fail against the discrete ASHRAE 140 setpoint profile because the
implemented boundary condition carries a documented 2-hour linear setback ramp
(see `docs/investigations/issue-4196-setback-ramp-deviation.md` for the
committed failure output). They are ignored pending the keep/remove decision
in #4226 — not a LIMIT-* gap, a recorded specification deviation.

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `fluxion-core/tests/schedule_conformance.rs` | `conformance_case_640_integer_hours` | `structural` | #4196, #4226 | `unassigned` | Ramp keep/remove decision (#4226) lands; un-ignore after the decision is recorded | `pending` |
| `fluxion-core/tests/schedule_conformance.rs` | `conformance_case_640_sub_hour` | `structural` | #4196, #4226 | `unassigned` | Ramp keep/remove decision (#4226) lands; un-ignore after the decision is recorded | `pending` |
| `fluxion-core/tests/schedule_conformance.rs` | `conformance_case_940_integer_hours` | `structural` | #4196, #4226 | `unassigned` | Ramp keep/remove decision (#4226) lands; un-ignore after the decision is recorded | `pending` |
| `fluxion-core/tests/schedule_conformance.rs` | `conformance_case_940_sub_hour` | `structural` | #4196, #4226 | `unassigned` | Ramp keep/remove decision (#4226) lands; un-ignore after the decision is recorded | `pending` |

### LIMIT-05 / LIMIT-12 / LIMIT-14 / LIMIT-16 / LIMIT-17 / LIMIT-18 / LIMIT-19 / LIMIT-20

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/all_tests/ashrae_140_case_920.rs` | `test_case_920_strict_annual_energy_within_band` | `structural` | #2427, #2454, LIMIT-05 | `unassigned` | GaugeSolver (#1465/#1462) ships and closes peak cooling gap | `pending` |
| `tests/all_tests/ashrae_140_case_920.rs` | `test_case_920_per_month_attribution` | `structural` | #2454, LIMIT-05 | `unassigned` | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/all_tests/ashrae_140_case_920.rs` | `test_case_920_engine_vs_reference_per_month` | `structural` | #2454, LIMIT-05 | `unassigned` | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/all_tests/limit_05_inversion_regression.rs` | `test_limit_05_inversion_case_900_peak_cooling` | `structural` | #1280, LIMIT-05 | `unassigned` | GaugeSolver (#1465/#1462) ships; direction confirmed corrected | `pending` |
| `tests/all_tests/limit_05_inversion_regression.rs` | `test_limit_05_inversion_case_950_peak_cooling` | `structural` | #1280, LIMIT-05 | `unassigned` | GaugeSolver (#1465/#1462) ships; direction confirmed corrected | `pending` |
| `tests/all_tests/limit_05_inversion_regression.rs` | `test_limit_05_inversion_case_960_peak_cooling` | `structural` | #1280, LIMIT-05 | `unassigned` | GaugeSolver (#1465/#1462) ships; direction confirmed corrected | `pending` |
| `tests/all_tests/limit_05_inversion_regression.rs` | `test_limit_05_inversion_summary` | `structural` | #1280, LIMIT-05 | `unassigned` | GaugeSolver (#1465/#1462) ships; direction confirmed corrected | `pending` |
| `tests/all_tests/case_900_annual_energy_attribution.rs` | `test_issue_2448_case_910_shading_attribution` | `structural` | #2448, LIMIT-05 | `unassigned` | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/all_tests/case_900_series_seasonal_attribution.rs` | `test_case_900_series_seasonal_attribution` | `structural` | #2453, LIMIT-05 | `unassigned` | GaugeSolver (#1465/#1462) ships; bidirectional gap closed | `pending` |
| `tests/all_tests/case_900_multinode_validation.rs` | `test_case_900_peak_cooling_*` | `structural` | #1356, LIMIT-05 | `unassigned` | CTF transient wall modeling lands; peak cooling in band | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_600_annual_energy_ashrae140_tolerance` | `structural` | #2506, LIMIT-05 | `unassigned` | GaugeSolver (#1465/#1462) ships; annual cooling in band | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_900_annual_energy_ashrae140_tolerance` | `structural` | #2506, LIMIT-05 | `unassigned` | GaugeSolver (#1465/#1462) ships; annual cooling in band | `pending` |
| `tests/all_tests/known_issues_regression.rs` | `test_solar01_high_mass_peak_cooling_regression` | `structural` | LIMIT-05 | `unassigned` | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/all_tests/known_issues_regression.rs` | `test_solar02_high_mass_annual_cooling_regression` | `structural` | #275, SOLAR-02 | `unassigned` | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/all_tests/known_issues_regression.rs` | `test_solar03_shading_sensitivity_regression` | `structural` | #276, SOLAR-03 | `unassigned` | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/all_tests/known_issues_regression.rs` | `test_solar04_night_ventilation_regression` | `structural` | #276, SOLAR-04 | `unassigned` | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/all_tests/known_issues_regression.rs` | `test_free01_low_mass_max_temp_regression` | `structural` | FREE-01 | `unassigned` | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/all_tests/known_issues_regression.rs` | `test_free02_high_mass_min_temp_regression` | `structural` | ADR-0003 | `unassigned` | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/all_tests/known_issues_regression.rs` | `test_free03_temperature_swing_regression` | `structural` | FREE-03 | `unassigned` | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/all_tests/known_issues_regression.rs` | `test_limit05_high_mass_peak_cooling_model_limitation` | `structural` | LIMIT-05 | `unassigned` | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/all_tests/known_issues_regression.rs` | `test_issue532_case195_energy_regression` | `structural` | #532 | `unassigned` | Resolved or closed | `pending` |
| `tests/all_tests/known_issues_regression.rs` | `test_issue533_600_series_peak_load_regression` | `structural` | #533 | `unassigned` | Resolved or closed | `pending` |
| `tests/all_tests/issue_1860_5r1c_time_constant_aware.rs` | `test_case_600_annual_cooling_within_ashrae140_band` | `structural` | #1860, LIMIT-05 | `unassigned` | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/all_tests/issue_1860_5r1c_time_constant_aware.rs` | `test_case_600_annual_heating_within_ashrae140_band` | `structural` | #1860, LIMIT-05 | `unassigned` | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/all_tests/issue_1860_5r1c_time_constant_aware.rs` | `test_case_650_annual_cooling_within_ashrae140_band` | `structural` | #1860, LIMIT-05 | `unassigned` | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/all_tests/issue_1860_5r1c_time_constant_aware.rs` | `test_case_950_annual_cooling_within_ashrae140_band` | `structural` | #1860, LIMIT-05 | `unassigned` | GaugeSolver (#1465/#1462) ships | `pending` |
| `tests/all_tests/invariant_checker_test.rs` | `test_one_watt_artificial_gain_increases_imbalance` | `structural` | #3103, LIMIT-19 | `unassigned` | EnergyBalanceValidator (#1344) investigation resolves algebraic invariant confusion | `pending` |
| `tests/validation/hvac_bestest/runner.rs` | `comparative_e200_cooling_vs_iea_task22_ensemble` | `structural` | LIMIT-05, SOLAR-02 | `unassigned` | GaugeSolver (#1465/#1462) ships; Case-600-class cooling closes | `pending` |
| `tests/all_tests/ffd_cosimulation_validation.rs` | `test_peak_cooling_load_tolerance` | `structural` | #2612, FFD-02 | `unassigned` | Real coupled BES↔FFD solver ships; stub `BuoyancyDrivenFfdSolver` replaced | `pending` |

### Issue #4078 — solar/zone-balance hermeticity triage cohort

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/all_tests/solar_distribution_validation.rs` | `test_ashrae_140_solar_distribution_to_air_is_zero` | `structural` | #4078, LIMIT-30 | `unassigned` | GaugeSolver per-surface distribution rework (#1465/#1462) lands; `solar_distribution_to_air` reaches 0.0 without regressing Case 600/800 cooling | `pending` |
| `tests/all_tests/solar_distribution_validation.rs` | `test_ashrae_140_solar_beam_to_mass_fraction` | `structural` | #4078, LIMIT-30 | `unassigned` | GaugeSolver per-surface distribution rework (#1465/#1462) lands; `solar_beam_to_mass_fraction` reaches 1.0 without collapsing high-mass cooling | `pending` |
| `tests/all_tests/solar_distribution_validation.rs` | `test_solar_fractions_sum_to_one` | `structural` | #4078, LIMIT-30 | `unassigned` | GaugeSolver per-surface distribution rework (#1465/#1462) lands; distribution fractions sum to 1.0 | `pending` |
| `tests/all_tests/issue_1860_5r1c_time_constant_aware.rs` | `test_case_650_solar_lag_improves_annual_cooling` | `structural` | #4078 | `unassigned` | Solar-lag coupling root-caused; Case 650 annual cooling reaches ≥10% above the 3.0 MWh pre-fix baseline | `pending` |
| `tests/all_tests/solar_horizontal_isolation.rs` | `test_roof_surface_irradiance_matches_energyplus` | `structural` | #4078 | `unassigned` | Perez/E+ sky-diffuse discrepancy root-caused; annual error within the 1% tolerance (tolerance must NOT be weakened) | `pending` |
| `tests/all_tests/zone_balance_analytical.rs` | `test_single_timestep_convergence` | `structural` | #4078 | `unassigned` | Single-step convergence contract defined for the multi-state model; strict monotonic decrease holds or the assertion is reformulated | `pending` |

### Case 920 / 950 / 960 blind-mode cohort (Issue #1323 / #1213 / #3071 / #1422)

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/all_tests/ashrae_140_blind_validation.rs` | `test_blind_mode_case_960_infrastructure` | `structural` | LIMIT-18, #1465/#1462 | `unassigned` | GaugeSolver structural 5R1C multi-lumped-mass lands; Case 960 blind heating closes | `pending` |
| `tests/all_tests/ashrae_140_blind_validation.rs` | `test_blind_mode_case_920_annual_energy_within_band` | `structural` | #1213, #1323, #1346 AC | `unassigned` | Roof-solar / high-mass cooling physics fix (#1323) lands; Case 920 annual heating closes | `pending` |
| `tests/all_tests/ashrae_140_blind_validation.rs` | `test_blind_mode_case_950_annual_energy_within_band` | `structural` | #1323, #1347 AC2 | `unassigned` | Roof-solar / high-mass cooling physics fix (#1323) lands; Case 950 strict band closes | `pending` |
| `tests/all_tests/ashrae_140_blind_validation.rs` | `test_case_950_5r1c_free_float_uses_night_vent_overrides_issue_1422` | `structural` | #3071, #1422, #1465/#1462 | `unassigned` | GaugeSolver mass trajectory matches legacy night-flush pre-cool | `pending` |

### Ashrae 140 Case 900 / 920 paired-comparison cohort (Issue #2490 / LIMIT-05)

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/all_tests/ashrae_140_case_900.rs` | `test_case_900_annual_cooling_within_reference_range` | `structural` | LIMIT-05, #2490, #1465/#1462 | `unassigned` | GaugeSolver ships; high-mass 9R4C over-damping closes | `pending` |
| `tests/all_tests/ashrae_140_case_900.rs` | `test_case_900_peak_cooling_within_reference_range` | `structural` | LIMIT-05, #2490, #1465/#1462 | `unassigned` | GaugeSolver ships; instantaneous peak cooling closes | `pending` |
| `tests/all_tests/ashrae_140_case_900.rs` | `test_case_900ff_max_temperature_within_reference_range` | `structural` | LIMIT-05, #2490, #1465/#1462 | `unassigned` | GaugeSolver ships; free-float max-temp closes | `pending` |
| `tests/all_tests/ashrae_140_case_900.rs` | `test_case_900_annual_cooling_energy_with_correction` | `structural` | LIMIT-05, #2490, #1465/#1462 | `unassigned` | GaugeSolver ships | `pending` |
| `tests/all_tests/ashrae_140_case_900.rs` | `test_case_600ff_vs_900ff_paired_comparison` | `structural` | LIMIT-05, #2490, #1465/#1462 | `unassigned` | GaugeSolver ships; 900FF paired-comparison closes | `pending` |
| `tests/all_tests/ashrae_140_case_900.rs` | `test_900_series_regression` | `structural` | Test-pollution (superceded by individual case tests) | `unassigned` | Investigate; either un-ignore after pollution fix or delete | `pending` |
| `tests/all_tests/ashrae_140_integration.rs` | `test_case_600_full_reference_tolerance` | `structural` | #2683, SOLAR-02, LIMIT-05, #1465/#1462 | `unassigned` | All four Case 600 metrics re-enter reference bands | `pending` |
| `tests/all_tests/ashrae_140_integration.rs` | `test_case_610_shading` | `structural` | #62 | `unassigned` | Issue #62 merges; shading test wired into strict gate | `pending` |

### Case 195 / solid conduction cohort (Issue #3064 / LIMIT-20 / #3218)

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/all_tests/ashrae_140_solid_conduction_variants.rs` | `test_case_195_high_mass_walls` | `structural` | #3064, LIMIT-11, #1465/#1462 | `unassigned` | GaugeSolver ships; zero-energy assertion closes | `pending` |
| `tests/all_tests/ashrae_140_solid_conduction_variants.rs` | `test_solid_conduction_variants_integration` | `structural` | LIMIT-20, #3218, LIMIT-11, #3064, #1465/#1462 | `unassigned` | GaugeSolver ships; HighMass variant integration closes | `pending` |
| `tests/all_tests/gauge_validation_case_900.rs` | `test_case_900_gauge_fiver1c_diurnal_parity` | `structural` | #1669 | `unassigned` | GaugeSolver thermal mass implementation lands (Option A) | `pending` |

### Strict-energy-gate observation cohort (Issue #3572)

These tests are `#[ignore]`-quarantined because they assert the strict
ASHRAE 140 ±15% annual-energy band while the engine's measured H/C sits
outside it; the §3572 strict-energy-gate workflow observes the metric
(the gate fails only on regression beyond `regression_tolerance_pp`).
Baseline gap is recorded in each `#[ignore]` reason.

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_800_annual_energy_ashrae140_tolerance` | `structural` | #3572 | `unassigned` | Engine re-enters the ASHRAE 140-2023 Annex B ±15% band; lower the baseline in the test file when it does | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_810_annual_energy_ashrae140_tolerance` | `structural` | #3572 | `unassigned` | Engine re-enters the ASHRAE 140-2023 Annex B ±15% band; lower the baseline in the test file when it does | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_920_annual_energy_ashrae140_tolerance` | `structural` | #3572, #2453, LIMIT-05 | `unassigned` | GaugeSolver (#1465/#1462) ships; 900-series bidirectional annual-energy gap closed | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_950_annual_energy_ashrae140_tolerance` | `structural` | #3572, #3551, LIMIT-24, LIMIT-17 | `unassigned` | GaugeSolver (#1465/#1462) ships; Case 950 (HVAC mode) annual cooling re-enters band (~14× under per §LIMIT-24) | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_960_annual_energy_ashrae140_tolerance` | `structural` | #3572, #3061, LIMIT-14 | `unassigned` | GaugeSolver (#1465/#1462) ships; sunspace air-mass distribution gap closed | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_970_annual_energy_ashrae140_tolerance` | `structural` | #3572, #3552, LIMIT-23 | `unassigned` | GaugeSolver (#1465/#1462) ships; multi-zone air-mass distribution gap closed | `pending` |
| `tests/all_tests/ashrae_140_case_970_validation.rs` | `test_case_970_annual_energy_band` | `structural` | #3585, #3552, LIMIT-23 | `unassigned` | GaugeSolver (#1465/#1462) ships; Case 970 H/C re-enters the ASHRAE 140-2017 §B6.7 envelope | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_610_annual_energy_ashrae140_tolerance` | `structural` | #4170 | `unassigned` | Engine re-enters the ASHRAE 140-2023 Annex B ±15% band; lower the baseline gap in `strict_energy_gate_baseline.json` when it does | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_620_annual_energy_ashrae140_tolerance` | `structural` | #4170 | `unassigned` | Heating re-enters the ±15% band (cooling already passes); lower the baseline gap when it does | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_630_annual_energy_ashrae140_tolerance` | `structural` | #4170 | `unassigned` | Heating re-enters the ±15% band (cooling already passes on the blind path; §LIMIT-39 tracks the 600-series path); lower the baseline gap when it does | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_640_annual_energy_ashrae140_tolerance` | `structural` | #4170, #4167 | `unassigned` | Setback schedule resolved on the blind annual path (#4167) and the engine re-enters the ±15% band; lower the baseline gaps when they close | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_650_annual_energy_ashrae140_tolerance` | `structural` | #4170 | `unassigned` | Cooling re-enters the ±15% band (heating is degenerate [0,0] and passes); lower the baseline gap when it does | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_910_annual_energy_ashrae140_tolerance` | `structural` | #4170 | `unassigned` | Heating passes; cooling left its band 2026-10-09 (loop round 5, doc §18: 0.992 vs [1.147, 1.552], accepted fidelity tradeoff recorded in `strict_energy_gate_baseline.json`) — lower the cooling gap when the engine re-enters | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_930_annual_energy_ashrae140_tolerance` | `structural` | #4170 | `unassigned` | Heating re-enters the ±15% band (cooling already passes); lower the baseline gap when it does | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_940_annual_energy_ashrae140_tolerance` | `structural` | #4170, #4167 | `unassigned` | Setback schedule resolved on the blind annual path (#4167) and the engine re-enters the ±15% band; lower the baseline gaps when they close | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_195_annual_energy_ashrae140_tolerance` | `structural` | #4170, #4169 | `unassigned` | Heating re-enters the ±15% band (cooling already passes; the band itself is under reconciliation in #4169); lower the baseline gap when it does | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_600ff_free_float_strict_gate` | `structural` | #4170 | `unassigned` | Max free-float temperature re-enters the ±15% band (min already passes); lower the baseline gap when it does | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_650ff_free_float_strict_gate` | `structural` | #4170 | `unassigned` | Both min and max free-float temperatures re-enter their ±15% bands; lower the baseline gaps when they do | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_900ff_free_float_strict_gate` | `structural` | #4170, LIMIT-37 | `unassigned` | Min/max free-float temperatures re-enter their ±15% bands (5R1C mass-node damping, §LIMIT-37/#1465 family); lower the baseline gaps when they do | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_950ff_free_float_strict_gate` | `structural` | #4170 | `unassigned` | Min free-float temperature re-enters the ±15% band (max already passes); lower the baseline gap when it does | `pending` |

### LIMIT-22 (gauge-build-only, `cfg_attr(feature = "gauge-solver", ignore)`, Issue #3297)

Feature-gated quarantines: these tests FAIL only under `--features gauge-solver`
(the exact Crank-Nicolson mass-state proxy of `fd7ef13` exposes that each was
passing for a physically-wrong reason). Default-build assertions are fully live
and pass. See KNOWN_ISSUES.md §LIMIT-22 for the root-cause analysis. (The
§LIMIT-21 pre-existing gauge air-trajectory cohort is deliberately NOT
quarantined — it is the #3286 β-soak gate signal.)

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/all_tests/ashrae_140_blind_validation.rs` | `test_case_950_mass_temperature_precooled_issue_1422` | `structural` | #3297, LIMIT-22 | `unassigned` | Gauge air trajectory matches legacy night-flush pre-cool (τ_mass ≈ 61 h CN node swings +1.09 °C vs legacy +2.41 °C), or #1422 re-derives the band with maintainer sign-off | `pending` |
| `tests/all_tests/ashrae_140_case_960_sunspace.rs` | `test_case_960_inter_zone_heat_transfer_analysis` | `structural` | #3297, LIMIT-22 | `unassigned` | Gauge multi-zone integration stability lands for Case 960-class configs (±140 °C step ΔT spikes around the fail-closed guard) | `pending` |
| `tests/all_tests/ashrae_140_case_960_sunspace.rs` | `test_case_960_comprehensive_energy_validation` | `structural` | #3392, LIMIT-22, #273 (inter-zone radiation / condensation 20× over reference) | `unassigned` | Inter-zone radiation / condensation coupling that drives the 20× cooling gap is resolved (tracked under the GaugeSolver cohort #1465/#1462, #3297). When the underlying 5R1C/9R4C trajectory holds for the sunspace config, the comprehensive test can assert all 4 metrics without the documented cooling acceptance. | `pending` |
| `tests/all_tests/invariant_checker_test.rs` | `test_different_zones_respond_differently_to_targeted_gain` | `structural` | #3297, LIMIT-22 (§LIMIT-19/#3103 sibling) | `unassigned` | §LIMIT-19 / #1344 investigation resolves the checker's zero-leverage artificial-gain formula (gain enters the 5R1C residual only through φm·m_air_frac) | `pending` |

### Phase A8 / Issue #3599 — 9R4C legacy solver scratch pool (Issue #3291 / §LIMIT-21)

These `#[ignore]`-quarantined unit tests live in `src/` (not `tests/`) and so
fall outside the `tests/**` audit-invariant of `tests/QUARANTINE.md` (Issue
#3443). They are tracked here for completeness because they block the Code
Coverage Gate (Issue #1932) on every PR off `develop` until the 9R4C legacy
dispatch path is removed post-Phase A8 (Issue #3291). GaugeSolver is the
unconditional default zone solver (`src/sim/thermal_selector.rs`), and the
`9R4C` scratch pool is the LEGACY solver's scratch buffer — once the legacy
dispatch is removed, these tests can be deleted.

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `src/sim/thermal_model_physics/physics_impl/mod.rs` | `scratch_pool_9r4c_*` | `structural` | #3599, #3291, LIMIT-21 | `unassigned` | Legacy 9R4C dispatch removed; GaugeSolver-only path verified | `pending` |
| `src/sim/thermal_model_physics/physics_impl/mod.rs` | `scratch_pool_9r4c_*` (restored_on_free_float) | `structural` | #3599, #3291, LIMIT-21 | `unassigned` | Legacy 9R4C dispatch removed; GaugeSolver-only path verified | `pending` |
| `src/sim/thermal_model_physics/physics_impl/step_9r4c.rs` | `scratch_pool_9r4c_*` | `structural` | #3599, #3291, LIMIT-21 | `unassigned` | Legacy 9R4C dispatch removed; GaugeSolver-only path verified | `pending` |
| `fluxion-fluid/src/autodiff/components.rs` | `test_chiller_jacobian_accuracy` | `structural` | #4210 | `unassigned` | Analytical-vs-finite-diff Jacobians agree within tolerance at saturation points; resolve underlying Jacobian bugs | `pending` |
| `fluxion-fluid/src/autodiff/components.rs` | `test_chiller_jacobian_at_saturation` | `structural` | #4210 | `unassigned` | Analytical and finite-difference Jacobians agree at saturation test inputs; resolve underlying Jacobian bugs | `pending` |
| `fluxion-fluid/src/autodiff/components.rs` | `test_boiler_jacobian_accuracy` | `structural` | #4210 | `unassigned` | Analytical-vs-finite-diff Jacobians agree within tolerance at saturation points; resolve underlying Jacobian bugs | `pending` |
| `fluxion-fluid/src/autodiff/components.rs` | `test_boiler_jacobian_at_saturation` | `structural` | #4210 | `unassigned` | Analytical and finite-difference Jacobians agree at saturation test inputs; resolve underlying Jacobian bugs | `pending` |
| `fluxion-fluid/src/autodiff/components.rs` | `test_vav_box_jacobian_accuracy` | `structural` | #4210 | `unassigned` | Analytical-vs-finite-diff Jacobians agree within tolerance at saturation points; resolve underlying Jacobian bugs | `pending` |
| `fluxion-fluid/src/autodiff/components.rs` | `test_vav_box_jacobian_at_saturation` | `structural` | #4210 | `unassigned` | Analytical and finite-difference Jacobians agree at saturation test inputs; resolve underlying Jacobian bugs | `pending` |
| `fluxion-fluid/src/autodiff/components.rs` | `test_vav_box_gradient_descent_convergence` | `structural` | #4210 | `unassigned` | Gradient descent converges within 1e-2 tolerance; optimizer tuning | `pending` |
| `fluxion-fluid/src/autodiff/components.rs` | `test_pump_jacobian_accuracy` | `structural` | #4210 | `unassigned` | Analytical-vs-finite-diff Jacobians agree within tolerance at saturation points; resolve underlying Jacobian bugs | `pending` |
| `fluxion-fluid/src/autodiff/components.rs` | `test_pump_jacobian_at_saturation` | `structural` | #4210 | `unassigned` | Analytical and finite-difference Jacobians agree at saturation test inputs; resolve underlying Jacobian bugs | `pending` |
| `fluxion-fluid/src/autodiff/components.rs` | `test_cooling_coil_jacobian_accuracy` | `structural` | #4210 | `unassigned` | Analytical-vs-finite-diff Jacobians agree within tolerance at saturation points; resolve underlying Jacobian bugs | `pending` |
| `fluxion-fluid/src/autodiff/components.rs` | `test_cooling_coil_jacobian_at_saturation` | `structural` | #4210 | `unassigned` | Analytical and finite-difference Jacobians agree at saturation test inputs; resolve underlying Jacobian bugs | `pending` |

> `src/` and workspace-member quarantines (like the Phase A8 rows above) are
> covered by the audit invariant of this registry (Issues #3443 / #4178) —
> the auditor scans `tests/`, `src/`, and every workspace member's `src/`
> and `tests/`, so these rows match real `#[ignore]` attributes. Wildcard
> names use the audit's documented wildcard convention (matched against the
> union, never counted as ghosts).

---

## Category: Performance / Memory Profiling (dhat tests)

These tests are `#[ignore]` because dhat backtrace capture makes them too slow for
unit-test CI. They are run manually for memory profiling.

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/dhat_alloc_budget.rs` | `batch_oracle_hot_loop_alloc_budget` | `performance` | Performance | `unassigned` | CI profile budget defined; run in perf CI | `pending` |
| `tests/dhat_batched_surrogate_zero_growth.rs` | `predict_loads_batched_into_zero_steady_state_growth` | `performance` | Performance | `unassigned` | CI profile budget defined; run in perf CI | `pending` |
| `tests/dhat_batched_surrogate_zero_growth.rs` | `submit_with_sender_pingpong_steady_state_floor` | `performance` | Performance | `unassigned` | CI profile budget defined; run in perf CI | `pending` |
| `tests/dhat_batched_surrogate_zero_growth.rs` | `predict_loads_into_with_scratch_zero_steady_state_growth` | `performance` | Performance | `unassigned` | CI profile budget defined; run in perf CI (Issue #4204 single-sample zero-alloc gate) | `pending` |
| `tests/dhat_evaluate_population_numpy_zero_copy.rs` | `evaluate_population_from_slice_zero_steady_state_growth` | `performance` | Performance | `unassigned` | CI profile budget defined; run in perf CI | `pending` |
| `tests/dhat_hybrid_zero_alloc.rs` | `hybrid_solve_timesteps_surrogate_load_branch_zero_steady_state_growth` | `performance` | Performance | `unassigned` | CI profile budget defined; run in perf CI | `pending` |
| `tests/dhat_step_physics_zero_alloc.rs` | `step_physics_day_mode_steady_state_alloc_budget` | `performance` | Performance | `unassigned` | CI profile budget defined; run in perf CI | `pending` |
| `tests/dhat_zone_solar_gain_zero_alloc.rs` | `zone_solar_gain_zero_steady_state_alloc` | `performance` | Performance | `unassigned` | CI profile budget defined; run in perf CI | `pending` |

### BDF / batch-oracle benchmarks (slow, manual)

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/all_tests/bdf_solver_tests.rs` | `benchmark_bdf_stiff_network_100` | `performance` | Performance (manual benchmark) | `unassigned` | Run in perf CI under `--release` with `--nocapture` | `pending` |
| `tests/all_tests/bdf_solver_tests.rs` | `benchmark_bdf_stiff_network_100_throughput` | `performance` | Performance (manual benchmark) | `unassigned` | Run in perf CI under `--release` with `--nocapture` | `pending` |
| `tests/batch_oracle_memory_budget.rs` | `multi_zone_10k_population_peak_rss_under_budget` | `performance` | Performance (10k x 8760-step workload; ~60 s release) | `unassigned` | Nightly memory-budget workflow runs it via `scripts/memory-budget-gate.sh` (Issue #4189) | `pending` |
| `tests/all_tests/lib_batch_oracle.rs` | `test_batch_oracle_*` (5 tests) | `performance` | Slow (full-year simulation) | `unassigned` | Integration CI profile; run on perf runner | `pending` |

### TIMING suites (debug-mode absolute thresholds — #3957)

Quarantined 2026-09-24 from the #3952 deferred-suite triage: these fail
deterministically even in isolated low-load runs because their absolute
throughput/latency thresholds assume a measurement mode the debug per-PR
lanes cannot provide. Disposition (foreman-decided): migrated to a
release-mode nightly perf lane per #3957 and re-baselined there.

**Migrated 2026-09-26** (Issue #3957): the designated measurement home is
`.github/workflows/perf_lane.yml` (nightly, `ubuntu-24.04` pinned,
`cargo nextest run --release` with a `-E` filter over the 8 TIMING modules
and `--run-ignored=all` ignore-stripping). The `#[ignore]` attributes are
kept so the debug lanes stay green; the perf lane executes the full cohort
including these 7 tests. Thresholds re-baselined in release mode — see
`docs/ci/perf-lane.md` for the measurement environment and margin policy.

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/all_tests/throughput_benchmark.rs` | `test_batch_oracle_throughput_100` | `performance` | #3957 (debug-mode absolute threshold, deterministic miss) | `unassigned` | Migrated: runs in perf_lane.yml release nightly; threshold re-baselined 2026-09-26 | `migrated` |
| `tests/all_tests/throughput_benchmark.rs` | `test_batch_oracle_throughput_1000` | `performance` | #3957 (debug-mode absolute threshold, deterministic miss) | `unassigned` | Migrated: runs in perf_lane.yml release nightly; threshold re-baselined 2026-09-26 | `migrated` |
| `tests/all_tests/test_batch_oracle_throughput.rs` | `test_throughput_analytical_1000_configs_sec` | `performance` | #3957 (debug-mode absolute threshold, deterministic miss) | `unassigned` | Migrated: runs in perf_lane.yml release nightly; threshold re-baselined 2026-09-26 | `migrated` |
| `tests/all_tests/performance_ci_test.rs` | `test_multi_zone_throughput` | `performance` | #3957 (debug-mode absolute threshold, deterministic miss) | `unassigned` | Migrated: runs in perf_lane.yml release nightly; threshold re-baselined 2026-09-26 | `migrated` |
| `tests/all_tests/performance_regression_test.rs` | `test_performance_regression` | `performance` | #3957 (debug-mode absolute threshold, deterministic miss) | `unassigned` | Migrated: runs in perf_lane.yml release nightly; baseline regenerated 2026-09-26 via scripts/generate_perf_baseline.py (median-of-7) | `migrated` |
| `tests/all_tests/performance_regression_test.rs` | `test_performance_smoke_test` | `performance` | #3957 (debug-mode absolute threshold, deterministic miss) | `unassigned` | Migrated: runs in perf_lane.yml release nightly; threshold re-baselined 2026-09-26 | `migrated` |
| `tests/all_tests/quantum_topology_stress.rs` | `stress_20_zone_qubo_encoding_performance_k16` | `performance` | #3957 (debug-mode latency budget < 5 ms vs 8.9 ms measured, deterministic miss) | `unassigned` | Migrated: runs in perf_lane.yml release nightly; threshold re-baselined 2026-09-26 | `migrated` |

---

## Category: Hardware-Dependent Tests

These tests require special hardware and are `#[ignore]` on machines without that hardware.

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/all_tests/surrogate_backend_parity.rs` | `test_cpu_vs_cuda_parity` | `hardware` | Hardware (GPU) | `unassigned` | Run on GPU hardware-in-loop CI with `--include-ignored` | `pending` |
| `src/ai/surrogate.rs` | `hardware_cuda_ep_probe_reports_activation_and_runs_inference` | `hardware` | #3313 | `unassigned` | Requires NVIDIA GPU + CUDA runtime; run on CUDA-capable hardware (see docs/ORT_EP_VALIDATION.md) | `pending` |
| `src/ai/surrogate.rs` | `hardware_coreml_ep_probe_reports_activation_and_runs_inference` | `hardware` | #3313 | `unassigned` | Requires Apple Silicon macOS hardware (see docs/ORT_EP_VALIDATION.md) | `pending` |
| `src/ai/surrogate.rs` | `hardware_directml_ep_probe_reports_activation_and_runs_inference` | `hardware` | #3313 | `unassigned` | Requires Windows + DirectX 12 GPU (see docs/ORT_EP_VALIDATION.md) | `pending` |

---

## Category: Calibration / Pending Data

These tests are `#[ignore]` because they await external calibration data or verification.

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/all_tests/solar_peak_cooling_tdd.rs` | `test_case_600_peak_cooling_red` | `calibration` | Calibration | `unassigned` | Expected values verified against ASHRAE 140 reference | `pending` |
| `tests/all_tests/solar_peak_cooling_tdd.rs` | `test_case_900_peak_cooling_red` | `calibration` | Calibration | `unassigned` | Expected values verified against ASHRAE 140 reference | `pending` |
| `tests/all_tests/thermal_comfort_prediction_validation.rs` | `test_eplus_thermal_comfort_reference_pending` | `calibration` | Data | `unassigned` | EnergyPlus thermal comfort benchmark data available | `pending` |
| `tests/all_tests/thermal_comfort_prediction_validation.rs` | `test_eplus_thermal_comfort_reference_pending` | `calibration` | Data | `unassigned` | EnergyPlus thermal comfort benchmark data available | `pending` |
| `tests/all_tests/test_statistical_validation.rs` | `test_cli_statistical_flag` | `calibration` | Environment | `unassigned` | Compiled `fluxion` binary at `target/release/fluxion` | `pending` |
| `tests/all_tests/surface_flux_parity.rs` | `test_parity_roof_zero_followup_1323` | `calibration` | #1323 | `unassigned` | Post-#1323 roof-solar physics fix lands | `pending` |
| `tests/all_tests/gauge_validation_case_900.rs` | `test_case_900_gauge_fiver1c_diurnal_parity` | `calibration` | #1669 | `unassigned` | GaugeSolver thermal mass implementation | `pending` |

### Pending reference CSVs (Issue #1331 / #1168 / #1166)

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/all_tests/ashrae_140_blind_validation.rs` | `test_blind_mode_case_800_annual_energy_within_band` | `calibration` | #1331, #1168 | `unassigned` | `case_800_energy_reference.csv` regenerated from EnergyPlus | `pending` |
| `tests/all_tests/ashrae_140_blind_validation.rs` | `test_blind_mode_case_810_annual_energy_within_band` | `calibration` | #1331, #1168 | `unassigned` | `case_810_energy_reference.csv` regenerated from EnergyPlus | `pending` |
| `tests/all_tests/ashrae_140_blind_validation.rs` | `test_blind_mode_case_960_annual_energy_within_band` | `calibration` | #1331, #1168 | `unassigned` | `case_960_energy_reference.csv` regenerated from EnergyPlus | `pending` |
| `src/solar/surface_irradiance.rs` | `test_perez_f1_coefficients_monotonic_non_decreasing` | `calibration` | #1695 | `unassigned` | Perez Table 3 source-data correction (F1 monotonicity at bin 5/6); #1695 scope excludes coefficient changes — separate data-correction effort | `pending` |
| `src/solar/surface_irradiance.rs` | `test_perez_f1_increases_at_each_sky_clearness_transition` | `calibration` | #1695 | `unassigned` | Perez Table 3 source-data correction (F1 monotonicity at epsilon=4.5 transition); #1695 scope excludes coefficient changes — separate data-correction effort | `pending` |

### FD-vs-EnergyPlus step-response redesign cohort (Issue #4058 / #3981)

Former flux channel was a circular identity (back-calculation through the pre-#3981
`h*(T_zone - T_0)` extraction); the conservative Robin extraction broke the circularity.
The harness is redesigned to the temperature channel (interior surface temperature under
the real film), but the ~96-row reference datasets cannot spin up multi-day thermal
states (concrete tau ~ 29 h) and the E+ interior-film composition is undocumented.

The 6 FD step-response cohort rows that previously lived here were un-ignored by PR #4111 / commit 8903331 (closes #4058). They are now live tests; their rows were removed from the registry in PR #4111 follow-up commits once the tests were confirmed passing (`test_inventory.json` reflects the new ignored count).

---

## Category: CI Infrastructure

These tests are `#[ignore]` because CI is broken, not because the test logic is wrong.

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/all_tests/idf_ashrae_140_acceptance.rs` | `idf_case_600_annual_heating_within_15_percent_strict` | `ci-broken` | #1577 | `unassigned` | CI fixed; develop CI can run tests to verify | `pending` |

---

## Category: Manual Baseline Regeneration

These tests are `#[ignore]` because they regenerate baselines and should only be run
manually after legitimate changes.

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/all_tests/surrogate_drift_fallback_regression.rs` | `fallback_annual_hvac_diagnostic` | `manual-baseline` | Manual | `unassigned` | Run manually after surrogate change; not in CI | `pending` |
| `tests/all_tests/surrogate_cold_start_test.rs` | `diagnostic_print_cold_warm_cycles` | `manual-baseline` | Manual | `unassigned` | Run manually after ort version bump; not in CI | `pending` |

---

## Category: Other / Unclassified

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/all_tests/lib_batch_oracle.rs` | `test_evaluate_population_u_value_impact` | `other` | Slow (full-year simulation) | `unassigned` | Integration CI profile | `pending` |
| `tests/all_tests/lib_batch_oracle.rs` | `test_evaluate_population_setpoint_impact` | `other` | Slow (full-year simulation) | `unassigned` | Integration CI profile | `pending` |
| `tests/all_tests/lib_batch_oracle.rs` | `test_evaluate_population_with_large_population` | `other` | Slow (full-year simulation) | `unassigned` | Integration CI profile | `pending` |
| `tests/all_tests/lib_batch_oracle.rs` | `test_evaluate_population_with_surrogates_no_model` | `other` | Slow (hangs when surrogates=true without model loaded) | `unassigned` | Investigate; either fix or delete | `pending` |
| `tests/all_tests/lib_batch_oracle.rs` | `test_evaluate_population_parallel_execution` | `other` | Slow (full-year simulation) | `unassigned` | Integration CI profile | `pending` |
| `tests/all_tests/bdf_solver_tests.rs` | `test_bdf_*` (2 tests) | `other` | Unknown | `unassigned` | Investigate; determine un-ignore criteria | `pending` |
| `tests/all_tests/weather_vs_energyplus.rs` | `test_humidity_ratio_psychrometrics_vs_energyplus` | `other` | #2673 | `unassigned` | Formula generator embedded; issue #2673 resolves | `pending` |
| `tests/all_tests/weather_vs_energyplus.rs` | `test_synthetic_miami_tmy_matches_reference` | `other` | #2673 | `unassigned` | Formula generator embedded; issue #2673 resolves | `pending` |
| `tests/all_tests/energyplus_comparison_tests.rs` | `test_900_series_comprehensive_comparison` | `other` | Long-running | `unassigned` | Run explicitly when needed; not in CI | `pending` |
| `src/validation/thermal_mass.rs` | `test_thermal_mass_*` | `other` | #3770 | `unassigned` | RESOLVED via PR #3770: empirical tuning factor removed by #2717; `target_tau_hours` derivation fix (Phase B2a per PR #3845 audit) restored the Case 900 mass node to within the -50..100 °C plausibility band. Final mass temperature observed 20.75 °C after 24 h (was 141 °C). `#[ignore]` removed; original assertions restored and passing | `closed (resolve #3770)` |
| `fluxion-behavior/src/occupancy.rs` | `test_statistical_validation_*` | `other` | #3705 (flaky: OS-seeded 10k-step Markov statistical assertion) | `unassigned` | Fixed-seed sampling or variance-aware tolerance; keep the distribution check live in CI. Closed: fixed seeds (`SmallRng::seed_from_u64`, distinct per Monte Carlo trajectory); `#[ignore]` removed; 5% relative-error assertion kept (resolves #3705, absorbs duplicate #3683) | `closed` |
| `fluxion-wasm/tests/wasm_integration_tests.rs` | `wasm_run_full_annual_*` | `other` | #3703 (wasm step() toy model; #3595 smoke test) | `unassigned` | RESOLVED via #3624: `step()` now drives the embedded WD600 annual weather schedule (`weather: "ASHRAE_600"`), test un-ignored. Cooling asserts the published ±15% band (satisfied); heating asserts a recorded regression band — the toy single-node RC model has no solar aperture, so published heating-band parity still requires an engine-backed wasm surface | `closed (resolve #3624)` |
| `tests/all_tests/energyplus_comparison_tests.rs` | `test_900_series_comprehensive_comparison` | `other` | Long-running | `unassigned` | Run explicitly when needed; not in CI | `pending` |
| `crates/fluxion-twin/src/telemetry/mqtt.rs` | `test_mqtt_consumer_integration` | `other` | #4211 | `unassigned` | TLS MQTT broker provisioned at mqtts://localhost:8883 in the test environment, or documented manual broker setup | `pending` |

## Category: Strict-Energy-Gate & Structural Diagnostics (Issue #3443 reconciliation)

> Reconciliation section: these 10 `#[ignore]` attributes landed on `develop`
> (via #3572 / #3585 Wave-5/8 and the #3551/#3552 diagnostic placeholders)
> without registry rows, tripping the Issue #3443 downward-only ratchet. This
> section back-fills the missing rows and raises `BASELINE_ORPHANED_IGNORES`
> to 10 with the freeze-set entries in `scripts/generate_quarantine_registry.py`.

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_800_annual_energy_ashrae140_tolerance` | `structural` | #3572 | `unassigned` | Engine re-enters ±15% ASHRAE 140-2023 Annex B band; lower in-file baseline | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_810_annual_energy_ashrae140_tolerance` | `structural` | #3572 | `unassigned` | Engine re-enters ±15% band; lower in-file baseline | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_920_annual_energy_ashrae140_tolerance` | `structural` | #3572 | `unassigned` | Engine re-enters ±15% band; lower in-file baseline | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_950_annual_energy_ashrae140_tolerance` | `structural` | #3572 | `unassigned` | Engine re-enters ±15% band; lower in-file baseline | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_960_annual_energy_ashrae140_tolerance` | `structural` | #3572 | `unassigned` | Engine re-enters ±15% band; lower in-file baseline | `pending` |
| `tests/all_tests/zone_balance_eplus_isolation.rs` | `test_case_970_annual_energy_ashrae140_tolerance` | `structural` | #3572 | `unassigned` | Engine re-enters ±15% band; lower in-file baseline | `pending` |
| `tests/all_tests/ashrae_140_case_970_validation.rs` | `test_case_970_annual_energy_band` | `structural` | #3585 / §LIMIT-23 (#3552) | `unassigned` | Multi-zone air-mass distribution gap closed (GaugeSolver #1465/#1462 rework) | `pending` |
| `tests/diagnostics/case_950_hvac_mode_seasonal_attribution.rs` | `test_case_950_hvac_mode_seasonal_attribution` | `structural` | #3551 / §LIMIT-24 | `unassigned` | Implement per-month attribution walk (follow-up PR, GaugeSolver #1465/#1462) | `pending` |
| `tests/diagnostics/case_970_multi_zone_seasonal_attribution.rs` | `case_970_per_zone_seasonal_attribution_placeholder` | `structural` | #3552 / §LIMIT-23 | `unassigned` | Implement per-month per-zone attribution | `pending` |
| `fluxion-wasm/tests/wasm_integration_tests.rs` | `wasm_run_full_annual_*` | `structural` | #3703 (wasm step() toy model; #3595 smoke test) | `unassigned` | RESOLVED via #3624: `step()` now drives the embedded WD600 annual weather schedule (`weather: "ASHRAE_600"`), test un-ignored. Cooling asserts the published ±15% band (satisfied); heating asserts a recorded regression band — the toy single-node RC model has no solar aperture, so published heating-band parity still requires an engine-backed wasm surface | `closed (resolve #3624)` |

### Issue #3803 (Phase C1, BENCH-01) — Case 620 narrower ASHRAE 140-2023 band exposes physics gap

`src/validation/benchmark.rs` Case 620 reference was migrated from the
historical "Calibrated for 5R1C model" values to the raw ASHRAE 140-2023
Annex B Tables B8-1..B8-4 inter-program range from
`data/ashrae140_reference.json` (Std140_TF_Results.pdf, TESS 19-Aug-2024;
programs BSIMAC 9.0.74, CSE 0.861.1, DeST 2.0, EnergyPlus 9.0.1, ESP-r 13.3,
TRNSYS 18.01.0001). Per RULES.md / ADR-0001 the band is not widened; the
four `ashrae_140_case_600_series::case_620::test_*` assertions are quarantined
while the engine sits outside the narrower inter-program spread on every
metric (H=5.83 vs [4.09,4.72]; C=2.47 vs [3.84,4.40]; pH=3.58 vs [3.04,3.38];
pC=3.43 vs [3.96,4.80]). A follow-up issue should add a §LIMIT- entry to
`docs/KNOWN_ISSUES.md` for the Case 620 east/west-window physics gap; the
daily `ashrae_140_blind_validation` runner still prints the engine value vs
the new band for triage.

| Test File | Test Name | Category | Blocking Issue | Owner | Un-Ignore Criteria | Status |
|-----------|-----------|----------|----------------|-------|-------------------|--------|
| `tests/all_tests/ashrae_140_case_600_series.rs` | `test_annual_heating` | `structural` | #3803 | `unassigned` | Engine annual heating re-enters [4.094, 4.719] MWh | `pending` |
| `tests/all_tests/ashrae_140_case_600_series.rs` | `test_annual_cooling` | `structural` | #3803 | `unassigned` | Engine annual cooling re-enters [3.841, 4.404] MWh | `pending` |
| `tests/all_tests/ashrae_140_case_600_series.rs` | `test_peak_heating` | `structural` | #3803 | `unassigned` | Engine peak heating re-enters [3.038, 3.385] kW | `pending` |
| `tests/all_tests/ashrae_140_case_600_series.rs` | `test_peak_cooling` | `structural` | #3803 | `unassigned` | Engine peak cooling re-enters [3.955, 4.797] kW | `pending` |

---

## Summary

| Category | Count | Status |
|----------|-------|--------|
| Diagnostic tests (#2536) | 22 | `pending` |
| Structural gaps (LIMIT-*) | 88 pending + 1 closed (3 9R4C legacy pool, Issue #3599; 7 strict-energy-gate / Case 970 cohort, Issue #3572 / #3585; 4 Case 620 BENCH-01 narrower-band cohort, Issue #3803; 11 fluxion-fluid Jacobian cohort, Issue #4210) | `pending` |
| Performance/memory (dhat + BDF + batch) | 17 (10 pending + 7 migrated to perf_lane.yml, Issue #3957) | `pending` / `migrated` |
| Hardware-dependent (GPU) | 4 (1 parity + 3 ONNX EP hardware probes, Issue #3313) | `pending` |
| Calibration/pending data | 12 (incl. 3 pending reference CSVs #1331/#1168; 2 Perez F1 source-data rows, Issue #1695) | `pending` |
| CI infrastructure | 1 | `pending` |
| Manual baseline regen | 2 | `pending` |
| Other/unclassified | 11 pending + 3 closed (1 TLS-MQTT integration row, Issue #4211) | `pending` |
| **Total** | **161** | |

(82 orphan entries were triaged into this registry by Issue #3443; the totals
above include the 23 pre-existing entries and the 82 newly-added ones. The 3
gauge-build-only `cfg_attr(...)` ignores live in the structural-cohort section
above; the audit scanner counts only unconditional `#[ignore]` attributes.)

2026-09-11 (PR improve/quarantine-burndown): 9 orphan `#[ignore]`s audited and
registered — 7 strict-energy/Case-970 structural rows (Issue #3572 / #3585,
LIMIT-05/14/17/23/24) and 2 diagnostic placeholder rows (Issue #3551 / #3552).
`test_case_970_validator_accepts_canonical_midpoints` was un-ignored (its
assertions validate the validator, not the engine band, and it passes).

2026-09-16 (PR fix/issue-3803-phase-c1-bench-01): 4 Case 620 quarantines added
— Phase C1 BENCH-01 migration from 5R1C-calibrated to raw ASHRAE 140-2023
Annex B narrower inter-program range (Issue #3803); see
`tests/reference_data/zone_balance/PROVENANCE.md` for the full provenance.

2026-09-28 (Issues #4178 / #4179): audit scan widened beyond `tests/` to
`src/` and every workspace member's `src/` / `tests/`; registry schema
frozen to Test File | Test Name | Category | Blocking Issue | Owner |
Un-Ignore Criteria | Status (Category backfilled from the section, Owner
backfilled `unassigned`). 17 newly-visible `#[ignore]`s registered — 3 ONNX
EP hardware probes (Issue #3313), 2 Perez F1 source-data rows (Issue #1695),
11 fluxion-fluid Jacobian rows (new tracking Issue #4210), 1 TLS-MQTT
integration row (new tracking Issue #4211). The Issue #3443 ratchet is now a
key-membership check with an empty freeze snapshot (`BASELINE_ORPHANED_IGNORES
= 0`): any new orphan fails `--strict` until it is registered. Removed a
stray `=======` conflict marker left in the Other section.

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
*Last Updated: 2026-09-26 (Issue #3957: 7 TIMING-suite tests migrated to the release-mode nightly perf lane `.github/workflows/perf_lane.yml` (ubuntu-24.04 pinned); thresholds re-baselined in release mode, measurement environment documented in docs/ci/perf-lane.md. Contention-sensitive `api_concurrent_throughput::concurrent_throughput_smoke` and `security_rate_limit::rate_limiter_lru_cap_respected_under_concurrent_cold_flood` stay in-suite with a `.config/nextest.toml` retries override.)*
| `tests/all_tests/ashrae_140_case_600_series.rs` | `test_annual_heating` | `structural` | #4332, LIMIT-35 | `unassigned` | Case 610 annual heating re-enters the published [4.36, 5.79] MWh range (loop round 5: 6.412 under spec-correct shading, accepted fidelity tradeoff, doc §18) | `pending` |
| `tests/all_tests/ashrae_140_case_600_series.rs` | `test_annual_cooling` | `structural` | #4332, LIMIT-35 | `unassigned` | Case 610 annual cooling re-enters the published [3.92, 6.14] MWh range (loop round 5: 2.724 under spec-correct shading; §17 base-deficit analysis) | `pending` |
| `tests/all_tests/ashrae_140_integration.rs` | `test_case_600_baseline` | `structural` | #4332, LIMIT-35 | `unassigned` | Case 600 annual heating re-enters the reference [4.0, 7.5] MWh band (loop round 9 PR-c: self-consistent gauge air-node metering replaced the network-foreign 5R1C series coefficient; pre-PR-d envelope runs cold in free float) | `pending` |
