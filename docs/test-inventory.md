# Fluxion Test Inventory

> **TL;DR**: Comprehensive catalog of all test files in the fluxion repository.
> **Key decisions**: Isolation tests per ARCHITECTURE.md Module 1-5 | Integration tests cover E+ parity, ASHRAE 140, HVAC, API
> **Owned by**: QA / Test Infrastructure
> **Reviewed**: 2026-07-13

## Isolation Tests

Tests that validate individual physics modules in isolation against reference data or analytical solutions.

### Weather Module

| Test File | Module | What it Tests | Reference Data |
|-----------|--------|---------------|----------------|
| `tests/weather_isolation.rs` | Weather | EPW parser, hourly data parsing, psychrometrics | `reference_data/weather/denver_tmy3_reference.csv` |
| `tests/weather_epw.rs` | Weather | EPW version detection, field parsing | N/A (unit tests) |
| `tests/weather_psychrometrics.rs` | Weather | Saturation vapor pressure, humidity ratio, enthalpy vs ASHRAE Table 3 | ASHRAE Handbook Fundamentals Ch. 1 |

### Solar Module

| Test File | Module | What it Tests | Reference Data |
|-----------|--------|---------------|----------------|
| `tests/solar_isolation.rs` | Solar | Solar position (altitude/azimuth/zenith), surface irradiance, sol-air temperature | `reference_data/solar/solar_position_denver.csv`, `reference_data/solar/surface_irradiance_south.csv` |
| `tests/surface_irradiance_vs_energyplus.rs` | Solar | Surface irradiance on south-facing wall vs E+ 25.2 | `reference_data/solar/surface_irradiance_south.csv` |
| `tests/solar_calculation_validation.rs` | Solar | DNI/DHI calculations, window solar gain, orientation effects | N/A (unit tests) |
| `tests/solar_distribution_validation.rs` | Solar | Solar distribution across surfaces | N/A |
| `tests/solar_distribution_tests.rs` | Solar | Perez transposition model, sky diffuse | N/A |

### Conduction Module

| Test File | Module | What it Tests | Reference Data |
|-----------|--------|---------------|----------------|
| `tests/conduction_5r1c_isolation.rs` | Conduction | 5R1C solver: steady-state, transient step response, time constant | Analytical (no CSV needed) |
| `tests/conduction_step_response_vs_energyplus.rs` | Conduction | Finite difference solver step response | `reference_data/conduction/step_response_200mm_concrete.csv`, `step_response_fixed_zone_20c.csv` |
| `tests/conduction_ctf_step_response_vs_energyplus.rs` | Conduction | CTF coefficient calculation | N/A |
| `tests/ctf_coefficient_validation.rs` | Conduction | CTF material properties, time constant | N/A |
| `tests/ctf_analytical_step_response.rs` | Conduction | CTF vs analytical transient conduction (Incropera Ch. 5) | Analytical |
| `tests/per_surface_conduction_isolation.rs` | Conduction | Per-surface conduction solver | N/A |
| `tests/surface_flux_provider_isolation.rs` | Conduction | SurfaceHeatFluxProvider trait | N/A |
| `tests/conduction_solver_manager.rs` | Conduction | SolverManager orchestration | N/A |

### Ventilation Module

| Test File | Module | What it Tests | Reference Data |
|-----------|--------|---------------|----------------|
| `tests/ventilation_isolation.rs` | Ventilation | WeatherDependentVentilation ACH vs ASHRAE 140 §5.5.3.6 default (0.5 ACH) | `reference_data/ventilation/infiltration_denver.csv` |
| `tests/ventilation_schedule_trait.rs` | Ventilation | ConstantVentilation, ScheduledVentilation, WeatherDependentVentilation trait impls | N/A |
| `tests/ventilation_infiltration_vs_energyplus.rs` | Ventilation | Infiltration ACH vs E+ | `reference_data/ventilation/infiltration_*.csv` |

### Zone Balance Module

| Test File | Module | What it Tests | Reference Data |
|-----------|--------|---------------|----------------|
| `tests/zone_balance_trait_isolation.rs` | Zone Balance | ThermalModelTrait implementations, PhysicsThermalModel, UnifiedThermalModel | N/A |
| `tests/zone_balance_eplus_isolation.rs` | Zone Balance | PhysicsThermalModel vs E+ Case 600 reference, free-floating temps | `reference_data/zone_balance/case_600_energy_reference.csv` |
| `tests/zone_balance_analytical.rs` | Zone Balance | Heat balance equation vs analytical solutions (steady-state, transient, energy conservation) | N/A |
| `tests/thermal_mass_coupling_tests.rs` | Zone Balance | h_tr_ms, h_tr_is conductance, tau calculation | N/A |
| `tests/thermal_mass_time_constant_validation.rs` | Zone Balance | Thermal time constant validation | N/A |

### Physics Core

| Test File | Module | What it Tests | Reference Data |
|-----------|--------|---------------|----------------|
| `tests/test_conductance_calculations.rs` | Physics Core | 5R1C conductances: h_tr_em, h_tr_ms, h_tr_is, h_tr_w, h_ve | ASHRAE 140 Case 600 |
| `tests/test_differentiation.rs` | Physics Core | Auto-differentiation scalar (dual numbers) | N/A |
| `tests/test_interpolation.rs` | Physics Core | Linear, cubic spline, piecewise Hermite interpolation | N/A |
| `tests/step_physics_unit_tests.rs` | Physics Core | step_physics boundary conditions | N/A |
| `tests/test_8r3c_evaluation.rs` | Physics Core | 8R3C thermal network evaluation | N/A |
| `tests/test_6r2c_comprehensive.rs` | Physics Core | 6R2C comprehensive tests | N/A |

## Integration Tests

Tests that validate multiple modules working together, system-level behavior, or compliance with standards.

### ASHRAE 140 Compliance

| Test File | Scenario | Pass Criteria |
|-----------|----------|---------------|
| `tests/ashrae_140_case_900.rs` | High-mass concrete building (Case 900), annual energy, peak loads | Annual heating 1.17-2.04 MWh, cooling 2.13-3.67 MWh |
| `tests/ashrae_140_validation.rs` | Comprehensive ASHRAE 140 validator framework | 18+ cases instantiated, all metrics validated |
| `tests/ashrae_140_free_floating.rs` | Cases 600FF, 650FF, 900FF, 950FF free-floating temps | Min/max temps within ASHRAE 140 ranges |
| `tests/ashrae_140_case_600.rs` | Low-mass building (Case 600), annual energy, peak loads | Reference ranges per ASHRAE 140 |
| `tests/ashrae_140_case_non_residential.rs` | Non-residential cases | Per-case reference ranges |
| `tests/ashrae_140_solar_gain_variants.rs` | Solar gain diagnostic cases | Per-case reference ranges |
| `tests/ashrae_140_blind_validation.rs` | Blind validation mode (no case ID) | Within ±15% of reference |
| `tests/ashrae_140_weather_comparison.rs` | Weather data impact on cases | Consistent results across weather files |
| `tests/ashrae_140_setback_ventilation.rs` | Setback and ventilation interactions | Per-case reference ranges |
| `tests/ashrae_140_input_validation.rs` | Invalid input handling | Proper error messages |
| `tests/ashrae_140/diagnostics.rs` | Cases 195-470, 800-810 diagnostic validation | Consolidated validation logic |
| `tests/ashrae_140_coverage.rs` | Test coverage for ASHRAE 140 cases | Coverage metrics |
| `tests/ashrae_140_integration.rs` | End-to-end ASHRAE 140 workflow | Full case execution |

### EnergyPlus Parity

| Test File | Scenario | Pass Criteria |
|-----------|----------|---------------|
| `tests/energyplus_comparison_tests.rs` | E+ comparison framework | Within 1% for energy, 0.5°C for temperature |
| `tests/surface_flux_parity.rs` | Surface heat flux vs E+ | Within tolerance |
| `tests/diag_917_energy.rs` | Diagnostic 917 energy metrics | Per-reference tolerance |
| `tests/diag_917_solar.rs` | Diagnostic 917 solar metrics | Per-reference tolerance |
| `tests/diag_917_v2.rs` | Diagnostic 917 v2 | Per-reference tolerance |

### HVAC Tests

| Test File | Scenario | Pass Criteria |
|-----------|----------|---------------|
| `tests/hvac/zone_control_tests.rs` | Zone-level HVAC control, setpoints, deadband | Correct mode determination |
| `tests/test_hvac_load_calculation.rs` | Ti_free calculation, HVAC mode, load calculations | Correct sign convention |
| `tests/hvac_bestest_validation.rs` | ASHRAE RP-865 BESTEST HVAC suite | Per-case pass/fail |
| `tests/hvac_predictive_modulation.rs` | Predictive modulation | Validation metrics |
| `tests/issue_365_hvac_sensitivity_verification.rs` | HVAC sensitivity analysis | Sensitivity bounds |

### API & CLI

| Test File | Scenario | Pass Criteria |
|-----------|----------|---------------|
| `tests/api_integration_tests.rs` | REST API (axum) end-to-end | HTTP 200, correct JSON schema |
| `tests/cli_integration.rs` | CLI commands and exit codes | Exit code 0 for success |
| `tests/test_guardrail_exit_codes.rs` | Guardrail enforcement exit codes | Correct error codes |
| `tests/test_api_error.rs` | API error handling | Proper error responses |

### Surrogate & AI

| Test File | Scenario | Pass Criteria |
|-----------|----------|---------------|
| `tests/test_modular_surrogates.rs` | ComponentSurrogate, CompositeSurrogate | Correct fallback to analytical |
| `tests/surrogate_models.rs` | Surrogate model loading/execution | Valid predictions |
| `tests/surrogate_backend_parity.rs` | Surrogate vs analytical backend parity | Within surrogate tolerance |
| `tests/test_batched_inference.rs` | Batch inference throughput | Performance targets |
| `tests/surrogate_golden_output.rs` | Surrogate golden output validation | Match expected outputs |

### Validation Framework

| Test File | Scenario | Pass Criteria |
|-----------|----------|---------------|
| `tests/validation/ab_testing.rs` | A/B testing framework for thermal network variants | NMBE, CV(RMSE) calculations |
| `tests/validation/benchmark_report.rs` | BenchmarkReport aggregation | Valid metric aggregation |
| `tests/validation/free_floating_tests.rs` | Free-floating temperature validation | Case 900FF known limitation documented |
| `tests/validation/multi_reference_integration.rs` | Multi-reference DB enrichment | Per-program references present |
| `tests/validation/high_mass_tests.rs` | High-mass case validation | Per-case tolerance |
| `tests/validation/tolerance_test.rs` | Validation tolerance framework | Tolerance calculations correct |
| `tests/validation/case_900_control_audit.rs` | Case 900 control audit | Control action validation |
| `tests/validation/case_900_solar_audit.rs` | Case 900 solar audit | Solar distribution validation |
| `tests/validation/case_900_peak_attribution.rs` | Case 900 peak load attribution | Peak attribution breakdown |
| `tests/validation/case_900_peak_diagnostic.rs` | Case 900 peak diagnostic | Peak metrics |
| `tests/validation/case_900_annual_energy_attribution.rs` | Case 900 annual energy attribution | Energy attribution breakdown |
| `tests/validation/multi_zone_validation.rs` | Multi-zone cases | Per-zone tolerance |
| `tests/validation/night_ventilation_air_side.rs` | Night ventilation | Energy savings validation |
| `tests/validation/esp_r_test.rs` | ESP-r reference comparison | Cross-program tolerance |
| `tests/validation/cross_validation_test.rs` | Cross-validation framework | Statistical metrics |
| `tests/validation/hourly_ff_profile.rs` | Hourly free-floating profile | Profile shape validation |
| `tests/validation/empirical_validation_test.rs` | Empirical validation | Statistical significance |
| `tests/validation_report.rs` | Validation report generation | Markdown output |

### Performance & Concurrency

| Test File | Scenario | Pass Criteria |
|-----------|----------|---------------|
| `tests/performance_regression_test.rs` | Performance vs stored baseline | <10% regression threshold |
| `tests/performance_integration_test.rs` | ValidationSuite case-coverage aggregation | Pass/warn/fail counts, MAE/RMSE, BenchmarkReport generation |
| `tests/performance_completion_test.rs` | Phase47CompletionValidator | 14 requirements, completion_percentage math, report generation |
| `tests/concurrency/loom_concurrency_tests.rs` | Parallel solver race conditions | LOOM=1 mode checking |
| `tests/test_parallel_validation.rs` | Parallel validation | Correct results |
| `tests/test_deterministic_parallel.rs` | Deterministic parallel execution | Reproducible results |

### Physics Edge Cases

| Test File | Scenario | Pass Criteria |
|-----------|----------|---------------|
| `tests/test_edge_cases.rs` | Extreme parameters, NaN, Inf, zero, negative | No panics, finite output |
| `tests/test_energy_conservation.rs` | Energy conservation in analytical path | Residual < 0.1% |
| `tests/thermal_invariants.rs` | Thermal invariant checking | Invariants preserved |
| `tests/limit_05_inversion_regression.rs` | LIMIT-05 inversion regression | Correct behavior |
| `tests/regression_exterior_film_unification.rs` | Exterior film coefficient | Consistent values |

### Miscellaneous Integration

| Test File | Scenario | Pass Criteria |
|-----------|----------|---------------|
| `tests/examples_smoke.rs` | Example files execute | No panics |
| `tests/testing_tdd_framework_validates_against_ep.rs` | TDD framework validation | Pass/fail criteria |
| `tests/test_constants_module.rs` | Physical constants | Values within tolerance |
| `tests/test_constants_integration.rs` | Constants integration | Consistent usage |
| `tests/test_statistical_validation.rs` | Statistical validation metrics | NMBE, CV(RMSE), MAE |
| `tests/test_allocation_tracking.rs` | Memory allocation tracking | No leaks |
| `tests/test_batch_oracle_throughput.rs` | Batch oracle throughput | Performance targets |
| `tests/test_coverage_enhancement.rs` | Coverage improvement tracking | Coverage metrics |
| `tests/test_critical_paths.rs` | Critical path testing | Path coverage |
| `tests/test_delta_analysis.rs` | Delta analysis framework | Correct deltas |

### Diagnostic Tests

| Test File | Scenario | Pass Criteria |
|-----------|----------|---------------|
| `tests/ashrae_140_diagnostic_test.rs` | Diagnostic case framework | Diagnostic output |
| `tests/ashrae_140_diagnostic_integration_test.rs` | Diagnostic integration | End-to-end diagnostics |
| `tests/ashrae_140_output_validation.rs` | Output validation | Per-metric tolerance |
| `tests/case_900_quick_check.rs` | Case 900 quick validation | Sanity checks |
| `tests/case_900_cooling_diagnostic.rs` | Case 900 cooling diagnostic | Cooling metrics |
| `tests/diag_phim.rs` | Phi*_m diagnostic metrics | Per-metric tolerance |
| `tests/diag_mass_traj.rs` | Mass trajectory diagnostics | Trajectory validation |
| `tests/case_900_determinism.rs` | Deterministic execution | Reproducible results |

## Reference Data Directory Structure

```
tests/reference_data/
├── ashrae140/
│   └── monthly/
│       ├── case_600_monthly_reference.csv
│       └── case_900_monthly_reference.csv
├── conduction/
│   ├── step_response_200mm_concrete.csv     (E+ 25.2.0 output)
│   ├── step_response_composite.csv          (synthetic)
│   ├── step_response_fixed_zone_20c.csv     (E+ output)
│   ├── step_response_floor.csv              (synthetic)
│   ├── step_response_lightweight.csv         (synthetic)
│   └── step_response_roof.csv               (synthetic)
├── solar/
│   ├── ashrae_140_surface_incident_solar.csv
│   ├── ashrae_140_surface_incident_solar_summary.csv
│   ├── case_900_roof_solar_hourly.csv
│   ├── solar_gain_distribution.csv
│   ├── solar_position_denver.csv
│   ├── solar_position_miami.csv
│   ├── solar_position_phoenix.csv
│   ├── surface_irradiance_south.csv
│   ├── surface_irradiance_south_miami.csv
│   ├── surface_irradiance_south_minneapolis.csv
│   └── surface_irradiance_south_phoenix.csv
├── ventilation/
│   ├── infiltration_denver.csv
│   ├── infiltration_denver_01ach.csv
│   ├── infiltration_denver_05ach.csv
│   ├── infiltration_denver_10ach.csv
│   ├── infiltration_dulles_05ach.csv
│   ├── infiltration_miami_05ach.csv
│   ├── infiltration_minneapolis_05ach.csv
│   └── infiltration_phoenix_05ach.csv
├── weather/
│   ├── denver_tmy3_reference.csv
│   ├── miami_tmy3_reference.csv
│   ├── minneapolis_tmy3_reference.csv
│   └── phoenix_tmy3_reference.csv
├── zone_balance/
│   ├── case_600_energy_reference.csv
│   ├── case_900_energy_reference.csv
│   ├── case_920_energy_reference.csv
│   ├── case_950_energy_reference.csv
│   ├── case_960_energy_reference.csv
│   ├── case_970_energy_reference.csv
│   ├── case_920_energy_hourly.csv
│   ├── case_950_energy_hourly.csv
│   ├── case_960_energy_hourly.csv
│   └── fixed_inputs_zone_temp.csv
└── gauge/
    └── case_900_diurnal_reference.csv
```

## Test Execution

```bash
# Run all tests
cargo test --all

# Run isolation tests only
cargo test --test '*_isolation'

# Run ASHRAE 140 tests
cargo test ashrae_140

# Run with output
cargo test -- --nocapture

# Run specific integration test
cargo test --test api_integration_tests
```
