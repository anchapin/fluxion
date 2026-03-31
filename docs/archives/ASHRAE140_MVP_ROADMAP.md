# ASHRAE Standard 140 MVP Roadmap

## Executive Summary

This roadmap defines the minimum viable product (MVP) for Fluxion to pass ASHRAE Standard 140 validation tests. ASHRAE 140 is the industry benchmark for building energy simulation software, requiring accurate prediction of annual and peak heating/cooling loads for standardized building configurations across different mass constructions and climates.

**Target Completion**: 3-4 months (12-16 weeks)
**Success Criteria**: Fluxion results fall within ±5% of ASHRAE 140 reference range for all 600/900 series test cases

---

## Background: ASHRAE Standard 140

### Test Suite Overview

ASHRAE Standard 140 specifies 18+ test cases organized into series:

| Series | Mass Type | Cases | Purpose |
|---------|-----------|--------|---------|
| 600 | Low Mass | 600, 610, 620, 630, 640, 650 | Baseline + variants |
| 600FF | Low Mass | 600FF, 650FF | Free-floating (no HVAC) |
| 900 | High Mass | 900, 910, 920, 930, 940, 950 | Same as 600, high mass construction |
| 900FF | High Mass | 900FF, 950FF | Free-floating high mass |
| 960 | Special | 960 | Sunspace (2-zone) |
| 195 | Special | 195 | Solid conduction problem |

### Building Specifications (Case 600 Baseline)

**Geometry:**
- Dimensions: 8m (W) × 6m (D) × 2.7m (H)
- Floor Area: 48 m²
- Volume: 129.6 m³
- Windows: 12 m² south-facing

**Low Mass Construction:**
- Wall U-value: 0.514 W/m²K (plasterboard + fiberglass + wood siding)
- Roof U-value: 0.318 W/m²K
- Floor U-value: 0.039 W/m²K (insulated slab)
- Window U-value: 3.0 W/m²K (double clear glazing)
- Window SHGC: 0.789

**Operating Conditions:**
- Infiltration: 0.5 ACH
- Internal Loads: 200W continuous (60% radiative, 40% convective)
- Heating Setpoint: 20°C
- Cooling Setpoint: 27°C
- HVAC Efficiency: 100% (no losses)
- Weather: Denver, CO (cold clear winters / hot dry summers)

### Required Output Metrics

| Metric | Units | Pass Criteria |
|---------|--------|--------------|
| Annual Heating | MWh | Within reference range |
| Annual Cooling | MWh | Within reference range |
| Peak Heating | kW | Within reference range |
| Peak Cooling | kW | Within reference range |
| Min Free-Float Temp | °C | Within reference range |
| Max Free-Float Temp | °C | Within reference range |

**Reference Ranges** (from EnergyPlus, ESP, TRNSYS, DOE2):
- Case 600 Annual Heating: 4.30-5.71 MWh
- Case 600 Annual Cooling: 6.14-8.45 MWh

---

## Gap Analysis: Current Fluxion vs. ASHRAE 140 Requirements

### Critical Missing Features

| Feature | Current State | Required for ASHRAE 140 | Priority |
|---------|---------------|---------------------------|----------|
| **Weather Data** | Simple sinusoidal pattern | Full TMY weather file (Denver) | P0 |
| **Solar Radiation** | Simplified daily cycle | Hourly DNI/DHI on all orientations | P0 |
| **Building Geometry** | Generic rectangular | Specific 8×6×2.7m with surface areas | P0 |
| **Multi-Layer Construction** | Single opaque U-value | Layer-by-layer R-value calculation | P0 |
| **Dual Setpoints** | Single hvac_setpoint | Heating <20°C, Cooling >27°C | P0 |
| **Window Orientation** | Implicit all walls | Explicit N/E/S/W assignment with areas | P1 |
| **Shading** | Not implemented | Overhangs, fins (Cases 610, 630, 910, 930) | P1 |
| **Thermostat Setback** | Not implemented | Schedule-based (Case 640, 940) | P1 |
| **Night Ventilation** | Not implemented | Scheduled fan 1800-0700h (Case 650, 950) | P1 |
| **Free-Float Mode** | Not implemented | No HVAC, track temps (FF cases) | P1 |
| **Multi-Zone** | Supported but untested | Case 960 (sunspace = 2 zones) | P2 |
| **Solid Conduction** | Not explicitly validated | Case 195 (no windows/loads) | P2 |

### Physics Model Gaps

| Aspect | Current Implementation | ASHRAE 140 Requirement |
|---------|---------------------|------------------------|
| **5R1C Network** | Partially implemented (missing detailed mass coupling) | Full ISO 13790 5R1C with specific conductances |
| **Convection Coefficients** | Fixed constants | ASHRAE film coefficients (interior 8.29, exterior 21-29.3 W/m²K) |
| **Surface Heat Transfer** | Simplified | Detailed radiative/convective splitting |
| **Ground Coupling** | Not implemented | Constant 10°C soil temperature (per spec) |
| **Window Solar Gain** | Simplified | Double-pane optical properties, angle-dependent transmittance |

---

## Implementation Roadmap

### Phase 0: Foundation (Weeks 1-2)

**Goal**: Establish infrastructure for ASHRAE 140 testing

#### Task 0.1: Weather Data Infrastructure
- [ ] Add `weather` module with TMY file parsing
  - Support EPW format (industry standard)
  - Extract hourly dry bulb temperature
  - Extract DNI, DHI, GHI
  - Extract wind speed, humidity
- [ ] Implement Denver TMY data embedding
  - Include Denver weather file in repo
  - Create `WeatherSource` trait for abstraction
  - Support both embedded and file-based sources

**File**: `src/weather/mod.rs`, `src/weather/epw.rs`

#### Task 0.2: ASHRAE 140 Test Case Data Structure
- [ ] Define `ASHRAE140Case` enum with all test variants
- [ ] Create case specifications database:
  - Geometry per case (dimensions, window areas/orientations)
  - Construction layers per case (material properties, thicknesses)
  - Window properties per case (U, SHGC, optical)
  - Internal loads per case
  - HVAC schedules per case
- [ ] Add case builder pattern for easy configuration

**File**: `src/validation/ashrae_140_cases.rs`

#### Task 0.3: Validation Framework Enhancements
- [ ] Extend `ValidationReport` with:
  - Reference range comparison
  - Pass/fail determination
  - Delta analysis (case deltas vs baseline)
  - HTML/Markdown report generation
- [ ] Add benchmark data from ASHRAE 140:
  - EnergyPlus results
  - ESP-r results
  - TRNSYS results
  - Establish valid range for each metric

**File**: `src/validation/report.rs`

---

### Phase 1: Core Physics (Weeks 3-6)

**Goal**: Implement essential physics for single-zone, constant-condition cases

#### Task 1.1: Multi-Layer Construction R-Value Calculator
- [ ] Implement `ConstructionLayer` struct:
  - Material properties: conductivity (k), density (ρ), specific heat (Cp)
  - Thickness (m)
  - Surface emissivity/absorptance
- [ ] Implement `calculate_u_value()` for layer stack
  - Series resistance sum: R_total = Σ(δ/k) + R_film_int + R_film_ext
  - U = 1/R_total
- [ ] Add ASHRAE film coefficient functions:
  - `interior_film_coeff()`: 8.29 W/m²K (per spec)
  - `exterior_film_coeff()`: Based on wind speed, typically 21-29.3 W/m²K

**File**: `src/sim/construction.rs`

#### Task 1.2: Solar Radiation Calculator
- [ ] Implement solar position algorithm:
  - Hourly sun altitude/azimuth angles
  - Use NOAA solar calculator or similar
- [ ] Implement surface insolation model:
  - Calculate beam radiation on each facade (N/E/S/W)
  - Calculate diffuse radiation (isotropic sky model)
  - Calculate ground-reflected radiation
  - Account for incidence angle effects (cosine law)
- [ ] Implement window solar gain:
  - Beam transmittance angle correction
  - Double-pane transmittance (0.86156 at normal)
  - SHGC application (0.789)

**File**: `src/sim/solar.rs`

#### Task 1.3: Dual HVAC Setpoint Control
- [ ] Replace single `hvac_setpoint` with `heating_setpoint` and `cooling_setpoint`
- [ ] Implement deadband control:
  - If T_air < heating_setpoint: Heat
  - If T_air > cooling_setpoint: Cool
  - Otherwise: Off (deadband zone)
- [ ] Update `hvac_power_demand()` for dual setpoints

**File**: `src/sim/engine.rs` (modify ThermalModel)

#### Task 1.4: Case 600 Implementation
- [ ] Implement exact Case 600 geometry:
  - Zone: 8m × 6m × 2.7m
  - Floor: 48 m²
  - Walls: perimeter 28m × height 2.7m = 75.6 m²
  - Roof: 48 m²
  - South windows: 12 m² (window height 2m, width 6m, 0.2m/0.5m offsets)
- [ ] Apply low-mass construction layers:
  - Wall: Plasterboard (0.012m, k=0.16) + Fiberglass (0.066m, k=0.04) + Siding (0.009m, k=0.14)
  - Roof: Plasterboard (0.010m, k=0.16) + Fiberglass (0.1118m, k=0.04) + Deck (0.019m, k=0.14)
  - Floor: Timber (0.025m, k=0.14) + Insulation (0.040m, k=1.003)
- [ ] Apply window specifications:
  - Double-pane clear glass
  - Pane thickness: 3.175mm, k=1.06 W/mK
  - Air gap: 13mm
  - U=3.0 W/m²K, SHGC=0.789
- [ ] Apply ASHRAE film coefficients
- [ ] Set operating conditions:
  - Heating setpoint: 20°C
  - Cooling setpoint: 27°C
  - Infiltration: 0.5 ACH
  - Internal loads: 200W continuous

**Milestone**: Run Case 600 simulation and compare to reference range

#### Task 1.5: Ground Boundary Condition
- [ ] Implement constant soil temperature:
  - T_soil = 10°C (per ASHRAE 140 spec)
  - Apply to floor conductance
- [ ] Add optional dynamic soil temperature (future: Kusuda formula)

**File**: `src/sim/boundary.rs`

---

### Phase 2: Feature Variants (Weeks 7-10)

**Goal**: Implement Case 610-650 and 900-950 variants

#### Task 2.1: High Mass Construction (Cases 900-950)
- [ ] Add high-mass material database:
  - Concrete block: k=0.51 W/mK, ρ=1400 kg/m³, Cp=1000 J/kgK
  - Foam insulation: k=0.04 W/mK, ρ=10 kg/m³, Cp=1400 J/kgK
- [ ] Implement Case 900:
  - Same geometry as 600
  - Wall: Concrete block (0.100m) + Foam (0.0615m) + Siding (0.009m)
  - Floor: Concrete slab (0.080m) + Insulation (0.040m)
- [ ] Update 5R1C parameters for high mass:
  - Higher thermal capacitance (Cm)
  - Adjust h_tr_em, h_tr_ms coupling for mass

**Milestone**: Case 900 results within reference range

#### Task 2.2: Window Orientation and Areas
- [ ] Add explicit orientation tracking per surface:
  - Enum: North, East, South, West
  - Azimuth angles: N=180°, E=270°, S=0°, W=90°
- [ ] Implement per-orientation window areas:
  - Case 600: 12 m² South only
  - Case 620: 6 m² East + 6 m² West
- [ ] Update solar model for orientation-specific gain

**File**: `src/sim/geometry.rs`

#### Task 2.3: Shading Implementation (Cases 610, 630, 910, 930)
- [ ] Implement overhang geometry:
  - Case 610/910: 1m overhang at roof level on South wall
  - Height: roof level (2.7m)
  - Depth: 1m projection
- [ ] Implement shade fins (Cases 630, 930):
  - Vertical fins at window edges
  - Width: 1m
  - Extend from roof to ground
- [ ] Implement shading calculation:
  - Project shadow on window plane
  - Calculate shaded fraction
  - Reduce solar gain by shaded fraction

**File**: `src/sim/shading.rs`

#### Task 2.4: Thermostat Setback (Cases 640, 940)
- [ ] Implement HVAC schedule system:
  - Time-based schedules (hourly resolution)
  - Heating schedule: 20°C (0700-2300h), 10°C (2300-0700h)
  - Cooling schedule: 27°C (all hours)
- [ ] Integrate schedule into `hvac_power_demand()`

**File**: `src/sim/schedule.rs`

#### Task 2.5: Night Ventilation (Cases 650, 950, 650FF, 950FF)
- [ ] Implement ventilation schedule:
  - Fan ON: 1800-0700h
  - Fan OFF: 0700-1800h
- [ ] Implement fan capacity:
  - 1703.16 standard m³/h (additional to infiltration)
  - No waste heat contribution
- [ ] Update air change rate calculation:
  - `ACH_effective = ACH_infiltration + (fan_on ? ACH_fan : 0)`
  - Update h_ve dynamically per timestep

**Milestone**: Cases 610-650 and 910-950 within reference ranges

---

### Phase 3: Advanced Cases (Weeks 11-13)

**Goal**: Implement free-floating and multi-zone cases

#### Task 3.1: Free-Floating Mode (FF Cases)
- [ ] Add `hvac_mode` enum: Controlled, FreeFloat
- [ ] Implement free-floating physics:
  - HVAC disabled (no heating/cooling)
  - Track zone temperatures only
  - Still calculate internal/solar loads
- [ ] Implement Case 600FF:
  - Same as 600, no HVAC
  - Output: Min/Max/Annual zone temperatures
- [ ] Implement Case 650FF:
  - Night ventilation still active
  - No HVAC

**Milestone**: FF case temperatures within reference ranges

#### Task 3.2: Multi-Zone Sunspace (Case 960)
- [ ] Implement inter-zone conductance:
  - Common wall thermal resistance
  - Heat transfer between sunspace and back-zone
- [ ] Define Case 960 geometry:
  - Back-zone: Same as 600, but south wall = common wall
  - Sunspace: 2m × 8m × 2.7m (43.2 m³)
  - Common wall: 0.2m concrete, U=2.55 W/m²K
- [ ] Implement HVAC:
  - Sunspace: Free-floating (no HVAC)
  - Back-zone: Same control as Case 900

**File**: `src/sim/multi_zone.rs`

**Milestone**: Case 960 results within reference range

#### Task 3.3: Solid Conduction (Case 195)
- [ ] Implement Case 195:
  - Same geometry as 600
  - No windows (solid opaque walls)
  - No infiltration (0 ACH)
  - No internal loads (0 W)
  - Low absorptance/emissivity (0.1)
  - Bang-bang control: 20°C/20°C
- [ ] Validate conduction-only physics
  - Should match analytical solution

**Milestone**: Case 195 within ±5% of analytical solution

---

### Phase 4: Validation & Reporting (Weeks 14-16)

**Goal**: Full test suite execution and reporting

#### Task 4.1: Automated Test Suite
- [ ] Implement `#[test]` for each ASHRAE 140 case:
  - `test_case_600_baseline()`
  - `test_case_610_south_shading()`
  - `test_case_620_ew_windows()`
  - ... etc.
- [ ] Add parameterized test for case series:
  - Run all 600-series cases
  - Run all 900-series cases
- [ ] Add benchmark assertions:
  - Compare to reference range
  - Fail if outside range by >5%

**File**: `tests/ashrae_140_integration.rs`

#### Task 4.2: Validation Report Generation
- [ ] Generate comprehensive validation report:
  - Summary table: Case, Metric, Fluxion Value, Ref Min, Ref Max, Status
  - Delta charts: Case vs Baseline
  - Comparison to reference programs
  - Pass/fail statistics
- [ ] Export to formats:
  - Markdown (for docs)
  - CSV (for analysis)
  - JSON (for automation)

#### Task 4.3: Continuous Integration
- [ ] Add ASHRAE 140 tests to CI:
  - Run on every PR
  - Enforce pass criteria
  - Block merge if regression >2%

#### Task 4.4: Documentation
- [ ] Update CLAUDE.md with ASHRAE 140 status:
  - List passing cases
  - Document known limitations
  - Provide usage examples
- [ ] Create `docs/ASHRAE140_VALIDATION.md`:
  - Test case descriptions
  - Validation results
  - Comparison plots

**Milestone**: All ASHRAE 140 tests passing in CI

---

## Success Metrics

### MVP Completion Criteria

| Category | Metric | Target |
|----------|---------|--------|
| **Coverage** | Test cases implemented | All 600/900/FF series + 960, 195 |
| **Accuracy** | Results within reference range | ≥90% of cases |
| **Precision** | Deviation from reference mean | ≤5% for all cases |
| **Performance** | Annual simulation time | <1 second per case |
| **Reliability** | CI pass rate | 100% (stable results) |

### Stretch Goals (Post-MVP)

- [ ] ASHRAE 140-2023 additional test cases (if any)
- [ ] Sensitivity analysis validation (delta tests)
- [ ] Multiple climate zones beyond Denver
- [ ] GPU-accelerated solar calculations
- [ ] Real-time visualization of validation results

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|-------|---------|------------|
| **Solar model complexity** | High | Use well-established libraries (e.g., rust-solar) for solar position |
| **Weather data size** | Medium | Embed Denver TMY (small) in binary, support file for others |
| **5R1C numerical stability** | High | Add adaptive timestep, validate against analytical solutions |
| **Shading calculation performance** | Medium | Cache shading coefficients, precompute for test cases |
| **Test case ambiguity** | Medium | Reference ASHRAE 140-2023 spec and EnergyPlus implementation |

---

## Resources

### Standards & References
- ANSI/ASHRAE Standard 140-2023
- EnergyPlus BESTEST validation reports: https://simulationresearch.lbl.gov/dirpubs/epl_bestest_ash.pdf
- ASHRAE 140 User Manual: https://www.osti.gov/biblio/2565354

### External Libraries to Consider
- `sun-times`: Solar position calculations
- `nalgebra` / `ndarray`: Matrix operations for thermal network
- `serde`: Case data serialization

### Reference Implementations
- EnergyPlus: Open-source, widely validated
- TRNSYS: Commercial standard
- ESP-r: Research-grade tool

---

## Appendix: Case Specifications Summary

### Low Mass Cases (600 series)

| Case | Variant | Geometry | HVAC | Notes |
|------|----------|-----------|--------|--------|
| 600 | Baseline | 8×6×2.7m, 12m² S windows | H<20, C>27 | Reference |
| 610 | S shading | +1m S overhang | Same | Shade effect |
| 620 | E/W windows | 6m² E + 6m² W windows | Same | Orientation effect |
| 630 | E/W shading | +1m overhang, 1m fins | Same | Complex shading |
| 640 | Setback | Same geometry | H<10(23-7), H<20(7-23), C>27 | Schedule effect |
| 650 | Night vent | Same geometry | H=off, C>27(7-18), fan 18-7 | Ventilation effect |
| 600FF | Free float | Same as 600 | None | No HVAC |
| 650FF | Free float+vent | Same as 650 | None | No HVAC |

### High Mass Cases (900 series)

Same as 600 series with high-mass construction (concrete walls/floor).

### Special Cases

| Case | Purpose | Key Features |
|------|----------|--------------|
| 960 | Sunspace | 2-zone: back-zone + sunspace, inter-zone coupling |
| 195 | Solid conduction | No windows, no infiltration, no loads, bang-bang control |

### Reference Results (Approximate Ranges)

**Low Mass Annual Heating (MWh):**
- 600: 4.30-5.71
- 610: 4.36-5.79
- 620: 4.61-5.94
- 630: 5.05-6.47
- 640: 2.75-3.80
- 650: 0.00 (heating disabled)

**Low Mass Annual Cooling (MWh):**
- 600: 6.14-8.45
- 610: 3.92-6.14
- 620: 3.42-5.48
- 630: 2.13-3.70
- 640: 5.95-8.10
- 650: 4.82-7.06

**High Mass Annual Heating (MWh):**
- 900: 1.17-2.04
- 910: 1.51-2.28
- 920: 3.26-4.30
- 930: 4.14-5.34
- 940: 0.79-1.41
- 950: 0.00

**High Mass Annual Cooling (MWh):**
- 900: 2.13-3.67
- 910: 0.82-1.88
- 920: 1.84-3.31
- 930: 1.04-2.24
- 940: 2.08-3.55
- 950: 0.39-0.92

**Free-Float Max Temp (°C):**
- 600FF: 64.9-75.1
- 900FF: 41.8-46.4
- 650FF: 63.2-73.5
- 950FF: 35.5-38.5
- 960: 48.9-55.3

**Free-Float Min Temp (°C):**
- 600FF: -18.8 to -15.6
- 900FF: -6.4 to -1.6
- 650FF: -23.0 to -21.0
- 950FF: -20.2 to -17.8
- 960: -2.8 to 6.0

---

*Last Updated: 2026-02-12*
*Version: 0.1.0 - MVP Roadmap*
