# Fluxion ASHRAE 140 Blind Validation Plan

**Goal:** Achieve ASHRAE 140 validation with true blind test methodology — no calibration factors, no case-specific corrections, physics-only model.

**Current State:** 9.4% pass rate WITH corrections. 0% without them.

**Why This Plan Exists:** The current validation is "informed" not "blind" — case IDs are known before simulation, correction factors are applied post-simulation based on case type, and benchmark ranges are "calibrated for 5R1C" rather than being true ASHRAE reference values. This plan establishes the physics-only path to genuine compliance.

---

## Part 1: SPEC

### 1.1 Definition of Done

A validation run is considered successful when ALL of the following are true:

1. **Blind execution:** The simulation engine receives only building description, weather data, and HVAC specifications. No case ID, no case-type hint, no pre-configuration based on what case it is.

2. **Zero correction factors:** All `time_constant_sensitivity_correction`, `cooling_sensitivity_correction`, `thermal_mass_correction_factor`, `adaptive_calibration`, and post-simulation empirical multipliers are set to 1.0 or removed.

3. **True reference values:** Benchmark data represents actual EnergyPlus/ESP-r/TRNSYS outputs from ASHRAE 140 standard, not "calibrated for 5R1C" adjusted ranges.

4. **Pass tolerance:** For each case and metric:
   - Annual energy: ±15% of reference mean
   - Monthly energy: ±10% of reference mean
   - Peak loads: ±15% of reference mean
   - Free-floating temperature: ±1.0°C of reference mean

5. **Coverage:** At least 80% of ASHRAE 140 standard test cases pass all tolerance bands.

### 1.2 What "Physics Only" Means

The model uses ONLY:
- First-principles thermal network (5R1C, 6R2C, or whatever topology)
- ISO 13790 solar distribution formulas
- Construction properties from ASHRAE 140 specification (no fitting to reference results)
- Weather data as provided (TMY format)
- HVAC characteristics from standard equipment curves

The model does NOT use:
- Case ID to select model configuration
- Reference results to adjust any parameter
- Empirical corrections tuned against validation cases
- Machine learning fine-tuning on ASHRAE cases
- Any form of calibration that uses validation output as feedback

### 1.3 Architecture Constraints

**Seam constraint:** There must exist a regression test seam that can reproduce the bug pattern. If no such seam exists, the finding is "codebase architecture prevents validation" — not "bug fixed."

This means:
- The blind validation must be executable via `cargo test --test ashrae_140_blind`
- Individual case failures must be traceable to specific physical model components
- The feedback loop must be deterministic (same input → same output, within floating-point tolerance)

---

## Part 2: ROADMAP

### Phase A: Baseline Stripping (Weeks 1-2)

**Goal:** Remove all corrections, confirm true baseline failure mode.

#### A.1: Catalog All Correction Infrastructure

Find and document every location where corrections are applied:

| Correction Type | Location | Current Value | Effect if Removed |
|-----------------|----------|---------------|-------------------|
| Post-simulation multipliers | `ashrae_140_validator.rs:1129-1146` | ÷4.0, ×0.35 for 900 series | Unknown - likely 0% pass |
| 6R2C time constant | `thermal_model_core.rs:327` | 5.2 | Model topology changes |
| 6R2C cooling sensitivity | `thermal_model_core.rs:328` | 1.74 | Unknown |
| Thermal mass correction | `thermal_mass.rs` | Per-case factors | Unknown |
| Adaptive calibration | `adaptive_calibration.rs` | Hourly recalibration | Unknown |
| Case-specific 6R2C config | `thermal_model_core.rs:1211-1222` | 75% envelope for 900 series | Unknown |
| Benchmark calibrated ranges | `benchmark.rs:108-110` | Adjusted for 5R1C | Different pass/fail thresholds |

#### A.2: Create Blind Validation Harness

Build `tests/ashrae_140_blind_validation.rs` that:
- Loads case definitions from data files (no case ID pre-load)
- Runs simulation with physics-only configuration
- Compares output against true reference values (not calibrated ranges)
- Reports pass/fail per metric per case

This harness must be executable with `cargo test --test ashrae_140_blind` and produce deterministic results.

#### A.3: Confirm Baseline

Run blind validation harness against current codebase WITH corrections enabled. Record:
- Which cases pass with corrections
- Which cases fail with corrections
- Magnitude of failure for each

Then run with corrections disabled. This is the true baseline.

### Phase B: Physics Fixes (Weeks 3-20)

**Goal:** Fix the thermal model so it matches reference results without corrections.

#### B.1: Solar Distribution Fix (Weeks 3-8)

**Root cause hypothesis:** The Perez sky model + ISO 13790 distribution produces wrong solar fraction to air vs mass for heavy-mass cases.

**Verification:** Compare hourly solar gain profiles for Case 900 between Fluxion and EnergyPlus. Find the specific timestep range where divergence occurs.

**Fix approach:**
1. Implement detailed sky diffuse / ground reflectance split per ISO 13790 Section 10
2. Verify beam radiation uses correct angle-of-incidence calculations
3. Check that diffuse vs beam distribution matches reference programs
4. Validate against all 900 series cases without corrections

**Deliverable:** `tests/solar_distribution_validation.rs` that compares Fluxion solar gains against EnergyPlus hourly data for at least 4 cases (600, 620, 900, 940).

#### B.2: Thermal Mass Time Constant Fix (Weeks 9-14)

**Root cause hypothesis:** The thermal time constant τ = Cm/(h_tr_ms + h_tr_em) is wrong because h_tr_ms (conductance from thermal mass to interior surface) is incorrectly calculated for heavy constructions.

**Fix approach:**
1. Implement ISO 13790 Table C.2 effective capacitance per unit area (κ values)
2. Verify h_tr_ms calculation uses actual construction layer properties
3. Check 6R2C model configuration — currently defaults to 40% of ISO 13790 h_tr_ms which is empirically calibrated
4. Derive 6R2C correction factors from first principles, not calibration

**Deliverable:** `tests/thermal_mass_time_constant_validation.rs` that verifies τ matches ISO 13790 calculated values for high-mass and low-mass constructions.

#### B.3: Free-Floating Temperature Fix (Weeks 15-20)

**Root cause hypothesis:** Free-floating cases (600FF, 900FF, etc.) have extreme temperature failures (125°C max for 900FF when reference is 41-46°C). This suggests either:
- HVAC is incorrectly stil active in "free-floating" mode
- Internal gains are being double-counted
- Thermal mass is not damping correctly

**Fix approach:**
1. Verify that free-floating mode truly disables HVAC (no heating/cooling load)
2. Check that internal gains (occupants, equipment, lighting) are correctly summed
3. Validate that thermal damping matches reference diurnal temperature swing
4. Check for numerical instability in implicit solver for long timesteps

**Deliverable:** `tests/free_floating_temperature_validation.rs` covering min/max temperature and diurnal swing for at least 4 free-floating cases.

### Phase C: Benchmark Correction (Weeks 21-24)

**Goal:** Replace "calibrated for 5R1C" benchmark ranges with true ASHRAE 140 reference values.

**Note:** This phase may reveal that the current "pass" cases were only passing because the benchmark was adjusted downward to match the broken model. This is a critical finding.

**Deliverable:** `data/ashrae_140_true_reference/` containing:
- Actual EnergyPlus output hourly data for each case
- Reference mean/std per metric per case
- Source program identification (EnergyPlus version, ESP-r version, etc.)

### Phase D: Blind Validation Pass (Weeks 25-28)

**Goal:** Run full ASHRAE 140 blind validation suite and achieve 80%+ pass rate.

**Validation suite must include:**
- All 600 series (low-mass baseline)
- All 900 series (high-mass baseline)
- 800 series (non-residential HVAC)
- 195-470 diagnostic cases
- Free-floating variants (FF suffix)
- Multi-zone cases (960, 970)

**Pass criteria:** 80%+ cases pass all tolerance bands.

### Phase E: Sustained Validation (Ongoing)

**Goal:** Maintain blind validation pass rate as code evolves.

**Mechanisms:**
- CI gate: Blind validation must pass before merge to main
- Regression tracking: Any drop below 80% triggers automatic investigation
- Annual re-validation: Run against latest ASHRAE 140 reference data

---

## Part 3: TEST APPROACH

### 3.1 Feedback Loop Construction

The primary feedback loop is `cargo test --test ashrae_140_blind`. This must:
1. Load case definitions from `data/ashrae_140_cases/` (YAML or JSON)
2. Load weather data from `data/ashrae_140_weather/`
3. Load reference values from `data/ashrae_140_true_reference/`
4. Run each case through the simulation engine with NO case-type hints
5. Compare outputs against reference tolerances
6. Report pass/fail per case per metric

**Loop execution time target:** < 5 minutes for full suite (58+ cases).

### 3.2 Instrumentation Strategy

**For solar distribution issues:**
- Tag logs with `[DEBUG-solar]` prefix
- Log hourly solar gain breakdown (beam, diffuse, ground-reflected) for diagnostic cases
- Compare against reference hourly profiles

**For thermal mass issues:**
- Tag logs with `[DEBUG-thermal-mass]`
- Log computed τ per zone per timestep
- Log h_tr_ms, h_tr_em values

**For free-floating temperature issues:**
- Tag logs with `[DEBUG-free-float]`
- Log HVAC active/inactive state
- Log internal gain totals per timestep

### 3.3 Differential Testing

For each fix, run:
1. Old version (before fix) vs new version (after fix)
2. Both against same input case
3. Diff the outputs

This confirms the fix actually changes behavior and doesn't just shift numerical results.

### 3.4 Non-Deterministic Handling

If any test exhibits >1% flakiness rate:
1. Parallelize the trigger (run 100x in parallel)
2. Narrow timing windows
3. Pin random seed
4. Isolate filesystem

Until flake rate is below 0.1%.

---

## Part 4: RISK LOG

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 5R1C topology insufficient for heavy-mass cases | High | High - may need complete model rewrite | Plan for 6R2C/8R3C alternative if 5R1C can't match reference |
| Benchmark data sourcing | Medium | High - can't validate against true references | Source from ASHRAE 140 official reference data files |
| Free-floating extreme temperatures indicate deeper bug | High | High - solver instability or HVAC logic error | Address in Phase B.3 before attempting validation |
| 6R2C corrections (5.2, 1.74) are empirically derived and can't be fixed with first principles | Medium | Medium | Document finding, evaluate if 6R2C is right topology |
| Performance regression from removing corrections | Low | Medium - may slow CI | Optimize thermal model after accuracy is confirmed |

---

## Part 5: DEPENDENCIES

**Before Phase B begins:**
- [ ] Phase A.3 baseline confirmed (know true failure magnitudes)
- [ ] Blind validation harness is deterministic and reproducible
- [ ] True reference data sourced and validated

**Before Phase D begins:**
- [ ] Solar distribution fixed and validated against reference hourly data
- [ ] Thermal mass time constant fixed and validated
- [ ] Free-floating temperature fixed and validated
- [ ] Benchmark data corrected to true ASHRAE values

---

## Part 6: PHASED ESTIMATE

| Phase | Duration | Key Deliverable | Pass Criterion |
|-------|----------|-----------------|----------------|
| A: Baseline Stripping | 2 weeks | Confirmed baseline (0% pass rate without corrections) | Know exact failure magnitudes per case |
| B.1: Solar Distribution | 6 weeks | Solar gains match reference hourly profiles | 900 series cooling within ±15% |
| B.2: Thermal Mass τ | 6 weeks | τ matches ISO 13790 calculated values | 900 series annual energy within ±20% |
| B.3: Free-Float Fix | 6 weeks | Free-float temps match reference diurnal swing | FF cases max/min within ±2°C |
| C: Benchmark Correction | 4 weeks | True reference data replaces calibrated ranges | Reference data validated against official ASHRAE sources |
| D: Blind Validation Pass | 4 weeks | Full suite pass rate ≥ 80% | 80%+ cases pass all tolerance bands |
| **Total** | **28 weeks** | — | — |

**Estimate confidence:** Low-Medium. The 0% baseline is known, but the effort to fix each physics component is uncertain. Could be 20 weeks or 40 weeks depending on how many iterations each fix requires.

**If 5R1C topology proves insufficient:** Add 8-12 weeks for topology change evaluation and implementation.

---

*This spec is a living document. Update based on findings from each phase.*