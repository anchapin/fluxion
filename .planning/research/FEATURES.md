# Feature Research

**Domain:** ASHRAE Standard 140 Validation for Building Energy Modeling (BEM) Engines
**Researched:** 2026-03-08
**Confidence:** MEDIUM (Source: Fluxion codebase analysis, official ASHRAE 140 documentation review, web search unavailable)

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Baseline test cases (600/900 series)** | ASHRAE 140 specifies 16+ baseline cases covering low/high mass construction | MEDIUM | Cases 600, 610, 620, 630, 640, 650 + 900-950 variants. Required for any credible validation. Without these, engineers cannot trust the engine. |
| **Free-floating mode** | ASHRAE 140 includes FF cases to test thermal dynamics without HVAC | LOW | Cases 600FF, 650FF, 900FF, 950FF. Tests passive thermal response, validates thermal mass dynamics. |
| **Multi-zone capability** | Real buildings have multiple zones; ASHRAE 140 Case 960 is 2-zone | MEDIUM | Case 960 (sunspace + back-zone). Tests inter-zone heat transfer, essential for multi-story buildings. |
| **Special conduction cases** | Validates envelope heat transfer independently | LOW | Case 195 (solid conduction, no windows/loads). Isolates conduction physics from solar/load effects. |
| **Annual/peak load metrics** | Core energy use metrics for building design | LOW | Annual heating/cooling (MWh), peak loads (kW). Standard output for all BEM tools. |
| **Reference range comparison** | ASHRAE 140 provides accepted ranges from EnergyPlus, ESP-r, TRNSYS | MEDIUM | Must compare results to reference ranges. Pass/fail status within ±5% tolerance is expected. |
| **Temperature metrics (FF cases)** | Free-floating cases report min/max/avg temperatures | LOW | Min/max free-floating temperatures (°C). Critical for validating thermal mass response. |
| **Weather data integration** | All cases require hourly weather data (TMY format) | HIGH | Denver TMY weather is standard. Must parse EPW format or embed weather data. |
| **Solar radiation modeling** | Solar gains are dominant heating/cooling driver | HIGH | Hourly DNI/DHI on all orientations, incidence angle effects, window transmittance. |
| **Thermostat control** | HVAC control logic determines energy use | LOW | Dual setpoints (heating <20°C, cooling >27°C), deadband control. Required for all non-FF cases. |
| **Multi-layer construction** | Buildings have layered assemblies (wall, roof, floor) | MEDIUM | Layer-by-layer R-value calculation, ASHRAE film coefficients. Tests heat transfer accuracy. |
| **Window properties** | Windows are critical envelope component | LOW | U-value, SHGC, normal transmittance, glass type (double clear, low-E). Required for all cases. |
| **Infiltration modeling** | Air leakage affects heat loss/gain significantly | LOW | Air change rate (ACH), typically 0.5 ACH for baseline cases. |
| **Internal loads** | Occupancy/equipment loads add heat | LOW | Continuous internal gains (200W typical), convective/radiative split. |
| **Ground boundary condition** | Floor conducts heat to/from ground | LOW | Constant soil temperature (10°C per ASHRAE 140 spec) or dynamic soil models. |
| **Validation report generation** | Engineers need documented results | MEDIUM | Markdown/CSV/JSON export, pass/fail summary, comparison tables. Required for audit trails. |

### Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Diagnostic logging & debugging tools** | Accelerates physics debugging by exposing internal simulation state | MEDIUM | Hourly data collection, energy breakdowns, peak timing, temperature profiles. Helps identify why cases fail. |
| **Sensitivity analysis** | Shows which parameters most affect results | HIGH | Perturbs inputs and measures output changes. Valuable for optimization and uncertainty quantification. |
| **Automated CI/CD integration** | Prevents regressions by running validation on every commit | MEDIUM | GitHub Actions integration, pass/fail thresholds, blocking merge on regressions. Ensures long-term stability. |
| **GPU-accelerated calculations** | 10-100x speedup for large population evaluations | VERY HIGH | Batch inference for surrogates, parallel solar calculations. Critical for quantum/GA optimization loops. |
| **Interactive visualization** | Makes results immediately understandable | HIGH | Real-time plots of temperature profiles, HVAC demand curves, energy breakdowns. Reduces debugging time. |
| **Delta testing** | Shows how design changes affect energy use | MEDIUM | Compare case variants (e.g., 600 vs 610 shading) to isolate individual effects. Excellent for sensitivity studies. |
| **Batch validation** | Run all cases in parallel, collect comprehensive results | MEDIUM | Execute 18+ cases simultaneously, aggregate results. Saves time for full validation suites. |
| **Extensible test case framework** | Add custom cases beyond ASHRAE 140 | MEDIUM | Builder pattern for custom specs, support additional climate zones beyond Denver. Enables domain-specific validation. |
| **Hourly CSV export** | Enables detailed post-simulation analysis | LOW | Export full hourly time series for external tools (Python, Excel). Researchers love this. |
| **Environment variable configuration** | Easy toggling of diagnostic output | LOW | `ASHRAE_140_DEBUG`, `ASHRAE_140_VERBOSE` flags. Low-friction debugging. |
| **Multi-reference comparison** | Compare to multiple reference programs simultaneously | MEDIUM | Compare results to EnergyPlus, ESP-r, TRNSYS in single report. Shows consistency across tools. |
| **Thermal mass response analysis** | Quantify thermal lag and damping | HIGH | Extract time constants, phase shifts, amplitude reduction. Valuable for high-mass building optimization. |
| **Peak load timing validation** | Verify peaks occur at correct time of year/day | LOW | Peak heating in winter, peak cooling in summer. Tests seasonality and solar timing. |
| **Component-level energy breakdown** | Show where energy goes (conduction, infiltration, solar) | MEDIUM | Envelope conduction, infiltration losses, solar gains, internal gains. Helps diagnose over/under-prediction. |
| **Free-floating temperature swing analysis** | Quantify thermal mass effectiveness | LOW | Min/max/avg temperatures, swing range. Tests passive cooling/heating potential. |
| **Regression guardrails** | Prevent performance degradation over time | MEDIUM | Track MAE, max deviation, pass rate trends. Alert on >2% regression. |
| **Neural surrogate integration** | Replace expensive physics with fast ML models | VERY HIGH | ONNX Runtime, batch inference, GPU backends. Enables 10,000+ configs/sec for optimization. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Real-time validation during simulation** | Engineers want instant feedback | Adds overhead, breaks batching, violates "collect data once" principle | Run full simulation, then validate. Use quick smoke tests for basic correctness. |
| **Adaptive timestep in validation** | "We should use smaller steps for accuracy" | Breaks reproducibility, reference programs use fixed hourly timesteps, makes debugging harder | Use fixed hourly timesteps for validation (ASHRAE 140 spec). Adaptive timesteps for optimization only. |
| **Custom tolerance bands per case** | "Some cases are harder, we should relax them" | Undermines validation credibility, reference ranges are already lenient (±5-15%), slippery slope | Use consistent tolerance (±5% of reference midpoint). Document why cases fail, don't change thresholds. |
| **Parallel test execution within CI** | "Running all 18 cases takes too long" | Increases CI flakiness, makes debugging harder, obscures which case failed | Run tests sequentially in CI, use batch validation for development. |
| **Skipping failing cases** | "We can't pass Case 900 yet, let's skip it" | Masks real physics issues, loses regression protection, violates validation purpose | Mark cases as "expected failure" with tickets, run them to track progress. |
| **Excessive diagnostic output by default** | "I want to see everything" | Slows down runs, floods CI logs, hides actual failures | Disabled by default, enable via environment variables when debugging. |
| **Reference data embedding in source code** | "Hardcoding benchmark data is easier" | Violates single-source-of-truth, hard to update, mixes data and logic | External CSV/JSON files in `benchmarks/` directory, parsed at runtime. |
| **Manual result verification** | "I'll check the numbers myself" | Not scalable, error-prone, impossible for large validation suites | Automated comparison to reference ranges, generate reports for manual review. |
| **Test case randomization** | "We should test random designs too" | Reference programs don't test random cases, makes comparison impossible | Stick to ASHRAE 140 standard cases. Use separate "exploratory validation" for random designs. |
| **Optimizing validation results** | "We can tweak parameters to pass" | Violates purpose of validation, hides physics bugs, results don't generalize | Fix physics issues causing failures, don't tune to pass tests. |
| **Over-compliance reporting** | "Let's show we're better than reference" | Misleading, reference is a range not a target, encourages overfitting | Report deviation from reference midpoint, show pass/fail status clearly. |

## Feature Dependencies

```
[Baseline 600/900 Cases]
    └──requires──> [Weather Data Integration]
                    └──requires──> [TMY EPW Parsing]
    └──requires──> [Solar Radiation Modeling]
                    └──requires──> [Solar Position Calculation]
                    └──requires──> [Surface Insolation Model]
    └──requires──> [Multi-layer Construction]
                    └──requires──> [Layer R-Value Calculator]
                    └──requires──> [ASHRAE Film Coefficients]
    └──requires──> [Thermostat Control]
                    └──requires──> [Dual Setpoint Logic]
                    └──requires──> [Deadband Control]
    └──requires──> [Window Properties]
                    └──requires──> [U-value Calculation]
                    └──requires──> [SHGC Application]
    └──requires──> [Infiltration Modeling]
    └──requires──> [Internal Loads]
    └──requires──> [Ground Boundary Condition]

[Shading Cases (610, 630, 910, 930)]
    └──requires──> [Baseline Cases]
    └──requires──> [Overhang Geometry]
    └──requires──> [Shade Fin Geometry]
    └──requires──> [Shadow Projection Calculation]
    └──enhances──> [Solar Radiation Modeling]

[Thermostat Setback Cases (640, 940)]
    └──requires──> [Baseline Cases]
    └──requires──> [HVAC Schedule System]
    └──requires──> [Time-based Schedules]

[Night Ventilation Cases (650, 950)]
    └──requires──> [Baseline Cases]
    └──requires──> [HVAC Schedule System]
    └──requires──> [Fan Capacity Model]
    └──enhances──> [Infiltration Modeling]

[Free-Floating Cases (FF series)]
    └──requires──> [Baseline Cases]
    └──enhances──> [Thermostat Control]
    └──enhances──> [Temperature Profile Analysis]

[Multi-Zone Case (960)]
    └──requires──> [Baseline Cases]
    └──requires──> [Inter-Zone Heat Transfer]
    └──requires──> [Zone Coupling Calculation]
    └──enhances──> [Thermal Model]

[Solid Conduction Case (195)]
    └──requires──> [Baseline Cases]
    └──enhances──> [Construction Physics]

[Validation Report Generation]
    └──requires──> [All Test Cases]
    └──requires──> [Reference Range Comparison]
    └──requires──> [Status Determination]

[Diagnostic Logging]
    └──requires──> [Test Cases]
    └──enhances──> [Validation Report Generation]

[Sensitivity Analysis]
    └──requires──> [All Test Cases]
    └──requires──> [Batch Validation]
    └──enhances──> [Diagnostic Logging]

[Automated CI/CD Integration]
    └──requires──> [Validation Report Generation]
    └──requires──> [Regression Guardrails]

[GPU Acceleration]
    └──enhances──> [Solar Radiation Modeling]
    └──enhances──> [Neural Surrogate Integration]

[Interactive Visualization]
    └──requires──> [Hourly Data Collection]
    └──requires──> [Diagnostic Logging]
    └──enhances──> [Validation Report Generation]

[Delta Testing]
    └──requires──> [All Test Cases]
    └──requires──> [Batch Validation]
    └──enhances──> [Sensitivity Analysis]
```

### Dependency Notes

- **Baseline Cases require Weather Data Integration**: All cases need hourly TMY weather data. Without this, no validation is possible.
- **Shading Cases enhance Solar Radiation Modeling**: Shading adds complexity (overhangs, fins) but builds on the solar model used in baseline cases.
- **Thermostat Setback requires HVAC Schedule System**: Time-based schedules (0700-2300h heating, 2300-0700h setback) are more complex than constant setpoints.
- **Night Ventilation enhances Infiltration Modeling**: Adds fan capacity (1703 m³/h) to infiltration, requires dynamic ACH calculation.
- **Free-Floating enhances Thermostat Control**: HVAC is disabled, but thermostat logic still exists (just unused). Simpler than controlled cases.
- **Multi-Zone enhances Thermal Model**: Requires inter-zone conductance and zone coupling, but most 5R1C physics is unchanged.
- **Diagnostic Logging enhances Validation Report Generation**: Optional detailed output that adds context to pass/fail results.
- **Sensitivity Analysis requires Batch Validation**: Running multiple perturbations needs efficient batch execution.
- **GPU Acceleration enhances Multiple Components**: Parallelizes solar calculations and surrogate inference, but doesn't change validation logic.
- **Interactive Visualization requires Diagnostic Logging**: Needs hourly data collection to plot time series.

## MVP Definition

### Launch With (v1)

Minimum viable product — what's needed to validate the concept.

- [x] **Baseline test cases (600/900 series)** — Core ASHRAE 140 validation, without these the engine cannot be trusted
- [x] **Free-floating mode** — Tests thermal mass dynamics independently of HVAC
- [x] **Multi-zone capability** — Case 960 validates inter-zone heat transfer
- [x] **Special conduction cases** — Case 195 isolates conduction physics
- [x] **Annual/peak load metrics** — Core energy use metrics required for all BEM tools
- [x] **Reference range comparison** — Compare to EnergyPlus/ESP-r/TRNSYS, determine pass/fail
- [x] **Temperature metrics (FF cases)** — Min/max free-floating temperatures for thermal mass validation
- [x] **Weather data integration** — Denver TMY weather in EPW format
- [x] **Solar radiation modeling** — Hourly DNI/DHI, incidence angle effects, window transmittance
- [x] **Thermostat control** — Dual setpoints, deadband control
- [x] **Multi-layer construction** — Layer-by-layer R-value calculation, ASHRAE film coefficients
- [x] **Window properties** — U-value, SHGC, normal transmittance, glass type
- [x] **Infiltration modeling** — Air change rate (ACH), typical 0.5 ACH
- [x] **Internal loads** — Continuous internal gains (200W), convective/radiative split
- [x] **Ground boundary condition** — Constant soil temperature (10°C)
- [x] **Validation report generation** — Markdown/CSV export, pass/fail summary

### Add After Validation (v1.x)

Features to add once core is working.

- [ ] **Diagnostic logging & debugging tools** — Hourly data collection, energy breakdowns, peak timing. Accelerates physics debugging.
- [ ] **Automated CI/CD integration** — GitHub Actions, pass/fail thresholds, blocking merge on regressions. Prevents drift.
- [ ] **Batch validation** — Run all cases in parallel, aggregate results. Saves time for full validation suites.
- [ ] **Hourly CSV export** — Export full hourly time series for external analysis. Researchers love this.
- [ ] **Environment variable configuration** — Easy toggling of diagnostic output. Low-friction debugging.
- [ ] **Peak load timing validation** — Verify peaks occur at correct time of year/day. Tests seasonality.
- [ ] **Component-level energy breakdown** — Show where energy goes (conduction, infiltration, solar). Diagnoses over/under-prediction.
- [ ] **Free-floating temperature swing analysis** — Quantify thermal mass effectiveness. Tests passive potential.
- [ ] **Regression guardrails** — Track MAE, max deviation, pass rate trends. Alert on >2% regression.

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] **Sensitivity analysis** — Shows which parameters most affect results. Valuable for optimization and uncertainty quantification.
- [ ] **GPU-accelerated calculations** — 10-100x speedup for large population evaluations. Critical for quantum/GA optimization loops.
- [ ] **Interactive visualization** — Makes results immediately understandable. Real-time plots of temperature profiles, HVAC demand curves.
- [ ] **Delta testing** — Compare case variants to isolate individual effects. Excellent for sensitivity studies.
- [ ] **Extensible test case framework** — Add custom cases beyond ASHRAE 140. Builder pattern for custom specs.
- [ ] **Multi-reference comparison** — Compare to multiple reference programs simultaneously. Shows consistency across tools.
- [ ] **Thermal mass response analysis** — Quantify thermal lag and damping. Valuable for high-mass building optimization.
- [ ] **Neural surrogate integration** — Replace expensive physics with fast ML models. ONNX Runtime, batch inference, GPU backends.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Baseline test cases (600/900) | HIGH | MEDIUM | P1 |
| Weather data integration | HIGH | HIGH | P1 |
| Solar radiation modeling | HIGH | HIGH | P1 |
| Thermostat control | HIGH | LOW | P1 |
| Multi-layer construction | HIGH | MEDIUM | P1 |
| Validation report generation | HIGH | MEDIUM | P1 |
| Reference range comparison | HIGH | MEDIUM | P1 |
| Free-floating mode | MEDIUM | LOW | P1 |
| Multi-zone capability | MEDIUM | MEDIUM | P1 |
| Special conduction cases | MEDIUM | LOW | P1 |
| Window properties | MEDIUM | LOW | P1 |
| Infiltration modeling | MEDIUM | LOW | P1 |
| Internal loads | MEDIUM | LOW | P1 |
| Ground boundary condition | MEDIUM | LOW | P1 |
| Temperature metrics (FF) | MEDIUM | LOW | P1 |
| Diagnostic logging | HIGH | MEDIUM | P2 |
| Automated CI/CD integration | MEDIUM | MEDIUM | P2 |
| Batch validation | MEDIUM | MEDIUM | P2 |
| Hourly CSV export | MEDIUM | LOW | P2 |
| Peak load timing validation | MEDIUM | LOW | P2 |
| Component-level energy breakdown | MEDIUM | MEDIUM | P2 |
| Regression guardrails | MEDIUM | LOW | P2 |
| Sensitivity analysis | HIGH | HIGH | P3 |
| GPU-accelerated calculations | HIGH | VERY HIGH | P3 |
| Interactive visualization | HIGH | HIGH | P3 |
| Delta testing | MEDIUM | MEDIUM | P3 |
| Extensible test case framework | LOW | MEDIUM | P3 |
| Multi-reference comparison | LOW | MEDIUM | P3 |
| Thermal mass response analysis | LOW | HIGH | P3 |
| Neural surrogate integration | HIGH | VERY HIGH | P3 |

**Priority key:**
- P1: Must have for launch (ASHRAE 140 compliance)
- P2: Should have, add when possible (enhances usability)
- P3: Nice to have, future consideration (competitive advantages)

## Competitor Feature Analysis

| Feature | EnergyPlus | ESP-r | TRNSYS | Our Approach |
|---------|-----------|-------|--------|--------------|
| Baseline 600/900 cases | ✅ Full | ✅ Full | ✅ Full | ✅ Full (18+ cases) |
| Weather data integration | ✅ EPW + TMY3 | ✅ EPW + TMY2 | ✅ Custom formats | ✅ EPW (Denver embedded) |
| Solar radiation model | ✅ Perez, ASHRAE | ✅ Perez, Hay | ✅ Custom | ✅ Isotropic sky model |
| Multi-layer construction | ✅ Full | ✅ Full | ✅ Full | ✅ Layer-by-layer R-value |
| Thermostat control | ✅ Advanced | ✅ Advanced | ✅ Advanced | ✅ Dual setpoint, deadband |
| Free-floating mode | ✅ | ✅ | ✅ | ✅ FF series (4 cases) |
| Multi-zone | ✅ Unlimited | ✅ Unlimited | ✅ Unlimited | ✅ 2-zone (Case 960) |
| Diagnostic logging | ✅ Limited | ✅ Research tools | ✅ Custom | ✅ Comprehensive (hourly, breakdowns, timing) |
| GPU acceleration | ❌ | ❌ | ❌ | ✅ ONNX Runtime with GPU backends |
| Neural surrogates | ❌ | ❌ | ❌ | ✅ ONNX Runtime integration |
| Batch validation | ❌ | ❌ | ❌ | ✅ Parallel execution with rayon |
| CI/CD integration | ❌ Manual | ❌ Manual | ❌ Manual | ✅ GitHub Actions, blocking merges |
| Interactive visualization | ✅ OpenStudio | ❌ | ✅ TRNSYS Studio | ❌ Future (v2+) |
| Sensitivity analysis | ✅ Plugins | ✅ Research | ✅ Custom | ❌ Future (v2+) |
| Delta testing | ❌ | ❌ | ❌ | ❌ Future (v2+) |

**Key Differentiators:**
- **GPU acceleration & neural surrogates**: Fluxion uniquely integrates ONNX Runtime with GPU backends for 100x speedup
- **Comprehensive diagnostics**: More detailed logging than EnergyPlus/ESP-r, with hourly data, energy breakdowns, and peak timing
- **Automated CI/CD**: Native GitHub Actions integration prevents regressions (competitors require manual validation)
- **Batch validation**: Parallel execution of all 18+ cases with rayon (competitors run sequentially)

## Sources

- Fluxion codebase analysis: `/home/alex/Projects/fluxion/` (comprehensive review of validation implementation)
- Fluxion ASHRAE 140 documentation:
  - `docs/ASHRAE140_VALIDATION.md` - Test case overview and validation process
  - `docs/ASHRAE140_RESULTS.md` - Current validation results and known issues
  - `docs/ASHRAE140_MVP_ROADMAP.md` - Implementation roadmap and gap analysis
  - `docs/ASHRAE_140_5R1C_MODEL.md` - Thermal network model details
  - `docs/ASHRAE_140_DIAGNOSTICS.md` - Diagnostic features and debugging tools
- Fluxion validation implementation:
  - `src/validation/ashrae_140_cases.rs` - Test case definitions and specifications
  - `src/validation/ashrae_140_validator.rs` - Validation framework and execution
  - `tests/ashrae_140_validation.rs` - Integration tests
- Web search: Tool returned empty results for all ASHRAE 140 queries, limiting external validation. Research based on internal codebase analysis and documentation (MEDIUM confidence).

---
*Feature research for: ASHRAE Standard 140 Validation for Building Energy Modeling (BEM) Engines*
*Researched: 2026-03-08*
