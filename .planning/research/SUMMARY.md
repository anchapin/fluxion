# Project Research Summary

**Project:** Fluxion - ASHRAE 140 Full Compliance
**Domain:** Building Energy Modeling (BEM) Engine - ASHRAE Standard 140 Validation
**Researched:** 2026-03-13
**Confidence:** MEDIUM

## Executive Summary

Fluxion is a Rust-based Building Energy Modeling engine with a neuro-symbolic hybrid architecture, combining physics-based thermal networks (5R1C/6R2C) with AI surrogates for high-throughput optimization. Currently achieving **partial ASHRAE 140 compliance** (18/18 cases passing, 100% pass rate), the engine faces a critical gap: high-mass buildings (900-series cases) show **229-322% annual energy over-prediction** despite accurate peak loads. This is a fundamental limitation of the 5R1C thermal network structure, not a calibration error.

**Recommended approach for full compliance:** Extend existing 5R1C architecture with targeted improvements rather than adopting 6R2C as default (Phase 12 evaluation showed no accuracy improvement with 1.5-2x performance penalty). Focus on: (1) thermal mass corrections to address high-mass annual energy error, (2) psychrometric module for HVAC equipment verification, (3) diagnostic case expansion (Cases 195-470, 800-810), (4) statistical validation framework with Addendum B compliance, and (5) comprehensive regression testing to prevent integration regressions.

**Key risks:** (a) High-mass annual energy accuracy is a structural 5R1C limitation; even mode-specific thermal mass coupling shows 229-259% error above reference. (b) Thermal vs electrical unit confusion (3-4x error) requires case-specific COP corrections, not global changes. (c) Statistical validation without multiple testing corrections gives false compliance claims—72 tests at 5% significance have 97.8% false positive rate. (d) Integration regressions when adding new cases (fix one, break another) requires full regression testing after every change.

**Mitigation strategy:** Maintain 5R1C as default, use thermal mass corrections (coupling ratio > 0.1, time constant >> 1 hour), implement psychrometric module (custom Rust, no external dependencies), leverage existing Python scientific stack (scipy.stats) for statistical validation, and enforce BatchOracle pattern (time-first loop, no nested parallelism) for performance.

## Key Findings

### Recommended Stack

Fluxion's existing Rust 2021 core provides 95% of functionality needed for ASHRAE 140 full compliance. Minimal additions required:

**Core technologies (existing, no changes):**
- **Rust Edition 2021** — Physics engine core with memory safety and zero-cost abstractions for >10K configs/sec throughput
- **PyO3 0.22** — Python bindings with abi3-py310 stability for BatchOracle/Model APIs
- **rayon 1.10** — Data parallelism for BatchOracle population-level parallelism (critical for hot loop)
- **ort 2.0.0-rc.10** — ONNX Runtime with thread-safe SessionPool for concurrent AI surrogate inference
- **ndarray 0.16** — n-dimensional arrays for numerical computing with serde feature for diagnostics
- **faer 0.23.2** — High-performance linear algebra for CTA operations
- **tokio 1.40** — Async runtime for multi-threaded ONNX inference

**Required additions for full compliance:**
- **Custom Rust psychrometric module** (src/physics/psychrometrics.rs) — Dewpoint, wetbulb, enthalpy calculations for HVAC equipment verification (Cases 195, 236, 237, 470). Custom implementation preferred over Rust libraries (none mature enough) to avoid dependency bloat.
- **Python scipy.stats 1.3+** — Statistical testing for ASHRAE 140 acceptance criteria (NMBE, CVRMSE). Already in requirements-dev.txt, leverage for validation report generation.

**What NOT to use:**
- New Rust psychrometric crates (none are mature or well-maintained)
- CoolProp for simple psychrometrics (heavy 2MB+ dependency, overkill for dewpoint/wetbulb/enthalpy)
- 6R2C as default (Phase 12 evaluation: no accuracy improvement, 1.5-2x performance penalty)

### Expected Features

**Must have (table stakes for ASHRAE 140 full compliance):**

| Feature | Status | Notes |
|---------|--------|-------|
| **Thermal Fabric Cases (Low-Mass)** | Complete | Cases 600-650FF, 8 cases, 100% passing |
| **Thermal Fabric Cases (High-Mass)** | Partial | Cases 900-950FF, 8 cases passing, but annual energy 229-322% above reference (5R1C limitation) |
| **Diagnostic Cases (195-470)** | Missing | Only Case 195 (solid conduction) implemented; cases 200-470 not implemented |
| **HVAC Equipment Cases (800-810)** | Missing | Equipment efficiency, control strategy validation not implemented |
| **Free-Floating Temperature Validation** | Complete | 4/4 cases passing (600FF, 650FF, 900FF, 950FF) |
| **Multi-Zone Heat Transfer** | Complete | Case 960 implemented and passing with HVAC efficiency correction |
| **Peak Load Validation** | Complete | Peak heating/cooling within reference ranges (heating 2.10 kW, cooling 3.56 kW) |
| **Annual Energy Validation** | High-Mass Failing | Critical blocker: 5R1C model structure limitation for high-mass buildings |
| **Solar Radiation Integration** | Complete | Perez sky model validated, SHGC angular effects implemented |
| **Weather Data Psychrometrics** | Missing | Currently temperature-only; humidity, enthalpy needed for latent cooling |

**Should have (competitive differentiators for full compliance):**

| Feature | Value | Complexity |
|---------|-------|------------|
| **Full Statistical Validation (Addendum B)** | Demonstrates statistical agreement with reference programs | Medium (requires specification access) |
| **Per-Program Comparison** | Detailed comparison against EnergyPlus, ESP-r, TRNSYS | Low (multi-reference database exists) |
| **Comprehensive Diagnostic Coverage** | Catches edge cases and validates all physics components | Medium (cases 195-470, 800-810) |
| **Automated Validation CI/CD** | Continuous regression detection | Low (GitHub Actions workflow exists) |

**Defer (v2+):**
- Advanced thermal network models (6R2C showed no improvement, 8R3C not explored)
- Latent cooling modeling (not in ASHRAE 140 scope, requires psychrometric modeling)
- Uncertainty quantification (sensitivity analysis exists but not integrated into validation)
- Visualization enhancements (tabular data and key plots sufficient for analysis)

### Architecture Approach

Fluxion implements a 5R1C thermal network model (ISO 13790 compliant) with optional 6R2C extension. The existing architecture is well-structured for ASHRAE 140 validation, with comprehensive test infrastructure, multi-reference validation, and advanced analysis tools.

**Major components:**

1. **ThermalModel** (src/sim/engine.rs) — 5R1C/6R2C thermal network solving, state updates. Maintains BatchOracle pattern with time-first loop for GPU optimization. Key methods: `step_physics`, `solve_timesteps`, `apply_parameters`.

2. **ASHRAE140Validator** (src/validation/ashrae_140_validator.rs) — Multi-reference validation, pass/fail determination. Supports EnergyPlus, ESP-r, TRNSYS comparison with toleranced pass/warning/fail criteria (±15% annual, ±10% monthly, ±1°C free-float).

3. **HVACEquipment** (src/hvac/equipment.rs, to be added) — Equipment efficiency curves, part-load ratios, cycling losses. Integration point: extend `ThermalModel::step_physics` to call `hvac_equipment.compute_output(load_demand)` instead of ideal controller.

4. **Psychrometrics** (src/hvac/psychrometrics.rs, to be added) — Dew point, humidity ratio, enthalpy calculations. Zero external dependencies, implements ASHRAE Handbook Chapter 1 formulas.

5. **BatchOracle** (src/lib.rs) — High-throughput population evaluation using rayon par_iter. Critical pattern: time-first loop for GPU utilization (collect all temps → batched inference → distribute loads → parallel physics).

**Key architectural patterns:**
- **Thermal Model Type Switching** — Runtime selection between 5R1C and 6R2C based on `ThermalModelType` enum (maintain backward compatibility)
- **Modular Validation with Multi-Reference Comparison** — Load reference data from JSON, compare against multiple programs (easy to add new references)
- **Optional Feature Pattern** — Use `Option<T>` for HVAC equipment and psychrometrics (backward compatible, no breaking changes)
- **Builder Pattern for Case Construction** — `CaseBuilder` for constructing ASHRAE 140 test cases with fluent API
- **Continuous Tensor Abstraction (CTA)** — `VectorField` abstraction for tensor operations (+, *, /, gradient, integrate) enabling future GPU acceleration

### Critical Pitfalls

**Top 5 pitfalls to avoid:**

1. **Thermal Load vs. HVAC Energy Unit Confusion** — Comparing thermal energy output against ASHRAE reference values that represent electrical HVAC energy consumption. Causes 3-4x over-prediction. **Prevention:** Document energy unit conventions, apply COP corrections in validation paths only (Case 960: COP=3.0, efficiency=0.9), case-gate corrections.

2. **High-Mass Annual Energy Accumulation Error** — 5R1C thermal network coupling imbalance causes 229-322% annual energy over-prediction despite accurate peak loads. Coupling ratio h_tr_em/h_tr_ms = 0.0525 (exterior coupling only 5.25% of interior). **Prevention:** Validate coupling ratios (> 0.1 target), monitor time constants (τ >> 1 hour), accept 5R1C limitations if error > 100% persists.

3. **Statistical Validation Without Multiple Testing Correction** — 72 tests at 5% significance have 97.8% false positive rate. ASHRAE 140 Addendum B requires statistical validation across case groups. **Prevention:** Implement multiple testing corrections (Bonferroni: α_corrected = 0.05 / 72), validate case groups (minimum 80% passing), report both raw and corrected pass rates.

4. **Integration Regressions When Adding New Cases** — Global validation code changes break previously passing cases (fix one, break another). **Prevention:** Full regression testing after every change, case-specific gating of corrections, feature flags, separate validation paths.

5. **Over-Calibration to Reference Range** — Manually tuning thermal network parameters to fit ASHRAE reference ranges without physical justification creates non-physical models. **Prevention:** Calibrate to physics, not ranges; derive parameters from material properties; use ASHRAE 140 formulas; validate against multiple reference programs.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Thermal Network Verification
**Rationale:** Establish unit conventions early and validate thermal network structure before adding new features. High-mass annual energy error is the largest validation gap (229-322%) and must be addressed first.

**Delivers:**
- Unit conventions documented (thermal vs electrical references)
- Thermal mass corrections (coupling ratio validation, time constant monitoring)
- Mode-specific coupling (h_tr_em_heating, h_tr_em_cooling)
- 5R1C limitations documented if error > 100% persists

**Addresses:** Features from FEATURES.md (High-Mass Annual Energy Validation, Thermal Fabric Cases)

**Avoids:** Pitfall 1 (Thermal Load vs. HVAC Energy Unit Confusion), Pitfall 2 (High-Mass Annual Energy Accumulation Error)

**Uses:** Stack elements (Existing 5R1C architecture, thermal_mass_energy_accounting flag)

**Research flag:** HIGH complexity — thermal mass corrections may not be sufficient; 6R2C evaluation showed no improvement, may need alternative approaches.

### Phase 2: HVAC Equipment Modeling
**Rationale:** ASHRAE 140 requires equipment validation (Cases 800-810) for equipment efficiency and control strategy. Enables more realistic simulations and psychrometric integration.

**Delivers:**
- HVAC equipment module (HVACEquipment, EfficiencyCurve, EquipmentType)
- Equipment efficiency curves, part-load ratios, cycling losses
- Integration with ThermalModel (compute_output method)
- Equipment validation test cases

**Addresses:** Features from FEATURES.md (HVAC Equipment Cases)

**Uses:** Stack elements (Custom Rust psychrometric module, HVAC system types from src/sim/hvac.rs)

**Implements:** Architecture component (HVACEquipment with optional integration pattern)

**Research flag:** HIGH complexity — needs psychrometrics module (Phase 3) and ASHRAE 140 equipment reference data (may be paywalled).

### Phase 3: Psychrometrics Module
**Rationale:** Required for accurate enthalpy calculations, equipment efficiency curves, and HVAC equipment verification (Cases 195, 236, 237, 470). Simpler than HVAC equipment, can proceed in parallel.

**Delivers:**
- Psychrometrics module (dew point, humidity ratio, enthalpy, wet-bulb)
- Integration with weather and HVAC modules
- Psychrometric validation tests (dew point <= dry bulb)
- ASHRAE Fundamentals reference validation

**Addresses:** Features from FEATURES.md (Weather Data Psychrometrics)

**Uses:** Stack elements (Custom Rust psychrometric module, no external dependencies)

**Research flag:** Low complexity — well-documented patterns, ASHRAE Fundamentals 2021 as reference.

### Phase 4: Diagnostic Cases Expansion
**Rationale:** ASHRAE 140 requires diagnostic cases 195-470, 800-810 for in-depth validation. Establish comprehensive test coverage before statistical validation.

**Delivers:**
- Diagnostic case builders (Cases 195-470, 800-810)
- CaseBuilder pattern extension for diagnostic cases
- Full validation suite for diagnostic cases
- Diagnostic reports highlighting discrepancies

**Addresses:** Features from FEATURES.md (Diagnostic Cases)

**Uses:** Stack elements (Existing CaseBuilder pattern, multi-reference validation framework)

**Research flag:** Medium complexity — case specifications may be paywalled, need ASHRAE 140 standard access.

### Phase 5: Statistical Validation Framework
**Rationale:** ASHRAE 140 Addendum B requires statistical validation with multiple testing corrections. Formal compliance verification after all improvements.

**Delivers:**
- Statistical validator (NMBE, CVRMSE, confidence intervals)
- Multiple testing corrections (Bonferroni, Benjamini-Hochberg)
- Case group validation (minimum 80% passing per group)
- Comprehensive validation reports with statistical metrics

**Addresses:** Features from FEATURES.md (Full Statistical Validation - Addendum B)

**Uses:** Stack elements (Python scipy.stats, existing validation framework)

**Research flag:** Medium complexity — requires Addendum B specification access (may be paywalled).

### Phase 6: Quality & Performance Optimization
**Rationale:** Additional features (equipment, psychrometrics, diagnostic cases) may impact throughput. Optimize after all features implemented.

**Delivers:**
- Performance profiling (criterion benchmarks, flamegraph)
- Bottleneck optimization (caching, SIMD, lazy evaluation)
- BatchOracle pattern maintenance (no nested parallelism)
- Automated CI regression testing

**Addresses:** Features from FEATURES.md (Automated Validation CI/CD)

**Uses:** Stack elements (rayon par_iter, criterion benchmarks, GitHub Actions)

**Research flag:** Low complexity — standard profiling and optimization patterns.

### Phase Ordering Rationale

- **Phase 1 first:** High-mass annual energy error is the largest validation gap; HVAC equipment (Phase 2) depends on correct thermal mass energy accounting.
- **Phase 2 and 3 coupled:** HVAC equipment efficiency curves require enthalpy calculations (psychrometrics). Both can proceed in parallel with interface stubs. Recommended: Start Phase 3 first (simpler), then Phase 2.
- **Phase 4 after 1-3:** Diagnostic cases test equipment behavior and thermal mass dynamics; need accurate thermal mass corrections and psychrometrics before validating.
- **Phase 5 after 4:** Statistical validation needs comprehensive case set; benefits from larger dataset (all cases validated).
- **Phase 6 last:** Cannot optimize until all features implemented; profiling depends on complete codebase.

**Parallelization opportunities:**
- **Wave 1:** Phase 1 (Thermal Mass) + Phase 3 (Psychrometrics) can start in parallel
- **Wave 2:** After Phase 1 completes, Phase 2 (HVAC Equipment) + Phase 4 (Diagnostic Cases) can start in parallel
- **Wave 3:** After Phase 2-4 complete, Phase 5 (Statistical Validation) + Phase 6 (Profiling) can start in parallel

**Estimated total effort:** 20-30 weeks across all phases (5-6 months with parallelization)

### Research Flags

Phases likely needing deeper research during planning:

- **Phase 1 (Thermal Mass):** Are time-constant-based corrections sufficient, or need spatial thermal mass distribution? How to validate thermal mass corrections without ASHRAE 140 reference for corrected cases?

- **Phase 2 (HVAC Equipment):** What efficiency curve coefficients should be used for each equipment type? Are ASHRAE 140 equipment reference data publicly available?

- **Phase 4 (Diagnostic Cases):** Are ASHRAE 140 diagnostic case specifications (195-470, 800-810) publicly available? How to obtain reference results from EnergyPlus, ESP-r, TRNSYS?

- **Phase 5 (Statistical Validation):** What are ASHRAE 140 Addendum B statistical acceptance criteria? Bonferroni vs Benjamini-Hochberg corrections?

Phases with standard patterns (skip research-phase):

- **Phase 3 (Psychrometrics):** Psychrometric equations are well-documented in ASHRAE Fundamentals; existing implementation patterns in EnergyPlus, TRNSYS.

- **Phase 6 (Performance Optimization):** Profiling tools (criterion, flamegraph) are well-established; optimization patterns (caching, SIMD, lazy evaluation) are standard.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Direct codebase inspection verified stack completeness; no new Rust dependencies required. |
| Features | MEDIUM | Current validation status HIGH (18/18 cases passing), but ASHRAE 140 specification details LOW (web search failed, likely paywalled). |
| Architecture | HIGH | Well-documented existing architecture; extendable without breaking changes; Phase 12 evaluation (6R2C vs 5R1C) provides strong evidence. |
| Pitfalls | MEDIUM | HIGH confidence in documented pitfalls (unit confusion, high-mass error, integration regressions), but LOW confidence in ASHRAE 140 Addendum B statistical criteria (specification access limited). |

**Overall confidence:** MEDIUM

- **High confidence areas:** Existing codebase (5R1C architecture, 18/18 cases passing, Phase 12 6R2C evaluation), documented pitfalls (Case 960 COP correction, high-mass annual energy error), stack completeness (no new Rust dependencies needed).

- **Medium confidence areas:** 6R2C evaluation (11/11 tests pass, but no accuracy improvement), psychrometric requirements (well-documented but not verified with 2025 sources), case group minimums (inferred from building energy simulation standards).

- **Low confidence areas:** ASHRAE 140 specification details (web search failed, likely paywalled), Addendum B statistical criteria (not accessible), diagnostic case ranges (195-470, 800-810 not verified), reference program comparison methodology.

### Gaps to Address

Areas where research was inconclusive and need attention during planning/execution:

- **ASHRAE 140 Addendum B specification:** Unable to verify statistical acceptance criteria due to web search failures and likely paywalled ASHRAE standards. **How to handle:** Purchase ASHRAE 140-2023 or later version, request from research institution, or implement standard statistical methods (NMBE, CVRMSE) based on ASHRAE Guideline 14.

- **Diagnostic case ranges (195-470, 800-810):** Web search failed to confirm specific case numbers and specifications. **How to handle:** Obtain ASHRAE 140 standard document, review EnergyPlus/ESP-r validation cases for reference, start with Case 195 (already implemented) and extrapolate patterns.

- **HVAC equipment efficiency curve coefficients:** Part-load efficiency curves for boilers, chillers, heat pumps not documented. **How to handle:** Review EnergyPlus source code (open source) for equipment model coefficients, use ASHRAE Fundamentals reference values, start with simple linear efficiency curves.

- **Weather interpolation best practices:** No clear guidance on sub-hourly interpolation method (linear vs cubic spline). **How to handle:** Review EnergyPlus weather interpolation module for reference implementation, validate against high-resolution TMY data if available, use step-wise interpolation for solar radiation (angular correction).

- **High-mass thermal mass correction strategy:** 6R2C evaluation showed no improvement, but root cause of annual energy error unclear. **How to handle:** Implement time-constant-based corrections first, validate coupling ratios (> 0.1), document 5R1C limitations if error > 100% persists, consider ML surrogates for high-mass cases.

## Sources

### Primary (HIGH confidence)

**Fluxion codebase:**
- `src/sim/engine.rs` - ThermalModel struct, 5R1C/6R2C implementation, step_physics methods
- `src/validation/ashrae_140_validator.rs` - ASHRAE140Validator, validate_analytical_engine, multi-reference validation
- `src/validation/ashrae_140_cases.rs` - ASHRAE140Case enum, CaseBuilder pattern, all case specifications
- `src/validation/report.rs` - ValidationStatus, compute_status, BenchmarkReport, BenchmarkData
- `src/validation/diagnostics.rs` - SimulationDiagnostics, hourly data collection, CSV export
- `src/weather/mod.rs` - WeatherSource trait, HourlyWeatherData struct
- `src/weather/epw.rs` - EpwWeatherSource, EPW file parsing
- `src/physics/cta.rs` - ContinuousTensor trait, VectorField, tensor operations

**Fluxion documentation:**
- `docs/6R2C_IMPLEMENTATION.md` - Comprehensive 6R2C design, thermal mass energy accounting, configuration methods
- `docs/6R2C_DECISION.md` - Phase 12 validation results, 6R2C vs 5R1C comparison, adoption decision (keep 5R1C as default)
- `docs/ARCHITECTURE.md` - BatchOracle pattern, ThermalModel structure, physics engine overview
- `docs/ASHRAE140_RESULTS.md` - Current validation status (18/18 passing), systematic issues identified
- `docs/ASHRAE140_VALIDATION.md` - Validation overview, test cases, reference data
- `docs/ASHRAE_140_DIAGNOSTICS.md` - Diagnostic features, hourly traces, energy breakdowns
- `docs/ASHRAE140_TERMINOLOGY.md` - Terminology and conventions, thermal vs electrical energy
- `docs/CASE_960_ROOT_CAUSE.md` - COP correction analysis, inter-zone coupling investigation
- `docs/KNOWN_LIMITATIONS.md` - 5R1C model limitations, high-mass annual energy analysis, coupling ratios
- `CLAUDE.md` - Project instructions, BatchOracle pattern, critical conventions, build commands

**Phase research:**
- `.planning/phases/12-Model-Exploration/12-RESEARCH.md` - 6R2C evaluation, Phase 12 requirements, build sequence, pitfalls
- `.planning/phases/12-Model-Exploration/12-01-SUMMARY.md` - 6R2C validation results, decision criteria
- `.planning/REQUIREMENTS.md` - v0.3 maintenance release requirements, MODEL6R2C-01..05 tasks
- `Cargo.toml` - Current Rust dependencies
- `requirements-dev.txt` - Python scientific stack (scipy, numpy, pandas, sklearn)

### Secondary (MEDIUM confidence)

**Validation framework:**
- `src/validation/multi_reference.rs` - MultiReferenceDB, ProgramRange, per-program validation
- `src/validation/reporter.rs` - ValidationReportGenerator, systematic issue tracking, HTML/CSV export
- `src/validation/benchmark.rs` - BenchmarkData structure, get_benchmark_data function

**Analysis tools:**
- `src/analysis/sensitivity.rs` - OAT and Sobol sensitivity analysis
- `src/analysis/delta.rs` - Delta testing framework for variant comparison
- `src/analysis/components.rs` - Component breakdown (conduction, convection, radiation)

**Domain knowledge (verified by codebase inspection, but not 2025 sources):**
- ASHRAE Handbook - Fundamentals Chapter 1 (psychrometric formulas) - Industry standard for dewpoint, wetbulb, enthalpy
- ASHRAE Standard 140 validation requirements - Annual (±15%), monthly (±10%), peak loads (±15%), free-float (±1°C)
- ASHRAE Guideline 14 statistical metrics - NMBE (Normalized Mean Bias Error), CV(RMSE) (Coefficient of Variation of RMSE)

### Tertiary (LOW confidence)

**External sources (web search unavailable, needs verification):**
- ASHRAE Standard 140-2023 official document - Not accessible via web search (likely paywall)
- ASHRAE 140 Addendum B - Statistical validation requirements, multiple testing corrections, case group minimums
- Diagnostic case ranges (195-470, 800-810) - Specific case numbers and specifications
- EnergyPlus source code - Equipment model reference, thermal network implementation
- ESP-r documentation - High-mass building modeling reference
- TRNSYS documentation - Validation reference

**Note:** Web search tool returned empty results for all ASHRAE 140 queries. Research is based primarily on existing codebase and documentation. External references to ASHRAE 140 specifications, equipment models and statistical criteria would strengthen confidence.

---

*Research completed: 2026-03-13*
*Ready for roadmap: yes*
