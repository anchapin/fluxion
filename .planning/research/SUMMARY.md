# Project Research Summary

**Project:** Fluxion - Building Energy Modeling (BEM) Validation Framework
**Domain:** ASHRAE Standard 140 Validation for Building Energy Modeling Engines
**Researched:** 2026-03-08
**Confidence:** MEDIUM

## Executive Summary

Fluxion is a Rust-based Building Energy Modeling (BEM) engine with a neuro-symbolic hybrid architecture that combines physics-based thermal networks with AI surrogates for 100x speedups. The project's immediate focus is ASHRAE Standard 140 validation—a critical credibility milestone for any BEM engine. Expert-built validation frameworks follow a layered architecture with clear separation between test case definitions, simulation execution, result comparison, and reporting. The recommended approach uses the ISO 13790-compliant 5R1C thermal network model, Rust for performance-critical physics calculations, PyO3 for Python bindings, and ONNX Runtime for neural surrogate inference. Key differentiators include GPU-accelerated batch inference, comprehensive diagnostic logging, and automated CI/CD integration—features that competitors like EnergyPlus, ESP-r, and TRNSYS lack.

The research reveals critical implementation risks that commonly derail BEM engines. The most dangerous pitfalls include incorrect 5R1C conductance parameterization (causing systematic heating load over-prediction), HVAC load calculation errors in setpoint control, and thermal mass dynamics mishandling in high-mass cases. Fluxion's current validation status shows these exact issues: 61% of cases failing with 78.79% mean absolute error and 471.66% max deviation. Prevention requires rigorous unit testing of individual 5R1C conductance paths, validation of HVAC setpoint control against analytical solutions, and progressive case testing from simple (lightweight Case 600) to complex (high-mass Case 900) building configurations. The research strongly suggests fixing these foundational physics issues before adding advanced features like GPU acceleration or neural surrogates.

## Key Findings

### Recommended Stack

Rust-based validation framework with Python bindings for accessibility and ONNX Runtime for AI surrogate integration. The stack prioritizes performance, safety, and ecosystem maturity for numerical computing.

**Core technologies:**
- **Rust 2021 Edition** — Physics engine core providing memory safety and zero-cost abstractions for >10K evaluations/second throughput
- **PyO3 0.22** — Python bindings with abi3-py310 compatibility for stable BatchOracle API pattern
- **rayon 1.10** — Data parallelism for population-level parallelism without nested parallelism anti-patterns
- **ort (ONNX Runtime) 2.0.0-rc.10** — Thread-safe session pool for concurrent AI surrogate inference with GPU backends
- **tokio 1.40** — Production-grade async runtime for concurrent ONNX inference operations
- **ndarray 0.16** — Numerical computing with serde feature for efficient diagnostic output serialization
- **faer 0.23.2** — High-performance linear algebra optimized for scientific computing

**Testing infrastructure:**
- **pytest** — Python test framework for ASHRAE 140 case generator tests
- **cargo test** — Rust unit/integration tests with parallel execution
- **criterion 0.5** — Performance regression testing for maintaining <100μs per-config latency

### Expected Features

ASHRAE 140 validation requires baseline test cases, comprehensive physics modeling, and robust reporting capabilities. Features fall into three categories based on user expectations.

**Must have (table stakes):**
- **Baseline test cases (600/900 series)** — ASHRAE 140 specifies 16+ baseline cases covering low/high mass construction required for any credible validation
- **Free-floating mode** — Tests thermal dynamics without HVAC, validating thermal mass response
- **Multi-zone capability** — Case 960 validates inter-zone heat transfer for multi-story buildings
- **Weather data integration** — Denver TMY weather in EPW format for all cases
- **Solar radiation modeling** — Hourly DNI/DHI, incidence angle effects, window transmittance (dominant heating/cooling driver)
- **Thermostat control** — Dual setpoints (heating <20°C, cooling >27°C) with deadband control
- **Validation report generation** — Markdown/CSV/JSON export with pass/fail summary for audit trails

**Should have (competitive):**
- **Diagnostic logging & debugging tools** — Hourly data collection, energy breakdowns, peak timing accelerates physics debugging
- **Automated CI/CD integration** — GitHub Actions with pass/fail thresholds prevents regressions (competitors require manual validation)
- **Batch validation** — Parallel execution of 18+ cases with rayon saves time for full validation suites
- **GPU-accelerated calculations** — 10-100x speedup for large population evaluations critical for quantum/GA optimization
- **Neural surrogate integration** — ONNX Runtime with batch inference enables 10,000+ configs/sec for optimization

**Defer (v2+):**
- **Sensitivity analysis** — Shows parameter impact, valuable for optimization but not essential for validation
- **Interactive visualization** — Real-time plots enhance usability but don't affect validation accuracy
- **Delta testing** — Compare case variants to isolate effects, useful for sensitivity studies
- **Multi-reference comparison** — Compare to multiple reference programs, shows tool consistency
- **Thermal mass response analysis** — Quantify thermal lag and damping, valuable for high-mass building optimization

### Architecture Approach

Layered architecture with clear separation between validation framework, physics engine, and mathematical foundations. The validator orchestrates test execution, the physics engine simulates building thermal behavior, and the CTA layer provides tensor-like operations for future GPU acceleration.

**Major components:**
1. **ASHRAE140Validator** — Core validation logic coordinating test execution and result collection with diagnostic support
2. **Case Specifications (CaseSpec)** — Defines building geometry, materials, HVAC, weather, and control strategies using builder pattern
3. **BenchmarkData** — Reference values from EnergyPlus, ESP-r, TRNSYS, DOE2 with min/max ranges per metric
4. **ThermalModel (5R1C)** — ISO 13790-compliant thermal network simulation with ideal HVAC control
5. **DiagnosticCollector** — Captures detailed simulation data (hourly temperatures, loads, energy breakdowns) for debugging
6. **Comparison Engine** — Tolerance-based validation with Pass/Warning/Fail status (±5% tolerance band)
7. **Report Generator** — Produces human-readable validation summaries in Markdown/JSON/CSV formats

**Architectural patterns:**
- **Builder Pattern** — Flexible construction of test case specifications with optional parameters
- **Diagnostic Collector** — Event-driven data collection with configurable output granularity
- **Tolerance-Based Validation** — Standard ASHRAE 140 methodology with configurable tolerance bands

### Critical Pitfalls

ASHRAE 140 validation presents systematic challenges that commonly derail BEM engine development. The top pitfalls align with Fluxion's current validation failures.

1. **Incorrect 5R1C Conductance Parameterization** — Mix conductance vs resistance units, incorrect window U-value application to h_tr_em and h_tr_w, missing thermal bridge effects. Prevent by unit testing each conductance independently and validating against ASHRAE 140 Case 600 where conductances are well-documented.

2. **HVAC Load Calculation Errors in Setpoint Control** — Wrong temperature for load calculation (Ti vs Ti_free), applying heating instead of cooling, incorrect load sign convention. Prevent by implementing unit tests for HVAC control logic, separating Ti and Ti_free in code, and validating with analytical steady-state cases.

3. **Thermal Mass Dynamics Mishandling (High-Mass Cases)** — Wrong thermal mass capacitance value, incorrect time step integration method, missing coupling between Ti and Tm. Prevent by using implicit/semi-implicit integration, validating thermal mass coupling, and testing with lightweight cases (Case 600) before attempting high-mass (Case 900).

4. **Solar Radiation and External Boundary Condition Errors** — Incorrect beam/diffuse solar decomposition, wrong solar incidence angle, missing shading, incorrect external convection coefficient. Prevent by validating solar gain calculations against reference values, implementing proper shading calculations, and checking external convection against ASHRAE fundamentals handbook correlations.

5. **Incorrect Peak Load Timing and Identification** — Using instantaneous load instead of hourly average, identifying peak at wrong time, not accounting for thermal mass lag. Prevent by separating heating and cooling peaks, validating peak timing against reference, and checking peak units (kW not W).

## Implications for Roadmap

Based on research, suggested phase structure follows physics complexity from simple to complex, addressing foundational issues before adding advanced features.

### Phase 1: Foundation and Core Validation
**Rationale:** Address critical 5R1C conductance and HVAC load calculation errors causing 61% failure rate and 78.79% MAE. Fix foundational physics before adding complexity.
**Delivers:** Working 5R1C thermal network with correct conductance parameterization, HVAC setpoint control, and baseline validation passing lightweight cases
**Addresses:** Baseline test cases (600/900 series), free-floating mode, thermostat control, validation report generation
**Avoids:** Incorrect 5R1C conductance parameterization (Pitfall 1), HVAC load calculation errors (Pitfall 2)
**Features from FEATURES.md:** P1 features (baseline cases, weather data, solar modeling, thermostat control, validation reports)

### Phase 2: Thermal Mass and Complex Physics
**Rationale:** High-mass cases (900 series) are failing due to thermal mass dynamics errors. Address after lightweight cases pass to validate mass-air coupling.
**Delivers:** Thermal mass integration with implicit/semi-implicit method, correct mass-air coupling (h_tr_em, h_tr_ms), high-mass case validation
**Uses:** faer linear algebra for mass matrix operations, ndarray for thermal mass temperature arrays
**Implements:** Thermal mass validation helpers, multi-layer construction validation
**Addresses:** High-mass cases (900, 910, 920, 930, 940, 950), thermal mass response analysis
**Avoids:** Thermal mass dynamics mishandling (Pitfall 3)
**Features from FEATURES.md:** High-mass case validation, thermal mass response analysis

### Phase 3: Solar and External Boundaries
**Rationale:** Cooling load under-prediction and peak cooling errors suggest solar radiation issues. Address after thermal mass is working correctly.
**Delivers:** Correct solar gain calculations, proper beam/diffuse decomposition, shading geometry support, external convection validation
**Uses:** CTA VectorField for solar radiation distribution, tokio async for concurrent solar calculations
**Implements:** Solar radiation modeling with incidence angle effects, shading calculations (overhangs, fins)
**Addresses:** Shading cases (610, 630, 910, 930), peak cooling loads, solar gain validation
**Avoids:** Solar radiation and external boundary condition errors (Pitfall 4)
**Features from FEATURES.md:** Shading cases, peak cooling load validation, component-level energy breakdown

### Phase 4: Multi-Zone and Advanced Cases
**Rationale:** Multi-zone cases require inter-zone heat transfer. Address after single-zone physics is validated.
**Delivers:** Inter-zone heat transfer calculation, zone coupling, multi-zone validation (Case 960)
**Uses:** rayon for parallel zone execution, ndarray for multi-zone state management
**Implements:** Inter-zone conductance validation, zone temperature gradient checks
**Addresses:** Multi-zone capability (Case 960), inter-zone heat transfer
**Avoids:** Inter-zone heat transfer errors (Pitfall 5)
**Features from FEATURES.md:** Multi-zone capability, advanced case validation

### Phase 5: Diagnostic and Developer Experience
**Rationale:** After physics is correct, add diagnostic tools to accelerate debugging and improve developer productivity.
**Delivers:** Comprehensive diagnostic logging, hourly CSV export, environment variable configuration, debugging tools
**Uses:** tempfile for test artifact management, serde_json for diagnostic data serialization
**Implements:** DiagnosticCollector with configurable output, hourly data collection, energy breakdown tracking
**Addresses:** Diagnostic logging, batch validation, hourly CSV export, environment variable configuration
**Features from FEATURES.md:** P2 features (diagnostic logging, batch validation, hourly CSV export)

### Phase 6: Performance and Scale
**Rationale:** Once validation is accurate, optimize performance for high-throughput optimization use cases (10,000+ configs/second).
**Delivers:** GPU-accelerated calculations, neural surrogate integration, batch inference optimization, regression guardrails
**Uses:** ort (ONNX Runtime) with CUDA/CoreML backends, rayon for population-level parallelism, tokio for async inference
**Implements:** SessionPool for concurrent ONNX inference, GPU kernel optimization, performance benchmarking
**Addresses:** GPU acceleration, neural surrogate integration, performance regression testing
**Features from FEATURES.md:** P3 features (GPU acceleration, neural surrogates, batch validation at scale)

### Phase 7: Advanced Analysis and Visualization
**Rationale:** Add advanced features for research and sensitivity analysis after core validation and performance are solid.
**Delivers:** Sensitivity analysis, delta testing, interactive visualization, extensible test case framework
**Uses:** matplotlib/seaborn for visualization, scikit-learn for sensitivity metrics, pandas for data analysis
**Implements:** Parameter perturbation studies, case variant comparison, real-time plotting, custom case builder
**Addresses:** Sensitivity analysis, delta testing, interactive visualization, extensible framework
**Features from FEATURES.md:** P3 features (sensitivity analysis, delta testing, visualization, multi-reference comparison)

### Phase Ordering Rationale

- **Physics-first approach:** Phases 1-4 address physics accuracy (conductances, HVAC, mass, solar, multi-zone) before optimization (Phases 6-7). This prevents optimizing incorrect physics.
- **Complexity gradient:** Start with simple lightweight cases (Phase 1), add thermal mass complexity (Phase 2), then external boundaries (Phase 3), then multi-zone coupling (Phase 4). Each phase builds on the previous.
- **Developer experience:** Diagnostic tools (Phase 5) come after physics is correct but before optimization, ensuring tools help debug correct behavior.
- **Performance last:** GPU acceleration and neural surrogates (Phase 6) are deferred until validation is accurate. Optimizing incorrect physics wastes effort.
- **Research deferred:** Advanced analysis (Phase 7) requires accurate physics and performant simulation, making it the final phase.

**Pitfall avoidance:**
- Phase 1 directly addresses Pitfalls 1 and 2 (conductances, HVAC loads)
- Phase 2 addresses Pitfall 3 (thermal mass dynamics)
- Phase 3 addresses Pitfall 4 (solar and external boundaries)
- Phase 4 addresses Pitfall 5 (inter-zone heat transfer)
- Later phases focus on usability and performance, avoiding the critical physics pitfalls

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3 (Solar and External Boundaries):** Solar radiation modeling with shading geometry requires research into beam/diffuse decomposition algorithms, incidence angle calculations, and ASHRAE film coefficient correlations. Sparse documentation in open-source tools.
- **Phase 6 (Performance and Scale):** GPU-accelerated thermal calculations and ONNX Runtime integration need research into CUDA kernel optimization, memory management strategies, and async inference patterns. Domain-specific with limited open-source examples.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Foundation):** Well-documented 5R1C thermal network and HVAC control patterns in ASHRAE 140 standard and ISO 13790. Extensive Fluxion codebase examples.
- **Phase 2 (Thermal Mass):** Standard implicit/semi-implicit integration methods for thermal mass. Numerical analysis literature provides proven approaches.
- **Phase 4 (Multi-Zone):** Multi-zone heat transfer uses established coupled differential equation solvers. Case 960 reference values are well-documented.
- **Phase 5 (Diagnostics):** Diagnostic logging and CSV export follow standard Rust patterns. serde and tempfile libraries have comprehensive documentation.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | Based on Fluxion codebase analysis (HIGH) and current dependency versions (HIGH). Web search limitations prevented verification with 2025 sources. |
| Features | MEDIUM | Fluxion codebase and ASHRAE 140 documentation provide strong evidence. Competitor analysis based on general BEM knowledge without recent verification. |
| Architecture | MEDIUM | Layered architecture validated against Fluxion codebase. Patterns (builder, diagnostic collector, tolerance-based) are well-documented in existing implementation. |
| Pitfalls | HIGH | Pitfalls directly inferred from Fluxion's validation failures (61% failing, 78.79% MAE). Strong correlation between predicted pitfalls and actual issues. |

**Overall confidence:** MEDIUM

### Gaps to Address

- **ONNX Runtime version verification:** Current recommendation (2.0.0-rc.10) is a release candidate. Verify with official documentation before production use.
- **PyO3 0.22 compatibility:** Confirm Python 3.10+ compatibility with abi3-py310 feature using official PyO3 documentation.
- **Rayon 1.10 stability:** Verify compatibility with current Rust toolchain and rayon 1.10 stability for data parallelism patterns.
- **ASHRAE 140 standard access:** Did not have direct access to ASHRAE 140-2017 standard document. Secondary sources used for tolerance bands and reference values.
- **Solar radiation algorithm details:** Beam/diffuse decomposition and shading calculations require implementation-specific research during Phase 3.
- **GPU optimization patterns:** Limited open-source examples for GPU-accelerated thermal calculations. Research needed during Phase 6.
- **External verification:** Web search tool limitations prevented verification of recommendations with 2025 sources. All stack, features, architecture, and pitfalls based on Fluxion codebase analysis (HIGH confidence) and general domain knowledge (MEDIUM confidence).

**Handling gaps during planning:**
- Schedule `/gsd:research-phase` for Phase 3 (solar and external boundaries) to validate algorithm approaches
- Schedule `/gsd:research-phase` for Phase 6 (performance and scale) to research GPU optimization patterns
- Validate dependency versions (ort, pyo3, rayon) against official documentation during Phase 1 setup
- Use Fluxion's existing ASHRAE 140 documentation and test cases as primary source for validation requirements
- Prototype solar radiation calculations during Phase 3 planning to verify algorithm correctness
- Create proof-of-concept GPU integration during Phase 6 planning before full implementation

## Sources

### Primary (HIGH confidence)
- **Fluxion codebase analysis** — Comprehensive review of validation implementation, physics engine, and architecture at `/home/alex/Projects/fluxion/`
  - `src/validation/ashrae_140_validator.rs` — Validation framework and execution
  - `src/validation/ashrae_140_cases.rs` — Test case definitions and specifications
  - `src/sim/engine.rs` — ThermalModel and 5R1C physics engine
  - `src/physics/cta.rs` — Continuous Tensor Abstraction implementation
  - `tests/ashrae_140_validation.rs` — Integration tests and validation results
  - `Cargo.toml` — Current dependency versions
  - `requirements-dev.txt` — Python development dependencies

### Secondary (MEDIUM confidence)
- **Fluxion ASHRAE 140 documentation** — Test case overview, validation process, known issues
  - `docs/ASHRAE140_VALIDATION.md` — Test case overview and validation process
  - `docs/ASHRAE140_RESULTS.md` — Current validation results and systematic issues
  - `docs/ASHRAE140_MVP_ROADMAP.md` — Implementation roadmap and gap analysis
  - `docs/ASHRAE_140_5R1C_MODEL.md` — Thermal network model details
  - `docs/ASHRAE_140_DIAGNOSTICS.md` — Diagnostic features and debugging tools

- **ASHRAE Standard 140 methodology** — Referenced in building simulation literature, standard method of test for BEM programs (based on domain knowledge, not verified with 2025 sources due to web search limitations)

- **ISO 13790** — Energy performance of buildings - Calculation of energy use for space heating and cooling (5R1C thermal network standard, based on domain knowledge)

### Tertiary (LOW confidence)
- **Web search tool limitations** — Search service returned no results for ASHRAE 140 validation queries, limiting external verification. Recommendations based on Fluxion codebase analysis and general BEM domain knowledge.

- **Wikipedia articles** — Verified technical definitions of thermal resistance/conductance, RC network analogies, building performance simulation references

- **Competitor analysis** — EnergyPlus, ESP-r, TRNSYS features based on general BEM knowledge without recent verification. Fluxion differentiators (GPU acceleration, neural surrogates, CI/CD integration) inferred from competitor lack of these features.

---
*Research completed: 2026-03-08*
*Ready for roadmap: yes*
