================================================================================
PARALLEL ISSUES EXECUTION PLAN
================================================================================

Total tracks: 4
Total issues to work: 23 (ASHRAE 140 MVP phases 0-4) + 2 (Future: Phase 6, 8)

Current Branch: issue/46-surrogate-batch-inference (already in progress)

================================================================================
ISSUE PRIORITY MATRIX
================================================================================

Phase-based priority scoring (Phase × Weight):
  Phase 0 (Foundation): CRITICAL - 150 pts - Blocks all ASHRAE 140 work
  Phase 1 (Core Physics): CRITICAL - 150 pts - Baseline cases required for validation
  Phase 2 (Feature Variants): HIGH - 120 pts - Expands validation coverage
  Phase 3 (Advanced): HIGH - 120 pts - Special cases for completeness
  Phase 4 (Validation): MEDIUM - 80 pts - Reporting and CI (can start early)
  Phase 6+ (Future): LOW - 20 pts - Post-MVP enhancements

================================================================================
TRACK 1: FOUNDATION & DATA (Phase 0)
================================================================================

  Issue #52: Weather Data Infrastructure for ASHRAE 140
  └─ Priority: CRITICAL | Phase 0 | Score: 150
  └─ Labels: enhancement, good first issue, phase-0, weather
  └─ Worktree: ../feature-issue-52-weather-data-infrastructure-for-ashrae-140
  └─ Branch: feature/issue-52
  └─ Tasks:
    - Create src/weather/mod.rs and src/weather/epw.rs
    - Implement EPW format parsing for TMY weather files
    - Add embedded Denver weather data
    - Implement WeatherSource trait

  Issue #53: ASHRAE 140 Test Case Data Structure
  └─ Priority: CRITICAL | Phase 0 | Score: 150
  └─ Labels: enhancement, data-pipeline, validation, phase-0
  └─ Worktree: ../feature-issue-53-ashrae-140-test-case-data-structure
  └─ Branch: feature/issue-53
  └─ Tasks:
    - Create src/validation/ashrae_140_cases.rs
    - Define ASHRAE140Case enum with all test variants
    - Create case specifications database (geometry, construction, windows, HVAC)
    - Add case builder pattern

  Issue #54: Validation Framework Enhancements
  └─ Priority: CRITICAL | Phase 0 | Score: 150
  └─ Labels: documentation, enhancement, validation, phase-0
  └─ Worktree: ../feature-issue-54-validation-framework-enhancements
  └─ Branch: feature/issue-54
  └─ Tasks:
    - Extend ValidationReport with reference range comparison
    - Add benchmark data from EnergyPlus, ESP-r, TRNSYS
    - Implement pass/fail determination logic
    - Add HTML/Markdown report generation

  PARALLEL STATUS: ✅ CAN WORK IN PARALLEL
  └─ Different modules (weather, validation, data structures)
  └─ Minimal dependencies (can be merged sequentially)

================================================================================
TRACK 2: CORE PHYSICS (Phase 1)
================================================================================

  Issue #55: Multi-Layer Construction R-Value Calculator
  └─ Priority: CRITICAL | Phase 1 | Score: 150
  └─ Labels: enhancement, physics, phase-1, construction
  └─ Worktree: ../feature-issue-55-multi-layer-construction-r-value-calculator
  └─ Branch: feature/issue-55
  └─ Tasks:
    - Create src/sim/construction.rs
    - Implement ConstructionLayer struct (k, ρ, Cp, thickness)
    - Implement calculate_u_value() for layer stacks
    - Add ASHRAE film coefficient functions

  Issue #56: Solar Radiation Calculator
  └─ Priority: CRITICAL | Phase 1 | Score: 150
  └─ Labels: enhancement, physics, phase-1, solar
  └─ Worktree: ../feature-issue-56-solar-radiation-calculator
  └─ Branch: feature/issue-56
  └─ Tasks:
    - Create src/sim/solar.rs
    - Implement solar position algorithm (NOAA or similar)
    - Implement surface insolation model (beam, diffuse, ground-reflected)
    - Implement window solar gain with angle-dependent transmittance

  Issue #57: Dual HVAC Setpoint Control
  └─ Priority: CRITICAL | Phase 1 | Score: 150
  └─ Labels: enhancement, physics, phase-1, hvac
  └─ Worktree: ../feature-issue-57-dual-hvac-setpoint-control
  └─ Branch: feature/issue-57
  └─ Tasks:
    - Modify src/sim/engine.rs ThermalModel
    - Replace hvac_setpoint with heating_setpoint and cooling_setpoint
    - Implement deadband control logic
    - Update hvac_power_demand() for dual setpoints

  Issue #58: Case 600 Baseline Implementation
  └─ Priority: CRITICAL | Phase 1 | Score: 150
  └─ Labels: enhancement, validation, phase-1, ashrae-140
  └─ Worktree: ../feature-issue-58-case-600-baseline-implementation
  └─ Branch: feature/issue-58
  └─ Tasks:
    - Implement exact Case 600 geometry (8×6×2.7m)
    - Apply low-mass construction layers
    - Apply window specifications (double-pane, U=3.0, SHGC=0.789)
    - Apply ASHRAE film coefficients
    - Set operating conditions (setpoints, infiltration, internal loads)
    - BLOCKED BY: #55, #56, #57 (depends on core physics features)

  Issue #59: Ground Boundary Condition
  └─ Priority: CRITICAL | Phase 1 | Score: 150
  └─ Labels: enhancement, physics, phase-1
  └─ Worktree: ../feature-issue-59-ground-boundary-condition
  └─ Branch: feature/issue-59
  └─ Tasks:
    - Create src/sim/boundary.rs
    - Implement constant soil temperature (T_soil = 10°C)
    - Apply to floor conductance
    - Add optional dynamic soil temperature (Kusuda formula placeholder)

  PARALLEL STATUS: ⚠️ PARTIAL PARALLEL
  └─ #55, #56, #57, #59 CAN work in parallel (independent modules)
  └─ #58 MUST wait for #55, #56, #57 to be merged (integrates all features)

================================================================================
TRACK 3: FEATURE VARIANTS (Phase 2)
================================================================================

  Issue #60: High Mass Construction (Cases 900-950)
  └─ Priority: HIGH | Phase 2 | Score: 120
  └─ Labels: enhancement, physics, phase-2, construction
  └─ Worktree: ../feature-issue-60-high-mass-construction-cases-900-950
  └─ Branch: feature/issue-60
  └─ Tasks:
    - Add high-mass material database (concrete, foam)
    - Implement Case 900 with high-mass construction layers
    - Update 5R1C parameters for high mass (higher Cm)
    - BLOCKED BY: #55 (depends on construction layer infrastructure)

  Issue #61: Window Orientation and Areas
  └─ Priority: HIGH | Phase 2 | Score: 120
  └─ Labels: enhancement, physics, phase-2, geometry
  └─ Worktree: ../feature-issue-61-window-orientation-and-areas
  └─ Branch: feature/issue-61
  └─ Tasks:
    - Create src/sim/geometry.rs
    - Add explicit orientation tracking (N/E/S/W with azimuth angles)
    - Implement per-orientation window areas
    - Update solar model for orientation-specific gain
    - BLOCKED BY: #56 (depends on solar model)

  Issue #62: Shading Implementation (Cases 610, 630, 910, 930)
  └─ Priority: HIGH | Phase 2 | Score: 120
  └─ Labels: enhancement, physics, phase-2, shading
  └─ Worktree: ../feature-issue-62-shading-implementation-cases-610-630-910-930
  └─ Branch: feature/issue-62
  └─ Tasks:
    - Create src/sim/shading.rs
    - Implement overhang geometry (1m projection at roof level)
    - Implement shade fins (vertical fins at window edges)
    - Implement shading calculation (shadow projection, shaded fraction)
    - BLOCKED BY: #56, #61 (depends on solar model and geometry)

  Issue #63: Thermostat Setback (Cases 640, 940)
  └─ Priority: HIGH | Phase 2 | Score: 120
  └─ Labels: enhancement, phase-2, hvac, scheduling
  └─ Worktree: ../feature-issue-63-thermostat-setback-cases-640-940
  └─ Branch: feature/issue-63
  └─ Tasks:
    - Create src/sim/schedule.rs
    - Implement HVAC schedule system (hourly resolution)
    - Define heating schedule (20°C 0700-2300h, 10°C 2300-0700h)
    - Define cooling schedule (27°C all hours)
    - BLOCKED BY: #57 (depends on dual HVAC setpoints)

  Issue #64: Night Ventilation (Cases 650, 950, FF variants)
  └─ Priority: HIGH | Phase 2 | Score: 120
  └─ Labels: enhancement, physics, phase-2, scheduling
  └─ Worktree: ../feature-issue-64-night-ventilation-cases-650-950-ff-variants
  └─ Branch: feature/issue-64
  └─ Tasks:
    - Implement ventilation schedule (fan ON 1800-0700h)
    - Implement fan capacity (1703.16 standard m³/h)
    - Update ACH calculation dynamically per timestep
    - BLOCKED BY: #58 (depends on Case 600 baseline)

  PARALLEL STATUS: ⚠️ BLOCKED BY PHASE 1
  └─ All Phase 2 issues depend on Phase 1 features being complete
  └─ Once Phase 1 merges, Phase 2 CAN work in parallel (different case variants)

================================================================================
TRACK 4: ADVANCED CASES & VALIDATION (Phase 3-4)
================================================================================

  Issue #65: Free-Floating Mode (600FF, 900FF, 650FF, 950FF)
  └─ Priority: HIGH | Phase 3 | Score: 120
  └─ Labels: enhancement, phase-3, hvac, ashrae-140
  └─ Worktree: ../feature-issue-65-free-floating-mode-600ff-900ff-650ff-950ff
  └─ Branch: feature/issue-65
  └─ Tasks:
    - Add hvac_mode enum (Controlled, FreeFloat)
    - Implement free-floating physics (no HVAC, track temps only)
    - Implement Case 600FF and 650FF
    - BLOCKED BY: #58, #64 (depends on baseline and ventilation)

  Issue #66: Multi-Zone Sunspace (Case 960)
  └─ Priority: HIGH | Phase 3 | Score: 120
  └─ Labels: enhancement, physics, phase-3, multi-zone
  └─ Worktree: ../feature-issue-66-multi-zone-sunspace-case-960
  └─ Branch: feature/issue-66
  └─ Tasks:
    - Create src/sim/multi_zone.rs
    - Implement inter-zone conductance (common wall)
    - Define Case 960 geometry (back-zone + sunspace)
    - Implement HVAC (sunspace free-floating, back-zone controlled)
    - BLOCKED BY: #58, #60 (depends on baseline and high mass construction)

  Issue #67: Solid Conduction Problem (Case 195)
  └─ Priority: HIGH | Phase 3 | Score: 120
  └─ Labels: enhancement, physics, accuracy, phase-3
  └─ Worktree: ../feature-issue-67-solid-conduction-problem-case-195
  └─ Branch: feature/issue-67
  └─ Tasks:
    - Implement Case 195 (no windows, no infiltration, no loads)
    - Apply low absorptance/emissivity (0.1)
    - Implement bang-bang control (20°C/20°C)
    - Validate against analytical solution
    - BLOCKED BY: #55 (depends on construction layers)

  Issue #68: Automated ASHRAE 140 Test Suite
  └─ Priority: MEDIUM | Phase 4 | Score: 80
  └─ Labels: enhancement, testing, phase-4, ashrae-140
  └─ Worktree: ../feature-issue-68-automated-ashrae-140-test-suite
  └─ Branch: feature/issue-68
  └─ Tasks:
    - Create tests/ashrae_140_integration.rs
    - Implement #[test] for each ASHRAE 140 case
    - Add parameterized tests for case series
    - Add benchmark assertions (pass/fail criteria)
    - CAN START EARLY (mock tests, then update with real cases)

  Issue #69: Validation Report Generation
  └─ Priority: MEDIUM | Phase 4 | Score: 80
  └─ Labels: documentation, enhancement, validation, phase-4
  └─ Worktree: ../feature-issue-69-validation-report-generation
  └─ Branch: feature/issue-69
  └─ Tasks:
    - Generate comprehensive validation report (summary table, charts)
    - Export to Markdown, CSV, JSON formats
    - Compare to reference programs (EnergyPlus, ESP-r, TRNSYS)
    - BLOCKED BY: #54 (depends on validation framework)

  Issue #70: Continuous Integration for ASHRAE 140
  └─ Priority: MEDIUM | Phase 4 | Score: 80
  └─ Labels: enhancement, tools, phase-4, ci
  └─ Worktree: ../feature-issue-70-continuous-integration-for-ashrae-140
  └─ Branch: feature/issue-70
  └─ Tasks:
    - Add ASHRAE 140 tests to CI pipeline
    - Enforce pass criteria on every PR
    - Block merge if regression >2%
    - BLOCKED BY: #68, #69 (depends on test suite and reports)

  Issue #71: Documentation Updates for ASHRAE 140
  └─ Priority: MEDIUM | Phase 4 | Score: 80
  └─ Labels: documentation, enhancement, phase-4
  └─ Worktree: ../feature-issue-71-documentation-updates-for-ashrae-140
  └─ Branch: feature/issue-71
  └─ Tasks:
    - Update CLAUDE.md with ASHRAE 140 status
    - Create docs/ASHRAE140_VALIDATION.md
    - Create docs/ASHRAE140_RESULTS.md
    - Update PyO3 and Rust API docs
    - Update README and CONTRIBUTING.md
    - CAN START EARLY (documentation can be drafted and refined)

  PARALLEL STATUS: ⚠️ DEPENDENCY CHAINS
  └─ Phase 3 depends on Phase 1-2 features
  └─ Phase 4 validation depends on Phase 1-3 completion
  └─ #68 and #71 CAN START EARLY (tests and docs can be prepared incrementally)

================================================================================
TRACK 5: FUTURE ENHANCEMENTS (Phase 6-8)
================================================================================

  Issue #46: perf(core): Vectorize surrogate inference across population
  └─ Priority: LOW | Phase 6 | Score: 20
  └─ Labels: performance, rust, phase-6
  └─ Worktree: feature/issue-46 (CURRENT BRANCH)
  └─ Status: ALREADY IN PROGRESS
  └─ Tasks:
    - Batch surrogate inference for entire population
    - Use GPU tensor cores for batch processing
    - Minimize Python-Rust boundary crossings
    - NOTE: Currently on issue/46-surrogate-batch-inference branch

  Issue #45: feat(ml): Implement Physics-Informed Loss (PINN) for surrogates
  └─ Priority: LOW | Phase 8 | Score: 20
  └─ Labels: accuracy, ml, phase-8
  └─ Worktree: ../feature-issue-45-feat-ml-implement-physics-informed-loss-pinn-for-surrogates
  └─ Branch: feature/issue-45
  └─ Tasks:
    - Implement PINN loss function
    - Add energy balance constraints to surrogate training
    - Validate surrogate accuracy against physics
    - BLOCKED BY: ASHRAE 140 MVP completion

  PARALLEL STATUS: ✅ CAN WORK IN PARALLEL (post-MVP)
  └─ These are Phase 6+ enhancements, not blocking ASHRAE 140 MVP

================================================================================
GIT WORKTREE SETUP COMMANDS
================================================================================

# TRACK 1: Foundation (Phase 0) - CAN START NOW
git worktree add ../feature-issue-52-weather-data-infrastructure-for-ashrae-140 -b feature/issue-52
git worktree add ../feature-issue-53-ashrae-140-test-case-data-structure -b feature/issue-53
git worktree add ../feature-issue-54-validation-framework-enhancements -b feature/issue-54

# TRACK 2: Core Physics (Phase 1) - CAN START NOW (except #58)
git worktree add ../feature-issue-55-multi-layer-construction-r-value-calculator -b feature/issue-55
git worktree add ../feature-issue-56-solar-radiation-calculator -b feature/issue-56
git worktree add ../feature-issue-57-dual-hvac-setpoint-control -b feature/issue-57
git worktree add ../feature-issue-59-ground-boundary-condition -b feature/issue-59

# TRACK 3: Validation Early Start (Phase 4) - CAN START NOW
git worktree add ../feature-issue-68-automated-ashrae-140-test-suite -b feature/issue-68
git worktree add ../feature-issue-71-documentation-updates-for-ashrae-140 -b feature/issue-71

================================================================================
EXECUTION SUMMARY
================================================================================

All open issues by phase:
  Phase 0 (Foundation): 3 issues - CRITICAL - 450 pts total
  Phase 1 (Core Physics): 5 issues - CRITICAL - 750 pts total
  Phase 2 (Feature Variants): 5 issues - HIGH - 600 pts total
  Phase 3 (Advanced): 3 issues - HIGH - 360 pts total
  Phase 4 (Validation): 4 issues - MEDIUM - 320 pts total
  Phase 6+ (Future): 2 issues - LOW - 40 pts total

  Total: 22 issues for ASHRAE 140 MVP (Phase 0-4)
  Total: 2 issues for post-MVP (Phase 6-8)

RECOMMENDED EXECUTION ORDER (with max 4 parallel worktrees):

WAVE 1 (Weeks 1-2): Foundation & Core Physics - ALL CRITICAL
  └─ Worktree 1: #52 Weather Data Infrastructure
  └─ Worktree 2: #55 Multi-Layer Construction
  └─ Worktree 3: #56 Solar Radiation Calculator
  └─ Worktree 4: #57 Dual HVAC Setpoint Control

  PREREQUISITES: None (all can start immediately)

WAVE 2 (Weeks 3-4): Core Physics Completion + Validation Framework
  └─ Worktree 1: #53 ASHRAE 140 Test Case Data Structure
  └─ Worktree 2: #54 Validation Framework Enhancements
  └─ Worktree 3: #59 Ground Boundary Condition
  └─ Worktree 4: #58 Case 600 Baseline (INTEGRATES #55, #56, #57, #59)

  PREREQUISITES: Wave 1 complete

WAVE 3 (Weeks 5-6): Feature Variants - Phase 2
  └─ Worktree 1: #60 High Mass Construction (depends on #55)
  └─ Worktree 2: #61 Window Orientation (depends on #56)
  └─ Worktree 3: #62 Shading Implementation (depends on #56, #61)
  └─ Worktree 4: #63 Thermostat Setback (depends on #57)

  PREREQUISITES: Wave 2 complete (Case 600 baseline)

WAVE 4 (Weeks 7-8): Night Ventilation + Early Validation
  └─ Worktree 1: #64 Night Ventilation (depends on #58)
  └─ Worktree 2: #68 Automated Test Suite (can start early)
  └─ Worktree 3: #71 Documentation Updates (can start early)
  └─ Worktree 4: Complete Wave 3 issues if needed

  PREREQUISITES: Wave 3 complete

WAVE 5 (Weeks 9-10): Advanced Cases - Phase 3
  └─ Worktree 1: #65 Free-Floating Mode (depends on #58, #64)
  └─ Worktree 2: #66 Multi-Zone Sunspace (depends on #58, #60)
  └─ Worktree 3: #67 Solid Conduction (depends on #55)
  └─ Worktree 4: #69 Validation Report Generation (depends on #54)

  PREREQUISITES: Wave 4 complete

WAVE 6 (Weeks 11-12): Validation & CI - Phase 4
  └─ Worktree 1: #70 Continuous Integration (depends on #68, #69)
  └─ Worktree 2: Finalize #68 Test Suite
  └─ Worktree 3: Finalize #69 Reports
  └─ Worktree 4: Finalize #71 Documentation

  PREREQUISITES: Wave 5 complete

WAVE 7+ (Post-MVP): Future Enhancements
  └─ Worktree 1: #46 Vectorize surrogate inference (already in progress)
  └─ Worktree 2: #45 Physics-Informed Loss (PINN)

  PREREQUISITES: ASHRAE 140 MVP complete

================================================================================
PARALLEL WORKABILITY MATRIX
================================================================================

                    #52  #53  #54  #55  #56  #57  #58  #59  #60  #61  #62  #63  #64  #65  #66  #67  #68  #69  #70  #71
#52 Weather          -    ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅
#53 Case Structure   ✅   -    ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅
#54 Validation       ✅   ✅   -    ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅
#55 Construction     ✅   ✅   ✅   -    ✅   ✅   ⛔   ✅   ⛔   ✅   ✅   ✅   ✅   ✅   ✅   ⛔   ✅   ✅   ✅   ✅
#56 Solar            ✅   ✅   ✅   ✅   -    ✅   ⛔   ✅   ✅   ⛔   ⛔   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅
#57 HVAC Dual        ✅   ✅   ✅   ✅   ✅   -    ⛔   ✅   ✅   ✅   ✅   ⛔   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅
#58 Case 600         ✅   ✅   ✅   ⛔   ⛔   ⛔   -    ⛔   ⛔   ⛔   ⛔   ⛔   ⛔   ⛔   ⛔   ⛔   ⛔   ⛔   ⛔   ⛔
#59 Ground           ✅   ✅   ✅   ✅   ✅   ✅   ⛔   -    ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅
#60 High Mass        ✅   ✅   ✅   ⛔   ✅   ✅   ⛔   ✅   -    ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅
#61 Orientation      ✅   ✅   ✅   ✅   ⛔   ✅   ⛔   ✅   ✅   -    ⛔   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅
#62 Shading          ✅   ✅   ✅   ✅   ⛔   ✅   ⛔   ✅   ⛔   ⛔   -    ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅
#63 Setback          ✅   ✅   ✅   ✅   ✅   ⛔   ⛔   ✅   ✅   ✅   ✅   ✅   -    ✅   ✅   ✅   ✅   ✅   ✅   ✅
#64 Night Vent       ✅   ✅   ✅   ✅   ✅   ✅   ⛔   ✅   ✅   ✅   ✅   ✅   ✅   -    ⛔   ⛔   ✅   ✅   ✅   ✅   ✅
#65 Free Float       ✅   ✅   ✅   ✅   ✅   ✅   ⛔   ✅   ✅   ✅   ✅   ✅   ✅   ⛔   -    ✅   ✅   ✅   ✅   ✅   ✅
#66 Multi-Zone       ✅   ✅   ✅   ✅   ✅   ⛔   ⛔   ✅   ⛔   ✅   ✅   ✅   ✅   ⛔   ✅   -    ✅   ✅   ✅   ✅   ✅   ✅
#67 Solid Cond       ✅   ✅   ✅   ⛔   ✅   ✅   ⛔   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   -    ✅   ✅   ✅   ✅   ✅
#68 Test Suite       ✅   ✅   ✅   ✅   ✅   ✅   ⛔   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   -    ⛔   ✅   ✅
#69 Reports          ✅   ✅   ✅   ✅   ✅   ✅   ⛔   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ⛔   ✅   -    ⛔   ✅
#70 CI               ✅   ✅   ✅   ✅   ✅   ✅   ⛔   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ⛔   ⛔   -    ✅
#71 Docs             ✅   ✅   ✅   ✅   ✅   ✅   ⛔   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   ✅   -

Legend:
  ✅ = CAN WORK IN PARALLEL (no direct file conflicts)
  ⛔ = BLOCKED (depends on other issue to complete first)
  - = Self

================================================================================
CURRENT WORK STATUS
================================================================================

IN PROGRESS:
  └─ Issue #46: perf(core): Vectorize surrogate inference across population
     └─ Current Branch: issue/46-surrogate-batch-inference
     └─ Status: On branch, likely has uncommitted changes (check git status)
     └─ Recommendation: Complete or pause this to focus on ASHRAE 140 MVP (Phase 0-4)

RECOMMENDATION:
  The ASHRAE 140 MVP (Phase 0-4) is the critical path. Issue #46 (Phase 6) is a
  post-MVP enhancement and should be deprioritized unless explicitly requested.

================================================================================
NEXT STEPS
================================================================================

1. Review this plan and confirm prioritization
2. Decide whether to pause #46 or continue in parallel
3. Set up git worktrees for Wave 1 (4 parallel tracks):
   ```bash
   git worktree add ../feature-issue-52-weather-data-infrastructure-for-ashrae-140 -b feature/issue-52
   git worktree add ../feature-issue-55-multi-layer-construction-r-value-calculator -b feature/issue-55
   git worktree add ../feature-issue-56-solar-radiation-calculator -b feature/issue-56
   git worktree add ../feature-issue-57-dual-hvac-setpoint-control -b feature/issue-57
   ```
4. Launch background agents for each worktree using the Task tool
5. Monitor progress with TaskOutput
6. Create PRs as issues complete and remove worktrees

*Generated: 2026-02-12*
*Parallel Planner for Fluxion ASHRAE 140 MVP*
