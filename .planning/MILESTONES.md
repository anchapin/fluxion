# Fluxion Milestones

## v1.3 Blind ASHRAE 140 Validation (Physics Only) — IN PROGRESS

**Status:** 🚧 IN PROGRESS
**Started:** 2026-04-19
**Target close:** ≥80% case coverage on the ASHRAE 140-2023 blind-validation suite (no calibration factors, no case-ID hints, true reference values, ±15% annual / ±10% monthly / ±15% peak / ±1.0°C free-float gates). See `.planning/ASHRAE_140_BLIND_VALIDATION_PLAN.md` for the full spec.

### Live state (as of 2026-09-09)

- **Headline pass rate (metric-level):** **20.3%** (13/64) — release-gate target ≥60% (`release_gates.yaml → validation.individual`)
- **Mean Absolute Error (MAE):** **55.09%** — target ≤50%
- **Cases fully passing:** 1/18 (5.6%)
- **Strict ±15% annual-energy gate (Cases 600/900):** *heating* passes for both cases; *cooling* remains an unresolved structural failure on the 2026-08-11 baseline (the two `#[ignore]`'d strict tests run with `--include-ignored` and compare against `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`)

### Phase A8 — GaugeSolver production-path switchover (SHIPPED 2026-09-07)

- **Issue #3291 / PR #3482** closed and merged on **2026-09-07** (commit `e811df6`).
- `ThermalSelector::default()` resolves to `ZoneSolverKind::Gauge`; with `--features gauge-solver`, the dispatcher's `step_physics` routes to `GaugeSolver` unconditionally (no silent fall-through to legacy 5R1C/9R4C). `FiveROneC` and `NineRFourC` remain available as explicit opt-in legacy paths via `ThermalSelector`.
- The `gauge-solver` cargo feature is intentionally retained as the production-path gate pending §LIMIT-21 (Issue #3297) closure — the β-soak gate (Issue #3286, `#3286 β-soak` convention in CI comment threads) is currently at **0/30 nights green**. Once green, the unconditional production default applies.
- See `AGENTS.md:12`, `ARCHITECTURE.md:716-735` + `:1267-1269`, `docs/KNOWN_ISSUES.md` §LIMIT-21 + §LIMIT-22, and `src/sim/thermal_selector.rs`.

### Phase A–E roadmap

| Phase | Goal | Duration | Requirements | Status |
|-------|------|----------|--------------|--------|
| **A — Baseline Stripping** | Catalog and remove all correction infrastructure; measure the true physics-only baseline | 2 weeks | BASELINE-01, BASELINE-02, BASELINE-03 | 📋 in planning |
| **B — Physics Fixes** | Solar distribution (ISO 13790), thermal-mass time constant, free-floating temperature fixes | 18 weeks (B.1/B.2/B.3) | PHYSICS-01, PHYSICS-02, PHYSICS-03 | 📋 in planning |
| **C — Benchmark Correction** | Replace "calibrated for 5R1C" ranges with true EnergyPlus/ESP-r/TRNSYS reference values | 4 weeks | BENCH-01 | 📋 in planning |
| **D — Blind Validation Pass** | Run the full blind suite targeting ≥80% pass | 4 weeks | VALIDATE-01 | 📋 in planning |
| **E — Sustained Validation** | CI gate + regression tracking to hold the pass rate as code evolves | Ongoing | SUSTAIN-01, SUSTAIN-02 | 📋 in planning |

**Definition of Done:** blind execution (no case ID / case-type hint), zero correction factors, true ASHRAE reference values, pass tolerances of ±15% annual energy / ±10% monthly energy / ±15% peak loads / ±1.0°C free-floating temperature, and ≥80% case coverage.

### Linked ADRs and governance documents

- **ADR-0007** — GaugeSolver structural work: aggressive-baseline cohort unblocker (status: Accepted; production-path switchover shipped 2026-09-07)
- **ADR-0001** — No-Parameter-Tuning Rule (binding — fixes must address the underlying physics, not retune constants to make tests pass)
- **ADR-0003** — 5R1C high-mass limitations (root cause of the Case 900-series structural failure)
- `.planning/ASHRAE_140_BLIND_VALIDATION_PLAN.md` — full spec, roadmap, and test approach
- `.planning/PROJECT.md` — live state mirror (v1.3 phase breakdown + requirements index)
- `.planning/ROADMAP.md` — milestone timeline
- `docs/KNOWN_ISSUES.md` — open physics limitations; the `*Last Updated*` line is enforced fresh by `scripts/check_known_issues_stale.py` (≤60-day window)
- `SCORECARD.md` — auto-generated release-readiness snapshot (canonical status)
- `docs/ASHRAE140_RESULTS.md` — validation snapshot
- `release_gates.yaml` — required checks + thresholds (canonical source of truth for CI gates)

### Key dependent issues (open + recently closed)

| Issue | Status | Role in v1.3 |
|-------|--------|--------------|
| [#3291](https://github.com/anchapin/fluxion/issues/3291) | **Closed** (PR #3482, 2026-09-07) | Phase A8 GaugeSolver production-path switchover |
| [#3482](https://github.com/anchapin/fluxion/pull/3482) | **Merged** (2026-09-07) | The merged PR for #3291 |
| [#3297](https://github.com/anchapin/fluxion/issues/3297) | Closed (§LIMIT-21 mass-state exposure) | Gates the `gauge-solver` cargo feature → unconditional production default |
| [#3290](https://github.com/anchapin/fluxion/issues/3290) | Open | Phase A8 PR4 follow-up: remove the `#[cfg(feature = "gauge-solver")]` cargo feature flag from `ConductionBackend` (gated behind the β-soak gate reaching 30/30 nights green) |
| [#3573](https://github.com/anchapin/fluxion/issues/3573) | Closed (security) | TOCTOU window between `verify_onnx_signature` and ONNX session instantiation — fail-closed security contract across the surrogate load path |
| [#3624](https://github.com/anchapin/fluxion/issues/3624) | Open (FFI) | Fix zeroed energy surfaces in `fluxion-wasm` and NAPI `state_extractor` (surfaced by #3595 regression tests) — required for wasm/napi to publish v1.3 metrics |
| [#3286](https://github.com/anchapin/fluxion/issues/3286) | Open | β-soak gate; production-path switchover waits on `#3286 β-soak` reaching 30/30 nights green |
| [#3172](https://github.com/anchapin/fluxion/issues/3172) | Open | ADR-0007 implementation plan tracker |
| [#3072](https://github.com/anchapin/fluxion/issues/3072) | Open (meta) | Aggressive-baseline cohort (195 / 600 / 620 / 940 / 960) — unblocked by Phase A8 |
| [#3511](https://github.com/anchapin/fluxion/issues/3511) | Open | Post-#3291 GaugeSolver status refresh |

### Files of record

- **Spec / roadmap:** `.planning/ASHRAE_140_BLIND_VALIDATION_PLAN.md`
- **Project state mirror:** `.planning/PROJECT.md` (and `.planning/STATE.md`, refreshed by issue #3632)
- **Validation status snapshot:** `SCORECARD.md`, `docs/ASHRAE140_RESULTS.md`
- **Canonical CI gate list:** `release_gates.yaml` → `ci.required_checks`
- **Architecture source of truth:** `ARCHITECTURE.md`, `CODEBASE_MAP.md`
- **Hard constraints:** `RULES.md` (numerical-reasoning-via-code, energy balance, no parameter tuning)

### Notes

- Strict energy-conservation, `h_tr_em`, and surrogate-drift tolerance bands are *not* relaxed to compensate for nextest concurrency behavior — `scripts/ci/nextest-rollout.md` plus `.config/nextest.toml` are the canonical knobs (Issue #3366 / ADR-0014).
- The strict ±15% Cases 600/900 annual-energy gate is transparent + regression-catching: the two strict tests are `#[ignore]`'d by default and run via `--include-ignored`; never raise `tests/reference_data/zone_balance/strict_energy_gate_baseline.json` to hide the cooling-side gap.
- Goal #6 (contributor docs accurate) is the reason this entry exists: prior to issue #3632, `.planning/STATE.md` still described v1.2 as the current milestone and 2026-04-19 as the last update; this section re-establishes the canonical "current milestone page" for the work being done today.

**Last updated:** 2026-09-09

---

## v1.0 Multi-Zone Support (Shipped: 2026-04-07)

**Phases completed:** 3 phases (M1-M3), 12 plans, 36 tasks

**Key accomplishments:**

1. **Multi-Zone Thermal Network Foundation:** Extended single-zone 5R1C thermal network to support N zones with inter-zone conductance and coupled ODE solver
2. **Zone-Level HVAC Controls:** Implemented independent HVAC control per zone with zone-specific setpoints, Python bindings, and CLI support
3. **ASHRAE 140 Multi-Zone Validation:** Comprehensive validation framework with Case 960 validation tests and Case 970 framework foundation
4. **Energy Conservation:** Verified energy conservation within 1W tolerance across zones
5. **Performance Maintenance:** Multi-zone simulation performance maintained at <50ms per timestep for 10-zone simulations
6. **Full API Support:** Complete Python API and CLI support for multi-zone configuration and simulation

**Statistics:**
- Phases: 3 (M1-M3)
- Plans: 12 total
- Tasks: 36 total
- Files modified: 50+
- Lines of code: ~124,509 Rust lines
- Timeline: 1 day (2026-04-06 → 2026-04-07)
- Git range: Multiple commits across v1.0 development

---

## v1.1 ASHRAE 140 Completion (Partial) (Shipped: 2026-04-08)

**Phases completed:** 1 phase (Phase 40), 9 plans, 27 tasks

**Key accomplishments:**

1. **ASHRAE 140 Case Expansion:** Extended support for Cases 800-810 (HVAC equipment validation) and 195-470 (diagnostic validation)
2. **Extended Reference Database:** Comprehensive reference data infrastructure with 96,361 + 2,417,761 data rows and complete hourly coverage
3. **Cross-Validation Framework:** Implemented EnergyPlus and TRNSYS adapters with trait-based architecture for ASHRAE 140 compliance verification
4. **CLI Integration:** Enhanced command-line interface with validation execution and cross-validation commands
5. **Performance Optimization:** Real performance monitoring with high-resolution timing, parallel processing, and bottleneck analysis
6. **Validation Infrastructure:** Complete validation framework with comprehensive reporting and analysis capabilities

**Statistics:**
- Phases: 1 (Phase 40 only)
- Plans: 9 total
- Tasks: 27 total  
- Files modified: 50+
- Lines of code: ~124,509 Rust lines
- Timeline: 2 months (2026-02-16 → 2026-04-08)
- Git range: Multiple commits across Phase 40 development
- Requirements completion: 6/16 (37.5%) - Partial completion with Phases 41-43 deferred to v1.2

**Notes:**
- Partial milestone completion: Only Phase 40 (Case Expansion Foundation) completed
- Deferred work: Phases 41-43 (High-Mass Physics, Advanced Cross-Validation, Validation Optimization) moved to v1.2
- Requirements status: 6 requirements completed, 10 requirements deferred

---

## v0.8.0 v0.8.0 (Shipped: 2026-04-07)

**Phases completed:** 4 phases, 10 plans, 24 tasks

**Key accomplishments:**

- Issue
- 1. [Scaling Calibration] Adjusted τ scaling factor
- Phase 34 Status:
- Adjusted thermal time constant τ and free-float capacitance for ASHRAE 140 peak loads and free-floating temperatures
- ASHRAE 140 validation executed with 25% pass rate - Phase 34/35 fixes appear not fully integrated
- Comprehensive documentation update for v0.8.0 release with peak load and free-float validation improvements
- v0.8.0 release script created and verified with comprehensive publication automation
- Gap Closure Status:

---

## v0.4 ASHRAE 140 Compliance (Shipped: 2026-03-15)

**Phases completed:** 20 phases, 138 plans, 187 tasks

**Key accomplishments:**

- Comprehensive test-driven test suite for all 5R1C thermal network conductances with placeholder implementations
- ISO 13790-compliant 5R1C conductance calculations with TDD validation and all Plan 01 tests passing
- Test infrastructure for directional conductance, Stefan-Boltzmann radiation, and stack effect ACH calculations
- Complete three-component inter-zone heat transfer integration with conductive, radiative, and ventilation components
- ASHRAE-compliant PsychrometricCalculations trait implementation with dew point, wet-bulb, humidity ratio, and enthalpy methods for weather data integration
- Economizer enthalpy mode uses psychrometrics module for accurate ASHRAE-compliant enthalpy calculations
- Trait-based material layer abstraction with AssemblyBuilder pattern and ISO 13790 Annex C thermal mass auto-calculation

---

This document records completed milestone releases with summaries, statistics, and achievements.

---

## v0.2: ASHRAE 140 Partial Validation

**Shipped:** 2026-03-11
**Phases:** 1-7 (all complete)
**Plans:** 48 completed
**Duration:** 4 days (March 8-11, 2026)
**Git Stats:** 228 files changed, +56,270 / -1,151 lines

### What Was Built

Substantial ASHRAE 140 validation progress with comprehensive infrastructure, but **critical gaps remain**:

- **Physics Validation:** Peak loads within reference; solar integration complete; free-floating validation passing; multi-zone physics validated. **FAILURE:** High-mass annual energy 229-322% above reference (fundamental 5R1C limitation).
- **Diagnostics:** Automated validation reports (`docs/ASHRAE140_RESULTS.md`), hourly CSV export with metadata, systematic issue classification
- **Performance:** Full validation suite <5 minutes; GPU acceleration with ONNX Runtime; parallel rayon execution; regression guardrails
- **Research Tools:** Sensitivity analysis (OAT + Sobol), delta testing (YAML-driven), interactive HTML visualization (Plotly), multi-reference comparison (EnergyPlus, ESP-r, TRNSYS)
- **Extensibility:** Extended CaseBuilder API for custom geometries and assemblies; CLI with 7 subcommands

### Key Achievements

1. **MAE improved 37.5%** (78.79% → 49.21%) through conductance fixes, HVAC correction, thermal mass integration, and solar radiation
2. **Peak loads validated** — heating 2.10 kW, cooling 3.56 kW within ASHRAE reference ranges ✅
3. **Free-floating validation passing** — 10/10 tests passing, temperature swing reduction 22.4% ✅
4. **Solar integration complete** — all 4 SOLAR requirements satisfied with full beam/diffuse decomposition ✅
5. **Multi-zone physics validated** — directional conductance, nonlinear Stefan-Boltzmann radiation, stack effect ACH ✅
6. **Performance optimized** — rayon parallelization, GPU-accelerated surrogates, <5 min full suite ✅
7. **Advanced analysis tools** — sensitivity, delta, component breakdown, swing metrics, visualization, multi-reference ✅

### Requirements Coverage

- **Total v0.2 requirements:** 51
- **Satisfied:** 51 (100%)
- **Partially Satisfied:** 0
- **Unsatisfied:** 0

All requirements mapped to phases and validated through automated tests. **However**, validation status is **partial** due to fundamental model limitations.

### Critical Gap: High-Mass Annual Energy

**Status:** ❌ **NOT FIXED** — Fundamental 5R1C model limitation

**Evidence (Case 900):**

- Annual heating: **5.35 MWh** vs [1.17, 2.04] MWh reference (**262-322% above**)
- Annual cooling: **4.75 MWh** vs [2.13, 3.67] MWh reference (**229-259% above**)
- Peak heating: 2.10 kW ✅ (within [1.10, 2.10] kW)
- Peak cooling: 3.56 kW ✅ (within [2.10, 3.50] kW)

**Root Cause:** High h_tr_em/h_tr_ms coupling ratio (0.0525) causes thermal mass to exchange 95% with interior instead of exterior. 8 sophisticated approaches attempted (Plans 03-07 through 03-14), all failed to achieve annual energy targets. Mode-specific coupling provided 22% heating improvement but still far from reference.

**Impact:** Model is **not suitable for production building energy analysis** where annual energy accuracy is required. Suitable for research, prototyping, and low-mass building analysis.

### Other Known Issues

1. **Case 960 annual cooling** (4.53 MWh) — Fails validation, issue #273 under investigation
2. **Temperature swing reduction** 13.7% vs target 19.6% — partial achievement

### Technical Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Physics-first approach | Fix accuracy before optimization | ✅ Core physics validated before performance work |
| Diagnostics before performance | Tools needed to debug validation | ✅ CSV export and reports were essential |
| HTML+Plotly visualization | Easy sharing, interactive exploration | ✅ Single-file HTML with embedded data works well |
| BatchOracle pattern (rayon par_iter) | Maximize GPU utilization, avoid nested parallelism | ✅ Pre-commit hook enforces pattern |
| Multi-reference JSON architecture | Easy updates without code changes | ✅ Remote fetching implemented |
| Modular surrogates | Separate component models for flexibility | ✅ Architecture supports future additions |
| CLI subcommands for research | Expose advanced tools directly | ✅ sensitivity, delta, viz all accessible |
| Known limitations documentation | Transparent about gaps | ✅ Comprehensive KNOWN_ISSUES.md |

### What Worked

- **Sequential physics domains** (foundation → mass → solar → multi-zone) ensured each layer was solid before adding complexity
- **Comprehensive test scaffolding** (45+ test functions for inter-zone physics) caught edge cases early
- **Automated validation reporting** kept progress visible and guided debugging
- **Gap documentation** (KNOWN_ISSUES.md) allowed honest assessment without blocking release

### What Was Inefficient

- **Phase 1 success criteria too aggressive** — <15% MAE target not achievable in single phase; took 3 phases to reach best state
- **Some verification artifacts inconsistent** — VALIDATION.md vs VERIFICATION.md naming drift; could standardize
- **Manual requirements traceability** — REQUIREMENTS.md needed final manual update; could integrate with SUMMARY frontmatter earlier

### Reusable Artifacts

- Validation infrastructure: `src/validation/` (modular, extensible)
- Diagnostic system: `SimulationDiagnostics` with CSV export
- Analysis modules: `src/analysis/` (sensitivity, delta, components, swing)
- CLI framework: 7 subcommands with common output formatting
- Test data: `docs/ashrae_140_references.json` (multi-reference database)
- Documentation templates: `KNOWN_ISSUES.md`, `ASHRAE140_RESULTS.md`

---

### Next Steps

**To reach v1.0, address:**

1. Investigate 6R2C or 8R3C thermal network for high-mass buildings
2. Fix Case 960 annual cooling failure (#273)
3. Consider alternative integration methods or time step strategies
4. Detailed comparison with EnergyPlus/ESP-r to understand reference implementation differences

**Current State:**
All infrastructure is in place and working. The gap is **fundamental physics modeling**, not implementation quality. Proceed with caution: this codebase is suitable for research but **not** for production energy compliance without addressing the high-mass annual energy issue.

---

*Next milestone: TBD — see PROJECT.md for recommendations*
