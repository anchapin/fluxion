# Fluxion - Building Energy Modeling Engine

## What This Is

Fluxion is a Rust-based Building Energy Modeling (BEM) engine with a Neuro-Symbolic hybrid architecture. It combines physics-based thermal networks with AI surrogates for 100x speedups, designed to serve as a high-throughput oracle for quantum optimization and genetic algorithms.

## Current State (v1.3 — IN PROGRESS: Blind ASHRAE 140 Validation)

**Milestone:** v1.3 "Blind ASHRAE 140 Validation" (Physics Only) 🚧 IN PROGRESS
**Status:** Physics-only baseline established; corrections catalogued; underlying physics fixes in flight.
**Last Updated:** 2026-08-11

The goal of v1.3 is to achieve ASHRAE 140 validation with **true blind test methodology** — no calibration factors, no case-specific corrections, physics-only model. The current "informed" validation (case IDs known before simulation, post-simulation corrections applied, benchmark ranges "calibrated for 5R1C") is being replaced with genuine compliance.

### Current Validation Snapshot (2026-08-07 run)

| Metric | Current | Release-gate Target | Status |
|--------|---------|---------------------|--------|
| Pass rate (metric-level) | **20.3%** (13/64) | ≥ 60% | ❌ Fail |
| Mean Absolute Error (MAE) | **55.09%** | ≤ 50% | ❌ Fail |
| Cases fully passing | 1/18 (5.6%) | — | ❌ |
| Max single-case deviation | 499.89% | — | ℹ️ |

**Known structural failures** (`release_gates.yaml → validation.individual.known_failures`):
- **Baseline 600-series (low-mass):** All 6 cases FAIL. Simplified envelope model over-predicts peak loads (e.g. peak heating ~4.36 kW vs 2.80–3.80 kW reference band).
- **High-mass 900-series:** All 6 cases FAIL. Heating over-predicted by ~**200%** due to a 5R1C/CTF thermal-mass limitation (e.g. Case 900 annual heating 5,449 kWh vs 1,170–2,040 kWh reference band).

Per `RULES.md`, the fix path is the underlying physics — **no parameter tuning** to make tests pass. The strict ±15% annual-energy gate for Cases 600/900 (`ASHRAE 140 Strict Energy Gate (Issue #1333)`) is transparent + regression-catching: the two `#[ignore]`'d strict tests run with `--include-ignored` and are compared against `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`. Heating currently passes the ±15% band for both Cases 600 and 900; the Case 600/900 annual **cooling** gap is an unresolved structural failure on the 2026-08-11 baseline.

### v1.3 Phase Breakdown (Phases A–E)

| Phase | Goal | Duration | Requirements | Status |
|-------|------|----------|--------------|--------|
| **A — Baseline Stripping** | Catalog and remove all correction infrastructure; measure the true physics-only baseline | 2 weeks | BASELINE-01, BASELINE-02, BASELINE-03 | 📋 in planning |
| **B — Physics Fixes** | Solar distribution (ISO 13790), thermal-mass time constant, free-floating temperature fixes | 18 weeks (3 sub-phases B.1/B.2/B.3) | PHYSICS-01, PHYSICS-02, PHYSICS-03 | 📋 in planning |
| **C — Benchmark Correction** | Replace "calibrated for 5R1C" ranges with true EnergyPlus/ESP-r/TRNSYS reference values | 4 weeks | BENCH-01 | 📋 in planning |
| **D — Blind Validation Pass** | Run the full blind suite targeting ≥80% pass | 4 weeks | VALIDATE-01 | 📋 in planning |
| **E — Sustained Validation** | CI gate + regression tracking to hold the pass rate as code evolves | Ongoing | SUSTAIN-01, SUSTAIN-02 | 📋 in planning |

### v1.3 Requirements Index

| ID | Requirement | Phase |
|----|-------------|-------|
| BASELINE-01 | Complete correction factors inventory documented | A |
| BASELINE-02 | All corrections marked with removal TODO markers in source | A |
| BASELINE-03 | True baseline failure state measured and documented | A |
| PHYSICS-01 | Solar distribution matches ISO 13790 / reference hourly profiles | B.1 |
| PHYSICS-02 | Thermal time constant τ matches ISO 13790 calculated values | B.2 |
| PHYSICS-03 | Free-floating temperatures match reference diurnal swing (±2°C) | B.3 |
| BENCH-01 | True ASHRAE reference data replaces calibrated ranges | C |
| VALIDATE-01 | 80%+ cases pass all tolerance bands in blind mode | D |
| SUSTAIN-01 | CI gate prevents merges below 80% pass rate | E |
| SUSTAIN-02 | Annual re-validation against latest ASHRAE reference data | E |

**Definition of Done:** blind execution (no case ID / case-type hint), zero correction factors, true ASHRAE reference values, pass tolerances of ±15% annual energy / ±10% monthly energy / ±15% peak loads / ±1.0°C free-floating temperature, and ≥80% case coverage. See `.planning/ASHRAE_140_BLIND_VALIDATION_PLAN.md` for the full spec, roadmap, and test approach.

### Codebase Stats (post-crate-split, see `STACK.md` / `STRUCTURE.md`)

- **Workspace crates:** 11 (1 root + 10 members) — see `STRUCTURE.md`
- **Rust edition:** 2021, toolchain pinned to stable via `rust-toolchain.toml`
- **Feature flags:** default = none; opt in via `--features` (see `AGENTS.md` §Toolchain Quirks)

---

## Milestone History

<details>
<summary>v1.2 — Validation & Testing Completion (SHIPPED 2026-04-08)</summary>

**Milestone:** v1.2 "Validation & Testing Completion" ✅ SHIPPED 2026-04-08
**Phases completed:** 4 (Phases 44–47), completing the v1.1 deferred work.

**Key accomplishments:**
- High-mass building physics validation and thermal-mass diagnostics (Phase 44)
- Advanced cross-validation infrastructure (ESP-r integration) and testing automation (Phase 45)
- Expanded ASHRAE 140 case coverage toward the 500-699 series (Phase 46)
- Performance validation, benchmarking, and CI/CD regression testing (Phase 47)

**Scope:** 20 requirements spanning validation completion, new test cases, automated testing infrastructure, and performance validation. See `.planning/MILESTONE_v1.2_SUMMARY.md` for the full scope and `.planning/v1.2-MILESTONE-VERIFICATION.md` for verification.

</details>

<details>
<summary>v1.1 — ASHRAE 140 Completion (Partial) (SHIPPED 2026-04-08)</summary>

**Milestone:** v1.1 ASHRAE 140 Completion (Partial) ✅ SHIPPED 2026-04-08
**Status:** Partial ASHRAE 140 compliance with expanded validation framework
**Phases completed:** 1 (Phase 40), 9 plans, 27 tasks. 6/16 v1.1 requirements completed (37.5%).

**Key accomplishments:**
- ASHRAE 140 case expansion: Cases 800-810 (HVAC equipment) and 195-470 (diagnostic validation)
- Extended reference database: 96,361 + 2,417,761 data rows with complete hourly coverage
- Cross-validation framework: EnergyPlus and TRNSYS adapters with file-based comparison
- CLI integration: Validation commands for case execution and cross-validation
- Performance infrastructure: Parallel execution framework and monitoring

**Validated requirements:** CASE-01, CASE-02, CASE-03, CROSS-01, CROSS-02, PERF-01, MZ-01–MZ-10.
**Deferred to v1.2:** CASE-04, CROSS-03-05, MASS-01-04, PERF-02-04.

</details>

<details>
<summary>v1.0 — Multi-Zone Support (SHIPPED 2026-04-07)</summary>

**Milestone:** v1.0 Multi-Zone Support ✅ SHIPPED 2026-04-07
**Phases completed:** 3 (M1–M3), 12 plans, 36 tasks. 100% requirements completion (10/10).

**Key accomplishments:**
- Multi-zone thermal network foundation with N-zone support
- Independent zone-level HVAC controls with per-zone setpoints
- ASHRAE 140 multi-zone validation framework (Case 960/970)
- Energy conservation verified within 1W tolerance
- Performance maintained: <50ms per timestep for 10-zone simulations
- Full Python API and CLI support for multi-zone configuration

</details>

<details>
<summary>v0.8 — Peak Load & Free-Float Validation (SHIPPED 2026-04-07)</summary>

**Milestone:** v0.8 Peak Load & Free-Float Validation ✅ SHIPPED 2026-04-07

- Peak loads within ±10% for all ASHRAE 140 cases
- Free-floating temperature max/min within ±0.5°C of reference
- Hourly profile alignment with EnergyPlus/ESP-r/TRNSYS references
- Performance: 1,237 configs/sec maintained

> **Note:** The v0.8 "near-passing" framing is superseded by the current v1.3 blind-validation figures above. See `docs/archive/ASHRAE140_RESULTS_v0.8.0.md` for the historical snapshot (archived per Issue #2764).

</details>

---

## Constraints

- **ASHRAE 140 Tolerance Bands:** ±15% annual energy, ±10% monthly energy, ±15% peak loads, ±1.0°C free-floating temperature (blind validation targets)
- **ISO 13790 Compliance:** Maintain 5R1C thermal network structure unless alternative approach proven superior (gauge-solver / finite-volume work pending per Phase B)
- **Performance:** Maintain ≥150 configs/sec throughput (release gate); design target ~900 configs/sec in release mode
- **Latency:** ≤10 ms/config (release gate)
- **Backwards Compatibility:** Preserve BatchOracle/Model API for Python users
- **No Parameter Tuning:** Fix the underlying math, never tune to make tests pass (`RULES.md`)
- **Documentation:** All public APIs must have docstrings and examples

---

## Key Decisions

This section records architectural and process decisions made across v0.4 → v1.3.

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Physics-first approach | Address accuracy before optimization to avoid optimizing incorrect physics | ✅ Successful — Analytical physics path validated, mocks removed |
| Trait-based swap points | `HeatConductionSolver`, `VentilationSchedule`, `ThermalModelTrait` make physics / AI-surrogate implementations interchangeable | ✅ Successful — Configurable building assemblies, ML-surrogate swap-point traits |
| Comprehensive HVAC modeling | Implement all major equipment types for ASHRAE 140 compliance | ✅ Successful — VAV, CAV, HeatPump, Chiller, Boiler all implemented |
| Psychrometrics compliance | ASHRAE-compliant calculations for weather integration | ✅ Successful — Dew point, humidity ratio, enthalpy, wet-bulb validated |
| Statistical validation framework | Addendum B acceptance criteria (NMBE, CV(RMSE), FDR corrections, group validation) | ✅ Successful |
| Data quality finalization | Remove all mocks, replace hardcodes, document parameters | ✅ Successful — 37/37 requirements satisfied, codebase audited |
| 8R3C not recommended | Research shows no accuracy improvement, significant performance penalty | ✅ Correct — Avoided 2,000+ lines of physics code for no benefit |
| v0.8 peak load focus | Address peak load accuracy before annual energy improvements | ✅ Successful — Peak loads within ±10% tolerance |
| Crate split (#1255, #1349, #2462, #2463) | Break cycle dependencies; extract leaf modules into `fluxion-core`; isolate CFD / city / fluid / MCP / WASM / TOON / twin | ✅ Successful — 11-crate workspace with cycle-breaking CI guards |
| BatchOracle population-level parallelism | `Clone`-by-design `ThermalModel` + rayon `par_iter()` at the population level only (no nested parallelism) | ✅ Successful — enforced by `.githooks/batch-oracle-check.sh` |
| Blind ASHRAE 140 validation (v1.3) | Replace "informed" validation with physics-only blind methodology; no calibration factors | 🚧 In progress — Phase A baseline established |

---

## User Feedback Themes

*No formal user feedback collected yet — Fluxion is not externally released. Internal validation status is reported in `SCORECARD.md` (auto-generated) and `docs/ASHRAE140_RESULTS.md`.*

---

## Technical Debt

**Blocking technical debt** from the v1.3 perspective — the engine is **not yet ASHRAE 140-compliant** and is not production-ready:

- **High-mass annual energy (Case 900 series):** ~200% over-prediction — fundamental 5R1C/CTF thermal-mass limitation. Fix path is Phase B (PHYSICS-02) and/or the planned `gauge-solver` finite-volume work.
- **Low-mass peak loads (Case 600 series):** envelope model over-predicts peak loads — Phase B (PHYSICS-01, solar distribution).
- **Free-floating temperatures:** extreme temperature failures in FF cases — Phase B (PHYSICS-03).
- **Calibrated benchmark ranges:** "calibrated for 5R1C" ranges must be replaced with true ASHRAE reference values — Phase C (BENCH-01).

See `docs/KNOWN_ISSUES.md` for the open physics limitations (CI gate `scripts/check_known_issues_stale.py` fails if the `*Last Updated: YYYY-MM-DD*` line is >60 days old).

---

*Last updated: 2026-08-11 — reflects v1.3 Blind ASHRAE 140 Validation in progress.*
