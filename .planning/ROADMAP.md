# Fluxion Roadmap

**Project:** Building Energy Modeling Engine (Rust + Python)
**Milestone:** v1.3 Blind ASHRAE 140 Validation (PHYSICS ONLY)
**Current Phase:** Planning
**Last Updated:** 2026-05-05

---

## Milestones

- 🚧 **v1.3 Blind ASHRAE 140 Validation (Physics Only)** — Phases A-E (planning)
- ✅ **v1.2 Validation & Testing Completion** — Phases 44-47 (shipped 2026-04-08)
- ✅ **v1.1 ASHRAE 140 Completion (Partial)** — Phase 40 (shipped 2026-04-08)
- ✅ **v1.0 Multi-Zone Support** — Phases M1-M3 (shipped 2026-04-07)
- ✅ **v0.8 Peak Load & Free-Float Validation** — Phases 33-36 (shipped 2026-04-07)

---

## v1.3: Blind ASHRAE 140 Validation (Physics Only)

**Goal:** Achieve ASHRAE 140 validation with true blind test methodology — no calibration factors, no case-specific corrections, physics-only model.

**Why:** Current validation is "informed" not "blind" — case IDs are known before simulation, correction factors are applied post-simulation, and benchmark ranges are "calibrated for 5R1C." This milestone establishes genuine compliance.

### Phase A: Baseline Stripping
**Duration:** 2 weeks
**Goal:** Remove all corrections, confirm true baseline failure mode.

- [ ] A-01: Catalog all correction infrastructure
- [ ] A-02: Measure true baseline (0% pass rate expected without corrections)

**Requirements:** BASELINE-01, BASELINE-02, BASELINE-03

### Phase B: Physics Fixes
**Duration:** 18 weeks (3 sub-phases)
**Goal:** Fix thermal model so it matches reference results without corrections.

- [ ] B.1: Solar Distribution Fix (Weeks 3-8)
  - Implement detailed sky diffuse / ground reflectance split per ISO 13790
  - Validate against 900 series cases without corrections
- [ ] B.2: Thermal Mass Time Constant Fix (Weeks 9-14)
  - Implement ISO 13790 Table C.2 effective capacitance
  - Verify h_tr_ms calculation from actual construction layers
  - Derive 6R2C corrections from first principles
- [ ] B.3: Free-Floating Temperature Fix (Weeks 15-20)
  - Verify HVAC truly disabled in free-float mode
  - Validate thermal damping matches reference diurnal swing

**Requirements:** PHYSICS-01, PHYSICS-02, PHYSICS-03

### Phase C: Benchmark Correction
**Duration:** 4 weeks
**Goal:** Replace "calibrated for 5R1C" benchmark ranges with true ASHRAE 140 reference values.

**Requirements:** BENCH-01

### Phase D: Blind Validation Pass
**Duration:** 4 weeks
**Goal:** Run full ASHRAE 140 blind validation suite and achieve 80%+ pass rate.

**Requirements:** VALIDATE-01

### Phase E: Sustained Validation
**Duration:** Ongoing
**Goal:** Maintain blind validation pass rate as code evolves.

**Requirements:** SUSTAIN-01, SUSTAIN-02

---

## Phase Status (Auto-synced)

| Phase | Progress | Status | Completed |
|-------|----------|--------|-----------|
| A-baseline-stripping | 0/2 | 📋 planning | |
| B-physics-fixes | 0/0 | 📋 planning | |
| C-benchmark-correction | 0/0 | 📋 planning | |
| D-blind-validation-pass | 0/0 | 📋 planning | |
| E-sustained-validation | 0/0 | 📋 planning | |

---

## v1.2 Status (Archived)

See .planning/ROADMAP_v1.2.md for archived v1.2 milestone details.

---

## v1.1 Status (Archived)

**Milestone:** v1.1 ASHRAE 140 Completion (Partial) ✅ SHIPPED 2026-04-08

**Requirements completed:** CASE-01, CASE-02, CASE-03, CROSS-01, CROSS-02, PERF-01, MZ-01-MZ-10

**Deferred:** CASE-04, CROSS-03-05, MASS-01-04, PERF-02-04 (moved to v1.3)

---

## Phase History

| Phase | Milestone | Status | Completed |
|-------|-----------|--------|-----------|
| M1-multi-zone-foundation | v1.0 | ✅ complete | 2026-04-07 |
| M2-zone-hvac-controls | v1.0 | ✅ complete | 2026-04-07 |
| M3-ashrae-140-validation | v1.0 | ✅ complete | 2026-04-07 |
| 31-full-validation-release | v1.1 | ✅ complete | 2026-04-08 |
| 32-ctf-thermal-mass-fix | v1.1 | ✅ complete | 2026-04-08 |
| 33-peak-load-diagnostics | v1.1 | ✅ complete | 2026-04-08 |
| 34-peak-load-physics-fix | v1.1 | ✅ complete | 2026-04-08 |
| 36-01-validation | v1.1 | ✅ complete | 2026-04-08 |
| 36-v0.8.0-release | v1.1 | ✅ complete | 2026-04-08 |
| 40-case-expansion-foundation | v1.1 | ✅ complete | 2026-04-08 |
| 41-high-mass-physics-performance | v1.1 | ✅ complete | 2026-04-08 |
| 44-high-mass-physics-validation | v1.1 | ✅ complete | 2026-04-08 |
| 45-advanced-cross-validation-automation | v1.1 | ✅ complete | 2026-04-08 |
| 46-expanded-validation-coverage | v1.1 | ✅ complete | 2026-04-08 |
| 47-performance-validation-optimization | v1.2 | ✅ complete | 2026-04-08 |

---

## Requirements Index

### v1.3 Requirements

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

---

*Last updated: 2026-05-05*