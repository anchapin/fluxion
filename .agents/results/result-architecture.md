# Architecture Review: Bottom-Up Physics Validation Restructuring

**Status**: COMPLETE
**Date**: 2026-06-11
**Scope**: Strategic pivot from top-down ASHRAE 140 calibration to bottom-up module validation

---

## Problem Statement

The Fluxion engine was stuck in a "vibe coding" loop: running ASHRAE 140 system-level tests, observing failures, and tuning individual parameters (furniture factors, h_tr_ms, ventilation coefficients) to force pass rates higher. This approach:
1. Cannot distinguish which module (solar, conduction, ventilation, zone balance) is causing failures
2. Encourages parameter-tweaking over physics correctness
3. Makes ML surrogate integration impossible (can't swap unverified modules)

## Diagnosis

The root cause is **missing module boundaries**: the codebase has 241 Rust source files with significant coupling between `sim/` and `physics/` modules. The existing `HeatConductionSolver` trait is a good pattern but is undermined by:
- Trait methods taking `BuildingAssembly` (full building context leaks into solver)
- No isolated unit tests against known-correct reference data
- 53 open GitHub issues attempting to calibrate a black box

## Recommendation

### Adopted: Bottom-Up Module Validation

**Phase 1** (Completed):
- Freeze 53 vibe-coding issues with clear explanation
- Create `ARCHITECTURE.md` as single source of truth with Mermaid diagrams
- Define explicit input/output contracts for all 5 modules
- Create reference data directory structure

**Phase 2** (Next):
- Generate EnergyPlus reference CSVs for isolated physics scenarios
- Solar position, surface irradiance, conduction step response, ventilation ACH

**Phase 3** (After data):
- Refactor solar calculations into pure functions with no building dependencies
- Test conduction solvers (FD, CTF, 5R1C) against step-response CSVs
- Each module must match E+ within 1% tolerance before integration

**Phase 4** (After modules verified):
- Reconnect modules, resume ASHRAE 140
- Any failure now points to a specific module (we have per-module tests)

### Rejected: Continue Top-Down Calibration
- Would continue the vibe-coding cycle
- No diagnostic power when tests fail
- Parameter tuning masks deeper mathematical errors

---

## Tradeoffs

| Factor | Bottom-Up (Adopted) | Top-Down (Rejected) |
|--------|---------------------|---------------------|
| Time to first passing test | Longer (must generate E+ data first) | Immediate (tweak params) |
| Diagnostic power | High (per-module tests) | Low (only system tests) |
| ML surrogate readiness | High (traits already defined) | Low (coupled code) |
| Confidence in results | High (matches E+ module-by-module) | Low (tuned to look right) |
| Team complexity | Low (work on one module at a time) | High (everything coupled) |

## Risks

1. **EnergyPlus availability**: Must have E+ installed to generate reference data. Mitigation: Pre-generated CSVs can be committed to the repo.
2. **1% tolerance may be too tight**: E+ itself has internal tolerances. Mitigation: Start with 1%, relax to 2% if E+ self-consistency is the limit.
3. **CTF solver may fail step-response**: Known architectural limitation. Mitigation: This is exactly the diagnostic power we gain — knowing CTF is wrong for high-mass is valuable.
4. **Scope creep**: Could spend too long on individual modules. Mitigation: Each module has a clear "done" criterion (test passes).

## Validation Steps

- [x] 53 issues frozen with explanation comments
- [x] ARCHITECTURE.md created with Mermaid diagrams and module I/O tables
- [x] 5 new issues created (#942-#946) with acceptance criteria
- [x] Reference data directory structure created
- [ ] E+ reference CSVs generated (blocked on EnergyPlus availability)
- [ ] Solar module isolated and tested
- [ ] Conduction module tested against step-response

## Artifacts Created

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` | Source of truth for module boundaries and interfaces |
| `tests/reference_data/README.md` | Protocol for generating E+ reference data |
| `tests/reference_data/{solar,conduction,ventilation,zone_balance,energyplus_models}/` | Directory structure for reference CSVs |
| GitHub Issues #942-#946 | Tracked work items for Phase 1-3 |

## GitHub Issue Summary

| Issue | Title | Phase |
|-------|-------|-------|
| #942 | ARCHITECTURE.md — Define strict module inputs/outputs | Phase 1 |
| #943 | Implement ThermalModel trait for ML surrogate prep | Phase 1 |
| #944 | Generate isolated EnergyPlus CSV reference data | Phase 2 |
| #945 | Isolate Solar Module and test against E+ | Phase 3 |
| #946 | Isolate 1D Conduction Module and test against E+ | Phase 3 |

## Kept Open (Not Frozen)

| Issue | Title | Why Kept |
|-------|-------|----------|
| #728 | Architecture review: solver abstraction | Aligned with new approach |
| #726 | Promote FD solver for high-mass | Critical for conduction testing |
| #724 | Validation harness empirical corrections | Must remove before clean testing |
| #739 | Empirical correction fields | Architecture bug |

## Follow-Up: Architecture Drift Detection

After the initial restructuring, we added automated drift prevention:

### AGENTS.md Update
Added a "Required Reading" section to AGENTS.md that:
- Instructs AI agents to read ARCHITECTURE.md before working on any issue
- Documents the current validation strategy (Phase 1: Module Isolation)
- Lists the 3 key traits (HeatConductionSolver, VentilationSchedule, ThermalModelTrait)
- Provides a module boundary diagram

### Drift Detection Script
`scripts/check_architecture_drift.py` — Python script that:
1. Scans all `src/**/*.rs` for `pub trait` definitions
2. Cross-references against ARCHITECTURE.md (trait names, file paths)
3. Checks key module files still exist
4. Exits 1 if drift detected (with actionable remediation)
5. Exits 0 if everything is consistent

### CI Workflow
`.github/workflows/architecture_drift.yml` — Runs:
- **Nightly** at 03:00 UTC
- **On PR** that touches `src/**/*.rs` or `ARCHITECTURE.md`
- **On demand** via workflow_dispatch
- Auto-comments on PRs if drift detected

### Design Decision: Validate, Don't Auto-Generate
We chose drift detection over auto-generated Mermaid diagrams because:
- ARCHITECTURE.md is curated — shows the *target* architecture, not messy reality
- Auto-generated diagrams from 241 Rust files would be noisy and overwhelming
- Drift detection prevents staleness; auto-generation only documents it
- The curated diagrams are more useful for AI context than raw code dumps
| #712 | Migrate to step_all() | Infrastructure |
| #764 | Phase 4a MLP Surrogate Architecture | ML prep, aligned |
