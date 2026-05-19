# Fluxion Issue Wave Plan

**Status**: ✅ Complete
**PM Agent**: SeniorProjectManager
**Date**: 2026-05-16
**Total Open Issues**: 43
**Open PRs**: 0
**Current Branch**: `fix/issue-853-ci-failures`

---

## Situation Summary

Fluxion is a building energy modeling engine (Rust + Python) targeting **v1.3: ASHRAE 140 Blind Validation (Physics Only)**. The project is in Phase A (baseline stripping) of the v1.3 roadmap with all 5 sub-phases at 0% progress.

**43 open issues** fall into these categories:
- **P0 Blockers** (4): CI broken, gates failing — nothing merges until fixed
- **P1 Critical Bugs** (7): Physics accuracy issues blocking validation
- **P2 v1.3 Validation** (8): Reporting gaps and documentation needed for compliance
- **P3 Architecture/Performance** (5): Refactoring and profiling for future health
- **Backlog** (19): Ecosystem integration, AI/surrogate, research — post-v1.3

---

## Wave Plan: Next 4 Parallel Workstreams

### 🔴 WAVE 1: CI Unblock (P0 — Immediate, Sequential)

> **Nothing else ships until CI is green. Must land first.**

| Task | Issue | Agent | Acceptance Criteria | Dependencies |
|------|-------|-------|---------------------|--------------|
| T1.1 Fix format trait errors | #853 (partial) | debug-investigator | `{:.3f}` → `{:.3}` in `thermal_mass_energy_accounting.rs`; compiles under `pr821-diag` feature | None |
| T1.2 Fix MAE test data | #853 (partial) | debug-investigator | `test_quality_metrics_mae_calculation` passes; test data corrected | None |
| T1.3 Fix CI matrix failures | #853 (partial) | debug-investigator | All 4 matrix jobs green: `multi-zone`, `no-default`, `wiring-tracing`, `Code Coverage` | T1.1 |
| T1.4 Raise CI validation gate threshold | #790 | backend-engineer | Gate threshold raised to 15; 2 structurally-limited cases suppressed; new regressions still caught | T1.3 |

**Timeline**: 1-2 days

---

### 🟠 WAVE 2: Physics Accuracy Fixes (P1 — Core Validation, Partially Parallel)

> **These are the root causes of validation failures. Sequential where dependencies exist.**

| Task | Issue | Agent | Acceptance Criteria | Dependencies |
|------|-------|-------|---------------------|--------------|
| T2.1 Ground temp boundary for floor slab | #746 | debug-investigator | Kusuda-Achenbach ground temp model implemented per Annex B §B3.3; floor slab uses ground BC not fixed 10°C | Wave 1 |
| T2.2 Shading overlap fix | #747 | debug-investigator | Overhang+fin shadow overlap corrected for Cases 630/930; no double-counting | Wave 1 |
| T2.3 Warm-up / pre-conditioning | #744 | backend-engineer | Annual simulation runs warm-up period per §B2; results represent periodic steady state | Wave 1 |
| T2.4 600-series heating investigation | #851, #803 | debug-investigator | Root cause identified for ~2x heating overestimate; fix brings within 2x of reference | T2.1, T2.3 |
| T2.5 Solar gain distribution to mass node | #700 | backend-engineer | Solar radiation correctly distributed to thermal mass node per ISO 13790 | T2.4 (may inform) |
| T2.6 Window properties for high-mass | #699 | debug-investigator | Window U-value and SHGC verified against ASHRAE 140 spec for 900-series | Wave 1 |
| T2.7 Ground coupling / h_tr_floor | #680 | backend-engineer | h_tr_floor calculated from physics; not hardcoded | T2.1 |

**Timeline**: 2-4 weeks (overlapping)

---

### 🟡 WAVE 3: Reporting & Documentation (P2 — Compliance Gaps)

> **Section 8 output requirements and certification docs. Can run in parallel with Wave 2.**

| Task | Issue | Agent | Acceptance Criteria | Dependencies |
|------|-------|-------|---------------------|--------------|
| T3.1 Peak load timestamps | #761, #749-G3 | backend-engineer | ValidationResult includes peak load date+hour | None |
| T3.2 IncidentSolar metric type | #762, #749-G4 | backend-engineer | Per-surface incident solar radiation reported | None |
| T3.3 Hourly FF temperature profiles | #763, #749-G5 | backend-engineer | Full hourly zone temperature profiles stored for free-float cases | None |
| T3.4 Output units correct (G1, G3, G5) | From recent PRs | backend-engineer | All output metrics use correct SI units per Section 8 | None |
| T3.5 Compliance documentation | #750, #749 | docs-curator | Missing ASHRAE 140 certification submission docs created | T3.1-T3.3 |
| T3.6 Empirical corrections removal | #724, #739 | backend-engineer | No correction factors in validation harness; clean physics-only model | Wave 2 complete |
| T3.7 Placeholder conductance removal | #731 | backend-engineer | All thermal conductances in `ashrae_140_cases.rs` are physics-calculated | T2.7 |

**Timeline**: 2-3 weeks (parallel with Wave 2)

---

### 🟢 WAVE 4: Architecture & Performance (P3 — After Validation Passes)

> **Health and performance. Only start after Waves 1-3 are substantially complete.**

| Task | Issue | Agent | Acceptance Criteria | Dependencies |
|------|-------|-------|---------------------|--------------|
| T4.1 SolverManager deprecation | #712 | backend-engineer | All code uses `step_all()`; deprecated methods removed | Wave 2 |
| T4.2 Promote FD solver for high-mass | #726 | backend-engineer | FD solver is primary for 900-series; CTF not used for high-mass | T2.5, T2.7 |
| T4.3 Flamegraph profiling | #721 | qa-reviewer | Hot-path identified; optimization targets documented | Wave 2 |
| T4.4 Solver abstraction review | #728 | architecture-reviewer | Architecture review produced; zone model hierarchy documented | T4.2 |
| T4.5 Surrogate benchmarking | #720 | qa-reviewer | ONNX inference timing vs physics baseline documented | Wave 2 |

**Timeline**: 3-4 weeks (after Wave 2)

---

## Explicitly Deferred (Backlog, Post-v1.3)

These 19 issues are tagged `backlog`, `ai`, or `v2.x` and should NOT be worked on until v1.3 ships:

- **Ecosystem Integration Epic** #777 and children (#778-#785): IDF, OSM, gbXML, IFC, Python bindings
- **AI/Surrogate** #718, #719, #764: Training data, ONNX export, MLP architecture
- **Research** #751, #781: Standards landscape, ecosystem survey
- **900-series specific** #703-#705, #714-#715: Will be resolved by Wave 2 physics fixes
- **Architecture** #728: Solver review (moved to Wave 4)

---

## Dependency Graph (Simplified)

```
Wave 1 (CI) ──┬── T1.1 Format fixes
              ├── T1.2 MAE test fix
              ├── T1.3 Matrix fixes ──→ T1.4 Gate threshold
              │
Wave 2 (Physics) ─┬── T2.1 Ground temp ──→ T2.7 Ground coupling
                   ├── T2.2 Shading fix
                   ├── T2.3 Warm-up ──→ T2.4 600-series investigation
                   ├── T2.5 Solar gain ──→ (informed by T2.4)
                   └── T2.6 Window props
                   │
Wave 3 (Reporting) ─┬── T3.1 Peak timestamps ──┐
                    ├── T3.2 IncidentSolar     ├── T3.5 Compliance docs
                    └── T3.3 Hourly profiles ──┘
                        T3.6 Remove corrections (after Wave 2)
                        T3.7 Remove placeholders (after T2.7)
                   │
Wave 4 (Arch/Perf) ──── After Waves 2+3
```

---

## Recommended Execution Order

1. **Today**: Complete #853 (CI fix) — already on branch `fix/issue-853-ci-failures`
2. **This week**: Merge CI fix → Start T1.4 (gate threshold) + T2.1, T2.2, T2.3 in parallel
3. **Week 2**: T2.4 investigation (blocks on T2.1, T2.3 results) + T3.1-T3.3 reporting tasks
4. **Week 3-4**: Complete remaining Wave 2 physics fixes + T3.5-T3.7 cleanup
5. **Week 5+**: Wave 4 architecture/performance

---

## Files Changed
- `.agents/results/result-pm.md` — this plan

## Acceptance Criteria Checklist
- [x] All 43 open issues categorized and prioritized
- [x] Dependency chain identified (CI first, then physics, then reporting)
- [x] 900-series / ecosystem / AI issues explicitly deferred
- [x] Each task has assigned agent, acceptance criteria, and dependencies
- [x] Timeline estimates realistic based on issue complexity
