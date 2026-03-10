# Phase 7 Planning Overview

**Phase:** 07 - Advanced Analysis & Visualization
**Goal:** Implement sensitivity analysis, delta testing, interactive visualization, component-level energy breakdown, and extensible case specification to support building design optimization and diagnostic investigation.
**Requirements:** 24 requirements across 7 functional areas
**Plans:** 8 implementation plans, structured in 2 waves
**Total Tasks:** 22 tasks

---

## Plan Summary

| Plan | Title | Requirements | Tasks | Wave | Key Files Modified |
|------|------|--------------|-------|------|--------------------|
| 07-01 | Analysis Core: Sensitivity & Delta | SENS-01..04 | 3 | 1 | src/analysis/, Cargo.toml, src/lib.rs |
| 07-02 | Delta Testing Framework | DELTA-01..03 | 3 | 1 | src/analysis/delta.rs |
| 07-03 | Diagnostic Utilities: Components & Swing | COMP-01..03, SWING-01..03 | 3 | 1 | src/analysis/components.rs, src/analysis/swing.rs |
| 07-04 | Interactive Visualization | VIZ-01..04 | 3 | 1 | src/analysis/visualization.rs, src/analysis/mod.rs |
| 07-05 | Multi-Reference Comparison | MREF-01..03 | 2 | 1 | docs/ashrae_140_references.json, src/validation/multi_reference.rs, src/validation/reporter.rs, src/validation/ashrae_140_validator.rs |
| 07-06 | CLI Integration (Initial) | (none) | 2 | 2 | src/bin/fluxion.rs, tests/cli_integration.rs |
| 07-07 | CLI Integration (Completion) | (none) | 2 | 2 | src/bin/fluxion.rs, tests/cli_integration.rs |
| 07-08 | Extensible Case Framework | EXT-01..04 | 4 | 1 | src/validation/ashrae_140_cases.rs, src/validation/assembly_library.rs, config/assemblies.yaml, docs/cases/quickstart.md |

**Wave Structure:**
- **Wave 1 (Plans 01-05, 08):** Core feature implementations and extensibility additions. No overlapping file modifications between plans → parallel execution.
- **Wave 2 (Plans 06-07):** CLI subcommand registration and integration testing. Plan 06 depends on Plan 05; Plan 07 depends on all previous Wave 1 and Plan 06 → sequential execution (06 then 07).

---

## Rationale

### Vertical Slice Organization
Each plan delivers a self-contained functional area, from core logic to CLI wiring. Analysis features (sensitivity, delta, components, swing) are grouped to minimize cross-plan dependencies; visualization, multi-reference, extensibility are separate concerns.

### Conflict Avoidance
- **Scaffolding:** Plan 01 creates the `src/analysis` module and declares submodules. Subsequent analysis plans fill their respective files without conflicting.
- **File ownership:** No two Wave 1 plans modify the same source file. Detailed:
  - Plan 01: creates analysis scaffolding and adds dependencies.
  - Plan 02: fills `src/analysis/delta.rs`.
  - Plan 03: fills `src/analysis/components.rs` and `src/analysis/swing.rs`.
  - Plan 04: creates/fills `src/analysis/visualization.rs`.
  - Plan 05: creates multi-reference modules and modifies validation framework.
  - Plan 08: extends `CaseBuilder`, creates assembly library and docs.
- **Wave 2 plans** both touch `src/bin/fluxion.rs` and `tests/cli_integration.rs`, but they are executed sequentially (06 before 07) to avoid conflicts.

### Task Sizing & Context Budget
Each plan contains 2-4 tasks, targeting ~40% context per plan. Tasks are scoped to 15-60 minutes of Claude execution time with clear verification.

### Verification Strategy
Every task includes an automated `cargo test` command or explicit command check to confirm correctness. No manual-only verification.

---

## Must-Have Distribution

**Overall Phase Must-Haves (derived from success criteria):**

**Truths:**
- Parameter sensitivity studies produce ranked impact metrics.
- Delta test reports show isolated variant effects.
- Interactive HTML visualizations provide zoom/pan and PNG/SVG export.
- Component energy breakdown (annual and hourly) is available for diagnosis.
- Temperature swing metrics (min, max, avg, range, comfort hours) are computed for free-floating cases.
- Swing interpretation identifies thermal mass effectiveness and passive cooling/heating potential.
- Simplified geometry methods and assembly library enable custom case specification.
- Custom EPW weather can be used.
- Multi-program validation results (EnergyPlus, ESP-r, TRNSYS) are compared and reported.
- All Phase 7 features are accessible via the fluxion CLI.

**Artifacts:**
- `src/analysis/sensitivity.rs` (sampling, evaluation, metrics)
- `src/analysis/delta.rs` (YAML parsing, comparison, Markdown/CSV output)
- `src/analysis/components.rs` (component aggregation, conservation check)
- `src/analysis/swing.rs` (swing metrics and interpretation)
- `src/analysis/visualization.rs` (HTML charts + animation)
- Extended `CaseBuilder` in `src/validation/ashrae_140_cases.rs` (simplified geometry, custom weather)
- `src/validation/assembly_library.rs` (assembly library loader)
- `config/assemblies.yaml` (default assemblies)
- `docs/cases/quickstart.md` (extensibility guide)
- `docs/ashrae_140_references.json` (versioned multi-program references)
- `src/validation/multi_reference.rs` (reference DB loader)
- Extensions to `src/validation/reporter.rs` and `src/validation/ashrae_140_validator.rs`
- `src/bin/fluxion.rs` (CLI subcommands)
- `tests/cli_integration.rs` (integration tests)

**Key Links:**
- Sensitivity → BatchOracle (high-throughput evaluation)
- Delta → CaseSpec → simulation pipeline
- Component/Swing → validation diagnostics (EnergyBreakdown, TemperatureProfile)
- Visualization → CSV diagnostics export
- Extensibility → CaseBuilder + assembly library + EPW
- Multi-Reference → BenchmarkData + ValidationReport
- CLI → all analysis and visualization modules

---

## Discovery Level

**Level 0:** All tasks follow established codebase patterns (diagnostics, validation, builder). No new external dependencies beyond those identified in research (`sobol`, `linregress`). Implementation conforms to existing Rust conventions (Result, serde, clap).

---

## Next Steps

Execute: `/gsd:execute-phase 07`

Plans will be executed in wave order:
1. Run all Wave 1 plans (07-01, 07-02, 07-03, 07-04, 07-05, 07-08) in parallel.
2. After Wave 1 completion, run Wave 2 plan 07-06.
3. After 07-06 completion, run Wave 2 plan 07-07.
