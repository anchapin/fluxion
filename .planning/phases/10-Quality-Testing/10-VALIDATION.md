---
phase: 10
slug: Quality-Testing
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-12
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust built-in) + cargo-tarpaulin (coverage) + proptest (property-based) |
| **Config file** | Cargo.toml + .cargo/config.toml |
| **Quick run command** | `cargo test` |
| **Full suite command** | `cargo test --all-features && cargo tarpaulin --out Xml` |
| **Estimated runtime** | ~120 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test` (quick subset if possible)
- **After every plan wave:** Run `cargo test --all-features && cargo tarpaulin --out Xml`
- **Before `/gsd:verify-work`:** Full suite must be green with >80% coverage
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 1 | TEST-01 | integration | `cargo test --lib` | ✅ Cargo.toml | ⬜ pending |
| 10-02-01 | 02 | 1 | TEST-02 | property | `cargo test --test thermal_invariants` | ⬜ pending W0 | ⬜ pending |
| 10-03-01 | 03 | 2 | TEST-03 | integration | `cargo test --test test_edge_cases` | ⬜ pending W0 | ⬜ pending |
| 10-04-01 | 04 | 2 | TEST-04 | regression | `cargo test --seeded --threads 1` (10x) | ⬜ pending W0 | ⬜ pending |
| 10-05-01 | 05 | 3 | TEST-05 | benchmark | `cargo bench -- --save-baseline phase10` | ⬜ pending W0 | ⬜ pending |
| 10-06-01 | 06 | 3 | TEST-06 | unit | `cargo test --test test_isolation` | ⬜ pending W0 | ⬜ pending |
| 10-07-01 | 07 | 3 | BUG-04 | unit | `cargo test --test test_flaky_fixes` | ⬜ pending W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `Cargo.toml` — proptest dependency (Plan 10-01)
- [ ] `tests/properties/thermal_invariants.rs` — property-based test framework (TEST-02, Plan 10-02)
- [ ] `tests/test_edge_cases.rs` — integration tests for edge cases (TEST-03, Plan 10-03)
- [ ] `tests/test_deterministic_parallel.rs` — seeded thread pool tests (TEST-04, Plan 10-04)
- [ ] `benches/baseline.toml` — performance baselines (TEST-05, Plan 10-05)
- [ ] `tests/test_isolation.rs` — test isolation verification (TEST-06, Plan 10-06)
- [ ] `tests/test_flaky_fixes.rs` — BUG-04 fixes for flaky tests (Plan 10-07)
- [x] `cargo install proptest` — property-based testing framework (Plan 10-01 user_setup)

*Wave 0 creates the test infrastructure framework for all phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Coverage report review | TEST-01 | Requires human judgment on acceptable exclusions | 1. Review generated coverage report (cobertura.xml) 2. Verify uncovered lines are acceptable (debug code, unsafe blocks) 3. Document exclusions in tests/coverage_exclusions.toml |
| Performance trend analysis | TEST-05 | CI gating automated, but trend interpretation manual | 1. Compare new benchmark results against phase10 baseline 2. Verify variance <5% across runs 3. Document any regressions in docs/PERFORMANCE.md |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 120s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
