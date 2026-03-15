---
phase: 19
slug: statistical-validation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 19 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test |
| **Config file** | Cargo.toml |
| **Quick run command** | `cargo test --lib validation::statistical` |
| **Full suite command** | `cargo test` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --lib validation::statistical`
- **After every plan wave:** Run `cargo test`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 19-01-01 | 01 | 1 | STATS-01 | unit | `cargo test --lib validation::statistical` | ✅ W0 | ⬜ pending |
| 19-01-02 | 01 | 1 | STATS-02 | unit | `cargo test --lib validation::statistical` | ✅ W0 | ⬜ pending |
| 19-01-03 | 01 | 1 | STATS-03 | unit | `cargo test --lib validation::statistical` | ✅ W0 | ⬜ pending |
| 19-01-04 | 01 | 1 | STATS-04 | unit | `cargo test --lib validation::statistical` | ✅ W0 | ⬜ pending |
| 19-02-01 | 02 | 1 | STATS-05 | unit | `cargo test --lib validation::statistical` | ✅ W0 | ⬜ pending |
| 19-03-01 | 03 | 2 | STATS-06 | unit | `cargo test --lib validation::statistical` | ✅ W0 | ⬜ pending |
| 19-04-01 | 04 | 2 | STATS-06 | integration | `cargo test --bin fluxion` | ✅ W0 | ⬜ pending |
| 19-04-02 | 04 | 2 | STATS-06 | integration | `cargo test --bin fluxion` | ✅ W0 | ⬜ pending |
| 19-05-01 | 05 | 3 | STATS-06 | integration | `cargo test test_statistical_validation` | ✅ W0 | ⬜ pending |
| 19-05-02 | 05 | 3 | STATS-06 | integration | `cargo test test_statistical_validation` | ✅ W0 | ⬜ pending |
| 19-05-03 | 05 | 3 | STATS-06 | integration | `cargo test test_statistical_validation` | ✅ W0 | ⬜ pending |
| 19-05-04 | 05 | 3 | STATS-06 | e2e | `grep "## Statistical Validation" docs/ASHRAE140_RESULTS.md` | ⬜ W0 | ⬜ pending |
| 19-05-05 | 05 | 3 | STATS-06 | integration | `cargo test && fluxion validate --all` | ⬜ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `src/validation/statistical.rs` — module structure for STAT-01 through STAT-06
- [ ] `tests/test_statistical_validation.rs` — comprehensive tests for statistical metrics and group validation (W0 file from RESEARCH.md)
- [ ] `tests/test_report_integration.rs` — integration tests for statistical report generation (W0 file from RESEARCH.md)

*Existing infrastructure covers ASHRAE140Validator and validation reporting frameworks.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Statistical compliance report review | STATS-06 | Requires domain expertise to interpret confidence intervals and FDR results | Review generated reports, verify NMBE/CV(RMSE) calculations against manual calculations for known cases |
| ASHRAE 140 Addendum B threshold validation | STATS-01 | Requires verification against external standard | Manually compare computed thresholds with Addendum B specification |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
