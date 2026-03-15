---
phase: 18
slug: diagnostic-cases
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 18 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test |
| **Config file** | none — Cargo.toml config |
| **Quick run command** | `cargo test ashrae_140_case_195 --lib` |
| **Full suite command** | `cargo test --test ashrae_140_case_195_470 --test ashrae_140_case_800_810` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test ashrae_140 --lib`
- **After every plan wave:** Run `cargo test --test ashrae_140_case_195_470 --test ashrae_140_case_800_810`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 18-01-01 | 01 | 1 | DIAG-01 | unit | `cargo test test_case_195_solid_conduction --lib` | ✅ W0 | ⬜ pending |
| 18-01-02 | 01 | 1 | DIAG-01 | integration | `cargo test --test ashrae_140_case_195_470` | ❌ W1 | ⬜ pending |
| 18-02-01 | 02 | 1 | DIAG-02 | unit | `cargo test test_case_800_hvac_equipment --lib` | ✅ W0 | ⬜ pending |
| 18-02-02 | 02 | 1 | DIAG-02 | integration | `cargo test --test ashrae_140_case_800_810` | ❌ W2 | ⬜ pending |
| 18-03-01 | 03 | 2 | DIAG-03 | unit | `cargo test test_non_residential_cases --lib` | ❌ W3 | ⬜ pending |
| 18-04-01 | 04 | 2 | DIAG-04 | unit | `cargo test test_solar_gain_variants --lib` | ❌ W4 | ⬜ pending |
| 18-05-01 | 05 | 2 | DIAG-05 | integration | `cargo test validate_all_diagnostic_cases` | ❌ W5 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/ashrae_140_case_195_470.rs` — stubs for DIAG-01 (Cases 195-470)
- [ ] `tests/ashrae_140_case_800_810.rs` — stubs for DIAG-02 (Cases 800-810) [exists with TODO markers]
- [ ] `src/validation/ashrae_140_cases.rs` — case spec functions for diagnostic cases [exists with Case 195]
- [ ] `tests/ashrae_140/diagnostics.rs` — consolidated validation module (new)

*Note: Existing infrastructure from Phase 5-17 covers validation framework. Wave 0 adds diagnostic-specific test stubs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Validation report review | DIAG-05 | Requires human judgment on tolerance and diagnostic quality | Run `fluxion validate --full` and review output for passing all cases |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
