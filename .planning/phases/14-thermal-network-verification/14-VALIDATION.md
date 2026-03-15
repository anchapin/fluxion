---
phase: 14
slug: thermal-network-verification
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-13
---

# Phase 14 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust built-in) |
| **Config file** | Cargo.toml (existing) |
| **Quick run command** | `cargo test --lib verification::` |
| **Full suite command** | `cargo test` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --lib verification::`
- **After every plan wave:** Run `cargo test`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 14-01-01 | 01 | 1 | PHYS-01 | unit | `cargo test --lib test_energy_conservation` | ✅ stub | ⬜ pending |
| 14-02-01 | 02 | 2 | PHYS-04 | integration | `cargo test --lib validation::thermal_mass_tests::test_thermal_mass_coupling_ratio_high_mass` | ✅ stub | ⬜ pending |
| 14-03-01 | 03 | 2 | PHYS-05 | integration | `cargo test --lib validation::thermal_mass_tests::test_mode_specific_coupling_factors` | ✅ stub | ⬜ pending |
| 14-04-01 | 04 | 2 | DATA-01 | unit | `cargo test --bin audit_codebase -- --validate` | ✅ stub | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `src/bin/audit_codebase.rs` — stubs for DATA-01 audit tool
- [x] `tests/validation/thermal_mass_tests.rs` — fixtures for PHYS-04/PHYS-05 validation
- [x] Existing infrastructure covers all other phase requirements (cargo test framework)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| ASHRAE 140 full suite comparison | PHYS-01, PHYS-04, PHYS-05 | Requires domain expertise interpretation of 18 cases | Run `fluxion validate --all`, compare against docs/ASHRAE140_RESULTS.md, document deviations |
| Codebase audit review | DATA-01 | Requires human judgment on priority/severity | Review docs/AUDIT_REPORT.md, verify critical findings addressed in issues |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
