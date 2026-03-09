---
phase: 1
slug: foundation
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-03-09
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (via rstest) |
| **Config file** | None (use rstest config) |
| **Quick run command** | `cargo test` |
| **Full suite command** | `cargo test --all` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test`
- **After every plan wave:** Run `cargo test --all`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|-----------|------|-------|------------|-------------------|---------------|--------|
| 01-01 | 01 | 1 | REQ-01 | unit | `cargo test` | ✅ pending | ⬜ pending |
| 01-02 | 01 | 2 | REQ-01 | unit | `cargo test` | ✅ pending | ⬜ pending |
| 01-03 | 01 | 3 | REQ-01 | unit | `cargo test` | ✅ pending | ⬜ pending |
| 01-04 | 02 | 1 | THERM-01 | unit | `cargo test` | ✅ pending | ⬜ pending |
| 01-05 | 02 | 2 | THERM-02 | unit | `cargo test` | ✅ pending | ⬜ pending |
| 02-01 | 02 | 1 | REQ-01 | unit | `cargo test` | ✅ pending | ⬜ pending |
| 02-02 | 02 | 2 | REQ-01 | unit | `cargo test` | ✅ pending | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_conductance_calculations.py` — stubs for REQ-01
- [ ] `tests/conftest.py` — shared fixtures
- [ ] `pytest` installation — if no framework detected
- [ ] `rstest` installation — if no framework detected

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| {behavior} | REQ-{XX} | {reason} | {steps} |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending YYYY-MM-DD
