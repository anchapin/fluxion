---
phase: 16
slug: psychrometrics-module
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-03-13
---

# Phase 16 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust built-in) |
| **Config file** | .clippy.toml (clippy config) |
| **Quick run command** | `cargo test psychrometrics --lib` |
| **Full suite command** | `cargo test --lib` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test psychrometrics --lib`
- **After every plan wave:** Run `cargo test --lib`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 16-01-01 | 01 | 1 | WEATHER-02 | unit | `cargo test psychrometrics --lib` | ❌ W0 | ⬜ pending |
| 16-01-02 | 01 | 1 | WEATHER-02 | unit | `cargo test psychrometrics --lib` | ❌ W0 | ⬜ pending |
| 16-01-03 | 01 | 1 | WEATHER-02 | unit | `cargo test psychrometrics --lib` | ❌ W0 | ⬜ pending |
| 16-02-01 | 02 | 1 | WEATHER-02 | unit | `cargo test psychrometrics --lib` | ❌ W0 | ⬜ pending |
| 16-02-02 | 02 | 1 | WEATHER-02 | unit | `cargo test psychrometrics --lib` | ❌ W0 | ⬜ pending |
| 16-03-01 | 03 | 1 | WEATHER-02 | unit | `cargo test psychrometrics --lib` | ❌ W0 | ⬜ pending |
| 16-03-02 | 03 | 1 | WEATHER-02 | unit | `cargo test psychrometrics --lib` | ❌ W0 | ⬜ pending |
| 16-04-01 | 04 | 2 | WEATHER-02 | integration | `cargo test economizer --lib` | ❌ W0 | ⬜ pending |
| 16-04-02 | 04 | 2 | WEATHER-02 | integration | `cargo test economizer --lib` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `src/weather/psychrometrics.rs` — psychrometric calculation functions with test stubs
- [ ] `src/weather/mod.rs` — module exports
- [ ] `src/sim/hvac/economizer.rs` — integration test stubs

*Note: Existing infrastructure covers framework; psychrometrics.rs is new module.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Review ASHRAE reference value comparison | WEATHER-02 | Reference tables may require human verification | Compare test output against ASHRAE Fundamentals Chapter 1 tables for selected points |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
