---
phase: 4
slug: multi-zone-inter-zone-transfer
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-09
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (built-in Rust test framework) |
| **Config file** | None (validation tolerances in benchmark.rs) |
| **Quick run command** | `cargo test test_case_960_sunspace -- --nocapture` |
| **Full suite command** | `cargo test ashrae_140 -- --nocapture` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test test_case_960_sunspace -- --nocapture`
- **After every plan wave:** Run `cargo test ashrae_140 -- --nocapture`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | MULTI-01 | unit | `cargo test test_interzone_conductance_calculation -- --nocapture` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 1 | MULTI-01 | unit | `cargo test test_stefan_boltzmann_radiation -- --nocapture` | ❌ W0 | ⬜ pending |
| 04-01-03 | 01 | 1 | MULTI-01 | unit | `cargo test test_stack_effect_ach -- --nocapture` | ❌ W0 | ⬜ pending |
| 04-01-04 | 01 | 1 | MULTI-01 | unit | `cargo test test_directional_conductance -- --nocapture` | ❌ W0 | ⬜ pending |
| 04-02-01 | 02 | 1 | MULTI-01 | integration | `cargo test ashrae_140_validation::test_case_960_sunspace_simulation -- --nocapture` | ✅ | ⬜ pending |
| 04-03-01 | 03 | 1 | MULTI-01 | system | `cargo test ashrae_140 -- --nocapture` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_interzone_conductance.rs` — validate h_tr_iz = A/R calculation against Case 960 specs
- [ ] `tests/test_stefan_boltzmann_radiation.rs` — test full nonlinear radiative exchange vs linearized approximation
- [ ] `tests/test_stack_effect_ach.rs` — validate stack effect ACH formula and air enthalpy calculation
- [ ] `tests/test_directional_conductance.rs` — test bidirectional h_tr_iz for asymmetric insulation

*Wave 0 creates test infrastructure for all three physics components (conductance, radiation, ventilation).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Zone temperature gradients match ASHRAE reference | MULTI-01 | Requires visual inspection of hourly temperature profiles for physical reasonableness | Run `cargo test test_case_960_sunspace -- --nocapture`, check that sunspace temperature is between outdoor and back-zone temperatures, verify typical ΔT ≈ 2-5°C for sunspace buildings |
| Radiative heat transfer sign correctness | MULTI-01 | Difficult to test automatically with realistic temperature swings | Manually inspect debug output: radiative heat transfer should be positive when T_sunspace > T_back-zone, negative when T_sunspace < T_back-zone |

*Two manual verifications for physical reasonableness and sign correctness.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
