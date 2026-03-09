---
phase: 2
slug: Thermal-Mass-Dynamics
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-09
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust) |
| **Config file** | Cargo.toml |
| **Quick run command** | `cargo test -- thermal_mass` |
| **Full suite command** | `cargo test --all` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -- thermal_mass`
- **After every plan wave:** Run `cargo test --all`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|
| 02-01-01 | 01 | FREE-02 | unit | `cargo test -- thermal_mass` | tests/test_thermal_mass.rs | ⬜ pending |
| 02-01-02 | 01 | FREE-02 | unit | `cargo test -- thermal_mass` | tests/test_thermal_mass.rs | ⬜ pending |
| 02-01-03 | 01 | FREE-02 | integration | `cargo test -- thermal_mass` | tests/test_thermal_mass.rs | ⬜ pending |
| 02-02-01 | 02 | TEMP-01 | unit | `cargo test -- thermal_mass` | tests/test_thermal_mass.rs | ⬜ pending |
| 02-02-02 | 02 | TEMP-01 | integration | `cargo test -- thermal_mass` | tests/ashrae_140_validation.rs | ⬜ pending |
| 02-03-01 | 03 | FREE-02, TEMP-01 | integration | `cargo test -- all` | tests/ashrae_140_validation.rs | ⬜ pending |
| 02-03-02 | 03 | FREE-02, TEMP-01 | validation | `cargo test -- all` | docs/ASHRAE140_RESULTS.md | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_thermal_mass.rs` — stubs for FREE-02 (thermal mass dynamics)
- [ ] `tests/test_thermal_mass_integration.rs` — integration tests for thermal mass + HVAC
- [ ] `tests/ashrae_140_validation.rs` — existing validation infrastructure covers Case 900, 900FF

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Free-floating temperature swing reduction | FREE-02 | Requires comparison to ASHRAE 140 reference values (41.8-46.4°C max temp) | 1. Run Case 900FF simulation; 2. Compare max temperature to reference 41.8-46.4°C; 3. Verify swing (max - min) is ~19.6% narrower than low-mass Case 600FF (65-75°C) |
| Thermal lag response time (2-6 hours) | FREE-02 | Requires empirical validation against ASHRAE reference thermal response curves | 1. Simulate Case 900FF with step change (e.g., solar pulse); 2. Measure time to reach 90% steady-state; 3. Verify within 2-6 hours per ASHRAE reference |
| Mass-air coupling correctness | FREE-02 | Requires validating h_tr_em, h_tr_ms conductances match ISO 13790 formulas | 1. Review implementation in `src/sim/engine.rs`; 2. Verify h_tr_em = 1 / ((1/h_tr_op) - (1/(h_ms*A_m))); 3. Validate no negative or unrealistic conductance values |
| Case 900 annual energy validation | FREE-02 | ASHRAE 140 reference values are in standard document | 1. Run Case 900 simulation; 2. Compare annual heating/cooling energy to ASHRAE reference ranges; 3. Verify within ±15% annual tolerance |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
