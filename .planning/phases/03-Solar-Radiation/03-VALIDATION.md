---
phase: 3
slug: solar-radiation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-09
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust) |
| **Config file** | Cargo.toml |
| **Quick run command** | `cargo test -- solar` |
| **Full suite command** | `cargo test --all` |
| **Estimated runtime** | ~90 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -- solar`
- **After every plan wave:** Run `cargo test --all`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 180 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | SOLAR-01 | unit | `cargo test -- solar` | tests/test_solar_radiation.rs | ⬜ pending |
| 03-01-02 | 01 | SOLAR-02 | unit | `cargo test -- solar` | tests/test_solar_incidence.rs | ⬜ pending |
| 03-01-03 | 01 | SOLAR-03 | unit | `cargo test -- solar` | tests/test_window_properties.rs | ⬜ pending |
| 03-01-04 | 01 | SOLAR-04 | unit | `cargo test -- solar` | tests/test_beam_diffuse.rs | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_solar_radiation.rs` — unit tests for NOAA solar position, Perez sky model
- [ ] `tests/test_solar_incidence.rs` — unit tests for solar incidence angle calculations
- [ ] `tests/test_window_properties.rs` — unit tests for SHGC, normal transmittance values
- [ ] `tests/test_beam_diffuse.rs` — unit tests for beam/diffuse decomposition

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Solar gain calculation accuracy | SOLAR-01 | Requires comparison to ASHRAE 140 reference values (hourly DNI/DHI for all orientations) | 1. Run Case 600, 610, 620, 630, 640, 650 simulations with solar gains enabled; 2. Compare hourly DNI/DHI values to ASHRAE 140 reference (e.g., from ASHRAE 140 standard document or online calculators); 3. Verify within ±5% tolerance for all orientations |
| Shading effect correctness | SOLAR-04 | Requires simulation-based validation of shading on specific cases (610, 630, 910, 930) | 1. Run Case 610, 630, 910, 930 simulations with shading enabled; 2. Compare cooling loads to ASHRAE 140 reference ranges; 3. Verify that shading reduces cooling loads correctly (e.g., 610 should have lower cooling than 600); 4. Validate against ASHRAE 140 shading reference values |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Wave 0 covers all test file references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
