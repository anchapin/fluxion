---
phase: 3
slug: solar-radiation
status: draft
nyquist_compliant: true
wave_0_complete: true
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
| 03-01-01 | 01 | SOLAR-01 | unit | `cargo test solar_integration --lib` | src/sim/engine.rs | ✅ exists | ⬜ pending |
| 03-01-02 | 01 | SOLAR-02 | unit | `cargo test --test ashrae_140_free_floating -- --nocapture` | tests/ashrae_140_free_floating.rs | ✅ exists | ⬜ pending |
| 03-01-03 | 01 | SOLAR-04 | unit | `cargo test --test solar_calculation_validation test_beam_diffuse_decomposition -- --nocapture` | tests/solar_calculation_validation.rs | ✅ exists | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Wave 0 completed by Plan 03-00 which created:
- ✅ `tests/solar_calculation_validation.rs` — unit tests for Perez sky model beam/diffuse decomposition
- ✅ `tests/solar_integration.rs` — unit tests for solar gains integration into thermal network
- ✅ `tests/ashrae_140_free_floating.rs` — validation tests for free-floating temperature with solar effects

*Wave 0 complete: all required test files created by Plan 03-00*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Solar gain calculation accuracy | SOLAR-01 | Requires comparison to ASHRAE 140 reference values (hourly DNI/DHI for all orientations) | 1. Run Case 600, 610, 620, 630, 640, 650 simulations with solar gains enabled; 2. Compare hourly DNI/DHI values to ASHRAE 140 reference (e.g., from ASHRAE 140 standard document or online calculators); 3. Verify within ±5% tolerance for all orientations |
| Shading effect correctness | SOLAR-04 | Requires simulation-based validation of shading on specific cases (610, 630, 910, 930) | 1. Run Case 610, 630, 910, 930 simulations with shading enabled; 2. Compare cooling loads to ASHRAE 140 reference ranges; 3. Verify that shading reduces cooling loads correctly (e.g., 610 should have lower cooling than 600); 4. Validate against ASHRAE 140 shading reference values |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Wave 0 covers all test file references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
