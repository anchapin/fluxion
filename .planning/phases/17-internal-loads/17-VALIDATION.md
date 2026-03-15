---
phase: 17
slug: internal-loads
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-13
---

# Phase 17 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust) |
| **Config file** | Cargo.toml (existing) |
| **Quick run command** | `cargo test --lib internal_loads -- --nocapture` |
| **Full suite command** | `cargo test --lib` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --lib internal_loads`
- **After every plan wave:** Run `cargo test --lib`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 17-01-01 | 01 | 1 | LOADS-01 | unit | `cargo test weekly_schedule --lib` | ✅ W0 | ⬜ pending |
| 17-01-02 | 01 | 1 | LOADS-01 | unit | `cargo test day_type_override --lib` | ✅ W0 | ⬜ pending |
| 17-01-03 | 01 | 1 | LOADS-01 | integration | `cargo test schedule_factory_methods --lib` | ✅ W0 | ⬜ pending |
| 17-02-01 | 02 | 1 | LOADS-02 | unit | `cargo test equipment_trait --lib` | ✅ W0 | ⬜ pending |
| 17-02-02 | 02 | 1 | LOADS-02 | unit | `cargo test equipment_types --lib` | ✅ W0 | ⬜ pending |
| 17-02-03 | 02 | 1 | LOADS-02 | integration | `cargo test equipment_convective_radiative --lib` | ✅ W0 | ⬜ pending |
| 17-02-04 | 02 | 1 | LOADS-02 | unit | `cargo test mass_coupled_radiative --lib` | ✅ W0 | ⬜ pending |
| 17-03-01 | 03 | 2 | LOADS-03 | unit | `cargo test thermal_model_internal_loads --lib` | ✅ W0 | ⬜ pending |
| 17-03-02 | 03 | 2 | LOADS-03 | integration | `cargo test schedule_indexing --lib` | ✅ W0 | ⬜ pending |
| 17-03-03 | 03 | 2 | LOADS-03 | integration | `cargo test energy_balance_with_loads --lib` | ✅ W0 | ⬜ pending |
| 17-04-01 | 04 | 2 | LOADS-04 | unit | `cargo test building_profile_loading --lib` | ✅ W0 | ⬜ pending |
| 17-04-02 | 04 | 2 | LOADS-04 | integration | `cargo test building_type_defaults --lib` | ✅ W0 | ⬜ pending |
| 17-04-03 | 04 | 2 | LOADS-04 | integration | `cargo test ashrae_internal_loads --lib` | ✅ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `src/sim/schedule.rs` — stubs for LOADS-01 weekly schedule tests
- [ ] `src/sim/equipment.rs` — stubs for LOADS-02 Equipment trait tests
- [ ] `src/sim/profiles.rs` — stubs for LOADS-04 profile loading tests
- [ ] `tests/internal_loads.rs` — integration tests for ThermalModel with internal loads
- [ ] Existing cargo test infrastructure covers all phase requirements

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Weekly schedule visual inspection | LOADS-01 | Validate schedule values match expected patterns (8-18 weekdays) | Run `cargo test schedule_factory_methods --lib -- --nocapture` and verify output shows correct hour values |
| Equipment thermal behavior | LOADS-02 | Verify mass coupling factor affects heat distribution correctly | Run test with different mass_coupling_factor values and compare Ti/Tm changes |
| Profile file format validation | LOADS-04 | Ensure JSON/YAML schema matches expected structure | Manually inspect `data/building_profiles.json` after creation |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
