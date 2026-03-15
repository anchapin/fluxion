---
phase: 15
slug: hvac-equipment-modeling
status: ready
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-13
---

# Phase 15 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust built-in) |
| **Config file** | None (uses Cargo.toml dev-dependencies) |
| **Quick run command** | `cargo test --package fluxion --lib sim::hvac -- --nocapture` |
| **Full suite command** | `cargo test --package fluxion --lib` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --package fluxion --lib sim::hvac -- --nocapture`
- **After every plan wave:** Run `cargo test --package fluxion --lib`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 15-01-01 | 01 | 1 | HVAC-01 | unit | `cargo test test_vav_variable_capacity -- --nocapture` | ✅ W0 | ⬜ pending |
| 15-01-02 | 01 | 1 | HVAC-02 | unit | `cargo test test_cav_variable_capacity -- --nocapture` | ✅ W0 | ⬜ pending |
| 15-01-03 | 01 | 1 | HVAC-03 | unit | `cargo test test_heatpump_efficiency_curves -- --nocapture` | ✅ W0 | ⬜ pending |
| 15-02-01 | 02 | 1 | HVAC-04 | unit | `cargo test test_chiller_efficiency_curves -- --nocapture` | ✅ W0 | ⬜ pending |
| 15-02-02 | 02 | 1 | HVAC-05 | unit | `cargo test test_boiler_efficiency_curves -- --nocapture` | ✅ W0 | ⬜ pending |
| 15-03-01 | 03 | 2 | HVAC-07 | unit | `cargo test test_polynomial_efficiency_curves -- --nocapture` | ✅ W0 | ⬜ pending |
| 15-03-02 | 03 | 2 | HVAC-08 | unit | `cargo test test_cycling_losses -- --nocapture` | ✅ W0 | ⬜ pending |
| 15-04-01 | 04 | 3 | HVAC-09 | unit | `cargo test test_predictive_control -- --nocapture` | ✅ W0 | ⬜ pending |
| 15-04-02 | 04 | 3 | HVAC-06 | integration | `cargo test test_economizer_mode -- --nocapture` | ✅ W0 | ⬜ pending |
| 15-04-03 | 04 | 3 | HVAC-01-09 | integration | `cargo test test_ashrae_140_cases_800_810 -- --nocapture` | ✅ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `src/sim/hvac/equipment.rs` — VariableCapacityEquipment trait, Chiller, Boiler structs
- [x] `src/sim/hvac/efficiency_curves.rs` — Polynomial curve evaluation, coefficient structs
- [x] `src/sim/hvac/control.rs` — Predictive control logic with thermal inertia
- [x] `src/sim/hvac/cycling.rs` — CyclingTracker, startup penalties, minimum runtime
- [x] `src/sim/hvac/tests/equipment_tests.rs` — Unit tests for VariableCapacityEquipment trait
- [x] `src/sim/hvac/tests/efficiency_curve_tests.rs` — Unit tests for polynomial curves
- [x] `src/sim/hvac/tests/control_tests.rs` — Unit tests for predictive control
- [x] `src/sim/hvac/tests/cycling_tests.rs` — Unit tests for cycling losses
- [x] `tests/ashrae_140_cases_800_810.rs` — Integration tests for ASHRAE 140 Cases 800-810
- [x] Framework install: None needed (cargo test is built-in)
- [x] AHRI coefficient data: Create `src/sim/hvac/ahri_coefficients.json` with default values

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| AHRI coefficient accuracy | HVAC-03, HVAC-04, HVAC-05 | Requires external reference data validation | Compare default coefficient curves against AHRI Standard 550/590 (chillers) and 210/240 (heat pumps) when available; adjust defaults to match reference |
| Thermal inertia stability | HVAC-09 | Requires analysis of control signal behavior | Run ASHRAE 140 Cases 800-810; inspect control signal (modulation factor) for oscillation; adjust thermal_inertia_gain and temp_rate_gain if control is unstable |
| Minimum runtime effectiveness | HVAC-08 | Requires analysis of annual cycling patterns | Run 1-year simulation; verify startup count << annual runtime hours; if startup count > 100, increase minimum_runtime_timesteps |

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
