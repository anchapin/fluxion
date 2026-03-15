---
phase: 22
slug: validation-gap-resolution
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 22 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust built-in) |
| **Config file** | Cargo.toml (test profiles) |
| **Quick run command** | `cargo test --test ashrae_140_case_900` |
| **Full suite command** | `cargo test --test ashrae_140_comprehensive_regression` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --test ashrae_140_case_900`
- **After every plan wave:** Run `cargo test --test ashrae_140_comprehensive_regression`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 22-01-01 | 01 | 1 | VAL-07 | integration | `cargo test test_900_series_regression` | ❌ W0 | ⬜ pending |
| 22-02-01 | 02 | 1 | VAL-08, VAL-06 | unit | `cargo test test_thermal_mass_energy_accounting` | ❌ W0 | ⬜ pending |
| 22-02-02 | 02 | 1 | VAL-08, VAL-06 | unit | `cargo test test_case_600_energy_accounting` | ❌ W0 | ⬜ pending |
| 22-03-01 | 03 | 1 | VAL-09 | integration | `cargo test ab_testing -- --nocapture` | ❌ W0 | ⬜ pending |
| 22-04-01 | 04 | 2 | VAL-01, VAL-07 | integration | `cargo test test_case_960_comprehensive_energy_validation` | ✅ existing | ⬜ pending |
| 22-05-01 | 05 | 2 | VAL-02, VAL-03 | integration | `cargo test ab_testing_8r3c -- --nocapture` | ❌ W0 | ⬜ pending |
| 22-05-02 | 05 | 2 | VAL-04 | benchmark | `cargo test --bench batch_oracle -- --bench` | ✅ existing | ⬜ pending |
| 22-05-03 | 05 | 2 | VAL-05 | integration | `cargo test ab_testing_8r3c -- --nocapture` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `src/validation/thermal_mass_energy_accounting.rs` — thermal mass energy accounting validation functions (VAL-08, VAL-06)
- [ ] `tests/validation/thermal_mass_energy_accounting.rs` — energy accounting unit tests for 900 + 600 series (VAL-08, VAL-06)
- [ ] `src/validation/ab_testing.rs` — A/B testing framework with ThermalNetworkVariant enum (VAL-09)
- [ ] `tests/validation/ab_testing.rs` — A/B test runner and comparison reports (VAL-09)
- [ ] `tests/ashrae_140_case_900.rs` — extend with 900-series sequential regression test (VAL-07)
- [ ] `src/sim/engine_8r3c.rs` — 8R3C thermal network implementation stub (VAL-02, VAL-03)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| 8R3C reference implementation research | VAL-02 | Requires external documentation/code review | Analyze ASHRAE 140 reference programs (EnergyPlus, TRNSYS, ESP-r) to determine thermal network structure before implementing 8R3C |
| 8R3C performance evaluation | VAL-04 | Requires benchmarking with population vectors | Run BatchOracle::evaluate_population() with 10,000 configs and measure throughput via `cargo test --bench batch_oracle` |
| A/B testing variant comparison | VAL-03, VAL-05 | Requires manual analysis of statistical metrics | Review NMBE, CV(RMSE), pass rates from ab_testing reports to determine if 8R3C provides measurable improvement |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
