---
phase: 20
slug: data-quality-finalization
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 20 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust built-in) |
| **Config file** | Cargo.toml |
| **Quick run command** | `cargo test -p fluxion --lib` |
| **Full suite command** | `cargo test --all` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p fluxion --lib`
- **After every plan wave:** Run `cargo test --all`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 20-01-01 | 01 | 1 | PHYS-02 | unit | `cargo test -p fluxion construction` | ❌ W0 | ⬜ pending |
| 20-01-02 | 01 | 1 | PHYS-03 | unit | `cargo test -p fluxion constants` | ❌ W0 | ⬜ pending |
| 20-02-01 | 02 | 1 | PHYS-06 | unit | `cargo test -p fluxion thermal_mass` | ❌ W0 | ⬜ pending |
| 20-02-02 | 02 | 1 | PHYS-07 | unit | `cargo test -p fluxion validation` | ❌ W0 | ⬜ pending |
| 20-03-01 | 03 | 2 | WEATHER-01 | integration | `cargo test -p fluxion epw` | ❌ W0 | ⬜ pending |
| 20-03-02 | 03 | 2 | WEATHER-03 | unit | `cargo test -p fluxion tmy3` | ❌ W0 | ⬜ pending |
| 20-03-03 | 03 | 2 | WEATHER-04 | integration | `cargo test -p fluxion weather` | ❌ W0 | ⬜ pending |
| 20-03-04 | 03 | 2 | WEATHER-05 | unit | `cargo test -p fluxion cache` | ❌ W0 | ⬜ pending |
| 20-04-01 | 04 | 2 | DATA-02 | unit | `cargo test -p fluxion config_validation` | ❌ W0 | ⬜ pending |
| 20-04-02 | 04 | 2 | DATA-03 | integration | `cargo test -p fluxion assembly` | ❌ W0 | ⬜ pending |
| 20-04-03 | 04 | 2 | DATA-04 | unit | `cargo test -p fluxion docs` | ❌ W0 | ⬜ pending |
| 20-04-04 | 04 | 2 | DATA-05 | integration | `cargo test --all` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `src/construction/mod.rs` — ISO 13790 Annex C material property library
- [ ] `src/constants/mod.rs` — Physical constants with source references
- [ ] `src/thermal_mass/mod.rs` — Auto-calculation and classification module
- [ ] `src/validation/mod.rs` — Configuration validation infrastructure
- [ ] `src/weather/tmy3.rs` — TMY3 download and caching
- [ ] `src/config/mod.rs` — Structured error types
- [ ] `tests/` — Integration tests for weather and assembly systems

*Existing infrastructure covers some phase requirements, but new modules are needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Physical constant source references | PHYS-03 | Documentation review required | Review docs/PHYSICAL_CONSTANTS.md for each constant |
| Assembly composition documentation | DATA-04 | Domain knowledge review | Verify ASHRAE 140 assemblies match standard |
| TMY3 data quality check | WEATHER-03 | Visual inspection of plots | Plot sample TMY3 files to check continuity |
| Configuration error messages | DATA-02 | UX review required | Trigger each validation error, review message clarity |

*Some behaviors have automated verification; others require manual review.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
