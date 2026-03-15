---
phase: 21
slug: integration-testing-framework
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 21 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust) + pytest 7.x (Python) |
| **Config file** | .github/workflows/ci.yml (existing) |
| **Quick run command** | `cargo test --test integration --lib --quiet` |
| **Full suite command** | `cargo test --test integration && pytest -q && cargo bench --bench integration_bench` |
| **Estimated runtime** | ~60 seconds (quick) / ~5 minutes (full) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --test integration --lib --quiet`
- **After every plan wave:** Run `cargo test --test integration && pytest -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 21-01-01 | 01 | 1 | INTEG-01, INTEG-02 | integration | `cargo test --test integration test_e2e_scenarios` | ❌ W0 | ⬜ pending |
| 21-01-02 | 01 | 1 | INTEG-03 | integration | `cargo test --test integration test_wiring_validation` | ❌ W0 | ⬜ pending |
| 21-02-01 | 02 | 1 | INTEG-04 | integration | `pytest tests/test_pyo3_bindings.py -q` | ❌ W0 | ⬜ pending |
| 21-03-01 | 03 | 2 | INTEG-05 | integration | `cargo test --test integration test_ashrae_140_regression` | ❌ W0 | ⬜ pending |
| 21-04-01 | 04 | 2 | INTEG-06 | unit | `cargo test test_data_manager --lib` | ❌ W0 | ⬜ pending |
| 21-05-01 | 05 | 2 | INTEG-07 | integration | `cargo test --test integration && cargo bench --bench integration_bench` | ❌ W0 | ⬜ pending |
| 21-05-02 | 05 | 2 | INTEG-08 | integration | `cargo test --test integration test_wiring_tracer` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `src/testing/integration/fixtures.rs` — stubs for BuildingScenario builder
- [ ] `src/testing/integration/wiring.rs` — stubs for WiringTracer
- [ ] `tests/integration/test_e2e.rs` — stubs for E2E scenarios
- [ ] `tests/integration/test_wiring.rs` — stubs for wiring validation
- [ ] `tests/conftest.py` — pytest fixtures for Python tests (extend existing)
- [ ] `tests/test_pyo3_bindings.py` — stubs for NumPy array validation
- [ ] `tests/data/v0.5/` — test data directory for v0.5 reference results

*Existing infrastructure covers: pytest setup, ASHRAE 140 validation, CI/CD workflow, dev dependencies (tempfile, rstest, approx, proptest, mockito).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Nightly regression test creates GitHub issues on failure | INTEG-05 | Requires GitHub API integration | Verify `.github/workflows/nightly_regression.yml` triggers and creates issues |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
