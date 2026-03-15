---
phase: 11
slug: api-robustness
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-13
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (existing Rust test infrastructure) |
| **Config file** | .claude/skills/ or .agents/skills/ (if exists) |
| **Quick run command** | `cargo test --lib` |
| **Full suite command** | `cargo test` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --lib`
- **After every plan wave:** Run `cargo test`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| {N}-01-01 | 01 | 1 | API-01 | unit | `cargo test --lib` | ⬜ pending |
| {N}-01-02 | 01 | 1 | API-02 | unit | `cargo test --lib` | ⬜ pending |
| {N}-02-01 | 02 | 1 | API-03 | unit | `cargo test --lib` | ⬜ pending |
| {N}-02-02 | 02 | 1 | API-04 | unit | `cargo test --lib` | ⬜ pending |
| {N}-03-01 | 03 | 1 | API-05 | integration | `cargo test --lib` | ⬜ pending |
| {N}-04-01 | 04 | 1 | ROBUST-01 | unit | `cargo test --lib` | ⬜ pending |
| {N}-04-02 | 04 | 1 | ROBUST-02 | unit | `cargo test --lib` | ⬜ pending |
| {N}-05-01 | 05 | 1 | ROBUST-03 | unit | `cargo test --lib` | ⬜ pending |
| {N}-05-02 | 05 | 1 | ROBUST-04 | integration | `cargo test --lib` | ⬜ pending |
| {N}-05-03 | 05 | 1 | ROBUST-05 | unit | `cargo test --lib` | ⬜ pending |
| {N}-06-01 | 06 | 1 | BUG-03 | unit | `cargo test --lib` | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_api_exceptions.rs` — stubs for API-01, API-02 custom exceptions
- [ ] `tests/test_parameter_bounds.rs` — stubs for API-02, API-03 parameter discovery
- [ ] `tests/test_validation_api.rs` — stubs for API-03, API-04 validation
- [ ] `tests/test_onnx_fallback.rs` — stubs for ROBUST-02 ONNX fallback
- [ ] `tests/test_nan_inf_detection.rs` — stubs for ROBUST-01 NaN/Inf detection
- [ ] `tests/test_logging_control.rs` — stubs for ROBUST-03 logging verbosity
- [ ] `tests/test_extreme_data.rs` — stubs for ROBUST-04 extreme weather data
- [ ] `tests/test_error_messages.rs` — stubs for BUG-03 correct error messages

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Python API usability | API-01, API-02, API-03 | Requires Python interpreter testing | `python -c "import fluxion; oracle = fluxion.BatchOracle(); ..."` |
| ONNX graceful degradation | ROBUST-02 | Requires ONNX Runtime failure simulation | Simulate ONNX failure, verify fallback to analytical |
| Error message clarity | BUG-03 | Subjective user experience assessment | Manually test with invalid inputs, review message clarity |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** {pending / approved YYYY-MM-DD}

---
*Phase: 11-API-Robustness*
*Validation strategy created: 2026-03-13*
