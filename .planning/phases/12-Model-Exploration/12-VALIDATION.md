---
phase: 12
slug: model-exploration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-13
---

# Phase 12 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (native Rust testing) |
| **Config file** | None — tests directly use ASHRAE140Case enum |
| **Quick run command** | `cargo test test_6r2c_model --test` |
| **Full suite command** | `cargo test --all -- --nocapture` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test test_6r2c_model --test`
- **After every plan wave:** Run `cargo test --all -- --nocapture`
- **Before `/gsd:verify-work`:** Full ASHRAE 140 suite for both 5R1C and 6R2C models before decision
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 12-01-01 | 01 | 1 | MODEL6R2C-01 | unit | `cargo test test_configure_6r2c_model --test` | ✅ tests/test_6r2c_model.rs | ⬜ pending |
| 12-01-02 | 01 | 1 | MODEL6R2C-02 | unit | `cargo test test_6r2c_model_single_timestep --test` | ✅ tests/test_6r2c_model.rs (currently failing) | ⬜ pending |
| 12-01-03 | 01 | 1 | MODEL6R2C-03 | integration | `cargo test ashrae_140_case_900 --test` | ✅ tests/ashrae_140_case_900.rs | ⬜ pending |
| 12-01-04 | 01 | 1 | MODEL6R2C-04 | manual | N/A (documentation review) | ✅ docs/6R2C_IMPLEMENTATION.md | ⬜ pending |
| 12-01-05 | 01 | 1 | MODEL6R2C-05 | manual | N/A (decision document) | ❌ docs/6R2C_DECISION.md (to be created) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Fix failing test_6r2c_model_single_timestep (temperatures not updating)
- [ ] Add 6R2C benchmarks (criterion suite comparing 5R1C vs 6R2C throughput)
- [ ] Create 6R2C-specific ASHRAE 140 validation tests (900 series with 6R2C enabled)
- [ ] Add parameter sweep tests for envelope_mass_fraction and h_tr_me values
- [ ] Document decision framework for MODEL6R2C-05 adoption criteria

*Existing infrastructure covers most phase requirements, but 6R2C-specific validation needs Wave 0 completion.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| 6R2C findings documentation | MODEL6R2C-04 | Qualitative assessment of accuracy improvements, performance trade-offs, and migration path requires human judgment | Review docs/6R2C_IMPLEMENTATION.md after validation; verify accuracy comparison tables, benchmark results, and decision recommendations are complete |
| Adoption decision (6R2C as default vs keep 5R1C) | MODEL6R2C-05 | Binary decision based on quantitative criteria (error reduction, no regression, throughput) with business judgment | Create docs/6R2C_DECISION.md; document: (1) accuracy improvement on 900 series, (2) regression check on 600 series, (3) throughput comparison, (4) final decision with rationale |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (fix test_6r2c_model_single_timestep, add benchmarks, add 6R2C ASHRAE tests, add parameter sweep, document decision framework)
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
