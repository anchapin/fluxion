---
phase: 6
slug: performance-optimization
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust/cargo test |
| **Config file** | none — tests use existing validation framework |
| **Quick run command** | `cargo test --release --test validation_smoke` |
| **Full suite command** | `cargo test --release -- --test-threads=1 test_validation` |
| **Estimated runtime** | ~300 seconds (5 min target after optimization) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --release --test test_parallel_validation` (target: <30s)
- **After every plan wave:** Run `cargo test --release --test test_guardrails` (target: <60s)
- **Before `/gsd:verify-work`:** Full validation suite `fluxion validate --all` (target: <300s after optimization)
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 6-01-01 | 01 | 0 | BATCH-01 | integration | `cargo test --test test_parallel_validation` | ✅ W0 | ⬜ pending |
| 6-01-02 | 01 | 0 | BATCH-02 | integration | `cargo test --test test_result_aggregation` | ✅ W0 | ⬜ pending |
| 6-01-03 | 01 | 1 | GPU-01 | integration | `cargo test --test test_gpu_autodetect` | ❌ W1 | ⬜ pending |
| 6-02-01 | 02 | 0 | SURR-01 | unit | `cargo test --test test_session_pool` | ✅ W0 | ⬜ pending |
| 6-02-02 | 02 | 0 | SURR-02 | integration | `cargo test --test test_batched_inference` | ✅ W0 | ⬜ pending |
| 6-02-03 | 02 | 1 | SURR-03 | integration | `cargo test --test test_modular_surrogates` | ❌ W1 | ⬜ pending |
| 6-03-01 | 03 | 2 | REG-01 | unit | `cargo test --test test_mae_tracking` | ❌ W2 | ⬜ pending |
| 6-03-02 | 03 | 2 | REG-02 | unit | `cargo test --test test_max_deviation_tracking` | ❌ W2 | ⬜ pending |
| 6-03-03 | 03 | 2 | REG-03 | integration | `cargo test --test test_performance_history` | ❌ W2 | ⬜ pending |
| 6-03-04 | 03 | 2 | REG-04 | integration | `cargo test --test test_guardrail_exit_codes` | ❌ W2 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_parallel_validation.rs` — rayon parallelization of ASHRAE cases
- [ ] `tests/test_result_aggregation.rs` — aggregated benchmark report collection
- [ ] `tests/test_session_pool.rs` — SurrogateManager SessionPool concurrency test
- [ ] `tests/test_batched_inference.rs` — predict_loads_batched() correctness validation
- [ ] `tests/validation/benchmark_report.rs` — performance metrics recording infrastructure

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GPU acceleration achieves 10-100x speedup | GPU-02, GPU-03 | Hardware-dependent; requires specific GPU hardware and driver setup | 1. Run `fluxion validate --all --gpu` on RTX 4090 or similar<br>2. Record throughput: configs/sec<br>3. Compare to CPU baseline (expected 10-100x improvement) |
| Baseline performance established | REG-01, REG-02, REG-03 | Requires 3 Clean-run measurements for reference | 1. On reference hardware (8-core CPU + GPU), run `fluxion validate --all` 3 times<br>2. Record mean values for MAE, MaxDev, PassRate, Time<br>3. Update `docs/performance_baseline.json` with these values |
| Multi-GPU scaling | GPU-03 | Requires multi-GPU system not universally available | 1. Configure `validation.gpu.devices = [0,1]`<br>2. Run large population (10,000 configs)<br>3. Monitor per-GPU utilization via `nvidia-smi`<br>4. Verify both GPUs are used (near-equal utilization) |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter after Wave 0 completion

**Approval:** pending
