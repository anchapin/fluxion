# Issue #3325: BatchOracle `fast-math` Audit — Documented No-Op

<!-- 7-line summary for AI agents: lines 1-7 -->
<!-- 1: Profile-first audit per issue #3325 of the BatchOracle / src/ai/batch_*.rs code -->
<!-- 2: paths for measurable reduction/accumulation hotspots eligible for fast-math conversion. -->
<!-- 3: All inspected aggregate-metric call sites are either orchestration (no numerics) -->
<!-- 4: or feed the ASHRAE 140 / energy-conservation ledger, which the fp_algebraic helper -->
<!-- 5: module explicitly excludes (`src/physics/fp_algebraic.rs` "Do NOT use in energy- -->
<!-- 6: balance-critical paths"). Bench evidence: 8 bench IDs captured, no site eligible. -->
<!-- 7: Recommendation: close #3325 as a documented no-op; profile evidence is the resolution. -->

## Scope

Follow-up to the `fast-math` feature ([#3322](https://github.com/anchapin/fluxion/issues/3322), merged via PR #3344). The original draft assumed hot hand-rolled loss/divergence reduction loops in `src/ai/`; the upstream audit found the opposite, and #3325 makes this a **measure-first** rather than blanket-conversion task. Acceptance criteria explicitly allow a documented no-op when profiling shows no meaningful reduction loops.

## Method

1. **Static inspection** of every file referenced in the issue (`src/ai/batch_runner.rs`, `src/ai/shared_batch_service.rs`, `src/ai/batch_inference.rs`) plus the actual `BatchOracle` definition (`src/batch_oracle.rs`) and its two hot-loop orchestrators (`src/sim/orchestrator.rs`).
2. **Profile evidence** via `cargo bench --bench batch_oracle_bench` (the `batch_oracle` target in #3325 maps to this binary) and `cargo bench --bench surrogate_vs_physics` for the head-to-head Case 900/600/920 physics-vs-hybrid comparison that BatchOracle actually drives.
3. **Reduction-site classification** against the `src/physics/fp_algebraic.rs` "Do NOT use in energy-balance-critical paths" exclusion list, which enumerates the same paths AGENTS.md / RULES.md / SCORECARD.md protect as bit-identical IEEE.

## Static inspection findings

### `src/ai/batch_runner.rs` (745 lines) — orchestration

Defines `ParameterSpec` / `ParameterSample` / `ParameterManifest` / `SimulationOutput` / `BatchResults` / `BatchConfig` / `BatchRunner` plus the `sampling` and `io` submodules. No reduction loop:

- `validate_parameters` (`src/ai/batch_runner.rs:80`) — range checks, no accumulation.
- `run_single_sample` (`src/ai/batch_runner.rs:407`) — single-config physics, returns one `SimulationOutput`.
- `run` (`src/ai/batch_runner.rs:456`) — `par_iter` over samples; each worker delegates to `run_single_sample` and pushes a single output into the result vec. No MAE/RMSE/CV(RMSE) computation.
- `run_smoke_test` (`src/ai/batch_runner.rs:491`) — same shape, smaller N.
- `sampling::sample_from_distribution` (`src/ai/batch_runner.rs:526`) — single-sample RNG draw.
- `sampling::generate_samples` (`src/ai/batch_runner.rs:556`) — builds a `Vec<ParameterSample>` from RNG draws; one f64 per call, no accumulation.

**Orchestration-only.**

### `src/ai/shared_batch_service.rs` (792 lines) — scheduling + channels

Defines `DynamicBatchConfig` / `SchedulerConfig` / `BatchMetrics` / `BatchMetricsSnapshot` / `SharedBatchInferenceService`. No reduction loop:

- `SchedulerConfig::resolve_num_workers` (`src/ai/shared_batch_service.rs:104`) — single integer arithmetic.
- `BatchMetrics::ema_inference_ms` (`src/ai/shared_batch_service.rs:154`) — single multiply-add of EMA state (1-element EMA, not a population aggregate).
- `submit` / `submit_with_sender` (`src/ai/shared_batch_service.rs:286` and `:339`) — pushes a request onto a `crossbeam::channel`, returns the matching `Receiver`.
- `run_worker` (`src/ai/shared_batch_service.rs:374`) — drains requests, dispatches to `process_batch`, replies. No accumulation of results.
- `process_batch` (`src/ai/shared_batch_service.rs:477`) — splits a batch into `BatchProcessor::process_single` chunks; each result is a per-element `Vec<f64>`.
- `Inner::drop` (`src/ai/shared_batch_service.rs:543`) — sends `WorkerCommand::Shutdown`; no math.

**Orchestration-only.** The single EMA multiply-add at `:154` is a one-element running average and is also a telemetry metric (consumed by `metrics()` at `:354`); converting it to `algebraic_add` / `algebraic_mul` would perturb CI advisory metrics for no measurable speedup on a hot loop that does ~1 op per request.

### `src/ai/batch_inference.rs` (480 lines) — scheduling + benchmarking helpers

Defines `DynamicBatchConfig` / `BatchStats` / `BatchProcessor` / `BatchBenchmarkResult` / `benchmark_batch_inference`. Only the test stub at `src/ai/batch_inference.rs:370` contains a sum: `inputs.iter().map(|v| vec![v.iter().sum()]).collect()`, which is a single-call mock inside a test that drives the `BatchProcessor::optimize_batch_size` adapter (the mock only computes the input checksum that the real `inference_fn` would have processed).

**Orchestration-only.**

### `src/batch_oracle.rs` (BatchOracle struct + `evaluate_population_from_slice`) — physics dispatch, not aggregate metrics

`BatchOracle::evaluate_population` (`src/batch_oracle.rs:195`) returns one `EUI = energy / area` per candidate (the per-member result the issue mentions). The only reduction loop on the **BatchOracle** code path is the per-member energy accumulation inside `RayonChunksOrchestrator`:

- `run_cpu_surrogate` (`src/sim/orchestrator.rs:289`): `energy_kwh += model.step_physics(t, outdoor_temp, 3600.0)` over 8 760 timesteps.
- `run_cpu_surrogate_batched` (`src/sim/orchestrator.rs:362` + `:435`): same `energy[t]` accumulation.
- `run_cpu_analytical` (`src/sim/orchestrator.rs:660`): `total_energy += model.solve_single_step(...)` over 8 760 timesteps.

These *are* hot reduction loops (~8 760 iterations × N configs), **but** every element being summed is an IEEE result from `model.step_physics` / `model.solve_single_step`, and the final sum directly feeds `EUI = total_energy / total_area` — the zone-balance / energy-conservation quantity that:

- `src/physics/fp_algebraic.rs` "Do NOT use in energy-balance-critical paths" §calls out as **never** to be routed through `algebraic_*`.
- AGENTS.md §"Physics and Validation Guardrails" forbids tuning for zone-balance invariants.
- `tests/reference_data/zone_balance/strict_energy_gate_baseline.json` + `SCORECARD.md` baseline are generated and judged under strict IEEE semantics (RULES.md).
- The surrogate ASHRAE 140 MAE Gate (#2924) and per-timestep surrogate drift gate (`surrogate_drift_gate.yml`, #1784) would be perturbed by last-ulp drift in the EUI numerator.

Converting `energy_kwh += ...` to `energy_kwh = energy_kwh.algebraic_add(...)` would be a direct violation of the fp_algebraic helper's own exclusion list and AGENTS.md's BatchOracle hard rule ("Per-member physics solves inside `BatchOracle` stay IEEE"). **Not eligible.**

### Aggregate-metric sites (`src/validation/high_mass/metrics.rs`, `src/validation/reference_data.rs`, `src/thermal/mass/validator.rs`, `src/performance/parallel/validation.rs`)

The only `sum` / `sum_squared` / `fold` patterns matching MAE/RMSE/CV(RMSE) over populations live outside `src/ai/`:

| File | Function | Input length | Output feeds |
|------|----------|--------------|--------------|
| `src/validation/high_mass/metrics.rs:58` | `calculate_nmbe` | 12 monthly totals | ASHRAE 140 validation report |
| `src/validation/high_mass/metrics.rs:90` | `calculate_cv_rmse` | 12 monthly totals | ASHRAE 140 validation report |
| `src/validation/high_mass/metrics.rs:130` | `calculate_mae` | 12 monthly totals | ASHRAE 140 validation report |
| `src/validation/high_mass/metrics.rs:155` | `calculate_max_error` | 12 monthly totals | ASHRAE 140 validation report |
| `src/validation/reference_data.rs:315` | `calculate_rmse` | 12 monthly totals | ASHRAE 140 validation report |
| `src/thermal/mass/validator.rs:111` + `:132` | `calculate_nmbe` / `calculate_cv_rmse` | length 8 760 hourly | Thermal mass validation |
| `src/performance/parallel/validation.rs:94` | `validate_single_case` (NMBE/CV(RMSE)/max_deviation) | length 8 760 hourly | Parallel validation pipeline (used by `benches/parallel_validation`) |

Each of these outputs **feeds either** the ASHRAE 140 baseline ledger, the surrogate MAE/drift gate, or an EnergyPlus-validation comparison. The `fp_algebraic` module's "Do NOT use in energy-balance-critical paths" exclusion explicitly enumerates `crate::validation::*` as out-of-scope for fast-math:

> "It must never be applied to paths whose floating-point rounding feeds an energy conservation ledger or a validation baseline."

**Not eligible.** (They are also one-shot per ASHRAE 140 case — not a hot loop.)

### Bench coverage

`benches/batch_oracle_bench.rs` and `benches/surrogate_vs_physics_bench.rs` are the only bench files that exercise `BatchOracle::evaluate_population` end-to-end. Neither computes MAE/RMSE/CV(RMSE):

- `benches/batch_oracle_bench.rs` measures `oracle.evaluate_population(pop, false)` and `... true)` only — throughput per population size, no reductions.
- `benches/surrogate_vs_physics_bench.rs` measures `solve_timesteps` per case (600/900/920) — throughput and ms/timestep, no reductions.

The other `BatchOracle`-touching benches (`benchmark_8760_timesteps`, `multi_zone_throughput`, `rayon_chunks_bench`, `hybrid_hotloop_tracing_bench`, `shared_batch_service_bench`, `performance_regression`) are all throughput-only.

## Profile evidence

### `cargo bench --bench batch_oracle_bench -- --sample-size 10 --measurement-time 5`

(Local default-features build, fluxion v1.0.0, c7b3bbf, rustc 1.98.0; bench IDs as reported by Criterion.)

| Bench ID | Criterion time | Notes |
|----------|---------------:|-------|
| `batch_oracle_analytical/100` | (warm-up only) | Under 5 s budget — fast per-config |
| `batch_oracle_analytical/200` | 5.6067 s (range 4.84–6.48) | Population of 200 |
| `batch_oracle_analytical/500` | 6.0267 s (range 5.61–6.47) | Population of 500 |
| `batch_oracle_analytical/1000` | 10.315 s (range 9.30–11.59) | Population of 1000 |
| `batch_oracle_surrogates/100` | (warm-up only) | Under 5 s budget — fast per-config |
| `batch_oracle_surrogates/200` | 3.8819 s (range 3.59–4.17) | Population of 200 |
| `batch_oracle_surrogates/500` | 6.2857 s (range 6.05–6.59) | Population of 500 |
| `batch_oracle_surrogates/1000` | 4.7335 s (range 4.51–5.03) | Population of 1000 |

All time is dominated by per-config physics inside `BatchOracle::evaluate_population` (the `RayonChunksOrchestrator::run_cpu_analytical` and `run_cpu_surrogate` 8 760-step inner loops). No aggregate-metric reduction happens inside the bench closure.

### `cargo bench --bench surrogate_vs_physics -- --sample-size 10 --measurement-time 5 weekly_168`

| Bench ID | Criterion time | Throughput |
|----------|---------------:|-----------|
| `head_to_head/case900/physics_only_short/weekly_168` | 193.54 µs (180.3–207.6) | 868 Kelem/s |
| `head_to_head/case900/hybrid_default_short/weekly_168` | 163.88 µs (160.9–167.4) | 1.025 Melem/s |
| `head_to_head/case600/physics_only_short/weekly_168` | 194.49 µs (184.8–206.3) | 864 Kelem/s |
| `head_to_head/case600/hybrid_default_short/weekly_168` | 211.17 µs (197.7–225.9) | 796 Kelem/s |
| `head_to_head/case920/physics_only_short/weekly_168` | 171.34 µs (166.2–177.1) | 980 Kelem/s |
| `head_to_head/case920/hybrid_default_short/weekly_168` | 331.38 µs (300.9–363.5) | 507 Kelem/s |

(Annual `head_to_head/case900/*/annual_8760` and `bench_ms_per_timestep_8760` benchmarks were skipped in this run — they take >2 min each and are not relevant to fast-math site identification.)

Surrogate ONNX benchmarks in `surrogate_vs_physics_bench.rs` are skipped because `ort` is not enabled in this build (`Surrogate benchmarks will be skipped.` from the bench). This is the expected default-features build state and is unrelated to the audit.

## Hot-spot assessment: **none found**

The audit located exactly three classes of f64 reduction loops on the BatchOracle / `src/ai/` / aggregate-metric path:

1. **Per-member energy accumulation** in `RayonChunksOrchestrator::run_cpu_*` (`src/sim/orchestrator.rs:289`, `:435`, `:660`). Each is a 8 760-step `+=` that feeds EUI = energy / area. Excluded by `fp_algebraic` ("Do NOT use in energy-balance-critical paths") and AGENTS.md BatchOracle hard rule.
2. **One-element EMA telemetry** in `BatchMetrics::ema_inference_ms` (`src/ai/shared_batch_service.rs:154`). Single multiply-add, not a hot loop, feeds CI advisory metrics.
3. **One-shot MAE/RMSE/CV(RMSE) reductions** in `src/validation/high_mass/metrics.rs`, `src/validation/reference_data.rs`, `src/thermal/mass/validator.rs`, `src/performance/parallel/validation.rs`. Called once per ASHRAE 140 case on 12-element or 8 760-element slices; output feeds validation baselines. Excluded by `fp_algebraic` (explicitly enumerates `crate::validation::*`).

**Class (1)** is the only hot reduction loop on the BatchOracle path, but it is a per-member physics-solve sum (the `step_physics` / `solve_single_step` results are themselves IEEE and feed the energy ledger). Converting it would violate the AGENTS.md rule and the fp_algebraic helper's own exclusion list.

**Classes (2) and (3)** are off-limits or not hot.

## Recommendation: close #3325 as a documented no-op

The issue text explicitly allows this outcome:

> "If profiling shows no meaningful reduction loops, close this issue with the profile evidence attached — a documented no-op is a valid outcome."

This investigation IS the resolution. No code changes to `src/ai/`, `src/batch_oracle.rs`, or `src/sim/orchestrator.rs` are appropriate. `fast-math` remains available for the in-flight kernel-conversion issues (#3324 solar/irradiance reductions and any future, genuinely-aggregate batch-metric call site that is *not* in the energy-conservation path).

## Constraints upheld (regardless of outcome)

- Per-member physics solves inside `BatchOracle` stay IEEE — no `algebraic_*` introduced anywhere in `src/sim/orchestrator.rs` or `src/batch_oracle.rs`.
- No nested inner-loop Rayon parallelism added — `BatchOracle` continues to parallelize populations only via `par_chunks` in `RayonChunksOrchestrator`.
- ONNX surrogate inference untouched — the `ort`-gated `predict_loads_onnx` / `predict_loads_batched` paths were not opened; fail-closed `.sha256` verification unaffected.
- Default-feature behavior bit-identical — zero code changes, zero `cfg(feature = "fast-math")` introduced.

## Acceptance

- [x] Profile artifacts linked (8 bench IDs above; criterion estimates saved under `target/criterion/batch_oracle_*` and `target/criterion/head_to_head_*`).
- [x] Surrogate outputs preserved on default features (no change). #2924 and #1784 not touched.
- [x] Default-feature behavior bit-identical (zero source diff in default build).
- [x] `cargo fmt --check` clean (exit 0).
- [x] `cargo clippy --lib -- -D warnings` clean (exit 0 on root crate lib).
- [x] `cargo test --workspace` — see "Pre-existing failures (out of scope)" below.

## Pre-existing failures (out of scope for #3325)

`cargo test --workspace` in **debug** mode is red on develop tip `c7b3bbf` due to two pre-existing issues that are **not** introduced by this audit and are **not** on the CI required-checks list (`release_gates.yaml` → `ci.required_checks`):

- `tests/batch_oracle_hotloop_equivalence.rs::analytical_path_eui_is_bit_identical_to_baseline` — pre-existing; the golden `[2.099, 2.959, 0.0]` (3-decimal place, see #3232) does not bit-equal the actual `[2.099433588906497, 2.9595394892940883, 0.0]` produced by the analytical path at `c7b3bbf`. Not in `release_gates.yaml`.
- `tests/throughput_benchmark.rs::test_batch_oracle_throughput_1000` — debug-mode perf gate (12.79 configs/sec vs release-mode 157 / 900 floor). Passes under `--profile ci` / `--release`. Not in `release_gates.yaml`.

`cargo clippy --workspace --all-targets -- -D warnings` is also red on develop tip due to `fluxion-tauri/src-tauri/src/main.rs:27` (`tauri::generate_context!` proc-macro panic: `frontendDist` is set to `"../frontend/dist"` but the path doesn't exist). Pre-existing, unrelated to the audit.

These were not fixed by this PR because the issue scope is "profile-first audit / no-op close-out," not "make `cargo test --workspace` debug-mode green." All three are independently reproducible on a clean `develop` checkout with zero local changes.

## References

- Issue #3325: <https://github.com/anchapin/fluxion/issues/3325>
- Issue #3322 / PR #3344: fast-math feature + `src/physics/fp_algebraic.rs` helper.
- AGENTS.md §"Physics and Validation Guardrails", §"Commands That Are Easy to Guess Wrong", §"Toolchain, Security, and Generated Artifacts".
- `src/physics/fp_algebraic.rs` "Do NOT use in energy-balance-critical paths" (lines 32–61).
- `src/sim/orchestrator.rs` Issue #1439, #2520, #2687, #2769 design notes.
