# Performance Profile — Annual Single-Zone Run (fluxion #4688, workload 1)

> **Status: COMPLETE for workload 1 (narrowed scope).** The measurements were
> made by worker run `fluxion-4688-1794206213` (flagship, MiniMax-M3), which
> hit its step cap before filling in §§4–7; those sections were completed by
> the supervisor from the worker's own logged measurement artifacts
> (`/tmp/prof4688/`: `criterion_quick.log`, `attrib_run1.log`,
> `attrib2_runs.log`, `getenv_8760.txt`, `getenv_timing.txt`), and the
> headline wall clock was independently re-run by the supervisor
> (`attrib2`, §4.2). No number below comes from the worker's prose summary.
>
> **Scope note (narrowed):** issue #4688 asks for a profile-first ranking of ALL
> performance spikes. This run covers **workload 1 only** — a single-zone
> annual hourly simulation (8760 timesteps). Workloads 2 (DAE/BDF sub-hourly)
> and 3 (multi-zone / BatchOracle) are **NOT measured** in this run. See
> [§8 Not measured](#8-not-measured-workloads-23).

---

## 1. Base commit and branch

| Item | Value |
|---|---|
| Repository | `fluxion` (anchapin/fluxion) |
| Branch | `worker/4688-profile-workload1` |
| Base commit | `6cef68bd1c6b72340086988010cafa3ec0084aa1` |
| Base commit subject | `Merge pull request #4704 from anchapin/worker/4666-section9b-harness` |
| Base commit date | `2026-11-09 01:23:20 -0500` |
| Upstream base | `origin/develop @ 6cef68bd` |
| Working-tree changes | `docs/PERF_PROFILE_ANNUAL.md` only (this report). No production code changes. |

Command:

```sh
git rev-parse HEAD
git log -1 --format='%H%n%ci%n%s'
git status --porcelain
```

## 2. Machine and build profile

| Item | Value |
|---|---|
| CPU | AMD EPYC 9D25 126-Core Processor (as reported by `/proc/cpuinfo`; the container is limited to 4 CPUs) |
| Cores visible to the run | 4 (`nproc`) |
| RAM | 8 124 908 kB total (≈7.75 GiB) |
| OS | Ubuntu 24.04.5 LTS, Linux 7.0.0-43-generic x86_64 |
| Kernel | `7.0.0-43-generic` |
| Toolchain | rustc 1.98.0 (88d9e12ae 2026-08-18), cargo 1.98.0 (797e8a9bc 2026-08-05) |
| Toolchain pin | `rust-toolchain.toml` → `channel = "1.98.0"` (workspace MSRV, issue #3321) |
| Cargo profile used | `bench` profile (inherits `release`): `opt-level = 3`, `lto = "thin"`, `codegen-units = 4`, `split-debuginfo = "packed"`, `strip = true` |
| Criterion | `criterion = "0.8"` (workspace dev-dependency) |
| Profiling tools | **None installed.** Wall-clock timing + the repo's own criterion harness only. No `perf`, no flamegraph, no sampler. |
| Load on machine | Single-run measurements; the build was the only concurrent activity. |

Commands:

```sh
grep -m1 "model name" /proc/cpuinfo; nproc
grep MemTotal /proc/meminfo
uname -a
export PATH="$HOME/.cargo/bin:$PATH"; cargo --version; rustc --version
```

## 3. Method

**Workload 1 as executed.** The repo's own 8760-timestep criterion benchmark
(`benches/benchmark_8760_timesteps.rs`) is used as the "single-zone annual
hourly" workload. Its `single_config_8760/analytical` sub-benchmark builds a
one-zone `ThermalModel<VectorField>` and runs `solve_timesteps(8760, …)` with
`use_ai = false` (analytical loads). It is **not** a literal ASHRAE 140 Case 600
geometry run — the bench constructs its own model via `SurrogateManager`, so
numbers from it are labelled "8760-timestep single-zone annual (bench
harness)" throughout this report. The Case 600 entry point is timed separately
in §5.3 where it exists.

**Commands to be run (exact):**

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cd ~/workspace/fixtrees/fluxion-worker-4688

# (1) Build the bench in release
cargo bench --bench benchmark_8760_timesteps --no-run

# (2) Workload-1 measurement, workload-1 sub-benchmark only, shortened window
cargo bench --bench benchmark_8760_timesteps -- 'single_config_8760' --quick

# (3) Case 600 live entry point (separate package under examples/)
cargo run --release --manifest-path examples/Cargo.toml --bin ashrae_case_demo
```

## 4. Results — workload 1 wall clock

### 4.1 Criterion sub-benchmark `single_config_8760/analytical`

| Run | Command / flags | Criterion mean (time/iteration) | 95% CI | Notes |
|---|---|---|---|---|
| 1 | `cargo bench --bench benchmark_8760_timesteps -- 'single_config_8760' --quick` | **97.982 ms** | [97.689, 98.055] ms | From `criterion_quick.log`; one full 8760-step solve per iteration |

### 4.2 End-to-end wall clock, direct repetitions

Measured with the worker's `attrib2` harness (full `solve_timesteps(8760)`
on the same bench model, 7 reps, min/median reported):

| Run set | Min (ms) | Median (ms) | Notes |
|---|---|---|---|
| Worker, quiet machine | **88.776** | 89.521 | `attrib2_runs.log` |
| Worker, earlier session (loaded) | 104.4 | 186.9 (max) | `attrib_run1.log`: reps [106.0, 186.9, 104.4] ms |
| Supervisor re-run (loaded) | 113.959 | 124.073 | Independent confirmation, same binary, busier machine |

Fixed (non-step) cost is negligible: the 100-step solve takes ~1.03 ms, and
the decomposition puts fixed cost at ~0.015 ms (~0.02% of the annual run).
Marginal cost is **~0.0101 ms/step** on the quiet machine (~0.0130 ms/step
under load). Warm-up + 8760 steps measured [110.8, 162.3, 131.8] ms under
load (`attrib_run1.log`).

### 4.3 ASHRAE 140 Case 600 live entry point (`examples/ashrae_case_demo`)

**Not timed.** The example lives in the separate `examples/` workspace;
building it was not completed within the run's step budget. The workload-1
numbers above are therefore the bench harness's own single-zone model,
labelled as such — not a literal Case 600 geometry run.

## 5. Attribution — % wall-clock breakdown by area (workload 1)

**Attribution method:** two independent decompositions, both from the
worker's logged artifacts. (1) A manual-loop decomposition (`attrib2`):
full solve vs hand-rolled step loop with gains vs without gains — the
differences isolate the gains-application share; everything else is the
physics step as a whole (no finer sub-phase split is exposed without code
changes). (2) An `LD_PRELOAD` `getenv` interposer (`countenv2`) counting
and timing environment-variable reads during one annual run: the hot loop
reads `FLX_*` feature-flag env vars per timestep instead of caching them.

| Area | Share of annual run | Measured value | How it was obtained |
|---|---|---|---|
| **`getenv` env-flag reads (`FLX_*`)** | **≈22–50 ms net of a ~89–98 ms quiet-machine run (roughly a quarter to a half; method-sensitive, see §7)** | **762,121 calls per annual run (≈87/step)**; two interposer variants measured raw totals 57.2 / 85.40 ms, clock-pair overhead ~46 ns/call, **net 22.27 / 49.95 ms** | `getenv_timing.txt`, `getenv_8760.txt` (interposer counts + timing, second variant); worker's first-variant figures from its run log |
| Physics step (everything else in the step loop) | essentially all remaining marginal time | marginal 0.0101 ms/step total; gains application ≈ 0 (within noise, −1.3%) | `attrib2_runs.log` decomposition: full solve 88.776 ms ≈ manual-with-gains 88.304 ms ≈ manual-no-gains 89.431 ms |
| Fixed setup (model construction etc.) | ~0.02% | ~0.015 ms | 100-step vs 8760-step decomposition |

Top `getenv` callers per annual run (`getenv_8760.txt`): `FLX_R89_MASS_DT_ALIGN`
210,240; `FLX_R84_PERCLASS` 113,880; `FLX_R86_VENT_AIR_ONLY` 113,880;
`FLX_R76_NO_NUM_REFRESH` 105,120; `FLX_R76_NO_MASS` 105,120; ~20 further
`FLX_*` flags at 8,760 each (once per step each).

## 6. Top area (named and quantified)

**Per-timestep environment-variable flag reads.** One annual run makes
762,121 `getenv` calls (~87 per timestep) re-reading `FLX_*` feature flags
in the hot loop; direct interposer timing attributes ≈22–50 ms net
(57–85 ms raw, two variants) to those calls, against a quiet-machine annual wall clock
of 88.8–98.0 ms. No other single area comes close: the gains application
is unmeasurable (≈0) and fixed setup is ~0.02%. The obvious follow-up
candidate for the perf-spike series is caching flag reads at model
construction — that is a recommendation for a later spike, not part of
this measurement-only run, and no production code was changed.

## 7. Failures, deviations and honest limits

- **Contention:** this is a shared VM; wall clock for the identical run
  ranged 88.8 ms (quiet) to 186.9 ms (loaded). Only min/median statistics
  and same-run ratios should be quoted; the getenv share is stated
  against the quiet-machine wall clock and would shrink under load.
- **Interposer method:** the net getenv figures subtract a measured
  clock-pair overhead (~46 ns/call) from the raw totals; the raw totals
  (57.2 / 85.40 ms across two interposer variants) are the hard upper
  bounds and the nets (22.27 / 49.95 ms) depend on that subtraction being
  right — the spread between variants is itself the method uncertainty.
  The call count (762,121) is exact.
- **Bench model ≠ Case 600:** workload 1 as executed is the bench
  harness's own single-zone model (see §3, §4.3). The DAE/BDF path is not
  exercised at all — it is workload 2, not measured.
- **No solver counters:** this path (analytical 5R1C-style solve) exposes
  no Newton/BDF counters, so the issue's solver-counter table could not
  be produced for workload 1.
- **No sampler:** no `perf`/flamegraph was available or installed (per
  the run's hard limits); attribution rests on the two decompositions
  above, not on sampled stacks.

## 8. Not measured (workloads 2–3)

- **Workload 2 — DAE / BDF sub-hourly annual run: NOT MEASURED.** No timing, no
  breakdown, no ranking for this workload in this run.
- **Workload 3 — multi-zone / BatchOracle annual run: NOT MEASURED** *in the
  workload-1 run above* (this section was written before workload 3 was
  measured). Workload 3 is measured in **§§15–19** below, and the closing
  cross-workload ranking is **§20**.

**Issue #4688 therefore remains OPEN after this run.** The full spike
(ranking *all* the performance spikes) is **not** complete: only workload 1 of
three was measured, and only on one machine with one harness configuration. The
numbers above are a single-machine, single-harness observation and are not a
general claim about fluxion's performance.

## 9. Reproduction

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cd ~/workspace/fixtrees/fluxion-worker-4688
cargo bench --bench benchmark_8760_timesteps -- 'single_config_8760' --quick
git status --porcelain   # must show ONLY docs/PERF_PROFILE_ANNUAL.md
```

---

# Workload 2 — DAE/BDF high-mass annual run, sub-hourly steps (fluxion #4688)

> **Scope:** workload 2 only. Workload 1 (above) is already complete and is
> **not** redone here. This section profiles the **high-mass (Case 900 family)
> annual run on the DAE/BDF path at sub-hourly steps** (6-minute steps =
> 87,600 steps/year), and reports (a) solver counters aggregated over the run
> and (b) a wall-clock breakdown by area.
>
> Every number below is **this VM's number** for the machine/build named in
> §10.1–10.2 and is not a general claim about fluxion's performance.

## 10. Base commit, machine and build profile (workload 2)

### 10.1 Base commit and branch

| Item | Value |
|---|---|
| Repository | `fluxion` (anchapin/fluxion) |
| Branch | `worker/4688-workload2-profile` |
| Base commit | `711cbbf7982b8a38e0d54f71ce5dd672d3104744` |
| Base commit subject | `Merge pull request #4721 from anchapin/worker/4691-frozen-jacobian` |
| Base commit date | `2026-11-09 10:22:44 -0500` |
| Upstream base | `origin/develop @ 711cbbf7` |
| Working-tree changes | `docs/PERF_PROFILE_ANNUAL.md` only (this report). No production code changes. |

### 10.2 Machine and build profile (this VM)

| Item | Value |
|---|---|
| CPU | AMD EPYC 9D25 126-Core Processor (as reported by `/proc/cpuinfo`; the container is limited to 4 CPUs) |
| Cores visible to the run | 4 (`nproc`) |
| RAM | 8 124 908 kB total (≈7.75 GiB) |
| OS | Ubuntu 24.04.5 LTS, Linux 7.0.0-43-generic x86_64 |
| Kernel | `7.0.0-43-generic` |
| Toolchain | rustc 1.98.0 (88d9e12ae 2026-08-18), cargo 1.98.0 (797e8a9bc 2026-08-05) |
| Toolchain pin | `rust-toolchain.toml` → `channel = "1.98.0"` |
| Cargo profile used | `release` (`opt-level = 3`, `lto = "thin"`, `codegen-units = 4`, `split-debuginfo = "packed"`, `strip = true`) |
| Profiling tools | **None installed** (no `perf`, no flamegraph, no sampler). Wall-clock timing + the engine's own counters only. |

### 10.3 Method (workload 2)

A temporary integration-test harness (deleted at collection; it is not
part of this diff) drove the shipped `BdfTimeStepper` directly on the
Case-900-family DAE (33 states) at fixed 6-minute steps for a full
87,600-step year-equivalent, reading the engine's own per-step counters
(Newton iterations, Jacobian assemblies/factorizations, accepted and
rejected steps, final residual — the Issue #3339 counter set). A second,
instrumented arm intended to split wall clock by area (solar, longwave,
conduction, zone balance, HVAC, other) did not complete before the
worker run reached its step cap; §12 records that gap instead of
inventing the split.

## 11. Results — workload 2 solver counters

Full 87,600-step run, shipped BDF stepper, release build, this VM:

| Counter | Value |
|---|---|
| Steps accepted | 87,600 |
| Steps rejected | 0 (rejected-step fraction 0.0) |
| Newton iterations, total | 87,602 |
| Newton iterations / step | 1.00002 (87,599 steps converged in 1 iteration; 1 step took 3) |
| Jacobian assemblies | 2 |
| Jacobian factorizations | 2 |
| Solve calls | 87,600 |
| Final residual | 3.04e-23 |
| Stepper wall clock | 123.4 ms for the full 87,600 steps |

Headline: on this workload the Newton solve is already effectively
free — one iteration per step, and the Jacobian is assembled and
factorised only twice in a full year of sub-hourly steps (the cached /
frozen-Jacobian behaviour is already what the shipped path does). Any
spike premised on Newton iteration cost on this path (frozen-Jacobian
reuse, warm-start predictors, tolerance ladders) has almost nothing to
win here; that matches the #4690 KILL (the premise of an expensive
iterative solve on the annual path did not hold).

## 12. Results — workload 2 wall-clock breakdown by area

NOT MEASURED in this slice. The instrumented area-split arm did not
complete (see §13). The one defensible wall-clock statement is §11's:
the DAE stepper itself accounts for ~123 ms over a full sub-hourly
year on this VM, so for a full production run the wall clock must be
dominated by everything around the stepper (envelope/solar/weather/
output work in the production step), not by the BDF Newton solve. A
future slice that wants the area split should instrument the
production `step_physics` path directly rather than the stepper.

## 13. Failures, deviations and honest limits (workload 2)

- The harness drove the shipped `BdfTimeStepper` on the Case-900 DAE
  directly; it did NOT run the full production binary end-to-end. The
  counters are the engine's own and are exact for the stepper path;
  they do not by themselves characterise a production annual run.
- The area-breakdown arm (§12) is incomplete — the worker run hit its
  60-step cap while extending the instrumentation. No area percentages
  are claimed.
- Wall-clock figures are this VM's (4 visible CPUs, shared host) and
  are stated only as this VM's numbers.
- Sections 10.3–14 were completed by the supervisor at collection from
  the worker's logged test output (raw counter line preserved in the
  run transcript); the temporary harness file was deleted and never
  formed part of the diff. Worker run: fluxion-4688 (balanced,
  MiniMax-M2.7, 60/60 steps).

## 14. Reproduction (workload 2)

The harness was temporary and is not in the tree. To re-measure: write
a test that constructs the Case-900 DAE residual used by the shipped
`BdfTimeStepper`, steps it 87,600 times at dt = 360 s, and prints the
stepper's Issue #3339 counters (`newton_iterations`, assemblies,
factorizations, accepted/rejected steps, `last_final_residual`).

---

# Workload 3 — Multi-zone (10 zones) + BatchOracle population context (fluxion #4688)

> **Scope:** workload 3 only. Workload 1 (§§1–9) and workload 2 (§§10–14) are
> already complete and are **not** redone here; §20 cites them.
>
> Every number below is **this VM's number** for the machine/build named in
> §15.2 and is stated only as this VM's measurement, never as a general claim
> about fluxion's performance.

## 15. Base commit, machine and build profile (workload 3)

### 15.1 Base commit and branch

| Item | Value |
|---|---|
| Repository | `fluxion` (anchapin/fluxion) |
| Branch | `worker/4688-profile-workload3` |
| Base commit | `3e66c9b1ff4513aac7856b79bad29811f8934c37` |
| Base commit subject | `Merge pull request #4735 from anchapin/docs/repo-contract` |
| Base commit date | `2026-11-10 06:39:23 -0700` |
| Working-tree changes | `docs/PERF_PROFILE_ANNUAL.md` only (this report). No production code changes. |

Commands:

```sh
git rev-parse HEAD
git log -1 --format='%H%n%ci%n%s'
git status --porcelain
```

### 15.2 Machine and build profile (this VM)

| Item | Value |
|---|---|
| CPU | AMD EPYC 9D25 126-Core Processor (`/proc/cpuinfo`; container limited to 4 CPUs) |
| Cores visible to the run | 4 (`nproc`) |
| RAM | 8 124 900 kB total (≈7.75 GiB); heavily shared host — only ≈0.5 GiB available at build time |
| OS | Ubuntu 24.04.5 LTS, Linux 7.0.0-44-generic x86_64 |
| Kernel | `7.0.0-44-generic` |
| Toolchain | rustc 1.98.0 (88d9e12ae 2026-08-18), cargo 1.98.0 (797e8a9bc 2026-08-05) |
| Toolchain pin | `rust-toolchain.toml` → `channel = "1.98.0"` |
| Cargo profile used | `bench` profile (inherits `release`): `opt-level = 3`, `lto = "thin"`, `codegen-units = 4`, `split-debuginfo = "packed"`, `strip = true` |
| Criterion | `criterion = "0.8"` (workspace dev-dependency); `--quick` mode used to shorten the measurement window |
| Profiling tools | **None installed.** Wall-clock timing + the repo's own criterion harnesses only. |

### 15.3 Method (workload 3)

**Workload 3a — multi-zone throughput.** The repo's own criterion benchmark
`benches/multi_zone_throughput.rs` (Issue #2522) is used. What it actually
constructs (read from the source):

- a **10-zone** `ThermalModel<VectorField>` (`ZONE_COUNT = 10`, matching the
  `multi_zone.zones: 10` release-gate setting);
- one `BatchOracle::from_model(base_model)` wrapper;
- a synthetic population of configurations
  `[u_value, heating_setpoint, cooling_setpoint]` (StdRng, seed 42) of size
  `{10, 100, 1000, 10 000}`;
- per iteration: clone the population and call
  `BatchOracle::evaluate_population(pop, false)` (analytical mode, no ONNX).

**This is a population-throughput proxy, NOT a full 8760-step annual
multi-zone simulation.** The bench never runs a timestep loop; it measures
how many *configurations* per second the batch oracle can evaluate against
one 10-zone model. It is therefore reported throughout as
"10-zone batch-population throughput (bench harness)", not as an annual
multi-zone run. `sample_size(10)` is set by the bench for the larger
populations to bound wall time; criterion `Throughput::Elements(size)`
reports configs/sec directly, the same metric the
`multi_zone.min_configs_per_sec: 10` release gate uses.

**Workload 3b — BatchOracle population context.** The repo's own criterion
benchmark `benches/batch_oracle_bench.rs` (Issue #4696) is used. It measures
the same `BatchOracle::evaluate_population` call on a 10-zone `ThermalModel`
over populations `{100, 200, 500, 1000}` in two modes — `analytical`
(`use_surrogates = false`) and `surrogates` (`use_surrogates = true`) — plus
three "output materialisation ceiling" sub-benchmarks (API-boundary output
per config, counterfactual full-hourly output, per-config model clone) that
bound what any aggregate-only output mode could save.

**Exact commands (workload 3):**

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cd ~/workspace/fixtrees/fluxion-worker-4688

# (1) Build the multi-zone bench (release, bench profile). Requires the
#     `multi-zone` feature (Cargo.toml: required-features = ["multi-zone"]).
cargo bench --bench multi_zone_throughput --features multi-zone --no-run

# (2) Build the BatchOracle bench.
cargo bench --bench batch_oracle_bench --no-run

# (3) Workload 3a — shortened criterion window (one bench binary at a time).
cargo bench --bench multi_zone_throughput --features multi-zone -- 'population_100' --quick

# (4) Workload 3b — shortened criterion window.
cargo bench --bench batch_oracle_bench -- 'batch_oracle_analytical' --quick
```

## 16. Results — workload 3a wall clock (10-zone batch population throughput)

Measured by the supervisor at collection (the worker run was killed by a
runtime restart at step 28/60 during the bench build; the build was finished
and the benches run by the supervisor from the worker's method in §15.3,
exactly as for workloads 1 and 2). Command: `cargo bench --bench
multi_zone_throughput --features multi-zone -- --quick`, bench profile, this
VM (shared host, ~1.5 effective CPU — see §15.2).

| Sub-benchmark | Criterion time / iteration | Throughput | Per-config cost |
|---|---|---|---|
| `multi_zone_throughput_10_zones/population_10` | [3.2660 s, **3.2676 s**, 3.2739 s] | [**3.0604 elem/s**] | ~327 ms |
| `multi_zone_throughput_10_zones/population_100` | **63.660 s** (single-sample estimate as printed) | **1.5709 elem/s** | ~637 ms |
| `multi_zone_throughput_10_zones/population_1000` | NOT COMPLETED — see below | — | — |
| `multi_zone_throughput_10_zones/population_10000` | NOT COMPLETED — see below | — | — |

The run was stopped by the supervisor after `population_100`: at the
measured ~0.64 s/config, a single `population_1000` iteration would take
~640 s and `population_10000` ~6,400 s (extrapolation from the measured
per-config cost, labelled as such — not a measurement). Two defensible
points are recorded instead of two extrapolated ones.

Context, with the comparison the issue asks for: the release gate the bench
encodes is `multi_zone.min_configs_per_sec: 10` on CI hardware; this VM
measured 1.57–3.06 elem/s. This VM is a heavily shared host and these are
this VM's numbers only — they do not show the gate failing on CI hardware.

## 17. Results — workload 3b (BatchOracle population context)

Same collection circumstances as §16. Commands: `cargo bench --bench
batch_oracle_bench --no-run` (build), then filtered `--quick` runs.

| Sub-benchmark | Criterion time / iteration | Derived |
|---|---|---|
| `batch_oracle_analytical/100` | **58.494 s** (single-sample estimate as printed) | 1.71 configs/s; ~585 ms/config |
| `batch_oracle_analytical/1000` | NOT COMPLETED — started, did not finish inside a 600 s cap | consistent with ~0.59 s/config extrapolation |
| `batch_oracle_surrogates/*` | NOT RUN — the analytical points already fix the per-config cost scale on this VM; surrogate mode is #3987's subject, not this profile's | — |

The `batch_oracle_analytical/100` figure (58.494 s) independently
corroborates §16's `population_100` figure (63.660 s) for the same
underlying call (`BatchOracle::evaluate_population`, 10-zone model,
analytical mode) through a second bench harness: ~0.59–0.64 s/config on
this VM.

The issue's context figures (~157 configs/sec CI throughput gate vs a 150
floor, ~900/sec release per SCORECARD) are for the single-zone BatchOracle
path on CI/release hardware. The 10-zone per-config cost measured here is
not comparable to them beyond showing that per-config evaluation — not
population plumbing — is where the time is (§18).

## 18. Attribution — workload 3 top area

**Top area: the per-config evaluation compute inside
`BatchOracle::evaluate_population` itself — essentially 100% of the
population-path wall clock on this VM.** The two candidate overheads the
bench suite was built to bound were both measured and are negligible:

| Candidate overhead | Measured | Share of ~585–637 ms/config |
|---|---|---|
| Per-config base-model clone (`batch_oracle_per_config_model_clone/100`: [763.18 µs, **765.92 µs**, 776.86 µs] per 100 clones; `/1000`: [30.232 ms, **30.442 ms**, 31.284 ms] per 1000) | ~7.7 µs/config (pop 100); ~30 µs/config (pop 1000) | <0.01% |
| API-boundary output materialisation (`batch_oracle_api_boundary_output/100`: ~**130 ns**; `/1000`: ~**1.42 µs**) | ~1.3 ns/config | ~0% |
| **Per-config evaluation (residual: total minus the above)** | **~585–637 ms/config** | **>99.99%** |

So on the population path there is no plumbing win to be had: cloning,
output materialisation, and population management are noise. Any speedup
must come from the per-config analytical evaluation itself — which is the
same physics step whose single-zone cost workload 1 profiled (§§4–6),
plus the 10-zone model size.

A finer split *inside* per-config evaluation (solar / conduction / zone
balance / gains) is **NOT MEASURED** for workload 3: no sampling profiler
is installed on this VM, and the workload-1 decomposition method (manual
loop with/without gains) was built around the single-zone bench harness,
not the BatchOracle population path. Recorded as a gap, not filled in.

## 19. Failures, deviations and honest limits (workload 3)

- The worker run (`fluxion-4688-1794321818`, balanced, MiniMax-M2.7) was
  killed by a runtime restart at step 28/60 while the multi-zone bench was
  still compiling. Its method sections (§15) and report skeleton survived
  in the worktree; the supervisor finished the builds and ran every
  measurement in §§16–18, and no number above comes from the worker's
  prose. This is the same supervisor-completion pattern as workloads 1
  and 2.
- Workload 3 as the repo's benches define it is a **population-throughput
  proxy, not a literal 10-zone 8760-step annual simulation** (§15.3). No
  bench in the repo runs a multi-zone annual timestep loop; that workload
  does not exist to be measured without writing a new harness, which this
  measurement-only spike does not do.
- `population_1000` / `population_10000` (3a) and
  `batch_oracle_analytical/1000` (3b) were not completed — wall-time on
  this shared VM, recorded openly in §§16–17 rather than extrapolated
  into print as results.
- The allocation profile from the issue's spike design (allocations and
  bytes per timestep by module) was not produced for any workload; no
  allocation profiler is installed on this VM. Still an open gap in
  #4688's design, alongside the §12 workload-2 area split.
- All figures are this VM's (4 visible CPUs on a heavily shared host,
  ~1.5 effective CPU, bench profile at base commit `3e66c9b1`). They rank
  areas; they are not release or CI numbers.

## 20. Closing summary — cross-workload ranking (workloads 1, 2, 3)

The ranking table #4688 exists to produce — top area per workload, each
row citing the sections that measured it. All figures are the named
harnesses' numbers on the VMs that ran them; see each row's sections for
machine and commit.

| Workload | Top area | Quantified | Source |
|---|---|---|---|
| 1 — single-zone annual, hourly (8760-step bench harness) | Per-timestep `FLX_*` environment-flag reads (`getenv`) | 762,121 calls/run (~87/step); net ~22–50 ms of a ~89–98 ms quiet-machine annual run — roughly a quarter to a half | §§5–6 |
| 2 — high-mass DAE/BDF annual, sub-hourly (87,600 steps) | NOT the Newton solve: the BDF stepper totals 123.4 ms/year at ~1.0 Newton iter/step, Jacobian assembled/factorised 2×/year — the wall clock must sit in the production work around the stepper; its area split is NOT MEASURED | Stepper ~123.4 ms per 87,600-step year | §11; gap at §12 |
| 3 — 10-zone BatchOracle population (throughput proxy) | Per-config evaluation compute inside `evaluate_population` | ~0.59–0.64 s/config on this VM (1.57–3.06 elem/s); per-config clone ~7.7–30 µs and output materialisation ~ns — <0.01% combined | §§16–18 |

What this ranking does to the sibling spikes, stated once, from the
measurements above and the kills already on record: anything premised on
Newton-iteration cost on the DAE path (frozen-Jacobian reuse #4691,
warm-start #4692, tolerance ladder #4690 — killed) has almost nothing to
win on workload 2 (§11). The two measured, quantified wins available are
(1) workload 1's per-step env-flag reads — a caching fix, not a solver
change — and (2) whatever dominates per-config evaluation on the
population path, which needs the finer intra-evaluation split that no
workload has yet measured (§18) before a spike can be aimed at it.
Remaining #4688 gaps, honestly open: the §12 workload-2 area split, the
§18 intra-evaluation split, and the allocation profile (§19).
