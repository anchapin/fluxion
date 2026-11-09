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
- **Workload 3 — multi-zone / BatchOracle annual run: NOT MEASURED.** No timing,
  no breakdown, no ranking for this workload in this run.

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
