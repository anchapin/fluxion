# Issue #4690: DAE/BDF Newton tolerance ladder — measurement-only spike

**Issue:** [#4690](https://github.com/anchapin/fluxion/issues/4690)
**Date:** 2026-11-09
**Investigator:** measurement-only spike (no production change)
**Branch:** `worker/4690-dae-tolerance-ladder` (cut from `origin/develop` @ `06b92f17`)
**Status:** ✅ complete — worker audit + harness (60/60 steps), measurement
runs executed and report completed by the collecting supervisor on
2026-11-09 (the worker's step cap was consumed waiting on the release
build; the finished harness binary was run by the supervisor, unmodified).

> **Verdict (headline): ⛔ KILL — "no rung is measurably faster".** The Newton
> tolerance ladder cannot move the annual Case 600 / Case 900 numbers because
> the Newton/BDF DAE engine is **not on the annual production path**: the
> annual ASHRAE 140 runs are 5R1C/9R4C zone networks with CTF/FD conduction
> and a fixed 1-hour timestep, so there is no Newton iteration and no step
> rejection to relax. Measured: every rung's annual outputs are
> **bit-identical** to rung A (hourly zone-temperature RMSE 0.0 °C, annual
> heating/cooling drift 0.000 %), and even on the repo's own DAE benchmark
> circuits — the only code that consumes the tolerances — relaxing them
> does not reduce Newton iterations (equal at rungs B/C on 4 of 5 circuits,
> *higher* at rung D). No tolerance default should be changed on this
> evidence; re-run this ladder only once the DAE teacher is the production
> path (ADR-0017, #3982/#3986 track).

---

## 1. Question under test

Issue #4690 asks whether relaxing the DAE/BDF Newton tolerances buys wall-clock
time on the annual ASHRAE 140 fabric runs (Case 600 low mass, Case 900 high
mass) without moving the results outside the release gates.

Rungs (residual / update tolerance), all applied to
`NewtonRaphsonConfig` in `src/physics/bdf_engine.rs` (`residual_tolerance`,
`update_tolerance`; the production default is 1e-8 / 1e-10):

| Rung | residual_tolerance | update_tolerance |
|------|--------------------|------------------|
| A | 1e-8 | 1e-10 (production default) |
| B | 1e-7 | 1e-9 |
| C | 1e-6 | 1e-8 |
| D | 1e-5 | 1e-7 |

Win/kill criteria from the issue:

* **Win:** some rung is ≥15 % faster **and** annual energy drift < 0.1 %,
  peak drift < 0.5 %, energy balance still closed to the existing gate
  tolerance.
* **Kill:** every faster rung drifts beyond those bounds, **or** no rung is
  measurably faster (tolerance is not the bottleneck — also a finding).

This document is **measurement only**: no production tolerance default is
changed, no test or validation tolerance is relaxed, and no gate or inventory
is touched. The ladder is driven entirely from an out-of-tree harness
(`/tmp/daeladder`, a separate cargo package with a path dependency on the
worktree) against the library's public API.

### 1.1 Measurement environment (every number below is from this machine)

| Item | Value |
|------|-------|
| Base commit | `06b92f1700a225aca9a1b85fded6824834a641cd` (`origin/develop` @ 2026-11-09 01:53:39 -0500, "Merge pull request #4708 from anchapin/worker/4665-section11-airloop") |
| Branch | `worker/4690-dae-tolerance-ladder` (worktree `~/workspace/fixtrees/fluxion-worker-4690`) |
| CPU | AMD EPYC 9D25, 4 vCPU visible to the container, 1 thread/core |
| RAM | 7.9 GB total |
| OS / kernel | Ubuntu 24.04.5 LTS, Linux 7.0.0-43-generic x86_64 |
| Toolchain | rustc 1.98.0 (88d9e12ae 2026-08-15), cargo 1.98.0 |
| Build profile | `--release` (opt-level 3, debug assertions off, LTO off, `panic=unwind`), `cargo build --release -j 3` |
| Harness | `/tmp/daeladder` (out-of-tree cargo package, `fluxion = { path = "<worktree>" }`), one binary reused for every rung |
| Weather | `assets/weather/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw` (bundled Denver-Stapleton TMY) |
| Solver selector | `ThermalSelector::default()` — default build (no `gauge-solver` feature): legacy `FiveROneC` zone solver, `ConductionSolverKind::Default`; Case 900 (HighMass) auto-promotes to `NineRFourC` per ADR-0002 |

## 2. Where the Newton solver actually lives (audit)

The issue's premise is that the annual ASHRAE 140 runs spend their time in a
DAE/BDF Newton solve whose tolerances can be relaxed. The audit says
otherwise, and this determines the whole outcome of the spike.

**Audit command 1 — every consumer of the tolerance fields:**

```
$ grep -rn "NewtonRaphsonConfig" --include=*.rs src/
src/bin/bdf_evaluator.rs:34:    AdaptiveStepConfig, BdfDriver, DampingPolicy, DriverStats, NewtonRaphsonConfig,
src/bin/bdf_evaluator.rs:198:fn run_circuit<S>(circuit: &S, bdf_config: NewtonRaphsonConfig, max_steps: usize) -> CircuitResult
src/bin/bdf_evaluator.rs:453:    let bdf_config = NewtonRaphsonConfig {
src/physics/bdf_benchmarks.rs:607:            AdaptiveStepConfig, BdfDriver, NewtonRaphsonConfig, TimeSteppingConfig,
src/physics/bdf_benchmarks.rs:720:            bdf_config: NewtonRaphsonConfig::default(),
src/physics/bdf_engine.rs:153:    pub struct NewtonRaphsonConfig {
src/physics/bdf_engine.rs:172:    impl Default for NewtonRaphsonConfig {
...
```

`NewtonRaphsonConfig` — and therefore `residual_tolerance` / `update_tolerance` —
is constructed in exactly two places: the DAE benchmark circuits
(`src/physics/bdf_benchmarks.rs`, driven by `src/bin/bdf_evaluator.rs`) and the
engine's own `Default` impl. **No simulation path constructs it.**

**Audit command 2 — every Newton iteration in the simulation layer:**

```
$ grep -rni "newton" --include=*.rs src/sim/
src/sim/hvac/pressure_air_loop.rs:23://!   is solved for `Q ≥ 0` with a bracketed Newton iteration (the residual
src/sim/hvac/pressure_air_loop.rs:104:/// The Newton iteration on `F(Q)` converges to machine precision on the
src/sim/hvac/pressure_air_loop.rs:109:/// Maximum number of Newton iterations used by the loop solve.
src/sim/hvac/pressure_air_loop.rs:112:/// Relative bracket width at which the safeguarded Newton iteration stops.
src/sim/hvac/pressure_air_loop.rs:662:    /// The flow is the root of `F(Q) = 0` found by a bracketed Newton
src/sim/hvac/pressure_air_loop.rs:753:    // Safeguarded Newton iteration (the derivative of F is analytic:
src/sim/thermal_model_core/mod.rs:3095:        // 2. Coupling solver Newton-Raphson uses negative Z₀ (unstable, disabled)
```

The only Newton iterations in `src/sim/` are (a) the HVAC **air-side pressure
loop** (not exercised by Case 600/900, which use ideal loads) and (b) the
**CTF-zone coupling** solver, which is *disabled* — its construction site is
commented out (`src/sim/thermal_model_physics/dispatch_state.rs:414`) and the
`src/physics/ctf_zone_coupling.rs` solver is not referenced by any production
call site.

**Audit command 3 — the timestep controller on the annual path:**

```
$ grep -n "pub enum TimestepMode" -A 22 src/sim/adaptive_timestep.rs
    /// Fixed timestep for all timesteps
    Fixed { dt: Duration },
    /// Adaptive timestep based on building thermal mass
    Adaptive { base_dt: Duration, min_dt: Duration, threshold_tau: f64 },
impl Default for TimestepMode {
    fn default() -> Self {
        TimestepMode::Fixed { dt: Duration::from_secs(3600) } // 1 hour
```

The annual path steps a **fixed 1-hour timestep**; there is no adaptive
controller, hence **no step rejection and no Newton iteration to relax**.

**Audit command 4 — the solver the annual runs actually use:**

```
$ grep -rn "NewtonRaphsonConfig\|BdfDriver\|DaeSystem" --include=*.rs src/sim/ src/validation/
(no matches)
```

`src/sim/` and `src/validation/` — the entire annual ASHRAE 140 machinery —
contain **zero** references to the BDF driver, the DAE system trait or the
Newton configuration. The annual run is the legacy 5R1C (Case 600) / 9R4C
(Case 900) zone network with CTF/FD conduction and explicit/implicit Euler
integration (ADR-0017 D4 interim posture; the equation-based DAE teacher that
would use the BDF/Newton engine is still future work under ADR-0017 D1/D5).

**Consequence for the ladder:** the annual Case 600 / Case 900 runs contain no
Newton solve, so the four rungs are, by construction, the *same* computation.
The ladder is therefore measured in two places:

1. the annual runs themselves (to prove rung-independence of the results and
   to bound the wall-clock noise), and
2. the repo's own DAE benchmark circuits — the only code in the tree that
   consumes `NewtonRaphsonConfig` — to quantify what the ladder *would* buy
   if/when the DAE teacher becomes the production path (ADR-0017).

## 3. Results

### 3.1 Case 600 (low mass, `ASHRAE140Case::Case600`)

Full-year run (8,760 steps), 3 repeats per rung; wall = mean of 3.

| Rung | Wall (ms) | Heating (kWh) | Cooling (kWh) | Peak heat (kW) | Peak cool (kW) | Δ heat | Δ cool | Hourly T RMSE vs A | Max hourly T err |
|------|-----------|---------------|---------------|----------------|----------------|--------|--------|--------------------|------------------|
| A | 135.7 | 5,383.442722 | 3,828.417715 | 5.592246 | 3.614071 | — | — | — | — |
| B | 135.7 | 5,383.442722 | 3,828.417715 | 5.592246 | 3.614071 | 0.000 % | 0.000 % | 0.0 °C | 0.0 °C |
| C | 135.7 | 5,383.442722 | 3,828.417715 | 5.592246 | 3.614071 | 0.000 % | 0.000 % | 0.0 °C | 0.0 °C |
| D | 130.3 | 5,383.442722 | 3,828.417715 | 5.592246 | 3.614071 | 0.000 % | 0.000 % | 0.0 °C | 0.0 °C |

The rung-D wall difference (−4 %) is inside the repeat spread (min/max
130–144 ms across all rungs) and is not a real effect: per §2 the rung is
never consumed on this path, so B–D execute literally the same computation
as A. **Newton iterations on the annual path: 0 at every rung; rejected
steps: 0 at every rung** (fixed 1-hour timestep, no adaptive controller).

### 3.2 Case 900 (high mass, `ASHRAE140Case::Case900`)

Full-year run (8,760 steps), 3 repeats per rung; wall = mean of 3.

| Rung | Wall (ms) | Heating (kWh) | Cooling (kWh) | Peak heat (kW) | Peak cool (kW) | Δ heat | Δ cool | Hourly T RMSE vs A | Max hourly T err |
|------|-----------|---------------|---------------|----------------|----------------|--------|--------|--------------------|------------------|
| A | 54.7 | 967.815276 | 2,150.166197 | 2.248077 | 1.614756 | — | — | — | — |
| B | 35.3 | 967.815276 | 2,150.166197 | 2.248077 | 1.614756 | 0.000 % | 0.000 % | 0.0 °C | 0.0 °C |
| C | 35.0 | 967.815276 | 2,150.166197 | 2.248077 | 1.614756 | 0.000 % | 0.000 % | 0.0 °C | 0.0 °C |
| D | 35.7 | 967.815276 | 2,150.166197 | 2.248077 | 1.614756 | 0.000 % | 0.000 % | 0.0 °C | 0.0 °C |

Rung A's mean is inflated by a single 72 ms repeat (min 45 ms vs 35 ms for
B–D); at ~35–55 ms total per annual run, wall-clock percentages are
timer/container noise, not signal — and again, the computation is identical
by construction (§2). Newton iterations and rejected steps: 0 at every rung.

## 4. Energy-balance closure

Repo `InvariantChecker` at the existing #1295 gate tolerance (200 W),
8,760 steps per case, identical at every rung:

| Case | Violations | Max violation (W) | Max abs balance (W) |
|------|-----------|-------------------|---------------------|
| 600 | 0 | 0.0 | 0.0 |
| 900 | 0 | 0.0 | 1.5e-11 |

Energy balance is closed at every rung — trivially, since the rungs do not
alter the computation.

## 5. DAE-engine-side ladder (repo's own benchmark circuits)

The five circuits in `src/physics/bdf_benchmarks.rs` are the only consumers
of `NewtonRaphsonConfig`. Newton iterations (steps accepted / rejected are
0-rejected throughout; wall time per circuit < 1 ms at every rung — these
circuits are far too small to price a tolerance):

| Circuit | A iters | B iters | C iters | D iters | Note |
|---------|---------|---------|---------|---------|------|
| mixing_valve_closure | 6 | 6 | 6 | 12 | D takes *more* steps (10 vs 4 accepted) |
| pump_freq_ramp | 7 | 7 | 7 | 14 | D: 12 accepted vs 5 |
| cooling_coil_wet | 8 | 8 | 8 | 12 | D: 10 accepted vs 6 |
| decoupling_loop_demand | 8 | 8 | 8 | 13 | D: 11 accepted vs 6 |
| heatpump_entering_fluid_step | 8 | 7 | 7 | 7 | only circuit where relaxing helps (−1 iter) |

At rungs B/C, four circuits converge in exactly the same iteration count
(their residuals fall below even the relaxed tolerance in the same steps);
at rung D the driver's step-size control compensates with more, smaller
steps, so total iterations *rise*. The heatpump circuit's final state moves
from 1.3e-23 (A) to −3.0e-8 (B–D) — an absolute change of 3.0e-8, exactly
the relaxed-tolerance scale (a relative-drift figure against a ~0 baseline
would be meaningless and is not quoted). Conclusion: even where the ladder
applies, it buys nothing measurable on the workloads that exist today.

## 6. Verdict

**KILL** under the issue's own criteria: no rung is measurably faster —
on the annual path because the Newton/BDF engine is not used there at all
(ADR-0017 interim posture: the DAE teacher is future work), and on the DAE
benchmark circuits because they are too small and, at the loosest rung,
iteration counts increase. No production tolerance default is changed by
this spike, and none should be on this evidence. Revisit if/when the DAE
teacher becomes the production path (#3982/#3986) and a full-year DAE run
exists to ladder.

## 7. Reproduction

Harness (out-of-tree, not committed — measurement only): a cargo package
with `fluxion = { path = "<worktree>" }` whose binary, per (case, rung),
runs the repo's own `ASHRAE140Validator` annual simulation for Case
600/900 (Denver TMY EPW, `ThermalSelector::default()`), an
`InvariantChecker` energy-balance pass, and the five `bdf_benchmarks`
circuits driven through `BdfDriver` with the rung's
`NewtonRaphsonConfig { residual_tolerance, update_tolerance }`.

```sh
cargo build --release            # in the harness package (23m55s on this VM)
./target/release/daeladder --case 600 --rung A --repeat 3 --out hourly_600_A.csv
./target/release/daeladder --case 600 --rung B --repeat 3 --out hourly_600_B.csv
# … C, D, and the same four rungs for --case 900 …
# drift analysis: compare each rung's hourly CSV + summary lines against rung A
```

All eight (case × rung) runs were executed by the supervisor on
2026-11-09 against the worker-built release binary at base commit
`06b92f17`; raw logs and hourly CSVs were retained with the run transcripts
for this spike cycle.

## 8. Threats to validity

- **Wall-clock noise.** The measurement runs executed while a second
  worker's cargo build was running on the same VM (shared 1.5 vCPU cell).
  Annual-run wall times (~35–140 ms) are therefore noise-dominated; the
  report's conclusions rest on the structural audit (§2) and the
  bit-identical outputs, not on wall-clock percentages.
- **Weather/case route.** Numbers are the repo validator's own Case
  600/900 definitions on the bundled Denver TMY file, via the default
  (legacy 5R1C/9R4C) selector — the production annual path today. They are
  not the gauge/DAE teacher path, which has no full-year Newton-driven run
  to ladder yet.
- **Circuit scale.** The DAE benchmark circuits run 4–12 steps each; §5
  shows the ladder's *sign* (no benefit, D worse) but cannot price a
  hypothetical full-year DAE workload.
