# Issue #4697: Per-surface parallelism within a timestep — measurement-only spike

**Issue:** [#4697](https://github.com/anchapin/fluxion/issues/4697)
**Date:** 2026-11-09
**Investigator:** measurement-only spike (flag-gated prototype NOT built — see verdict)
**Branch:** `worker/4697-per-surface-parallelism` (cut from `origin/develop` @ `017b8bf8`)
**Status:** ✅ complete — worker audit + instrumentation + bench (60/60 steps);
measurement runs executed by the worker and independently re-run by the
collecting supervisor on 2026-11-09; this report was completed by the
supervisor because the worker's step cap was consumed by release builds.

> **Verdict (headline): ⛔ KILL — the per-surface phase is ~5% of step time.**
> On a single-config 50-zone run, the two parallelisable per-surface phases
> (conduction/sol-air preparation + the 5R1C per-surface exterior-boundary
> loop) together are 4.5–5.0% of wall clock across three independent
> measurement runs. The Amdahl ceiling is 1.05× at infinite threads and
> 1.04× at 4 threads — far under the issue's 1.3× kill threshold and nowhere
> near the 2× win bar. The coupled zone solve dominates; per-surface Rayon
> parallelism within a timestep is not worth building for this model shape.

## What was measured

Instrumentation (measurement-only, never feeds a thermal result):
`PerSurfacePhaseAudit` counters on `ConductionState`
(`src/sim/thermal_model_data/conduction_state.rs`), timing
`ThermalModel::prepare_solvers_and_sol_air` (preparation) in
`src/sim/thermal_model_core/mod.rs` and the per-surface exterior-boundary
loop in `src/sim/thermal_model_physics/zone_balance.rs`, exposed via
`ThermalModel::per_surface_phase_audit()`.

Bench: `benches/per_surface_phase_bench.rs` — a NEW single-config bench
driving one `ThermalModel` (50 zones, 720 timesteps = 30 days, release
profile). It is deliberately NOT a `BatchOracle` population bench
(`benches/rayon_chunks_bench.rs` / `benches/multi_zone_throughput.rs` are
population benches for issue #1439 and were used only as Rayon-pattern
references).

```text
FLUXION_BENCH_REPEATS=5 cargo test --release --bench per_surface_phase_bench -- --nocapture --test-threads=1
```

## Results (medians)

| Run | Repeats | Median wall | Preparation | Boundary | Phase share | Amdahl ∞ | Amdahl 4-thread |
|---|---|---|---|---|---|---|---|
| worker | 5 | 139.76 ms | 0.131 ms | 6.141 ms | 4.49% | 1.047× | 1.035× |
| worker | 9 | 125.62 ms | 0.132 ms | 6.180 ms | 5.02% | 1.053× | 1.039× |
| supervisor re-run | 5 | 124.46 ms | 0.129 ms | 5.947 ms | 4.88% | 1.051× | 1.038× |

Energy output was bit-identical across every repeat
(`-3542.671549296761`), confirming the counters do not perturb results.
A 2-zone / 24-step smoke run showed the same shape (3.49% share).

## Reading

- The boundary loop is ~98% of the measured phase; preparation is noise.
- ~95% of step time is outside both phases (coupled zone solve and the
  rest of the step path) — that is the Amdahl wall for this spike.
- Per the narrowed brief's decision gate (phase < 20% → stop), no
  flag-gated prototype and no bit-identity test were built: there is
  nothing worth parallelising until the coupled-solve share changes
  (e.g. a future profile after DAE-teacher migration, #3982 track).

## Deferred from the issue's full design (explicit, per the narrowed brief)

The 50-zone annual speedup curve, the chunk-size sweep, core-utilisation
reporting, and the 1-zone no-regression run were out of scope for this
slice. They are moot unless the phase share changes materially: the
bounded-window ceiling (1.05×) already sits under the issue's kill bar
(1.3×) with no dependence on window length.

## Refs

Refs #4697. No physics-default, tolerance, baseline, or gate changes;
no Rayon added anywhere; BatchOracle nesting rules untouched.
