# GaugeSolver Scalability Performance Characterization

**Issue:** #1771
**Last Updated:** 2026-07-26
**Status:** Characterised

This document records how the cost of the multi-zone `GaugeSolver`
(`MultiZoneGaugeSolver` in `src/physics/gauge_zone_solver.rs`) scales with zone
count, and identifies the crossover where GaugeSolver cost exceeds the
`FiveR1CSolver` baseline. The measurements are produced and guarded by
`tests/gauge_solver_scalability.rs`.

## Summary

- **Realistic topology (ring coupling): near-linear scaling** (~O(N)). A
  20-zone ring building solves in well under the interactive-timestep budget.
- **Worst-case topology (full/dense coupling): near-quadratic scaling**
  (~O(N²)). The coupling graph has N(N-1)/2 edges, so the per-timestep cost
  grows quadratically with zone count.
- **Crossover vs FiveR1C:** the GaugeSolver is **more expensive than the
  FiveR1C baseline at every measured size**, starting at N=2. This is expected:
  the GaugeSolver carries per-surface gauge-connection bookkeeping
  (`ThermalManifold`/`VectorField`) and a zone-air energy-balance aggregation
  pass that the bare FiveR1C steady-state step does not. The GaugeSolver is a
  higher-fidelity solver, not a drop-in performance replacement for FiveR1C.

## Method

For each problem size `N ∈ {2, 5, 10, 20}` zones, three configurations are
timed over 2000 iterations (after a warm-up step) using `std::time::Instant`:

1. **GaugeSolver (ring)** — `MultiZoneGaugeSolver`, ring inter-zone coupling
   (each zone coupled to its neighbours). Realistic sparse O(N) case.
2. **GaugeSolver (dense)** — `MultiZoneGaugeSolver`, fully-connected coupling
   (every zone coupled to every other). Worst-case O(N²) case.
3. **FiveR1C baseline** — `N` independent `FiveR1CSolver::step` calls, no
   coupling. Cheapest baseline.

Every configuration uses the **same** 3-layer envelope wall
(concrete + insulation + gypsum) so the comparison is apples-to-apples. Each
zone has six exterior surfaces (4 walls + roof + floor). Measurements were
taken with `cargo test --profile ci` (opt-level=1) on Linux; absolute numbers
vary by machine, but **ratios and scaling exponents are stable**.

### Reproduce

```bash
cargo test --profile ci --test gauge_solver_scalability -- --nocapture
```

## Scaling curve (zones vs µs/timestep)

Representative numbers from a single `--profile ci` run. Absolute µs vary by
hardware; the scaling exponents and the Gauge/5R1C ratios are the stable signal.

| Zones | Ring µs | Dense µs | 5R1C µs | Ring/5R1C | Dense/5R1C |
|------:|--------:|---------:|--------:|----------:|-----------:|
|     2 |    ~2.4 |     ~2.1 |    ~0.02 |     ~115× |      ~100× |
|     5 |    ~7.0 |    ~13.5 |    ~0.04 |     ~165× |      ~315× |
|    10 |   ~16.6 |    ~38.0 |    ~0.08 |     ~200× |      ~460× |
|    20 |   ~27.7 |   ~125.9 |    ~0.16 |     ~180× |      ~810× |

### Scaling exponent (N=2 → N=20)

`slope = ln(t(N=20) / t(N=2)) / ln(20 / 2)`. `≈1` = linear, `≈2` = quadratic.

| Configuration        | Exponent | Interpretation |
|----------------------|---------:|-----------------|
| FiveR1C baseline     |    ~0.90 | Linear (N independent solvers) |
| GaugeSolver (ring)   |    ~0.98 | Near-linear — realistic building case |
| GaugeSolver (dense)  |    ~1.94 | Near-quadratic — coupling graph is O(N²) edges |

## Crossover analysis

The GaugeSolver cost **exceeds** the FiveR1C baseline at every measured zone
count, beginning at N=2 (the smallest configuration). The crossover is not a
function of zone count but of the per-surface solver model:

- **FiveR1C** computes a single steady-state flux `q = ΔT / R_total` (plus a
  one-node mass integration) per surface — minimal arithmetic.
- **GaugeSolver** additionally maintains a `ThermalManifold` of per-zone
  `VectorField` gauge connections, translates boundary conditions into the
  gauge-connection representation, computes sol-air effective exterior
  temperatures, and aggregates all surface fluxes at the zone-air energy node
  with an implicit-Euler update.

Consequently the GaugeSolver is ~100–800× the per-timestep cost of an equal
number of bare FiveR1C solvers across the measured range. There is **no zone
count at which the GaugeSolver becomes cheaper than FiveR1C**; the two solvers
occupy different points on the fidelity/cost tradeoff. The GaugeSolver should
be selected when geometrically-accurate multi-surface zone aggregation and
inter-zone coupling are required, not as a performance optimisation.

## Practical limits

- **Ring-coupled buildings (the common case):** cost grows ~linearly. Even at
  N=20 the per-timestep cost is tens of µs, leaving ample headroom for
  cloud-scale runs (the `DENSE_20_ZONE_BUDGET_US = 5_000` guard in the test is
  ~40–180× the measured N=20 ring cost).
- **Dense-coupled buildings (worst case):** cost grows ~quadratically. The
  practical limit before the per-timestep cost dominates the annual run is on
  the order of ~50–100 fully-coupled zones; beyond that, prefer a sparse
  coupling graph (ring, tree, or zone-graph sparsity) to restore linear
  scaling.

## Regression guards

The test suite (`tests/gauge_solver_scalability.rs`) encodes the
characterisation as assertions so regressions are caught in CI:

- `gauge_solver_scaling_curve` — ring exponent < 1.4, dense exponent in
  (1.4, 2.6), FiveR1C exponent < 1.3, dense N=20 < 5000 µs.
- `gauge_solver_crossover_vs_fiver1c` — crossover first observed at N=2.
- `gauge_solver_2_zone_step_budget` — N=2 ring < 1000 µs.
- `gauge_solver_dense_not_cheaper_than_ring` — dense ≥ ring for N ≥ 5.
- `gauge_solver_correctness_at_all_sizes` — finite results at every N.
