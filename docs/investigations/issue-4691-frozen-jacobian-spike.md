# Issue #4691: Frozen Jacobian / factorisation reuse in the BDF Newton loop — spike

**Issue:** [#4691](https://github.com/anchapin/fluxion/issues/4691)
**Date:** 2026-11-09
**Investigator:** flag-gated prototype + measurement spike (no production default change)
**Branch:** `worker/4691-frozen-jacobian` (cut from `origin/develop` @ `017b8bf8`)
**Status:** ✅ complete — prototype + harness by the worker (two runs; first
infra-killed at step 37/60, continuation hit the 60/60 cap after landing the
full diff); measurements executed and this report completed by the collecting
supervisor on 2026-11-09, plus supervisor clippy fixes in the LU solve.

> **Verdict (headline): ✅ WIN on the harness workloads.** On the high-mass
> workload the frozen Jacobian cuts Newton-solve wall clock by **−68% at
> N=4** (6,202.8 ms → 2,004.9 ms) and −71% at N=16, with **zero rejected
> steps** at every N and a maximum hourly zone-temperature drift of
> **1.2e-6 °C** vs the baseline arm — far inside the solver's own
> convergence tolerance. Jacobian assemblies/factorisations fall 53–64%.
> The prototype stays **flag-gated and default-OFF**; adopting it as a
> production default is a separate decision for Alex.

## 1. What was built

All in `src/physics/bdf_engine.rs` unless noted:

- `counters` module — always-live relaxed-atomic counters (Jacobian
  assemblies, factorisations, Newton iterations, Jacobian reuses, step
  rejections, solve calls, forced-refresh retries) with `snapshot()` /
  `reset()`. Pure instrumentation; no behaviour depends on them.
- `freeze` module — `FrozenJacobianPolicy::from_env()` reading
  `FLX_BDF_FREEZE_JACOBIAN` (default OFF, via `util::env_bool::env_bool`)
  and `FLX_BDF_FREEZE_MAX_REUSE` (default 8).
- `LuFactors` — LU factorisation with partial pivoting (factorise once,
  O(n²) triangular solves per reuse) and `NewtonJacobianCache` carrying
  the factors across Newton iterations *and* consecutive BDF steps.
- Refresh policy, exactly as the issue specifies: reuse while the
  residual decreases monotonically; refresh on stall (residual not
  strictly decreasing, NaN included), on step rejection, or after N
  reuses. `BdfTimeStepper` invalidates the cache on rejection and
  retries a failed frozen solve **once** with a freshly assembled
  Jacobian before rejecting the step.
- Default path is unchanged: `NewtonRaphsonSolver::solve` delegates to
  `solve_with_cache` with a throwaway cache and the flag off, which
  takes the legacy branch (one assembly + one elimination per
  iteration). No tolerance, convergence criterion, baseline, or gate
  was changed.

Measurement harness: `tests/all_tests/bdf_frozen_jacobian_spike.rs`
(registered in `tests/all_tests/main.rs`). It drives the **shipped**
Newton solver + cache through a harness-assembled BDF-2 residual for
two ASHRAE-140-style annual workloads at hourly steps (8,760 steps):
Case 600 lightweight (9 states) and Case 900 high-mass (33 states),
using the repo's own construction assemblies. Optional
`FLX4691_DUMP=<path>` writes the hourly zone-temperature trajectory
per arm for exact drift measurement.

## 2. Why the harness does not drive the full engine (side finding)

The harness deliberately does **not** use `BdfTimeStepper::step`:
the shipped `BdfResidual` wrapper overwrites the `DaeSystem` residual
with `sum(alpha_j) * y / (dt*beta)`, i.e. the DAE right-hand side
never reaches the Newton solve through that path (consistent with
#4690's finding that the Newton engine is not on the production
annual path). The numbers below are therefore the Newton-solve
component on realistic state sizes (9 / 33 states), **not** a
whole-engine annual wall clock. This caveat bounds the claim: the
win is proven for the BDF Newton loop wherever it runs; what share
of a production annual run that loop is remains #4688-territory.

## 3. Measurements (supervisor-run, 2026-11-09, debug test build)

Command per arm:
`cargo test -p fluxion --test all_tests bdf_frozen_jacobian_spike -- --nocapture --test-threads=1`
with the arm's env vars; baseline arm has no `FLX_*` vars set.

### Case 900 — high mass (33 states, 8,760 hourly steps)

| Arm | Wall (ms) | Δ wall | Assemblies | Factorisations | Newton iters | Reuses | Rejected steps |
|---|---|---|---|---|---|---|---|
| Baseline | 6,202.8 | — | 9,183 | 9,183 | 17,943 | 0 | 0 |
| Freeze N=4 | 2,004.9 | −67.7% | 4,274 | 4,274 | 18,709 | 5,675 | 0 |
| Freeze N=8 | 1,990.7 | −67.9% | 3,562 | 3,562 | 19,474 | 7,152 | 0 |
| Freeze N=16 | 1,770.6 | −71.5% | 3,347 | 3,347 | 21,254 | 9,147 | 0 |

### Case 600 — low mass (9 states, 8,760 hourly steps)

| Arm | Wall (ms) | Δ wall | Assemblies | Factorisations | Newton iters | Reuses | Rejected steps |
|---|---|---|---|---|---|---|---|
| Baseline | 302.8 | — | 9,067 | 9,067 | 17,827 | 0 | 0 |
| Freeze N=4 | 143.4 | −52.6% | 3,886 | 3,886 | 18,590 | 5,944 | 0 |
| Freeze N=8 | 139.1 | −54.1% | 3,229 | 3,229 | 19,303 | 7,314 | 0 |
| Freeze N=16 | 131.7 | −56.5% | 2,919 | 2,919 | 20,944 | 9,265 | 0 |

Notes:

- Baseline assemblies ≈ one per *converging* iteration short of one
  per solve (the final, converged iteration assembles nothing): the
  Jacobian is re-assembled and re-eliminated on essentially every
  Newton iteration that takes a step — the cost the issue suspected.
- Freezing trades iterations for assemblies: iterations rise +4.3%
  (N=4) to +18.5% (N=16) on the high-mass case while assemblies fall
  53–64%. N=4 captures nearly all the wall-clock win with the least
  iteration growth; N=16's extra wall gain is small and its
  iteration inflation is the trend to watch at larger state sizes.
- Post-completion supervisor re-runs of the N=8 arm after clippy
  fixes reproduced the pre-fix counters and zone sums **exactly**
  (assemblies 3,229 / 3,562; zone sums bit-identical), confirming
  the fixes were numerics-neutral.

## 4. Output drift vs baseline

Hourly zone-temperature trajectories (`FLX4691_DUMP` CSVs) compared
cell-by-cell against the baseline arm, both workloads:

| Arm | Max abs zone-temp drift |
|---|---|
| Freeze N=4 | 1.07e-6 °C |
| Freeze N=8 | 1.22e-6 °C |
| Freeze N=16 | 0.99e-6 °C |

Final residuals per arm remain at the baseline's order (1e-10 –
1e-11). A converged frozen solve is the same solve to within the
solver's own convergence behaviour; the observed ~1e-6 °C drift is
the expected consequence of taking a slightly different iteration
path to the same tolerance, **not** a changed acceptance criterion.

## 5. Win / kill criteria (from the issue)

- **Win:** ≥15% wall-clock reduction on the high-mass case → **met**
  (−67.7% at N=4, the most conservative arm).
- **Win:** drift within the solver's own convergence tolerance →
  **met** (max 1.22e-6 °C hourly zone temperature).
- **Win:** rejected steps up no more than +10% → **met** (0 at
  every arm, baseline included; forced-refresh retries 0).
- **Kill:** factorisation <5% of step cost → **not triggered**;
  assembly+factorisation together are the dominant Newton cost on
  these workloads (freezing both removes ~⅔ of wall clock). The
  counters count assemblies and factorisations 1:1 in both paths,
  so their individual shares are not separable from this spike —
  stated plainly rather than guessed.

## 6. Verification

- `cargo test -p fluxion --lib physics::` — 630 passed, 0 failed
  (supervisor run).
- Harness test passes in baseline and all freeze arms (supervisor
  runs above).
- `cargo fmt --check` — clean (after supervisor `cargo fmt`).
- `cargo clippy -p fluxion --lib` — no warnings (4 warnings in the
  new LU code fixed by the supervisor; re-run arms reproduced
  identical counters/zone sums).
- Test-inventory procedure run (new harness test): inventory
  regenerated and ratchets bumped per `scripts/check_test_inventory_drift.py`.

## 7. Recommendation

Land the prototype as-is (flag-gated, default OFF). If Alex wants a
production trial, N=4 is the conservative starting budget: nearly
all of the measured win, least iteration growth, and the refresh
policy already handles stall/rejection. Any default-on flip is a
physics call for a separate issue.
