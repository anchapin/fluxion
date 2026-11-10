# Warm-start predictor for the BDF Newton solve (fluxion #4692, spike)
> Summary: flag-gated linear warm-start predictor for the BDF Newton
> initial guess, prototype complete and unit-tested in
> `src/physics/bdf_engine.rs`. The flag defaults OFF and the flag-off
> path is bit-identical (unit test). The audit found the legacy guess
> is already a BDF-coefficient history predictor. The iteration-count
> measurement leg did NOT complete in the worker run, so the issue's
> win/kill criteria are UNRESOLVED in this report — see §4.

## 1. Audit — the current initial-guess policy

Before this change, the Newton initial guess in
`BdfTimeStepper::step` came from `predict()` (`src/physics/bdf_engine.rs`):
with fewer than two history states it copies the last accepted state;
otherwise it extrapolates the history with the current BDF order's
coefficients. So the engine does **not** start Newton from the last
state unchanged — it already uses a polynomial history predictor. The
issue's proposed linear extrapolation `x_prev + (x_prev - x_prev2)` is
therefore an *alternative* guess (cheaper, order-independent), not the
introduction of prediction where none existed.

## 2. Prototype (flag-gated, default OFF)

- `FLUXION_BDF_WARM_START_PREDICTOR` enables the linear predictor.
  The flag is read **once at construction** (`BdfTimeStepper::new` /
  `BdfDriver::new`), never per step — per-step `getenv` reads were the
  #4688 profile's dominant avoidable cost.
- When enabled, the guess is `warm_start_predict(x_prev, x_prev2)`,
  clamped to the physical bounds the DAE system optionally declares
  via the new defaulted `DaeSystem::state_bounds()` (default `None` =
  unbounded; existing implementors unaffected). A non-finite
  extrapolation falls back to `x_prev` element-wise.
- Fallbacks keep the legacy `predict()` guess on the first two steps
  and on the step after any rejected/failed step
  (`warm_start_guess()` returns `None` in those cases); a failed
  Newton solve or a non-converged step marks `last_step_rejected`.
- With the flag off, the code path is exactly the pre-change path.

## 3. Verification (run by the supervisor at collection)

```sh
cargo test -p fluxion --lib physics::bdf_engine
# test result: ok. 17 passed; 0 failed — including
# issue_4692_flag_off_is_bit_identical (same trajectory + iteration
# counts with the predictor compiled in but disabled),
# issue_4692_warm_start_predict_math, _clamps_to_declared_bounds,
# _first_two_steps_fallback, _post_rejection_fallback,
# _failed_step_marks_rejection, _flag_defaults_off.
cargo fmt --check   # clean after supervisor fmt pass
```

Base commit: `017b8bf8` (develop, merge of PR #4713). Machine: the
supervisor VM (shared vCPU, debug test profile) — no wall-clock
claims are made from it.

## 4. Measurement — NOT completed (win/kill unresolved)

The worker's Case 600 measurement harness (an out-of-tree release
binary under `/tmp`, never part of the diff) was still compiling when
the worker reached its step cap, so no baseline-vs-predictor
iteration table exists yet. A supervisor fixed-step DAE harness was
also tried and discarded: under the default stepper config its state
trajectory collapsed to ~0 within ten steps, so its iteration counts
describe a degenerate workload and are not reported as evidence.

What remains to measure, per the issue: Newton iterations/year and
rejected steps on Case 600 and a high-mass sub-hourly case, flag off
vs on, plus wall clock and output drift. Win = ≥25% fewer Newton
iterations and a measurable wall-clock reduction with rejections not
increased; kill = iterations drop but wall clock doesn't, or
rejections rise at control events. Until that table exists, the
predictor must stay default OFF (it is), and no default change is
proposed by this spike.

## 5. Verdict

Prototype: **landed, default OFF**. Win/kill: **unresolved** —
measurement leg outstanding (follow-up run with the release harness
pre-built). Nothing in this change alters tolerances, convergence
criteria, baselines, or physics defaults.
