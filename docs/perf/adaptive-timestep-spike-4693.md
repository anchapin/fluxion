# Error-controlled adaptive timestepping (fluxion #4693, spike)

> Summary: flag-gated error-controlled step-size controller prototype in
> `src/sim/adaptive_timestep.rs`, sharing the BDF step-size law via a
> `pub(crate)` extraction in `src/physics/bdf_engine.rs` (bit-identical
> delegation; 41 controller tests + 17 BDF tests pass). Measured verdict:
> KILL on the path that actually drives annual runs — at tolerance 0.05
> the controller saturates at the 1 h cap for the whole year on Cases
> 600 and 900, so "adaptive" is bit-identical to fixed-hourly stepping
> and misses the issue's win bar against the 6-minute reference.

## 1. What was built

- `error_controlled_step_size` extracted from the private
  `AdaptiveStepController::compute_step_size` (src/physics/bdf_engine.rs)
  into a crate-level `pub(crate)` helper; the method now delegates to it
  with unchanged arithmetic.
- `ErrorControlledStepper` in src/sim/adaptive_timestep.rs: drives the
  shared law with min/max clamps (60 s / 3600 s), forced landings on hour
  boundaries, accepted/rejected step accounting, a step-size histogram,
  and min-step fraction tracking. Opt-in only (env flag, default off);
  no default behaviour changes anywhere.
- Measurement harness: `examples/adaptive_timestep_spike.rs` (bin
  registered in examples/Cargo.toml), modes `ref360` (fixed 6-minute),
  `ref3600` (fixed hourly), `adaptive`; Golden CO TMY3 EPW weather;
  full-year runs of ASHRAE 140 Cases 600 and 900.

## 2. Measurements (supervisor-run, release build, base 711cbbf7 + this diff)

| Case | Mode | Steps | Heating MWh | Cooling MWh | Peak heat kW | Peak cool kW | Wall s |
|---|---|---|---|---|---|---|---|
| 900 | ref360 (6-min) | 87,600 | 1.413677 | 1.772947 | 2.287221 | 1.569032 | 0.274 |
| 900 | ref3600 (hourly) | 8,760 | 1.404894 | 1.765974 | 2.271614 | 1.551569 | 0.034 |
| 900 | adaptive tol 0.05 | 8,760 accepted, 0 rejected | 1.404894 | 1.765974 | 2.271614 | 1.551569 | 0.034 |
| 600 | ref360 (6-min) | 87,600 | 6.044604 | 3.468576 | 5.628640 | 3.860233 | 1.191 |
| 600 | ref3600 (hourly) | 8,760 | 5.899111 | 3.299931 | 5.549076 | 3.708284 | 0.124 |
| 600 | adaptive tol 0.05 | 8,760 accepted, 0 rejected | 5.899111 | 3.299931 | 5.549076 | 3.708284 | 0.126 |

Adaptive step histogram (both cases): all 8,760 steps at 3,600,000 ms;
min-step fraction 0.0; hour landings 8,760. The adaptive hourly zone
temperature series is bit-identical to the ref3600 series on both cases.

Error vs the 6-minute reference:

| Case | Heating Δ | Cooling Δ | Temp RMSE | Temp max abs |
|---|---|---|---|---|
| 900 | −0.62% | −0.39% | 0.0547 °C | 0.3401 °C |
| 600 | −2.41% | −4.86% | 0.4493 °C | 3.1364 °C |

## 3. Why the controller never engages

The harness drives the legacy `ThermalModel` annual path, not the BDF
DAE engine — the BDF engine's own error estimate does not drive annual
runs on this path. The harness therefore feeds the controller a
first-order extrapolation proxy error (|change in the hourly zone-temp
increment|). At tolerance 0.05 that proxy stays below tolerance
essentially always, so the shared law doubles dt to the 3600 s cap on
the first step and holds it all year. The step-count reduction (10×)
is real but degenerate: it is exactly fixed-hourly stepping, with
fixed-hourly accuracy.

## 4. Verdict against the issue's win/kill criteria

Win required: ≥2× fewer accepted steps than fixed 6-minute AND annual
energy within 0.5% AND hourly temp RMSE within 0.1 °C of the 6-minute
reference. Step count passes (10×); energy fails on both cases
(900 heating −0.62%; 600 heating −2.41%, cooling −4.86%); temp RMSE
fails for Case 600 (0.4493 °C). **Verdict: KILL** for error-controlled
stepping as configured on the legacy annual path. The prototype and
controller stay behind the default-off flag as scaffolding: a future
attempt would need either the BDF/DAE path driving annual runs (so a
true local error estimate exists) or a much tighter tolerance with a
demonstrated accuracy case — neither is claimed here.

## 5. Verification

- `cargo test -p fluxion --lib sim::adaptive_timestep` — 41 passed
  (supervisor fixed one worker test whose rejection-count assertion
  assumed an absolute count; it now asserts the increment).
- `cargo test -p fluxion --lib physics::bdf_engine` — 17 passed; the
  extraction is arithmetic-identical, so the BDF path is unchanged.
- Test inventory regenerated; drift gate re-run (see PR).
