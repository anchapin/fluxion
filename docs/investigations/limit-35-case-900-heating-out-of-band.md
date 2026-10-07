# §LIMIT-05 companion — Case 900 annual heating out of band: reproduction, isolation, and a falsified mass-node hypothesis

**Status:** investigation (PREPARE-ONLY, DO-NOT-MERGE). No engine behavior changes by
default; two `FLUXION_MASS_SUBSTEPS` measurement hooks are included, both
bit-identical to develop when unset. **Do not merge before Alex's physics
review.**

**Scope note:** this is the Case 900 (high-mass) annual **heating**
over-prediction. It is pre-existing on develop `7d144c80`, unrelated to the
§LIMIT-33 fix (Case 900 is 9R4C-path ideal-loads-insensitive; variants of the
LIMIT-33 fix leave Case 900 bit-identical), and distinct from the §LIMIT-05
*peak-cooling under-prediction* symptom, though both live in the high-mass
path.

## 1. Reproduction (develop `7d144c80`, default build, Denver-Stapleton TMY EPW)

Harness: `ASHRAE140Validator::simulate_case_with_diagnostics` on
`ASHRAE140Case::Case900` (the production validator path; per
`step_dispatcher.rs` the high-mass spec is promoted selector-independently to
the 9R4C network, so `new_with_selector(NineRFourC)` returns identical
numbers to `new()`).

| Quantity | Value | Reference band |
|---|---|---|
| Annual heating | **3,477.88 kWh** | [1,170, 2,040] kWh → **+70.6% over the upper bound** |
| Annual cooling | 923.56 kWh | [2,130, 3,670] kWh → under |
| Peak heating | 3.304 kW | [1.80, 2.40] kW |

Standalone `ThermalModel` harnesses agree (3,249.6 kWh over 8,760 h with
14-day warmup via `DenverTmyWeather`); the ASHRAE140_RESULTS.md snapshot's
5,052.83 kWh is a stale/legacy build, not current develop.

Temperature trace: zone air never drops below the heating setpoint (min
exactly 20.00 °C, mean 23.32 °C) while heating energy is positive in
**3,722 of 8,760 hours** — the metered heating is the full net load of a
zone held at setpoint, and it is the *magnitude* of that net load that is
wrong.

Monthly split: Jan 934 / Feb 588 / Mar 187 / Apr 7 / Oct 85 / Nov 521 /
Dec 928 kWh — a physically plausible seasonal shape; the excess is spread
over the whole heating season, not a spike artifact.

Hourly split: nearly uniform across the day (63 kWh at 15:00 to 207 kWh at
06:00, annual totals per clock hour) — midday solar offsets only ~50–70% of
the noon-hour load, which for a high-mass single-zone building with large
south glazing is the signature of either under-captured solar gain or
over-counted envelope loss.

## 2. Decisive experiment: whole-step timestep refinement

Running the same annual loop with the physics step divided into N sub-steps
per hour (same hourly weather drivers, `step_physics(dt=3600/N)`):

| N | dt | Annual heating (kWh) | Annual cooling (kWh) |
|---|---|---|---|
| 1 | 3600 s | 3,249.6 | 1,602.4 |
| 2 | 1800 s | 4,717.9 | 555.5 |
| 4 | 900 s | **1,802.7** | 192.2 |
| 6 | 600 s | **1,305.7** | 126.9 |

At dt = 600 s the heating energy lands **inside the published band
[1,170, 2,040] kWh**, and the sequence converges toward it (the N=2
overshoot is the kind of non-monotone step seen when a stiff node crosses an
integration stability boundary). Refining the *whole* discretized hour
reproduces the reference behavior. This localizes the error to the
**hour-scale discretization of the coupled air-node/HVAC/gain system**, not
to any slow (building-mass-scale) drift.

## 3. Falsification: sub-stepping the mass node alone does NOT converge

§LIMIT-29 hypothesis 3 attributes the high-mass errors to the mass-node
integrator (dt/τ ≈ 0.81 at Case 900's τ ≈ 1.23 h on the 5R1C air
trajectory). Two targeted tests of that hypothesis, both now in the tree as
env-gated hooks:

1. **Lumped 9R4C mass block** (`step_9r4c.rs` mass-temperature backward
   Euler; hook applied temporarily, `FLUXION_MASS_SUBSTEPS=12`): validator
   Case 900 heating unchanged
   to 5 decimals (3,477.88 kWh). That block's output demonstrably does not
   drive the conditioned-air trajectory (consistent with its own comment:
   the multi-node network, not the lumped mass, sets the high-mass air
   temperature).
2. **Multi-node solver sub-step override** (`multi_node_solver/mod.rs`
   `step_with_gains`, same env var; the adaptive path there already
   sub-steps to 12 when dt > 4τ but τ for the 900 envelope exceeds dt/4, so
   it normally runs 1): N = 2/4/6/12 gives 3,484.2 / 3,487.4 / 3,488.5 /
   3,489.6 kWh — a +12 kWh drift **away** from the band.

Conclusion: **the mass-node backward Euler is not the root cause** of the
heating over-prediction. The convergence seen in §2 must come from refining
the air-node balance and the HVAC setpoint-feedback/metering loop at
sub-hour resolution (the same loop §LIMIT-33's state feedback closes at
hour resolution). The τ-based hypotheses in §LIMIT-29 remain valid for the
peak-cooling under-prediction (§LIMIT-05) but are not what pushes annual
heating out of band.

## 4. Proposed fix direction (not implemented — review first)

Refine the **conditioned-hour inner loop**: with the zone held at setpoint,
solve the air-node balance and meter the HVAC energy with sub-hour steps
(or in closed form for the piecewise-constant drivers), leaving the mass
node hour-stepped. Success criterion: Case 900 heating moves monotonically
toward [1,170, 2,040] kWh as the inner refinement increases, with Case 600
(NAPI/WD600 5,590.48 kWh in [4,314, 5,836]) and the recorded-value baselines
re-checked per the re-record discipline — never tuned to pass.

## 5. Measurement hooks in this branch

- `FLUXION_MASS_SUBSTEPS` (multi-node solver): forces N sub-steps in
  `step_with_gains`; unset/0 → adaptive logic bit-identical to develop.
- The same variable name in the 9R4C **lumped** mass block is intentionally
  not kept (zero observable effect; removed to avoid dead knobs).

Both hooks exist to make §3's numbers reproducible by anyone; neither is
wired into any gate.
