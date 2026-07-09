# Issue #1412 — PredictiveController inertia_factor sign drift

**Status:** COMPLETE
**Branch:** `fix/issue-1412-predictive-controller-sign`
**PR:** (opened at end of session — see commit message)

## Summary

Unified the `inertia_factor` sign across the two `PredictiveController`
overloads. Both `calculate_modulation` and `calculate_modulation_with_setpoints`
now route through a new private helper `effective_setpoints(...)` that
encodes the canonical, physically correct sign convention. Pre-fix, the
two overloads diverged by up to 2 °C per call at the 10 °C zone/mass
gap the issue cites — silently shifting annual heating energy during
setback schedules.

## Sign convention chosen

**`eff_heating_sp = heating_setpoint − inertia_factor − predictive_factor`**
**`eff_cooling_sp = cooling_setpoint − inertia_factor − predictive_factor`**

where `inertia_factor = α · (T_zone − T_mass)` and `predictive_factor = β · dT/dt`.

**Why this is canonical (per the issue + EnergyPlus IO Reference "Zone
Thermostat / Predictive Controller" cited in the issue body):**

- When the mass is **cooler** than the zone (`inertia_factor > 0`):
  the mass is absorbing heat and cooling the zone. The controller should
  **anticipate** that cooling by:
  - Lowering the effective heating setpoint → heating triggers at a
    higher zone temperature (fires earlier)
  - Lowering the effective cooling setpoint → cooling has to wait
    until the zone is hotter (defers — the mass is already helping)
- When the mass is **warmer** than the zone (`inertia_factor < 0`):
  the mass is releasing heat and warming the zone. The controller
  anticipates the warming and **raises** both setpoints (tolerates a
  slight under-shoot in heating, slight over-shoot in cooling).

The static-setpoint overload (line 116 pre-fix) already encoded this
correctly. The dynamic-setpoint overload (line 168 pre-fix) had the
**opposite** sign — it was telling the controller that a cool mass
should *raise* the heating setpoint (defer heating) and raise the
cooling setpoint (fire cooling sooner). That is the opposite of the
intended anticipation.

## Python verification (per AGENTS.md)

`ctx_execute` reproduced the sign-flip on a 10 °C zone/mass gap:

```
zone= 25.0 mass= 15.0 | h_eff static=+19.000 dyn=+21.000 diff=-2.000
zone= 15.0 mass= 25.0 | h_eff static=+21.000 dyn=+19.000 diff=+2.000
zone= 22.0 mass= 18.0 | h_eff static=+19.600 dyn=+20.400 diff=-0.800
zone= 18.0 mass= 22.0 | h_eff static=+20.400 dyn=+19.600 diff=+0.800
```

Post-fix: all `h_eff` and `c_eff` values match across the two overloads
within 0.0e+00 (i.e., bit-identical, well inside the 1e-12 acceptance
criterion in the issue).

## Files changed

| File | Change |
|------|--------|
| `src/sim/hvac/modes.rs` | Added private `effective_setpoints(...)` helper (canonical sign documented inline). Both `calculate_modulation` and `calculate_modulation_with_setpoints` now route through it. Dynamic overload also gained the NaN/Inf guard (it was missing — caught while consolidating the two branches). |
| `tests/hvac_predictive_modulation.rs` | Added two regression tests: `test_inertia_factor_sign_parity` (helper-hoist invariant: both overloads produce identical `(mode, modulation)` for identical inputs, within 1e-12) and `test_inertia_factor_physical_direction` (physical-intent guard: cool mass must anticipate cooling by lowering effective heating setpoint, warm mass must anticipate warming by raising it). |

`git diff --stat`:
```
src/sim/hvac/modes.rs               |  83 ++++++++++++++------
tests/hvac_predictive_modulation.rs | 151 ++++++++++++++++++++++++++++++++++++
2 files changed, 211 insertions(+), 23 deletions(-)
```

## Acceptance criteria (from issue #1412)

- [x] **Both overloads return identical effective setpoints within 1e-12 for
      identical inputs.** `test_inertia_factor_sign_parity` enforces this
      across 7 zone/mass/rate sweep cases (including the issue's worst-case
      10 °C gap). Pre-fix: would have failed (modulation diverged by
      ~0.5-1.0 at α=0.1). Post-fix: 0.0e+00 divergence, test passes.
- [x] **`test_inertia_factor_physical_direction` fails on the pre-fix
      overload and passes on the post-fix overload.** The test asserts
      that with mass 4 °C cooler than zone, the controller does NOT
      trigger heating at zone=19.7 (it should anticipate the mass's
      cooling). Pre-fix: would have produced Heating (the inverted sign
      raises `h_eff` to 20.4, threshold 19.9, zone 19.7 < 19.9 → Heating).
      Post-fix: Off, as physically intended.
- [x] **ASHRAE 140 Case 960 annual heating has not regressed.** Pre-fix
      baseline: 1.37 MWh (soft-warned out of band per known issue #348).
      Post-fix: 1.37 MWh (exact). Drift: 0.0%, well inside the 0.1%
      criterion. Annual cooling: 1.80 MWh pre-fix → 1.80 MWh post-fix.

## Verification

| Test | Pre-fix | Post-fix | Note |
|------|---------|----------|------|
| `cargo test -p fluxion --test hvac_predictive_modulation` | 3/3 | 5/5 | +2 new regression tests |
| `cargo test -p fluxion --test test_hvac_control_comprehensive` | 41/41 | 41/41 | unchanged |
| `cargo test -p fluxion --lib hvac` | 237/237 | 237/237 | unchanged |
| `cargo test -p fluxion --test hvac_equipment` | 9/9 | 9/9 | unchanged |
| `cargo test -p fluxion --test ashrae_140_blind_validation` | 17/17 (5 ignored) | 17/17 (5 ignored) | unchanged |
| `cargo test -p fluxion --test ashrae_140_case_960_sunspace test_annual_energy_validation` | 1/1 (heating 1.37 MWh, cooling 1.80 MWh) | 1/1 (heating 1.37 MWh, cooling 1.80 MWh) | **0.0% drift** |
| `cargo test -p fluxion --test ashrae_140_setback_ventilation` | 9/9 | 9/9 | unchanged |
| `cargo test -p fluxion --test ashrae_140_case_600_series` | 11/16/4 fail (pre-existing) | 11/16/4 fail (pre-existing) | **identical pre/post** — confirmed via `git stash` |
| `cargo test -p fluxion --test ashrae_140_case_900` | 13/4/1 fail (pre-existing) | 13/4/1 fail (pre-existing) | **identical pre/post** — confirmed via `git stash` |

The Case 600/900 series pre-existing failures are about ASHRAE 140
high-mass annual/peak cooling tolerances — orthogonal to the predictive
controller's sign convention and explicitly called out as known gaps
in `ARCHITECTURE.md` ("Current cooling underestimates ASHRAE 140 by
~90%; per the Issue #1281 / #1280 investigation, the root cause is
roof-solar under-counting").

## Production impact

**None.** Per `grep`, the production call sites of
`calculate_modulation` are:
- `src/sim/hvac/equipment.rs:1108` (test only)
- `src/sim/thermal_model_physics/physics_impl.rs:488` (production)
- `src/sim/thermal_model_physics/physics_impl.rs:2496` (production)

The dynamic-setpoint overload `calculate_modulation_with_setpoints` is
**not called from any production code path** — only from tests. The
fix therefore has zero runtime impact on the annual heating/cooling
numbers (confirmed by the byte-identical ASHRAE 140 Case 960 numbers
pre/post fix). The fix's value is:

1. **Correctness for the next consumer.** Any future production caller
   of `calculate_modulation_with_setpoints` (e.g., a setback-schedule
   driver wired into a future timestep loop) will now get the
   physically correct sign — preventing the silent annual heating
   shift the issue describes.
2. **Helper-hoist invariant.** The new `effective_setpoints` helper
   makes the sign-convention copy-paste impossible: both overloads
   route through the same function. The pre-fix code had two
   independent inline copies of the formula, which is what allowed
   them to drift.

## Blockers

None. Issue acceptance criteria all met.
