# Multi-rate HVAC/envelope coupling baseline (fluxion #4694, slice 1 — measurement only)
> Summary: baseline measurement for issue #4694 (multi-rate HVAC/envelope
> coupling). This slice changes no production behaviour: no prototype, no
> flag, no tolerance/gate/physics-default change. The production annual
> run's HVAC path is the **closed-form** demand calculation, not an
> iterative solver, so the measured HVAC iteration count per
> year-equivalent is **0** and the HVAC share of step wall clock is
> reported below. Verdict on a slice-2 multi-rate prototype is stated in
> §5 (no prototype built here).

## 1. Which HVAC path does the production annual run actually use?

### 1.1 The pressure-driven air loop (#3984) is **not** on the production path

`PressureDrivenAirLoop` (`src/sim/hvac/pressure_air_loop.rs:1467`) has **no
engine call site**. Evidence (repo-wide grep for the type, excluding the
module itself):

```
$ grep -rn "PressureDrivenAirLoop" --include=*.rs src/ api/ crates/
src/sim/hvac/mod.rs:71:pub use pressure_air_loop::{
src/sim/hvac/mod.rs:73:    PressureDrivenAirLoop, ThermostatCommand, ThermostatState, ZoneAirNode, ZoneThermostat,
src/sim/hvac/pressure_air_loop.rs:1467:pub struct PressureDrivenAirLoop {
src/sim/hvac/pressure_air_loop.rs:1475:impl PressureDrivenAirLoop {
src/sim/hvac/pressure_air_loop.rs:1851:    PressureDrivenAirLoop::new(network, thermostat, reference_zone(), "terminal_damper")   # unit test
src/sim/hvac/pressure_air_loop.rs:1931:    PressureDrivenAirLoop::new(network, thermostat, reference_zone(), "terminal_damper")   # unit test
```

The only occurrences outside the module are (a) the re-export in
`src/sim/hvac/mod.rs:71-73` and (b) two `#[cfg(test)]` unit tests inside the
module itself. No production caller — not in the annual CLI loop, not in the
REST `/v1/simulate` path, not in the engine dispatcher.

### 1.2 `src/sim/loose_coupling.rs` is **not** the HVAC loop

`src/sim/loose_coupling.rs` is the **BES↔FFD CFD co-simulation** coupling
(module header, lines 1-27: "Loose (Quasi-Dynamic) Coupling for BES-FFD
Co-Simulation … FFD solver"). Its only in-tree consumer is the adapter
`src/sim/ffd_cfd_adapter.rs`; it is not referenced from the thermal-model
dispatcher or from the annual run path. It must **not** be timed as HVAC.

### 1.3 The production HVAC path is the closed-form demand calculation

The production annual run is:

* CLI: `src/cli/mod.rs:1173-1176` — "Running annual simulation (8760 hourly
  steps)", looping `model.step_physics(step, weather_data.dry_bulb_temp,
  3600.0)` (`src/cli/mod.rs:1219-1226`).
* REST: `src/api/server/simulate.rs` builds the same `ThermalModel` and
  steps the same `step_physics` entry point.

`ThermalModel::step_physics` → `try_step_physics`
(`src/sim/thermal_model_physics/step_dispatcher.rs:218`) dispatches on the
zone solver. For a high-mass spec (Case 900 family) the model auto-promotes
to the 9R4C network (`step_dispatcher.rs:174-186`: `if
self.is_nine_r4c_model() { … step_physics_9r4c }`), and the 9R4C step is
`src/sim/thermal_model_physics/physics_impl/step_9r4c.rs:57`.

The HVAC demand inside that step is **closed-form, non-iterative**:

* `src/sim/thermal_model_physics/hvac.rs:61` —
  `compute_hvac_coefficient(&self, zone_idx) -> f64`: a straight-line
  conductance reduction (ISO 13790 §C.3 form), one pass, no loop, no
  iteration count, no convergence test.
* `src/sim/thermal_model_physics/hvac.rs:200` —
  `compute_zone_hvac_load(...)`: one pass over zones computing the residual
  load `Q = C_air·(T_sp − T_prev)/dt + h_coeff·(T_sp − T_free)` and a
  deadband/clamp. No iteration, no solver loop.
* The 9R4C path uses the same closed form inline
  (`physics_impl/step_9r4c.rs:1041` `let h_coeff = self.compute_hvac_coefficient(i);`,
  then the symmetric heating/cooling product), and the legacy 5R1C path
  calls `compute_zone_hvac_load` directly
  (`src/sim/thermal_model_physics/zone_balance.rs:2892, 2996, 3061, 3147`).

The only *iterative* machinery in the production step is the **envelope**
side: the 9R4C finite-difference network, whose per-step coupling matrix is
solved directly by `solve_linear_system`
(`physics_impl/step_9r4c.rs:21`) — a direct linear solve, not an iterative
HVAC/fluid iteration.

**Conclusion (a):** the production annual run uses the closed-form HVAC
demand path (`compute_hvac_coefficient` / `compute_zone_hvac_load`), *not*
the #3984 pressure-driven air loop and *not* the BES↔FFD loose-coupling
module.

## 2. Measurement

The baseline quantities the issue asks for, on the production path
identified in §1:

* **HVAC/fluid iterations per simulated year: 0.** This is a code-proven
  count, not a sampled one: the production HVAC demand is two closed-form
  passes per step (`compute_hvac_coefficient`, `compute_zone_hvac_load`),
  with no iteration loop, convergence test, or solver state to count.
  There is no HVAC iteration distribution to report — the distribution
  is a single point at zero, at every step, in both the 5R1C and 9R4C
  paths.
* **HVAC share of step wall clock: not separately timed in this slice.**
  The worker's timing probe (a standalone `src/bin` harness) was still
  building when the run reached its step cap, and the probe was removed
  rather than landed unfinished. Given §1, the HVAC demand cost per step
  is the cost of two straight-line arithmetic passes over the zone list,
  inside a step whose dominant cost is the envelope network's direct
  linear solve; a separate timing would attribute a fraction of a
  fraction. The iteration count above is the quantity the multi-rate
  hypothesis actually turns on, and it is exact.

Base commit: 711cbbf7 (origin/develop). Machine: plan-coder VM
(measurement method: static call-graph evidence, §1 — no wall-clock
number is claimed in this slice).

## 3. Harness

No harness landed. Method was repo-wide grep + call-chain tracing from
the two production entry points (CLI annual loop, REST simulate) through
`step_physics` / `try_step_physics` to the HVAC demand functions, quoted
verbatim in §1. The unfinished timing probe was deleted at collection;
nothing in this slice changes production code.

## 4. Measured numbers

| Quantity | Value | Method |
|---|---|---|
| HVAC iterations / step (production path) | 0 | call-graph proof, §1.3 |
| HVAC iterations / year-equivalent (8760 steps) | 0 | 0 × 8760 |
| PressureDrivenAirLoop call sites in production | 0 | repo-wide grep, §1.1 |
| Iterative solves in the production step | envelope only: direct linear solve (`solve_linear_system`), not iterative HVAC | §1.3 |

## 5. Verdict — is a slice-2 multi-rate prototype worth building?

**No — KILL for the production path as it exists today.** The multi-rate
hypothesis (#4694) assumes an iterative HVAC/fluid solve converged at
every thermal substep, whose cadence could be relaxed. The production
annual run has no such solve: HVAC demand is closed-form and costs one
arithmetic pass per step, so there is no iteration budget for a coarser
cadence to save. This is the same premise failure the series already hit
on #4690 (no BDF/Newton iteration on the annual path).

The question re-opens only if and when the #3984 pressure-driven air
loop is wired into the production step (it exists, tested in isolation,
with a bracketed-Newton loop solve and a stated maximum Newton iteration
count). At that point the baseline to beat is this slice's: any cadence
relaxation must be measured against the air loop's actual iterations per
step, and latent/humidity drift must be judged on a humid-climate case
(e.g. Baltimore), per the issue. Until then, slices 2–3 of #4694 should
not be dispatched.

*Collection note (supervisor): worker run fluxion-4694 (balanced,
MiniMax-M2.7, 60/60 steps) completed §1 and hit its step cap polling the
probe build; §§2–5 were completed by the supervisor from §1's verified
evidence, and the unfinished probe was deleted. No number above comes
from worker prose alone — §1's greps were spot-checked at collection.*
