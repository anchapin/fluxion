# Fluxion Measures — Standard Library

This directory is the home of Fluxion's **standard library of baseline Python
Measures** — out-of-the-box, OpenStudio-equivalent scripts for the most common
energy-modeling tasks so users do not have to write them from scratch.

A Fluxion **measure** is a Python class that mutates a building model **once**,
before the Rust simulation engine consumes it. Measures are **AOT
(Ahead-of-Time) pre-processors** — they are forbidden from running inside the
timestepping loop, because doing so would force the CPython GIL to contend
against every Rust `rayon` worker and collapse the parallel `BatchOracle`
throughput (≥10k configs/sec on 8 cores). See [`docs/measures.md`](../docs/measures.md)
for the full design rationale.

## Standard-library measures

| File | Class | What it does |
|------|-------|--------------|
| [`SetWindowToWallRatio.py`](SetWindowToWallRatio.py) | `SetWindowToWallRatio` | Resize exterior glazing to a target window-to-wall ratio, preserving sill height and glazing topology. |
| [`ReplaceHVACWithVAV.py`](ReplaceHVACWithVAV.py) | `ReplaceHVACWithVAV` | Replace ideal-loads with an explicit VAV air system (economizer + cool-deck setpoint). |
| [`IncreaseInsulationRValue.py`](IncreaseInsulationRValue.py) | `IncreaseInsulationRValue` | Add ΔR of insulation to the opaque envelope by lowering surface U-values. |
| [`SetThermostatSetpoints.py`](SetThermostatSetpoints.py) | `SetThermostatSetpoints` | Sweep the heating / cooling thermostat setpoints by a delta or pin them to absolute values (clamped to the engine's admissible window, deadband preserved). |
| [`SwapSchedule.py`](SwapSchedule.py) | `SwapSchedule` | Replace the thermostat schedule with an alternative profile — `constant`, `setback` or `operating_hours`. The named schedule must exist in the model, otherwise the measure fails loudly. |
| [`SetLightingPowerDensity.py`](SetLightingPowerDensity.py) | `SetLightingPowerDensity` | Set the lighting power density (W/m²) of the model's lighting / internal-load inputs. The density must be positive and bounded; a model with no lighting object fails loudly. |
| [`SetHVACEfficiency.py`](SetHVACEfficiency.py) | `SetHVACEfficiency` | Set the heating / cooling coefficient of performance (COP) of the model's HVAC plant. A model with no HVAC system fails loudly. |
| [`SetEquipmentPowerDensity.py`](SetEquipmentPowerDensity.py) | `SetEquipmentPowerDensity` | Set the equipment / plug-load power density (W/m²) of the model's equipment inputs. The density must be positive and bounded; a model with no equipment object fails loudly. |
| [`AddOverhang.py`](AddOverhang.py) | `AddOverhang` | Attach a horizontal overhang (depth / mounting height, per façade orientation) to the model's exterior façades. A model with no eligible façade fails loudly. |
| [`SetWindowProperties.py`](SetWindowProperties.py) | `SetWindowProperties` | Set the glazing U-value (W/m²·K) and/or the solar heat gain coefficient (SHGC) of the model's window specs — the single-pane → low-e double-pane upgrade study. A model with no window spec fails loudly. |
| [`SetInfiltrationRate.py`](SetInfiltrationRate.py) | `SetInfiltrationRate` | Set the model's infiltration / air-leakage rate (ACH) to a constant target — the envelope-tightening study. The rate must be in `[0, 20]` ACH (0 is a sealed-envelope study). A model with no infiltration representation fails loudly. |
| [`SetOccupancyDensity.py`](SetOccupancyDensity.py) | `SetOccupancyDensity` | Set the occupancy density (persons/m²) of the model's occupancy / internal-load inputs. The density must be positive and bounded; a model with no occupancy object fails loudly. |
| [`ReplaceHVACWithHeatPump.py`](ReplaceHVACWithHeatPump.py) | `ReplaceHVACWithHeatPump` | Replace the model's HVAC system with an air-source heat pump (heating / cooling COP, optional plant capacity and outdoor operating envelope). The COPs must be positive and bounded; a model with no HVAC system fails loudly. |

> **Control-side persistence (Issue #3995).** `SetThermostatSetpoints` and
> `SwapSchedule` act on the model's *control* data. The native
> `fluxion.Model` binding does not expose a zone-setpoint or schedule setter
> yet, so on today's bindings these two measures are **advisory**: the
> transformed values are computed (and unit-tested) and logged, but they do
> not round-trip into the serialized model. The measures push their result
> back through `set_zone_setpoints` / `set_hvac_schedule` as soon as the
> binding provides one — the same advisory-field pattern documented for
> `ReplaceHVACWithVAV`.

> **Load / plant measures (Issue #3995).** `SetLightingPowerDensity` and
> `SetHVACEfficiency` are the load-side and plant-side members of the
> standard library. `SetLightingPowerDensity` probes the model for the
> lighting / internal-loads object that carries the density
> (`lighting_power_density_w_m2`, the field name of the Rust
> `ScheduleSet`) and raises `LightingNotFoundError` when the model
> exposes none — a lighting study against a model without lighting must
> fail loudly rather than report a change that never happened.
> `SetHVACEfficiency` writes the `cop_heating` / `cop_cooling` fields of
> the HVAC snapshot and raises `HVACSystemNotFoundError` when the model
> carries no HVAC system. The native `fluxion.Model` binding exposes
> neither a lighting accessor nor a COP setter, so on today's bindings
> the lighting measure fails loudly and the COP change is **advisory**
> (visible on the HVAC snapshot and preserved by the Python
> save/load round-trip, but not consumed by the Rust simulation).

> **Equipment densities (Issue #3995).** `SetEquipmentPowerDensity` is the
> equipment-side twin of `SetLightingPowerDensity`: it sets the plug /
> process-equipment power density (W/m²). It probes the model for the
> equipment / internal-loads object that carries the density — accepting
> both the canonical `equipment_power_density_w_m2` (the Rust `ScheduleSet`
> naming for a W/m² density) and the `equipment_power_density` alias used
> by the BEM checker (`src/validation/copilot/checker.rs`) — and raises
> `EquipmentNotFoundError` when the model exposes none, so an equipment
> study against a model without equipment fails loudly rather than report
> a change that never happened. Its bounds (0.1-200 W/m²) live in the
> measure: the lower floor rejects "no equipment" (a deletion), the upper
> bound sits above the checker's 30 W/m² "high for an office" warning so
> dense server rooms stay modellable. As with the lighting measure, the
> native `fluxion.Model` binding exposes no equipment accessor, so on
> today's bindings the measure fails loudly on a native model.

> **Occupancy density (Issue #3995).** `SetOccupancyDensity` is the
> people-side member of the standard library: it sets the occupancy
> density (persons/m² — the unit the repo uses everywhere, from the BEM
> checker's `occupancy_density` key in
> `src/validation/copilot/checker.rs` to the `people_per_area`
> OpenStudio-interop field in `src/interop/osm/types.rs`). The Rust
> model carries the occupancy *schedule* (`ScheduleSet.occupancy`,
> `src/api/schema.rs`) but no occupancy-density field, so the measure
> probes the model for the occupancy / internal-loads object that carries
> the density — accepting `occupancy_density`, `occupant_density` and
> `people_per_area` — and raises `OccupancyNotFoundError` when the model
> exposes none, so an occupancy study against a model without occupancy
> fails loudly rather than report a change that never happened. Its
> bounds (0.01-10 persons/m²) live in the measure: the lower floor
> rejects "nobody is ever here" (a deletion — the BEM checker treats a
> non-positive occupancy density as an error), the upper bound is the
> declared occupancy validity range (0 to 10 persons/m²,
> `docs/SURROGATE_GOVERNANCE.md`) and sits above the checker's 1.0
> persons/m² "very high" warning so dense spaces stay modellable. As
> with the lighting and equipment measures, the native `fluxion.Model`
> binding exposes no occupancy accessor, so on today's bindings the
> measure fails loudly on a native model.

> **HVAC electrification (Issue #3995).** `ReplaceHVACWithHeatPump` is the
> electrification member of the standard library: it replaces the model's
> HVAC system with an air-source heat pump — the canonical "what does the
> bill look like once the boiler becomes a heat pump?" study, and the
> plant-side sibling of `ReplaceHVACWithVAV` (which electrifies the air
> side). It writes the heat-pump COP onto the HVAC snapshot
> (`cop_heating` / `cop_cooling`, default 3.0 / 3.5) and, optionally, the
> plant capacities and the outdoor operating envelope of the unit
> (`min_outdoor_temp` / `max_outdoor_temp`). The COP bounds (0.1-20) are
> the ones `SetHVACEfficiency` already uses — the measure reuses them
> rather than inventing new ones — and they are validated *before* the
> model is touched, so an invalid request never leaves a half-mutated
> model behind. A model that carries no HVAC system fails loudly with
> `HVACSystemNotFoundError`, exactly like `SetHVACEfficiency`.
>
> The repo does know about heat pumps — the Rust engine has the
> `HVACSystemType::HeatPump` variant and a full `HeatPump` struct
> (`src/sim/hvac/mod.rs`), the Python binding exposes
> `fluxion.HVACSystemType.HeatPump` (`src/python/hvac_bindings.rs`), and
> the JSON schema accepts the type through `controls.hvac_system`
> (`src/api/schema.rs`). The *measure-facing* HVAC snapshot, however,
> carries no system-type field (`PyHVACSystem`,
> `src/python/model_bindings/hvac.rs`), so on today's bindings the
> measure sets the heat-pump **COP** (an advisory field, as documented for
> `ReplaceHVACWithVAV` and `SetHVACEfficiency`) and records the system
> type only where the model's representation can carry one — it never
> claims a system-type change that did not happen.

> **Shading (Issue #3995).** `AddOverhang` is the shading member of the
> standard library: it attaches a horizontal overhang (projection `depth`,
> mounting `height`) to every façade of the requested `orientation`
> (`north` / `east` / `south` / `west`, or `all` for every cardinal
> vertical façade — the native `Orientation` enum has no "all exterior"
> variant), optionally restricted to glazed surfaces with
> `only_with_windows`. It validates the depth / height bounds *before*
> touching the model and raises `NoEligibleSurfaceError` when no façade is
> eligible, so a shading study never reports a change that never happened.
> The overhangs are attached to the surface snapshots and pushed back
> through `model.set_surfaces(...)`; on a model without that setter the
> change is **advisory**.

> **Glazing properties (Issue #3995).** `SetWindowProperties` is the
> glazing-*property* sibling of `SetWindowToWallRatio` (which changes the
> glazing *area* only): it sets the `window_u_value` (W/m²·K) and/or the
> `window_shgc` (the solar heat gain coefficient) of the model's window
> specs — the canonical single-pane → low-e double-pane upgrade study.
> Both arguments are optional and independent (setting one leaves the
> other untouched); at least one must be given. The U-value must be
> `> 0` (the Rust rule in `src/api/schema.rs`) and `<= 10` W/m²·K (a
> measure-side sanity cap — single-pane clear glass is ≈ 5.8), the SHGC
> must be in `(0, 1]` (exactly the Rust validator's rule). Both are
> validated *before* the model is touched, so an invalid request never
> leaves a half-mutated model. The measure probes the model for its
> window specs — a `window_specs()` / `windows()` accessor, then the
> surfaces / zones carrying the glazing fields (directly or in a nested
> `window` attribute), then the fields held on the model itself — mutates
> **all** of them, and pushes the mutated collection back through the
> matching setter. A model that exposes no window spec fails loudly with
> `WindowNotFoundError`; the native `fluxion.Model` binding exposes no
> window-spec accessor today, so on a native model the measure fails
> loudly rather than reporting an upgrade that never happened.

> **Infiltration (Issue #3995).** `SetInfiltrationRate` is the envelope-
> leakage member of the standard library: it sets the model's infiltration
> rate to a constant target in **ACH (air changes per hour, 1/h)** — the
> unit the engine itself uses (`h_ve = infiltration_rate · V · ρ · c_p /
> 3600`, with `infiltration_rate` in ACH,
> `src/sim/thermal_model_physics/dispatch_state.rs`) and the unit the
> importers produce (`extract_infiltration_ach`,
> `src/io/idf/convert.rs`). The rate is validated **before** the model is
> touched: it must be `>= 0` (a sealed-envelope study is legitimate — the
> engine models a zero rate itself) and `<= 20` ACH, a generous ceiling
> above the most extreme envelope-leakage fixture in the repository
> (10 ACH, `ventilation_denver_10ach.idf`). The measure probes the model
> for every infiltration representation it exposes and writes the *same*
> rate into each: the schema's `ScheduleSet.infiltration` daily schedule
> (written as a flat 24-hour profile), the engine's per-zone
> `setpoints.infiltration_rate`, and the rate held directly on the model.
> A model that exposes none of them fails loudly with
> `InfiltrationNotFoundError` — the measure never materialises an
> infiltration schedule the model does not have (`ScheduleSet::default()`
> sets `infiltration: None`, and the engine defines no default schedule
> for it). On today's native `fluxion.Model` binding the measure therefore
> fails loudly: the binding exposes neither a schedule-set accessor nor a
> setpoints accessor, so a native model exposes no infiltration
> representation at all. Where a binding does expose the representation but
> no setter, the change is **advisory** — the new rate is visible on the
> snapshot but does not round-trip into the serialized model.

> **Named schedules (Issue #3995).** `SwapSchedule` resolves its
> `schedule_name` argument against the schedules the model exposes
> (`model.schedule_names()` or `model.schedules()`; a model with neither
> carries the single implicit thermostat schedule `"Thermostat"`). A name
> that does not exist raises `ScheduleNotFoundError` — the measure fails
> loudly rather than reporting a swap that never happened. Note that the
> binding exposes a single schedule slot per model, so the named lookup
> validates the request; it does not select among several writable
> schedules.

Reference / example measures live under [`examples/`](examples/):

| File | Class | What it does |
|------|-------|--------------|
| [`examples/add_overhang.py`](examples/add_overhang.py) | `AddSouthOverhang` | Attach a horizontal overhang to every south-facing surface. |
| [`examples/set_hvac_cop.py`](examples/set_hvac_cop.py) | `SetHVACCOP` | Set the heating/cooling plant capacities. |

## How the AOT runner discovers measures

`fluxion apply-measures` (and `fluxion.measures.discover_measures`) walks a
`--measures` directory **recursively** for `*.py` files and imports each one.
Discovery rules:

- Files starting with `_` are skipped (`__init__.py`, `_helpers.py`, …).
- Every **concrete** `FluxionMeasure` subclass declared in an imported file is
  collected.
- Abstract subclasses (`abc.ABCMeta` + `@abstractmethod` on `apply`, or a
  subclass that does not override `apply`) are skipped.
- Results are sorted by class name for deterministic CLI output.

Pointing `--measures` at this directory (`measures/`) discovers the
standard-library measures **and** the examples under `measures/examples/`,
because discovery is recursive. To run only the standard library, point at a
copy without the `examples/` subdir, or filter with `--list` / `--measure-args`.

## The measure format

Each measure is a single `*.py` file containing a concrete subclass of
[`fluxion.FluxionMeasure`](../fluxion/measures.py):

```python
from fluxion import FluxionMeasure

class MyMeasure(FluxionMeasure):
    name = "MyMeasure"            # optional; defaults to class name
    description = "What it does."

    def arguments(self) -> list[dict]:
        # OpenStudio-style argument spec. The CLI merges --measure-args
        # JSON with these declared defaults via parse_arguments().
        return [
            {"name": "target", "type": "double", "default": 0.4,
             "min": 0.0, "max": 1.0, "description": "Target value."},
        ]

    def apply(self, model, arguments: dict) -> None:
        # Mutate ``model`` in place. Use the snapshot/owned-value contract:
        #   surfaces = model.surfaces()      # read snapshots (ONCE)
        #   for s in surfaces: ...           # mutate the stored list
        #   model.set_surfaces(surfaces)     # push back — REQUIRED to persist
```

Each standard-library measure additionally ships:

- A **descriptor** — the `arguments()` method returns the argument spec,
  analogous to OpenStudio's `measure.xml`.
- The **Python implementation** — `apply()`.
- A **unit test** — see [`tests/python/test_standard_measures.py`](../tests/python/test_standard_measures.py) and, for the control-side measures, [`tests/python/test_standard_measures_control.py`](../tests/python/test_standard_measures_control.py).

## Provenance (Issue #1816)

Every measure run through `fluxion apply-measures` (or
`fluxion.measures.apply_measures` with an `applied_deltas` accumulator) appends
an `AppliedDelta` entry to the model's provenance chain, in application order.
The chain is embedded in the serialized output under `applied_deltas` so
downstream ML pipelines can reconstruct which measures produced a given
energy-use number. Measures perform no provenance bookkeeping themselves — the
runner owns the chain.

## Usage

```bash
# 1. Create a base model JSON
python -c "import fluxion; from fluxion.measures import save_model; \
    save_model(fluxion.Model(num_zones=1), 'base.json')"

# 2. Run the standard library against it
fluxion apply-measures \
    --model base.json \
    --measures measures/ \
    --measure-args args.json \
    --output model.baseline.json

# 3. Inspect the provenance chain
python -c "import json; print([d['name'] for d in \
    json.load(open('model.baseline.json'))['applied_deltas']])"
```

Where `args.json` maps measure names to argument dicts:

```json
{
  "SetWindowToWallRatio": {"target_wwr": 0.40},
  "ReplaceHVACWithVAV": {"heating_capacity": 18000.0, "cooling_capacity": 15000.0},
  "IncreaseInsulationRValue": {"delta_r": 2.5}
}
```

## See also

- [`docs/measures.md`](../docs/measures.md) — FluxionMeasure base class, AOT-only rule, provenance schema.
- [`docs/bindings.md`](../docs/bindings.md) — snapshot/owned-value memory-safety contract.
- [`docs/tutorials/writing_your_first_measure.md`](../docs/tutorials/writing_your_first_measure.md) — end-to-end authoring walkthrough.
