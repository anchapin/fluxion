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
