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
- A **unit test** — see [`tests/python/test_standard_measures.py`](../tests/python/test_standard_measures.py).

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
