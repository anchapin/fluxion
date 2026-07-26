# Fluxion Python Measures — AOT Runner (Issue #1814)

<!-- 7-line summary for AI agents: lines 1-7 -->
<!-- 1: This document describes the FluxionMeasure Python base class and AOT runner. -->
<!-- 2: Read it before writing a custom measure or modifying the measures subsystem. -->
<!-- 3: Key concepts: AOT-only rule, GIL/rayon rationale, snapshot/owned-value lifecycle. -->
<!-- 4: Also covers the AppliedDelta provenance chain (Issue #1816) and its schema. -->
<!-- 5: Companion to docs/bindings.md (memory safety) and ARCHITECTURE.md (module boundaries). -->
<!-- 6: Stable as of issue #1816 — measures API + provenance contract are feature-frozen pending OSM integration. -->
<!-- 7: After changes, run `maturin develop` then `pytest tests/python/test_apply_measures_cli.py`. -->

## TL;DR

A Fluxion **measure** is a Python class that mutates a building model **once**, before the Rust simulation engine consumes it. Measures are **AOT (Ahead-of-Time) pre-processors** — they are forbidden from running inside the timestepping loop, because doing so would force the GIL to contend against every Rust rayon worker and serialize the parallel BatchOracle path (>=10k configs/sec on 8 cores).

The runner is the `fluxion apply-measures` CLI. It loads a base model, walks a directory for `FluxionMeasure` subclasses, runs each in sequence, and writes the mutated model back to JSON (or `.msgpack` if `msgpack` is installed).

```bash
fluxion apply-measures \
    --model base.json \
    --measures measures/examples/ \
    --output model.with_overhangs.json
```

## The AOT-Only Rule (Non-Negotiable)

The Rust timestepping loop is parallelized via `rayon::par_iter`. If a measure runs *inside* that loop:

1. CPython's GIL serializes all rayon workers — the speedup collapses to 1x.
2. Every per-step callback pays the cost of acquiring the GIL.
3. The `BatchOracle.evaluate_population` hot path stops being parallel.

Therefore:

- Measures are pre-processing steps that mutate the model *once*, *before* simulation.
- The runtime emits a `RuntimeWarning` if a measure detects it is running on a thread named `rayon-*` or `tokio-*`, or when the env var `FLUXION_INSIDE_TIMESTEPPING=1` is set. The warning is informational (does not raise) so that legitimate parallel use cases can opt out.
- This is enforced by the `_FluxionMeasureMeta` metaclass, which wraps every subclass `apply()` with `_warn_if_inside_timestepping`.

If you need per-step Python callbacks, implement a Rust trait in `src/sim/` instead.

## FluxionMeasure API

```python
from fluxion import FluxionMeasure

class AddSouthOverhang(FluxionMeasure):
    name = "AddSouthOverhang"               # optional; defaults to class name
    description = "Attach overhang to south-facing opaque surfaces."

    def arguments(self) -> list[dict]:
        """Return the OpenStudio-style argument spec."""
        return [
            {"name": "depth",  "type": "double", "default": 1.0, "min": 0.0, "max": 5.0},
            {"name": "height", "type": "double", "default": 2.5, "min": 0.0, "max": 10.0},
        ]

    def apply(self, model, arguments: dict) -> None:
        """Mutate ``model`` in place using parsed ``arguments``."""
        # Snapshot the model's surfaces (owned values, see docs/bindings.md).
        surfaces = model.surfaces()
        for s in surfaces:
            if s.orientation == fluxion.Orientation.South:
                s.add_overhang(depth=arguments["depth"], height=arguments["height"])
        model.set_surfaces(surfaces)  # push back — REQUIRED for persistence
```

`arguments()` mirrors OpenStudio's `arguments()` method. Each entry is a dict with at minimum `name`, `type` (`string` / `double` / `integer` / `bool` / `choice`), and optional `default`, `min`, `max`, `description`, `choices`. The CLI parses `--measure-args JSON` and merges it with declared defaults via `parse_arguments()`.

## The CLI

```
fluxion apply-measures
    --model PATH            Base model JSON or .msgpack
    --measures DIR          Directory containing FluxionMeasure subclasses
    --output PATH           Output file (default: model.applied.json)
    --measure-args PATH     Optional JSON: {measure_name: {arg: value}}
    --dry-run               Print the plan (measure name, class, args) and exit
    --list                  List discovered measure classes and exit
    -v / -vv                Verbosity (warning, info, debug)
```

Discovery rules:

- Walks `--measures` recursively for `*.py` files.
- Skips files starting with `_` (e.g. `__init__.py`, `_helpers.py`).
- Imports each file in a uniquely-named module and collects concrete `FluxionMeasure` subclasses declared in that file.
- Stable alphabetical order on output.

## Model Serialization

Two formats are supported:

| Format | Extension | Notes |
|--------|-----------|-------|
| JSON   | `.json`   | Default; readable, diff-friendly. |
| msgpack| `.msgpack`| Binary; smaller; requires `pip install msgpack`. Falls back to JSON if not installed. |

The on-disk schema is versioned (`schema_version: "1.0.0"`). It is currently a *round-trip* format for tests and CI smoke checks; the Rust runtime does not yet consume it directly. The full schema is being stabilized in `ARCHITECTURE.md`.

## Memory-Safety Recap

`fluxion.Model` uses the **snapshot / owned-value** contract from issue #1812 (see `docs/bindings.md`):

1. `model.surfaces()` returns a fresh list of owned `Surface` snapshots.
2. Mutating a snapshot does NOT affect the model.
3. Call `model.set_surfaces(snapshots)` to push mutations back.
4. `model.hvac_system()` / `model.set_hvac_system(...)` follow the same pattern.

Not all `HVACSystem` fields round-trip back to the model — see `src/python/model_bindings.rs` for the current ownership story. The `SetHVACCOP` example demonstrates the right pattern (mutate fields that do persist, document the advisory ones).

## Reference Material

- Issue #1812 — PyO3 struct exposure (the snapshot types measures mutate): [`docs/bindings.md`](bindings.md).
- Issue #1814 — This document and the runner CLI.
- OpenStudio `ModelMeasure` reference: https://openstudio.net/docs/latest/Measure_Guide/.

## End-to-end Example

```bash
# 1. Create a base model JSON
python -c "import fluxion; from fluxion.measures import save_model; \
    save_model(fluxion.Model(num_zones=1), 'base.json')"

# 2. Run the example measures
fluxion apply-measures \
    --model base.json \
    --measures measures/examples/ \
    --output model.applied.json

# 3. Inspect the result
python -c "
import json
with open('model.applied.json') as f:
    data = json.load(f)
print('applied:', data['_fluxion_run']['applied'])
print('zones:', data['num_zones'])
south = [s for s in data['surfaces'] if s['orientation'] == 'South']
print('south surfaces with overhang:', sum(1 for s in south if s['overhang_depth']))
"
```

## Provenance — the `AppliedDelta` chain (Issue #1816)

When a building model is altered by either a JSON Patch (#1811) or a Python
Measure (#1814), the resulting output dataset must be able to answer *"why did
this model change?"*. This is critical for ML feature tracking, debugging, and
reproducibility — ML data engineers need to know exactly which parametric
permutations produced a given energy-use number.

### The contract

Every successful mutation appends an **`AppliedDelta`** entry to the model's
provenance chain, in application order. The chain travels with the serialized
model / `results.json` so downstream consumers can reconstruct the full
mutation history without re-running.

- **JSON Patches** applied via `fluxion::measures::json_patch::apply_delta`
  (Rust) append an entry automatically, with a SHA-256 `digest` of the patch
  payload.
- **Python Measures** executed via `apply_measures` (the AOT runner) append an
  entry per measure when the caller passes an `applied_deltas` accumulator.

### The `AppliedDelta` schema

```json
{
  "source": "json_patch",
  "name": "json_patch:a1b2c3d4e5f60718",
  "timestamp": "0000000000000000",
  "digest": "a1b2c3d4e5f60718..."
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `source` | `"json_patch"` \| `"python_measure"` | Which subsystem produced the mutation. |
| `name` | string | Identifier of the patch file or measure class. For auto-recorded JSON patches this defaults to `json_patch:<short-digest>`; use `apply_delta_with_name(model, patch, name)` (Rust) for a human-readable label. |
| `timestamp` | string | **Logical sequence** (zero-padded, pure-Rust path) or an ISO-8601 wall-clock instant (Python AOT-runner path). See *Determinism* below. |
| `digest` | string \| `null` | SHA-256 (hex) of the patch payload. `null` when there is no deterministic payload to hash (e.g. a Python measure that mutates via the snapshot API). |

The Rust source of truth is `src/measures/provenance.rs`
(`AppliedDelta` / `DeltaSource`); the Python mirror is
`fluxion.measures.make_applied_delta`. The two are kept byte-compatible.

### Determinism

The `timestamp` field is a **monotonic logical clock** (a zero-padded sequence
index assigned from `applied_deltas.len()`) on the pure-Rust path, **not** a
wall-clock instant. This is deliberate: the Fluxion determinism gate
(Issue #1351) requires that two byte-identical inputs produce byte-identical
serialized output, and `SystemTime::now()` would violate that contract. The
Python AOT runner — which is not subject to the determinism gate — overwrites
`timestamp` with a real ISO-8601 instant at the application layer.

### Round-trip example

A model run through a JSON patch **and** a Python measure produces a chain with
both entries, in application order:

```python
from fluxion.measures import make_applied_delta, save_model

# ... after apply_delta (Rust) and apply_measures (Python) ...
chain = [
    make_applied_delta("json_patch", "patches/insulation.json", digest="abc..."),
    make_applied_delta("python_measure", "AddSouthOverhang"),
]
save_model(model, "results.json", applied_deltas=chain)
```

`results.json` now carries:

```json
{
  "schema_version": "1.0.0",
  "...": "...",
  "applied_deltas": [
    { "source": "json_patch",      "name": "patches/insulation.json", "timestamp": "...", "digest": "abc..." },
    { "source": "python_measure",  "name": "AddSouthOverhang",         "timestamp": "..." }
  ],
  "_fluxion_run": {
    "applied": ["AddSouthOverhang"],
    "applied_deltas": [ /* same chain, echoed for convenience */ ]
  }
}
```

ML pipelines (cf. #1337) consume `applied_deltas` to reconstruct the provenance
chain and feature-hash identical permutations via the `digest`.
