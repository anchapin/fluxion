# Writing Your First Fluxion Measure

<!-- 7-line summary for AI agents: lines 1-7 -->
<!-- 1: End-to-end tutorial: author a FluxionMeasure, run it via the AOT CLI, inspect output. -->
<!-- 2: Read this before writing a custom measure or extending the standard library. -->
<!-- 3: Covers the FluxionMeasure API, the snapshot/owned-value mutation pattern, and provenance. -->
<!-- 4: Companion to docs/measures.md (reference) and measures/README.md (standard library). -->
<!-- 5: Prerequisites: `maturin develop` for native bindings; the fluxion CLI must be importable. -->
<!-- 6: Stable as of issue #1815 — the standard-library measures referenced here ship in measures/. -->
<!-- 7: After changes, run `pytest tests/python/test_standard_measures.py`. -->

This tutorial walks through the complete lifecycle of a Fluxion Python Measure:
authoring it, running it through the AOT (Ahead-of-Time) CLI, and inspecting the
mutated output. By the end you will have written a measure that adds glazing to
south-facing walls and verified its effect end-to-end.

## Prerequisites

```bash
# Native bindings (required to mutate a real fluxion.Model)
maturin develop --features python-bindings

# Verify the CLI is importable
python -m fluxion.cli --help
```

## Table of Contents

1. [What is a Measure?](#what-is-a-measure)
2. [The AOT-Only Rule](#the-aot-only-rule)
3. [The Snapshot / Owned-Value Contract](#the-snapshot--owned-value-contract)
4. [Step 1 — Author the Measure](#step-1--author-the-measure)
5. [Step 2 — Run It Through the CLI](#step-2--run-it-through-the-cli)
6. [Step 3 — Inspect the Mutated Output](#step-3--inspect-the-mutated-output)
7. [Step 4 — Provenance and the AppliedDelta Chain](#step-4--provenance-and-the-applieddelta-chain)
8. [Testing Your Measure](#testing-your-measure)

---

## What is a Measure?

A Fluxion **measure** is a Python class that mutates a building model **once**,
before the Rust simulation engine consumes it. It is the direct analogue of
OpenStudio's `OpenStudio::Measure::ModelMeasure`. Measures let you express
common modeling operations — "set the window-to-wall ratio to 40%", "replace
the HVAC system with a VAV", "add R-5 of wall insulation" — as reusable,
composable scripts.

Fluxion ships a **standard library** of baseline measures in
[`measures/`](../../measures/), including `SetWindowToWallRatio`,
`ReplaceHVACWithVAV`, and `IncreaseInsulationRValue`. This tutorial shows you
how to write your own.

## The AOT-Only Rule

Measures are **pre-processors**. They run *before* simulation, never inside the
timestepping loop. Why? The Rust timestepping loop is parallelized with
`rayon::par_iter`. If a Python measure ran inside that loop, the CPython GIL
would serialize every `rayon` worker and collapse the parallel speedup.

The `FluxionMeasure` base class enforces this: its metaclass wraps every
subclass `apply()` with a guard that emits a `RuntimeWarning` if it detects a
`rayon-*` / `tokio-*` worker thread or the `FLUXION_INSIDE_TIMESTEPPING=1` env
var. The warning is informational (it does not raise), but it is loud and
CI-friendly.

**Takeaway:** write measures as one-shot mutations. If you need per-timestep
logic, implement a Rust trait in `src/sim/` instead.

## The Snapshot / Owned-Value Contract

`fluxion.Model` exposes its data through **snapshots** — owned Python copies
with no references back into the Rust model:

1. `model.surfaces()` returns a fresh list of `Surface` snapshots.
2. Mutating a snapshot does **not** affect the model.
3. Call `model.set_surfaces(snapshots)` to push mutations back.
4. `model.hvac_system()` / `model.set_hvac_system(...)` follow the same pattern.

This avoids the PyO3 lifetime pitfalls (double-free, dangling references) that
would arise from exposing Rust-owned data directly. The cost is that you must
**explicitly push back** every mutation. See
[`docs/bindings.md`](../bindings.md) for the full story.

> **Gotcha:** access `model.surfaces()` **once** and store the result in a
> variable, then mutate and push back that same variable. Re-accessing
> `zone.surfaces` returns fresh clones each time (PyO3 `#[pyo3(get)]` on a
> `Vec` builds a new list), so mutations through re-read accessors are lost.

## Step 1 — Author the Measure

Create a file `measures/my_measures/add_south_glazing.py`:

```python
"""AddSouthGlazing — a minimal Fluxion measure."""

from __future__ import annotations

import logging
from typing import Any

from fluxion import FluxionMeasure

_logger = logging.getLogger(__name__)


def _orientation_name(value: Any) -> str:
    return repr(value).rsplit(".", 1)[-1]


class AddSouthGlazing(FluxionMeasure):
    """Add a fixed window area to every south-facing wall."""

    def arguments(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "window_area",
                "type": "double",
                "default": 3.0,
                "min": 0.0,
                "description": "Window area to add per south-facing surface (m²).",
            },
        ]

    def apply(self, model: Any, arguments: dict[str, Any]) -> None:
        window_area = float(arguments.get("window_area", 3.0))

        # Snapshot ONCE, mutate in place, push back the SAME list.
        import fluxion

        surfaces = model.surfaces()
        modified = 0
        for s in surfaces:
            if s.orientation == fluxion.Orientation.South:
                s.window_area = window_area
                modified += 1

        model.set_surfaces(surfaces)  # REQUIRED — persists the mutation
        _logger.info("AddSouthGlazing: set %d south surfaces to %.1f m²",
                     modified, window_area)
```

The structure:

- **`arguments()`** — returns an OpenStudio-style argument spec. Each entry has
  `name`, `type` (`string` / `double` / `integer` / `bool` / `choice`), optional
  `default`, `min`, `max`, `description`. The CLI merges `--measure-args` JSON
  with these defaults via `parse_arguments()`.
- **`apply(model, arguments)`** — mutates `model` in place. `arguments` is the
  merged dict (user overrides + declared defaults).

## Step 2 — Run It Through the CLI

First, create a base model JSON:

```bash
python -c "import fluxion; from fluxion.measures import save_model; \
    save_model(fluxion.Model(num_zones=1), 'base.json')"
```

Then run the AOT runner:

```bash
fluxion apply-measures \
    --model base.json \
    --measures measures/my_measures/ \
    --measure-args args.json \
    --output model.glazing.json
```

Where `args.json` is:

```json
{"AddSouthGlazing": {"window_area": 4.0}}
```

You can also run the standard library directly:

```bash
fluxion apply-measures \
    --model base.json \
    --measures measures/ \
    --output model.baseline.json
```

Useful CLI flags:

- `--list` — discover and print measure class names; do not run.
- `--dry-run` — print the plan (name, class, arguments); do not mutate.
- `-v` / `-vv` — increase logging verbosity.

## Step 3 — Inspect the Mutated Output

The runner writes the mutated model to `--output` as JSON. Inspect it:

```python
import json

with open("model.glazing.json") as f:
    data = json.load(f)

south = [s for s in data["surfaces"] if s["orientation"] == "South"]
print("south-facing surfaces:", len(south))
print("window areas:", [s["window_area"] for s in south])
```

Or reload the model and inspect the live snapshot:

```python
from fluxion.measures import load_model

m = load_model("model.glazing.json")
for s in m.surfaces():
    if repr(s.orientation).endswith("South"):
        print(f"area={s.area:.2f}, window_area={s.window_area:.2f}")
```

## Step 4 — Provenance and the AppliedDelta Chain

Every measure run through the CLI appends an `AppliedDelta` entry to the model's
provenance chain (Issue #1816). The chain lives under `applied_deltas` in the
output JSON, in application order:

```python
import json

data = json.load(open("model.glazing.json"))
for entry in data["applied_deltas"]:
    print(f"{entry['source']:16} {entry['name']:24} {entry['timestamp']}")
```

Output:

```
python_measure  AddSouthGlazing         2026-07-26T11:30:00+00:00
```

This is critical for ML feature tracking and reproducibility: downstream
pipelines can reconstruct exactly which measures produced a given energy-use
number without re-running the model. The `_fluxion_run` section of the output
echoes the same chain for convenience.

## Testing Your Measure

Add a test in `tests/python/` that exercises both the pure-Python logic and the
full integration path. Split numerical logic into a pure helper so it is
testable without the native bindings (per `RULES.md`):

```python
import pytest

requires_fluxion = pytest.mark.skipif(
    not importlib.util.find_spec("fluxion"),
    reason="fluxion bindings not available",
)


def test_window_area_math():
    from measures.my_measures.add_south_glazing import compute_window_area
    assert compute_window_area(20.0, 0.4) == pytest.approx(8.0)


@requires_fluxion
def test_measure_applies_glazing():
    import fluxion
    from fluxion.measures import apply_measures
    from measures.my_measures.add_south_glazing import AddSouthGlazing

    m = fluxion.Model(num_zones=1)
    apply_measures(m, [AddSouthGlazing], {"AddSouthGlazing": {"window_area": 4.0}})
    south = [s for s in m.surfaces() if repr(s.orientation).endswith("South")]
    assert all(s.window_area == pytest.approx(4.0) for s in south)
```

Run with:

```bash
pytest tests/python/test_standard_measures.py -v
```

## Next Steps

- Browse the standard library in [`measures/`](../../measures/) for real-world
  patterns (`SetWindowToWallRatio`, `ReplaceHVACWithVAV`).
- Read [`docs/measures.md`](../measures.md) for the full reference (serialization
  formats, memory-safety recap, provenance schema).
- Read [`docs/bindings.md`](../bindings.md) for the PyO3 ownership story.
