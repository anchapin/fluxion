# Fluxion Python Bindings — Lifetime and Memory-Safety Contract

This document describes the memory-safety and ownership contract for Fluxion's
PyO3 Python bindings, with particular attention to the issue #1812 surface
(`Zone`, `Surface`, `Material`, `HVACSystem`, `ShadingDevice`,
`Orientation`).

## TL;DR

Every interior struct returned from a `FluxionModel` is an **owned snapshot**.
There are **no** references from Python back into the Rust model. Python
garbage collection of a snapshot cannot invalidate the model, and mutating /
re-simulating the model cannot invalidate a held snapshot.

```python
import fluxion

model = fluxion.Model(num_zones=3)

# Snapshot — owned. No reference to `model` retained.
zones = model.zones()

# GC the snapshot — model is unaffected.
del zones
import gc; gc.collect()

# Model is still usable; subsequent snapshots are independent.
zones = model.zones()
zones[0].temperature = 25.0     # mutates this snapshot only
assert model.zones()[0].temperature != 25.0  # model untouched
```

## Snapshot / owned-value model

PyO3 exposes Rust objects to Python via two broad strategies:

1. **Borrow / Arc-shared** — Python holds a reference-counted reference to a
   Rust object that lives elsewhere (often via `PyClass` + `Arc`). GC of the
   Python object decrements the refcount; the underlying Rust object is
   freed when the last reference drops. This is efficient but introduces a
   coupling: if the model is dropped first, Python references become
   dangling (use-after-free).

2. **Snapshot / owned-value** — each call into the Rust API returns a fresh
   Python object whose fields are clones of the Rust state. The Python object
   has **no** reference back into the model. This is the strategy used by
   Fluxion's issue #1812 bindings (and matches the pattern established in
   PR #1795 / #1797 for 9R4C and HVAC config).

The snapshot strategy trades a small per-call copy cost for **strict
memory safety** — the same trade-off as e.g. Pandas `.copy()` or a typical
ORM's `.to_dict()`.

## What the bindings copy, and what they don't

| Binding      | Strategy | Notes |
|--------------|----------|-------|
| `Model.zones()`       | snapshot | copies `num_zones` × (zone metadata + per-zone surfaces) |
| `Model.surfaces()`    | snapshot | copies `num_zones` × `surfaces_per_zone` `WallSurface` records |
| `Model.hvac_system()` | snapshot | copies `HVACSystem` fields (heating/cooling capacity, COP, etc.) |
| `Surface.append_shading(...)` | local    | mutates the snapshot's `shading_devices` list |
| `Surface.add_overhang(...)`    | local    | mutates the snapshot's overhang shorthand fields |
| `Model.set_surfaces(snapshots)` | commit | replaces `model.surfaces` (clones data back) |
| `Model.set_hvac_system(snap)`   | commit | updates heating/cooling capacity in model |

## Reference / canonical pattern

```python
import fluxion

model = fluxion.Model(num_zones=3)

# 1. READ: take owned snapshots
zones = model.zones()
surfaces = model.surfaces()

# 2. MUTATE on snapshots — model is unchanged
for s in surfaces:
    if s.orientation == fluxion.Orientation.South:
        s.add_overhang(depth=1.0, height=2.5)

# 3. COMMIT: push snapshots back to the model
model.set_surfaces(surfaces)

# 4. VERIFY
assert model.surfaces()[0].overhang_depth == 1.0
```

## Why not hold an Arc reference into the model?

A borrow-based design (where Python holds a `Py<Model>` or `Arc<ThermalModel>`
inside each `Zone` / `Surface` PyClass) was considered. It was rejected for
the following reasons:

1. **GC ordering hazards.** If the `Model` is GC'd before a `Zone`, every
   `Zone` would have to either (a) keep a strong reference to the model —
   preventing its deallocation and creating a memory leak — or (b) hold a
   weak reference, requiring every access to first check that the model is
   still alive and raising a Python error otherwise. Both are bad UX.

2. **Mutex contention.** Borrowing across the Python boundary forces either a
   `Mutex<ThermalModel>` (which serializes all concurrent accesses) or a
   `RwLock` (which adds runtime overhead on every read). The snapshot
   strategy sidesteps the lock entirely: read paths copy the data and
   release the borrow before returning.

3. **Iterators and slicing.** Returning `Vec<PyZone>` (a Python list) is a
   natural Python idiom. Iterating with `for z in model.zones()` works
   because CPython's list iterator protocol handles it; no custom
   `__iter__` / `__next__` plumbing is needed.

4. **Consistency with PRs #1795 / #1797.** The 9R4C solver and HVAC config
   bindings use the snapshot pattern. Deviating from that pattern for #1812
   would make the codebase's PyO3 conventions inconsistent.

## Iteration semantics

`model.zones()` and `model.surfaces()` return Python `list` objects. The
standard list iterator protocol applies:

```python
for z in model.zones():       # works out of the box
    print(z.index, z.temperature)

south = [s for s in model.surfaces() if s.orientation == fluxion.Orientation.South]
```

We deliberately do not implement custom `__iter__` / `__next__` methods on
the snapshot types — list iteration is the standard Python way, and adding a
custom iterator would add complexity without changing user-visible
semantics.

## Memory safety verification

The following invariants are exercised by
`tests/python/test_model_mutations.py::TestMemorySafety`:

| Test | Invariant |
|------|-----------|
| `test_gc_zone_does_not_invalidate_model` | GC of a Zone snapshot does not invalidate the parent model |
| `test_holding_snapshot_during_simulation` | Two consecutive snapshots are independent (no aliasing) |
| `test_surface_snapshot_independent_of_model_mutation` | Snapshot mutation does not propagate to the underlying model |
| `test_repeated_snapshots_stable` | Repeated `model.zones()` / `model.surfaces()` calls return deterministic data |
| `test_gc_many_surfaces_no_crash` | Many GC cycles of surface snapshots do not crash the interpreter |

## Reference bindings (issues #1795 / #1797)

The same snapshot pattern is used by:

- `PyThermalMassNode`, `PyMultiNodeThermalMass`, `PyMassAirCouplingMode`,
  `PySurfaceExteriorTemperatures`, `PyMultiNodeSolver` (issue #1795,
  9R4C solver).
- `PyZoneSetpoints`, `PyZoneControl`, `PyDailySchedule`, `PyHVACSchedule`
  (issue #1797, HVAC config).

The new `PyZone`, `PySurface`, `PyMaterial`, `PyHVACSystem`,
`PyShadingDevice`, `PyOrientation`, `PyShadingType` types added by issue
#1812 follow the same conventions.

## Issue references

- **#1812** — Phase 2 (Python Measure API) of the Hybrid Measure Approach.
  Builds on #1795 (9R4C bindings) and #1797 (HVAC config bindings).
- **#1795** — Initial PyO3 binding pattern for the 9R4C solver.
- **#1797** — HVAC schedule / setpoint bindings.
- **#1031** — Original `Model` and `BatchOracle` runner bindings.
- **#782**  — Initial PyO3 surface.
