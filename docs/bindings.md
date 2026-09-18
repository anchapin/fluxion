# Fluxion Bindings — Surface Overview

This document is the **top-level entry point** for every Fluxion bindings
surface. It is the discoverability layer that points readers to the
authoritative per-surface docs (PyO3 contract, NAPI implementation,
`wasm-pack`-built WebAssembly). For the underlying physics/simulation
surface, see [`fluxion-wasm/README.md`](fluxion-wasm/README.md) and the
crate root `fluxion-wasm/`.

## Available surfaces

| Surface | Build / install | Authoritative doc |
|---|---|---|
| **Python** (PyO3 / maturin) | `pip install maturin && maturin develop` | This file's [Python section](#python-pyo3) holds the #1812 memory-safety contract (PyO3 snapshot pattern) |
| **Node.js** (NAPI-RS) | `cd npm && npm install && npm run build` | [`docs/NAPI_BINDINGS.md`](NAPI_BINDINGS.md) |
| **WebAssembly** (`wasm-pack`) | `wasm-pack build --target web -p fluxion-wasm` (or `--target nodejs`) | [`fluxion-wasm/README.md`](../fluxion-wasm/README.md) + status: [`fluxion-wasm/WASM_STATUS.md`](../fluxion-wasm/WASM_STATUS.md) |

Every workspace binding produces a sibling native artifact and is wired
into CI: the WASM build runs as a Lane-2 path-filtered required check
([`wasm-build.yml`](https://github.com/anchapin/fluxion/actions/workflows/wasm-build.yml);
`fluxion-wasm/**`, Cargo.toml/Cargo.lock, release_gates.yaml), Node via
[`node-bindings.yml`](https://github.com/anchapin/fluxion/actions/workflows/node-bindings.yml),
and Python via
[`python-bindings.yml`](https://github.com/anchapin/fluxion/actions/workflows/python-bindings.yml).
The full lane classification is documented in
[`docs/adr/0016-fast-lane-gates-nightly-authority.md`](adr/0016-fast-lane-gates-nightly-authority.md)
(ADR-0016).

## WebAssembly (fluxion-wasm)

WebAssembly bindings let Fluxion run **client-side** in web browsers, CAD
software, and web-based BIM tools — no server round-trip, no native
install. The crate is `fluxion-wasm` (a workspace member); it wraps
`fluxion-fluid` types and exposes an enhanced per-zone lumped-capacitance
model through `wasm-bindgen`. The build command is:

```bash
cargo install wasm-pack                                # one-time
wasm-pack build --target web -p fluxion-wasm          # browser target
wasm-pack build --target nodejs -p fluxion-wasm       # Node.js target
```

### What's exposed

A single `FluidSimulation` class plus a per-zone parameter API (v1.1.0,
Issue #3181) — see [`fluxion-wasm/README.md`](../fluxion-wasm/README.md)
for the full method table. Headlines:

- `new(configJson)` → JSON-configured constructor (building, num_zones,
  weather, setpoints, optional per-zone thermal-mass / conductance /
  infiltration / internal-gains arrays).
- `step(dtHours)` → advance the simulation; when the config names the
  `"ASHRAE_600"` weather preset, `step()` drives the embedded WD600
  8760-hour dry-bulb series (Issue #3624). Unconfigured consumers keep
  the neutral 20 °C fallback.
- `get_zone_temps()` / `get_zone_temp(i)` / `set_temperatures([..])`
  → bulk and per-zone temperature accessors.
- `set_control(loopId, setpoint)` / `get_control(loopId)` →
  `heating_zone_N` / `cooling_zone_N` lookups.
- `apply_parameters({U, heating, cooling})` and
  `apply_zone_parameters(paramsJson)` → optimization gene vectors and
  bulk per-zone edits.
- `hvac_power_demand(timestep, outdoorTemp)` → heating (+) / cooling (−)
  power in W, using the enhanced per-zone thermal parameters.
- `export_state()` / `load_state(stateJson)` → round-trip the full
  simulation state as JSON (useful for replay / save-state UIs).

### What is **not** exposed (yet)

- **No ONNX surrogate inference.** `ort` is not WASM-compatible
  (Issue #1998 / `fluxion-fluid/WASM_STATUS.md`); the
  `solve_timesteps(steps, useSurrogates)` method is a documented stub
  that always returns `0.0`. This is the canonical limitation called out
  in [`fluxion-wasm/WASM_STATUS.md`](../fluxion-wasm/WASM_STATUS.md).
- **No multi-threading.** WASM runs single-threaded; `rayon` parallelism
  inside `BatchOracle` is unavailable.
- **No 5R1C / 9R4C thermal network solver.** `FluidSimulation` is a
  per-zone lumped-capacitance model; a WASM-native 5R1C solver is on
  the roadmap (see Planned Enhancements in WASM_STATUS.md).

### Live status

The dependency-compatibility matrix and the `✅ Complete / ⚠️ Stub / ❌
Incompatible` per-method map live in
[`fluxion-wasm/WASM_STATUS.md`](../fluxion-wasm/WASM_STATUS.md). Treat
that file as the source of truth — this section is the **index**; the
status doc is the **inventory**. When a method is marked "Stub" or
"Incompatible" here, the status doc carries the rationale.

### CI gate (Lane-2 path-filtered, ADR-0016)

`wasm-build.yml` runs the cross-platform build matrix on every PR that
touches the WASM output path (`fluxion-wasm/**`, its `fluxion-fluid` /
`fluxion-core` transitive deps, workspace `Cargo.toml`/`Cargo.lock`,
the workflow itself, and `release_gates.yaml`). Two jobs per PR:

1. `wasm-build` (matrix: ubuntu / macos / windows) — installs
   `wasm-pack` (SHA256-pinned per Issue #3740 / PR #3844), runs
   `wasm-pack build --release --target web`, asserts the `.wasm` size
   ≤ 10 MiB, computes a `wasm-bindgen` interface hash over the generated
   `*.d.ts` + `package.json`, uploads `pkg/` as a per-OS artifact.
2. `interface-stability` — depends on all three matrix rows; re-hashes
   the three artifacts and asserts byte-identical interface hashes.
   A mismatch means a platform-specific flag or env var leaked into the
   JS surface — a silent consumer-facing regression that would
   otherwise only surface at npm-publish time.

Lane classification per
[`docs/adr/0016-fast-lane-gates-nightly-authority.md`](adr/0016-fast-lane-gates-nightly-authority.md):
docs/deps-only changes (e.g. edits to `docs/bindings.md`) do **not**
trigger this gate via the deny-list `paths-ignore`; a change to
`fluxion-wasm/src/lib.rs` or `wasm-build.yml` does.

### Cross-references

- Build + usage + API + browser demo: [`fluxion-wasm/README.md`](../fluxion-wasm/README.md)
- Per-method status matrix + memory model + limitations:
  [`fluxion-wasm/WASM_STATUS.md`](../fluxion-wasm/WASM_STATUS.md)
- Lane model: [`docs/adr/0016-fast-lane-gates-nightly-authority.md`](adr/0016-fast-lane-gates-nightly-authority.md)
- CI workflow: `.github/workflows/wasm-build.yml` (release_gates.yaml
  entry under `ci.workflow_index`)

## Node.js (NAPI-RS)

The Node bindings provide high-performance native access with full
TypeScript definitions, built via `napi-rs`. They share the same
underlying `FluxionModel` / `BatchOracle` surfaces as Python but expose
the FFI as a `*.node` addon. Full implementation detail, performance
benchmarks, and the package layout live in
[`docs/NAPI_BINDINGS.md`](NAPI_BINDINGS.md). Build:

```bash
cd npm
npm install
npm run build
```

## Python (PyO3)

The Python bindings are Fluxion's deepest bindings surface: they expose
the full `FluxionModel`, the `BatchOracle` population evaluator, the
9R4C solver (#1795), HVAC config (#1797), and the Phase-2 Measure API
(#1812 — `Zone`, `Surface`, `Material`, `HVACSystem`, `ShadingDevice`,
`Orientation`). Build:

```bash
pip install maturin
maturin develop
```

The memory-safety / ownership contract for the #1812 surface is
documented below; it is the same snapshot / owned-value pattern used by
the 9R4C and HVAC config bindings.

---

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

- **#3736** — Top-level bindings surface discoverability (this doc).
- **#3624** — WASM `step()` drives the embedded WD600 annual weather
  schedule (`weather: "ASHRAE_600"`), closes the zero-energy smoke gap.
- **#3181** — Enhanced WASM API surface with per-zone thermal parameters.
- **#2380** — WebAssembly Bindings Completion (`fluxion-wasm`).
- **#1996** — WASM build scaffolding (`wasm-pack`).
- **#1998** — `fluxion-fluid` WASM compatibility analysis.
- **#1812** — Phase 2 (Python Measure API) of the Hybrid Measure Approach.
  Builds on #1795 (9R4C bindings) and #1797 (HVAC config bindings).
- **#1795** — Initial PyO3 binding pattern for the 9R4C solver.
- **#1797** — HVAC schedule / setpoint bindings.
- **#1031** — Original `Model` and `BatchOracle` runner bindings.
- **#782**  — Initial PyO3 surface.
