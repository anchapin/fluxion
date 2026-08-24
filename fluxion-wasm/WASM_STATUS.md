# fluxion-wasm WASM Status

**Created:** 2026-08-04
**Issue:** #2380
**Enhanced:** 2026-08-24 (Issue #3181)

## Executive Summary

`fluxion-wasm` v1.1.0 provides WebAssembly bindings for building energy simulation via `wasm-bindgen`. The crate exposes a `FluidSimulation` wrapper that wraps `fluxion-fluid` types and provides an enhanced energy balance model with per-zone thermal parameters suitable for browser-based simulation.

**Build status:** `cargo check -p fluxion-wasm` ✅ SUCCEEDS
**WASM pack status:** `wasm-pack build -p fluxion-wasm` ✅ SUCCEEDS

## Enhanced API Surface (v1.1.0)

### FluidSimulation (wasm_bindgen)

| Method | ThermalModelTrait equivalent | Status | Notes |
|--------|------------------------------|--------|-------|
| `new(configJson)` | — | ✅ Complete | JSON-configured constructor |
| `step(dtHours)` | `solve_timesteps` | ✅ Complete | Enhanced energy balance with per-zone thermal params |
| `get_zone_temps()` | `get_temperatures()` | ✅ Complete | Returns `Vec<f64>` → `Float64Array` |
| `get_zone_temp(zoneId)` | `get_temperatures()[i]` | ✅ Complete | Bounds-checked single zone |
| `set_temperatures(temps)` | `set_temperatures()` | ✅ Complete | Bulk temperature initializer |
| `set_control(loopId, setpoint)` | heating/cooling setpoints | ✅ Complete | `heating_zone_N`, `cooling_zone_N` |
| `get_control(loopId)` | — | ✅ Complete | Custom control loop lookup |
| `num_zones()` | `num_zones()` | ✅ Complete | |
| `get_heating_setpoints()` | `heating_setpoint()` | ✅ Complete | Returns all zone setpoints |
| `get_cooling_setpoints()` | `cooling_setpoint()` | ✅ Complete | Returns all zone setpoints |
| `reset_temperatures(t)` | — | ✅ Complete | Bulk temperature reset |
| `current_hour()` | — | ✅ Complete | Simulation time tracker |
| `mode()` | `mode()` | ✅ Complete | Always returns `"Physics"` |
| `set_mode(mode)` | `set_mode()` | ✅ Complete | No-op (Physics-only) |
| `apply_parameters(params)` | `apply_parameters()` | ✅ Complete | Window U-value, heating/cooling SP |
| `zone_area()` | `zone_area()` | ✅ Complete | Returns configured zone area |
| `hvac_power_demand(timestep, outdoor_temp)` | `hvac_power_demand()` | ✅ Complete | Enhanced with per-zone thermal params |
| `is_valid()` | `is_valid()` | ✅ Complete | Validates zone count, setpoints |
| `solve_timesteps(steps, useSurrogates)` | `solve_timesteps()` | ⚠️ Stub | Always returns 0.0 (ONNX unavailable in WASM) |

### New Zone Parameter API (v1.1.0 - Issue #3181)

| Method | Description | Status |
|--------|-------------|--------|
| `get_zone_thermal_mass(zoneId)` | Get zone thermal mass in J/K | ✅ Complete |
| `set_zone_thermal_mass(zoneId, thermalMass)` | Set zone thermal mass | ✅ Complete |
| `get_all_thermal_masses()` | Get all zone thermal masses | ✅ Complete |
| `get_zone_conductance(zoneId)` | Get zone conductance in W/K | ✅ Complete |
| `set_zone_conductance(zoneId, conductance)` | Set zone conductance | ✅ Complete |
| `get_all_conductances()` | Get all zone conductances | ✅ Complete |
| `get_zone_infiltration(zoneId)` | Get zone infiltration in kg/s | ✅ Complete |
| `set_zone_infiltration(zoneId, ach)` | Set zone infiltration in ACH | ✅ Complete |
| `get_all_infiltration()` | Get all infiltration rates | ✅ Complete |
| `get_zone_internal_gains(zoneId)` | Get zone internal gains in W | ✅ Complete |
| `set_zone_internal_gains(zoneId, gainsW)` | Set zone internal gains | ✅ Complete |
| `get_all_internal_gains()` | Get all internal gains | ✅ Complete |
| `get_zone_area(zoneId)` | Get zone floor area in m² | ✅ Complete |
| `apply_zone_parameters(paramsJson)` | Apply multiple zone params at once | ✅ Complete |
| `export_state()` | Export full simulation state as JSON | ✅ Complete |
| `load_state(stateJson)` | Load simulation state from JSON | ✅ Complete |

### Exported Types (wasm_bindgen)

| Type | Source | Status |
|------|--------|--------|
| `FluidSimulation` | `fluxion-wasm` | ✅ Exported |
| `FluidSimulationConfig` | `fluxion-wasm` | ✅ Exported (serialization only) |
| `Air`, `Water`, `Medium` | `fluxion-fluid::mediums` | ✅ Re-exported |
| `AirPort`, `HydronicPort`, `BoundaryConditions` | `fluxion-fluid::ports` | ✅ Re-exported |

## WASM Compatibility

Per `fluxion-fluid/WASM_STATUS.md`:

| Module | WASM Status | Notes |
|--------|-------------|-------|
| `fluxion-fluid::mediums` | ✅ Compatible | Pure Rust, no platform code |
| `fluxion-fluid::ports` | ✅ Compatible | Strongly typed port traits |
| `fluxion-fluid::graph` | ✅ Compatible | petgraph with `alloc` feature |
| `fluxion-fluid::ecs` | ⚠️ Sequential | Uses `faer-rs` without rayon |
| `fluxion-ai` (ONNX) | ❌ Incompatible | Requires `ort`/GPU backend |
| `fluxion-core` | ✅ Compatible | Weather, assembly, multi_node |

## Memory Management

`wasm-bindgen` handles memory transfer automatically:

- **`Vec<f64>`** → `Float64Array` (JS-side zero-copy view, no heap allocation)
- **`String`** → `JSString` (automatic conversion)
- **`Option<T>`** → `T | null` (JS null maps to `None`)
- **Ownership**: `wasm-bindgen` uses reference counting; all exported types are `'static`

No explicit `free()` calls needed — the JS garbage collector handles cleanup.

## Limitations

1. **No ONNX surrogate inference** — `ort` is not WASM-compatible. The `solve_timesteps()` method returns `0.0` as a stub.
2. **Enhanced lumped-capacitance model** — `FluidSimulation` uses per-zone thermal parameters but is not a full 5R1C/9R4C thermal network.
3. **No multi-threading** — WASM runs in a single-threaded environment.
4. **No `rayon` parallelism** — parallel population evaluation in `BatchOracle` is unavailable.

## Enhanced Configuration (v1.1.0)

The JSON configuration now supports per-zone thermal parameters:

```json
{
  "building": "5_zone_office",
  "num_zones": 5,
  "weather": "TMY3_CHICAGO",
  "initial_temps": [22.0, 22.0, 22.0, 22.0, 22.0],
  "heating_setpoint": 20.0,
  "cooling_setpoint": 24.0,
  "zone_areas": [50.0, 50.0, 50.0, 50.0, 50.0],
  "zone_thermal_mass": [5e6, 5e6, 5e6, 5e6, 5e6],
  "zone_conductance": [50.0, 50.0, 50.0, 50.0, 50.0],
  "infiltration_ach": [0.5, 0.5, 0.5, 0.5, 0.5],
  "internal_gains_w": [200.0, 200.0, 200.0, 200.0, 200.0]
}
```

## Planned Enhancements

- [x] Enhanced WASM API surface with per-zone thermal parameters (Issue #3181)
- [x] JSON configuration for building specs (Issue #3181)
- [x] Working browser demo page (Issue #3181)
- [ ] WASM-native 5R1C thermal network solver
- [ ] `wasm-bindgen-threads` support for parallel zone evaluation
- [ ] WebGPU fallback for simple ML inference
- [ ] Browser integration test via Playwright
- [ ] IndexedDB persistence for building model persistence

## Verification Commands

```bash
# Native check
cargo check -p fluxion-wasm

# WASM build (Node.js target)
wasm-pack build --target nodejs -p fluxion-wasm

# WASM build (web target)
wasm-pack build --target web -p fluxion-wasm

# Run integration tests
cargo test wasm_integration

# Format and lint
cargo fmt -- --check
cargo clippy --lib -- -D warnings
```

## References

- Issue #2380: WebAssembly Bindings Completion (fluxion-wasm)
- Issue #1996: WASM build scaffolding
- Issue #1998: fluxion-fluid WASM compatibility analysis
- `fluxion-fluid/WASM_STATUS.md`: Detailed dependency compatibility matrix
