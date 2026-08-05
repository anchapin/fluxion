# fluxion-wasm WASM Status

**Created:** 2026-08-04
**Issue:** #2380

## Executive Summary

`fluxion-wasm` v1.0.0 provides WebAssembly bindings for building energy simulation via `wasm-bindgen`. The crate exposes a `FluidSimulation` wrapper that wraps `fluxion-fluid` types and provides a simplified energy balance model suitable for browser-based simulation.

**Build status:** `cargo check -p fluxion-wasm` ✅ SUCCEEDS
**WASM pack status:** `wasm-pack build -p fluxion-wasm` ✅ SUCCEEDS

## API Surface

### FluidSimulation (wasm_bindgen)

| Method | ThermalModelTrait equivalent | Status | Notes |
|--------|------------------------------|--------|-------|
| `new(configJson)` | — | ✅ Complete | JSON-configured constructor |
| `step(dtHours)` | `solve_timesteps` | ✅ Complete | Simplified energy balance step |
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
| `hvac_power_demand(timestep, outdoor_temp)` | `hvac_power_demand()` | ✅ Complete | Simplified power calculation |
| `is_valid()` | `is_valid()` | ✅ Complete | Validates zone count, setpoints |
| `solve_timesteps(steps, useSurrogates)` | `solve_timesteps()` | ⚠️ Stub | Always returns 0.0 (ONNX unavailable in WASM) |

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
2. **Simplified energy balance** — `FluidSimulation` uses a lumped-capacitance model, not the full 5R1C/9R4C thermal network.
3. **No multi-threading** — WASM runs in a single-threaded environment.
4. **No `rayon` parallelism** — parallel population evaluation in `BatchOracle` is unavailable.

## Planned Enhancements

- [ ] WASM-native 5R1C thermal network solver (替代 simplified model)
- [ ] `wasm-bindgen-threads` support for parallel zone evaluation
- [ ] WebGPU fallback for simple ML inference
- [ ] Browser integration test via Playwright

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
