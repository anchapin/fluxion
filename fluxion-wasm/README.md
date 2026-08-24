# fluxion-wasm

WebAssembly bindings for [Fluxion](https://github.com/anchapin/fluxion), a Rust-based building energy modeling engine.

Enables full acausal building energy simulations to run client-side in web browsers, CAD software, and web-based BIM tools via `wasm-pack`.

## Status

**Issue**: [#1996](https://github.com/anchapin/fluxion/issues/1996)
**Enhanced**: [#3181](https://github.com/anchapin/fluxion/issues/3181)

**WASM Compatibility**: See [fluxion-fluid/WASM_STATUS.md](../fluxion-fluid/WASM_STATUS.md) for the current dependency compatibility matrix.

## Building

```bash
# Install wasm-pack if not already installed
cargo install wasm-pack

# Build for the web target
wasm-pack build --target web -p fluxion-wasm

# Build for node.js
wasm-pack build --target nodejs -p fluxion-wasm
```

## Usage

```javascript
import init, { FluidSimulation } from '@fluxion/wasm';

await init();

const sim = new FluidSimulation(JSON.stringify({
  building: '5_zone_office',
  num_zones: 5,
  weather: 'TMY3_CHICAGO',
  heating_setpoint: 20.0,
  cooling_setpoint: 24.0,
}));

// Run a 1-hour timestep
sim.step(1.0);

// Read zone temperatures
const zoneTemps = sim.get_zone_temps();
console.log(zoneTemps);  // [21.2, 22.1, 23.0, 20.8, 22.5]

// Set a heating setpoint for zone 0
sim.set_control('heating_zone_0', 21.0);

// Run annual simulation (8760 hours)
for (let hour = 0; hour < 8760; hour++) {
  sim.step(1.0);
}
```

## API

### `FluidSimulation`

| Method | Description |
|--------|-------------|
| `new(configJson)` | Create simulation from JSON config |
| `step(dtHours)` | Advance simulation by `dtHours` hours |
| `get_zone_temps()` | Returns `Vec<f64>` of zone temperatures (°C) |
| `get_zone_temp(zoneId)` | Get temperature for zone `zoneId` |
| `set_temperatures(temps)` | Set all zone temperatures (must match `numZones`) |
| `set_control(loopId, setpoint)` | Set a control loop setpoint |
| `get_control(loopId)` | Get a control loop setpoint |
| `num_zones()` | Number of thermal zones |
| `get_heating_setpoints()` | Heating setpoints per zone |
| `get_cooling_setpoints()` | Cooling setpoints per zone |
| `reset_temperatures(temperature)` | Reset all zone temperatures |
| `current_hour()` | Current simulation time in hours |
| `mode()` | Current execution mode (always `"Physics"`) |
| `set_mode(mode)` | Set mode (no-op, WASM only supports Physics) |
| `apply_parameters(params)` | Apply optimization gene vector (window U-value, heating/cooling setpoints) |
| `zone_area()` | Zone floor area in m² |
| `hvac_power_demand(timestep, outdoorTemp)` | HVAC heating (+) or cooling (−) power in W |
| `is_valid()` | Returns `true` if simulation state is valid |
| `solve_timesteps(steps, useSurrogates)` | Stub — returns `0.0` (ONNX unavailable in WASM) |

### Zone Parameter API (v1.1.0)

| Method | Description |
|--------|-------------|
| `get_zone_thermal_mass(zoneId)` | Get zone thermal mass in J/K |
| `set_zone_thermal_mass(zoneId, thermalMass)` | Set zone thermal mass |
| `get_all_thermal_masses()` | Get all zone thermal masses |
| `get_zone_conductance(zoneId)` | Get zone conductance in W/K |
| `set_zone_conductance(zoneId, conductance)` | Set zone conductance |
| `get_all_conductances()` | Get all zone conductances |
| `get_zone_infiltration(zoneId)` | Get zone infiltration in kg/s |
| `set_zone_infiltration(zoneId, ach)` | Set zone infiltration in ACH |
| `get_all_infiltration()` | Get all infiltration rates |
| `get_zone_internal_gains(zoneId)` | Get zone internal gains in W |
| `set_zone_internal_gains(zoneId, gainsW)` | Set zone internal gains |
| `get_all_internal_gains()` | Get all internal gains |
| `get_zone_area(zoneId)` | Get zone floor area in m² |
| `apply_zone_parameters(paramsJson)` | Apply multiple zone params at once |
| `export_state()` | Export full simulation state as JSON |
| `load_state(stateJson)` | Load simulation state from JSON |

## Configuration

### Basic Configuration

```json
{
  "building": "5_zone_office",
  "num_zones": 5,
  "weather": "TMY3_CHICAGO",
  "initial_temps": [22.0, 22.0, 22.0, 22.0, 22.0],
  "heating_setpoint": 20.0,
  "cooling_setpoint": 24.0
}
```

### Enhanced Configuration (v1.1.0)

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

### Parameter Ranges

| Parameter | Min | Max | Default | Unit |
|-----------|-----|-----|---------|------|
| Temperature | 10.0 | 40.0 | 22.0 | °C |
| U-value | 0.1 | 10.0 | 2.0 | W/m²K |
| Thermal mass | 1e3 | 1e10 | 5e6 | J/K |
| Conductance | 0.1 | 1e6 | 50.0 | W/K |
| Infiltration ACH | 0.0 | 10.0 | 0.5 | ACH |
| Internal gains | 0.0 | 1e6 | 200.0 | W |

## Browser Demo

A demo page is available at `www/index.html`. To use it:

1. Build the WASM package: `wasm-pack build --target web -p fluxion-wasm`
2. Serve the `www` directory with a static file server
3. Open `index.html` in a browser

The demo provides:
- Interactive zone temperature visualization
- Real-time HVAC power demand display
- Per-zone parameter adjustment
- State export/import functionality
- Performance timing indicators

## Performance

The enhanced thermal model is optimized for interactive use:
- Single timestep (1 zone): < 1ms
- 24-hour simulation (5 zones): < 50ms
- 168-hour simulation (5 zones): < 100ms

Performance targets (< 100ms per timestep) are met for typical configurations.

## WASM-Compatible Subset

Per the [WASM compatibility analysis](../fluxion-fluid/WASM_STATUS.md):

- **fluxion-core**: Fully WASM-compatible
- **fluxion-fluid/ports**: Fully WASM-compatible
- **fluxion-fluid/graph**: Fully WASM-compatible
- **fluxion-fluid/solvers**: WASM-compatible with sequential fallback (no rayon)
- **ort/ONNX inference**: NOT WASM-compatible (requires co-processing)

## License

Apache-2.0
