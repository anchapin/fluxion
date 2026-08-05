# fluxion-wasm

WebAssembly bindings for [Fluxion](https://github.com/anchapin/fluxion), a Rust-based building energy modeling engine.

Enables full acausal building energy simulations to run client-side in web browsers, CAD software, and web-based BIM tools via `wasm-pack`.

## Status

**Issue**: [#1996](https://github.com/anchapin/fluxion/issues/1996)

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
const zoneTemps = sim.getZoneTemps();
console.log(zoneTemps);  // [21.2, 22.1, 23.0, 20.8, 22.5]

// Set a heating setpoint for zone 0
sim.setControl('heating_zone_0', 21.0);

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
| `getZoneTemps()` | Returns `Float64Array` of zone temperatures (°C) |
| `getZoneTemp(zoneId)` | Get temperature for zone `zoneId` |
| `setTemperatures(temps)` | Set all zone temperatures (must match `numZones`) |
| `setControl(loopId, setpoint)` | Set a control loop setpoint |
| `getControl(loopId)` | Get a control loop setpoint |
| `numZones()` | Number of thermal zones |
| `getHeatingSetpoints()` | Heating setpoints per zone |
| `getCoolingSetpoints()` | Cooling setpoints per zone |
| `resetTemperatures(temperature)` | Reset all zone temperatures |
| `currentHour()` | Current simulation time in hours |
| `mode()` | Current execution mode (always `"Physics"`) |
| `setMode(mode)` | Set mode (no-op, WASM only supports Physics) |
| `applyParameters(params)` | Apply optimization gene vector (window U-value, heating/cooling setpoints) |
| `zoneArea()` | Zone floor area in m² |
| `hvacPowerDemand(timestep, outdoorTemp)` | HVAC heating (+) or cooling (−) power in W |
| `isValid()` | Returns `true` if simulation state is valid |
| `solveTimesteps(steps, useSurrogates)` | Stub — returns `0.0` (ONNX unavailable in WASM) |

## Configuration

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

## WASM-Compatible Subset

Per the [WASM compatibility analysis](../fluxion-fluid/WASM_STATUS.md):

- **fluxion-core**: Fully WASM-compatible
- **fluxion-fluid/ports**: Fully WASM-compatible
- **fluxion-fluid/graph**: Fully WASM-compatible
- **fluxion-fluid/solvers**: WASM-compatible with sequential fallback (no rayon)
- **ort/ONNX inference**: NOT WASM-compatible (requires co-processing)

## License

Apache-2.0
