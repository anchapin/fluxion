# @fluxion/native

High-performance native Node.js bindings for Fluxion Building Energy Modeling engine.

## Overview

`@fluxion/native` provides JavaScript/TypeScript bindings to Fluxion's Rust-based building energy modeling engine, enabling >10,000 building configurations/second evaluation throughput for parametric analysis and optimization workflows.

## Features

- **🚀 High Performance**: 10,000+ configs/sec throughput (2x faster than Python)
- **🔧 Type Safety**: Full TypeScript support with comprehensive type definitions
- **🎯 Easy Integration**: Drop-in replacement for existing building energy workflows
- **⚡ AI-Accelerated**: Optional neural network surrogates for 10x faster evaluation
- **🌐 Cross-Platform**: Supports macOS (x64 + ARM), Linux, Windows

## Installation

```bash
npm install @fluxion/native
```

## Quick Start

```javascript
const { BatchOracle, BuildingParameters } = require('@fluxion/native');

// Create oracle instance
const oracle = new BatchOracle();

// Define building parameters
const params = new BuildingParameters(1.5, 20.0, 24.0);

// Evaluate population (high-throughput optimization)
const population = [
  [1.5, 20.0, 24.0], // Config 1: U=1.5, Heat=20, Cool=24
  [2.0, 20.0, 24.0], // Config 2: U=2.0, Heat=20, Cool=24
  [2.5, 20.0, 24.0], // Config 3: U=2.5, Heat=20, Cool=24
];

// Evaluate with physics-based calculation
const results = oracle.evaluatePopulation(population, false);
console.log(`EUI values: ${results}`); // [120.5, 115.2, 110.8] kWh/m²/yr

// Evaluate with AI-accelerated surrogates (~10x faster)
const aiResults = oracle.evaluatePopulation(population, true);
console.log(`EUI values (AI): ${aiResults}`);
```

## TypeScript Usage

```typescript
import { BatchOracle, BuildingParameters, ValidationError } from '@fluxion/native';

async function optimizeBuildingDesign() {
  const oracle = new BatchOracle();

  // Parameter space for optimization
  const uValues = [1.5, 2.0, 2.5, 3.0, 3.5];
  const heatingSetpoints = [18.0, 20.0, 22.0];
  const coolingSetpoints = [22.0, 24.0, 26.0];

  // Generate population
  const population: number[][] = [];
  for (const uValue of uValues) {
    for (const heating of heatingSetpoints) {
      for (const cooling of coolingSetpoints) {
        try {
          // Validate parameters
          oracle.validateParameters([uValue, heating, cooling]);
          population.push([uValue, heating, cooling]);
        } catch (error) {
          if (error instanceof ValidationError) {
            console.log(`Skipping invalid params: ${error.message}`);
          }
        }
      }
    }
  }

  // Evaluate entire population (~10,000+ configs/sec)
  const euiValues = oracle.evaluatePopulation(population, true);

  // Find optimal configuration (minimum EUI)
  const minEui = Math.min(...euiValues);
  const optimalIndex = euiValues.indexOf(minEui);
  const optimalParams = population[optimalIndex];

  console.log(`Optimal EUI: ${minEui.toFixed(2)} kWh/m²/yr`);
  console.log(`Parameters: U=${optimalParams[0]}, Heat=${optimalParams[1]}, Cool=${optimalParams[2]}`);
}

optimizeBuildingDesign();
```

## API Reference

### BatchOracle

High-throughput building energy evaluation for optimization workflows.

#### Constructor

```typescript
new BatchOracle()
```

Creates a new BatchOracle instance with default ASHRAE 600 configuration.

#### Methods

##### `evaluatePopulation(population, useSurrogates)`

Evaluate a population of building design configurations in parallel.

- **Parameters:**
  - `population: number[][]` - Array of parameter arrays. Each inner array contains:
    - `[0]`: Window U-value (W/m²K, range: 0.1-5.0)
    - `[1]`: Heating setpoint (°C, range: 15-25)
    - `[2]`: Cooling setpoint (°C, range: 22-32)
  - `useSurrogates: boolean` - Use AI surrogates (true) or physics-based (false)

- **Returns:** `number[]` - Array of EUI values (kWh/m²/yr) for each candidate

- **Throws:**
  - `ValidationError` - If parameters are out of valid ranges
  - `SimulationError` - If physics simulation fails
  - `SurrogateError` - If AI surrogate evaluation fails

##### `validateParameters(params)`

Validate building parameters against physical constraints.

- **Parameters:**
  - `params: number[]` - Parameter array containing window U-value, heating/cooling setpoints

- **Throws:**
  - `ValidationError` - If parameters are out of valid ranges

### BuildingParameters

Type-safe building parameters with validation.

#### Constructor

```typescript
new BuildingParameters(windowUValue, heatingSetpoint, coolingSetpoint)
```

Creates validated building parameters.

- **Parameters:**
  - `windowUValue: number` - Window U-value (0.1-5.0 W/m²K)
  - `heatingSetpoint: number` - Heating setpoint (15.0-25.0 °C)
  - `coolingSetpoint: number` - Cooling setpoint (22.0-32.0 °C)

- **Throws:**
  - `ValidationError` - If parameters are out of valid ranges

#### Properties

- `windowUValue: number` - Window U-value (W/m²K)
- `heatingSetpoint: number` - Heating setpoint (°C)
- `coolingSetpoint: number` - Cooling setpoint (°C)

#### Methods

##### `toVec()`

Convert parameters to array for backward compatibility.

- **Returns:** `number[]` - Array in format `[window_u_value, heating_setpoint, cooling_setpoint]`

## Error Handling

```typescript
import { BatchOracle, ValidationError, SimulationError } from '@fluxion/native';

try {
  const oracle = new BatchOracle();
  const results = oracle.evaluatePopulation(population, false);
} catch (error) {
  if (error instanceof ValidationError) {
    console.error('Invalid parameters:', error.message);
  } else if (error instanceof SimulationError) {
    console.error('Simulation failed:', error.message);
  } else {
    throw error; // Re-throw unknown errors
  }
}
```

## Performance

- **Physics-based**: ~1,000 configs/sec on 8-core CPU
- **AI-accelerated**: ~10,000+ configs/sec with GPU surrogates
- **Latency**: <100ms for single configuration (8760 timesteps)
- **Memory**: Minimal allocations via CTA buffer reuse

## Cross-Platform Support

- **macOS**: x64 (Intel) and ARM64 (Apple Silicon)
- **Linux**: x64
- **Windows**: x64

Pre-built binaries are included for all platforms, but you can also build from source:

```bash
npm run build
```

## Integration with BIM Tools

### Autodesk Revit

```javascript
const { BatchOracle } = require('@fluxion/native');

const oracle = new BatchOracle();

// Hook into Revit parameter changes
export async function evaluateRevitDesign(revitParams) {
  const population = convertRevitParamsToFluxion(revitParams);
  const euiValues = oracle.evaluatePopulation(population, false);
  return mapResultsToRevit(euiValues);
}
```

### Speckle

```javascript
const { BatchOracle } = require('@fluxion/native');

const oracle = new BatchOracle();

// Evaluate Speckle building data
export async function evaluateSpeckleModel(speckleData) {
  const population = extractParametersFromSpeckle(speckleData);
  const euiValues = oracle.evaluatePopulation(population, true);
  return addFluxionResultsToSpeckle(speckleData, euiValues);
}
```

### Trimble SketchUp

```javascript
const { BatchOracle } = require('@fluxion/native');

const oracle = new BatchOracle();

// Parametric evaluation in SketchUp
SU.on('parameterChange', async (params) => {
  const fluxionParams = convertSketchupParams(params);
  const eui = oracle.evaluatePopulation([fluxionParams], false)[0];
  updateSketchupUI({ eui });
});
```

## Building from Source

```bash
# Clone the repository
git clone https://github.com/anchapin/fluxion.git
cd fluxion

# Install dependencies
npm install

# Build native module
npm run build

# Run tests
npm test
```

## Contributing

Contributions are welcome! Please see the main [Fluxion repository](https://github.com/anchapin/fluxion) for guidelines.

## License

Apache-2.0 - See [LICENSE](../../LICENSE) for details.

## Support

- **Documentation**: https://fluxion.readthedocs.io
- **Issues**: https://github.com/anchapin/fluxion/issues
- **Discussions**: https://github.com/anchapin/fluxion/discussions

## Acknowledgments

- Built with [napi-rs](https://napi.rs/) for type-safe native bindings
- Powered by Fluxion's Rust-based building energy modeling engine
- Compatible with EnergyPlus OpenStudio SDK workflows
