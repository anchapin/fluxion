# Node.js/NAPI Bindings Implementation

## Overview

This document describes the implementation of Node.js/NAPI bindings for Fluxion, enabling high-performance building energy modeling in JavaScript/TypeScript applications.

## Architecture

### Framework Choice: NAPI-RS

We selected **napi-rs** over raw NAPI for several reasons:

1. **Type Safety**: Generates TypeScript definitions automatically
2. **Zero-Cost Abstractions**: Minimal overhead over raw NAPI
3. **Cross-Platform**: Simplifies cross-compilation for macOS, Linux, Windows
4. **Developer Experience**: Familiar Rust procedural macros
5. **Performance**: ~2x faster than Python bindings for ONNX workloads

### Performance Comparison

| Method | Throughput | Latency | Memory |
|---------|------------|---------|--------|
| Python (PyO3) | ~1,000 configs/sec | ~200ms | Moderate |
| Node.js (NAPI) | ~2,000 configs/sec | ~100ms | Low |
| Node.js (GPU Surrogates) | ~10,000+ configs/sec | ~50ms | Optimized |

## Implementation Details

### Core Components

#### 1. Rust Bindings (`src/napi/`)

- **mod.rs**: Main NAPI module registration and exports
- **batch_oracle.rs**: Bindings for `BatchOracle::evaluate_population()`
- **building_parameters.rs**: Type-safe parameter wrapper with validation
- **error.rs**: JavaScript-accessible error classes

#### 2. Node.js Package (`npm/`)

- **package.json**: NPM package configuration with napi-rs integration
- **index.js**: JavaScript wrapper and convenience functions
- **index.d.ts**: Complete TypeScript type definitions
- **build.js**: Build script for cross-platform compilation
- **example.js**: Comprehensive usage examples
- **test.js**: Test suite using Node.js native test runner

### Feature Flag Integration

The NAPI bindings use a Cargo feature flag:

```toml
[features]
napi-bindings = ["dep:napi", "dep:napi-derive", "dep:napi-build"]
```

This allows the bindings to be optional, keeping the pure Rust core lightweight.

### API Surface Design

The API follows JavaScript/TypeScript conventions while maintaining Rust performance:

```typescript
// Type-safe parameter creation
const params = new BuildingParameters(1.5, 20.0, 24.0);

// High-throughput evaluation
const oracle = new BatchOracle();
const results = oracle.evaluatePopulation([
  params.toVec(),
  [2.0, 20.0, 24.0],
  [2.5, 20.0, 24.0]
], false);
```

### One-Call Simulation: `runSimulation` (issue #3306)

`runSimulation(options)` is the convenience wrapper that issue #3282's Node
example assumed: it constructs a `StateExtractor` with the thermal selector,
calls `configure`, runs the simulation, and returns plain serializable data.

```javascript
const { runSimulation } = require('@fluxion/native');

const baseline = runSimulation({ years: 1 });
console.log(baseline.zoneSolver); // 'gauge'

const legacy = runSimulation({ years: 1, zoneSolver: '5r1c' });
console.log(legacy.zoneSolver); // '5r1c'
console.log(legacy.zoneTemperatures.length); // 8760 (per simulated year)
```

**Options**

| Option             | Type      | Default      | Description                                                              |
|--------------------|-----------|--------------|--------------------------------------------------------------------------|
| `years`            | `number`  | `1`          | Years to simulate (8760 hourly timesteps per year)                       |
| `zoneSolver`       | `string`  | `'gauge'`    | `'gauge'` \| `'5r1c'` \| `'9r4c'`; case-insensitive                      |
| `conductionSolver` | `string`  | `'default'`  | `'default'` \| `'ctf'` \| `'fd'`; case-insensitive                       |
| `useSurrogates`    | `boolean` | `false`      | Use AI surrogates for faster evaluation when available                   |
| `caseSpec`         | —         | —            | Reserved; rejected — the native `StateExtractor` runs the ASHRAE 600 baseline only |
| `schema`           | —         | —            | Reserved; rejected — same as `caseSpec`                                  |

**Selector wiring and observability.** The `zoneSolver` / `conductionSolver`
options are forwarded to `new StateExtractor({ zoneSolver, conductionSolver })`,
so the run uses the same shared Rust selector (`parse_zone_solver` /
`parse_conduction_solver`, `ThermalModel::from_spec_with_selector`) as the
Python binding and CLI. The returned object echoes the effective lowercase
labels (`zoneSolver`, `conductionSolver`); a returned `zoneSolver` of
`'5r1c'` guarantees the Rust parser accepted the identifier and built the
model with it — construction would have thrown otherwise.

**Experimental gate.** Experimental identifiers (`'6r2c'`, `'8r3c'`) are
rejected without the `FLUXION_EXPERIMENTAL_ZONE_SOLVERS=1` environment
variable, reusing the exact shared gate message from the Rust parser. Even
with the env var set they stay unavailable until the
`fluxion-experimental-zone-solvers` cargo feature ships (issue #3291);
unknown values surface the Rust vocabulary error verbatim.

**Return shape.** Plain serializable data (no class instances):
`{ years, timesteps, zoneSolver, conductionSolver, useSurrogates,
zoneTemperatures[], massTemperatures[], heatingLoads[], coolingLoads[],
solarGains[] }`, where the temperature/load/gain arrays are per-timestep
plain `number[]` arrays (`zoneTemperatures` has `years * 8760` entries for
the single-zone baseline case).

### Error Handling

Four distinct error types map to Rust error types:

- `FluxionError`: Base error class
- `ValidationError`: Parameter validation failures
- `SimulationError`: Physics simulation failures
- `SurrogateError`: AI surrogate evaluation failures

## Cross-Platform Support

### Supported Platforms

- **macOS**: x64 (Intel) and ARM64 (Apple Silicon)
- **Linux**: x64
- **Windows**: x64

### Build Configuration

The `package.json` includes napi-rs configuration:

```json
{
  "napi": {
    "name": "fluxion",
    "triples": {
      "defaults": true,
      "additional": [
        "x86_64-pc-windows-msvc",
        "aarch64-pc-windows-msvc",
        "x86_64-apple-darwin",
        "aarch64-apple-darwin",
        "x86_64-unknown-linux-gnu"
      ]
    }
  }
}
```

### Building for Different Platforms

```bash
# Build for current platform
npm run build

# Build for all platforms
npm run artifacts

# Build specific platform
napi build --target aarch64-apple-darwin
```

## Performance Characteristics

### Throughput Analysis

- **Physics-Based**: ~1,000 configs/sec on 8-core CPU
- **AI-Accelerated**: ~10,000+ configs/sec with GPU surrogates
- **Memory Efficiency**: Minimal allocations via CTA buffer reuse
- **Parallel Execution**: Preserves Rust multi-threading via Rayon

### Optimization Strategies

1. **Zero-Copy Data Transfer**: Where possible, arrays are passed by reference
2. **Thread Pool Reuse**: Maintains worker threads across calls
3. **Batch Processing**: Optimal GPU utilization for surrogate inference
4. **Validation Early**: Parameter validation filters invalid configs upfront

## Integration with BIM Ecosystem

### Autodesk Revit

```javascript
const { BatchOracle } = require('@fluxion/native');

function evaluateRevitDesign(revitParams) {
  const oracle = new BatchOracle();
  const population = convertRevitParamsToFluxion(revitParams);
  return oracle.evaluatePopulation(population, false);
}
```

### Speckle

```javascript
const { BatchOracle } = require('@fluxion/native');

function evaluateSpeckleModel(speckleData) {
  const oracle = new BatchOracle();
  const population = extractParametersFromSpeckle(speckleData);
  return oracle.evaluatePopulation(population, true);
}
```

### Trimble SketchUp

```javascript
const { BatchOracle } = require('@fluxion/native');

function evaluateSketchupModel(sketchupParams) {
  const oracle = new BatchOracle();
  const population = convertSketchupParams(sketchupParams);
  return oracle.evaluatePopulation(population, false);
}
```

## Testing

### Test Coverage

The implementation includes comprehensive tests:

- **Unit Tests**: Individual function validation
- **Integration Tests**: End-to-end workflows
- **Performance Tests**: Throughput and latency benchmarks
- **Error Handling**: Proper error propagation

### Running Tests

```bash
cd npm
npm test
```

## Future Enhancements

### Planned Features

1. **Streaming API**: For real-time evaluation of streaming parameters
2. **Web Workers**: Better browser integration via web workers
3. **Async API**: Non-blocking evaluation with promises
4. **GPU Control**: Explicit GPU/CPU selection for surrogates
5. **Advanced Metrics**: Additional output beyond EUI values

### Potential Optimizations

1. **WebAssembly**: Browser-only version via WASM
2. **Edge Computing**: Cloudflare Workers / Vercel Edge support
3. **Streaming Results**: Return results as they become available
4. **Caching**: Cache frequently evaluated configurations

## Comparison with Python Bindings

| Aspect | Python (PyO3) | Node.js (NAPI) |
|--------|----------------|-----------------|
| **Performance** | 1x baseline | ~2x faster |
| **Type Safety** | Runtime (optional) | Compile-time (TS) |
| **Ecosystem** | Scientific computing | Web/BIM tools |
| **Memory** | GIL overhead | Direct access |
| **Bundle Size** | Large (Python runtime) | Smaller (V8 only) |

## Troubleshooting

### Build Failures

If `npm run build` fails:

1. Ensure Node.js >= 18 is installed
2. Install build tools: `npm install -g @napi-rs/cli`
3. Check Rust toolchain: `rustc --version`
4. Verify target toolchains: `rustup target list --installed`

### Runtime Errors

Common issues and solutions:

- **"Cannot find module"**: Run `npm run build` first
- **"Segmentation fault"**: Check parameter ranges and data types
- **"Out of memory"**: Reduce population size or enable surrogates

## References

- [napi-rs Documentation](https://napi.rs/)
- [Node.js Native Addons](https://nodejs.org/api/n-api.html)
- [Fluxion Core API](docs/API_REFERENCE.md)
- [Python Bindings Implementation](pyproject.toml)

## Contributing

To contribute to the Node.js bindings:

1. Ensure Rust code follows existing conventions
2. Update TypeScript definitions when adding features
3. Add tests for new functionality
4. Update documentation for API changes
5. Test on multiple platforms when possible

## License

Apache-2.0 - Same as Fluxion core library.
