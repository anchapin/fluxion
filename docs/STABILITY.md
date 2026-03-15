# Fluxion Stability Guarantees

This document describes Fluxion's stability guarantees, including input validation, failure modes, error recovery strategies, and determinism guarantees.

## Input Validation

Fluxion performs comprehensive input validation on all parameters to prevent invalid simulations from running.

### Parameter Bounds

| Parameter | Minimum | Maximum | Units |
|-----------|---------|---------|-------|
| Window U-value | 0.1 | 5.0 | W/m²K |
| Heating setpoint | 15.0 | 25.0 | °C |
| Cooling setpoint | 22.0 | 32.0 | °C |

### Validation Rules

1. **Window U-value**: Must be in range [0.1, 5.0] W/m²K
   - Single glass: ~5.0 W/m²K
   - Double pane: ~2.7 W/m²K
   - Triple pane low-E: ~0.1-0.3 W/m²K

2. **Heating setpoint**: Must be in range [15.0, 25.0] °C and less than cooling setpoint

3. **Cooling setpoint**: Must be in range [22.0, 32.0] °C and greater than heating setpoint

4. **No NaN/Infinity**: All parameters must be finite numbers

### ParameterBounds Structure

The `ParameterBounds` struct provides programmatic access to validation rules:

```python
from fluxion import BatchOracle

oracle = BatchOracle()
bounds = oracle.get_parameter_bounds()

print(bounds.min_u_value)   # 0.1
print(bounds.max_u_value)   # 5.0
print(bounds.min_setpoint)  # 15.0
print(bounds.max_setpoint)  # 32.0
```

### Validation Error Messages

When validation fails, Fluxion provides detailed error messages:

```python
from fluxion import BatchOracle, ValidationError

oracle = BatchOracle()

# Invalid U-value
try:
    oracle.validate_parameters([0.05, 20.0])  # U-value too low
except ValidationError as e:
    print(e)  # "Parameter validation error: window_u_value must be in range [0.1, 5.0], got 0.05"

# NaN value
try:
    oracle.validate_parameters([float('nan'), 20.0])
except ValidationError as e:
    print(e)  # "Parameter validation error: window_u_value must be finite, got NaN"

# Heating > Cooling
try:
    oracle.validate_parameters([1.5, 26.0, 24.0])  # heating > cooling
except ValidationError as e:
    print(e)  # "Parameter validation error: heating_setpoint (26.0) must be less than cooling_setpoint (24.0)"
```

## Failure Modes

Fluxion defines specific error types for different failure scenarios.

### Error Types

#### ValidationError
Raised when input parameters are invalid:
- Parameter values outside valid ranges
- NaN or Infinity values in parameters
- Invalid parameter vector lengths
- Heating/cooling setpoint conflicts

```python
from fluxion import BatchOracle, ValidationError

oracle = BatchOracle()
try:
    # Invalid parameter
    result = oracle.evaluate_population([[0.05, 20.0]], use_surrogates=False)
except ValidationError as e:
    print(f"Validation failed: {e}")
```

#### SurrogateError
Raised when surrogate model operations fail:
- ONNX Runtime initialization failures
- Model loading errors (file not found, invalid format)
- Inference failures (GPU not available)
- Session pool exhaustion

```python
from fluxion import BatchOracle, SurrogateError

oracle = BatchOracle()
try:
    oracle.load_surrogate("nonexistent_model.onnx")
except SurrogateError as e:
    print(f"Surrogate error: {e}")
```

#### SimulationError
Raised when simulation calculations fail:
- Physics calculation failures (singular matrices)
- NaN/Infinity propagation during simulation
- Integration errors
- Invalid thermal network states

```python
from fluxion import Model, SimulationError

model = Model()
try:
    # This should not raise SimulationError with valid params
    eui = model.simulate(years=1, use_surrogates=False)
except SimulationError as e:
    print(f"Simulation failed: {e}")
```

### Error Recovery Patterns

Fluxion is designed for robust error handling:

1. **All errors return as Result types** - Rust code uses Result for all fallible operations
2. **Python bindings convert to exceptions** - PyO3 automatically converts Rust errors to Python exceptions
3. **Graceful degradation** - Surrogate failures fall back to analytical mode

```python
from fluxion import BatchOracle, SurrogateError, ValidationError

oracle = BatchOracle()

# Try surrogate mode, fall back to analytical on error
population = [[1.5, 20.0, 24.0], [2.0, 21.0, 25.0]]

try:
    # Try with surrogates
    results = oracle.evaluate_population(population, use_surrogates=True)
except SurrogateError:
    # Fall back to analytical mode
    print("Surrogate unavailable, using analytical mode")
    results = oracle.evaluate_population(population, use_surrogates=False)

# Validate parameters before simulation
valid_params = [[1.5, 20.0, 24.0], [2.0, 21.0, 25.0]]
invalid_params = [[0.05, 20.0, 24.0]]  # Invalid U-value

for params in valid_params:
    oracle.validate_parameters(params)  # Raises ValidationError if invalid

# Handle validation errors gracefully
for params in invalid_params:
    try:
        oracle.validate_parameters(params)
    except ValidationError as e:
        print(f"Skipping invalid parameters: {e}")
```

## Error Recovery Strategies

### No Panics in Production

Fluxion is designed to never panic in production code:

- All errors return as `Result` types
- Python bindings convert all errors to exceptions
- No `unwrap()` or `expect()` in hot paths
- All panics are caught and converted to errors

### Graceful Degradation

Fluxion supports graceful degradation when surrogates fail:

```python
from fluxion import BatchOracle

oracle = BatchOracle()

# Load surrogate (may fail if model unavailable)
try:
    oracle.load_surrogate("models/loads_predictor.onnx")
    use_surrogates = True
except Exception:
    print("Warning: Surrogate unavailable, using analytical mode")
    use_surrogates = False

# Use whatever is available
population = [[1.5, 20.0, 24.0] for _ in range(1000)]
results = oracle.evaluate_population(population, use_surrogates=use_surrogates)
```

### Parameter Validation Before Simulation

Always validate parameters before running expensive simulations:

```python
from fluxion import BatchOracle, ValidationError

oracle = BatchOracle()
population = [
    [1.5, 20.0, 24.0],  # Valid
    [0.05, 20.0, 24.0], # Invalid - will be rejected
    [2.0, 21.0, 25.0],  # Valid
]

# Filter valid parameters
valid_population = []
for params in population:
    try:
        oracle.validate_parameters(params)
        valid_population.append(params)
    except ValidationError as e:
        print(f"Skipping invalid: {e}")

# Only evaluate valid parameters
if valid_population:
    results = oracle.evaluate_population(valid_population, use_surrogates=False)
```

## Determinism Guarantees

### Analytical Mode (use_surrogates=False)

Analytical mode is **deterministic**:
- Same parameters produce same results
- No floating-point non-determinism
- Same inputs → same outputs across runs
- Order-independent (reductions use commutative operations)

```python
from fluxion import BatchOracle

oracle = BatchOracle()
population = [[1.5, 20.0, 24.0], [2.0, 21.0, 25.0]]

# Run multiple times - results are identical
results1 = oracle.evaluate_population(population, use_surrogates=False)
results2 = oracle.evaluate_population(population, use_surrogates=False)

assert results1 == results2  # Always true in analytical mode
```

### Surrogate Mode (use_surrogates=True)

Surrogate mode has **minor non-determinism**:
- GPU inference may have minor numerical variations
- Parallel execution order may affect intermediate results
- ONNX Runtime may use different kernel implementations

```python
from fluxion import BatchOracle

oracle = BatchOracle()
oracle.load_surrogate("models/loads_predictor.onnx")

population = [[1.5, 20.0, 24.0], [2.0, 21.0, 25.0]]

# Results may differ slightly due to GPU kernel implementation
results = oracle.evaluate_population(population, use_surrogates=True)
# Differences are typically < 0.1%
```

### Reproducibility Requirements

For **exact reproducibility**:

1. Use analytical mode (`use_surrogates=False`)
2. Use single-threaded execution if needed
3. Pin to specific hardware (avoid different CPU architectures)
4. Use fixed random seeds if any stochastic elements exist

```python
# For maximum reproducibility
import os
os.environ['RAYON_NUM_THREADS'] = '1'  # Single-threaded

oracle = BatchOracle()
results = oracle.evaluate_population(population, use_surrogates=False)
```

## Performance Guarantees

### Latency Targets

| Mode | Per-Config Latency | Per-Timestep Latency |
|------|-------------------|---------------------|
| Analytical | < 100ms | < 0.06µs |
| Surrogate | < 10ms | < 0.006µs |

### Throughput Targets

| Mode | Throughput |
|------|------------|
| Analytical | > 2,000 configs/sec |
| Surrogate | > 10,000 configs/sec |

### Hardware Requirements

- **Minimum**: 4-core CPU, 8GB RAM
- **Recommended**: 8-core CPU, 16GB RAM
- **High-Performance**: 16+ core CPU, 32GB RAM, GPU

## Version Stability

Fluxion maintains API stability:

- No breaking changes without deprecation
- Migration guides provided for major versions
- Backward compatibility maintained for minor versions
- See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for details
