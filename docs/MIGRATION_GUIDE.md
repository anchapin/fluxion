# Fluxion Migration Guide

This guide helps users migrate between versions of Fluxion.

## v0.4 to v0.5

Fluxion v0.5 maintains full backward compatibility with v0.4. No code changes are required.

### API Compatibility

v0.5 includes all v0.4 APIs with the same signatures and behavior:

```python
# v0.4 code works unchanged in v0.5
from fluxion import BatchOracle

oracle = BatchOracle()
population = [[1.5, 20.0, 24.0], [2.0, 21.0, 25.0]]
results = oracle.evaluate_population(population, use_surrogates=False)
```

### Parameter Format

The parameter vector format remains unchanged:

| Index | Parameter | Range | Units |
|-------|-----------|-------|-------|
| 0 | Window U-value | [0.1, 5.0] | W/m²K |
| 1 | Heating setpoint | [15.0, 25.0] | °C |
| 2 | Cooling setpoint | [22.0, 32.0] | °C |

### New Features in v0.5

While maintaining compatibility, v0.5 includes these improvements:

1. **Enhanced Validation Error Messages**

   v0.5 provides more detailed validation error messages:

   ```python
   # v0.5 shows the valid range in the error
   try:
       oracle.validate_parameters([0.05, 20.0])
   except ValidationError as e:
       print(e)
       # "Parameter validation error: window_u_value must be in range [0.1, 5.0], got 0.05"
   ```

2. **Improved Performance Benchmarks**

   v0.5 includes comprehensive benchmark infrastructure:

   ```python
   # Built-in benchmarking
   import time
   start = time.perf_counter()
   results = oracle.evaluate_population(population, use_surrogates=False)
   elapsed = time.perf_counter() - start
   print(f"Throughput: {len(population)/elapsed:.0f} configs/sec")
   ```

3. **ASHRAE 140 Compliance Improvements**

   All 18 ASHRAE 140 validation cases now pass:

   ```bash
   cargo test --test ashrae_140_validation
   # All 18 cases passing
   ```

4. **Stability Documentation**

   v0.5 includes comprehensive stability documentation:

   ```python
   # See docs/STABILITY.md for:
   # - Input validation rules
   # - Error handling patterns
   # - Determinism guarantees
   # - Performance targets
   ```

### Migration Checklist

- [x] No code changes required
- [x] Existing parameter validation still works
- [x] BatchOracle API unchanged
- [x] Model API unchanged
- [x] Error types unchanged

### Common Patterns (Unchanged)

These patterns work in both v0.4 and v0.5:

```python
from fluxion import BatchOracle, Model, ValidationError, SurrogateError

# Basic usage
oracle = BatchOracle()
results = oracle.evaluate_population([[1.5, 20.0, 24.0]], use_surrogates=False)

# Single building simulation
model = Model(num_zones=1)
eui = model.simulate(years=1, use_surrogates=False)

# Parameter validation
bounds = oracle.get_parameter_bounds()
oracle.validate_parameters([1.5, 20.0, 24.0])

# Error handling
try:
    results = oracle.evaluate_population(population, use_surrogates=False)
except ValidationError as e:
    print(f"Invalid parameters: {e}")
except SurrogateError as e:
    print(f"Surrogate error: {e}")

# Surrogate loading
oracle.load_surrogate("models/loads_predictor.onnx")
results = oracle.evaluate_population(population, use_surrogates=True)
```

### Version Compatibility Matrix

| Feature | v0.4 | v0.5 |
|---------|------|------|
| BatchOracle | ✓ | ✓ |
| Model | ✓ | ✓ |
| Parameter validation | ✓ | ✓ |
| Surrogate inference | ✓ | ✓ |
| ASHRAE 140 | 18/18 | 18/18 |
| API stability | ✓ | ✓ |

## Deprecation Policy

Fluxion follows semantic versioning:

- **Major versions** (v1.0): May include breaking changes with 6-month deprecation notice
- **Minor versions** (v0.5): Backward compatible additions
- **Patch versions** (v0.5.1): Bug fixes only

### Deprecation Patterns

When APIs are deprecated, they will:

1. Show deprecation warnings in documentation
2. Continue to work for at least 6 months
3. Include migration guidance in error messages

Example deprecation (if any in future):

```python
import warnings

# Old API (deprecated)
warnings.warn(
    "old_function() is deprecated, use new_function() instead",
    DeprecationWarning,
    stacklevel=2
)
```

## Getting Help

- **Documentation**: [docs/](docs/)
- **API Reference**: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- **Stability Guarantees**: [docs/STABILITY.md](docs/STABILITY.md)
- **Performance Benchmarks**: [docs/PERFORMANCE_BENCHMARKS.md](docs/PERFORMANCE_BENCHMARKS.md)
- **GitHub Issues**: [https://github.com/fluxion/fluxion/issues](https://github.com/fluxion/fluxion/issues)

## Changelog Summary

### v0.5.0 (Current)
- Full API compatibility with v0.4
- Enhanced validation error messages
- Improved ASHRAE 140 compliance (18/18 cases)
- Comprehensive stability documentation
- Performance benchmark infrastructure

### v0.4.0 (Previous)
- Initial stable release
- BatchOracle for population evaluation
- Model for single-building simulation
- Surrogate inference support
- ASHRAE 140 validation (18/18 cases)
