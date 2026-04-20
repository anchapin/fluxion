# Surrogate Benchmarking Pack

This pack provides workflows for benchmarking surrogate model accuracy against physics-based simulations.

## Files
- `manifest.json` - Pack configuration and metadata
- `config.yaml` - Surrogate and benchmark configurations
- `run_benchmark.sh` - Benchmark execution script

## Use Case
Use this pack to:
- Validate ONNX surrogate models against reference physics
- Measure surrogate speedup vs full simulation
- Verify accuracy across various climate zones
- Test fallback behavior when surrogates are unavailable

## Expected Results
- Surrogate speedup: 10-100x vs physics simulation
- Accuracy: R² > 0.95 vs reference
- Max absolute error: < 0.5°C on temperatures

## Quick Start
```bash
cd examples/packs/surrogate_benchmarking
./run_benchmark.sh
```

## Configuration
- Reference cases: ASHRAE 140 Cases 600, 900, 960
- Climate zones: Denver (5B), Boston (5A)
- Surrogate model: examples/dummy_surrogate.onnx
- Batch sizes: 1, 10, 100 for throughput testing

## References
- See `examples/validate_surrogate.py` for surrogate validation
- See `docs/SURROGATE_GOVERNANCE.md` for surrogate safety guidelines
- See `examples/run_oracle.py` for batch oracle usage
