# Fluxion Building Packs

Pre-configured building simulation packs for quick validation, testing, and analysis.

## Available Packs

### 1. Single Zone Validation (`single_zone_validation/`)
**Purpose:** Quick validation testing for Fluxion installation and basic functionality.

- Minimal single-zone building model
- ASHRAE 140 Case 600 baseline
- < 1 minute runtime

```bash
cd examples/packs/single_zone_validation
./run_validation.sh
```

### 2. Multi Zone (`multi_zone/`)
**Purpose:** Multi-zone thermal modeling with inter-zone heat transfer.

- Two zones with independent setpoints
- Inter-zone conductance modeling
- Energy conservation validation

```bash
cd examples/packs/multi_zone
./run_simulation.sh
```

### 3. Retrofit (`retrofit/`)
**Purpose:** Evaluate energy savings from building upgrade scenarios.

- Baseline vs Retrofit comparison
- Wall, window, HVAC, lighting upgrades
- 15-40% expected savings

```bash
cd examples/packs/retrofit
./compare_scenarios.sh
```

### 4. Surrogate Benchmarking (`surrogate_benchmarking/`)
**Purpose:** Validate ONNX surrogate models against physics simulations.

- Accuracy vs reference cases
- Performance benchmarking
- 10-100x speedup validation

```bash
cd examples/packs/surrogate_benchmarking
./run_benchmark.sh
```

## Pack Structure

Each pack contains:

| File | Description |
|------|-------------|
| `manifest.json` | Pack metadata, parameters, expected outputs |
| `config.yaml` | Building/hvac configuration for simulation |
| `README.md` | Pack documentation and references |
| `*.sh` | Executable scripts for running simulations |

## Definition of Done (from Issue DX-02)

- ✅ Single-zone validation pack
- ✅ Multi-zone pack
- ✅ Retrofit benchmarking pack
- ✅ Surrogate benchmarking pack

## Dependencies

Required prior to running packs:
- Rust toolchain (cargo)
- Python 3.8+ (for surrogate benchmarking)
- maturin (for Python bindings)

## References

- `examples/README.md` - General example usage
- `docs/EXAMPLES.md` - Detailed input/output formats
- `docs/SURROGATE_GOVERNANCE.md` - Surrogate safety guidelines
- `tests/ashrae_140/` - ASHRAE 140 validation test suite
