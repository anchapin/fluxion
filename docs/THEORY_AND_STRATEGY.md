# Theory and Strategy

This document gives a high-level explanation of the physical model used in the repository, the surrogate strategy, and recommended usage patterns for optimization workflows.

1) Thermal Model (RC network)

The `ThermalModel` implements a simplified resistor-capacitor (RC) network approximation for building thermal dynamics. Key points:

- Zones: The model divides the building into `num_zones` thermal nodes, each with a temperature state (°C).
- Time discretization: Hourly timesteps are used (1 hour per step); a full year is `8760` steps.
- Loads: Thermal loads are applied per zone (W/m²) and influence the node temperatures via simplified state updates.
- Conduction: A simplified conduction term uses the window U-value to reduce temperature according to a conduction_loss calculation in `solve_timesteps`.
- Control: HVAC is represented implicitly via a setpoint; energy is computed as the cumulative temperature deviation from the setpoint.

Limitations of the current toy model
- The current implementation is intentionally simplified for fast development and testing. It does not:
  - Include weather station inputs (external temperature or solar radiation).
  - Model convection, detailed radiation exchange, or transient heat storage with realistic capacitances.
  - Output calibrated energy units (kWh/m²/year) without additional scaling parameters.

2) Surrogate strategy

Surrogates are intended to approximate expensive physics (CFD, ray-tracing, or detailed zone coupling) with a neural network. The design goals:

- Speed: Reduce per-timestep or per-zone computations by using pre-trained neural networks that predict loads or fluxes.
- Physics-awareness: Surrogates should be designed to respect conservation constraints where possible (for example, predict components that sum to known totals).
- Safety: The system uses a `use_surrogates` flag so developers can validate results against analytical baselines.

Current placeholder
- `SurrogateManager` currently returns a constant vector of mock loads when `model_loaded=false`. This allows deterministic tests and API validation without shipping large ONNX assets.

Recommended surrogate integration steps

1. Train surrogate(s) offline against high-fidelity simulation outputs.
2. Export surrogate(s) to ONNX and place the model file under `assets/` (or similar).
3. Implement ONNX runtime session in `SurrogateManager::new()` and inference in `predict_loads()`.
4. Validate surrogate predictions against holdout physics-based cases, and expose uncertainty estimates when possible.

3) Batch processing & performance

Batch processing design
- The `BatchOracle` pattern is critical: it clones a base `ThermalModel` per candidate, applies parameters, and runs `solve_timesteps` in parallel using `rayon::par_iter()`.
- Cross the Python-Rust boundary once per population call. Provide the entire population as a single data structure to avoid FFI overhead.

Performance tips
- Keep surrogate inference as a vectorized operation when possible (batch multiple zones or candidates per ONNX run).
- Use `cargo build --release` and `maturin develop --release` for performance testing.
- Avoid nested parallelism: parallelize at the population level only.

4) Optimization strategies

- Parameter vector semantics: Document all gene indices and valid ranges (0: window U-value, 1: HVAC setpoint, ...).
- Use normalized parameter ranges when passing vectors to optimizers; map them in `apply_parameters` to physical ranges.
- When using surrogates in optimization loops, periodically validate top candidates using the analytical physics engine to detect surrogate bias.

5) Validation and calibration

- Calibration: Use measured building energy data or high-fidelity simulation outputs to calibrate the physics constants and surrogate outputs.
- Validation tests: Implement unit tests that compare surrogate-enabled runs vs analytical runs on known test cases.

6) Safety and reproducibility

- Fix random seeds for experiments and CI tests.
- Log candidate parameters and seed values for each optimization run to enable traceability.

7) Next steps for production-readiness

1. Implement ONNX inference in `SurrogateManager` and add a small sample model in `assets/` for demos.
2. Add weather processing and a standard weather file input format (e.g., EPW) for realistic external forcing.
3. Add per-zone geometry, area and thermal capacity parameters to `ThermalModel` so simulation outputs can be converted to kWh/m²/year.
4. Add more comprehensive tests and benchmarks for batch throughput (1000+ candidates) in CI using release builds.
