# Examples: Inputs, Outputs, and Expected Behaviour

This document explains the inputs and outputs used by the examples in `examples/`, how to interpret the printed results, and provides small recipes for normalizing the toy-metric produced by the current physics engine.

1) Input formats

- `run_oracle.py` (Batch style): The `BatchOracle.evaluate_population` function expects `Vec<Vec<f64>>` in Rust, which maps to a Python `List[List[float]]` or a `numpy.ndarray.tolist()` result. Each inner vector is a design candidate with the following semantics:
  - `params[0]`: Window U-value (W/m²K). Example allowed range: `0.5` to `3.0`.
  - `params[1]`: HVAC setpoint (°C). Example allowed range: `19.0` to `24.0`.

- `run_model.py` (Single model): The `Model` constructor currently accepts a `config_path` string but ignores file contents; `simulate(years: u32, use_surrogates: bool)` runs the physics for `years * 8760` timesteps.

2) Output explained

- The examples print three things:
  - `Elapsed`: Wall-clock time to evaluate the population or run the model. Useful for measuring throughput.
  - `Best candidate index` and its `EUI`: Index in the population with the lowest (best) EUI and the numeric EUI value.
  - `Sample results`: Per-candidate printout showing `U`, `setpoint`, and `EUI`.

3) Why are EUI values large?

The current `ThermalModel::solve_timesteps` accumulates a simple metric: at each timestep it sums |temperature - setpoint| across all zones and adds this to a running total. This produces a cumulative number across zones and hours (e.g., 10 zones * 8760 hours = 87,600 contributions), hence large numeric outputs. These are intentionally uncalibrated and intended for algorithm correctness and performance testing.

4) Normalizing the toy metric

To create a more human-friendly, average metric you can normalize like this (performed in Python after `results` are returned):

```python
num_zones = 10  # matches ThermalModel default
timesteps_per_year = 8760

def normalize(raw_eui):
    return raw_eui / (num_zones * timesteps_per_year)

# Example usage
# normalized = normalize(results[best_idx])
```

This yields an average hourly temperature-gap per zone which is useful for relative comparisons between candidates.

5) Converting to physical units

To convert the normalized metric to physical energy (kWh/m²/year) you need additional data:

- Thermal capacity / heat capacity (J/K) of zones or mass
- Area (m²) that the metric should be expressed per
- Time-step duration (hours) — here `1 hour` per step

Rough pipeline:

1. Convert average temperature-gap (°C) to energy using heat capacity (J/°C).
2. Convert Joules to kWh (1 kWh = 3.6e6 J).
3. Divide by area (m²) and by simulation years to get kWh/m²/year.

6) Example: compute normalized EUI and print

```python
raw = results[best_idx]
normalized = raw / (10 * 8760)
print(f"Raw: {raw:.1f}, normalized avg temp-gap per zone (°C-hr): {normalized:.6f}")
```

7) Tips for using examples in tests or CI

- Use small populations (20-100) for CI to keep run-time small.
- Pin your Python interpreter (venv) in CI to match maturin-built wheel platform.
- To avoid flakiness, set the random seed in `run_oracle.py` (or use NumPy RNG) and/or mock the `SurrogateManager`.
