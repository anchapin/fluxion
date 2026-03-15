# Plan 07-02 Summary: Delta Testing Framework

## Implementation

The delta testing framework (`src/analysis/delta.rs`) has been fully implemented, providing a robust system for comparing building energy simulation results across multiple configuration variants.

### Core Structures

- **DeltaConfig**: Top-level configuration containing a base `CaseSpec` and a list of `Variant` definitions.
- **Variant**: Defines modifications to the base case via either a `patch` (field overwrites) or a `sweep` (parametric range). Supports both single-valued patches and multi-valued sweeps.
- **DeltaReport**: Contains the base case name and a vector of `VariantResult` entries, each with annual heating/cooling (MWh), peak heating/cooling (kW), and optional hourly differences.
- **HourlyDelta**: Detailed per-hour, per-zone, per-component differences between base and variant.
- **SimulationResult**: Internal structure returned by `run_simulation`, holding the key metrics and optional hourly `HourlyData`.

### Key Functions

1. **`parse_config(path: &Path) -> Result<DeltaConfig>`**
   - Deserializes a YAML file into a `DeltaConfig`.
   - Leverages `serde_yaml` and the existing `CaseSpec` Deserialize implementation.

2. **`expand_variants(config: &DeltaConfig) -> Result<Vec<(String, CaseSpec)>>`**
   - For each variant, applies the patch (if any) to the base `CaseSpec` using deep merge with dot notation.
   - If a sweep is defined, generates the Cartesian product of all parameter values, creating a separate `CaseSpec` for each combination. Variant names are suffixed with `key=value` pairs for identification.
   - Returns a flat list of `(variant_name, CaseSpec)` tuples ready for simulation.

3. **`run_simulation(spec: &CaseSpec, collect_hourly: bool) -> Result<SimulationResult>`**
   - Creates a `ThermalModel` from the `CaseSpec`.
   - Runs an 8760‑hour simulation using the same physics engine as the ASHRAE 140 validator.
   - Handles HVAC schedules (including multi‑zone setpoints), night ventilation, and internal loads (converted to density).
   - Collects hourly diagnostics (temperatures, solar gains, infiltration, envelope conduction, HVAC power) when requested.
   - Returns annual energies (MWh), peak powers (kW), and optional hourly time series.

4. **`run_comparison(base: &CaseSpec, variants: &[(String, CaseSpec)], include_hourly: bool) -> Result<(DeltaReport, SimulationResult)>`**
   - Orchestrates the base simulation and all variant simulations.
   - Computes `HourlyDelta` vectors when `include_hourly` is true using `compute_hourly_deltas`.
   - Returns both the `DeltaReport` (for reporting) and the base `SimulationResult` (for markdown generation).

5. **`generate_markdown_report(report: &DeltaReport, base: &SimulationResult) -> String`**
   - Produces a Markdown table with a row for the base case and one row per variant.
   - Columns: Variant name, annual heating (MWh) with absolute and percent delta, annual cooling (MWh) with deltas, peak heating (kW) with deltas, peak cooling (kW) with deltas.
   - Includes a “Sweep Statistics” section that groups sweep variants (by name prefix) and tabulates mean and sample standard deviation for each metric.

6. **`export_hourly_deltas_csv(report: &DeltaReport, path: &Path) -> Result<()>`**
   - Writes a long‑format CSV with columns: `Hour`, `Zone`, `Component`, `Base_Value`, `Variant_Value`, `Difference`.
   - All variants are written sequentially; the variant name itself is not included in the CSV (users can split by hour count).

7. **`run_and_report(config: DeltaConfig, output_dir: &Path, include_hourly: bool) -> Result<()>`**
   - High‑level convenience function: parses config, expands variants, runs comparison, writes `delta_report.md` and (if requested) `hourly_differences.csv`.

### Testing

Comprehensive unit tests cover:
- YAML configuration parsing (`test_config_parsing`)
- Patch application (`test_patch_application`)
- Sweep expansion (`test_sweep_expansion`)
- Simulation execution with and without hourly collection (`test_run_simulation_basic`, `test_run_simulation_with_hourly`)
- Comparison engine (`test_comparison`) verifying non‑zero differences
- Markdown generation (`test_markdown_generation`)
- CSV export (`test_csv_export`)
- Sweep statistics calculation (`test_sweep_statistics`)

All tests pass in release mode (`cargo test --release --lib delta`).

### Usage Example

```yaml
# delta_config.yaml
base:
  case_id: "600"
  description: "Base Case 600"
  # ... full CaseSpec fields ...
variants:
  - name: "high_infil"
    patch:
      infiltration_ach: 1.5
  - name: "u_sweep"
    sweep:
      window_u_value: [2.0, 3.0, 4.0]
```

```rust
use fluxion::analysis::delta::{parse_config, run_and_report};
let config = parse_config("delta_config.yaml")?;
run_and_report(config, "output/delta", true)?;
```

## Decisions

- Reused the existing validator’s simulation loop pattern (ThermalModel, step_physics, Denver weather) to ensure consistency.
- Implemented independent infiltration and envelope conduction calculation for hourly diagnostics only; model’s internal energy accounting remains authoritative.
- Provided both Markdown and CSV outputs for flexibility in reporting and downstream analysis.
- Sweep statistics are computed in `generate_markdown_report` and presented in a separate summary table, avoiding clutter in the main comparison table.
