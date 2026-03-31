# Diagnostics API Usage

This document describes how to use the diagnostic logging features in Fluxion.

## Enabling Diagnostics

Set the `RUST_LOG` environment variable to control verbosity:

- `info`: Summary statistics only
- `debug`: Hourly values printed to stderr
- `trace`: Full per-timestep details

## Using `validate_case_with_diagnostics`

The `validate_case_with_diagnostics` function provides a simple way to run a simulation with diagnostics collection:

```rust
use fluxion::validation::{validate_case_with_diagnostics, ASHRAE140Case};

let (report, diagnostics) = validate_case_with_diagnostics(ASHRAE140Case::Case900, true);

if let Some(diag) = diagnostics {
    // Print summary
    diag.print_summary();

    // Export hourly data to CSV
    diag.export_csv("case900_hourly.csv").unwrap();
}
```

## Data Collected

`SimulationDiagnostics` collects hourly:

- Zone temperatures (°C)
- Mass temperatures (°C)
- Surface temperatures (°C) (estimated)
- Load breakdown: solar, internal, HVAC, inter-zone, infiltration (Watts)
- Cumulative energy per zone (kWh)

## Performance Impact

Diagnostics collection has minimal overhead when disabled. When enabled, memory usage scales with number of zones and timesteps (approximately 10KB per zone per year).

## Export Format

CSV export includes columns: Hour, Zone_Temps, Mass_Temps, Surface_Temps, Solar_Watts, Internal_Watts, HVAC_Watts, InterZone_Watts, Infiltration_Watts. Multiple zones are separated by semicolons within a column.
