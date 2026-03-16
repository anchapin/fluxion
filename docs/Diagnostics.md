# Diagnostics and Reporting Guide

Fluxion provides comprehensive diagnostic capabilities for analyzing ASHRAE 140 validation results and simulation time series data. This guide covers enabling diagnostics, exporting CSV data, and using external tools for analysis.

## Table of Contents

- [Enabling Diagnostics](#enabling-diagnostics)
- [CSV Export Tool](#csv-export-tool)
- [Output Structure](#output-structure)
- [CSV Columns](#csv-columns)
- [Metadata JSON](#metadata-json)
- [Loading Data in Python](#loading-data-in-python)
- [Example Visualizations](#example-visualizations)
- [Advanced Usage](#advanced-usage)

## Enabling Diagnostics

Diagnostics are disabled by default to minimize performance overhead. To enable:

### Environment Variables

```bash
# Enable all diagnostic output
export ASHRAE_140_DEBUG=1

# Enable hourly CSV export (automatic when using validator's with_full_diagnostics)
export ASHRAE_140_HOURLY_OUTPUT=1

# Set output path for hourly data
export ASHRAE_140_HOURLY_PATH=output/hourly.csv

# Enable verbose logging
export ASHRAE_140_VERBOSE=1
```

### Programmatic Configuration

```rust
use fluxion::validation::{DiagnosticConfig, DiagnosticCollector};

let config = DiagnosticConfig::full(); // enable everything
let collector = DiagnosticCollector::new(config);
```

## CSV Export Tool

Fluxion includes a dedicated CLI for exporting simulation data to CSV format. This tool runs validation cases and produces per-zone time series files and metadata.

### Installation

The tool is built as part of the Fluxion binary. Build it with:

```bash
cargo build --release --bin export_csv
```

The binary will be located at `target/release/export_csv`.

### Basic Usage

Export default cases (900 and 960):

```bash
cargo run --bin export_csv --
```

Export specific cases:

```bash
cargo run --bin export_csv -- --cases 600,650,900
```

Change output directory:

```bash
cargo run --bin export_csv -- --output-dir results/csv
```

Use European-style semicolon delimiter:

```bash
cargo run --bin export_csv -- --delimiter ';'
```

### Arguments

| Argument      | Description                                     | Default                |
|---------------|-------------------------------------------------|------------------------|
| `--cases`     | Comma-separated list of case IDs to export     | `900,960`              |
| `--output-dir`| Base directory for CSV files                   | `output/csv`           |
| `--delimiter` | CSV field delimiter (`,` or `;`)               | `,`                    |

## Output Structure

For each case, the exporter creates a subdirectory containing one CSV file per zone and a metadata JSON file:

```
output/csv/
├── 900/
│   ├── case_900_zone0.csv
│   ├── case_900_zone1.csv  (if multi-zone)
│   └── metadata.json
├── 960/
│   ├── case_960_zone0.csv
│   ├── case_960_zone1.csv
│   └── metadata.json
└── ...
```

### CSV File Format

Each `case_XXX_zoneN.csv` contains hourly time series data with the following columns:

- **Hour**: Hour index (0–8759)
- **Month**: Month number (1–12)
- **Day**: Day of month (1–31)
- **HourOfDay**: Hour of day (0–23)
- **Outdoor_Temp**: Outdoor dry-bulb temperature (°C)
- **Zone_Temp**: Zone air temperature (°C)
- **Mass_Temp**: Thermal mass temperature (°C)
- **Solar_Gain**: Solar gains (Watts)
- **Internal_Load**: Internal loads (Lighting+Equipment+People) (Watts)
- **HVAC_Heating**: HVAC heating power (Watts, 0 if cooling)
- **HVAC_Cooling**: HVAC cooling power (Watts, 0 if heating)
- **Infiltration_Loss**: Infiltration heat loss/gain (Watts)
- **Envelope_Conduction**: Conduction through envelope (Watts)

All numeric values are formatted with two decimal places. The delimiter is configurable via `--delimiter`.

### Metadata JSON

The `metadata.json` file contains:

- **case_id**: Case identifier (e.g., "900")
- **case_spec**: Full ASHRAE 140 case specification (geometry, construction, HVAC, etc.) as JSON
- **validation_results**: Array of validation metrics, each with:
  - `metric`: Metric name (e.g., "Annual Heating", "Peak Cooling")
  - `fluxion_value`: Simulated value
  - `reference_min` / `reference_max`: ASHRAE reference range
  - `status`: "PASS", "WARN", or "FAIL"
  - `deviation_percent`: Percent deviation from reference midpoint
- **energy_breakdown** (if available):
  - `envelope_conduction_mwh`
  - `infiltration_mwh`
  - `solar_gains_mwh`
  - `internal_gains_mwh`
  - `heating_mwh`
  - `cooling_mwh`
  - `net_balance_mwh`
- **peak_timing** (if available):
  - `peak_heating_kw` and `peak_heating_hour`
  - `peak_cooling_kw` and `peak_cooling_hour`
- **export_info**:
  - `delimiter`: Delimiter used in CSV files
  - `columns`: List of column names in order

## Loading Data in Python

The CSV files can be easily loaded into pandas for analysis:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load a single zone's data
df = pd.read_csv('output/csv/900/case_900_zone0.csv')

# Inspect the first few rows
print(df.head())

# Plot zone temperature and outdoor temperature
plt.figure(figsize=(12, 6))
plt.plot(df['Hour'], df['Zone_Temp'], label='Zone Temp')
plt.plot(df['Hour'], df['Outdoor_Temp'], label='Outdoor Temp', alpha=0.5)
plt.xlabel('Hour of Year')
plt.ylabel('Temperature (°C)')
plt.title('Case 900 - Zone 0 Temperature Profile')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Compute heating and cooling energy
heating_energy_mwh = df['HVAC_Heating'].sum() / 1e6  # Convert Wh to MWh
cooling_energy_mwh = df['HVAC_Cooling'].sum() / 1e6
print(f"Heating: {heating_energy_mwh:.2f} MWh")
print(f"Cooling: {cooling_energy_mwh:.2f} MWh")
```

### Loading Metadata

```python
import json

with open('output/csv/900/metadata.json') as f:
    meta = json.load(f)

print(f"Case: {meta['case_id']}")
print("Validation Results:")
for result in meta['validation_results']:
    print(f"  {result['metric']}: {result['fluxion_value']:.2f} ({result['status']})")
```

## Example Visualizations

### Temperature Profile with Shaded Heating/Cooling Periods

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('output/csv/900/case_900_zone0.csv')

fig, ax1 = plt.subplots(figsize=(14, 6))

# Temperature plot
ax1.plot(df['Hour'], df['Zone_Temp'], 'b-', label='Zone Temp')
ax1.plot(df['Hour'], df['Outdoor_Temp'], 'k-', alpha=0.3, label='Outdoor Temp')
ax1.set_ylabel('Temperature (°C)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Shade heating/cooling periods on second y-axis
ax2 = ax1.twinx()
ax2.fill_between(df['Hour'], 0, df['HVAC_Heating']/1000, color='red', alpha=0.3, label='Heating (kW)')
ax2.fill_between(df['Hour'], 0, -df['HVAC_Cooling']/1000, color='blue', alpha=0.3, label='Cooling (kW)')
ax2.set_ylabel('HVAC Power (kW)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Case 900 - Temperature and HVAC Loads')
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.show()
```

## Advanced Usage

### Batch Export for All Cases

```bash
# Get all case IDs from the validator and export in a loop
for case in 600 610 620 630 640 650 600FF 650FF 900 910 920 930 940 950 900FF 950FF 960; do
    cargo run --bin export_csv -- --cases "$case"
done
```

### Custom Post-Processing Script

You can write a Rust program that uses the `CsvExporter` directly for more control:

```rust
use fluxion::validation::{ASHRAE140Validator, export::CsvExporter};
use std::path::PathBuf;

fn main() -> Result<()> {
    let validator = ASHRAE140Validator::with_full_diagnostics();
    let exporter = CsvExporter::new(PathBuf::from("custom/output"), ',');

    let case = fluxion::validation::ashrae_140_cases::ASHRAE140Case::Case900;
    let (report, collector) = validator.validate_single_case_with_diagnostics(case);
    let spec = case.spec();

    exporter.export_diagnostics("900", &collector, &spec)?;
    exporter.export_metadata("900", &spec, &report, &collector)?;

    Ok(())
}
```

## Troubleshooting

**Q: Some zones have zero HVAC heating/cooling values?**
A: The current implementation records HVAC loads for all zones as the total system load (zone 0 values). This is a known limitation; future versions will provide per-zone HVAC分配.

**Q: Interzone heat transfer is not shown in CSV?**
A: Interzone heat transfer is currently aggregated into envelope conduction and internal loads. A dedicated column may be added in a future update.

**Q: Export fails with "No hourly data collected"?**
A: Ensure the validator was created with `with_full_diagnostics()` or that `DiagnosticConfig::output_hourly` is set to `true`.

**Q: Can I export data for a case without running validation?**
A: The current tool requires running validation to collect diagnostics. For raw simulation without validation, use the `Model` API directly and collect your own data.

## Future Enhancements

- Per-zone HVAC load allocation for multi-zone cases
- Separate column for interzone heat transfer
- Direct export from `Model` without full validation
- Compression of output (gzip) for large datasets
- Streaming export for very large populations

---

*Last updated: 2026-03-10*
