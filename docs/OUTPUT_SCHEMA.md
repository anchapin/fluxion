# Fluxion Output Schema v1.0

## Overview

This document describes the **versioned, stable output contracts** for Fluxion simulation results. Consumers can rely on these schemas being consistent across versions.

## Schema Version

Current version: **1.0**

## Output Categories

The Fluxion simulation outputs data in these categories:

| Category | Description | Primary Structure |
|----------|-------------|-------------------|
| Zone State | Zone temperatures over time | `HourlyData` |
| Loads | Solar, internal, infiltration, conduction | `HourlyData` |
| Peaks | Peak heating/cooling power and timing | `PeakTiming` |
| Energy | Annual energy totals and breakdown | `EnergyBreakdown` |
| Comfort | Temperature statistics for free-floating cases | `TemperatureProfile` |
| Diagnostic Metadata | Validation results and export info | `DiagnosticReport`, `Metadata` |

---

## 1. Zone State

Zone state data represents the thermal conditions within each zone over time.

### HourlyData

Hourly time-series data for a single timestep.

```rust
pub struct HourlyData {
    pub hour: usize,              // Hour index (0-8759)
    pub month: u32,               // Month (1-12)
    pub day: u32,                 // Day of month (1-31)
    pub hour_of_day: u32,         // Hour of day (0-23)
    pub outdoor_temp: f64,        // Outdoor temperature (°C)
    pub zone_temps: Vec<f64>,     // Zone temperatures (°C) - one per zone
    pub mass_temps: Vec<f64>,    // Mass temperatures (°C) - one per zone
    pub solar_gains: Vec<f64>,    // Solar gains per zone (W)
    pub hvac_heating: Vec<f64>,  // HVAC heating power (W) - one per zone
    pub hvac_cooling: Vec<f64>,   // HVAC cooling power (W) - one per zone
    pub internal_loads: Vec<f64>, // Internal loads per zone (W)
    pub infiltration_loss: Vec<f64>,  // Infiltration heat loss per zone (W)
    pub envelope_conduction: Vec<f64>, // Envelope conduction per zone (W)
}
```

### CSV Export Format (Zone State)

File: `output_dir/{case_id}/case_{case_id}_zone{z}.csv`

| Column | Type | Description |
|--------|------|-------------|
| Hour | integer | Hour index (0-8759) |
| Month | integer | Month (1-12) |
| Day | integer | Day of month (1-31) |
| HourOfDay | integer | Hour of day (0-23) |
| Outdoor_Temp | float | Outdoor dry-bulb temperature (°C) |
| Zone_Temp | float | Zone air temperature (°C) |
| Mass_Temp | float | Thermal mass temperature (°C) |
| Solar_Gain | float | Solar gain (W) |
| Internal_Load | float | Internal gain from occupants/equipment (W) |
| HVAC_Heating | float | HVAC heating power (W) |
| HVAC_Cooling | float | HVAC cooling power (W) |
| Infiltration_Loss | float | Infiltration heat loss (W) |
| Envelope_Conduction | float | Envelope conduction heat transfer (W) |

---

## 2. Loads

Load data represents energy flows into and out of the building.

### Load Components

All loads are in **Watts (W)** and represent instantaneous power at each timestep:

| Component | Description | Sign Convention |
|-----------|-------------|-----------------|
| `solar_gains` | Solar radiation through windows | Always positive |
| `internal_loads` | Occupant and equipment gains | Always positive |
| `infiltration_loss` | Heat loss from air infiltration | Positive = loss from zone |
| `envelope_conduction` | Heat transfer through walls/roof/floor | Positive = loss from zone |
| `hvac_heating` | HVAC heating output | Positive when heating |
| `hvac_cooling` | HVAC cooling output | Positive when cooling |

### Load Aggregation

Annual energy loads are aggregated in `EnergyBreakdown`:

```rust
pub struct EnergyBreakdown {
    pub envelope_conduction_mwh: f64,  // Total envelope conduction (MWh)
    pub infiltration_mwh: f64,           // Total infiltration losses (MWh)
    pub solar_gains_mwh: f64,           // Total solar gains (MWh)
    pub internal_gains_mwh: f64,         // Total internal gains (MWh)
    pub heating_mwh: f64,                // Total heating energy (MWh)
    pub cooling_mwh: f64,                // Total cooling energy (MWh)
    pub net_balance_mwh: f64,           // Net energy balance (MWh)
}
```

**Note**: Net balance = solar_gains + internal_gains - heating + cooling

---

## 3. Peaks

Peak load data identifies maximum heating and cooling demands.

### PeakTiming

```rust
pub struct PeakTiming {
    pub peak_heating_kw: f64,    // Peak heating load (kW)
    pub peak_heating_hour: usize, // Hour of peak heating (0-8759)
    pub peak_cooling_kw: f64,     // Peak cooling load (kW)
    pub peak_cooling_hour: usize, // Hour of peak cooling (0-8759)
}
```

### Hour-to-DateTime Conversion

Hour index can be converted to datetime using:

```
hour_to_datetime(hour) -> "Mon D HH:00"

Examples:
- Hour 0    -> "Jan 1 00:00"
- Hour 500  -> "Jan 21 20:00"
- Hour 4380 -> "Jul 2 12:00"
- Hour 8759 -> "Dec 31 23:00"
```

---

## 4. Energy

Energy outputs represent annual totals in **Megawatt-hours (MWh)**.

### Energy Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| Annual Heating | MWh | Total HVAC heating energy delivered |
| Annual Cooling | MWh | Total HVAC cooling energy delivered |
| Peak Heating | kW | Maximum instantaneous heating power |
| Peak Cooling | kW | Maximum instantaneous cooling power |

### Delta Report Structure (Analysis)

For comparative analysis, the `DeltaReport` provides:

```rust
pub struct DeltaReport {
    pub base_name: String,
    pub variants: Vec<VariantResult>,
}

pub struct VariantResult {
    pub name: String,
    pub annual_heating_mwh: f64,
    pub annual_cooling_mwh: f64,
    pub peak_heating_kw: f64,
    pub peak_cooling_kw: f64,
    pub hourly_differences: Option<Vec<HourlyDelta>>,  // If include_hourly=true
}
```

---

## 5. Comfort

Comfort metrics apply to **free-floating** (uncontrolled) cases.

### TemperatureProfile

```rust
pub struct TemperatureProfile {
    pub case_id: String,
    pub min_temp: f64,       // Minimum temperature (°C)
    pub max_temp: f64,       // Maximum temperature (°C)
    pub avg_temp: f64,       // Average temperature (°C)
    pub swing: f64,          // Temperature swing: max - min (K)
    pub hourly_temps: Vec<f64>,  // Hourly temperatures (°C)
}
```

### Comfort Thresholds (ASHRAE 140)

For validation, comfort is evaluated against ASHRAE 140 reference ranges:

| Metric | Description | Typical Range |
|--------|-------------|---------------|
| Free-Floating Min | Lowest zone temperature | Varies by case |
| Free-Floating Max | Highest zone temperature | Varies by case |
| Free-Floating Swing | Daily temperature swing | < 5 K for well-built |

---

## 6. Diagnostic Metadata

Diagnostic metadata provides validation results and export information.

### ValidationResult

```rust
pub struct ValidationResult {
    pub case_id: String,
    pub metric: MetricType,       // e.g., AnnualHeating, PeakHeating
    pub fluxion_value: f64,       // Fluxion calculated value
    pub ref_min: f64,             // Reference minimum
    pub ref_max: f64,             // Reference maximum
    pub percent_error: f64,       // Percent error from reference midpoint
    pub status: ValidationStatus, // PASS or FAIL
    pub per_program: Option<HashMap<String, f64>>,  // Per-program breakdown
    pub peak_timestamp: Option<DateTime<Utc>>,  // Timestamp of peak value (for peak metrics)
}
```

### Metric Types

| MetricType | Description |
|------------|-------------|
| `AnnualHeating` | Annual heating energy (MWh) |
| `AnnualCooling` | Annual cooling energy (MWh) |
| `PeakHeating` | Peak heating load (kW) |
| `PeakCooling` | Peak cooling load (kW) |
| `FreeFloatingMin` | Minimum free-floating temperature (°C) |
| `FreeFloatingMax` | Maximum free-floating temperature (°C) |

### DiagnosticReport

```rust
pub struct DiagnosticReport {
    pub config: DiagnosticConfig,
    pub hourly_data: Vec<HourlyData>,
    pub energy_breakdowns: HashMap<String, EnergyBreakdown>,
    pub peak_timings: HashMap<String, PeakTiming>,
    pub temperature_profiles: HashMap<String, TemperatureProfile>,
    pub comparison_rows: Vec<ComparisonRow>,
}
```

### Export Metadata JSON

When exporting via `CsvExporter`, a `metadata.json` file is created:

```json
{
  "case_id": "600",
  "case_spec": { /* CaseSpec object */ },
  "validation_results": [ /* ValidationResult array */ ],
  "energy_breakdown": { /* EnergyBreakdown object */ },
  "peak_timing": { /* PeakTiming object */ },
  "export_info": {
    "delimiter": ",",
    "columns": ["Hour", "Month", "Day", "HourOfDay", ...]
  }
}
```

---

## Export Formats

### CSV (Hourly Zone Data)

- One file per zone: `case_{case_id}_zone{z}.csv`
- Delimiter: configurable (default `,`)
- Encoding: UTF-8

### JSON (Metadata)

- One file per case: `metadata.json`
- Pretty-printed for human readability
- Contains full case specification and validation results

### Markdown (Reports)

- Delta reports: `delta_report.md`
- Diagnostic reports: auto-generated from `DiagnosticReport`

---

## Versioning Policy

1. **Stable Outputs**: Listed categories are versioned and will not break without notice
2. **Additive Changes**: New fields may be added without version bump
3. **Breaking Changes**: Major version increments indicate breaking changes
4. **Deprecation**: Old fields will be marked deprecated before removal

Current outputs are considered **v1.0 stable**.

---

## Example Usage

### Rust

```rust
use fluxion::validation::export::CsvExporter;
use fluxion::validation::diagnostic::DiagnosticCollector;

let exporter = CsvExporter::new(output_dir, ',');
exporter.export_diagnostics("600", &collector, &spec)?;
exporter.export_metadata("600", &spec, &report, &collector)?;
```

### Python (via FFI or subprocess)

```python
# Run fluxion and parse outputs
import subprocess
import json

result = subprocess.run(['fluxion', 'validate', '--case', '600'], capture_output=True)
# Parse output_dir/600/metadata.json
with open('output_dir/600/metadata.json') as f:
    metadata = json.load(f)
```

---

## Appendix: Field Units Reference

| Field | SI Unit | Conversion |
|-------|---------|------------|
| Temperature | °C | - |
| Power (instantaneous) | W | - |
| Energy | MWh | 1 MWh = 3.6 GJ |
| Power (peak) | kW | 1 kW = 1000 W |
| Time | hour | 8760 hours/year |
| Area | m² | - |
| Volume | m³ | - |
| U-value | W/m²·K | - |
| ACH | 1/h | air changes per hour |
