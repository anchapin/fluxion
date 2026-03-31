# ASHRAE 140 Diagnostic Features

This document describes the diagnostic output and debugging tools available for ASHRAE 140 validation in Fluxion.

## Overview

The diagnostic tools provide detailed visibility into simulation internals to help debug validation discrepancies and understand building energy behavior.

## Features

### 1. DiagnosticConfig

Controls diagnostic output behavior:

```rust
use fluxion::validation::diagnostic::DiagnosticConfig;

// Disabled by default (no output)
let config = DiagnosticConfig::disabled();

// Full diagnostics enabled
let config = DiagnosticConfig::full();

// From environment variables
let config = DiagnosticConfig::from_env();
```

**Configuration Options:**
- `enabled`: Enable/disable all diagnostics
- `output_hourly`: Collect hourly simulation data
- `hourly_output_path`: Path for hourly CSV export
- `output_energy_breakdown`: Generate energy breakdown summaries
- `output_peak_timing`: Report peak load timing
- `output_temperature_profiles`: Track temperature profiles for free-floating cases
- `output_comparison_table`: Generate validation comparison table
- `verbose`: Print detailed console output

### 2. HourlyData

Tracks hourly simulation values:

```rust
use fluxion::validation::diagnostic::HourlyData;

let mut data = HourlyData::new(hour, num_zones);
data.outdoor_temp = 10.5;
data.zone_temps[0] = 20.1;
data.solar_gains[0] = 100.0;
data.hvac_heating[0] = 500.0;
data.internal_loads[0] = 200.0;
```

**Fields:**
- `hour`: Hour index (0-8759)
- `month`: Month (1-12)
- `day`: Day of month (1-31)
- `hour_of_day`: Hour of day (0-23)
- `outdoor_temp`: Outdoor temperature (°C)
- `zone_temps`: Zone temperatures (°C)
- `mass_temps`: Thermal mass temperatures (°C)
- `solar_gains`: Solar gains per zone (W)
- `hvac_heating`: HVAC heating power per zone (W)
- `hvac_cooling`: HVAC cooling power per zone (W)
- `internal_loads`: Internal loads per zone (W)
- `infiltration_loss`: Infiltration heat loss per zone (W)
- `envelope_conduction`: Envelope conduction per zone (W)

### 3. EnergyBreakdown

Component-level energy analysis:

```rust
use fluxion::validation::diagnostic::EnergyBreakdown;

let breakdown = EnergyBreakdown {
    envelope_conduction_mwh: 2.5,
    infiltration_mwh: 1.0,
    solar_gains_mwh: 5.0,
    internal_gains_mwh: 3.0,
    heating_mwh: 4.0,
    cooling_mwh: 6.0,
    net_balance_mwh: 5.0,
};

// Print formatted breakdown
breakdown.print("600");

// Get formatted string
let formatted = breakdown.to_formatted_string("600");
```

**Output Format:**
```
Case 600 Energy Breakdown:
  Envelope conduction: 2.500 MWh
  Infiltration:        1.000 MWh
  Solar gains:         5.000 MWh
  Internal gains:      3.000 MWh
  ─────────────────────────────────
  Heating energy:      4.000 MWh
  Cooling energy:      6.000 MWh
  Net balance:         5.000 MWh
```

### 4. PeakTiming

Peak load timing information:

```rust
use fluxion::validation::diagnostic::PeakTiming;

let timing = PeakTiming {
    peak_heating_kw: 5.5,
    peak_heating_hour: 123,
    peak_cooling_kw: 3.2,
    peak_cooling_hour: 4567,
};

// Print formatted timing
timing.print("600");

// Get formatted string
let formatted = timing.to_formatted_string("600");

// Get datetime strings
let heating_time = timing.peak_heating_time_str();
let cooling_time = timing.peak_cooling_time_str();
```

**Output Format:**
```
Case 600 Peak Load Timing:
  Peak Heating: 5.50 kW at Hour 123 (Month 01 Day 06, 03:00)
  Peak Cooling: 3.20 kW at Hour 4567 (Month 08 Day 20, 19:00)
```

### 5. TemperatureProfile

Free-floating case temperature analysis:

```rust
use fluxion::validation::diagnostic::TemperatureProfile;

let mut profile = TemperatureProfile::new("600FF");
profile.update(15.0);
profile.update(20.0);
profile.update(25.0);
profile.finalize();

println!("Min: {:.1}°C", profile.min_temp);
println!("Max: {:.1}°C", profile.max_temp);
println!("Avg: {:.1}°C", profile.avg_temp);
println!("Swing: {:.1}°C", profile.swing);
```

### 6. ComparisonRow

Validation comparison table generation:

```rust
use fluxion::validation::diagnostic::ComparisonRow;

let row = ComparisonRow::new("600", "Heating", 5.0, 4.30, 5.71);

println!("Case: {}", row.case_id);
println!("Metric: {}", row.metric);
println!("Fluxion: {:.2}", row.fluxion_value);
println!("Ref Range: {:.2} - {:.2}", row.ref_min, row.ref_max);
println!("Deviation: {:.1}%", row.deviation_percent);
println!("Status: {}", row.status);
```

### 7. DiagnosticCollector

Accumulates simulation data:

```rust
use fluxion::validation::diagnostic::DiagnosticCollector;
use fluxion::validation::ASHRAE140Validator;

let config = DiagnosticConfig::full();
let mut collector = DiagnosticCollector::new(config);

// Start a new case
collector.start_case("600", num_zones);

// Record hourly data
for hour in 0..8760 {
    let data = HourlyData::new(hour, num_zones);
    // ... populate data fields ...
    collector.record_hour(data);
}

// Finalize and compute summaries
collector.finalize_case(heating_mwh, cooling_mwh);

// Export to CSV
collector.export_hourly_csv("hourly.csv")?;
```

### 8. DiagnosticReport

Comprehensive diagnostic report:

```rust
use fluxion::validation::diagnostic::DiagnosticReport;

let config = DiagnosticConfig::full();
let mut report = DiagnosticReport::new(config);

// Add diagnostic data
report.add_energy_breakdown("600", breakdown);
report.add_peak_timing("600", timing);
report.add_temperature_profile(profile);
report.add_comparison_row(row);

// Generate Markdown report
let markdown = report.to_markdown();

// Save to file
report.save_to_file("diagnostic_report.md")?;

// Print summary
report.print_summary();
```

## Usage

### Environment Variables

Enable diagnostics via environment variables:

```bash
# Enable all diagnostics
ASHRAE_140_DEBUG=1 cargo test --test ashrae_140_validation

# Enable verbose output
ASHRAE_140_DEBUG=1 ASHRAE_140_VERBOSE=1 cargo test

# Export hourly data
ASHRAE_140_DEBUG=1 ASHRAE_140_HOURLY_OUTPUT=1 \
  ASHRAE_140_HOURLY_PATH=hourly.csv cargo test
```

**Available Environment Variables:**
- `ASHRAE_140_DEBUG`: Enable/disable diagnostics (1 or true)
- `ASHRAE_140_VERBOSE`: Enable verbose console output (1 or true)
- `ASHRAE_140_HOURLY_OUTPUT`: Enable hourly data collection (1 or true)
- `ASHRAE_140_HOURLY_PATH`: Path for hourly CSV export

### Programmatic Usage

Use the validator with diagnostics:

```rust
use fluxion::validation::{ASHRAE140Validator, diagnostic::DiagnosticConfig};

// Create validator with diagnostics
let config = DiagnosticConfig::full();
let mut validator = ASHRAE140Validator::with_diagnostics(config);

// Run full validation with diagnostics
let (report, diagnostic_report) = validator.validate_with_diagnostics();

// Access diagnostic data
println!("Collected {} hourly records", diagnostic_report.hourly_data.len());
for (case_id, breakdown) in &diagnostic_report.energy_breakdowns {
    breakdown.print(case_id);
}

// Export hourly CSV
if let Some(path) = &diagnostic_report.config.hourly_output_path {
    diagnostic_report.export_hourly_csv(path)?;
}
```

### Single Case Validation

Validate a single case with diagnostics:

```rust
use fluxion::validation::{ASHRAE140Validator, ASHRAE140Case};

let mut validator = ASHRAE140Validator::new();
let (report, collector) = validator.validate_single_case_with_diagnostics(
    ASHRAE140Case::Case600
);

// Access diagnostic collector
println!("Heating: {:.2} MWh", collector.energy_breakdowns["600"].heating_mwh);
println!("Peak heating: {:.2} kW", collector.peak_timings["600"].peak_heating_kw);
```

## Output Formats

### Hourly CSV

```
Hour,Month,Day,HourOfDay,OutdoorTemp,ZoneTemp,MassTemp,SolarGain,InternalLoad,HVACHeating,HVACCooling,InfiltrationLoss,EnvelopeConduction
0,1,1,0,10.50,20.10,19.80,100.00,200.00,500.00,0.00,50.00,25.00
1,1,1,1,11.00,20.20,19.90,150.00,200.00,480.00,0.00,45.00,24.00
...
```

### Markdown Report

```markdown
# ASHRAE 140 Diagnostic Report

## Energy Breakdowns

Case 600 Energy Breakdown:
  Envelope conduction: 2.500 MWh
  Infiltration:        1.000 MWh
  Solar gains:         5.000 MWh
  Internal gains:      3.000 MWh
  ─────────────────────────────────
  Heating energy:      4.000 MWh
  Cooling energy:      6.000 MWh
  Net balance:         5.000 MWh

## Peak Load Timing

Case 600 Peak Load Timing:
  | Type    | Peak (kW) | Time                |
  |---------|-----------|---------------------|
  | Heating |      5.50 | Jan 6 03:00        |
  | Cooling |      3.20 | Aug 20 19:00       |

## Temperature Profiles (Free-Floating Cases)

| Case | Min Temp (°C) | Max Temp (°C) | Avg Temp (°C) | Swing (°C) |
|------|---------------|---------------|---------------|------------|
| 600FF | 15.0 | 25.0 | 20.0 | 10.0 |

## Validation Comparison Table

| Case | Metric | Fluxion | Ref Min | Ref Max | Deviation | Status |
|------|--------|---------|---------|---------|-----------|--------|
| 600 | Heating | 5.00 | 4.30 | 5.71 | -0.1% | PASS |
```

## Debugging Workflow

### 1. Identify the Problem

Run validation with diagnostics enabled:

```bash
ASHRAE_140_DEBUG=1 ASHRAE_140_VERBOSE=1 cargo test --test ashrae_140_validation
```

Look for cases with large deviations or failures in the console output.

### 2. Analyze Energy Breakdown

Check which components contribute to energy use:

```bash
ASHRAE_140_DEBUG=1 cargo test 2>&1 | grep -A 10 "Energy Breakdown"
```

Look for:
- High envelope conduction → Check U-values, surface areas
- High infiltration → Check ACH rates
- High solar gains → Check SHGC, window areas, shading
- High internal gains → Check load schedules

### 3. Check Peak Timing

Verify that peaks occur at expected times:

```bash
ASHRAE_140_DEBUG=1 cargo test 2>&1 | grep -A 5 "Peak Load"
```

Peak heating should occur in winter (Jan/Feb), peak cooling in summer (Jul/Aug).

### 4. Export Hourly Data

Export hourly data for detailed analysis:

```bash
ASHRAE_140_DEBUG=1 ASHRAE_140_HOURLY_OUTPUT=1 \
  ASHRAE_140_HOURLY_PATH=case600.csv \
  cargo test --test ashrae_140_validation
```

Load into spreadsheet or plotting tool to visualize:
- Temperature profiles
- HVAC power curves
- Solar gain patterns
- Load duration curves

### 5. Compare with Reference

Use the validation comparison table to see how Fluxion results compare to reference programs:

```bash
ASHRAE_140_DEBUG=1 cargo test 2>&1 | grep -A 20 "Validation Comparison"
```

Look for systematic biases (e.g., always high/low) that indicate calibration issues.

## Performance Considerations

Diagnostic collection has minimal performance impact when disabled:

- **Disabled**: ~0.1% overhead (early returns in collector methods)
- **Enabled, no output**: ~1-2% overhead (data structures allocated but minimal recording)
- **Enabled, hourly output**: ~5-10% overhead (full hourly data collection and file I/O)

For production validation runs, use `DiagnosticConfig::disabled()` or omit the config parameter.

## Examples

See the test files for comprehensive examples:

- `tests/ashrae_140_diagnostic_test.rs` - Unit tests for all diagnostic features
- `tests/diagnostic_demo.rs` - Demonstration of diagnostic capabilities

## Related Issues

- Issue #282: Enhancement - Add diagnostic output and debugging tools
- Issue #292: Implementation of diagnostic features
- Issues #271-#281: Investigation issues supported by diagnostics
