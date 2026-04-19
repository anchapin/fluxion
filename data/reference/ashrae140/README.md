# ASHRAE 140 Reference Data

This directory contains reference data for ASHRAE 140 validation cases used by Fluxion's validation framework.

## Data Format Specification

### CSV Format

All reference data files use comma-separated values (CSV) format with the following conventions:

- **Header row**: First row contains column names
- **Comma delimiter**: Fields are separated by commas
- **No quotes**: Numeric values are not quoted unless necessary
- **Decimal separator**: Period (.) for decimal numbers
- **Missing values**: Empty fields represent missing/optional data

### Column Definitions

#### series_800.csv (HVAC Equipment Cases 800-810)

| Column | Type | Description | Units |
|--------|------|-------------|-------|
| case | integer | ASHRAE 140 case number (800-810) | - |
| hour | integer | Hour of year (1-8760) | - |
| zone1_temp | float | Zone 1 temperature | °C |
| zone1_heating | float | Zone 1 heating energy | W |
| zone1_cooling | float | Zone 1 cooling energy | W |
| zone2_temp | float | Zone 2 temperature (optional) | °C |
| zone2_heating | float | Zone 2 heating energy (optional) | W |
| zone2_cooling | float | Zone 2 cooling energy (optional) | W |
| total_energy | float | Total building energy consumption | W |

#### series_195.csv (Diagnostic Cases 195-470)

| Column | Type | Description | Units |
|--------|------|-------------|-------|
| case | integer | ASHRAE 140 case number (195-470) | - |
| hour | integer | Hour of year (1-8760) | - |
| zone1_temp | float | Zone 1 temperature | °C |
| zone1_heating | float | Zone 1 heating energy | W |
| zone1_cooling | float | Zone 1 cooling energy | W |
| total_energy | float | Total building energy consumption | W |
| peak_load | float | Peak load for the hour | W |

## Source of Reference Data

Reference data is derived from:

1. **ASHRAE Standard 140-2017**: Published reference values for standard cases
2. **Synthetic generation**: For cases without published reference data, values are generated using:
   - Base values from similar validated cases
   - Adjustments based on case parameters (thermal mass, window area, HVAC equipment)
   - Documented generation methodology

## Usage Instructions

### Loading Reference Data in Rust

```rust
use fluxion::validation::reference_data::load_csv_reference;

// Load reference data for a specific case
let references = load_csv_reference("data/reference/ashrae140/series_800.csv")?;

// Find reference data for Case 800
let case_800_ref = references.iter()
    .find(|r| r.case_id == "800")
    .expect("Case 800 reference not found");
```

### Data Validation

Reference data is validated for:
- **Completeness**: All 8760 hours per case
- **Reasonableness**: Temperature ranges, energy values within expected bounds
- **Consistency**: Format matches specification

## File Organization

- `series_800.csv`: Cases 800-810 (HVAC equipment validation)
- `series_195.csv`: Cases 195-470 (diagnostic validation)
- `README.md`: This documentation file

## Data Generation Methodology

For synthetic data generation:

1. **Base case selection**: Choose most similar validated case
2. **Parameter adjustment**: Apply scaling factors based on case differences
3. **Validation**: Ensure generated data falls within ASHRAE 140 tolerance bands
4. **Documentation**: Record generation parameters and assumptions

## Contact

For questions about reference data format or content, please refer to the Fluxion documentation or open an issue in the GitHub repository.