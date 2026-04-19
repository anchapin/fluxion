# ASHRAE 140 Case 195 Calibration System

## Overview

This document describes the Case 195 calibration system implemented for the Fluxion building energy modeling engine. The system provides automated tuning of thermal model parameters to match ASHRAE 140 reference ranges.

## Calibration Results Summary

### Case 195 Calibration Status

- **Initial Results**: 11.90 MWh annual heating (target: 3.50-6.00 MWh)
- **Initial Peak**: 3.93 kW peak heating (target: 1.40-2.20 kW)
- **Calibration Status**: ❌ Did not converge within tolerance
- **Final Error**: 1.4083 (150.43% annual heating error, 118.44% peak heating error)

### Case Series Testing (195-470)

The system successfully tested multiple cases in the 195-470 series:

- **Case 195**: Solid conduction test case (calibration target)
- **Case 196**: Lighting diagnostics
- **Case 197**: Equipment diagnostics  
- **Case 198**: Occupancy diagnostics

## Usage Examples

### Running Case 195 Calibration

```bash
# Run Case 195 calibration with default parameters
cargo run --bin fluxion -- validation calibrate-case-195

# Run with custom parameters
cargo run --bin fluxion -- validation calibrate-case-195 \
  --max-iterations 100 \
  --learning-rate 0.1 \
  --tolerance 0.001 \
  --output ./custom_calibration
```

### Running Individual Cases

```bash
# Run a single ASHRAE 140 case
cargo run --bin fluxion -- validation run 195 --output ./results

# Run Case 196 (lighting diagnostics)
cargo run --bin fluxion -- validation run 196 --output ./results
```

### Running Case Series

```bash
# Run the diagnostic case series (195-470)
cargo run --bin fluxion -- validation run-series 195-470 --output ./results
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Calibration does not converge

**Symptoms**: 
- Final error remains high (> 0.1)
- Results are far from target ranges
- "Calibration did not converge within tolerance" message

**Solutions**:
1. **Increase max_iterations**: Try values like 100-200
   ```bash
   cargo run --bin fluxion -- validation calibrate-case-195 --max-iterations 200
   ```

2. **Adjust learning_rate**: Try smaller values (0.01-0.05) for finer tuning
   ```bash
   cargo run --bin fluxion -- validation calibrate-case-195 --learning-rate 0.02
   ```

3. **Check parameter bounds**: Ensure parameters can physically reach target values

#### Issue 2: Zero results for diagnostic cases

**Symptoms**:
- Annual heating/cooling results are 0.0 MWh
- Case completes quickly with no errors

**Solutions**:
1. **Verify case configuration**: Check that the case has proper loads configured
2. **Check weather data**: Ensure weather data is available and correct
3. **Run with verbose mode**: Add `--verbose` flag to see detailed output

#### Issue 3: Missing reference data

**Symptoms**:
- "Reference data not found" errors
- Validation fails with missing target ranges

**Solutions**:
1. **Check reference files**: Ensure `docs/ashrae_140_references.json` exists
2. **Update references**: Add missing case references to the JSON file
3. **Use default ranges**: Modify calibration code to use fallback ranges

## Technical Details

### Calibration Algorithm

The system uses a gradient descent optimization approach:

1. **Parameter Space**: Wall U-value, roof U-value, floor U-value, convection coefficients, thermal mass factor
2. **Error Function**: Weighted combination of annual heating error (70%) and peak heating error (30%)
3. **Optimization**: Iterative parameter adjustment with learning rate control

### Calibration Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `wall_u_value` | 0.5 W/m²K | Wall thermal transmittance |
| `roof_u_value` | 0.3 W/m²K | Roof thermal transmittance |
| `floor_u_value` | 0.4 W/m²K | Floor thermal transmittance |
| `h_ce` | 25.0 W/m²K | External convection coefficient |
| `h_ci` | 8.3 W/m²K | Internal convection coefficient |
| `thermal_mass_factor` | 1.0 | Thermal mass adjustment |

### Target Ranges (Case 195)

| Metric | Minimum | Maximum |
|--------|---------|---------|
| Annual Heating | 3.50 MWh | 6.00 MWh |
| Peak Heating | 1.40 kW | 2.20 kW |
| Annual Cooling | 0.00 MWh | 0.00 MWh |
| Peak Cooling | 0.00 kW | 0.00 kW |

## Files Generated

### Calibration Outputs

- `case_195_calibration_results.json`: JSON results with final parameters and error metrics
- `case_*_results.json`: Individual case results in JSON format
- `case_*_report.txt`: Human-readable validation reports

### Example Calibration Results

```json
{
  "success": false,
  "parameters": {
    "wall_u_value": 0.5,
    "roof_u_value": 0.3,
    "floor_u_value": 0.4,
    "h_ce": 25.0,
    "h_ci": 8.3,
    "thermal_mass_factor": 1.0
  },
  "annual_heating": 11.895366762186434,
  "peak_heating": 3.9320016726870044,
  "annual_heating_error": 1.5042877394076704,
  "peak_heating_error": 1.1844453737150025,
  "iterations": 50,
  "final_error": 1.40833502969987
}
```

## Next Steps for Improvement

1. **Enhanced Optimization**: Implement more sophisticated optimization algorithms (e.g., particle swarm, genetic algorithms)
2. **Parameter Constraints**: Add physical constraints to prevent unrealistic parameter values
3. **Multi-case Calibration**: Extend calibration to optimize across multiple cases simultaneously
4. **Reference Data Expansion**: Add complete ASHRAE 140 reference data for all cases
5. **Performance Optimization**: Improve parallel execution for large case batches

## Contact Information

For issues or questions about the calibration system:

- **Repository**: https://github.com/your-org/fluxion
- **Issues**: Report bugs in the GitHub issue tracker
- **Documentation**: See `docs/ASHRAE_140_VALIDATION.md` for complete validation documentation
