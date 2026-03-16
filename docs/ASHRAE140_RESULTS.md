# ASHRAE Standard 140 Validation Results

Latest validation run results for Fluxion v0.1.0.

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Passed | 16 (25%) |
| Warnings | 9 (14%) |
| Failed | 39 (61%) |
| Mean Absolute Error | 78.79% |
| Max Deviation | 471.66% |
| Status | ⚠️ IN PROGRESS |

## Detailed Results

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 10.91 MWh (Ref: 5.50-7.50) | 8.40 MWh (Ref: 8.00-10.50) | 4.81 kW (Ref: 2.80-3.80) | 5.70 kW (Ref: 2.80-4.10) | ⚠️ Partial |
| 610 | 10.91 MWh (Ref: 4.36-5.79) | 8.40 MWh (Ref: 3.92-6.14) | 4.81 kW | 5.70 kW | ⚠️ Partial |
| 620 | 9.59 MWh (Ref: 4.50-6.50) | 3.20 MWh (Ref: 3.20-5.00) | 4.81 kW (Ref: 2.80-3.80) | 3.21 kW | ⚠️ Partial |
| 630 | 9.59 MWh (Ref: 5.05-6.47) | 3.20 MWh (Ref: 2.13-3.70) | 4.81 kW | 3.21 kW | ⚠️ Partial |
| 640 | 10.91 MWh (Ref: 2.75-3.80) | 8.40 MWh (Ref: 5.95-8.10) | 4.81 kW | 5.70 kW | ⚠️ Partial |
| 650 | 0.00 MWh | 7.54 MWh (Ref: 4.82-7.06) | 0.00 kW | 6.85 kW | ⚠️ Partial |
| 900 | 6.17 MWh (Ref: 1.17-2.04) | 2.23 MWh (Ref: 2.13-3.67) | 6.63 kW (Ref: 1.10-2.10) | 4.27 kW (Ref: 2.10-3.50) | ❌ FAIL |
| 960 | 10.20 MWh (Ref: 5.00-15.00) | 2.96 MWh (Ref: 1.00-3.50) | 4.57 kW | 3.01 kW | ✅ PASS |
| 195 | 5.53 MWh (Ref: 3.50-6.00) | 0.00 MWh | 1.87 kW | 0.00 kW | ⚠️ Partial |

*Note: Results shown are from the latest validation run with 64 total validation metrics.*

## Known Issues

The validation reveals systematic discrepancies in heating calculations, particularly for:
- Annual heating loads (consistently over-predicted)
- Peak heating values (significant deviations from reference)
- High-mass building cases (900-series)

These issues are actively being investigated as part of ongoing ASHRAE 140 compliance work.
