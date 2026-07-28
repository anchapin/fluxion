# Quality Metrics Tracker

*Generated: 2026-07-28 01:37 UTC

## Current Status

- **Pass Rate:** 5.6% (1 / 18 cases)
- **MAE:** 201.65%
- **Max Deviation:** 1552.89%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| WARN | 9 | 14.1% |
| PASS | 11 | 17.2% |
| FAIL | 44 | 68.8% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 5.6% | 201.7% | 1553% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 950 | Peak Cooling Load (kW) | 100.00 | 0.70-0.90 | 1552.9% | Unknown |
| 910 | Annual Cooling Energy (kWh) | 19.48 | 0.82-1.88 | 1342.9% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 100.00 | N/A | 1260.5% | Unknown |
| 900 | Annual Heating Energy (kWh) | 14.99 | 1.17-2.04 | 834.1% | ModelLimitation |
| 910 | Annual Heating Energy (kWh) | 17.16 | 1.51-2.28 | 805.8% | ModelLimitation |
| 900 | Annual Cooling Energy (kWh) | 19.09 | 2.13-3.67 | 558.2% | ModelLimitation |
| 950 | Annual Cooling Energy (kWh) | 34.83 | 0.39-0.92 | 513.8% | ModelLimitation |
| 940 | Annual Cooling Energy (kWh) | 30.13 | 2.08-3.55 | 493.7% | ModelLimitation |
| 910 | Peak Cooling Load (kW) | 8.06 | 1.20-1.60 | 475.7% | Unknown |
| 950 | Annual Heating Energy (kWh) | 34.80 | N/A | 443.8% | ModelLimitation |
| 900 | Peak Heating Load (kW) | 8.27 | 1.80-2.40 | 416.9% | Unknown |
| 920 | Annual Heating Energy (kWh) | 15.76 | 3.26-4.30 | 316.8% | ModelLimitation |
| 920 | Annual Cooling Energy (kWh) | 15.10 | 1.84-3.31 | 299.6% | ModelLimitation |
| 910 | Peak Heating Load (kW) | 8.49 | 1.90-2.50 | 285.9% | Unknown |
| 940 | Annual Heating Energy (kWh) | 21.39 | 0.79-1.41 | 235.5% | ModelLimitation |
| 940 | Peak Cooling Load (kW) | 18.20 | 1.70-2.30 | 233.9% | Unknown |
| 930 | Annual Heating Energy (kWh) | 15.77 | 4.14-5.34 | 232.8% | ModelLimitation |
| 900FF | Minimum Free-Floating Temperature (°C) | -12.70 | -6.40--1.60 | 217.6% | ThermalMass |
| 960 | Annual Cooling Energy (kWh) | 19.21 | 1.55-2.78 | 207.3% | InterZoneTransfer |
| 900 | Peak Cooling Load (kW) | 8.28 | 1.60-2.10 | 195.5% | Unknown |
| 930 | Annual Cooling Energy (kWh) | 13.69 | 1.04-2.24 | 188.8% | ModelLimitation |
| 940 | Peak Heating Load (kW) | 15.38 | 1.90-2.50 | 124.6% | Unknown |
| 920 | Peak Heating Load (kW) | 8.46 | 2.10-2.80 | 123.8% | Unknown |
| 650 | Peak Cooling Load (kW) | 4.88 | 1.90-2.50 | 121.9% | SolarGains |
| 920 | Peak Cooling Load (kW) | 8.24 | 1.40-1.90 | 117.9% | Unknown |
| 195 | Peak Heating Load (kW) | 3.70 | 1.40-2.20 | 105.5% | Unknown |
| 960 | Annual Heating Energy (kWh) | 15.08 | 1.65-2.45 | 101.1% | Unknown |
| 930 | Peak Cooling Load (kW) | 8.37 | 1.10-1.50 | 76.5% | Unknown |
| 930 | Peak Heating Load (kW) | 8.29 | 2.30-3.00 | 74.9% | Unknown |
| 610 | Peak Cooling Load (kW) | 4.42 | 2.20-2.90 | 73.3% | SolarGains |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 950 | 4 | 3771.0% |
| 910 | 4 | 2910.2% |
| 900 | 4 | 2004.8% |
| 940 | 4 | 1087.6% |
| 920 | 4 | 858.1% |
| 930 | 4 | 573.0% |
| 960 | 3 | 331.2% |
| 900FF | 2 | 231.3% |
| 195 | 2 | 160.6% |
| 650 | 2 | 147.8% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
