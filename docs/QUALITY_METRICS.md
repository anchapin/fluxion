# Quality Metrics Tracker

*Generated: 2026-07-26 15:14 UTC

## Current Status

- **Pass Rate:** 5.6% (1 / 18 cases)
- **MAE:** 110.79%
- **Max Deviation:** 1417.86%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| FAIL | 44 | 68.8% |
| PASS | 11 | 17.2% |
| WARN | 9 | 14.1% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 5.6% | 110.8% | 1418% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 950 | Annual Cooling Energy (kWh) | 86.14 | 0.39-0.92 | 1417.9% | ModelLimitation |
| 950 | Peak Cooling Load (kW) | 73.11 | 0.70-0.90 | 1108.5% | Unknown |
| 910 | Annual Cooling Energy (kWh) | 8.87 | 0.82-1.88 | 556.9% | ModelLimitation |
| 900 | Annual Heating Energy (kWh) | 6.64 | 1.17-2.04 | 313.9% | ModelLimitation |
| 910 | Annual Heating Energy (kWh) | 7.69 | 1.51-2.28 | 305.8% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 26.62 | N/A | 262.1% | Unknown |
| 910 | Peak Cooling Load (kW) | 4.85 | 1.20-1.60 | 246.5% | Unknown |
| 900 | Annual Cooling Energy (kWh) | 9.52 | 2.13-3.67 | 228.3% | ModelLimitation |
| 950 | Annual Heating Energy (kWh) | 18.73 | N/A | 192.6% | ModelLimitation |
| 900 | Peak Heating Load (kW) | 4.65 | 1.80-2.40 | 190.7% | Unknown |
| 940 | Annual Cooling Energy (kWh) | 14.01 | 2.08-3.55 | 176.2% | ModelLimitation |
| 900FF | Minimum Free-Floating Temperature (°C) | -10.17 | -6.40--1.60 | 154.3% | ThermalMass |
| 920 | Annual Heating Energy (kWh) | 8.48 | 3.26-4.30 | 124.5% | ModelLimitation |
| 920 | Annual Cooling Energy (kWh) | 8.44 | 1.84-3.31 | 123.2% | ModelLimitation |
| 650 | Peak Cooling Load (kW) | 4.88 | 1.90-2.50 | 121.9% | SolarGains |
| 910 | Peak Heating Load (kW) | 4.70 | 1.90-2.50 | 113.8% | Unknown |
| 195 | Peak Heating Load (kW) | 3.70 | 1.40-2.20 | 105.5% | Unknown |
| 930 | Annual Heating Energy (kWh) | 8.89 | 4.14-5.34 | 87.5% | ModelLimitation |
| 900 | Peak Cooling Load (kW) | 5.01 | 1.60-2.10 | 79.0% | Unknown |
| 940 | Peak Cooling Load (kW) | 9.62 | 1.70-2.30 | 76.5% | Unknown |
| 610 | Peak Cooling Load (kW) | 4.42 | 2.20-2.90 | 73.3% | SolarGains |
| 940 | Peak Heating Load (kW) | 11.20 | 1.90-2.50 | 63.5% | Unknown |
| 930 | Annual Cooling Energy (kWh) | 7.58 | 1.04-2.24 | 59.8% | ModelLimitation |
| 640 | Peak Cooling Load (kW) | 5.13 | 2.80-3.70 | 57.8% | SolarGains |
| 195 | Annual Heating Energy (kWh) | 7.37 | 3.50-6.00 | 55.1% | Unknown |
| 960 | Annual Cooling Energy (kWh) | 9.57 | 1.55-2.78 | 53.2% | InterZoneTransfer |
| 950FF | Maximum Free-Floating Temperature (°C) | 53.61 | 35.50-38.50 | 44.9% | ThermalMass |
| 960 | Peak Heating Load (kW) | 4.70 | 2.00-8.00 | 44.7% | Unknown |
| 630 | Peak Cooling Load (kW) | 2.96 | 1.80-2.40 | 40.7% | SolarGains |
| 940 | Annual Heating Energy (kWh) | 8.81 | 0.79-1.41 | 38.3% | ModelLimitation |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 950 | 4 | 2981.1% |
| 910 | 4 | 1223.0% |
| 900 | 4 | 811.9% |
| 940 | 4 | 354.4% |
| 920 | 4 | 310.5% |
| 900FF | 2 | 181.3% |
| 195 | 2 | 160.6% |
| 930 | 3 | 157.7% |
| 650 | 2 | 147.8% |
| 960 | 4 | 134.5% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
