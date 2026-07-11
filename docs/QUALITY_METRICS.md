# Quality Metrics Tracker

*Generated: 2026-07-11 06:25 UTC

## Current Status

- **Pass Rate:** 5.6% (1 / 18 cases)
- **MAE:** 114.17%
- **Max Deviation:** 1431.40%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| WARN | 7 | 10.9% |
| PASS | 15 | 23.4% |
| FAIL | 42 | 65.6% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 5.6% | 114.2% | 1431% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 950 | Annual Cooling Energy (kWh) | 86.91 | 0.39-0.92 | 1431.4% | ModelLimitation |
| 950 | Peak Cooling Load (kW) | 66.53 | 0.70-0.90 | 999.7% | Unknown |
| 910 | Annual Cooling Energy (kWh) | 9.99 | 0.82-1.88 | 640.1% | ModelLimitation |
| 950 | Annual Heating Energy (kWh) | 32.21 | N/A | 403.4% | ModelLimitation |
| 900 | Annual Heating Energy (kWh) | 7.75 | 1.17-2.04 | 382.8% | ModelLimitation |
| 910 | Annual Heating Energy (kWh) | 8.97 | 1.51-2.28 | 373.5% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 29.82 | N/A | 305.7% | Unknown |
| 900 | Annual Cooling Energy (kWh) | 9.64 | 2.13-3.67 | 232.5% | ModelLimitation |
| 910 | Peak Cooling Load (kW) | 4.37 | 1.20-1.60 | 212.4% | Unknown |
| 900FF | Minimum Free-Floating Temperature (°C) | -12.43 | -6.40--1.60 | 210.8% | ThermalMass |
| 940 | Annual Cooling Energy (kWh) | 15.51 | 2.08-3.55 | 205.7% | ModelLimitation |
| 900 | Peak Heating Load (kW) | 4.27 | 1.80-2.40 | 167.0% | Unknown |
| 650 | Peak Cooling Load (kW) | 4.88 | 1.90-2.50 | 122.0% | SolarGains |
| 195 | Peak Heating Load (kW) | 3.98 | 1.40-2.20 | 120.8% | Unknown |
| 920 | Annual Heating Energy (kWh) | 8.29 | 3.26-4.30 | 119.3% | ModelLimitation |
| 920 | Annual Cooling Energy (kWh) | 7.89 | 1.84-3.31 | 108.7% | ModelLimitation |
| 910 | Peak Heating Load (kW) | 4.30 | 1.90-2.50 | 95.5% | Unknown |
| 930 | Annual Heating Energy (kWh) | 9.01 | 4.14-5.34 | 90.1% | ModelLimitation |
| 940 | Annual Heating Energy (kWh) | 11.44 | 0.79-1.41 | 79.4% | ModelLimitation |
| 930 | Annual Cooling Energy (kWh) | 8.12 | 1.04-2.24 | 71.2% | ModelLimitation |
| 610 | Peak Cooling Load (kW) | 4.36 | 2.20-2.90 | 70.9% | SolarGains |
| 940 | Peak Cooling Load (kW) | 9.21 | 1.70-2.30 | 68.9% | Unknown |
| 900 | Peak Cooling Load (kW) | 4.57 | 1.60-2.10 | 63.4% | Unknown |
| 195 | Annual Heating Energy (kWh) | 7.66 | 3.50-6.00 | 61.2% | Unknown |
| 960 | Annual Cooling Energy (kWh) | 9.66 | 1.55-2.78 | 54.6% | InterZoneTransfer |
| 640 | Peak Cooling Load (kW) | 4.95 | 2.80-3.70 | 52.2% | SolarGains |
| 960 | Peak Heating Load (kW) | 4.27 | 2.00-8.00 | 49.7% | Unknown |
| 600 | Annual Cooling Energy (kWh) | 5.34 | 3.92-6.14 | 37.2% | Unknown |
| 940 | Peak Heating Load (kW) | 9.33 | 1.90-2.50 | 36.2% | Unknown |
| 630 | Peak Cooling Load (kW) | 2.85 | 1.80-2.40 | 35.5% | SolarGains |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 950 | 4 | 3140.2% |
| 910 | 4 | 1321.5% |
| 900 | 4 | 845.7% |
| 940 | 4 | 390.2% |
| 920 | 4 | 258.9% |
| 900FF | 2 | 231.3% |
| 195 | 2 | 182.1% |
| 930 | 3 | 171.4% |
| 960 | 3 | 136.9% |
| 650 | 1 | 122.0% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
