# Quality Metrics Tracker

*Generated: 2026-08-06 03:58 UTC

## Current Status

- **Pass Rate:** 5.6% (1 / 18 cases)
- **MAE:** 103.66%
- **Max Deviation:** 1757.96%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| PASS | 14 | 21.9% |
| FAIL | 41 | 64.1% |
| WARN | 9 | 14.1% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 5.6% | 103.7% | 1758% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 950 | Annual Cooling Energy (kWh) | 105.44 | 0.39-0.92 | 1758.0% | ModelLimitation |
| 950 | Peak Cooling Load (kW) | 68.60 | 0.70-0.90 | 1033.8% | Unknown |
| 910 | Annual Cooling Energy (kWh) | 8.27 | 0.82-1.88 | 512.4% | ModelLimitation |
| 950 | Annual Heating Energy (kWh) | 34.21 | N/A | 434.5% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 27.44 | N/A | 273.3% | Unknown |
| 900 | Annual Heating Energy (kWh) | 5.78 | 1.17-2.04 | 260.4% | ModelLimitation |
| 910 | Annual Heating Energy (kWh) | 6.64 | 1.51-2.28 | 250.3% | ModelLimitation |
| 900FF | Minimum Free-Floating Temperature (°C) | -12.70 | -6.40--1.60 | 217.6% | ThermalMass |
| 900 | Annual Cooling Energy (kWh) | 7.41 | 2.13-3.67 | 155.7% | ModelLimitation |
| 940 | Annual Cooling Energy (kWh) | 12.31 | 2.08-3.55 | 142.5% | ModelLimitation |
| 910 | Peak Cooling Load (kW) | 3.37 | 1.20-1.60 | 141.1% | Unknown |
| 650 | Peak Cooling Load (kW) | 5.11 | 1.90-2.50 | 132.2% | SolarGains |
| 900 | Peak Heating Load (kW) | 3.34 | 1.80-2.40 | 108.5% | Unknown |
| 195 | Peak Heating Load (kW) | 3.70 | 1.40-2.20 | 105.5% | Unknown |
| 610 | Peak Cooling Load (kW) | 5.21 | 2.20-2.90 | 104.3% | SolarGains |
| 920 | Annual Heating Energy (kWh) | 6.37 | 3.26-4.30 | 68.6% | ModelLimitation |
| 920 | Annual Cooling Energy (kWh) | 6.24 | 1.84-3.31 | 65.2% | ModelLimitation |
| 640 | Peak Cooling Load (kW) | 5.35 | 2.80-3.70 | 64.5% | SolarGains |
| 630 | Peak Cooling Load (kW) | 3.40 | 1.80-2.40 | 62.0% | SolarGains |
| 960 | Peak Heating Load (kW) | 3.35 | 2.00-8.00 | 60.6% | Unknown |
| 195 | Annual Heating Energy (kWh) | 7.37 | 3.50-6.00 | 55.1% | Unknown |
| 910 | Peak Heating Load (kW) | 3.34 | 1.90-2.50 | 51.7% | Unknown |
| 960 | Peak Cooling Load (kW) | 3.40 | 0.00-4.00 | 49.6% | Unknown |
| 940 | Peak Cooling Load (kW) | 7.44 | 1.70-2.30 | 36.5% | Unknown |
| 930 | Annual Heating Energy (kWh) | 6.43 | 4.14-5.34 | 35.7% | ModelLimitation |
| 620 | Peak Heating Load (kW) | 4.45 | 2.80-3.80 | 34.7% | Unknown |
| 940 | Annual Heating Energy (kWh) | 8.58 | 0.79-1.41 | 34.6% | ModelLimitation |
| 600 | Annual Cooling Energy (kWh) | 5.69 | 3.92-6.14 | 33.0% | Unknown |
| 600 | Peak Heating Load (kW) | 4.36 | 2.80-3.80 | 32.0% | Unknown |
| 930 | Peak Heating Load (kW) | 3.43 | 2.30-3.00 | 27.7% | Unknown |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 950 | 4 | 3499.6% |
| 910 | 4 | 955.5% |
| 900 | 4 | 545.4% |
| 900FF | 2 | 231.3% |
| 940 | 3 | 213.6% |
| 195 | 2 | 160.6% |
| 960 | 4 | 151.8% |
| 920 | 3 | 144.6% |
| 650 | 1 | 132.2% |
| 930 | 4 | 116.7% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
