# Quality Metrics Tracker

*Generated: 2026-08-15 19:34 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 52.23%
- **Max Deviation:** 484.61%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| WARN | 6 | 9.4% |
| FAIL | 51 | 79.7% |
| PASS | 7 | 10.9% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 52.2% | 485% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 910 | Annual Cooling Energy (kWh) | 7.89 | 0.82-1.88 | 484.6% | ModelLimitation |
| 900 | Annual Heating Energy (kWh) | 5.41 | 1.17-2.04 | 236.9% | ModelLimitation |
| 910 | Annual Heating Energy (kWh) | 5.97 | 1.51-2.28 | 214.8% | ModelLimitation |
| 900 | Annual Cooling Energy (kWh) | 7.60 | 2.13-3.67 | 162.2% | ModelLimitation |
| 900 | Peak Heating Load (kW) | 3.99 | 1.80-2.40 | 149.1% | Unknown |
| 910 | Peak Cooling Load (kW) | 3.32 | 1.20-1.60 | 137.4% | Unknown |
| 195 | Peak Heating Load (kW) | 3.65 | 1.40-2.20 | 102.7% | Unknown |
| 950 | Annual Heating Energy (kWh) | 0.00 | N/A | 100.0% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 950 | Annual Cooling Energy (kWh) | 0.03 | 0.39-0.92 | 99.6% | ModelLimitation |
| 950 | Peak Cooling Load (kW) | 0.33 | 0.70-0.90 | 94.5% | Unknown |
| 900FF | Minimum Free-Floating Temperature (°C) | -7.27 | -6.40--1.60 | 81.7% | ThermalMass |
| 910 | Peak Heating Load (kW) | 3.99 | 1.90-2.50 | 81.2% | Unknown |
| 600 | Annual Cooling Energy (kWh) | 3.24 | 3.92-6.14 | 61.9% | Unknown |
| 650 | Annual Cooling Energy (kWh) | 2.42 | 4.82-7.06 | 59.2% | Unknown |
| 920 | Annual Cooling Energy (kWh) | 6.01 | 1.84-3.31 | 59.0% | ModelLimitation |
| 640 | Annual Cooling Energy (kWh) | 3.24 | 5.95-8.10 | 53.9% | Unknown |
| 960 | Peak Heating Load (kW) | 4.00 | 2.00-8.00 | 52.9% | Unknown |
| 650 | Peak Cooling Load (kW) | 3.33 | 1.90-2.50 | 51.3% | SolarGains |
| 940 | Annual Cooling Energy (kWh) | 7.60 | 2.08-3.55 | 49.8% | ModelLimitation |
| 960 | Peak Cooling Load (kW) | 3.41 | 0.00-4.00 | 49.4% | Unknown |
| 610 | Annual Cooling Energy (kWh) | 2.61 | 3.92-6.14 | 48.1% | Unknown |
| 620 | Annual Cooling Energy (kWh) | 2.32 | 3.20-5.00 | 43.5% | Unknown |
| 640 | Annual Heating Energy (kWh) | 4.70 | 2.75-3.80 | 43.4% | Unknown |
| 940 | Peak Heating Load (kW) | 3.99 | 1.90-2.50 | 41.8% | Unknown |
| 920 | Annual Heating Energy (kWh) | 5.32 | 3.26-4.30 | 40.7% | ModelLimitation |
| 940 | Peak Cooling Load (kW) | 3.34 | 1.70-2.30 | 38.8% | Unknown |
| 195 | Annual Heating Energy (kWh) | 6.55 | 3.50-6.00 | 37.9% | Unknown |
| 620 | Peak Heating Load (kW) | 4.53 | 2.80-3.80 | 37.2% | Unknown |
| 610 | Peak Cooling Load (kW) | 3.43 | 2.20-2.90 | 34.4% | SolarGains |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 910 | 4 | 918.1% |
| 900 | 4 | 567.4% |
| 950 | 4 | 394.1% |
| 960 | 4 | 152.4% |
| 940 | 4 | 145.6% |
| 195 | 2 | 140.7% |
| 600 | 3 | 125.9% |
| 920 | 3 | 112.3% |
| 650 | 2 | 110.6% |
| 640 | 2 | 97.3% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
