# Quality Metrics Tracker

*Generated: 2026-08-16 08:18 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 52.04%
- **Max Deviation:** 476.39%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| FAIL | 50 | 78.1% |
| WARN | 5 | 7.8% |
| PASS | 9 | 14.1% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 52.0% | 476% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 910 | Annual Cooling Energy (kWh) | 7.78 | 0.82-1.88 | 476.4% | ModelLimitation |
| 900 | Annual Heating Energy (kWh) | 5.34 | 1.17-2.04 | 232.7% | ModelLimitation |
| 910 | Annual Heating Energy (kWh) | 5.80 | 1.51-2.28 | 205.8% | ModelLimitation |
| 900 | Annual Cooling Energy (kWh) | 7.66 | 2.13-3.67 | 164.1% | ModelLimitation |
| 900 | Peak Heating Load (kW) | 3.90 | 1.80-2.40 | 143.5% | Unknown |
| 910 | Peak Cooling Load (kW) | 3.34 | 1.20-1.60 | 138.9% | Unknown |
| 940 | Annual Cooling Energy (kWh) | 11.40 | 2.08-3.55 | 124.6% | ModelLimitation |
| 195 | Peak Heating Load (kW) | 3.65 | 1.40-2.20 | 102.7% | Unknown |
| 950 | Annual Heating Energy (kWh) | 0.00 | N/A | 100.0% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 950 | Annual Cooling Energy (kWh) | 0.03 | 0.39-0.92 | 99.5% | ModelLimitation |
| 950 | Peak Cooling Load (kW) | 0.36 | 0.70-0.90 | 94.1% | Unknown |
| 900FF | Minimum Free-Floating Temperature (°C) | -7.27 | -6.40--1.60 | 81.7% | ThermalMass |
| 920 | Annual Cooling Energy (kWh) | 6.20 | 1.84-3.31 | 64.1% | ModelLimitation |
| 600 | Annual Cooling Energy (kWh) | 3.30 | 3.92-6.14 | 61.1% | Unknown |
| 650 | Annual Cooling Energy (kWh) | 2.46 | 4.82-7.06 | 58.5% | Unknown |
| 910 | Peak Heating Load (kW) | 3.43 | 1.90-2.50 | 55.8% | Unknown |
| 960 | Peak Heating Load (kW) | 3.98 | 2.00-8.00 | 53.1% | Unknown |
| 640 | Annual Cooling Energy (kWh) | 3.30 | 5.95-8.10 | 53.1% | Unknown |
| 650 | Peak Cooling Load (kW) | 3.35 | 1.90-2.50 | 52.4% | SolarGains |
| 960 | Peak Cooling Load (kW) | 3.28 | 0.00-4.00 | 51.4% | Unknown |
| 610 | Annual Cooling Energy (kWh) | 2.67 | 3.92-6.14 | 46.9% | Unknown |
| 920 | Annual Heating Energy (kWh) | 5.42 | 3.26-4.30 | 43.4% | ModelLimitation |
| 620 | Annual Cooling Energy (kWh) | 2.37 | 3.20-5.00 | 42.3% | Unknown |
| 195 | Annual Heating Energy (kWh) | 6.55 | 3.50-6.00 | 37.9% | Unknown |
| 620 | Peak Heating Load (kW) | 4.51 | 2.80-3.80 | 36.6% | Unknown |
| 940 | Peak Cooling Load (kW) | 7.42 | 1.70-2.30 | 36.1% | Unknown |
| 610 | Peak Cooling Load (kW) | 3.45 | 2.20-2.90 | 35.2% | SolarGains |
| 600 | Peak Heating Load (kW) | 4.39 | 2.80-3.80 | 33.2% | Unknown |
| 630 | Annual Cooling Energy (kWh) | 2.03 | 2.13-3.70 | 30.3% | Unknown |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 910 | 4 | 876.9% |
| 900 | 4 | 559.5% |
| 950 | 4 | 393.7% |
| 940 | 3 | 178.1% |
| 960 | 4 | 151.7% |
| 195 | 2 | 140.7% |
| 600 | 3 | 124.1% |
| 920 | 3 | 119.5% |
| 650 | 2 | 110.9% |
| 640 | 3 | 99.1% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
