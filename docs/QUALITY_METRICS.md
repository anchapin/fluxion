# Quality Metrics Tracker

*Generated: 2026-08-16 18:24 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 21 cases)
- **MAE:** 51.07%
- **Max Deviation:** 470.11%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| WARN | 8 | 9.5% |
| FAIL | 64 | 76.2% |
| PASS | 12 | 14.3% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 51.1% | 470% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 910 | Annual Cooling Energy (kWh) | 7.70 | 0.82-1.88 | 470.1% | ModelLimitation |
| 900 | Annual Heating Energy (kWh) | 5.05 | 1.17-2.04 | 214.8% | ModelLimitation |
| 970 | Annual Cooling Energy (kWh) | 21.07 | 7.39-10.00 | 201.0% | ModelLimitation |
| 910 | Annual Heating Energy (kWh) | 5.43 | 1.51-2.28 | 186.5% | ModelLimitation |
| 900 | Annual Cooling Energy (kWh) | 7.75 | 2.13-3.67 | 167.4% | ModelLimitation |
| 900 | Peak Heating Load (kW) | 3.93 | 1.80-2.40 | 145.8% | Unknown |
| 910 | Peak Cooling Load (kW) | 3.36 | 1.20-1.60 | 140.1% | Unknown |
| 970 | Annual Heating Energy (kWh) | 18.58 | 10.54-14.26 | 135.2% | ModelLimitation |
| 940 | Annual Cooling Energy (kWh) | 11.06 | 2.08-3.55 | 118.0% | ModelLimitation |
| 950 | Annual Heating Energy (kWh) | 0.00 | N/A | 100.0% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 950 | Annual Cooling Energy (kWh) | 0.03 | 0.39-0.92 | 99.4% | ModelLimitation |
| 950 | Peak Cooling Load (kW) | 0.39 | 0.70-0.90 | 93.5% | Unknown |
| 910 | Peak Heating Load (kW) | 3.93 | 1.90-2.50 | 78.8% | Unknown |
| 810 | Annual Cooling Energy (kWh) | 7.75 | 3.80-5.00 | 76.2% | Unknown |
| 810 | Annual Cooling Energy (kWh) | 7.75 | 3.80-5.00 | 76.2% | Unknown |
| 920 | Annual Cooling Energy (kWh) | 6.46 | 1.84-3.31 | 71.0% | ModelLimitation |
| 970 | Peak Cooling Load (kW) | 2.58 | 2.50-5.50 | 66.7% | Unknown |
| 900FF | Minimum Free-Floating Temperature (°C) | -6.65 | -6.40--1.60 | 66.4% | ThermalMass |
| 600 | Annual Cooling Energy (kWh) | 3.30 | 3.92-6.14 | 61.2% | Unknown |
| 970 | Peak Heating Load (kW) | 3.80 | 4.00-8.00 | 58.9% | Unknown |
| 650 | Annual Cooling Energy (kWh) | 2.46 | 4.82-7.06 | 58.5% | Unknown |
| 800 | Annual Cooling Energy (kWh) | 2.61 | 5.00-6.50 | 54.6% | Unknown |
| 800 | Annual Cooling Energy (kWh) | 2.61 | 5.00-6.50 | 54.6% | Unknown |
| 640 | Annual Cooling Energy (kWh) | 3.29 | 5.95-8.10 | 53.2% | Unknown |
| 650 | Peak Cooling Load (kW) | 3.35 | 1.90-2.50 | 52.3% | SolarGains |
| 960 | Peak Heating Load (kW) | 4.16 | 2.00-8.00 | 51.0% | Unknown |
| 610 | Annual Cooling Energy (kWh) | 2.67 | 3.92-6.14 | 47.0% | Unknown |
| 960 | Peak Cooling Load (kW) | 3.64 | 0.00-4.00 | 46.0% | Unknown |
| 620 | Annual Cooling Energy (kWh) | 2.36 | 3.20-5.00 | 42.4% | Unknown |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 910 | 4 | 875.5% |
| 900 | 4 | 547.8% |
| 970 | 4 | 461.8% |
| 950 | 4 | 393.0% |
| 810 | 6 | 286.3% |
| 800 | 6 | 259.0% |
| 940 | 3 | 162.8% |
| 960 | 3 | 138.7% |
| 920 | 3 | 124.8% |
| 600 | 3 | 123.9% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
