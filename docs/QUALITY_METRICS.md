# Quality Metrics Tracker

*Generated: 2026-08-16 14:49 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 50.68%
- **Max Deviation:** 470.11%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| WARN | 6 | 9.4% |
| FAIL | 49 | 76.6% |
| PASS | 9 | 14.1% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 50.7% | 470% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 910 | Annual Cooling Energy (kWh) | 7.70 | 0.82-1.88 | 470.1% | ModelLimitation |
| 900 | Annual Heating Energy (kWh) | 5.05 | 1.17-2.04 | 214.8% | ModelLimitation |
| 910 | Annual Heating Energy (kWh) | 5.43 | 1.51-2.28 | 186.5% | ModelLimitation |
| 900 | Annual Cooling Energy (kWh) | 7.75 | 2.13-3.67 | 167.4% | ModelLimitation |
| 900 | Peak Heating Load (kW) | 3.93 | 1.80-2.40 | 145.8% | Unknown |
| 910 | Peak Cooling Load (kW) | 3.36 | 1.20-1.60 | 140.1% | Unknown |
| 940 | Annual Cooling Energy (kWh) | 11.06 | 2.08-3.55 | 118.0% | ModelLimitation |
| 950 | Annual Heating Energy (kWh) | 0.00 | N/A | 100.0% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 950 | Annual Cooling Energy (kWh) | 0.03 | 0.39-0.92 | 99.4% | ModelLimitation |
| 950 | Peak Cooling Load (kW) | 0.39 | 0.70-0.90 | 93.5% | Unknown |
| 910 | Peak Heating Load (kW) | 3.93 | 1.90-2.50 | 78.8% | Unknown |
| 920 | Annual Cooling Energy (kWh) | 6.46 | 1.84-3.31 | 71.0% | ModelLimitation |
| 900FF | Minimum Free-Floating Temperature (°C) | -6.65 | -6.40--1.60 | 66.4% | ThermalMass |
| 600 | Annual Cooling Energy (kWh) | 3.30 | 3.92-6.14 | 61.2% | Unknown |
| 650 | Annual Cooling Energy (kWh) | 2.46 | 4.82-7.06 | 58.5% | Unknown |
| 960 | Peak Heating Load (kW) | 3.79 | 2.00-8.00 | 55.4% | Unknown |
| 640 | Annual Cooling Energy (kWh) | 3.29 | 5.95-8.10 | 53.2% | Unknown |
| 650 | Peak Cooling Load (kW) | 3.35 | 1.90-2.50 | 52.3% | SolarGains |
| 960 | Peak Cooling Load (kW) | 3.42 | 0.00-4.00 | 49.3% | Unknown |
| 610 | Annual Cooling Energy (kWh) | 2.67 | 3.92-6.14 | 47.0% | Unknown |
| 620 | Annual Cooling Energy (kWh) | 2.36 | 3.20-5.00 | 42.4% | Unknown |
| 920 | Annual Heating Energy (kWh) | 5.35 | 3.26-4.30 | 41.6% | ModelLimitation |
| 620 | Peak Heating Load (kW) | 4.49 | 2.80-3.80 | 36.1% | Unknown |
| 940 | Peak Cooling Load (kW) | 7.38 | 1.70-2.30 | 35.5% | Unknown |
| 610 | Peak Cooling Load (kW) | 3.45 | 2.20-2.90 | 35.1% | SolarGains |
| 930 | Annual Cooling Energy (kWh) | 6.32 | 1.04-2.24 | 33.3% | ModelLimitation |
| 600 | Peak Heating Load (kW) | 4.38 | 2.80-3.80 | 32.8% | Unknown |
| 195 | Annual Heating Energy (kWh) | 3.24 | 3.95-4.22 | 31.8% | Unknown |
| 630 | Annual Cooling Energy (kWh) | 2.03 | 2.13-3.70 | 30.4% | Unknown |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 910 | 4 | 875.5% |
| 900 | 4 | 547.8% |
| 950 | 4 | 393.0% |
| 940 | 3 | 162.8% |
| 960 | 4 | 159.3% |
| 920 | 3 | 124.8% |
| 600 | 3 | 123.9% |
| 650 | 2 | 110.8% |
| 930 | 4 | 105.7% |
| 640 | 3 | 99.5% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
