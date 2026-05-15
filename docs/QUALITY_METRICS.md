# Quality Metrics Tracker

*Generated: 2026-05-15 22:00 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 78.71%
- **Max Deviation:** 216.70%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| FAIL | 57 | 89.1% |
| WARN | 1 | 1.6% |
| PASS | 6 | 9.4% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 78.7% | 217% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 650 | Peak Cooling Load (kW) | 6.97 | 1.90-2.50 | 216.7% | SolarGains |
| 195 | Annual Heating Energy (MWh) | 13.34 | 3.50-6.00 | 180.8% | Unknown |
| 900FF | Minimum Free-Floating Temperature (°C) | -10.84 | -6.40--1.60 | 170.9% | ThermalMass |
| 640 | Annual Heating Energy (MWh) | 8.66 | 2.75-3.80 | 164.4% | Unknown |
| 640 | Peak Heating Load (kW) | 11.58 | 4.30-5.70 | 131.7% | Unknown |
| 610 | Annual Heating Energy (MWh) | 11.66 | 4.36-5.79 | 129.7% | Unknown |
| 620 | Peak Cooling Load (kW) | 6.67 | 2.50-3.50 | 122.5% | SolarGains |
| 640 | Peak Cooling Load (kW) | 7.10 | 2.80-3.70 | 118.3% | SolarGains |
| 630 | Peak Cooling Load (kW) | 4.57 | 1.80-2.40 | 117.6% | SolarGains |
| 610 | Peak Cooling Load (kW) | 5.14 | 2.20-2.90 | 101.4% | SolarGains |
| 900 | Annual Heating Energy (MWh) | 0.00 | 1.17-2.04 | 100.0% | ModelLimitation |
| 900 | Annual Cooling Energy (MWh) | 0.00 | 2.13-3.67 | 100.0% | ModelLimitation |
| 900 | Peak Heating Load (kW) | 0.00 | 1.80-2.40 | 100.0% | Unknown |
| 900 | Peak Cooling Load (kW) | 0.00 | 1.60-2.10 | 100.0% | Unknown |
| 910 | Annual Heating Energy (MWh) | 0.00 | 1.51-2.28 | 100.0% | ModelLimitation |
| 910 | Annual Cooling Energy (MWh) | 0.00 | 0.82-1.88 | 100.0% | ModelLimitation |
| 910 | Peak Heating Load (kW) | 0.00 | 1.90-2.50 | 100.0% | Unknown |
| 910 | Peak Cooling Load (kW) | 0.00 | 1.20-1.60 | 100.0% | Unknown |
| 920 | Annual Heating Energy (MWh) | 0.00 | 3.26-4.30 | 100.0% | ModelLimitation |
| 920 | Annual Cooling Energy (MWh) | 0.00 | 1.84-3.31 | 100.0% | ModelLimitation |
| 920 | Peak Heating Load (kW) | 0.00 | 2.10-2.80 | 100.0% | Unknown |
| 920 | Peak Cooling Load (kW) | 0.00 | 1.40-1.90 | 100.0% | Unknown |
| 930 | Annual Heating Energy (MWh) | 0.00 | 4.14-5.34 | 100.0% | ModelLimitation |
| 930 | Annual Cooling Energy (MWh) | 0.00 | 1.04-2.24 | 100.0% | ModelLimitation |
| 930 | Peak Heating Load (kW) | 0.00 | 2.30-3.00 | 100.0% | Unknown |
| 930 | Peak Cooling Load (kW) | 0.00 | 1.10-1.50 | 100.0% | Unknown |
| 940 | Annual Heating Energy (MWh) | 0.00 | 0.79-1.41 | 100.0% | ModelLimitation |
| 940 | Annual Cooling Energy (MWh) | 0.00 | 2.08-3.55 | 100.0% | ModelLimitation |
| 940 | Peak Heating Load (kW) | 0.00 | 1.90-2.50 | 100.0% | Unknown |
| 940 | Peak Cooling Load (kW) | 0.00 | 1.70-2.30 | 100.0% | Unknown |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 640 | 4 | 460.0% |
| 910 | 4 | 400.0% |
| 930 | 4 | 400.0% |
| 920 | 4 | 400.0% |
| 940 | 4 | 400.0% |
| 900 | 4 | 400.0% |
| 950 | 4 | 400.0% |
| 960 | 4 | 342.9% |
| 610 | 4 | 302.1% |
| 620 | 3 | 284.7% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
