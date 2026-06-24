# Quality Metrics Tracker

*Generated: 2026-06-24 00:54 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 37.59%
- **Max Deviation:** 100.00%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| WARN | 2 | 3.1% |
| FAIL | 50 | 78.1% |
| PASS | 12 | 18.8% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 37.6% | 100% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 900FF | Minimum Free-Floating Temperature (°C) | -13.01 | -6.40--1.60 | 225.3% | ThermalMass |
| 950 | Annual Heating Energy (kWh) | 0.00 | N/A | 100.0% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 960 | Annual Cooling Energy (kWh) | 0.11 | 1.55-2.78 | 98.3% | InterZoneTransfer |
| 960 | Peak Cooling Load (kW) | 0.51 | 0.00-4.00 | 92.4% | Unknown |
| 960 | Peak Heating Load (kW) | 1.05 | 2.00-8.00 | 87.7% | Unknown |
| 940 | Annual Heating Energy (kWh) | 1.19 | 0.79-1.41 | 81.4% | ModelLimitation |
| 930 | Annual Cooling Energy (kWh) | 1.00 | 1.04-2.24 | 79.0% | ModelLimitation |
| 930 | Peak Cooling Load (kW) | 1.05 | 1.10-1.50 | 77.8% | Unknown |
| 940 | Peak Heating Load (kW) | 1.77 | 1.90-2.50 | 74.2% | Unknown |
| 950 | Annual Cooling Energy (kWh) | 1.76 | 0.39-0.92 | 68.9% | ModelLimitation |
| 950 | Peak Cooling Load (kW) | 1.88 | 0.70-0.90 | 68.9% | Unknown |
| 930 | Peak Heating Load (kW) | 1.53 | 2.30-3.00 | 67.6% | Unknown |
| 920 | Peak Cooling Load (kW) | 1.28 | 1.40-1.90 | 66.2% | Unknown |
| 920 | Annual Cooling Energy (kWh) | 1.28 | 1.84-3.31 | 66.1% | ModelLimitation |
| 940 | Peak Cooling Load (kW) | 1.93 | 1.70-2.30 | 64.6% | Unknown |
| 600 | Annual Cooling Energy (kWh) | 3.12 | 8.00-10.50 | 63.3% | Unknown |
| 940 | Annual Cooling Energy (kWh) | 2.02 | 2.08-3.55 | 60.2% | ModelLimitation |
| 920 | Peak Heating Load (kW) | 1.53 | 2.10-2.80 | 59.5% | Unknown |
| 620 | Annual Cooling Energy (kWh) | 1.79 | 3.20-5.00 | 56.4% | Unknown |
| 640 | Annual Cooling Energy (kWh) | 3.08 | 5.95-8.10 | 56.2% | Unknown |
| 630 | Annual Cooling Energy (kWh) | 1.29 | 2.13-3.70 | 55.6% | Unknown |
| 930 | Annual Heating Energy (kWh) | 2.19 | 4.14-5.34 | 53.8% | ModelLimitation |
| 630 | Peak Heating Load (kW) | 2.59 | 4.70-6.10 | 52.0% | Unknown |
| 650 | Annual Cooling Energy (kWh) | 2.88 | 4.82-7.06 | 51.6% | Unknown |
| 610 | Annual Cooling Energy (kWh) | 2.55 | 3.92-6.14 | 49.2% | Unknown |
| 640 | Peak Heating Load (kW) | 2.62 | 4.30-5.70 | 47.6% | Unknown |
| 610 | Peak Heating Load (kW) | 2.63 | 4.30-5.70 | 47.5% | Unknown |
| 920 | Annual Heating Energy (kWh) | 2.14 | 3.26-4.30 | 43.4% | ModelLimitation |
| 600 | Peak Cooling Load (kW) | 3.09 | 4.80-6.20 | 41.6% | SolarGains |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 950 | 4 | 337.8% |
| 960 | 4 | 307.5% |
| 940 | 4 | 280.4% |
| 930 | 4 | 278.2% |
| 900FF | 2 | 235.8% |
| 920 | 4 | 235.2% |
| 600 | 4 | 168.1% |
| 630 | 4 | 157.2% |
| 620 | 4 | 139.5% |
| 640 | 3 | 138.4% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
