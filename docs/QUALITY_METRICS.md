# Quality Metrics Tracker

*Generated: 2026-05-13 20:50 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 97.16%
- **Max Deviation:** 489.82%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| WARN | 3 | 4.7% |
| FAIL | 56 | 87.5% |
| PASS | 5 | 7.8% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 97.2% | 490% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 195 | Annual Heating Energy (MWh) | 28.02 | 3.50-6.00 | 489.8% | Unknown |
| 900FF | Minimum Free-Floating Temperature (°C) | -18.47 | -6.40--1.60 | 361.8% | ThermalMass |
| 640 | Annual Heating Energy (MWh) | 13.89 | 2.75-3.80 | 324.2% | Unknown |
| 610 | Annual Heating Energy (MWh) | 19.99 | 4.36-5.79 | 294.0% | Unknown |
| 650 | Peak Cooling Load (kW) | 7.85 | 1.90-2.50 | 256.9% | SolarGains |
| 600 | Annual Heating Energy (MWh) | 19.25 | 5.50-7.50 | 234.8% | Unknown |
| 600 | Peak Heating Load (kW) | 10.42 | 2.80-3.80 | 215.7% | Unknown |
| 620 | Annual Heating Energy (MWh) | 16.60 | 4.50-6.50 | 201.8% | Unknown |
| 620 | Peak Heating Load (kW) | 9.86 | 2.80-3.80 | 198.8% | Unknown |
| 630 | Peak Cooling Load (kW) | 6.22 | 1.80-2.40 | 196.2% | SolarGains |
| 630 | Annual Heating Energy (MWh) | 17.05 | 5.05-6.47 | 196.0% | Unknown |
| 640 | Peak Cooling Load (kW) | 8.11 | 2.80-3.70 | 149.5% | SolarGains |
| 620 | Peak Cooling Load (kW) | 6.86 | 2.50-3.50 | 128.6% | SolarGains |
| 610 | Peak Cooling Load (kW) | 5.61 | 2.20-2.90 | 120.0% | SolarGains |
| 610 | Peak Heating Load (kW) | 10.43 | 4.30-5.70 | 108.7% | Unknown |
| 640 | Peak Heating Load (kW) | 10.30 | 4.30-5.70 | 105.9% | Unknown |
| 950FF | Maximum Free-Floating Temperature (°C) | 74.72 | 35.50-38.50 | 101.9% | ThermalMass |
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

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 640 | 3 | 579.6% |
| 610 | 4 | 549.5% |
| 600 | 4 | 530.0% |
| 620 | 3 | 529.2% |
| 195 | 2 | 519.2% |
| 630 | 3 | 474.9% |
| 900FF | 2 | 417.8% |
| 900 | 4 | 400.0% |
| 920 | 4 | 400.0% |
| 940 | 4 | 400.0% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
