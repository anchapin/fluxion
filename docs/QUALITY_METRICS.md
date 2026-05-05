# Quality Metrics Tracker

*Generated: 2026-05-05 06:23 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 142.37%
- **Max Deviation:** 803.33%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| PASS | 6 | 9.4% |
| FAIL | 57 | 89.1% |
| WARN | 1 | 1.6% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 142.4% | 803% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 910 | Peak Cooling Load (kW) | 12.65 | 1.20-1.60 | 803.3% | Unknown |
| 920 | Annual Cooling Energy (MWh) | 24.88 | 1.84-3.31 | 558.2% | ModelLimitation |
| 900FF | Minimum Free-Floating Temperature (°C) | -22.37 | -6.40--1.60 | 459.1% | ThermalMass |
| 900 | Peak Cooling Load (kW) | 15.37 | 1.60-2.10 | 449.1% | Unknown |
| 910 | Annual Cooling Energy (MWh) | 7.41 | 0.82-1.88 | 448.6% | ModelLimitation |
| 650 | Peak Cooling Load (kW) | 11.71 | 1.90-2.50 | 432.2% | SolarGains |
| 630 | Peak Cooling Load (kW) | 10.23 | 1.80-2.40 | 387.1% | SolarGains |
| 930 | Annual Cooling Energy (MWh) | 21.35 | 1.04-2.24 | 350.4% | ModelLimitation |
| 900 | Annual Cooling Energy (MWh) | 12.87 | 2.13-3.67 | 343.8% | ModelLimitation |
| 630 | Annual Cooling Energy (MWh) | 11.79 | 2.13-3.70 | 304.4% | Unknown |
| 610 | Peak Cooling Load (kW) | 9.46 | 2.20-2.90 | 271.1% | SolarGains |
| 960 | Annual Cooling Energy (MWh) | 22.99 | 1.55-2.78 | 267.8% | InterZoneTransfer |
| 640 | Peak Cooling Load (kW) | 11.88 | 2.80-3.70 | 265.5% | SolarGains |
| 920 | Peak Cooling Load (kW) | 13.81 | 1.40-1.90 | 265.4% | Unknown |
| 620 | Annual Cooling Energy (MWh) | 14.65 | 3.20-5.00 | 257.3% | Unknown |
| 620 | Peak Cooling Load (kW) | 10.63 | 2.50-3.50 | 254.3% | SolarGains |
| 950FF | Maximum Free-Floating Temperature (°C) | 123.99 | 35.50-38.50 | 235.1% | ThermalMass |
| 930 | Peak Cooling Load (kW) | 13.60 | 1.10-1.50 | 186.8% | Unknown |
| 900FF | Maximum Free-Floating Temperature (°C) | 125.49 | 41.80-46.40 | 184.6% | ThermalMass |
| 940 | Peak Cooling Load (kW) | 15.37 | 1.70-2.30 | 181.9% | Unknown |
| 610 | Annual Cooling Energy (MWh) | 12.72 | 3.92-6.14 | 153.0% | Unknown |
| 900 | Peak Heating Load (kW) | 4.04 | 1.80-2.40 | 152.5% | Unknown |
| 950 | Peak Cooling Load (kW) | 15.13 | 0.70-0.90 | 150.1% | Unknown |
| 650 | Annual Cooling Energy (MWh) | 13.84 | 4.82-7.06 | 132.9% | Unknown |
| 640 | Annual Cooling Energy (MWh) | 16.06 | 5.95-8.10 | 128.7% | Unknown |
| 940 | Annual Cooling Energy (MWh) | 11.40 | 2.08-3.55 | 124.6% | ModelLimitation |
| 600 | Peak Cooling Load (kW) | 11.89 | 4.80-6.20 | 124.3% | SolarGains |
| 960 | Peak Cooling Load (kW) | 15.00 | 0.00-4.00 | 122.2% | Unknown |
| 640 | Annual Heating Energy (MWh) | 6.89 | 2.75-3.80 | 110.4% | Unknown |
| 610 | Annual Heating Energy (MWh) | 10.59 | 4.36-5.79 | 108.6% | Unknown |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 910 | 3 | 1337.3% |
| 900 | 4 | 958.0% |
| 920 | 3 | 866.6% |
| 630 | 4 | 781.3% |
| 620 | 4 | 683.8% |
| 900FF | 2 | 643.7% |
| 930 | 4 | 573.5% |
| 610 | 4 | 566.6% |
| 650 | 2 | 565.1% |
| 640 | 4 | 536.1% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
