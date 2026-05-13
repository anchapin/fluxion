# Quality Metrics Tracker

*Generated: 2026-05-08 20:40 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 98.22%
- **Max Deviation:** 506.56%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| PASS | 11 | 17.2% |
| WARN | 4 | 6.2% |
| FAIL | 49 | 76.6% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 98.2% | 507% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 910 | Peak Cooling Load (kW) | 8.49 | 1.20-1.60 | 506.6% | Unknown |
| 900 | Peak Cooling Load (kW) | 13.32 | 1.60-2.10 | 375.7% | Unknown |
| 650 | Peak Cooling Load (kW) | 9.76 | 1.90-2.50 | 343.7% | SolarGains |
| 920 | Annual Cooling Energy (MWh) | 15.70 | 1.84-3.31 | 315.3% | ModelLimitation |
| 630 | Peak Cooling Load (kW) | 7.90 | 1.80-2.40 | 276.2% | SolarGains |
| 960 | Annual Cooling Energy (MWh) | 21.90 | 1.55-2.78 | 250.4% | InterZoneTransfer |
| 910 | Annual Cooling Energy (MWh) | 4.66 | 0.82-1.88 | 244.9% | ModelLimitation |
| 900 | Peak Heating Load (kW) | 5.46 | 1.80-2.40 | 241.6% | Unknown |
| 900 | Annual Cooling Energy (MWh) | 9.47 | 2.13-3.67 | 226.7% | ModelLimitation |
| 630 | Annual Cooling Energy (MWh) | 9.32 | 2.13-3.70 | 219.8% | Unknown |
| 640 | Peak Cooling Load (kW) | 10.15 | 2.80-3.70 | 212.4% | SolarGains |
| 920 | Peak Cooling Load (kW) | 11.75 | 1.40-1.90 | 210.8% | Unknown |
| 900FF | Minimum Free-Floating Temperature (°C) | -12.05 | -6.40--1.60 | 201.3% | ThermalMass |
| 620 | Annual Cooling Energy (MWh) | 12.05 | 3.20-5.00 | 194.0% | Unknown |
| 620 | Peak Cooling Load (kW) | 8.68 | 2.50-3.50 | 189.5% | SolarGains |
| 610 | Peak Cooling Load (kW) | 7.23 | 2.20-2.90 | 183.4% | SolarGains |
| 910 | Peak Heating Load (kW) | 5.53 | 1.90-2.50 | 151.4% | Unknown |
| 940 | Peak Cooling Load (kW) | 13.32 | 1.70-2.30 | 144.4% | Unknown |
| 930 | Annual Cooling Energy (MWh) | 11.31 | 1.04-2.24 | 138.6% | ModelLimitation |
| 930 | Peak Cooling Load (kW) | 10.98 | 1.10-1.50 | 131.6% | Unknown |
| 950 | Peak Cooling Load (kW) | 12.67 | 0.70-0.90 | 109.5% | Unknown |
| 610 | Annual Cooling Energy (MWh) | 10.48 | 3.92-6.14 | 108.4% | Unknown |
| 960 | Peak Cooling Load (kW) | 13.84 | 0.00-4.00 | 105.1% | Unknown |
| 950 | Annual Heating Energy (MWh) | 0.00 | N/A | 100.0% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 640 | Annual Cooling Energy (MWh) | 13.64 | 5.95-8.10 | 94.1% | Unknown |
| 600 | Peak Cooling Load (kW) | 10.15 | 4.80-6.20 | 91.6% | SolarGains |
| 650 | Annual Cooling Energy (MWh) | 10.71 | 4.82-7.06 | 80.3% | Unknown |
| 940 | Annual Heating Energy (MWh) | 1.67 | 0.79-1.41 | 73.8% | ModelLimitation |
| 600 | Peak Heating Load (kW) | 5.58 | 2.80-3.80 | 69.0% | Unknown |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 910 | 4 | 966.5% |
| 900 | 3 | 843.9% |
| 920 | 4 | 592.9% |
| 630 | 2 | 496.1% |
| 620 | 3 | 450.7% |
| 960 | 4 | 438.2% |
| 650 | 2 | 424.0% |
| 640 | 3 | 341.6% |
| 610 | 3 | 336.6% |
| 950 | 3 | 309.5% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
