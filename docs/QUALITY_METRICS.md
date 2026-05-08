# Quality Metrics Tracker

*Generated: 2026-05-08 15:26 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 100.86%
- **Max Deviation:** 528.91%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| PASS | 10 | 15.6% |
| FAIL | 50 | 78.1% |
| WARN | 4 | 6.2% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 100.9% | 529% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 910 | Peak Cooling Load (kW) | 8.80 | 1.20-1.60 | 528.9% | Unknown |
| 900 | Peak Cooling Load (kW) | 13.65 | 1.60-2.10 | 387.4% | Unknown |
| 920 | Annual Cooling Energy (MWh) | 18.02 | 1.84-3.31 | 376.7% | ModelLimitation |
| 650 | Peak Cooling Load (kW) | 9.76 | 1.90-2.50 | 343.7% | SolarGains |
| 910 | Annual Cooling Energy (MWh) | 5.26 | 0.82-1.88 | 289.6% | ModelLimitation |
| 630 | Peak Cooling Load (kW) | 7.90 | 1.80-2.40 | 276.2% | SolarGains |
| 900 | Annual Cooling Energy (MWh) | 10.58 | 2.13-3.67 | 264.8% | ModelLimitation |
| 960 | Annual Cooling Energy (MWh) | 21.83 | 1.55-2.78 | 249.2% | InterZoneTransfer |
| 630 | Annual Cooling Energy (MWh) | 9.32 | 2.13-3.70 | 219.8% | Unknown |
| 920 | Peak Cooling Load (kW) | 12.06 | 1.40-1.90 | 219.0% | Unknown |
| 640 | Peak Cooling Load (kW) | 10.15 | 2.80-3.70 | 212.4% | SolarGains |
| 620 | Annual Cooling Energy (MWh) | 12.05 | 3.20-5.00 | 194.0% | Unknown |
| 620 | Peak Cooling Load (kW) | 8.68 | 2.50-3.50 | 189.5% | SolarGains |
| 900 | Peak Heating Load (kW) | 4.57 | 1.80-2.40 | 185.6% | Unknown |
| 610 | Peak Cooling Load (kW) | 7.23 | 2.20-2.90 | 183.4% | SolarGains |
| 930 | Annual Cooling Energy (MWh) | 13.32 | 1.04-2.24 | 181.0% | ModelLimitation |
| 900FF | Minimum Free-Floating Temperature (°C) | -10.78 | -6.40--1.60 | 169.6% | ThermalMass |
| 940 | Peak Cooling Load (kW) | 13.65 | 1.70-2.30 | 150.4% | Unknown |
| 930 | Peak Cooling Load (kW) | 11.33 | 1.10-1.50 | 139.0% | Unknown |
| 950 | Peak Cooling Load (kW) | 13.04 | 0.70-0.90 | 115.5% | Unknown |
| 910 | Peak Heating Load (kW) | 4.64 | 1.90-2.50 | 110.7% | Unknown |
| 610 | Annual Cooling Energy (MWh) | 10.48 | 3.92-6.14 | 108.4% | Unknown |
| 960 | Peak Cooling Load (kW) | 13.83 | 0.00-4.00 | 104.8% | Unknown |
| 950 | Annual Heating Energy (MWh) | 0.00 | N/A | 100.0% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 640 | Annual Cooling Energy (MWh) | 13.64 | 5.95-8.10 | 94.1% | Unknown |
| 600 | Peak Cooling Load (kW) | 10.15 | 4.80-6.20 | 91.6% | SolarGains |
| 940 | Annual Cooling Energy (MWh) | 9.44 | 2.08-3.55 | 86.0% | ModelLimitation |
| 940 | Annual Heating Energy (MWh) | 1.22 | 0.79-1.41 | 80.9% | ModelLimitation |
| 650 | Annual Cooling Energy (MWh) | 10.71 | 4.82-7.06 | 80.3% | Unknown |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 910 | 3 | 929.2% |
| 900 | 4 | 855.7% |
| 920 | 3 | 605.8% |
| 630 | 2 | 496.1% |
| 620 | 3 | 450.7% |
| 960 | 4 | 435.0% |
| 650 | 2 | 424.0% |
| 930 | 4 | 351.3% |
| 940 | 4 | 350.6% |
| 640 | 3 | 341.6% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
