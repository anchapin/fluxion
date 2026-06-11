# Quality Metrics Tracker

*Generated: 2026-06-11 17:31 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 44.37%
- **Max Deviation:** 100.00%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| FAIL | 54 | 84.4% |
| PASS | 8 | 12.5% |
| WARN | 2 | 3.1% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 44.4% | 100% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 900FF | Minimum Free-Floating Temperature (°C) | -26.56 | -6.40--1.60 | 564.0% | ThermalMass |
| 950 | Annual Heating Energy (MWh) | 0.00 | N/A | 100.0% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 960 | Annual Cooling Energy (MWh) | 0.00 | 1.55-2.78 | 100.0% | InterZoneTransfer |
| 960 | Peak Cooling Load (kW) | 0.00 | 0.00-4.00 | 100.0% | Unknown |
| 900 | Annual Heating Energy (MWh) | 3.19 | 1.17-2.04 | 98.9% | ModelLimitation |
| 950 | Annual Cooling Energy (MWh) | 0.08 | 0.39-0.92 | 98.6% | ModelLimitation |
| 940 | Annual Cooling Energy (MWh) | 0.11 | 2.08-3.55 | 97.8% | ModelLimitation |
| 930 | Annual Cooling Energy (MWh) | 0.11 | 1.04-2.24 | 97.6% | ModelLimitation |
| 920 | Annual Cooling Energy (MWh) | 0.11 | 1.84-3.31 | 97.0% | ModelLimitation |
| 900 | Annual Cooling Energy (MWh) | 0.11 | 2.13-3.67 | 96.1% | ModelLimitation |
| 950 | Peak Cooling Load (kW) | 0.48 | 0.70-0.90 | 92.1% | Unknown |
| 940 | Peak Cooling Load (kW) | 0.43 | 1.70-2.30 | 92.1% | Unknown |
| 910 | Annual Cooling Energy (MWh) | 0.11 | 0.82-1.88 | 91.6% | ModelLimitation |
| 930 | Peak Cooling Load (kW) | 0.43 | 1.10-1.50 | 90.9% | Unknown |
| 630 | Annual Cooling Energy (MWh) | 0.27 | 2.13-3.70 | 90.7% | Unknown |
| 920 | Peak Cooling Load (kW) | 0.43 | 1.40-1.90 | 88.6% | Unknown |
| 610 | Annual Cooling Energy (MWh) | 0.66 | 3.92-6.14 | 86.9% | Unknown |
| 600 | Annual Cooling Energy (MWh) | 1.16 | 8.00-10.50 | 86.4% | Unknown |
| 650 | Annual Cooling Energy (MWh) | 0.91 | 4.82-7.06 | 84.7% | Unknown |
| 900 | Peak Cooling Load (kW) | 0.43 | 1.60-2.10 | 84.6% | Unknown |
| 960 | Peak Heating Load (kW) | 1.35 | 2.00-8.00 | 84.1% | Unknown |
| 640 | Annual Cooling Energy (MWh) | 1.16 | 5.95-8.10 | 83.5% | Unknown |
| 620 | Annual Cooling Energy (MWh) | 1.01 | 3.20-5.00 | 75.3% | Unknown |
| 940 | Peak Heating Load (kW) | 1.93 | 1.90-2.50 | 71.8% | Unknown |
| 910 | Peak Cooling Load (kW) | 0.43 | 1.20-1.60 | 69.3% | Unknown |
| 910 | Annual Heating Energy (MWh) | 3.19 | 1.51-2.28 | 68.5% | ModelLimitation |
| 930 | Peak Heating Load (kW) | 1.70 | 2.30-3.00 | 64.1% | Unknown |
| 940 | Annual Heating Energy (MWh) | 2.56 | 0.79-1.41 | 59.9% | ModelLimitation |
| 195 | Annual Heating Energy (MWh) | 7.55 | 3.50-6.00 | 59.0% | Unknown |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 900FF | 2 | 583.6% |
| 950 | 4 | 390.7% |
| 940 | 4 | 321.6% |
| 930 | 4 | 285.3% |
| 960 | 3 | 284.1% |
| 900 | 3 | 279.6% |
| 920 | 4 | 256.2% |
| 910 | 4 | 252.1% |
| 630 | 4 | 220.0% |
| 610 | 4 | 214.4% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
