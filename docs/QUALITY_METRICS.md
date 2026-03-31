# Quality Metrics Tracker

*Generated: 2026-03-31 19:40 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** -5.60%
- **Max Deviation:** 30.69%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| FAIL | 61 | 95.3% |
| PASS | 2 | 3.1% |
| WARN | 1 | 1.6% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | -5.6% | 31% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 900FF | Min Free-Float Temp (°C) | -6.06 | -6.40--1.60 | 51.5% | ThermalMass |
| 950FF | Min Free-Float Temp (°C) | -10.08 | -20.20--17.80 | 46.9% | ThermalMass |
| 650FF | Min Free-Float Temp (°C) | -11.86 | -23.00--21.00 | 46.1% | FreeFloat |
| 600FF | Min Free-Float Temp (°C) | -10.68 | -18.80--15.60 | 37.9% | FreeFloat |
| 600 | Peak Cooling (kW) | 3.81 | 4.80-6.20 | 30.7% | SolarGains |
| 600 | Annual Cooling (MWh) | 7.06 | 8.00-10.50 | 21.5% | Unknown |
| 600 | Annual Heating (MWh) | 7.04 | 5.50-7.50 | 17.3% | Unknown |
| 600FF | Max Free-Float Temp (°C) | 58.22 | 64.90-75.10 | 16.8% | FreeFloat |
| 650FF | Max Free-Float Temp (°C) | 58.20 | 63.20-73.50 | 14.9% | FreeFloat |
| 900FF | Max Free-Float Temp (°C) | 39.20 | 41.80-46.40 | 11.1% | ThermalMass |
| 950FF | Max Free-Float Temp (°C) | 36.27 | 35.50-38.50 | 2.0% | ThermalMass |
| 600 | Peak Heating (kW) | 3.53 | 2.80-3.80 | 0.9% | Unknown |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 600 | 3 | 69.5% |
| 650FF | 2 | 60.9% |
| 600FF | 2 | 54.7% |
| 950FF | 1 | 46.9% |
| 900FF | 1 | 11.1% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
