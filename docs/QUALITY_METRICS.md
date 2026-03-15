# Quality Metrics Tracker

*Generated: 2026-03-15 19:23 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 12.10%
- **Max Deviation:** 99.93%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| PASS | 1 | 1.6% |
| WARN | 2 | 3.1% |
| FAIL | 61 | 95.3% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 12.1% | 100% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 600 | Annual Cooling (MWh) | 0.01 | 8.00-10.50 | 99.9% | Unknown |
| 600 | Annual Heating (MWh) | 0.01 | 5.50-7.50 | 99.9% | Unknown |
| 600FF | Min Free-Float Temp (°C) | -5.01 | -18.80--15.60 | 70.9% | FreeFloat |
| 650FF | Min Free-Float Temp (°C) | -10.32 | -23.00--21.00 | 53.1% | FreeFloat |
| 950FF | Min Free-Float Temp (°C) | -9.51 | -20.20--17.80 | 50.0% | ThermalMass |
| 600 | Peak Heating (kW) | 2.10 | 2.80-3.80 | 40.0% | Unknown |
| 650FF | Max Free-Float Temp (°C) | 44.53 | 63.20-73.50 | 34.8% | FreeFloat |
| 600FF | Max Free-Float Temp (°C) | 47.89 | 64.90-75.10 | 31.6% | FreeFloat |
| 900FF | Min Free-Float Temp (°C) | -4.52 | -6.40--1.60 | 13.1% | ThermalMass |
| 900FF | Max Free-Float Temp (°C) | 38.37 | 41.80-46.40 | 13.0% | ThermalMass |
| 600 | Peak Cooling (kW) | 5.01 | 4.80-6.20 | 8.8% | SolarGains |
| 950FF | Max Free-Float Temp (°C) | 35.48 | 35.50-38.50 | 4.1% | ThermalMass |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 600 | 3 | 239.8% |
| 600FF | 2 | 102.5% |
| 650FF | 2 | 87.9% |
| 950FF | 1 | 50.0% |
| 900FF | 1 | 13.0% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
