# Quality Metrics Tracker

*Generated: 2026-03-25 17:16 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** -8.58%
- **Max Deviation:** 40.00%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| WARN | 1 | 1.6% |
| FAIL | 61 | 95.3% |
| PASS | 2 | 3.1% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | -8.6% | 40% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 900FF | Min Free-Float Temp (°C) | -0.85 | -6.40--1.60 | 78.7% | ThermalMass |
| 600FF | Min Free-Float Temp (°C) | -5.13 | -18.80--15.60 | 70.2% | FreeFloat |
| 950FF | Min Free-Float Temp (°C) | -8.31 | -20.20--17.80 | 56.3% | ThermalMass |
| 650FF | Min Free-Float Temp (°C) | -10.35 | -23.00--21.00 | 53.0% | FreeFloat |
| 600 | Peak Heating (kW) | 2.10 | 2.80-3.80 | 40.0% | Unknown |
| 650FF | Max Free-Float Temp (°C) | 44.95 | 63.20-73.50 | 34.2% | FreeFloat |
| 600FF | Max Free-Float Temp (°C) | 48.67 | 64.90-75.10 | 30.5% | FreeFloat |
| 600 | Annual Cooling (MWh) | 7.28 | 8.00-10.50 | 19.1% | Unknown |
| 600 | Annual Heating (MWh) | 6.99 | 5.50-7.50 | 16.5% | Unknown |
| 900FF | Max Free-Float Temp (°C) | 47.49 | 41.80-46.40 | 7.7% | ThermalMass |
| 600 | Peak Cooling (kW) | 5.73 | 4.80-6.20 | 4.1% | SolarGains |
| 950FF | Max Free-Float Temp (°C) | 35.88 | 35.50-38.50 | 3.0% | ThermalMass |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 600FF | 2 | 100.6% |
| 650FF | 2 | 87.2% |
| 900FF | 1 | 78.7% |
| 600 | 3 | 75.7% |
| 950FF | 1 | 56.3% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
