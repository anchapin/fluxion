# Quality Metrics Tracker

*Generated: 2026-04-02 11:26 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** -0.50%
- **Max Deviation:** 69.48%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| WARN | 1 | 1.6% |
| PASS | 3 | 4.7% |
| FAIL | 60 | 93.8% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | -0.5% | 69% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 600 | Annual Heating (MWh) | 10.17 | 5.50-7.50 | 69.5% | Unknown |
| 900FF | Min Free-Float Temp (°C) | -5.99 | -6.40--1.60 | 49.8% | ThermalMass |
| 600 | Peak Heating (kW) | 5.23 | 2.80-3.80 | 49.5% | Unknown |
| 950FF | Min Free-Float Temp (°C) | -10.05 | -20.20--17.80 | 47.1% | ThermalMass |
| 650FF | Min Free-Float Temp (°C) | -11.77 | -23.00--21.00 | 46.5% | FreeFloat |
| 600FF | Min Free-Float Temp (°C) | -9.92 | -18.80--15.60 | 42.3% | FreeFloat |
| 600FF | Max Free-Float Temp (°C) | 55.90 | 64.90-75.10 | 20.1% | FreeFloat |
| 650FF | Max Free-Float Temp (°C) | 55.79 | 63.20-73.50 | 18.4% | FreeFloat |
| 900FF | Max Free-Float Temp (°C) | 39.06 | 41.80-46.40 | 11.4% | ThermalMass |
| 600 | Annual Cooling (MWh) | 9.64 | 8.00-10.50 | 7.2% | Unknown |
| 950FF | Max Free-Float Temp (°C) | 36.06 | 35.50-38.50 | 2.5% | ThermalMass |
| 600 | Peak Cooling (kW) | 5.44 | 4.80-6.20 | 1.1% | SolarGains |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 600 | 2 | 118.9% |
| 650FF | 2 | 64.9% |
| 600FF | 2 | 62.5% |
| 950FF | 1 | 47.1% |
| 900FF | 1 | 11.4% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
