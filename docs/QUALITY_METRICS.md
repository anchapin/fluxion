# Quality Metrics Tracker

*Generated: 2026-04-14 00:49 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 38.88%
- **Max Deviation:** 346.87%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| FAIL | 60 | 93.8% |
| PASS | 3 | 4.7% |
| WARN | 1 | 1.6% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 38.9% | 347% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 900 | Annual Heating Energy (MWh) | 7.17 | 1.17-2.04 | 346.9% | ModelLimitation |
| 930 | Peak Cooling Load (kW) | 0.61 | 1.10-1.50 | 87.2% | Unknown |
| 930 | Annual Cooling Energy (MWh) | 1.00 | 1.04-2.24 | 78.8% | ModelLimitation |
| 920 | Annual Heating Energy (MWh) | 6.72 | 3.26-4.30 | 77.7% | ModelLimitation |
| 920 | Peak Cooling Load (kW) | 0.95 | 1.40-1.90 | 74.8% | Unknown |
| 900 | Annual Cooling Energy (MWh) | 5.06 | 2.13-3.67 | 74.6% | ModelLimitation |
| 930 | Peak Heating Load (kW) | 1.58 | 2.30-3.00 | 66.7% | Unknown |
| 900FF | Minimum Free-Floating Temperature (°C) | -6.57 | -6.40--1.60 | 64.1% | ThermalMass |
| 930 | Annual Heating Energy (MWh) | 7.65 | 4.14-5.34 | 61.4% | ModelLimitation |
| 920 | Peak Heating Load (kW) | 1.55 | 2.10-2.80 | 59.1% | Unknown |
| 920 | Annual Cooling Energy (MWh) | 1.93 | 1.84-3.31 | 48.9% | ModelLimitation |
| 650FF | Minimum Free-Floating Temperature (°C) | -11.91 | -23.00--21.00 | 45.8% | FreeFloat |
| 950FF | Minimum Free-Floating Temperature (°C) | -10.95 | -20.20--17.80 | 42.4% | ThermalMass |
| 900 | Peak Cooling Load (kW) | 1.68 | 1.60-2.10 | 39.9% | Unknown |
| 600FF | Minimum Free-Floating Temperature (°C) | -11.31 | -18.80--15.60 | 34.2% | FreeFloat |
| 950FF | Maximum Free-Floating Temperature (°C) | 48.01 | 35.50-38.50 | 29.8% | ThermalMass |
| 600FF | Maximum Free-Floating Temperature (°C) | 53.45 | 64.90-75.10 | 23.6% | FreeFloat |
| 650FF | Maximum Free-Floating Temperature (°C) | 53.45 | 63.20-73.50 | 21.8% | FreeFloat |
| 600 | Annual Heating Energy (MWh) | 6.49 | 5.50-7.50 | 8.2% | Unknown |
| 900FF | Maximum Free-Floating Temperature (°C) | 47.09 | 41.80-46.40 | 6.8% | ThermalMass |
| 600 | Peak Heating Load (kW) | 3.31 | 2.80-3.80 | 5.5% | Unknown |
| 900 | Peak Heating Load (kW) | 1.64 | 1.80-2.40 | 2.8% | Unknown |
| 600 | Annual Cooling Energy (MWh) | 9.25 | 8.00-10.50 | 2.8% | Unknown |
| 600 | Peak Cooling Load (kW) | 5.63 | 4.80-6.20 | 2.4% | SolarGains |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 900 | 3 | 461.4% |
| 930 | 4 | 294.1% |
| 920 | 4 | 260.6% |
| 950FF | 2 | 72.1% |
| 650FF | 2 | 67.6% |
| 900FF | 1 | 64.1% |
| 600FF | 2 | 57.9% |
| 600 | 2 | 13.7% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
