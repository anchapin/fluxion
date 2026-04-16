# Quality Metrics Tracker

*Generated: 2026-04-16 03:41 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 37.67%
- **Max Deviation:** 100.00%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| WARN | 2 | 3.1% |
| PASS | 4 | 6.2% |
| FAIL | 58 | 90.6% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 37.7% | 100% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 950 | Annual Heating Energy (MWh) | 0.00 | N/A | 100.0% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 195 | Annual Cooling Energy (MWh) | 0.00 | N/A | 100.0% | Unknown |
| 195 | Peak Cooling Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 950 | Annual Cooling Energy (MWh) | 0.63 | 0.39-0.92 | 88.8% | ModelLimitation |
| 930 | Peak Cooling Load (kW) | 0.61 | 1.10-1.50 | 87.2% | Unknown |
| 940 | Annual Heating Energy (MWh) | 1.25 | 0.79-1.41 | 80.4% | ModelLimitation |
| 940 | Peak Heating Load (kW) | 1.61 | 1.90-2.50 | 76.5% | Unknown |
| 960 | Annual Heating Energy (MWh) | 1.88 | 1.65-2.45 | 75.0% | Unknown |
| 920 | Peak Cooling Load (kW) | 0.95 | 1.40-1.90 | 74.8% | Unknown |
| 950 | Peak Cooling Load (kW) | 1.66 | 0.70-0.90 | 72.6% | Unknown |
| 940 | Peak Cooling Load (kW) | 1.68 | 1.70-2.30 | 69.1% | Unknown |
| 930 | Peak Heating Load (kW) | 1.58 | 2.30-3.00 | 66.7% | Unknown |
| 930 | Annual Cooling Energy (MWh) | 1.68 | 1.04-2.24 | 64.6% | ModelLimitation |
| 900FF | Minimum Free-Floating Temperature (°C) | -6.57 | -6.40--1.60 | 64.1% | ThermalMass |
| 195 | Annual Heating Energy (MWh) | 5.00 | 3.50-6.00 | 59.7% | Unknown |
| 920 | Peak Heating Load (kW) | 1.55 | 2.10-2.80 | 59.1% | Unknown |
| 195 | Peak Heating Load (kW) | 6.86 | 1.40-2.20 | 50.6% | Unknown |
| 960 | Peak Cooling Load (kW) | 3.40 | 0.00-4.00 | 49.7% | Unknown |
| 600 | Peak Cooling Load (kW) | 2.82 | 4.80-6.20 | 46.9% | SolarGains |
| 650FF | Minimum Free-Floating Temperature (°C) | -11.91 | -23.00--21.00 | 45.8% | FreeFloat |
| 940 | Annual Cooling Energy (MWh) | 2.80 | 2.08-3.55 | 44.8% | ModelLimitation |
| 950FF | Minimum Free-Floating Temperature (°C) | -10.95 | -20.20--17.80 | 42.4% | ThermalMass |
| 900 | Peak Cooling Load (kW) | 1.68 | 1.60-2.10 | 39.9% | Unknown |
| 600FF | Minimum Free-Floating Temperature (°C) | -11.31 | -18.80--15.60 | 34.2% | FreeFloat |
| 950FF | Maximum Free-Floating Temperature (°C) | 48.01 | 35.50-38.50 | 29.8% | ThermalMass |
| 960 | Peak Heating Load (kW) | 6.31 | 2.00-8.00 | 25.7% | Unknown |
| 600FF | Maximum Free-Floating Temperature (°C) | 53.45 | 64.90-75.10 | 23.6% | FreeFloat |
| 650FF | Maximum Free-Floating Temperature (°C) | 53.45 | 63.20-73.50 | 21.8% | FreeFloat |
| 960 | Annual Cooling Energy (MWh) | 7.33 | 1.55-2.78 | 17.3% | InterZoneTransfer |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 950 | 4 | 361.4% |
| 195 | 4 | 310.3% |
| 940 | 4 | 270.8% |
| 930 | 4 | 228.5% |
| 960 | 4 | 167.7% |
| 920 | 4 | 156.5% |
| 950FF | 2 | 72.1% |
| 650FF | 2 | 67.6% |
| 900FF | 1 | 64.1% |
| 600FF | 2 | 57.9% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
