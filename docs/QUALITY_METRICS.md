# Quality Metrics Tracker

*Generated: 2026-04-27 22:58 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 32.97%
- **Max Deviation:** 129.38%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| FAIL | 43 | 67.2% |
| PASS | 17 | 26.6% |
| WARN | 4 | 6.2% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 33.0% | 129% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 900FF | Minimum Free-Floating Temperature (°C) | -9.69 | -6.40--1.60 | 142.2% | ThermalMass |
| 630 | Peak Cooling Load (kW) | 4.82 | 1.80-2.40 | 129.4% | SolarGains |
| 950 | Annual Heating Energy (MWh) | 0.00 | N/A | 100.0% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 620 | Annual Cooling Energy (MWh) | 7.86 | 3.20-5.00 | 91.7% | Unknown |
| 940 | Annual Heating Energy (MWh) | 0.93 | 0.79-1.41 | 85.5% | ModelLimitation |
| 620 | Peak Heating Load (kW) | 0.48 | 2.80-3.80 | 85.3% | Unknown |
| 960 | Annual Heating Energy (MWh) | 1.11 | 1.65-2.45 | 85.2% | Unknown |
| 950 | Annual Cooling Energy (MWh) | 0.84 | 0.39-0.92 | 85.2% | ModelLimitation |
| 620 | Peak Cooling Load (kW) | 0.48 | 2.50-3.50 | 84.0% | SolarGains |
| 195 | Annual Heating Energy (MWh) | 8.71 | 3.50-6.00 | 83.3% | Unknown |
| 940 | Peak Heating Load (kW) | 1.25 | 1.90-2.50 | 81.8% | Unknown |
| 650 | Peak Cooling Load (kW) | 0.49 | 1.90-2.50 | 77.8% | SolarGains |
| 630 | Annual Cooling Energy (MWh) | 5.17 | 2.13-3.70 | 77.2% | Unknown |
| 640 | Peak Cooling Load (kW) | 5.45 | 2.80-3.70 | 67.7% | SolarGains |
| 950 | Peak Cooling Load (kW) | 2.21 | 0.70-0.90 | 63.5% | Unknown |
| 610 | Peak Cooling Load (kW) | 4.16 | 2.20-2.90 | 63.2% | SolarGains |
| 940 | Peak Cooling Load (kW) | 2.18 | 1.70-2.30 | 60.0% | Unknown |
| 960 | Peak Heating Load (kW) | 3.81 | 2.00-8.00 | 55.1% | Unknown |
| 960 | Peak Cooling Load (kW) | 3.06 | 0.00-4.00 | 54.7% | Unknown |
| 950FF | Maximum Free-Floating Temperature (°C) | 54.20 | 35.50-38.50 | 46.5% | ThermalMass |
| 930 | Peak Heating Load (kW) | 2.62 | 2.30-3.00 | 44.8% | Unknown |
| 600 | Peak Cooling Load (kW) | 2.94 | 4.80-6.20 | 44.5% | SolarGains |
| 650FF | Minimum Free-Floating Temperature (°C) | -12.28 | -23.00--21.00 | 44.2% | FreeFloat |
| 910 | Peak Heating Load (kW) | 1.32 | 1.90-2.50 | 40.0% | Unknown |
| 950FF | Minimum Free-Floating Temperature (°C) | -11.57 | -20.20--17.80 | 39.1% | ThermalMass |
| 920 | Annual Cooling Energy (MWh) | 5.08 | 1.84-3.31 | 34.5% | ModelLimitation |
| 940 | Annual Cooling Energy (MWh) | 3.39 | 2.08-3.55 | 33.1% | ModelLimitation |
| 920 | Peak Heating Load (kW) | 2.61 | 2.10-2.80 | 30.9% | Unknown |
| 600FF | Minimum Free-Floating Temperature (°C) | -11.94 | -18.80--15.60 | 30.6% | FreeFloat |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 950 | 4 | 348.7% |
| 620 | 3 | 261.0% |
| 940 | 4 | 260.3% |
| 630 | 3 | 224.7% |
| 960 | 4 | 205.4% |
| 900FF | 2 | 164.1% |
| 600 | 3 | 101.6% |
| 930 | 4 | 101.1% |
| 900 | 4 | 88.8% |
| 920 | 3 | 87.4% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
