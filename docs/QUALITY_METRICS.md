# Quality Metrics Tracker

*Generated: 2026-04-20 04:55 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 28.38%
- **Max Deviation:** 100.00%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| WARN | 5 | 7.8% |
| PASS | 23 | 35.9% |
| FAIL | 36 | 56.2% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 28.4% | 100% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 950 | Annual Heating Energy (MWh) | 0.00 | N/A | 100.0% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 620 | Peak Cooling Load (kW) | 0.28 | 2.50-3.50 | 90.5% | SolarGains |
| 950 | Annual Cooling Energy (MWh) | 0.69 | 0.39-0.92 | 87.8% | ModelLimitation |
| 620 | Peak Heating Load (kW) | 0.49 | 2.80-3.80 | 85.3% | Unknown |
| 195 | Annual Heating Energy (MWh) | 8.73 | 3.50-6.00 | 83.7% | Unknown |
| 650 | Peak Cooling Load (kW) | 0.37 | 1.90-2.50 | 83.0% | SolarGains |
| 940 | Annual Heating Energy (MWh) | 1.19 | 0.79-1.41 | 81.4% | ModelLimitation |
| 940 | Peak Heating Load (kW) | 1.55 | 1.90-2.50 | 77.4% | Unknown |
| 960 | Annual Heating Energy (MWh) | 1.88 | 1.65-2.45 | 74.9% | Unknown |
| 950 | Peak Cooling Load (kW) | 1.68 | 0.70-0.90 | 72.2% | Unknown |
| 940 | Peak Cooling Load (kW) | 1.69 | 1.70-2.30 | 68.9% | Unknown |
| 930 | Annual Cooling Energy (MWh) | 1.49 | 1.04-2.24 | 68.6% | ModelLimitation |
| 930 | Peak Cooling Load (kW) | 1.67 | 1.10-1.50 | 64.8% | Unknown |
| 900FF | Minimum Free-Floating Temperature (°C) | -6.57 | -6.40--1.60 | 64.2% | ThermalMass |
| 920 | Annual Cooling Energy (MWh) | 1.76 | 1.84-3.31 | 53.5% | ModelLimitation |
| 920 | Peak Cooling Load (kW) | 1.91 | 1.40-1.90 | 49.4% | Unknown |
| 960 | Peak Cooling Load (kW) | 3.45 | 0.00-4.00 | 48.9% | Unknown |
| 610 | Peak Cooling Load (kW) | 3.74 | 2.20-2.90 | 46.5% | SolarGains |
| 650FF | Minimum Free-Floating Temperature (°C) | -12.28 | -23.00--21.00 | 44.2% | FreeFloat |
| 940 | Annual Cooling Energy (MWh) | 2.88 | 2.08-3.55 | 43.3% | ModelLimitation |
| 950FF | Minimum Free-Floating Temperature (°C) | -10.95 | -20.20--17.80 | 42.4% | ThermalMass |
| 630 | Peak Cooling Load (kW) | 2.95 | 1.80-2.40 | 40.4% | SolarGains |
| 900 | Peak Cooling Load (kW) | 1.69 | 1.60-2.10 | 39.8% | Unknown |
| 600 | Peak Cooling Load (kW) | 3.28 | 4.80-6.20 | 38.2% | SolarGains |
| 930 | Peak Heating Load (kW) | 3.13 | 2.30-3.00 | 34.0% | Unknown |
| 600FF | Minimum Free-Floating Temperature (°C) | -11.94 | -18.80--15.60 | 30.6% | FreeFloat |
| 950FF | Maximum Free-Floating Temperature (°C) | 48.05 | 35.50-38.50 | 29.9% | ThermalMass |
| 640 | Peak Cooling Load (kW) | 4.16 | 2.80-3.70 | 28.0% | SolarGains |
| 960 | Peak Heating Load (kW) | 6.33 | 2.00-8.00 | 25.5% | Unknown |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 950 | 4 | 360.0% |
| 940 | 4 | 271.0% |
| 620 | 2 | 175.8% |
| 960 | 4 | 169.1% |
| 930 | 3 | 167.5% |
| 920 | 3 | 121.0% |
| 195 | 1 | 83.7% |
| 650 | 1 | 83.0% |
| 950FF | 2 | 72.2% |
| 650FF | 2 | 66.5% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
