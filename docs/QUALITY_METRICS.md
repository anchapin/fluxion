# Quality Metrics Tracker

*Generated: 2026-04-18 05:25 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 37.48%
- **Max Deviation:** 100.00%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| PASS | 17 | 26.6% |
| FAIL | 42 | 65.6% |
| WARN | 5 | 7.8% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 37.5% | 100% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 950 | Annual Heating Energy (MWh) | 0.00 | N/A | 100.0% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 195 | Annual Cooling Energy (MWh) | 0.00 | N/A | 100.0% | Unknown |
| 195 | Peak Cooling Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 630 | Peak Heating Load (kW) | 0.49 | 4.70-6.10 | 91.0% | Unknown |
| 620 | Peak Cooling Load (kW) | 0.28 | 2.50-3.50 | 90.5% | SolarGains |
| 640 | Peak Heating Load (kW) | 0.49 | 4.30-5.70 | 90.2% | Unknown |
| 610 | Peak Heating Load (kW) | 0.49 | 4.30-5.70 | 90.2% | Unknown |
| 640 | Peak Cooling Load (kW) | 0.38 | 2.80-3.70 | 88.4% | SolarGains |
| 950 | Annual Cooling Energy (MWh) | 0.69 | 0.39-0.92 | 87.8% | ModelLimitation |
| 630 | Peak Cooling Load (kW) | 0.27 | 1.80-2.40 | 87.2% | SolarGains |
| 610 | Peak Cooling Load (kW) | 0.34 | 2.20-2.90 | 86.7% | SolarGains |
| 195 | Peak Heating Load (kW) | 1.95 | 1.40-2.20 | 86.0% | Unknown |
| 620 | Peak Heating Load (kW) | 0.49 | 2.80-3.80 | 85.3% | Unknown |
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
| 650FF | Minimum Free-Floating Temperature (°C) | -12.28 | -23.00--21.00 | 44.2% | FreeFloat |
| 940 | Annual Cooling Energy (MWh) | 2.88 | 2.08-3.55 | 43.3% | ModelLimitation |
| 950FF | Minimum Free-Floating Temperature (°C) | -10.95 | -20.20--17.80 | 42.4% | ThermalMass |
| 900 | Peak Cooling Load (kW) | 1.69 | 1.60-2.10 | 39.8% | Unknown |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 950 | 4 | 360.0% |
| 195 | 4 | 315.6% |
| 940 | 4 | 271.0% |
| 640 | 2 | 178.6% |
| 630 | 2 | 178.2% |
| 610 | 2 | 176.9% |
| 620 | 2 | 175.8% |
| 960 | 4 | 169.1% |
| 930 | 3 | 167.5% |
| 920 | 3 | 121.0% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
