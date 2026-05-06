# Quality Metrics Tracker

*Generated: 2026-05-06 17:48 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 64.74%
- **Max Deviation:** 447.55%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| FAIL | 51 | 79.7% |
| WARN | 4 | 6.2% |
| PASS | 9 | 14.1% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 64.7% | 448% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 910 | Peak Cooling Load (kW) | 7.67 | 1.20-1.60 | 447.5% | Unknown |
| 900FF | Minimum Free-Floating Temperature (°C) | -21.18 | -6.40--1.60 | 429.4% | ThermalMass |
| 920 | Annual Cooling Energy (MWh) | 14.38 | 1.84-3.31 | 280.5% | ModelLimitation |
| 900 | Peak Cooling Load (kW) | 9.05 | 1.60-2.10 | 223.4% | Unknown |
| 910 | Annual Cooling Energy (MWh) | 4.31 | 0.82-1.88 | 219.1% | ModelLimitation |
| 650 | Peak Cooling Load (kW) | 6.36 | 1.90-2.50 | 189.3% | SolarGains |
| 900 | Annual Cooling Energy (MWh) | 7.99 | 2.13-3.67 | 175.6% | ModelLimitation |
| 930 | Annual Cooling Energy (MWh) | 11.45 | 1.04-2.24 | 141.6% | ModelLimitation |
| 630 | Peak Cooling Load (kW) | 5.00 | 1.80-2.40 | 138.0% | SolarGains |
| 900 | Peak Heating Load (kW) | 3.71 | 1.80-2.40 | 132.1% | Unknown |
| 640 | Annual Heating Energy (MWh) | 7.41 | 2.75-3.80 | 126.2% | Unknown |
| 610 | Annual Heating Energy (MWh) | 10.98 | 4.36-5.79 | 116.3% | Unknown |
| 920 | Peak Cooling Load (kW) | 8.09 | 1.40-1.90 | 114.0% | Unknown |
| 195 | Annual Heating Energy (MWh) | 10.06 | 3.50-6.00 | 111.9% | Unknown |
| 950FF | Maximum Free-Floating Temperature (°C) | 78.14 | 35.50-38.50 | 111.2% | ThermalMass |
| 640 | Peak Cooling Load (kW) | 6.59 | 2.80-3.70 | 102.7% | SolarGains |
| 600 | Peak Heating Load (kW) | 6.62 | 2.80-3.80 | 100.6% | Unknown |
| 950 | Annual Heating Energy (MWh) | 0.00 | N/A | 100.0% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 620 | Peak Heating Load (kW) | 6.55 | 2.80-3.80 | 98.5% | Unknown |
| 610 | Peak Cooling Load (kW) | 5.00 | 2.20-2.90 | 96.1% | SolarGains |
| 940 | Annual Heating Energy (MWh) | 0.78 | 0.79-1.41 | 87.8% | ModelLimitation |
| 620 | Peak Cooling Load (kW) | 5.62 | 2.50-3.50 | 87.3% | SolarGains |
| 600 | Annual Heating Energy (MWh) | 10.53 | 5.50-7.50 | 83.1% | Unknown |
| 900FF | Maximum Free-Floating Temperature (°C) | 79.54 | 41.80-46.40 | 80.4% | ThermalMass |
| 620 | Annual Heating Energy (MWh) | 9.55 | 4.50-6.50 | 73.7% | Unknown |
| 910 | Peak Heating Load (kW) | 3.76 | 1.90-2.50 | 71.0% | Unknown |
| 630 | Annual Heating Energy (MWh) | 9.81 | 5.05-6.47 | 70.4% | Unknown |
| 940 | Peak Cooling Load (kW) | 9.05 | 1.70-2.30 | 66.1% | Unknown |
| 600FF | Minimum Free-Floating Temperature (°C) | -28.32 | -18.80--15.60 | 64.6% | FreeFloat |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 910 | 3 | 737.6% |
| 900 | 4 | 577.5% |
| 900FF | 2 | 509.8% |
| 920 | 3 | 425.3% |
| 620 | 4 | 290.9% |
| 950 | 4 | 277.3% |
| 930 | 4 | 272.6% |
| 640 | 3 | 260.0% |
| 610 | 3 | 245.1% |
| 940 | 4 | 237.2% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
