# Quality Metrics Tracker

*Generated: 2026-05-14 20:12 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 95.55%
- **Max Deviation:** 489.85%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| PASS | 5 | 7.8% |
| FAIL | 59 | 92.2% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 95.6% | 490% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 195 | Annual Heating Energy (MWh) | 28.02 | 3.50-6.00 | 489.8% | Unknown |
| 900FF | Minimum Free-Floating Temperature (°C) | -19.95 | -6.40--1.60 | 398.8% | ThermalMass |
| 640 | Annual Heating Energy (MWh) | 15.93 | 2.75-3.80 | 386.3% | Unknown |
| 610 | Annual Heating Energy (MWh) | 20.11 | 4.36-5.79 | 296.3% | Unknown |
| 600 | Annual Heating Energy (MWh) | 21.04 | 5.50-7.50 | 266.0% | Unknown |
| 620 | Annual Heating Energy (MWh) | 18.89 | 4.50-6.50 | 243.5% | Unknown |
| 630 | Annual Heating Energy (MWh) | 19.52 | 5.05-6.47 | 239.0% | Unknown |
| 600 | Peak Heating Load (kW) | 10.38 | 2.80-3.80 | 214.6% | Unknown |
| 620 | Peak Heating Load (kW) | 9.83 | 2.80-3.80 | 197.8% | Unknown |
| 650 | Peak Cooling Load (kW) | 5.02 | 1.90-2.50 | 128.3% | SolarGains |
| 610 | Peak Heating Load (kW) | 10.41 | 4.30-5.70 | 108.1% | Unknown |
| 640 | Peak Heating Load (kW) | 10.30 | 4.30-5.70 | 106.0% | Unknown |
| 900 | Annual Heating Energy (MWh) | 0.00 | 1.17-2.04 | 100.0% | ModelLimitation |
| 900 | Annual Cooling Energy (MWh) | 0.00 | 2.13-3.67 | 100.0% | ModelLimitation |
| 900 | Peak Heating Load (kW) | 0.00 | 1.80-2.40 | 100.0% | Unknown |
| 900 | Peak Cooling Load (kW) | 0.00 | 1.60-2.10 | 100.0% | Unknown |
| 910 | Annual Heating Energy (MWh) | 0.00 | 1.51-2.28 | 100.0% | ModelLimitation |
| 910 | Annual Cooling Energy (MWh) | 0.00 | 0.82-1.88 | 100.0% | ModelLimitation |
| 910 | Peak Heating Load (kW) | 0.00 | 1.90-2.50 | 100.0% | Unknown |
| 910 | Peak Cooling Load (kW) | 0.00 | 1.20-1.60 | 100.0% | Unknown |
| 920 | Annual Heating Energy (MWh) | 0.00 | 3.26-4.30 | 100.0% | ModelLimitation |
| 920 | Annual Cooling Energy (MWh) | 0.00 | 1.84-3.31 | 100.0% | ModelLimitation |
| 920 | Peak Heating Load (kW) | 0.00 | 2.10-2.80 | 100.0% | Unknown |
| 920 | Peak Cooling Load (kW) | 0.00 | 1.40-1.90 | 100.0% | Unknown |
| 930 | Annual Heating Energy (MWh) | 0.00 | 4.14-5.34 | 100.0% | ModelLimitation |
| 930 | Annual Cooling Energy (MWh) | 0.00 | 1.04-2.24 | 100.0% | ModelLimitation |
| 930 | Peak Heating Load (kW) | 0.00 | 2.30-3.00 | 100.0% | Unknown |
| 930 | Peak Cooling Load (kW) | 0.00 | 1.10-1.50 | 100.0% | Unknown |
| 940 | Annual Heating Energy (MWh) | 0.00 | 0.79-1.41 | 100.0% | ModelLimitation |
| 940 | Annual Cooling Energy (MWh) | 0.00 | 2.08-3.55 | 100.0% | ModelLimitation |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 640 | 4 | 616.7% |
| 600 | 3 | 551.9% |
| 610 | 4 | 545.8% |
| 620 | 4 | 545.5% |
| 195 | 2 | 519.5% |
| 900FF | 2 | 465.3% |
| 630 | 4 | 449.4% |
| 920 | 4 | 400.0% |
| 900 | 4 | 400.0% |
| 950 | 4 | 400.0% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
