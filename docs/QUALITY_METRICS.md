# Quality Metrics Tracker

*Generated: 2026-04-02 23:29 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 31.75%
- **Max Deviation:** 321.63%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| FAIL | 26 | 40.6% |
| WARN | 16 | 25.0% |
| PASS | 22 | 34.4% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 31.7% | 322% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 950 | Peak Cooling (kW) | 3.37 | 0.70-0.90 | 321.6% | Unknown |
| 650 | Peak Cooling (kW) | 5.60 | 1.90-2.50 | 154.5% | SolarGains |
| 900FF | Min Free-Float Temp (°C) | -9.59 | -6.40--1.60 | 139.8% | ThermalMass |
| 900 | Peak Heating (kW) | 4.44 | 1.80-2.40 | 111.3% | Unknown |
| 910 | Peak Cooling (kW) | 2.91 | 1.20-1.60 | 108.0% | Unknown |
| 910 | Peak Heating (kW) | 4.44 | 1.90-2.50 | 102.0% | Unknown |
| 940 | Peak Heating (kW) | 4.21 | 1.90-2.50 | 91.6% | Unknown |
| 610 | Peak Cooling (kW) | 4.72 | 2.20-2.90 | 85.1% | SolarGains |
| 900 | Peak Cooling (kW) | 3.42 | 1.60-2.10 | 84.6% | Unknown |
| 920 | Peak Heating (kW) | 4.34 | 2.10-2.80 | 77.3% | Unknown |
| 640 | Peak Cooling (kW) | 5.61 | 2.80-3.70 | 72.5% | SolarGains |
| 940 | Peak Cooling (kW) | 3.42 | 1.70-2.30 | 70.8% | Unknown |
| 930 | Peak Heating (kW) | 4.37 | 2.30-3.00 | 65.0% | Unknown |
| 960 | Peak Cooling (kW) | 3.28 | 0.00-4.00 | 64.1% | Unknown |
| 960 | Annual Cooling (MWh) | 3.59 | 1.00-3.50 | 59.7% | InterZoneTransfer |
| 195 | Annual Heating (MWh) | 7.34 | 3.50-6.00 | 54.4% | Unknown |
| 650FF | Min Free-Float Temp (°C) | -11.88 | -23.00--21.00 | 46.0% | FreeFloat |
| 600 | Peak Heating (kW) | 4.77 | 2.80-3.80 | 44.7% | Unknown |
| 620 | Peak Heating (kW) | 4.77 | 2.80-3.80 | 44.5% | Unknown |
| 195 | Peak Heating (kW) | 2.59 | 1.40-2.20 | 44.1% | Unknown |
| 640 | Annual Heating (MWh) | 4.57 | 2.75-3.80 | 39.6% | Unknown |
| 950FF | Min Free-Float Temp (°C) | -11.52 | -20.20--17.80 | 39.3% | ThermalMass |
| 910 | Annual Cooling (MWh) | 1.87 | 0.82-1.88 | 38.3% | ModelLimitation |
| 600FF | Min Free-Float Temp (°C) | -10.77 | -18.80--15.60 | 37.4% | FreeFloat |
| 630 | Annual Cooling (MWh) | 2.02 | 2.13-3.70 | 30.6% | Unknown |
| 960 | Annual Heating (MWh) | 7.02 | 5.00-15.00 | 29.8% | Unknown |
| 610 | Annual Heating (MWh) | 6.52 | 4.36-5.79 | 28.5% | Unknown |
| 920 | Peak Cooling (kW) | 2.08 | 1.40-1.90 | 26.2% | Unknown |
| 950FF | Max Free-Float Temp (°C) | 46.50 | 35.50-38.50 | 25.7% | ThermalMass |
| 650 | Annual Cooling (MWh) | 7.33 | 4.82-7.06 | 23.5% | Unknown |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 950 | 1 | 321.6% |
| 910 | 2 | 210.0% |
| 900 | 2 | 195.9% |
| 940 | 2 | 162.3% |
| 650 | 1 | 154.5% |
| 900FF | 1 | 139.8% |
| 610 | 2 | 113.6% |
| 640 | 2 | 112.1% |
| 920 | 2 | 103.5% |
| 195 | 2 | 98.6% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
