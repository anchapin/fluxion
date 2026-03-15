# Quality Metrics Tracker

*Generated: 2026-03-10 21:16 UTC

## Current Status

- **Pass Rate:** 5.6% (1 / 18 cases)
- **MAE:** 61.48%
- **Max Deviation:** 527.03%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| PASS | 18 | 28.1% |
| FAIL | 36 | 56.2% |
| WARN | 10 | 15.6% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 5.6% | 61.5% | 527% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 950 | Peak Cooling (kW) | 5.02 | 0.70-0.90 | 527.0% | Unknown |
| 940 | Annual Heating (MWh) | 5.34 | 0.79-1.41 | 385.5% | ModelLimitation |
| 950 | Annual Cooling (MWh) | 2.58 | 0.39-0.92 | 294.4% | ModelLimitation |
| 900 | Annual Heating (MWh) | 5.35 | 1.17-2.04 | 233.2% | ModelLimitation |
| 910 | Annual Heating (MWh) | 5.77 | 1.51-2.28 | 204.6% | ModelLimitation |
| 650 | Peak Cooling (kW) | 6.45 | 1.90-2.50 | 193.2% | SolarGains |
| 910 | Annual Cooling (MWh) | 3.26 | 0.82-1.88 | 141.8% | ModelLimitation |
| 910 | Peak Cooling (kW) | 2.91 | 1.20-1.60 | 107.7% | Unknown |
| 640 | Annual Heating (MWh) | 6.78 | 2.75-3.80 | 107.1% | Unknown |
| 960 | Annual Cooling (MWh) | 4.53 | 1.00-3.50 | 101.5% | InterZoneTransfer |
| 900 | Peak Cooling (kW) | 3.56 | 1.60-2.10 | 92.3% | Unknown |
| 960 | Peak Cooling (kW) | 3.79 | 0.00-4.00 | 89.6% | Unknown |
| 940 | Peak Cooling (kW) | 3.55 | 1.70-2.30 | 77.4% | Unknown |
| 600FF | Min Free-Float Temp (°C) | -5.01 | -18.80--15.60 | 70.9% | FreeFloat |
| 940 | Annual Cooling (MWh) | 4.75 | 2.08-3.55 | 68.8% | ModelLimitation |
| 900 | Annual Cooling (MWh) | 4.75 | 2.13-3.67 | 63.9% | ModelLimitation |
| 630 | Peak Heating (kW) | 2.10 | 4.70-6.10 | 61.1% | Unknown |
| 610 | Peak Cooling (kW) | 4.10 | 2.20-2.90 | 60.7% | SolarGains |
| 630 | Annual Cooling (MWh) | 1.18 | 2.13-3.70 | 59.6% | Unknown |
| 610 | Peak Heating (kW) | 2.10 | 4.30-5.70 | 58.0% | Unknown |
| 640 | Peak Heating (kW) | 2.10 | 4.30-5.70 | 58.0% | Unknown |
| 960 | Peak Heating (kW) | 2.10 | 2.00-8.00 | 58.0% | Unknown |
| 930 | Annual Cooling (MWh) | 0.74 | 1.04-2.24 | 54.8% | ModelLimitation |
| 640 | Peak Cooling (kW) | 5.01 | 2.80-3.70 | 54.3% | SolarGains |
| 650FF | Min Free-Float Temp (°C) | -10.32 | -23.00--21.00 | 53.1% | FreeFloat |
| 950FF | Min Free-Float Temp (°C) | -9.37 | -20.20--17.80 | 50.7% | ThermalMass |
| 620 | Annual Cooling (MWh) | 2.33 | 3.20-5.00 | 43.2% | Unknown |
| 960 | Annual Heating (MWh) | 5.78 | 5.00-15.00 | 42.2% | Unknown |
| 610 | Annual Heating (MWh) | 7.12 | 4.36-5.79 | 40.3% | Unknown |
| 920 | Annual Cooling (MWh) | 1.60 | 1.84-3.31 | 37.9% | ModelLimitation |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 950 | 2 | 821.5% |
| 940 | 3 | 531.8% |
| 910 | 3 | 454.1% |
| 900 | 3 | 389.3% |
| 640 | 3 | 219.4% |
| 650 | 1 | 193.2% |
| 610 | 3 | 159.0% |
| 630 | 3 | 151.8% |
| 600FF | 2 | 102.5% |
| 960 | 1 | 101.5% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
