# Quality Metrics Tracker

*Generated: 2026-04-17 12:46 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 49.50%
- **Max Deviation:** 100.00%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| FAIL | 48 | 75.0% |
| WARN | 5 | 7.8% |
| PASS | 11 | 17.2% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 49.5% | 100% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 950 | Annual Heating Energy (MWh) | 0.00 | N/A | 100.0% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 195 | Annual Heating Energy (MWh) | 0.00 | 3.50-6.00 | 100.0% | Unknown |
| 195 | Annual Cooling Energy (MWh) | 0.00 | N/A | 100.0% | Unknown |
| 195 | Peak Heating Load (kW) | 0.00 | 1.40-2.20 | 100.0% | Unknown |
| 195 | Peak Cooling Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 600 | Peak Cooling Load (kW) | 0.19 | 4.80-6.20 | 96.4% | SolarGains |
| 600 | Peak Heating Load (kW) | 0.24 | 2.80-3.80 | 92.6% | Unknown |
| 600 | Annual Cooling Energy (MWh) | 0.66 | 8.00-10.50 | 92.3% | Unknown |
| 650 | Annual Cooling Energy (MWh) | 0.50 | 4.82-7.06 | 91.6% | Unknown |
| 630 | Peak Heating Load (kW) | 0.49 | 4.70-6.10 | 91.0% | Unknown |
| 640 | Annual Cooling Energy (MWh) | 0.65 | 5.95-8.10 | 90.7% | Unknown |
| 620 | Peak Cooling Load (kW) | 0.28 | 2.50-3.50 | 90.5% | SolarGains |
| 620 | Annual Cooling Energy (MWh) | 0.39 | 3.20-5.00 | 90.5% | Unknown |
| 640 | Peak Heating Load (kW) | 0.49 | 4.30-5.70 | 90.2% | Unknown |
| 610 | Peak Heating Load (kW) | 0.49 | 4.30-5.70 | 90.2% | Unknown |
| 610 | Annual Cooling Energy (MWh) | 0.54 | 3.92-6.14 | 89.2% | Unknown |
| 640 | Peak Cooling Load (kW) | 0.38 | 2.80-3.70 | 88.4% | SolarGains |
| 630 | Annual Cooling Energy (MWh) | 0.34 | 2.13-3.70 | 88.3% | Unknown |
| 950 | Annual Cooling Energy (MWh) | 0.69 | 0.39-0.92 | 87.8% | ModelLimitation |
| 630 | Peak Cooling Load (kW) | 0.27 | 1.80-2.40 | 87.2% | SolarGains |
| 610 | Peak Cooling Load (kW) | 0.34 | 2.20-2.90 | 86.7% | SolarGains |
| 620 | Peak Heating Load (kW) | 0.49 | 2.80-3.80 | 85.3% | Unknown |
| 650 | Peak Cooling Load (kW) | 0.37 | 1.90-2.50 | 83.0% | SolarGains |
| 940 | Annual Heating Energy (MWh) | 1.19 | 0.79-1.41 | 81.4% | ModelLimitation |
| 940 | Peak Heating Load (kW) | 1.55 | 1.90-2.50 | 77.4% | Unknown |
| 960 | Annual Heating Energy (MWh) | 1.88 | 1.65-2.45 | 74.9% | Unknown |
| 950 | Peak Cooling Load (kW) | 1.68 | 0.70-0.90 | 72.2% | Unknown |
| 940 | Peak Cooling Load (kW) | 1.69 | 1.70-2.30 | 68.9% | Unknown |
| 930 | Annual Cooling Energy (MWh) | 1.49 | 1.04-2.24 | 68.6% | ModelLimitation |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 195 | 4 | 400.0% |
| 950 | 4 | 360.0% |
| 600 | 3 | 281.3% |
| 940 | 4 | 271.0% |
| 640 | 3 | 269.3% |
| 630 | 3 | 266.5% |
| 620 | 3 | 266.3% |
| 610 | 3 | 266.1% |
| 650 | 2 | 174.6% |
| 960 | 4 | 169.1% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
