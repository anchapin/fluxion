# Quality Metrics Tracker

*Generated: 2026-05-08 00:39 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** 210.64%
- **Max Deviation:** 1520.80%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| WARN | 3 | 4.7% |
| FAIL | 53 | 82.8% |
| PASS | 8 | 12.5% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | 210.6% | 1521% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 920 | Annual Cooling Energy (MWh) | 61.27 | 1.84-3.31 | 1520.8% | ModelLimitation |
| 910 | Annual Cooling Energy (MWh) | 18.01 | 0.82-1.88 | 1234.4% | ModelLimitation |
| 910 | Peak Cooling Load (kW) | 15.60 | 1.20-1.60 | 1014.2% | Unknown |
| 930 | Annual Cooling Energy (MWh) | 52.58 | 1.04-2.24 | 1009.4% | ModelLimitation |
| 900 | Annual Cooling Energy (MWh) | 31.38 | 2.13-3.67 | 982.1% | ModelLimitation |
| 960 | Annual Cooling Energy (MWh) | 63.57 | 1.55-2.78 | 917.2% | InterZoneTransfer |
| 900 | Peak Cooling Load (kW) | 21.68 | 1.60-2.10 | 674.2% | Unknown |
| 940 | Annual Cooling Energy (MWh) | 28.23 | 2.08-3.55 | 456.3% | ModelLimitation |
| 920 | Peak Cooling Load (kW) | 19.31 | 1.40-1.90 | 410.9% | Unknown |
| 650 | Peak Cooling Load (kW) | 9.76 | 1.90-2.50 | 343.7% | SolarGains |
| 940 | Peak Cooling Load (kW) | 21.68 | 1.70-2.30 | 297.8% | Unknown |
| 630 | Peak Cooling Load (kW) | 7.90 | 1.80-2.40 | 276.2% | SolarGains |
| 930 | Peak Cooling Load (kW) | 17.77 | 1.10-1.50 | 275.0% | Unknown |
| 950 | Peak Cooling Load (kW) | 21.47 | 0.70-0.90 | 254.9% | Unknown |
| 630 | Annual Cooling Energy (MWh) | 9.32 | 2.13-3.70 | 219.8% | Unknown |
| 640 | Peak Cooling Load (kW) | 10.15 | 2.80-3.70 | 212.4% | SolarGains |
| 920 | Peak Cooling Load (kW) | 11.75 | 1.40-1.90 | 210.8% | Unknown |
| 900FF | Minimum Free-Floating Temperature (°C) | -12.05 | -6.40--1.60 | 201.3% | ThermalMass |
| 620 | Annual Cooling Energy (MWh) | 12.05 | 3.20-5.00 | 194.0% | Unknown |
| 620 | Peak Cooling Load (kW) | 8.68 | 2.50-3.50 | 189.5% | SolarGains |
| 610 | Peak Cooling Load (kW) | 7.23 | 2.20-2.90 | 183.4% | SolarGains |
| 900FF | Minimum Free-Floating Temperature (°C) | -10.78 | -6.40--1.60 | 169.6% | ThermalMass |
| 960 | Peak Cooling Load (kW) | 15.00 | 0.00-4.00 | 122.2% | Unknown |
| 950 | Annual Cooling Energy (MWh) | 12.27 | 0.39-0.92 | 116.1% | ModelLimitation |
| 610 | Annual Cooling Energy (MWh) | 10.48 | 3.92-6.14 | 108.4% | Unknown |
| 950 | Annual Heating Energy (MWh) | 0.00 | N/A | 100.0% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 195 | Annual Heating Energy (MWh) | 0.00 | 3.50-6.00 | 100.0% | Unknown |
| 940 | Annual Heating Energy (MWh) | 0.02 | 0.79-1.41 | 99.7% | ModelLimitation |
| 930 | Annual Heating Energy (MWh) | 0.04 | 4.14-5.34 | 99.1% | ModelLimitation |
| 920 | Annual Heating Energy (MWh) | 0.04 | 3.26-4.30 | 98.8% | ModelLimitation |
| 960 | Annual Heating Energy (MWh) | 0.10 | 1.65-2.45 | 98.7% | Unknown |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 910 | 3 | 2346.1% |
| 920 | 4 | 2082.1% |
| 900 | 4 | 1774.6% |
| 930 | 4 | 1444.8% |
| 960 | 4 | 1217.2% |
| 940 | 4 | 923.6% |
| 950 | 4 | 571.0% |
| 630 | 2 | 496.1% |
| 620 | 3 | 450.7% |
| 650 | 2 | 424.0% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
