# Quality Metrics Tracker

*Generated: 2026-06-14 01:11 UTC

## Current Status

- **Pass Rate:** 0.0% (0 / 18 cases)
- **MAE:** -inf%
- **Max Deviation:** 156688407098920499671599176703523835049053393089600476937248427270002086959617548118529236642264993536098982690441995762405069308997445931246993091573702227637762538928709299735107800767136447229159732556243428836614946535382354807001923451373945131103098418491807716366513906433570485245860289940815872.00%

### Status Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| FAIL | 56 | 87.5% |
| PASS | 6 | 9.4% |
| WARN | 2 | 3.1% |

## Phase Progression

| Phase | Pass Rate | MAE | Max Dev | Notes |
|-------|-----------|-----|---------|-------|
| Baseline | 25% | 78.79% | 512% | Initial state |
| Phase 1 | 30% | 49.21% | 512% | Foundation fixes |
| Phase 2 | 35% | 38.5% | 250% | Thermal mass |
| Phase 3 | 42% | 32.1% | 200% | Solar improvements |
| Phase 4 | 47% | 28.4% | 180% | Multi-zone correct |
| Current (Phase 5) | 0.0% | -inf% | 156688407098920499671599176703523835049053393089600476937248427270002086959617548118529236642264993536098982690441995762405069308997445931246993091573702227637762538928709299735107800767136447229159732556243428836614946535382354807001923451373945131103098418491807716366513906433570485245860289940815872% | Diagnostics |

## Metric Deviations

| Case | Metric | Actual | Ref Range | Error | Issue |
|------|--------|--------|-----------|-------|-------|
| 950FF | Minimum Free-Floating Temperature (°C) | -inf | -20.20--17.80 | inf% | ThermalMass |
| 950FF | Maximum Free-Floating Temperature (°C) | 57974710626600581452404738247491642281413759705781285924358413466655938510560128939928543460066370960288826996319306673193616031724445086314023038146807073475898480543705223098025069892892935910839612787588082920516423135553759955810618656361723975394190456278660631795858222414073504395012943609069568.00 | 35.50-38.50 | 156688407098920499671599176703523835049053393089600476937248427270002086959617548118529236642264993536098982690441995762405069308997445931246993091573702227637762538928709299735107800767136447229159732556243428836614946535382354807001923451373945131103098418491807716366513906433570485245860289940815872.0% | ThermalMass |
| 900FF | Minimum Free-Floating Temperature (°C) | 0.03 | -6.40--1.60 | 100.7% | ThermalMass |
| 620 | Annual Cooling Energy (MWh) | 0.00 | 3.20-5.00 | 100.0% | Unknown |
| 620 | Peak Cooling Load (kW) | 0.00 | 2.50-3.50 | 100.0% | SolarGains |
| 630 | Annual Cooling Energy (MWh) | 0.00 | 2.13-3.70 | 100.0% | Unknown |
| 630 | Peak Cooling Load (kW) | 0.00 | 1.80-2.40 | 100.0% | SolarGains |
| 950 | Annual Heating Energy (MWh) | 0.00 | N/A | 100.0% | ModelLimitation |
| 950 | Peak Heating Load (kW) | 0.00 | N/A | 100.0% | Unknown |
| 960 | Annual Cooling Energy (MWh) | 0.00 | 1.55-2.78 | 100.0% | InterZoneTransfer |
| 960 | Peak Cooling Load (kW) | 0.00 | 0.00-4.00 | 100.0% | Unknown |
| 610 | Annual Cooling Energy (MWh) | 0.02 | 3.92-6.14 | 99.6% | Unknown |
| 600 | Annual Cooling Energy (MWh) | 0.08 | 8.00-10.50 | 99.0% | Unknown |
| 640 | Annual Cooling Energy (MWh) | 0.08 | 5.95-8.10 | 98.8% | Unknown |
| 650 | Annual Cooling Energy (MWh) | 0.08 | 4.82-7.06 | 98.6% | Unknown |
| 950 | Annual Cooling Energy (MWh) | 0.10 | 0.39-0.92 | 98.3% | ModelLimitation |
| 940 | Annual Cooling Energy (MWh) | 0.10 | 2.08-3.55 | 97.9% | ModelLimitation |
| 930 | Annual Cooling Energy (MWh) | 0.10 | 1.04-2.24 | 97.8% | ModelLimitation |
| 920 | Annual Cooling Energy (MWh) | 0.10 | 1.84-3.31 | 97.2% | ModelLimitation |
| 900 | Annual Cooling Energy (MWh) | 0.10 | 2.13-3.67 | 96.4% | ModelLimitation |
| 910 | Annual Cooling Energy (MWh) | 0.10 | 0.82-1.88 | 92.2% | ModelLimitation |
| 600 | Peak Cooling Load (kW) | 0.74 | 4.80-6.20 | 86.0% | SolarGains |
| 960 | Peak Heating Load (kW) | 1.27 | 2.00-8.00 | 85.0% | Unknown |
| 610 | Peak Cooling Load (kW) | 0.51 | 2.20-2.90 | 80.0% | SolarGains |
| 900 | Annual Heating Energy (MWh) | 2.87 | 1.17-2.04 | 79.0% | ModelLimitation |
| 950 | Peak Cooling Load (kW) | 1.32 | 0.70-0.90 | 78.3% | Unknown |
| 640 | Peak Cooling Load (kW) | 0.74 | 2.80-3.70 | 77.2% | SolarGains |
| 940 | Peak Cooling Load (kW) | 1.25 | 1.70-2.30 | 77.1% | Unknown |
| 930 | Peak Cooling Load (kW) | 1.25 | 1.10-1.50 | 73.7% | Unknown |
| 940 | Peak Heating Load (kW) | 1.90 | 1.90-2.50 | 72.3% | Unknown |

## Problematic Cases

Cases with the highest number of failing metrics:

| Case | Failing Metrics | Total Error |
|------|-----------------|-------------|
| 950FF | 2 | inf% |
| 950 | 4 | 376.5% |
| 940 | 4 | 311.7% |
| 630 | 4 | 308.1% |
| 960 | 4 | 305.4% |
| 930 | 4 | 275.7% |
| 620 | 4 | 273.3% |
| 610 | 4 | 268.8% |
| 920 | 4 | 244.1% |
| 900 | 3 | 230.9% |

---
*Note: MAE = Mean Absolute Error of percent deviation from reference midpoints.*
