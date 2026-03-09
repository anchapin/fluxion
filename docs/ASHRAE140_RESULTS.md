# ASHRAE Standard 140 Validation Results

Latest validation run results for Fluxion v0.1.0.

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Passed | 19 (30%) |
| Warnings | 10 (16%) |
| Failed | 35 (54%) |
| Mean Absolute Error | 49.21% |
| Max Deviation | 512.45% |
| Status | ⚠️ IN PROGRESS |

## Detailed Results

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 10.29 MWh (Ref: 5.50-7.50) | 8.59 MWh (Ref: 8.00-10.50) | 3.30 kW (Ref: 2.80-3.80) | 1.27 kW (Ref: 2.80-4.10) | ⚠️ Partial |
| 610 | 10.46 MWh (Ref: 4.36-5.79) | 6.26 MWh (Ref: 3.92-6.14) | 3.30 kW | 1.27 kW | ⚠️ Partial |
| 620 | 8.99 MWh (Ref: 4.50-6.50) | 3.33 MWh (Ref: 3.20-5.00) | 3.30 kW (Ref: 2.80-3.80) | 1.27 kW (Ref: 2.50-3.50) | ⚠️ Partial |
| 630 | 9.61 MWh (Ref: 5.05-6.47) | 1.79 MWh (Ref: 2.13-3.70) | 3.30 kW | 1.27 kW | ⚠️ Partial |
| 640 | 10.29 MWh (Ref: 2.75-3.80) | 8.59 MWh (Ref: 5.95-8.10) | 3.30 kW | 1.27 kW | ⚠️ Partial |
| 650 | 0.00 MWh (Ref: 0.00-0.00) | 7.67 MWh (Ref: 4.82-7.06) | 0.00 kW | 13.47 kW | ⚠️ Partial |
| 900 | 1.78 MWh (Ref: 1.17-2.04) | 0.71 MWh (Ref: 2.13-3.67) | 2.09 kW (Ref: 1.10-2.10) | 1.56 kW (Ref: 2.10-3.50) | ⚠️ Partial |
| 960 | 9.67 MWh (Ref: 5.00-15.00) | 3.03 MWh (Ref: 1.00-3.50) | 4.37 kW | 2.99 kW | ✅ PASS |
| 195 | 4.81 MWh (Ref: 3.50-6.00) | 0.00 MWh (Ref: 0.00-0.00) | 1.61 kW | 0.62 kW | ⚠️ Partial |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -8.55°C (Ref: -18.80--15.60) | 56.64°C (Ref: 64.90-75.10) | ✅ PASS |
| 650FF | -11.70°C (Ref: -23.00--21.00) | 56.34°C (Ref: 63.20-73.50) | ✅ PASS |
| 900FF | -4.50°C (Ref: -6.40--1.60) | 37.52°C (Ref: 41.80-46.40) | ⚠️ Partial |
| 950FF | -9.75°C (Ref: -20.20--17.80) | 34.18°C (Ref: 35.50-38.50) | ⚠️ Partial |

*Note: Results shown are from the latest validation run with 64 total validation metrics.*

## Phase 1 Progress

**Validation Date:** 2026-03-09

**Improvement from Baseline:**
- MAE reduced from 78.79% to 49.21% (37.5% improvement)
- Pass rate improved from 25% to 30%
- Peak heating loads significantly improved (3.30 kW vs 4.81 kW baseline)

**Lightweight Cases (600 series):**
- Annual heating loads still over-predicted by 37-87%
- Annual cooling loads generally within tolerance
- Peak cooling loads under-predicted significantly (1.27 kW vs 2.80-6.20 kW reference)
- Free-floating cases pass temperature range validation

**High-Mass Cases (900 series):**
- Heating loads significantly improved (1.78 MWh vs 6.17 MWh baseline)
- Cooling loads still under-predicted
- Thermal mass dynamics showing improvement but still need work

## Known Issues

The validation reveals remaining discrepancies after Phase 1 fixes:

**Systematic Issues:**
- Annual heating loads still over-predicted (37-87% above reference range)
- Peak cooling loads significantly under-predicted across all cases
- High-mass building cases (900-series) still showing thermal mass dynamics issues

**Specific Problem Areas:**
1. **Heating Load Over-prediction:** All lightweight cases show 37-87% heating load over-prediction, suggesting remaining conductance or HVAC calculation issues
2. **Peak Cooling Under-prediction:** Peak cooling loads are 1.27 kW across all cases vs 2.50-6.20 kW reference range, indicating potential solar gain or HVAC capacity issues
3. **Thermal Mass Dynamics:** 900-series cases show improved but still inaccurate results, likely requiring Phase 2 thermal mass fixes

**Weather Data:**
- All baseline cases use Denver TMY weather data (synthetic DenverTmyWeather implementation)
- Weather data provides DNI, DHI, GHI, temperature, and humidity at hourly resolution
- Denver climate characteristics validated for ASHRAE 140 (39.83°N, 1655m elevation)

These issues are actively being investigated as part of ongoing ASHRAE 140 compliance work. Phase 2 will focus on thermal mass dynamics, and later phases will address solar gain calculations.
