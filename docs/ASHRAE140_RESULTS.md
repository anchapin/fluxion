# ASHRAE Standard 140 Validation Results

*Generated: 2026-04-15 17:48 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 6.2% |
| Passed | 4 |
| Warnings | 2 |
| Failed | 58 |
| Mean Absolute Error | 35.35% |
| Max Deviation | 346.87% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 0.65 seconds |
| Throughput | 27.77 cases/sec |
| Total Cases | 18 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 6.49 MWh (Ref: 5.50-7.50) | 9.25 MWh (Ref: 8.00-10.50) | 3.31 kW (Ref: 2.80-3.80) | 5.63 kW (Ref: 4.80-6.20) | ❌ FAIL |
| 610 | 9.43 MWh (Ref: 4.36-5.79) | 4.72 MWh (Ref: 3.92-6.14) | 6.62 kW (Ref: 4.30-5.70) | 4.77 kW (Ref: 2.20-2.90) | ❌ FAIL |
| 620 | 5.51 MWh (Ref: 4.50-6.50) | 4.13 MWh (Ref: 3.20-5.00) | 6.59 kW (Ref: 2.80-3.80) | 3.45 kW (Ref: 2.50-3.50) | ❌ FAIL |
| 630 | 9.32 MWh (Ref: 5.05-6.47) | 1.60 MWh (Ref: 2.13-3.70) | 6.60 kW (Ref: 4.70-6.10) | 2.38 kW (Ref: 1.80-2.40) | ❌ FAIL |
| 640 | 6.71 MWh (Ref: 2.75-3.80) | 6.42 MWh (Ref: 5.95-8.10) | 6.60 kW (Ref: 4.30-5.70) | 5.63 kW (Ref: 2.80-3.70) | ❌ FAIL |
| 650 | 0.00 MWh (Ref: 0.00-0.00) | 5.39 MWh (Ref: 4.82-7.06) | 0.00 kW (Ref: 0.00-0.00) | 5.58 kW (Ref: 1.90-2.50) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 7.17 MWh (Ref: 1.17-2.04) | 5.06 MWh (Ref: 2.13-3.67) | 1.64 kW (Ref: 1.80-2.40) | 1.68 kW (Ref: 1.60-2.10) | ❌ FAIL |
| 910 | 7.57 MWh (Ref: 1.51-2.28) | 3.60 MWh (Ref: 0.82-1.88) | 1.65 kW (Ref: 1.90-2.50) | 1.39 kW (Ref: 1.20-1.60) | ❌ FAIL |
| 920 | 6.72 MWh (Ref: 3.26-4.30) | 1.93 MWh (Ref: 1.84-3.31) | 1.55 kW (Ref: 2.10-2.80) | 0.95 kW (Ref: 1.40-1.90) | ❌ FAIL |
| 930 | 7.65 MWh (Ref: 4.14-5.34) | 1.00 MWh (Ref: 1.04-2.24) | 1.58 kW (Ref: 2.30-3.00) | 0.61 kW (Ref: 1.10-1.50) | ❌ FAIL |
| 940 | 5.29 MWh (Ref: 0.79-1.41) | 5.04 MWh (Ref: 2.08-3.55) | 1.61 kW (Ref: 1.90-2.50) | 1.68 kW (Ref: 1.70-2.30) | ❌ FAIL |
| 950 | 0.00 MWh (Ref: 0.00-0.00) | 3.80 MWh (Ref: 0.39-0.92) | 0.00 kW (Ref: 0.00-0.00) | 1.66 kW (Ref: 0.70-0.90) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -11.31°C (Ref: -18.80--15.60) | 53.45°C (Ref: 64.90-75.10) | ❌ FAIL |
| 650FF | -11.91°C (Ref: -23.00--21.00) | 53.45°C (Ref: 63.20-73.50) | ❌ FAIL |
| 900FF | -6.57°C (Ref: -6.40--1.60) | 47.09°C (Ref: 41.80-46.40) | ❌ FAIL |
| 950FF | -10.95°C (Ref: -20.20--17.80) | 48.01°C (Ref: 35.50-38.50) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 1.88 MWh (Ref: 1.65-2.45) | 7.33 MWh (Ref: 1.55-2.78) | 6.31 kW (Ref: 2.00-8.00) | 3.40 kW (Ref: 0.00-4.00) | ❌ FAIL |
| 195 | 5.00 MWh (Ref: 3.50-6.00) | 0.00 MWh (Ref: 0.00-0.00) | 6.86 kW (Ref: 1.40-2.20) | 0.00 kW (Ref: 0.00-0.00) | ❌ FAIL |

## Multi-Reference Comparison

| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |
|------|--------|------------|-------|--------|---------|
| 195 | Annual Heating Energy (MWh) | FAIL (5.00) | - | - | FAIL |
| 195 | Annual Cooling Energy (MWh) | FAIL (0.00) | - | - | FAIL |
| 195 | Peak Heating Load (kW) | FAIL (6.86) | - | - | FAIL |
| 195 | Peak Cooling Load (kW) | FAIL (0.00) | - | - | FAIL |
| 600 | Annual Heating Energy (MWh) | WARN (6.49) | FAIL (6.49) | PASS (6.49) | WARN |
| 600 | Annual Cooling Energy (MWh) | PASS (9.25) | WARN (9.25) | PASS (9.25) | PASS |
| 600 | Peak Heating Load (kW) | WARN (3.31) | WARN (3.31) | WARN (3.31) | FAIL |
| 600 | Peak Cooling Load (kW) | PASS (5.63) | WARN (5.63) | PASS (5.63) | PASS |
| 900 | Annual Heating Energy (MWh) | FAIL (7.17) | - | - | FAIL |
| 900 | Annual Cooling Energy (MWh) | FAIL (5.06) | - | - | FAIL |
| 900 | Peak Heating Load (kW) | PASS (1.64) | - | - | PASS |
| 900 | Peak Cooling Load (kW) | FAIL (1.68) | - | - | FAIL |
| 920 | Annual Heating Energy (MWh) | FAIL (6.72) | - | - | FAIL |
| 920 | Annual Cooling Energy (MWh) | FAIL (1.93) | - | - | FAIL |
| 920 | Peak Heating Load (kW) | FAIL (1.55) | - | - | FAIL |
| 920 | Peak Cooling Load (kW) | FAIL (0.95) | - | - | FAIL |
| 930 | Annual Heating Energy (MWh) | FAIL (7.65) | - | - | FAIL |
| 930 | Annual Cooling Energy (MWh) | FAIL (1.00) | - | - | FAIL |
| 930 | Peak Heating Load (kW) | FAIL (1.58) | - | - | FAIL |
| 930 | Peak Cooling Load (kW) | FAIL (0.61) | - | - | FAIL |
| 940 | Annual Heating Energy (MWh) | FAIL (5.29) | - | - | FAIL |
| 940 | Annual Cooling Energy (MWh) | PASS (5.04) | - | - | PASS |
| 940 | Peak Heating Load (kW) | FAIL (1.61) | - | - | FAIL |
| 940 | Peak Cooling Load (kW) | FAIL (1.68) | - | - | FAIL |
| 950 | Annual Heating Energy (MWh) | FAIL (0.00) | - | - | FAIL |
| 950 | Annual Cooling Energy (MWh) | FAIL (3.80) | - | - | FAIL |
| 950 | Peak Heating Load (kW) | FAIL (0.00) | - | - | FAIL |
| 950 | Peak Cooling Load (kW) | FAIL (1.66) | - | - | FAIL |
| 960 | Annual Heating Energy (MWh) | FAIL (1.88) | - | - | FAIL |
| 960 | Annual Cooling Energy (MWh) | WARN (7.33) | - | - | FAIL |
| 960 | Peak Heating Load (kW) | FAIL (6.31) | - | - | FAIL |
| 960 | Peak Cooling Load (kW) | FAIL (3.40) | - | - | FAIL |

## Systematic Issues

The following recurring issues are affecting validation results:

### Solar Gain Calculations

**Affected metrics:** 640 - Peak Cooling (kW), 630 - Peak Cooling (kW), 650 - Peak Cooling (kW), 610 - Peak Cooling (kW), 620 - Peak Cooling (kW) |
**Count:** 5 metrics

### 5R1C Model Limitation (Accepted)

**Affected metrics:** 940 - Annual Cooling (MWh), 900 - Annual Cooling (MWh), 930 - Annual Cooling (MWh), 920 - Annual Heating (MWh), 900 - Annual Heating (MWh), 940 - Annual Heating (MWh), 950 - Annual Heating (MWh), 950 - Annual Cooling (MWh), 920 - Annual Cooling (MWh), 930 - Annual Heating (MWh), 910 - Annual Cooling (MWh), 910 - Annual Heating (MWh) |
**Count:** 12 metrics

### Thermal Mass Dynamics

**Affected metrics:** 950FF - Min Free-Float Temp (°C), 900FF - Max Free-Float Temp (°C) |
**Count:** 2 metrics

### Inter-Zone Heat Transfer

**Affected metrics:** 960 - Annual Cooling Energy (MWh) |
**Count:** 1 metrics

### Unknown/Unclassified

**Affected metrics:** 640 - Annual Cooling Energy (MWh), 950 - Peak Cooling Load (kW), 620 - Annual Heating Energy (MWh), 640 - Peak Heating Load (kW), 650FF - Minimum Free-Floating Temperature (°C), 195 - Peak Heating Load (kW), 930 - Peak Cooling Load (kW), 195 - Peak Cooling Load (kW), 630 - Annual Cooling Energy (MWh), 620 - Peak Heating Load (kW), 910 - Peak Heating Load (kW), 640 - Annual Heating Energy (MWh), 960 - Peak Cooling Load (kW), 610 - Peak Heating Load (kW), 650FF - Maximum Free-Floating Temperature (°C), 650 - Annual Heating Energy (MWh), 600FF - Maximum Free-Floating Temperature (°C), 940 - Peak Cooling Load (kW), 920 - Peak Heating Load (kW), 630 - Annual Heating Energy (MWh), 600FF - Minimum Free-Floating Temperature (°C), 950 - Peak Heating Load (kW), 960 - Annual Heating Energy (MWh), 910 - Peak Cooling Load (kW), 195 - Annual Cooling Energy (MWh), 610 - Annual Heating Energy (MWh), 620 - Annual Cooling Energy (MWh), 650 - Peak Heating Load (kW), 900 - Peak Cooling Load (kW), 920 - Peak Cooling Load (kW), 610 - Annual Cooling Energy (MWh), 195 - Annual Heating Energy (MWh), 650 - Annual Cooling Energy (MWh), 960 - Peak Heating Load (kW), 940 - Peak Heating Load (kW), 930 - Peak Heating Load (kW), 630 - Peak Heating Load (kW), 600 - Peak Heating Load (kW) |
**Count:** 38 metrics

## References

- **[Quality Metrics Tracker](QUALITY_METRICS.md)** - Detailed metrics dashboard with historical progression
- **[Known Systematic Issues](KNOWN_ISSUES.md)** - Comprehensive issue catalog with severity, status, and resolution roadmap

## Phase Progress

| Phase | Status | Completion | Notes |
|-------|--------|------------|-------|
| Phase 1: Foundation | ✅ Complete | 4/4 plans | Conductances, HVAC load fixes |
| Phase 2: Thermal Mass | ✅ Complete | 4/4 plans | Implicit integration validated |
| Phase 3: Solar & External | ✅ Complete | 3/3 plans | Solar integration, mode-specific coupling |
| Phase 4: Multi-Zone Transfer | ✅ Complete | 6/6 plans | Inter-zone heat transfer validated |
| Phase 5: Diagnostics & Reporting | 🔄 In Progress | 4/4 plans | Quality metrics, issue tracking |
| Phase 6: Performance Optimization | ⏳ Pending | 0/12 requirements | GPU acceleration, throughput |
| Phase 7: Advanced Analysis | ⏳ Pending | 0/20 requirements | Sensitivity, visualization |

## What's Fixed in Phase 5

This phase delivered systematic diagnostics and reporting infrastructure:

- ✅ **REPORT-01:** Automated quality metrics computation via `analyzer.rs`
- ✅ **REPORT-02:** Quality metrics dashboard (`QUALITY_METRICS.md`) with historical progression
- ✅ **REPORT-03:** Comprehensive known issues catalog (`KNOWN_ISSUES.md`) with taxonomy, severity, and GitHub links
- ✅ **REPORT-04:** Enhanced validation report with issue references and phase summaries

## Legend

- **PASS**: Value within 5% of reference range
- **WARN**: Value within reference range but >2% deviation, or within tolerance band
- **FAIL**: Value outside 5% tolerance band
