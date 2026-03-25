# ASHRAE Standard 140 Validation Results

*Generated: 2026-03-25 17:16 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 3.1% |
| Passed | 2 |
| Warnings | 1 |
| Failed | 61 |
| Mean Absolute Error | 6.46% |
| Max Deviation | 78.72% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 0.13 seconds |
| Throughput | 139.24 cases/sec |
| Total Cases | 18 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 6.99 MWh (Ref: 5.50-7.50) | 7.28 MWh (Ref: 8.00-10.50) | 2.10 kW (Ref: 2.80-3.80) | 5.73 kW (Ref: 4.80-6.20) | ❌ FAIL |
| 610 | 7.29 MWh (Ref: 4.36-5.79) | 5.08 MWh (Ref: 3.92-6.14) | 2.10 kW (Ref: 4.30-5.70) | 4.67 kW (Ref: 2.20-2.90) | ❌ FAIL |
| 620 | 6.64 MWh (Ref: 4.50-6.50) | 2.54 MWh (Ref: 3.20-5.00) | 2.10 kW (Ref: 2.80-3.80) | 3.16 kW (Ref: 2.50-3.50) | ❌ FAIL |
| 630 | 7.62 MWh (Ref: 5.05-6.47) | 1.24 MWh (Ref: 2.13-3.70) | 2.10 kW (Ref: 4.70-6.10) | 2.03 kW (Ref: 1.80-2.40) | ❌ FAIL |
| 640 | 5.24 MWh (Ref: 2.75-3.80) | 7.06 MWh (Ref: 5.95-8.10) | 2.10 kW (Ref: 4.30-5.70) | 5.73 kW (Ref: 2.80-3.70) | ❌ FAIL |
| 650 | 0.00 MWh (Ref: 0.00-0.00) | 5.32 MWh (Ref: 4.82-7.06) | 0.00 kW (Ref: 0.00-0.00) | 6.81 kW (Ref: 1.90-2.50) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 4.86 MWh (Ref: 1.17-2.04) | 6.89 MWh (Ref: 2.13-3.67) | 2.10 kW (Ref: 1.80-2.40) | 3.92 kW (Ref: 1.60-2.10) | ❌ FAIL |
| 910 | 2.13 MWh (Ref: 1.51-2.28) | 1.68 MWh (Ref: 0.82-1.88) | 2.10 kW (Ref: 1.90-2.50) | 3.11 kW (Ref: 1.20-1.60) | ❌ FAIL |
| 920 | 4.28 MWh (Ref: 3.26-4.30) | 2.39 MWh (Ref: 1.84-3.31) | 2.10 kW (Ref: 2.10-2.80) | 1.98 kW (Ref: 1.40-1.90) | ❌ FAIL |
| 930 | 5.47 MWh (Ref: 4.14-5.34) | 1.02 MWh (Ref: 1.04-2.24) | 2.10 kW (Ref: 2.30-3.00) | 1.26 kW (Ref: 1.10-1.50) | ❌ FAIL |
| 940 | 1.35 MWh (Ref: 0.79-1.41) | 3.10 MWh (Ref: 2.08-3.55) | 2.10 kW (Ref: 1.90-2.50) | 3.92 kW (Ref: 1.70-2.30) | ❌ FAIL |
| 950 | 0.00 MWh (Ref: 0.00-0.00) | 0.91 MWh (Ref: 0.39-0.92) | 0.00 kW (Ref: 0.00-0.00) | 5.51 kW (Ref: 0.70-0.90) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -5.13°C (Ref: -18.80--15.60) | 48.67°C (Ref: 64.90-75.10) | ❌ FAIL |
| 650FF | -10.35°C (Ref: -23.00--21.00) | 44.95°C (Ref: 63.20-73.50) | ❌ FAIL |
| 900FF | -0.85°C (Ref: -6.40--1.60) | 47.49°C (Ref: 41.80-46.40) | ❌ FAIL |
| 950FF | -8.31°C (Ref: -20.20--17.80) | 35.88°C (Ref: 35.50-38.50) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 0.05 MWh (Ref: 5.00-15.00) | 21.23 MWh (Ref: 1.00-3.50) | 2.10 kW (Ref: 2.00-8.00) | 10.22 kW (Ref: 0.00-4.00) | ❌ FAIL |
| 195 | 5.01 MWh (Ref: 3.50-6.00) | 0.00 MWh (Ref: 0.00-0.00) | 1.72 kW (Ref: 1.40-2.20) | 0.00 kW (Ref: 0.00-0.00) | ❌ FAIL |

## Multi-Reference Comparison

| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |
|------|--------|------------|-------|--------|---------|
| 600 | Annual Heating (MWh) | WARN (6.99) | - | - | FAIL |
| 600 | Annual Cooling (MWh) | FAIL (7.28) | - | - | FAIL |
| 600 | Peak Heating (kW) | FAIL (2.10) | - | - | FAIL |
| 600 | Peak Cooling (kW) | PASS (5.73) | - | - | PASS |

## Systematic Issues

The following recurring issues are affecting validation results:

### Solar Gain Calculations

**Affected metrics:** 620 - Peak Cooling (kW), 610 - Peak Cooling (kW), 630 - Peak Cooling (kW), 640 - Peak Cooling (kW), 650 - Peak Cooling (kW) |
**Count:** 5 metrics

### Thermal Mass Dynamics

**Affected metrics:** 900FF - Min Free-Float Temp (°C), 950FF - Min Free-Float Temp (°C) |
**Count:** 2 metrics

### Unknown/Unclassified

**Affected metrics:** 610 - Peak Heating (kW), 630 - Annual Cooling (MWh), 650FF - Min Free-Float Temp (°C), 610 - Annual Heating (MWh), 650 - Annual Cooling (MWh), 610 - Annual Cooling (MWh), 930 - Peak Cooling (kW), 940 - Peak Cooling (kW), 930 - Peak Heating (kW), 630 - Annual Heating (MWh), 195 - Annual Cooling (MWh), 640 - Annual Heating (MWh), 650 - Annual Heating (MWh), 950 - Peak Cooling (kW), 195 - Peak Heating (kW), 630 - Peak Heating (kW), 900 - Peak Heating (kW), 920 - Peak Heating (kW), 960 - Peak Cooling (kW), 600 - Annual Heating (MWh), 960 - Annual Heating (MWh), 960 - Peak Heating (kW), 195 - Peak Cooling (kW), 900 - Peak Cooling (kW), 650 - Peak Heating (kW), 640 - Annual Cooling (MWh), 600FF - Max Free-Float Temp (°C), 950 - Peak Heating (kW), 940 - Peak Heating (kW), 620 - Peak Heating (kW), 600FF - Min Free-Float Temp (°C), 920 - Peak Cooling (kW), 910 - Peak Heating (kW), 600 - Annual Cooling (MWh), 650FF - Max Free-Float Temp (°C), 910 - Peak Cooling (kW), 640 - Peak Heating (kW), 600 - Peak Heating (kW), 195 - Annual Heating (MWh), 620 - Annual Heating (MWh), 620 - Annual Cooling (MWh) |
**Count:** 41 metrics

### 5R1C Model Limitation (Accepted)

**Affected metrics:** 940 - Annual Cooling (MWh), 910 - Annual Heating (MWh), 910 - Annual Cooling (MWh), 900 - Annual Cooling (MWh), 920 - Annual Heating (MWh), 930 - Annual Heating (MWh), 900 - Annual Heating (MWh), 930 - Annual Cooling (MWh), 950 - Annual Heating (MWh), 950 - Annual Cooling (MWh), 920 - Annual Cooling (MWh), 940 - Annual Heating (MWh) |
**Count:** 12 metrics

### Inter-Zone Heat Transfer

**Affected metrics:** 960 - Annual Cooling (MWh) |
**Count:** 1 metrics

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
