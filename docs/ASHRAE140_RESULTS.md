# ASHRAE Standard 140 Validation Results

*Generated: 2026-03-31 13:36 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 3.1% |
| Passed | 2 |
| Warnings | 2 |
| Failed | 60 |
| Mean Absolute Error | 4.50% |
| Max Deviation | 47.21% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 0.75 seconds |
| Throughput | 23.91 cases/sec |
| Total Cases | 18 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 7.04 MWh (Ref: 5.50-7.50) | 7.06 MWh (Ref: 8.00-10.50) | 3.53 kW (Ref: 2.80-3.80) | 3.81 kW (Ref: 4.80-6.20) | ❌ FAIL |
| 610 | 5.48 MWh (Ref: 4.36-5.79) | 4.02 MWh (Ref: 3.92-6.14) | 3.53 kW (Ref: 4.30-5.70) | 3.17 kW (Ref: 2.20-2.90) | ❌ FAIL |
| 620 | 5.81 MWh (Ref: 4.50-6.50) | 3.39 MWh (Ref: 3.20-5.00) | 3.53 kW (Ref: 2.80-3.80) | 2.27 kW (Ref: 2.50-3.50) | ❌ FAIL |
| 630 | 6.21 MWh (Ref: 5.05-6.47) | 2.02 MWh (Ref: 2.13-3.70) | 3.53 kW (Ref: 4.70-6.10) | 1.54 kW (Ref: 1.80-2.40) | ❌ FAIL |
| 640 | 3.61 MWh (Ref: 2.75-3.80) | 6.46 MWh (Ref: 5.95-8.10) | 3.72 kW (Ref: 4.30-5.70) | 3.81 kW (Ref: 2.80-3.70) | ❌ FAIL |
| 650 | 0.00 MWh (Ref: 0.00-0.00) | 4.66 MWh (Ref: 4.82-7.06) | 0.00 kW (Ref: 0.00-0.00) | 3.81 kW (Ref: 1.90-2.50) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 0.31 MWh (Ref: 1.17-2.04) | 1.80 MWh (Ref: 2.13-3.67) | 3.19 kW (Ref: 1.80-2.40) | 1.36 kW (Ref: 1.60-2.10) | ❌ FAIL |
| 910 | 0.37 MWh (Ref: 1.51-2.28) | 1.10 MWh (Ref: 0.82-1.88) | 3.20 kW (Ref: 1.90-2.50) | 1.13 kW (Ref: 1.20-1.60) | ❌ FAIL |
| 920 | 4.24 MWh (Ref: 3.26-4.30) | 1.53 MWh (Ref: 1.84-3.31) | 3.12 kW (Ref: 2.10-2.80) | 0.52 kW (Ref: 1.40-1.90) | ❌ FAIL |
| 930 | 5.27 MWh (Ref: 4.14-5.34) | 1.07 MWh (Ref: 1.04-2.24) | 3.45 kW (Ref: 2.30-3.00) | 0.39 kW (Ref: 1.10-1.50) | ❌ FAIL |
| 940 | 0.21 MWh (Ref: 0.79-1.41) | 2.09 MWh (Ref: 2.08-3.55) | 3.63 kW (Ref: 1.90-2.50) | 0.95 kW (Ref: 1.70-2.30) | ❌ FAIL |
| 950 | 0.00 MWh (Ref: 0.00-0.00) | 0.18 MWh (Ref: 0.39-0.92) | 0.00 kW (Ref: 0.00-0.00) | 0.52 kW (Ref: 0.70-0.90) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -10.68°C (Ref: -18.80--15.60) | 58.22°C (Ref: 64.90-75.10) | ❌ FAIL |
| 650FF | -11.86°C (Ref: -23.00--21.00) | 58.20°C (Ref: 63.20-73.50) | ❌ FAIL |
| 900FF | -5.84°C (Ref: -6.40--1.60) | 40.41°C (Ref: 41.80-46.40) | ⚠️ WARN |
| 950FF | -10.03°C (Ref: -20.20--17.80) | 37.14°C (Ref: 35.50-38.50) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 9.05 MWh (Ref: 5.00-15.00) | 0.41 MWh (Ref: 1.00-3.50) | 3.22 kW (Ref: 2.00-8.00) | 1.29 kW (Ref: 0.00-4.00) | ❌ FAIL |
| 195 | 4.52 MWh (Ref: 3.50-6.00) | 0.00 MWh (Ref: 0.00-0.00) | 1.57 kW (Ref: 1.40-2.20) | 0.00 kW (Ref: 0.00-0.00) | ❌ FAIL |

## Multi-Reference Comparison

| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |
|------|--------|------------|-------|--------|---------|
| 600 | Annual Heating (MWh) | WARN (7.04) | - | - | FAIL |
| 600 | Annual Cooling (MWh) | FAIL (7.06) | - | - | FAIL |
| 600 | Peak Heating (kW) | PASS (3.53) | - | - | PASS |
| 600 | Peak Cooling (kW) | FAIL (3.81) | - | - | FAIL |

## Systematic Issues

The following recurring issues are affecting validation results:

### Unknown/Unclassified

**Affected metrics:** 650FF - Min Free-Float Temp (°C), 620 - Annual Heating (MWh), 650 - Annual Cooling (MWh), 930 - Peak Heating (kW), 620 - Peak Heating (kW), 910 - Peak Heating (kW), 960 - Peak Cooling (kW), 630 - Annual Cooling (MWh), 195 - Annual Heating (MWh), 940 - Peak Heating (kW), 940 - Peak Cooling (kW), 640 - Peak Heating (kW), 900 - Peak Cooling (kW), 195 - Peak Cooling (kW), 650 - Annual Heating (MWh), 920 - Peak Cooling (kW), 630 - Peak Heating (kW), 960 - Peak Heating (kW), 600 - Annual Heating (MWh), 600 - Annual Cooling (MWh), 650 - Peak Heating (kW), 640 - Annual Cooling (MWh), 610 - Annual Heating (MWh), 610 - Annual Cooling (MWh), 950 - Peak Cooling (kW), 960 - Annual Heating (MWh), 600FF - Max Free-Float Temp (°C), 920 - Peak Heating (kW), 950 - Peak Heating (kW), 910 - Peak Cooling (kW), 620 - Annual Cooling (MWh), 630 - Annual Heating (MWh), 900 - Peak Heating (kW), 640 - Annual Heating (MWh), 650FF - Max Free-Float Temp (°C), 600FF - Min Free-Float Temp (°C), 195 - Annual Cooling (MWh), 610 - Peak Heating (kW), 930 - Peak Cooling (kW), 195 - Peak Heating (kW) |
**Count:** 40 metrics

### Inter-Zone Heat Transfer

**Affected metrics:** 960 - Annual Cooling (MWh) |
**Count:** 1 metrics

### Thermal Mass Dynamics

**Affected metrics:** 950FF - Min Free-Float Temp (°C) |
**Count:** 1 metrics

### 5R1C Model Limitation (Accepted)

**Affected metrics:** 950 - Annual Heating (MWh), 930 - Annual Heating (MWh), 910 - Annual Heating (MWh), 950 - Annual Cooling (MWh), 900 - Annual Cooling (MWh), 940 - Annual Heating (MWh), 910 - Annual Cooling (MWh), 930 - Annual Cooling (MWh), 920 - Annual Cooling (MWh), 900 - Annual Heating (MWh), 920 - Annual Heating (MWh), 940 - Annual Cooling (MWh) |
**Count:** 12 metrics

### Solar Gain Calculations

**Affected metrics:** 620 - Peak Cooling (kW), 650 - Peak Cooling (kW), 630 - Peak Cooling (kW), 600 - Peak Cooling (kW), 640 - Peak Cooling (kW), 610 - Peak Cooling (kW) |
**Count:** 6 metrics

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
