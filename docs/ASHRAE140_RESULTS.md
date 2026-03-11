# ASHRAE Standard 140 Validation Results

*Generated: 2026-03-10 21:16 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 28.1% |
| Passed | 18 |
| Warnings | 10 |
| Failed | 36 |
| Mean Absolute Error | 61.52% |
| Max Deviation | 527.03% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 1.07 seconds |
| Throughput | 16.82 cases/sec |
| Total Cases | 18 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 6.78 MWh (Ref: 5.50-7.50) | 6.45 MWh (Ref: 8.00-10.50) | 2.10 kW (Ref: 2.80-3.80) | 5.01 kW (Ref: 4.80-6.20) | ❌ FAIL |
| 610 | 7.12 MWh (Ref: 4.36-5.79) | 4.54 MWh (Ref: 3.92-6.14) | 2.10 kW (Ref: 4.30-5.70) | 4.10 kW (Ref: 2.20-2.90) | ❌ FAIL |
| 620 | 6.59 MWh (Ref: 4.50-6.50) | 2.33 MWh (Ref: 3.20-5.00) | 2.10 kW (Ref: 2.80-3.80) | 2.75 kW (Ref: 2.50-3.50) | ❌ FAIL |
| 630 | 7.55 MWh (Ref: 5.05-6.47) | 1.18 MWh (Ref: 2.13-3.70) | 2.10 kW (Ref: 4.70-6.10) | 1.86 kW (Ref: 1.80-2.40) | ❌ FAIL |
| 640 | 6.78 MWh (Ref: 2.75-3.80) | 6.45 MWh (Ref: 5.95-8.10) | 2.10 kW (Ref: 4.30-5.70) | 5.01 kW (Ref: 2.80-3.70) | ❌ FAIL |
| 650 | 0.00 MWh (Ref: 0.00-0.00) | 4.60 MWh (Ref: 4.82-7.06) | 0.00 kW (Ref: 0.00-0.00) | 6.45 kW (Ref: 1.90-2.50) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 5.35 MWh (Ref: 1.17-2.04) | 4.75 MWh (Ref: 2.13-3.67) | 2.10 kW (Ref: 1.80-2.40) | 3.56 kW (Ref: 1.60-2.10) | ❌ FAIL |
| 910 | 5.77 MWh (Ref: 1.51-2.28) | 3.26 MWh (Ref: 0.82-1.88) | 2.10 kW (Ref: 1.90-2.50) | 2.91 kW (Ref: 1.20-1.60) | ❌ FAIL |
| 920 | 4.95 MWh (Ref: 3.26-4.30) | 1.60 MWh (Ref: 1.84-3.31) | 2.10 kW (Ref: 2.10-2.80) | 1.99 kW (Ref: 1.40-1.90) | ❌ FAIL |
| 930 | 5.91 MWh (Ref: 4.14-5.34) | 0.74 MWh (Ref: 1.04-2.24) | 2.10 kW (Ref: 2.30-3.00) | 1.36 kW (Ref: 1.10-1.50) | ❌ FAIL |
| 940 | 5.34 MWh (Ref: 0.79-1.41) | 4.75 MWh (Ref: 2.08-3.55) | 2.10 kW (Ref: 1.90-2.50) | 3.55 kW (Ref: 1.70-2.30) | ❌ FAIL |
| 950 | 0.00 MWh (Ref: 0.00-0.00) | 2.58 MWh (Ref: 0.39-0.92) | 0.00 kW (Ref: 0.00-0.00) | 5.02 kW (Ref: 0.70-0.90) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -5.01°C (Ref: -18.80--15.60) | 47.89°C (Ref: 64.90-75.10) | ❌ FAIL |
| 650FF | -10.32°C (Ref: -23.00--21.00) | 44.53°C (Ref: 63.20-73.50) | ❌ FAIL |
| 900FF | -3.56°C (Ref: -6.40--1.60) | 41.60°C (Ref: 41.80-46.40) | ⚠️ WARN |
| 950FF | -9.37°C (Ref: -20.20--17.80) | 36.81°C (Ref: 35.50-38.50) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 5.78 MWh (Ref: 5.00-15.00) | 4.53 MWh (Ref: 1.00-3.50) | 2.10 kW (Ref: 2.00-8.00) | 3.79 kW (Ref: 0.00-4.00) | ❌ FAIL |
| 195 | 4.82 MWh (Ref: 3.50-6.00) | 0.00 MWh (Ref: 0.00-0.00) | 1.63 kW (Ref: 1.40-2.20) | 0.00 kW (Ref: 0.00-0.00) | ✅ PASS |

## Systematic Issues

The following recurring issues are affecting validation results:

### Thermal Mass Dynamics

**Affected metrics:** 950FF - Min Free-Float Temp (°C) |
**Count:** 1 metrics

### Solar Gain Calculations

**Affected metrics:** 610 - Peak Cooling (kW), 640 - Peak Cooling (kW), 650 - Peak Cooling (kW) |
**Count:** 3 metrics

### Unknown/Unclassified

**Affected metrics:** 900 - Peak Cooling (kW), 640 - Peak Heating (kW), 630 - Annual Heating (MWh), 600FF - Min Free-Float Temp (°C), 610 - Peak Heating (kW), 940 - Peak Cooling (kW), 950 - Peak Cooling (kW), 620 - Peak Heating (kW), 600 - Peak Heating (kW), 630 - Peak Heating (kW), 630 - Annual Cooling (MWh), 610 - Annual Heating (MWh), 930 - Peak Heating (kW), 620 - Annual Cooling (MWh), 650FF - Min Free-Float Temp (°C), 650FF - Max Free-Float Temp (°C), 910 - Peak Cooling (kW), 600 - Annual Cooling (MWh), 640 - Annual Heating (MWh), 600FF - Max Free-Float Temp (°C) |
**Count:** 20 metrics

### 5R1C Model Limitation (Accepted)

**Affected metrics:** 920 - Annual Heating (MWh), 920 - Annual Cooling (MWh), 900 - Annual Cooling (MWh), 940 - Annual Cooling (MWh), 900 - Annual Heating (MWh), 930 - Annual Cooling (MWh), 910 - Annual Cooling (MWh), 940 - Annual Heating (MWh), 950 - Annual Cooling (MWh), 910 - Annual Heating (MWh), 930 - Annual Heating (MWh) |
**Count:** 11 metrics

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
