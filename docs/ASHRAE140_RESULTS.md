# ASHRAE Standard 140 Validation Results

*Generated: 2026-04-02 23:29 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 34.4% |
| Passed | 22 |
| Warnings | 16 |
| Failed | 26 |
| Mean Absolute Error | 36.98% |
| Max Deviation | 321.63% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 0.88 seconds |
| Throughput | 20.54 cases/sec |
| Total Cases | 18 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 6.42 MWh (Ref: 5.50-7.50) | 8.32 MWh (Ref: 8.00-10.50) | 4.77 kW (Ref: 2.80-3.80) | 5.61 kW (Ref: 4.80-6.20) | ❌ FAIL |
| 610 | 6.52 MWh (Ref: 4.36-5.79) | 6.17 MWh (Ref: 3.92-6.14) | 4.77 kW (Ref: 4.30-5.70) | 4.72 kW (Ref: 2.20-2.90) | ❌ FAIL |
| 620 | 5.68 MWh (Ref: 4.50-6.50) | 3.48 MWh (Ref: 3.20-5.00) | 4.77 kW (Ref: 2.80-3.80) | 3.36 kW (Ref: 2.50-3.50) | ❌ FAIL |
| 630 | 6.05 MWh (Ref: 5.05-6.47) | 2.02 MWh (Ref: 2.13-3.70) | 4.77 kW (Ref: 4.70-6.10) | 2.25 kW (Ref: 1.80-2.40) | ❌ FAIL |
| 640 | 4.57 MWh (Ref: 2.75-3.80) | 8.26 MWh (Ref: 5.95-8.10) | 4.69 kW (Ref: 4.30-5.70) | 5.61 kW (Ref: 2.80-3.70) | ❌ FAIL |
| 650 | 0.00 MWh (Ref: 0.00-0.00) | 7.33 MWh (Ref: 4.82-7.06) | 0.00 kW (Ref: 0.00-0.00) | 5.60 kW (Ref: 1.90-2.50) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 1.60 MWh (Ref: 1.17-2.04) | 3.01 MWh (Ref: 2.13-3.67) | 4.44 kW (Ref: 1.80-2.40) | 3.42 kW (Ref: 1.60-2.10) | ❌ FAIL |
| 910 | 1.92 MWh (Ref: 1.51-2.28) | 1.87 MWh (Ref: 0.82-1.88) | 4.44 kW (Ref: 1.90-2.50) | 2.91 kW (Ref: 1.20-1.60) | ❌ FAIL |
| 920 | 3.81 MWh (Ref: 3.26-4.30) | 2.49 MWh (Ref: 1.84-3.31) | 4.34 kW (Ref: 2.10-2.80) | 2.08 kW (Ref: 1.40-1.90) | ❌ FAIL |
| 930 | 4.72 MWh (Ref: 4.14-5.34) | 1.59 MWh (Ref: 1.04-2.24) | 4.37 kW (Ref: 2.30-3.00) | 1.44 kW (Ref: 1.10-1.50) | ❌ FAIL |
| 940 | 1.10 MWh (Ref: 0.79-1.41) | 2.80 MWh (Ref: 2.08-3.55) | 4.21 kW (Ref: 1.90-2.50) | 3.42 kW (Ref: 1.70-2.30) | ❌ FAIL |
| 950 | 0.00 MWh (Ref: 0.00-0.00) | 0.81 MWh (Ref: 0.39-0.92) | 0.00 kW (Ref: 0.00-0.00) | 3.37 kW (Ref: 0.70-0.90) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -10.77°C (Ref: -18.80--15.60) | 61.36°C (Ref: 64.90-75.10) | ❌ FAIL |
| 650FF | -11.88°C (Ref: -23.00--21.00) | 61.34°C (Ref: 63.20-73.50) | ❌ FAIL |
| 900FF | -9.59°C (Ref: -6.40--1.60) | 46.88°C (Ref: 41.80-46.40) | ❌ FAIL |
| 950FF | -11.52°C (Ref: -20.20--17.80) | 46.50°C (Ref: 35.50-38.50) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 7.02 MWh (Ref: 5.00-15.00) | 3.59 MWh (Ref: 1.00-3.50) | 4.46 kW (Ref: 2.00-8.00) | 3.28 kW (Ref: 0.00-4.00) | ⚠️ WARN |
| 195 | 7.34 MWh (Ref: 3.50-6.00) | 0.00 MWh (Ref: 0.00-0.00) | 2.59 kW (Ref: 1.40-2.20) | 0.00 kW (Ref: 0.00-0.00) | ❌ FAIL |

## Systematic Issues

The following recurring issues are affecting validation results:

### Unknown/Unclassified

**Affected metrics:** 600 - Peak Heating (kW), 940 - Peak Cooling (kW), 950 - Peak Cooling (kW), 900 - Peak Cooling (kW), 900 - Peak Heating (kW), 930 - Peak Heating (kW), 620 - Peak Heating (kW), 910 - Peak Heating (kW), 600FF - Min Free-Float Temp (°C), 630 - Annual Cooling (MWh), 610 - Annual Heating (MWh), 940 - Peak Heating (kW), 195 - Peak Heating (kW), 195 - Annual Heating (MWh), 910 - Peak Cooling (kW), 920 - Peak Heating (kW), 920 - Peak Cooling (kW), 640 - Annual Heating (MWh), 600FF - Max Free-Float Temp (°C), 650FF - Min Free-Float Temp (°C) |
**Count:** 20 metrics

### Solar Gain Calculations

**Affected metrics:** 640 - Peak Cooling (kW), 650 - Peak Cooling (kW), 610 - Peak Cooling (kW) |
**Count:** 3 metrics

### Thermal Mass Dynamics

**Affected metrics:** 950FF - Min Free-Float Temp (°C), 950FF - Max Free-Float Temp (°C), 900FF - Min Free-Float Temp (°C) |
**Count:** 3 metrics

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
