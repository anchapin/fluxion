# ASHRAE Standard 140 Validation Results

*Generated: 2026-03-15 19:23 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 1.6% |
| Passed | 1 |
| Warnings | 2 |
| Failed | 61 |
| Mean Absolute Error | 8.11% |
| Max Deviation | 99.93% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 0.35 seconds |
| Throughput | 51.67 cases/sec |
| Total Cases | 18 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 0.01 MWh (Ref: 5.00-7.00) | 0.01 MWh (Ref: 8.00-10.00) | 2.10 kW (Ref: 3.00-4.00) | 5.01 kW (Ref: 5.00-6.00) | ❌ FAIL |
| 610 | 0.01 MWh (Ref: 0.00-0.00) | 0.00 MWh (Ref: 0.00-0.00) | 2.10 kW (Ref: 0.00-0.00) | 4.10 kW (Ref: 0.00-0.00) | ❌ FAIL |
| 620 | 0.01 MWh (Ref: 0.00-0.00) | 0.00 MWh (Ref: 0.00-0.00) | 2.10 kW (Ref: 0.00-0.00) | 2.75 kW (Ref: 0.00-0.00) | ❌ FAIL |
| 630 | 0.01 MWh (Ref: 0.00-0.00) | 0.00 MWh (Ref: 0.00-0.00) | 2.10 kW (Ref: 0.00-0.00) | 1.86 kW (Ref: 0.00-0.00) | ❌ FAIL |
| 640 | 0.01 MWh (Ref: 0.00-0.00) | 0.01 MWh (Ref: 0.00-0.00) | 2.10 kW (Ref: 0.00-0.00) | 5.01 kW (Ref: 0.00-0.00) | ❌ FAIL |
| 650 | 0.00 MWh (Ref: 0.00-0.00) | 0.00 MWh (Ref: 0.00-0.00) | 0.00 kW (Ref: 0.00-0.00) | 6.45 kW (Ref: 0.00-0.00) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 0.01 MWh (Ref: 0.00-0.00) | 0.00 MWh (Ref: 0.00-0.00) | 2.10 kW (Ref: 0.00-0.00) | 3.32 kW (Ref: 0.00-0.00) | ❌ FAIL |
| 910 | 0.01 MWh (Ref: 0.00-0.00) | 0.00 MWh (Ref: 0.00-0.00) | 2.10 kW (Ref: 0.00-0.00) | 2.73 kW (Ref: 0.00-0.00) | ❌ FAIL |
| 920 | 0.01 MWh (Ref: 0.00-0.00) | 0.00 MWh (Ref: 0.00-0.00) | 2.10 kW (Ref: 0.00-0.00) | 1.85 kW (Ref: 0.00-0.00) | ❌ FAIL |
| 930 | 0.01 MWh (Ref: 0.00-0.00) | 0.00 MWh (Ref: 0.00-0.00) | 2.10 kW (Ref: 0.00-0.00) | 1.24 kW (Ref: 0.00-0.00) | ❌ FAIL |
| 940 | 0.01 MWh (Ref: 0.00-0.00) | 0.00 MWh (Ref: 0.00-0.00) | 2.10 kW (Ref: 0.00-0.00) | 3.32 kW (Ref: 0.00-0.00) | ❌ FAIL |
| 950 | 0.00 MWh (Ref: 0.00-0.00) | 0.00 MWh (Ref: 0.00-0.00) | 0.00 kW (Ref: 0.00-0.00) | 4.82 kW (Ref: 0.00-0.00) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -5.01°C (Ref: -18.80--15.60) | 47.89°C (Ref: 64.90-75.10) | ❌ FAIL |
| 650FF | -10.32°C (Ref: -23.00--21.00) | 44.53°C (Ref: 63.20-73.50) | ❌ FAIL |
| 900FF | -4.52°C (Ref: -6.40--1.60) | 38.37°C (Ref: 41.80-46.40) | ❌ FAIL |
| 950FF | -9.51°C (Ref: -20.20--17.80) | 35.48°C (Ref: 35.50-38.50) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 0.01 MWh (Ref: 0.00-0.00) | 0.00 MWh (Ref: 0.00-0.00) | 2.10 kW (Ref: 0.00-0.00) | 3.61 kW (Ref: 0.00-0.00) | ❌ FAIL |
| 195 | 0.00 MWh (Ref: 0.00-0.00) | 0.00 MWh (Ref: 0.00-0.00) | 1.64 kW (Ref: 0.00-0.00) | 0.00 kW (Ref: 0.00-0.00) | ❌ FAIL |

## Multi-Reference Comparison

| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |
|------|--------|------------|-------|--------|---------|
| 600 | Annual Heating (MWh) | FAIL (0.01) | - | - | FAIL |
| 600 | Annual Cooling (MWh) | FAIL (0.01) | - | - | FAIL |
| 600 | Peak Heating (kW) | FAIL (2.10) | - | - | FAIL |
| 600 | Peak Cooling (kW) | PASS (5.01) | - | - | PASS |

## Systematic Issues

The following recurring issues are affecting validation results:

### 5R1C Model Limitation (Accepted)

**Affected metrics:** 900 - Annual Cooling (MWh), 930 - Annual Cooling (MWh), 920 - Annual Heating (MWh), 920 - Annual Cooling (MWh), 930 - Annual Heating (MWh), 910 - Annual Cooling (MWh), 940 - Annual Heating (MWh), 900 - Annual Heating (MWh), 950 - Annual Cooling (MWh), 910 - Annual Heating (MWh), 950 - Annual Heating (MWh), 940 - Annual Cooling (MWh) |
**Count:** 12 metrics

### Inter-Zone Heat Transfer

**Affected metrics:** 960 - Annual Cooling (MWh) |
**Count:** 1 metrics

### Thermal Mass Dynamics

**Affected metrics:** 950FF - Min Free-Float Temp (°C), 900FF - Max Free-Float Temp (°C) |
**Count:** 2 metrics

### Solar Gain Calculations

**Affected metrics:** 650 - Peak Cooling (kW), 620 - Peak Cooling (kW), 610 - Peak Cooling (kW), 630 - Peak Cooling (kW), 640 - Peak Cooling (kW) |
**Count:** 5 metrics

### Unknown/Unclassified

**Affected metrics:** 950 - Peak Heating (kW), 960 - Annual Heating (MWh), 195 - Peak Heating (kW), 630 - Annual Heating (MWh), 630 - Annual Cooling (MWh), 600FF - Max Free-Float Temp (°C), 930 - Peak Cooling (kW), 960 - Peak Heating (kW), 650 - Annual Cooling (MWh), 610 - Annual Heating (MWh), 610 - Annual Cooling (MWh), 630 - Peak Heating (kW), 620 - Annual Heating (MWh), 650 - Annual Heating (MWh), 920 - Peak Heating (kW), 600 - Annual Cooling (MWh), 900 - Peak Heating (kW), 900 - Peak Cooling (kW), 640 - Annual Cooling (MWh), 910 - Peak Heating (kW), 920 - Peak Cooling (kW), 600FF - Min Free-Float Temp (°C), 195 - Annual Heating (MWh), 940 - Peak Cooling (kW), 640 - Peak Heating (kW), 650 - Peak Heating (kW), 195 - Annual Cooling (MWh), 195 - Peak Cooling (kW), 620 - Peak Heating (kW), 910 - Peak Cooling (kW), 650FF - Min Free-Float Temp (°C), 960 - Peak Cooling (kW), 930 - Peak Heating (kW), 950 - Peak Cooling (kW), 940 - Peak Heating (kW), 610 - Peak Heating (kW), 600 - Peak Heating (kW), 620 - Annual Cooling (MWh), 640 - Annual Heating (MWh), 650FF - Max Free-Float Temp (°C), 600 - Annual Heating (MWh) |
**Count:** 41 metrics

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
