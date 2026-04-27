# ASHRAE Standard 140 Validation Results

*Generated: 2026-04-27 22:58 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 26.6% |
| Passed | 17 |
| Warnings | 4 |
| Failed | 43 |
| Mean Absolute Error | 38.91% |
| Max Deviation | 142.22% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 3.92 seconds |
| Throughput | 4.60 cases/sec |
| Total Cases | 18 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 6.41 MWh (Ref: 5.50-7.50) | 10.90 MWh (Ref: 8.00-10.50) | 2.35 kW (Ref: 2.80-3.80) | 2.94 kW (Ref: 4.80-6.20) | ❌ FAIL |
| 610 | 5.58 MWh (Ref: 4.36-5.79) | 5.58 MWh (Ref: 3.92-6.14) | 5.34 kW (Ref: 4.30-5.70) | 4.16 kW (Ref: 2.20-2.90) | ❌ FAIL |
| 620 | 5.02 MWh (Ref: 4.50-6.50) | 7.86 MWh (Ref: 3.20-5.00) | 0.48 kW (Ref: 2.80-3.80) | 0.48 kW (Ref: 2.50-3.50) | ❌ FAIL |
| 630 | 4.72 MWh (Ref: 5.05-6.47) | 5.17 MWh (Ref: 2.13-3.70) | 5.33 kW (Ref: 4.70-6.10) | 4.82 kW (Ref: 1.80-2.40) | ❌ FAIL |
| 640 | 3.59 MWh (Ref: 2.75-3.80) | 8.40 MWh (Ref: 5.95-8.10) | 5.33 kW (Ref: 4.30-5.70) | 5.45 kW (Ref: 2.80-3.70) | ❌ FAIL |
| 650 | 0.00 MWh (Ref: 0.00-0.00) | 7.36 MWh (Ref: 4.82-7.06) | 0.00 kW (Ref: 0.00-0.00) | 0.49 kW (Ref: 1.90-2.50) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 1.13 MWh (Ref: 1.17-2.04) | 3.41 MWh (Ref: 2.13-3.67) | 1.31 kW (Ref: 1.80-2.40) | 2.14 kW (Ref: 1.60-2.10) | ❌ FAIL |
| 910 | 1.59 MWh (Ref: 1.51-2.28) | 1.22 MWh (Ref: 0.82-1.88) | 1.32 kW (Ref: 1.90-2.50) | 1.37 kW (Ref: 1.20-1.60) | ❌ FAIL |
| 920 | 2.95 MWh (Ref: 3.26-4.30) | 5.08 MWh (Ref: 1.84-3.31) | 2.61 kW (Ref: 2.10-2.80) | 4.07 kW (Ref: 1.40-1.90) | ❌ FAIL |
| 930 | 4.05 MWh (Ref: 4.14-5.34) | 3.76 MWh (Ref: 1.04-2.24) | 2.62 kW (Ref: 2.30-3.00) | 3.74 kW (Ref: 1.10-1.50) | ❌ FAIL |
| 940 | 0.93 MWh (Ref: 0.79-1.41) | 3.39 MWh (Ref: 2.08-3.55) | 1.25 kW (Ref: 1.90-2.50) | 2.18 kW (Ref: 1.70-2.30) | ❌ FAIL |
| 950 | 0.00 MWh (Ref: 0.00-0.00) | 0.84 MWh (Ref: 0.39-0.92) | 0.00 kW (Ref: 0.00-0.00) | 2.21 kW (Ref: 0.70-0.90) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -11.94°C (Ref: -18.80--15.60) | 69.64°C (Ref: 64.90-75.10) | ❌ FAIL |
| 650FF | -12.28°C (Ref: -23.00--21.00) | 69.64°C (Ref: 63.20-73.50) | ❌ FAIL |
| 900FF | -9.69°C (Ref: -6.40--1.60) | 53.75°C (Ref: 41.80-46.40) | ❌ FAIL |
| 950FF | -11.57°C (Ref: -20.20--17.80) | 54.20°C (Ref: 35.50-38.50) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 1.11 MWh (Ref: 1.65-2.45) | 6.90 MWh (Ref: 1.55-2.78) | 3.81 kW (Ref: 2.00-8.00) | 3.06 kW (Ref: 0.00-4.00) | ❌ FAIL |
| 195 | 8.71 MWh (Ref: 3.50-6.00) | 0.00 MWh (Ref: 0.00-0.00) | 1.95 kW (Ref: 1.40-2.20) | 0.00 kW (Ref: 0.00-0.00) | ❌ FAIL |

## Multi-Reference Comparison

| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |
|------|--------|------------|-------|--------|---------|
| 195 | Annual Heating Energy (MWh) | FAIL (8.71) | - | - | FAIL |
| 195 | Annual Cooling Energy (MWh) | PASS (0.00) | - | - | PASS |
| 195 | Peak Heating Load (kW) | PASS (1.95) | - | - | PASS |
| 195 | Peak Cooling Load (kW) | PASS (0.00) | - | - | PASS |
| 600 | Annual Heating Energy (MWh) | PASS (6.41) | FAIL (6.41) | PASS (6.41) | PASS |
| 600 | Annual Cooling Energy (MWh) | FAIL (10.90) | FAIL (10.90) | FAIL (10.90) | FAIL |
| 600 | Peak Heating Load (kW) | FAIL (2.35) | FAIL (2.35) | FAIL (2.35) | FAIL |
| 600 | Peak Cooling Load (kW) | FAIL (2.94) | FAIL (2.94) | FAIL (2.94) | FAIL |
| 900 | Annual Heating Energy (MWh) | WARN (1.13) | - | - | FAIL |
| 900 | Annual Cooling Energy (MWh) | WARN (3.41) | - | - | FAIL |
| 900 | Peak Heating Load (kW) | WARN (1.31) | - | - | FAIL |
| 900 | Peak Cooling Load (kW) | WARN (2.14) | - | - | FAIL |
| 920 | Annual Heating Energy (MWh) | FAIL (2.95) | - | - | FAIL |
| 920 | Annual Cooling Energy (MWh) | FAIL (5.08) | - | - | FAIL |
| 920 | Peak Heating Load (kW) | FAIL (2.61) | - | - | FAIL |
| 920 | Peak Cooling Load (kW) | PASS (4.07) | - | - | PASS |
| 930 | Annual Heating Energy (MWh) | WARN (4.05) | - | - | FAIL |
| 930 | Annual Cooling Energy (MWh) | FAIL (3.76) | - | - | FAIL |
| 930 | Peak Heating Load (kW) | FAIL (2.62) | - | - | FAIL |
| 930 | Peak Cooling Load (kW) | FAIL (3.74) | - | - | FAIL |
| 940 | Annual Heating Energy (MWh) | FAIL (0.93) | - | - | FAIL |
| 940 | Annual Cooling Energy (MWh) | FAIL (3.39) | - | - | FAIL |
| 940 | Peak Heating Load (kW) | FAIL (1.25) | - | - | FAIL |
| 940 | Peak Cooling Load (kW) | FAIL (2.18) | - | - | FAIL |
| 950 | Annual Heating Energy (MWh) | FAIL (0.00) | - | - | FAIL |
| 950 | Annual Cooling Energy (MWh) | FAIL (0.84) | - | - | FAIL |
| 950 | Peak Heating Load (kW) | FAIL (0.00) | - | - | FAIL |
| 950 | Peak Cooling Load (kW) | FAIL (2.21) | - | - | FAIL |
| 960 | Annual Heating Energy (MWh) | FAIL (1.11) | - | - | FAIL |
| 960 | Annual Cooling Energy (MWh) | WARN (6.90) | - | - | FAIL |
| 960 | Peak Heating Load (kW) | FAIL (3.81) | - | - | FAIL |
| 960 | Peak Cooling Load (kW) | FAIL (3.06) | - | - | FAIL |

## Systematic Issues

The following recurring issues are affecting validation results:

### Unknown/Unclassified

**Affected metrics:** 620 - Annual Cooling Energy (MWh), 600 - Annual Cooling Energy (MWh), 630 - Annual Cooling Energy (MWh), 930 - Peak Heating Load (kW), 195 - Annual Heating Energy (MWh), 600 - Peak Heating Load (kW), 960 - Peak Cooling Load (kW), 940 - Peak Cooling Load (kW), 920 - Peak Heating Load (kW), 650FF - Minimum Free-Floating Temperature (°C), 900 - Peak Cooling Load (kW), 950 - Peak Cooling Load (kW), 930 - Peak Cooling Load (kW), 900 - Peak Heating Load (kW), 950 - Peak Heating Load (kW), 960 - Peak Heating Load (kW), 940 - Peak Heating Load (kW), 620 - Peak Heating Load (kW), 910 - Peak Heating Load (kW), 630 - Annual Heating Energy (MWh), 960 - Annual Heating Energy (MWh), 600FF - Minimum Free-Floating Temperature (°C) |
**Count:** 22 metrics

### 5R1C Model Limitation (Accepted)

**Affected metrics:** 950 - Annual Cooling Energy (MWh), 930 - Annual Cooling Energy (MWh), 920 - Annual Cooling Energy (MWh), 920 - Annual Heating Energy (MWh), 930 - Annual Heating Energy (MWh), 900 - Annual Heating Energy (MWh), 950 - Annual Heating Energy (MWh), 940 - Annual Cooling Energy (MWh), 940 - Annual Heating Energy (MWh), 900 - Annual Cooling Energy (MWh) |
**Count:** 10 metrics

### Inter-Zone Heat Transfer

**Affected metrics:** 960 - Annual Cooling Energy (MWh) |
**Count:** 1 metrics

### Thermal Mass Dynamics

**Affected metrics:** 900FF - Minimum Free-Floating Temperature (°C), 950FF - Maximum Free-Floating Temperature (°C), 900FF - Maximum Free-Floating Temperature (°C), 950FF - Minimum Free-Floating Temperature (°C) |
**Count:** 4 metrics

### Solar Gain Calculations

**Affected metrics:** 620 - Peak Cooling Load (kW), 650 - Peak Cooling Load (kW), 640 - Peak Cooling Load (kW), 610 - Peak Cooling Load (kW), 630 - Peak Cooling Load (kW), 600 - Peak Cooling Load (kW) |
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
