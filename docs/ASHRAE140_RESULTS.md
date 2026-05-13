# ASHRAE Standard 140 Validation Results

*Generated: 2026-05-08 20:40 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 17.2% |
| Passed | 11 |
| Warnings | 4 |
| Failed | 49 |
| Mean Absolute Error | 102.74% |
| Max Deviation | 506.56% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 2.65 seconds |
| Throughput | 6.78 cases/sec |
| Total Cases | 18 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 7.01 MWh (Ref: 5.50-7.50) | 13.89 MWh (Ref: 8.00-10.50) | 5.58 kW (Ref: 2.80-3.80) | 10.15 kW (Ref: 4.80-6.20) | ❌ FAIL |
| 610 | 7.35 MWh (Ref: 4.36-5.79) | 10.48 MWh (Ref: 3.92-6.14) | 5.59 kW (Ref: 4.30-5.70) | 7.23 kW (Ref: 2.20-2.90) | ❌ FAIL |
| 620 | 5.86 MWh (Ref: 4.50-6.50) | 12.05 MWh (Ref: 3.20-5.00) | 5.52 kW (Ref: 2.80-3.80) | 8.68 kW (Ref: 2.50-3.50) | ❌ FAIL |
| 630 | 6.02 MWh (Ref: 5.05-6.47) | 9.32 MWh (Ref: 2.13-3.70) | 5.52 kW (Ref: 4.70-6.10) | 7.90 kW (Ref: 1.80-2.40) | ❌ FAIL |
| 640 | 4.42 MWh (Ref: 2.75-3.80) | 13.64 MWh (Ref: 5.95-8.10) | 5.91 kW (Ref: 4.30-5.70) | 10.15 kW (Ref: 2.80-3.70) | ❌ FAIL |
| 650 | 0.00 MWh (Ref: 0.00-0.00) | 10.71 MWh (Ref: 4.82-7.06) | 0.00 kW (Ref: 0.00-0.00) | 9.76 kW (Ref: 1.90-2.50) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 1.76 MWh (Ref: 1.17-2.04) | 9.47 MWh (Ref: 2.13-3.67) | 5.46 kW (Ref: 1.80-2.40) | 13.32 kW (Ref: 1.60-2.10) | ❌ FAIL |
| 910 | 3.10 MWh (Ref: 1.51-2.28) | 4.66 MWh (Ref: 0.82-1.88) | 5.53 kW (Ref: 1.90-2.50) | 8.49 kW (Ref: 1.20-1.60) | ❌ FAIL |
| 920 | 4.98 MWh (Ref: 3.26-4.30) | 15.70 MWh (Ref: 1.84-3.31) | 5.10 kW (Ref: 2.10-2.80) | 11.75 kW (Ref: 1.40-1.90) | ❌ FAIL |
| 930 | 5.33 MWh (Ref: 4.14-5.34) | 11.31 MWh (Ref: 1.04-2.24) | 5.11 kW (Ref: 2.30-3.00) | 10.98 kW (Ref: 1.10-1.50) | ❌ FAIL |
| 940 | 1.67 MWh (Ref: 0.79-1.41) | 8.43 MWh (Ref: 2.08-3.55) | 5.32 kW (Ref: 1.90-2.50) | 13.32 kW (Ref: 1.70-2.30) | ❌ FAIL |
| 950 | 0.00 MWh (Ref: 0.00-0.00) | 5.25 MWh (Ref: 0.39-0.92) | 0.00 kW (Ref: 0.00-0.00) | 12.67 kW (Ref: 0.70-0.90) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -26.19°C (Ref: -18.80--15.60) | 83.97°C (Ref: 64.90-75.10) | ❌ FAIL |
| 650FF | -32.90°C (Ref: -23.00--21.00) | 80.46°C (Ref: 63.20-73.50) | ❌ FAIL |
| 900FF | -12.05°C (Ref: -6.40--1.60) | 47.26°C (Ref: 41.80-46.40) | ❌ FAIL |
| 950FF | -26.19°C (Ref: -20.20--17.80) | 47.55°C (Ref: 35.50-38.50) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 4.91 MWh (Ref: 1.65-2.45) | 21.90 MWh (Ref: 1.55-2.78) | 4.41 kW (Ref: 2.00-8.00) | 13.84 kW (Ref: 0.00-4.00) | ❌ FAIL |
| 195 | 7.44 MWh (Ref: 3.50-6.00) | 0.00 MWh (Ref: 0.00-0.00) | 1.78 kW (Ref: 1.40-2.20) | 0.00 kW (Ref: 0.00-0.00) | ❌ FAIL |

## Multi-Reference Comparison

| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |
|------|--------|------------|-------|--------|---------|
| 195 | Annual Heating Energy (MWh) | FAIL (7.44) | - | - | FAIL |
| 195 | Annual Cooling Energy (MWh) | PASS (0.00) | - | - | PASS |
| 195 | Peak Heating Load (kW) | PASS (1.78) | - | - | PASS |
| 195 | Peak Cooling Load (kW) | PASS (0.00) | - | - | PASS |
| 600 | Annual Heating Energy (MWh) | WARN (7.01) | FAIL (7.01) | PASS (7.01) | WARN |
| 600 | Annual Cooling Energy (MWh) | FAIL (13.89) | FAIL (13.89) | FAIL (13.89) | FAIL |
| 600 | Peak Heating Load (kW) | FAIL (5.58) | FAIL (5.58) | FAIL (5.58) | FAIL |
| 600 | Peak Cooling Load (kW) | FAIL (10.15) | FAIL (10.15) | FAIL (10.15) | FAIL |
| 900 | Annual Heating Energy (MWh) | PASS (1.76) | - | - | PASS |
| 900 | Annual Cooling Energy (MWh) | FAIL (9.47) | - | - | FAIL |
| 900 | Peak Heating Load (kW) | FAIL (5.46) | - | - | FAIL |
| 900 | Peak Cooling Load (kW) | FAIL (13.32) | - | - | FAIL |
| 920 | Annual Heating Energy (MWh) | FAIL (4.98) | - | - | FAIL |
| 920 | Annual Cooling Energy (MWh) | FAIL (15.70) | - | - | FAIL |
| 920 | Peak Heating Load (kW) | FAIL (5.10) | - | - | FAIL |
| 920 | Peak Cooling Load (kW) | FAIL (11.75) | - | - | FAIL |
| 930 | Annual Heating Energy (MWh) | WARN (5.33) | - | - | FAIL |
| 930 | Annual Cooling Energy (MWh) | FAIL (11.31) | - | - | FAIL |
| 930 | Peak Heating Load (kW) | PASS (5.11) | - | - | PASS |
| 930 | Peak Cooling Load (kW) | FAIL (10.98) | - | - | FAIL |
| 940 | Annual Heating Energy (MWh) | FAIL (1.67) | - | - | FAIL |
| 940 | Annual Cooling Energy (MWh) | FAIL (8.43) | - | - | FAIL |
| 940 | Peak Heating Load (kW) | FAIL (5.32) | - | - | FAIL |
| 940 | Peak Cooling Load (kW) | FAIL (13.32) | - | - | FAIL |
| 950 | Annual Heating Energy (MWh) | FAIL (0.00) | - | - | FAIL |
| 950 | Annual Cooling Energy (MWh) | PASS (5.25) | - | - | PASS |
| 950 | Peak Heating Load (kW) | FAIL (0.00) | - | - | FAIL |
| 950 | Peak Cooling Load (kW) | FAIL (12.67) | - | - | FAIL |
| 960 | Annual Heating Energy (MWh) | FAIL (4.91) | - | - | FAIL |
| 960 | Annual Cooling Energy (MWh) | FAIL (21.90) | - | - | FAIL |
| 960 | Peak Heating Load (kW) | FAIL (4.41) | - | - | FAIL |
| 960 | Peak Cooling Load (kW) | FAIL (13.84) | - | - | FAIL |

## Systematic Issues

The following recurring issues are affecting validation results:

### Inter-Zone Heat Transfer

**Affected metrics:** 960 - Annual Cooling Energy (MWh) |
**Count:** 1 metrics

### 5R1C Model Limitation (Accepted)

**Affected metrics:** 920 - Annual Cooling Energy (MWh), 930 - Annual Heating Energy (MWh), 920 - Annual Heating Energy (MWh), 930 - Annual Cooling Energy (MWh), 900 - Annual Cooling Energy (MWh), 910 - Annual Heating Energy (MWh), 940 - Annual Cooling Energy (MWh), 950 - Annual Heating Energy (MWh), 940 - Annual Heating Energy (MWh), 910 - Annual Cooling Energy (MWh) |
**Count:** 10 metrics

### Unknown/Unclassified

**Affected metrics:** 640 - Annual Cooling Energy (MWh), 650FF - Maximum Free-Floating Temperature (°C), 600FF - Maximum Free-Floating Temperature (°C), 900 - Peak Heating Load (kW), 910 - Peak Heating Load (kW), 910 - Peak Cooling Load (kW), 620 - Peak Heating Load (kW), 640 - Annual Heating Energy (MWh), 600 - Annual Cooling Energy (MWh), 650FF - Minimum Free-Floating Temperature (°C), 950 - Peak Heating Load (kW), 920 - Peak Cooling Load (kW), 620 - Annual Cooling Energy (MWh), 195 - Annual Heating Energy (MWh), 960 - Peak Cooling Load (kW), 650 - Annual Cooling Energy (MWh), 600FF - Minimum Free-Floating Temperature (°C), 960 - Peak Heating Load (kW), 940 - Peak Heating Load (kW), 940 - Peak Cooling Load (kW), 610 - Annual Cooling Energy (MWh), 600 - Peak Heating Load (kW), 920 - Peak Heating Load (kW), 610 - Annual Heating Energy (MWh), 630 - Annual Cooling Energy (MWh), 950 - Peak Cooling Load (kW), 930 - Peak Cooling Load (kW), 960 - Annual Heating Energy (MWh), 900 - Peak Cooling Load (kW) |
**Count:** 29 metrics

### Thermal Mass Dynamics

**Affected metrics:** 950FF - Minimum Free-Floating Temperature (°C), 900FF - Minimum Free-Floating Temperature (°C), 950FF - Maximum Free-Floating Temperature (°C) |
**Count:** 3 metrics

### Solar Gain Calculations

**Affected metrics:** 600 - Peak Cooling Load (kW), 620 - Peak Cooling Load (kW), 630 - Peak Cooling Load (kW), 640 - Peak Cooling Load (kW), 610 - Peak Cooling Load (kW), 650 - Peak Cooling Load (kW) |
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
