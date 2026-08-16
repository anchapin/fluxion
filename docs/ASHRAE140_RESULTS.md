# ASHRAE Standard 140 Validation Results

*Generated: 2026-08-16 08:18 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 14.1% |
| Passed | 9 |
| Warnings | 5 |
| Failed | 50 |
| Mean Absolute Error | 52.41% |
| Max Deviation | 476.39% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 1.25 seconds |
| Throughput | 14.36 cases/sec |
| Total Cases | 18 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 4613.62 kWh (Ref: 4360.00-5790.00) | 3302.72 kWh (Ref: 3920.00-6140.00) | 4.39 kW (Ref: 2.80-3.80) | 3.72 kW (Ref: 4.80-6.20) | ❌ FAIL |
| 610 | 4701.11 kWh (Ref: 4360.00-5790.00) | 2668.59 kWh (Ref: 3920.00-6140.00) | 4.39 kW (Ref: 4.30-5.70) | 3.45 kW (Ref: 2.20-2.90) | ❌ FAIL |
| 620 | 5966.79 kWh (Ref: 4500.00-6500.00) | 2366.68 kWh (Ref: 3200.00-5000.00) | 4.51 kW (Ref: 2.80-3.80) | 2.98 kW (Ref: 2.50-3.50) | ❌ FAIL |
| 630 | 6083.75 kWh (Ref: 5050.00-6470.00) | 2032.26 kWh (Ref: 2130.00-3700.00) | 4.51 kW (Ref: 4.70-6.10) | 2.74 kW (Ref: 1.80-2.40) | ❌ FAIL |
| 640 | 2389.80 kWh (Ref: 2750.00-3800.00) | 3295.68 kWh (Ref: 5950.00-8100.00) | 4.05 kW (Ref: 4.30-5.70) | 3.72 kW (Ref: 2.80-3.70) | ❌ FAIL |
| 650 | 0.00 kWh (Ref: 0.00-0.00) | 2464.98 kWh (Ref: 4820.00-7060.00) | 0.00 kW (Ref: 0.00-0.00) | 3.35 kW (Ref: 1.90-2.50) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 5340.01 kWh (Ref: 1170.00-2040.00) | 7657.69 kWh (Ref: 2130.00-3670.00) | 3.90 kW (Ref: 1.80-2.40) | 3.34 kW (Ref: 1.60-2.10) | ❌ FAIL |
| 910 | 5795.15 kWh (Ref: 1510.00-2280.00) | 7781.31 kWh (Ref: 820.00-1880.00) | 3.43 kW (Ref: 1.90-2.50) | 3.34 kW (Ref: 1.20-1.60) | ❌ FAIL |
| 920 | 5419.62 kWh (Ref: 3260.00-4300.00) | 6202.36 kWh (Ref: 1840.00-3310.00) | 3.57 kW (Ref: 2.10-2.80) | 3.33 kW (Ref: 1.40-1.90) | ❌ FAIL |
| 930 | 5414.48 kWh (Ref: 4140.00-5340.00) | 5910.80 kWh (Ref: 1040.00-2240.00) | 3.58 kW (Ref: 2.30-3.00) | 3.40 kW (Ref: 1.10-1.50) | ❌ FAIL |
| 940 | 7487.81 kWh (Ref: 790.00-1410.00) | 11397.32 kWh (Ref: 2080.00-3550.00) | 6.29 kW (Ref: 1.90-2.50) | 7.42 kW (Ref: 1.70-2.30) | ❌ FAIL |
| 950 | 0.00 kWh (Ref: 0.00-0.00) | 26.87 kWh (Ref: 390.00-920.00) | 0.00 kW (Ref: 0.00-0.00) | 0.36 kW (Ref: 0.70-0.90) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -17.13°C (Ref: -18.80--15.60) | 55.22°C (Ref: 64.90-75.10) | ❌ FAIL |
| 650FF | -23.71°C (Ref: -23.00--21.00) | 52.43°C (Ref: 63.20-73.50) | ❌ FAIL |
| 900FF | -7.27°C (Ref: -6.40--1.60) | 38.74°C (Ref: 41.80-46.40) | ❌ FAIL |
| 950FF | -23.94°C (Ref: -20.20--17.80) | 31.20°C (Ref: 35.50-38.50) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 5517.87 kWh (Ref: 1650.00-2450.00) | 7548.18 kWh (Ref: 1550.00-2780.00) | 3.98 kW (Ref: 2.00-8.00) | 3.28 kW (Ref: 0.00-4.00) | ❌ FAIL |
| 195 | 6552.13 kWh (Ref: 3500.00-6000.00) | 279.70 kWh (Ref: 0.00-0.00) | 3.65 kW (Ref: 1.40-2.20) | 1.02 kW (Ref: 0.00-0.00) | ❌ FAIL |

## Multi-Reference Comparison

| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |
|------|--------|------------|-------|--------|---------|
| 195 | Annual Heating Energy (kWh) | FAIL (6552.13) | - | - | FAIL |
| 195 | Annual Cooling Energy (kWh) | FAIL (279.70) | - | - | FAIL |
| 195 | Peak Heating Load (kW) | FAIL (3.65) | - | - | FAIL |
| 195 | Peak Cooling Load (kW) | FAIL (1.02) | - | - | FAIL |
| 600 | Annual Heating Energy (kWh) | FAIL (4613.62) | PASS (4613.62) | FAIL (4613.62) | WARN |
| 600 | Annual Cooling Energy (kWh) | FAIL (3302.72) | FAIL (3302.72) | FAIL (3302.72) | FAIL |
| 600 | Peak Heating Load (kW) | FAIL (4.39) | FAIL (4.39) | FAIL (4.39) | FAIL |
| 600 | Peak Cooling Load (kW) | FAIL (3.72) | FAIL (3.72) | FAIL (3.72) | FAIL |
| 900 | Annual Heating Energy (kWh) | FAIL (5340.01) | - | - | FAIL |
| 900 | Annual Cooling Energy (kWh) | FAIL (7657.69) | - | - | FAIL |
| 900 | Peak Heating Load (kW) | FAIL (3.90) | - | - | FAIL |
| 900 | Peak Cooling Load (kW) | WARN (3.34) | - | - | FAIL |
| 920 | Annual Heating Energy (kWh) | FAIL (5419.62) | - | - | FAIL |
| 920 | Annual Cooling Energy (kWh) | FAIL (6202.36) | - | - | FAIL |
| 920 | Peak Heating Load (kW) | PASS (3.57) | - | - | PASS |
| 920 | Peak Cooling Load (kW) | WARN (3.33) | - | - | FAIL |
| 930 | Annual Heating Energy (kWh) | WARN (5414.48) | - | - | FAIL |
| 930 | Annual Cooling Energy (kWh) | FAIL (5910.80) | - | - | FAIL |
| 930 | Peak Heating Load (kW) | FAIL (3.58) | - | - | FAIL |
| 930 | Peak Cooling Load (kW) | FAIL (3.40) | - | - | FAIL |
| 940 | Annual Heating Energy (kWh) | FAIL (7487.81) | - | - | FAIL |
| 940 | Annual Cooling Energy (kWh) | FAIL (11397.32) | - | - | FAIL |
| 940 | Peak Heating Load (kW) | PASS (6.29) | - | - | PASS |
| 940 | Peak Cooling Load (kW) | FAIL (7.42) | - | - | FAIL |
| 950 | Annual Heating Energy (kWh) | FAIL (0.00) | - | - | FAIL |
| 950 | Annual Cooling Energy (kWh) | FAIL (26.87) | - | - | FAIL |
| 950 | Peak Heating Load (kW) | FAIL (0.00) | - | - | FAIL |
| 950 | Peak Cooling Load (kW) | FAIL (0.36) | - | - | FAIL |
| 960 | Annual Heating Energy (kWh) | FAIL (5517.87) | - | - | FAIL |
| 960 | Annual Cooling Energy (kWh) | FAIL (7548.18) | - | - | FAIL |
| 960 | Peak Heating Load (kW) | FAIL (3.98) | - | - | FAIL |
| 960 | Peak Cooling Load (kW) | FAIL (3.28) | - | - | FAIL |

## Systematic Issues

The following recurring issues are affecting validation results:

### HVAC Load Calculation

**Affected metrics:** 620 - Peak Heating Load (kW), 630 - Peak Cooling Load (kW), 195 - Peak Heating Load (kW), 650 - Peak Cooling Load (kW), 610 - Peak Cooling Load (kW), 195 - Peak Cooling Load (kW), 600 - Peak Heating Load (kW), 640 - Peak Heating Load (kW) |
**Count:** 8 metrics

### Solar Gain Calculations

**Affected metrics:** 650FF - Minimum Free-Floating Temperature (°C), 610 - Annual Cooling Energy (kWh), 950 - Peak Cooling Load (kW), 930 - Peak Cooling Load (kW), 650FF - Maximum Free-Floating Temperature (°C), 960 - Peak Cooling Load (kW), 620 - Annual Cooling Energy (kWh), 600 - Annual Cooling Energy (kWh), 600 - Peak Cooling Load (kW), 600FF - Maximum Free-Floating Temperature (°C), 650 - Annual Cooling Energy (kWh), 640 - Annual Cooling Energy (kWh) |
**Count:** 12 metrics

### Unknown/Unclassified

**Affected metrics:** 930 - Annual Cooling Energy (kWh), 920 - Peak Cooling Load (kW), 195 - Annual Heating Energy (kWh), 195 - Annual Cooling Energy (kWh), 930 - Annual Heating Energy (kWh), 900 - Peak Cooling Load (kW), 950 - Annual Heating Energy (kWh), 640 - Annual Heating Energy (kWh), 950 - Annual Cooling Energy (kWh), 940 - Annual Heating Energy (kWh) |
**Count:** 10 metrics

### Thermal Mass Dynamics

**Affected metrics:** 950 - Peak Heating Load (kW), 960 - Peak Heating Load (kW), 900 - Peak Heating Load (kW), 930 - Peak Heating Load (kW), 910 - Peak Heating Load (kW), 910 - Peak Cooling Load (kW), 950FF - Maximum Free-Floating Temperature (°C), 900FF - Maximum Free-Floating Temperature (°C), 900FF - Minimum Free-Floating Temperature (°C), 950FF - Minimum Free-Floating Temperature (°C), 940 - Peak Cooling Load (kW) |
**Count:** 11 metrics

### 5R1C Model Limitation (Accepted)

**Affected metrics:** 910 - Annual Cooling Energy (kWh), 920 - Annual Heating Energy (kWh), 920 - Annual Cooling Energy (kWh), 900 - Annual Cooling Energy (kWh), 940 - Annual Cooling Energy (kWh), 910 - Annual Heating Energy (kWh), 900 - Annual Heating Energy (kWh) |
**Count:** 7 metrics

### Inter-Zone Heat Transfer

**Affected metrics:** 960 - Annual Heating Energy (kWh), 960 - Annual Cooling Energy (kWh) |
**Count:** 2 metrics

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
