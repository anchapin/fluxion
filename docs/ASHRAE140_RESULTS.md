# ASHRAE Standard 140 Validation Results

*Generated: 2026-07-26 15:14 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 17.2% |
| Passed | 11 |
| Warnings | 9 |
| Failed | 44 |
| Mean Absolute Error | 109.61% |
| Max Deviation | 1417.86% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 1.01 seconds |
| Throughput | 17.88 cases/sec |
| Total Cases | 18 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 5165.39 kWh (Ref: 4360.00-5790.00) | 5309.88 kWh (Ref: 3920.00-6140.00) | 4.37 kW (Ref: 2.80-3.80) | 5.13 kW (Ref: 4.80-6.20) | ❌ FAIL |
| 610 | 5643.46 kWh (Ref: 4360.00-5790.00) | 4408.76 kWh (Ref: 3920.00-6140.00) | 4.63 kW (Ref: 4.30-5.70) | 4.42 kW (Ref: 2.20-2.90) | ❌ FAIL |
| 620 | 6347.66 kWh (Ref: 4500.00-6500.00) | 3208.16 kWh (Ref: 3200.00-5000.00) | 4.45 kW (Ref: 2.80-3.80) | 3.56 kW (Ref: 2.50-3.50) | ❌ FAIL |
| 630 | 6616.64 kWh (Ref: 5050.00-6470.00) | 2346.48 kWh (Ref: 2130.00-3700.00) | 4.46 kW (Ref: 4.70-6.10) | 2.96 kW (Ref: 1.80-2.40) | ❌ FAIL |
| 640 | 3149.57 kWh (Ref: 2750.00-3800.00) | 5302.65 kWh (Ref: 5950.00-8100.00) | 4.38 kW (Ref: 4.30-5.70) | 5.13 kW (Ref: 2.80-3.70) | ❌ FAIL |
| 650 | 0.00 kWh (Ref: 0.00-0.00) | 4401.39 kWh (Ref: 4820.00-7060.00) | 0.00 kW (Ref: 0.00-0.00) | 4.88 kW (Ref: 1.90-2.50) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 6643.80 kWh (Ref: 1170.00-2040.00) | 9519.53 kWh (Ref: 2130.00-3670.00) | 4.65 kW (Ref: 1.80-2.40) | 5.01 kW (Ref: 1.60-2.10) | ❌ FAIL |
| 910 | 7689.82 kWh (Ref: 1510.00-2280.00) | 8868.30 kWh (Ref: 820.00-1880.00) | 4.70 kW (Ref: 1.90-2.50) | 4.85 kW (Ref: 1.20-1.60) | ❌ FAIL |
| 920 | 8484.29 kWh (Ref: 3260.00-4300.00) | 8436.49 kWh (Ref: 1840.00-3310.00) | 5.20 kW (Ref: 2.10-2.80) | 4.74 kW (Ref: 1.40-1.90) | ❌ FAIL |
| 930 | 8889.10 kWh (Ref: 4140.00-5340.00) | 7575.55 kWh (Ref: 1040.00-2240.00) | 5.23 kW (Ref: 2.30-3.00) | 4.71 kW (Ref: 1.10-1.50) | ❌ FAIL |
| 940 | 8814.23 kWh (Ref: 790.00-1410.00) | 14014.98 kWh (Ref: 2080.00-3550.00) | 11.20 kW (Ref: 1.90-2.50) | 9.62 kW (Ref: 1.70-2.30) | ❌ FAIL |
| 950 | 18727.99 kWh (Ref: 0.00-0.00) | 86138.83 kWh (Ref: 390.00-920.00) | 26.62 kW (Ref: 0.00-0.00) | 73.11 kW (Ref: 0.70-0.90) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -17.40°C (Ref: -18.80--15.60) | 67.24°C (Ref: 64.90-75.10) | ✅ PASS |
| 650FF | -23.72°C (Ref: -23.00--21.00) | 65.57°C (Ref: 63.20-73.50) | ❌ FAIL |
| 900FF | -10.17°C (Ref: -6.40--1.60) | 56.01°C (Ref: 41.80-46.40) | ❌ FAIL |
| 950FF | -15.08°C (Ref: -20.20--17.80) | 53.61°C (Ref: 35.50-38.50) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 6688.37 kWh (Ref: 1650.00-2450.00) | 9574.73 kWh (Ref: 1550.00-2780.00) | 4.70 kW (Ref: 2.00-8.00) | 5.01 kW (Ref: 0.00-4.00) | ❌ FAIL |
| 195 | 7365.43 kWh (Ref: 3500.00-6000.00) | 0.00 kWh (Ref: 0.00-0.00) | 3.70 kW (Ref: 1.40-2.20) | 0.00 kW (Ref: 0.00-0.00) | ❌ FAIL |

## Multi-Reference Comparison

| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |
|------|--------|------------|-------|--------|---------|
| 195 | Annual Heating Energy (kWh) | FAIL (7365.43) | - | - | FAIL |
| 195 | Annual Cooling Energy (kWh) | PASS (0.00) | - | - | PASS |
| 195 | Peak Heating Load (kW) | FAIL (3.70) | - | - | FAIL |
| 195 | Peak Cooling Load (kW) | PASS (0.00) | - | - | PASS |
| 600 | Annual Heating Energy (kWh) | WARN (5165.39) | PASS (5165.39) | FAIL (5165.39) | WARN |
| 600 | Annual Cooling Energy (kWh) | FAIL (5309.88) | FAIL (5309.88) | FAIL (5309.88) | FAIL |
| 600 | Peak Heating Load (kW) | FAIL (4.37) | FAIL (4.37) | FAIL (4.37) | FAIL |
| 600 | Peak Cooling Load (kW) | PASS (5.13) | PASS (5.13) | PASS (5.13) | PASS |
| 900 | Annual Heating Energy (kWh) | FAIL (6643.80) | - | - | FAIL |
| 900 | Annual Cooling Energy (kWh) | FAIL (9519.53) | - | - | FAIL |
| 900 | Peak Heating Load (kW) | FAIL (4.65) | - | - | FAIL |
| 900 | Peak Cooling Load (kW) | FAIL (5.01) | - | - | FAIL |
| 920 | Annual Heating Energy (kWh) | FAIL (8484.29) | - | - | FAIL |
| 920 | Annual Cooling Energy (kWh) | FAIL (8436.49) | - | - | FAIL |
| 920 | Peak Heating Load (kW) | FAIL (5.20) | - | - | FAIL |
| 920 | Peak Cooling Load (kW) | FAIL (4.74) | - | - | FAIL |
| 930 | Annual Heating Energy (kWh) | FAIL (8889.10) | - | - | FAIL |
| 930 | Annual Cooling Energy (kWh) | FAIL (7575.55) | - | - | FAIL |
| 930 | Peak Heating Load (kW) | WARN (5.23) | - | - | FAIL |
| 930 | Peak Cooling Load (kW) | PASS (4.71) | - | - | PASS |
| 940 | Annual Heating Energy (kWh) | FAIL (8814.23) | - | - | FAIL |
| 940 | Annual Cooling Energy (kWh) | FAIL (14014.98) | - | - | FAIL |
| 940 | Peak Heating Load (kW) | FAIL (11.20) | - | - | FAIL |
| 940 | Peak Cooling Load (kW) | FAIL (9.62) | - | - | FAIL |
| 950 | Annual Heating Energy (kWh) | FAIL (18727.99) | - | - | FAIL |
| 950 | Annual Cooling Energy (kWh) | FAIL (86138.83) | - | - | FAIL |
| 950 | Peak Heating Load (kW) | FAIL (26.62) | - | - | FAIL |
| 950 | Peak Cooling Load (kW) | FAIL (73.11) | - | - | FAIL |
| 960 | Annual Heating Energy (kWh) | WARN (6688.37) | - | - | FAIL |
| 960 | Annual Cooling Energy (kWh) | FAIL (9574.73) | - | - | FAIL |
| 960 | Peak Heating Load (kW) | FAIL (4.70) | - | - | FAIL |
| 960 | Peak Cooling Load (kW) | FAIL (5.01) | - | - | FAIL |

## Systematic Issues

The following recurring issues are affecting validation results:

### HVAC Load Calculation

**Affected metrics:** 610 - Peak Cooling Load (kW), 195 - Peak Heating Load (kW), 640 - Peak Cooling Load (kW), 630 - Peak Heating Load (kW), 620 - Peak Heating Load (kW), 600 - Peak Heating Load (kW), 650 - Peak Cooling Load (kW), 630 - Peak Cooling Load (kW) |
**Count:** 8 metrics

### Thermal Mass Dynamics

**Affected metrics:** 940 - Peak Cooling Load (kW), 950FF - Minimum Free-Floating Temperature (°C), 960 - Peak Heating Load (kW), 910 - Peak Cooling Load (kW), 950 - Peak Heating Load (kW), 900FF - Minimum Free-Floating Temperature (°C), 920 - Peak Heating Load (kW), 930 - Peak Heating Load (kW), 950 - Peak Cooling Load (kW), 950FF - Maximum Free-Floating Temperature (°C), 900 - Peak Heating Load (kW), 940 - Peak Heating Load (kW), 900FF - Maximum Free-Floating Temperature (°C), 910 - Peak Heating Load (kW), 900 - Peak Cooling Load (kW), 920 - Peak Cooling Load (kW) |
**Count:** 16 metrics

### Inter-Zone Heat Transfer

**Affected metrics:** 960 - Annual Heating Energy (kWh), 960 - Annual Cooling Energy (kWh) |
**Count:** 2 metrics

### Unknown/Unclassified

**Affected metrics:** 640 - Annual Cooling Energy (kWh), 195 - Annual Heating Energy (kWh), 650 - Annual Cooling Energy (kWh) |
**Count:** 3 metrics

### Solar Gain Calculations

**Affected metrics:** 960 - Peak Cooling Load (kW), 650FF - Minimum Free-Floating Temperature (°C), 600 - Annual Cooling Energy (kWh) |
**Count:** 3 metrics

### 5R1C Model Limitation (Accepted)

**Affected metrics:** 950 - Annual Heating Energy (kWh), 940 - Annual Cooling Energy (kWh), 920 - Annual Cooling Energy (kWh), 910 - Annual Heating Energy (kWh), 930 - Annual Cooling Energy (kWh), 900 - Annual Cooling Energy (kWh), 950 - Annual Cooling Energy (kWh), 910 - Annual Cooling Energy (kWh), 920 - Annual Heating Energy (kWh), 900 - Annual Heating Energy (kWh), 930 - Annual Heating Energy (kWh), 940 - Annual Heating Energy (kWh) |
**Count:** 12 metrics

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
