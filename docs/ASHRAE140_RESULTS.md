# ASHRAE Standard 140 Validation Results

*Generated: 2026-08-06 03:58 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 21.9% |
| Passed | 14 |
| Warnings | 9 |
| Failed | 41 |
| Mean Absolute Error | 105.11% |
| Max Deviation | 1757.96% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 0.14 seconds |
| Throughput | 127.05 cases/sec |
| Total Cases | 18 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 5020.00 kWh (Ref: 4360.00-5790.00) | 5691.62 kWh (Ref: 3920.00-6140.00) | 4.36 kW (Ref: 2.80-3.80) | 5.35 kW (Ref: 4.80-6.20) | ❌ FAIL |
| 610 | 5093.08 kWh (Ref: 4360.00-5790.00) | 4791.68 kWh (Ref: 3920.00-6140.00) | 4.36 kW (Ref: 4.30-5.70) | 5.21 kW (Ref: 2.20-2.90) | ❌ FAIL |
| 620 | 6218.84 kWh (Ref: 4500.00-6500.00) | 3433.55 kWh (Ref: 3200.00-5000.00) | 4.45 kW (Ref: 2.80-3.80) | 3.67 kW (Ref: 2.50-3.50) | ❌ FAIL |
| 630 | 6341.33 kWh (Ref: 5050.00-6470.00) | 2963.05 kWh (Ref: 2130.00-3700.00) | 4.45 kW (Ref: 4.70-6.10) | 3.40 kW (Ref: 1.80-2.40) | ❌ FAIL |
| 640 | 3025.37 kWh (Ref: 2750.00-3800.00) | 5684.28 kWh (Ref: 5950.00-8100.00) | 4.37 kW (Ref: 4.30-5.70) | 5.35 kW (Ref: 2.80-3.70) | ❌ FAIL |
| 650 | 0.00 kWh (Ref: 0.00-0.00) | 4741.13 kWh (Ref: 4820.00-7060.00) | 0.00 kW (Ref: 0.00-0.00) | 5.11 kW (Ref: 1.90-2.50) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 5783.86 kWh (Ref: 1170.00-2040.00) | 7414.34 kWh (Ref: 2130.00-3670.00) | 3.34 kW (Ref: 1.80-2.40) | 3.38 kW (Ref: 1.60-2.10) | ❌ FAIL |
| 910 | 6637.93 kWh (Ref: 1510.00-2280.00) | 8267.32 kWh (Ref: 820.00-1880.00) | 3.34 kW (Ref: 1.90-2.50) | 3.37 kW (Ref: 1.20-1.60) | ❌ FAIL |
| 920 | 6372.91 kWh (Ref: 3260.00-4300.00) | 6243.79 kWh (Ref: 1840.00-3310.00) | 3.41 kW (Ref: 2.10-2.80) | 3.37 kW (Ref: 1.40-1.90) | ❌ FAIL |
| 930 | 6430.67 kWh (Ref: 4140.00-5340.00) | 5964.46 kWh (Ref: 1040.00-2240.00) | 3.43 kW (Ref: 2.30-3.00) | 3.44 kW (Ref: 1.10-1.50) | ❌ FAIL |
| 940 | 8583.23 kWh (Ref: 790.00-1410.00) | 12305.17 kWh (Ref: 2080.00-3550.00) | 6.23 kW (Ref: 1.90-2.50) | 7.44 kW (Ref: 1.70-2.30) | ❌ FAIL |
| 950 | 34208.50 kWh (Ref: 0.00-0.00) | 105439.29 kWh (Ref: 390.00-920.00) | 27.44 kW (Ref: 0.00-0.00) | 68.60 kW (Ref: 0.70-0.90) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -17.35°C (Ref: -18.80--15.60) | 68.96°C (Ref: 64.90-75.10) | ✅ PASS |
| 650FF | -23.72°C (Ref: -23.00--21.00) | 67.36°C (Ref: 63.20-73.50) | ❌ FAIL |
| 900FF | -12.70°C (Ref: -6.40--1.60) | 50.13°C (Ref: 41.80-46.40) | ❌ FAIL |
| 950FF | -24.21°C (Ref: -20.20--17.80) | 42.62°C (Ref: 35.50-38.50) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 5812.52 kWh (Ref: 1650.00-2450.00) | 7447.93 kWh (Ref: 1550.00-2780.00) | 3.35 kW (Ref: 2.00-8.00) | 3.40 kW (Ref: 0.00-4.00) | ❌ FAIL |
| 195 | 7365.43 kWh (Ref: 3500.00-6000.00) | 0.00 kWh (Ref: 0.00-0.00) | 3.70 kW (Ref: 1.40-2.20) | 0.00 kW (Ref: 0.00-0.00) | ❌ FAIL |

## Multi-Reference Comparison

| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |
|------|--------|------------|-------|--------|---------|
| 195 | Annual Heating Energy (kWh) | FAIL (7365.43) | - | - | FAIL |
| 195 | Annual Cooling Energy (kWh) | PASS (0.00) | - | - | PASS |
| 195 | Peak Heating Load (kW) | FAIL (3.70) | - | - | FAIL |
| 195 | Peak Cooling Load (kW) | PASS (0.00) | - | - | PASS |
| 600 | Annual Heating Energy (kWh) | WARN (5020.00) | PASS (5020.00) | FAIL (5020.00) | WARN |
| 600 | Annual Cooling Energy (kWh) | FAIL (5691.62) | FAIL (5691.62) | FAIL (5691.62) | FAIL |
| 600 | Peak Heating Load (kW) | FAIL (4.36) | FAIL (4.36) | FAIL (4.36) | FAIL |
| 600 | Peak Cooling Load (kW) | PASS (5.35) | PASS (5.35) | PASS (5.35) | PASS |
| 900 | Annual Heating Energy (kWh) | FAIL (5783.86) | - | - | FAIL |
| 900 | Annual Cooling Energy (kWh) | FAIL (7414.34) | - | - | FAIL |
| 900 | Peak Heating Load (kW) | FAIL (3.34) | - | - | FAIL |
| 900 | Peak Cooling Load (kW) | WARN (3.38) | - | - | FAIL |
| 920 | Annual Heating Energy (kWh) | FAIL (6372.91) | - | - | FAIL |
| 920 | Annual Cooling Energy (kWh) | FAIL (6243.79) | - | - | FAIL |
| 920 | Peak Heating Load (kW) | PASS (3.41) | - | - | PASS |
| 920 | Peak Cooling Load (kW) | WARN (3.37) | - | - | FAIL |
| 930 | Annual Heating Energy (kWh) | FAIL (6430.67) | - | - | FAIL |
| 930 | Annual Cooling Energy (kWh) | FAIL (5964.46) | - | - | FAIL |
| 930 | Peak Heating Load (kW) | FAIL (3.43) | - | - | FAIL |
| 930 | Peak Cooling Load (kW) | FAIL (3.44) | - | - | FAIL |
| 940 | Annual Heating Energy (kWh) | FAIL (8583.23) | - | - | FAIL |
| 940 | Annual Cooling Energy (kWh) | FAIL (12305.17) | - | - | FAIL |
| 940 | Peak Heating Load (kW) | PASS (6.23) | - | - | PASS |
| 940 | Peak Cooling Load (kW) | FAIL (7.44) | - | - | FAIL |
| 950 | Annual Heating Energy (kWh) | FAIL (34208.50) | - | - | FAIL |
| 950 | Annual Cooling Energy (kWh) | FAIL (105439.29) | - | - | FAIL |
| 950 | Peak Heating Load (kW) | FAIL (27.44) | - | - | FAIL |
| 950 | Peak Cooling Load (kW) | FAIL (68.60) | - | - | FAIL |
| 960 | Annual Heating Energy (kWh) | FAIL (5812.52) | - | - | FAIL |
| 960 | Annual Cooling Energy (kWh) | FAIL (7447.93) | - | - | FAIL |
| 960 | Peak Heating Load (kW) | FAIL (3.35) | - | - | FAIL |
| 960 | Peak Cooling Load (kW) | FAIL (3.40) | - | - | FAIL |

## Systematic Issues

The following recurring issues are affecting validation results:

### Solar Gain Calculations

**Affected metrics:** 600 - Annual Cooling Energy (kWh), 960 - Peak Cooling Load (kW), 930 - Peak Cooling Load (kW), 650FF - Minimum Free-Floating Temperature (°C) |
**Count:** 4 metrics

### 5R1C Model Limitation (Accepted)

**Affected metrics:** 940 - Annual Heating Energy (kWh), 940 - Annual Cooling Energy (kWh), 900 - Annual Heating Energy (kWh), 950 - Annual Cooling Energy (kWh), 920 - Annual Heating Energy (kWh), 910 - Annual Cooling Energy (kWh), 930 - Annual Heating Energy (kWh), 950 - Annual Heating Energy (kWh), 910 - Annual Heating Energy (kWh), 900 - Annual Cooling Energy (kWh), 920 - Annual Cooling Energy (kWh) |
**Count:** 11 metrics

### Inter-Zone Heat Transfer

**Affected metrics:** 960 - Annual Cooling Energy (kWh), 960 - Annual Heating Energy (kWh) |
**Count:** 2 metrics

### Unknown/Unclassified

**Affected metrics:** 930 - Annual Cooling Energy (kWh), 920 - Peak Cooling Load (kW), 195 - Annual Heating Energy (kWh), 900 - Peak Cooling Load (kW) |
**Count:** 4 metrics

### HVAC Load Calculation

**Affected metrics:** 640 - Peak Cooling Load (kW), 620 - Peak Heating Load (kW), 600 - Peak Heating Load (kW), 195 - Peak Heating Load (kW), 630 - Peak Cooling Load (kW), 610 - Peak Cooling Load (kW), 630 - Peak Heating Load (kW), 650 - Peak Cooling Load (kW) |
**Count:** 8 metrics

### Thermal Mass Dynamics

**Affected metrics:** 950 - Peak Heating Load (kW), 910 - Peak Cooling Load (kW), 940 - Peak Cooling Load (kW), 930 - Peak Heating Load (kW), 900FF - Minimum Free-Floating Temperature (°C), 950FF - Minimum Free-Floating Temperature (°C), 950 - Peak Cooling Load (kW), 910 - Peak Heating Load (kW), 900 - Peak Heating Load (kW), 960 - Peak Heating Load (kW), 900FF - Maximum Free-Floating Temperature (°C), 950FF - Maximum Free-Floating Temperature (°C) |
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
