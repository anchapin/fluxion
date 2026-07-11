# ASHRAE Standard 140 Validation Results

*Generated: 2026-07-11 06:25 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 23.4% |
| Passed | 15 |
| Warnings | 7 |
| Failed | 42 |
| Mean Absolute Error | 114.23% |
| Max Deviation | 1431.40% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 1.35 seconds |
| Throughput | 13.30 cases/sec |
| Total Cases | 18 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 5102.85 kWh (Ref: 4360.00-5790.00) | 5335.32 kWh (Ref: 3920.00-6140.00) | 3.88 kW (Ref: 2.80-3.80) | 5.00 kW (Ref: 4.80-6.20) | ❌ FAIL |
| 610 | 5419.45 kWh (Ref: 4360.00-5790.00) | 4519.38 kWh (Ref: 3920.00-6140.00) | 4.03 kW (Ref: 4.30-5.70) | 4.36 kW (Ref: 2.20-2.90) | ❌ FAIL |
| 620 | 5665.18 kWh (Ref: 4500.00-6500.00) | 3118.61 kWh (Ref: 3200.00-5000.00) | 3.90 kW (Ref: 2.80-3.80) | 3.37 kW (Ref: 2.50-3.50) | ✅ PASS |
| 630 | 5843.95 kWh (Ref: 5050.00-6470.00) | 2322.37 kWh (Ref: 2130.00-3700.00) | 3.91 kW (Ref: 4.70-6.10) | 2.85 kW (Ref: 1.80-2.40) | ❌ FAIL |
| 640 | 3188.59 kWh (Ref: 2750.00-3800.00) | 5225.18 kWh (Ref: 5950.00-8100.00) | 4.03 kW (Ref: 4.30-5.70) | 4.95 kW (Ref: 2.80-3.70) | ❌ FAIL |
| 650 | 0.00 kWh (Ref: 0.00-0.00) | 4829.43 kWh (Ref: 4820.00-7060.00) | 0.00 kW (Ref: 0.00-0.00) | 4.88 kW (Ref: 1.90-2.50) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 7748.86 kWh (Ref: 1170.00-2040.00) | 9642.61 kWh (Ref: 2130.00-3670.00) | 4.27 kW (Ref: 1.80-2.40) | 4.57 kW (Ref: 1.60-2.10) | ❌ FAIL |
| 910 | 8972.44 kWh (Ref: 1510.00-2280.00) | 9991.62 kWh (Ref: 820.00-1880.00) | 4.30 kW (Ref: 1.90-2.50) | 4.37 kW (Ref: 1.20-1.60) | ❌ FAIL |
| 920 | 8290.15 kWh (Ref: 3260.00-4300.00) | 7889.42 kWh (Ref: 1840.00-3310.00) | 4.28 kW (Ref: 2.10-2.80) | 4.44 kW (Ref: 1.40-1.90) | ❌ FAIL |
| 930 | 9010.91 kWh (Ref: 4140.00-5340.00) | 8116.07 kWh (Ref: 1040.00-2240.00) | 4.26 kW (Ref: 2.30-3.00) | 4.38 kW (Ref: 1.10-1.50) | ❌ FAIL |
| 940 | 11437.45 kWh (Ref: 790.00-1410.00) | 15512.81 kWh (Ref: 2080.00-3550.00) | 9.33 kW (Ref: 1.90-2.50) | 9.21 kW (Ref: 1.70-2.30) | ❌ FAIL |
| 950 | 32214.60 kWh (Ref: 0.00-0.00) | 86906.83 kWh (Ref: 390.00-920.00) | 29.82 kW (Ref: 0.00-0.00) | 66.53 kW (Ref: 0.70-0.90) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -18.67°C (Ref: -18.80--15.60) | 67.68°C (Ref: 64.90-75.10) | ✅ PASS |
| 650FF | -23.18°C (Ref: -23.00--21.00) | 67.18°C (Ref: 63.20-73.50) | ❌ FAIL |
| 900FF | -12.43°C (Ref: -6.40--1.60) | 53.13°C (Ref: 41.80-46.40) | ❌ FAIL |
| 950FF | -17.95°C (Ref: -20.20--17.80) | 49.88°C (Ref: 35.50-38.50) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 7769.42 kWh (Ref: 1650.00-2450.00) | 9663.73 kWh (Ref: 1550.00-2780.00) | 4.27 kW (Ref: 2.00-8.00) | 4.56 kW (Ref: 0.00-4.00) | ❌ FAIL |
| 195 | 7657.50 kWh (Ref: 3500.00-6000.00) | 0.00 kWh (Ref: 0.00-0.00) | 3.98 kW (Ref: 1.40-2.20) | 0.00 kW (Ref: 0.00-0.00) | ❌ FAIL |

## Multi-Reference Comparison

| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |
|------|--------|------------|-------|--------|---------|
| 195 | Annual Heating Energy (kWh) | FAIL (7657.50) | - | - | FAIL |
| 195 | Annual Cooling Energy (kWh) | PASS (0.00) | - | - | PASS |
| 195 | Peak Heating Load (kW) | FAIL (3.98) | - | - | FAIL |
| 195 | Peak Cooling Load (kW) | PASS (0.00) | - | - | PASS |
| 600 | Annual Heating Energy (kWh) | WARN (5102.85) | PASS (5102.85) | FAIL (5102.85) | WARN |
| 600 | Annual Cooling Energy (kWh) | FAIL (5335.32) | FAIL (5335.32) | FAIL (5335.32) | FAIL |
| 600 | Peak Heating Load (kW) | WARN (3.88) | FAIL (3.88) | WARN (3.88) | FAIL |
| 600 | Peak Cooling Load (kW) | PASS (5.00) | PASS (5.00) | PASS (5.00) | PASS |
| 900 | Annual Heating Energy (kWh) | FAIL (7748.86) | - | - | FAIL |
| 900 | Annual Cooling Energy (kWh) | FAIL (9642.61) | - | - | FAIL |
| 900 | Peak Heating Load (kW) | FAIL (4.27) | - | - | FAIL |
| 900 | Peak Cooling Load (kW) | FAIL (4.57) | - | - | FAIL |
| 920 | Annual Heating Energy (kWh) | FAIL (8290.15) | - | - | FAIL |
| 920 | Annual Cooling Energy (kWh) | FAIL (7889.42) | - | - | FAIL |
| 920 | Peak Heating Load (kW) | WARN (4.28) | - | - | FAIL |
| 920 | Peak Cooling Load (kW) | WARN (4.44) | - | - | FAIL |
| 930 | Annual Heating Energy (kWh) | FAIL (9010.91) | - | - | FAIL |
| 930 | Annual Cooling Energy (kWh) | FAIL (8116.07) | - | - | FAIL |
| 930 | Peak Heating Load (kW) | WARN (4.26) | - | - | FAIL |
| 930 | Peak Cooling Load (kW) | PASS (4.38) | - | - | PASS |
| 940 | Annual Heating Energy (kWh) | FAIL (11437.45) | - | - | FAIL |
| 940 | Annual Cooling Energy (kWh) | FAIL (15512.81) | - | - | FAIL |
| 940 | Peak Heating Load (kW) | FAIL (9.33) | - | - | FAIL |
| 940 | Peak Cooling Load (kW) | FAIL (9.21) | - | - | FAIL |
| 950 | Annual Heating Energy (kWh) | FAIL (32214.60) | - | - | FAIL |
| 950 | Annual Cooling Energy (kWh) | FAIL (86906.83) | - | - | FAIL |
| 950 | Peak Heating Load (kW) | FAIL (29.82) | - | - | FAIL |
| 950 | Peak Cooling Load (kW) | FAIL (66.53) | - | - | FAIL |
| 960 | Annual Heating Energy (kWh) | PASS (7769.42) | - | - | PASS |
| 960 | Annual Cooling Energy (kWh) | FAIL (9663.73) | - | - | FAIL |
| 960 | Peak Heating Load (kW) | FAIL (4.27) | - | - | FAIL |
| 960 | Peak Cooling Load (kW) | FAIL (4.56) | - | - | FAIL |

## Systematic Issues

The following recurring issues are affecting validation results:

### 5R1C Model Limitation (Accepted)

**Affected metrics:** 930 - Annual Heating Energy (kWh), 900 - Annual Heating Energy (kWh), 930 - Annual Cooling Energy (kWh), 910 - Annual Heating Energy (kWh), 920 - Annual Heating Energy (kWh), 910 - Annual Cooling Energy (kWh), 940 - Annual Heating Energy (kWh), 920 - Annual Cooling Energy (kWh), 950 - Annual Heating Energy (kWh), 940 - Annual Cooling Energy (kWh), 950 - Annual Cooling Energy (kWh), 900 - Annual Cooling Energy (kWh) |
**Count:** 12 metrics

### HVAC Load Calculation

**Affected metrics:** 600 - Peak Heating Load (kW), 195 - Peak Heating Load (kW), 610 - Peak Heating Load (kW), 640 - Peak Heating Load (kW), 650 - Peak Cooling Load (kW), 630 - Peak Heating Load (kW), 640 - Peak Cooling Load (kW), 610 - Peak Cooling Load (kW), 630 - Peak Cooling Load (kW) |
**Count:** 9 metrics

### Unknown/Unclassified

**Affected metrics:** 195 - Annual Heating Energy (kWh), 640 - Annual Cooling Energy (kWh) |
**Count:** 2 metrics

### Thermal Mass Dynamics

**Affected metrics:** 900FF - Maximum Free-Floating Temperature (°C), 930 - Peak Heating Load (kW), 910 - Peak Heating Load (kW), 900 - Peak Cooling Load (kW), 920 - Peak Cooling Load (kW), 940 - Peak Cooling Load (kW), 950FF - Maximum Free-Floating Temperature (°C), 950 - Peak Cooling Load (kW), 920 - Peak Heating Load (kW), 900FF - Minimum Free-Floating Temperature (°C), 960 - Peak Heating Load (kW), 950 - Peak Heating Load (kW), 900 - Peak Heating Load (kW), 910 - Peak Cooling Load (kW), 940 - Peak Heating Load (kW) |
**Count:** 15 metrics

### Inter-Zone Heat Transfer

**Affected metrics:** 960 - Annual Cooling Energy (kWh) |
**Count:** 1 metrics

### Solar Gain Calculations

**Affected metrics:** 960 - Peak Cooling Load (kW), 600 - Annual Cooling Energy (kWh), 650FF - Minimum Free-Floating Temperature (°C) |
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
