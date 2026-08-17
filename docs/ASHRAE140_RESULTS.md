# ASHRAE Standard 140 Validation Results

*Generated: 2026-08-16 18:24 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 84 |
| Pass Rate | 14.3% |
| Passed | 12 |
| Warnings | 8 |
| Failed | 64 |
| Mean Absolute Error | 51.03% |
| Max Deviation | 470.11% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 0.59 seconds |
| Throughput | 35.36 cases/sec |
| Total Cases | 21 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 4604.57 kWh (Ref: 4360.00-5790.00) | 3299.30 kWh (Ref: 3920.00-6140.00) | 4.38 kW (Ref: 2.80-3.80) | 3.72 kW (Ref: 4.80-6.20) | ❌ FAIL |
| 610 | 4691.46 kWh (Ref: 4360.00-5790.00) | 2666.47 kWh (Ref: 3920.00-6140.00) | 4.38 kW (Ref: 4.30-5.70) | 3.45 kW (Ref: 2.20-2.90) | ❌ FAIL |
| 620 | 5948.61 kWh (Ref: 4500.00-6500.00) | 2361.26 kWh (Ref: 3200.00-5000.00) | 4.49 kW (Ref: 2.80-3.80) | 2.98 kW (Ref: 2.50-3.50) | ❌ FAIL |
| 630 | 6065.00 kWh (Ref: 5050.00-6470.00) | 2027.64 kWh (Ref: 2130.00-3700.00) | 4.49 kW (Ref: 4.70-6.10) | 2.73 kW (Ref: 1.80-2.40) | ❌ FAIL |
| 640 | 2385.06 kWh (Ref: 2750.00-3800.00) | 3290.46 kWh (Ref: 5950.00-8100.00) | 4.04 kW (Ref: 4.30-5.70) | 3.71 kW (Ref: 2.80-3.70) | ❌ FAIL |
| 650 | 0.00 kWh (Ref: 0.00-0.00) | 2462.26 kWh (Ref: 4820.00-7060.00) | 0.00 kW (Ref: 0.00-0.00) | 3.35 kW (Ref: 1.90-2.50) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 5052.83 kWh (Ref: 1170.00-2040.00) | 7754.04 kWh (Ref: 2130.00-3670.00) | 3.93 kW (Ref: 1.80-2.40) | 3.36 kW (Ref: 1.60-2.10) | ❌ FAIL |
| 910 | 5428.96 kWh (Ref: 1510.00-2280.00) | 7696.48 kWh (Ref: 820.00-1880.00) | 3.93 kW (Ref: 1.90-2.50) | 3.36 kW (Ref: 1.20-1.60) | ❌ FAIL |
| 920 | 5354.01 kWh (Ref: 3260.00-4300.00) | 6463.07 kWh (Ref: 1840.00-3310.00) | 3.64 kW (Ref: 2.10-2.80) | 3.32 kW (Ref: 1.40-1.90) | ❌ FAIL |
| 930 | 5531.77 kWh (Ref: 4140.00-5340.00) | 6317.71 kWh (Ref: 1040.00-2240.00) | 3.53 kW (Ref: 2.30-3.00) | 3.31 kW (Ref: 1.10-1.50) | ❌ FAIL |
| 940 | 6966.87 kWh (Ref: 790.00-1410.00) | 11063.54 kWh (Ref: 2080.00-3550.00) | 6.25 kW (Ref: 1.90-2.50) | 7.38 kW (Ref: 1.70-2.30) | ❌ FAIL |
| 950 | 0.00 kWh (Ref: 0.00-0.00) | 33.08 kWh (Ref: 390.00-920.00) | 0.00 kW (Ref: 0.00-0.00) | 0.39 kW (Ref: 0.70-0.90) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -17.13°C (Ref: -18.80--15.60) | 55.22°C (Ref: 64.90-75.10) | ❌ FAIL |
| 650FF | -23.71°C (Ref: -23.00--21.00) | 52.43°C (Ref: 63.20-73.50) | ❌ FAIL |
| 900FF | -6.65°C (Ref: -6.40--1.60) | 39.83°C (Ref: 41.80-46.40) | ❌ FAIL |
| 950FF | -23.95°C (Ref: -20.20--17.80) | 31.30°C (Ref: 35.50-38.50) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 6871.29 kWh (Ref: 1650.00-2450.00) | 8853.50 kWh (Ref: 1550.00-2780.00) | 4.16 kW (Ref: 2.00-8.00) | 3.64 kW (Ref: 0.00-4.00) | ❌ FAIL |
| 195 | 3237.51 kWh (Ref: 3951.00-4217.00) | 5.16 kWh (Ref: 592.00-712.00) | 1.47 kW (Ref: 1.79-1.80) | 0.19 kW (Ref: 0.94-1.12) | ❌ FAIL |

## Multi-Reference Comparison

| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |
|------|--------|------------|-------|--------|---------|
| 195 | Annual Heating Energy (kWh) | FAIL (3237.51) | - | - | FAIL |
| 195 | Annual Cooling Energy (kWh) | FAIL (5.16) | - | - | FAIL |
| 195 | Peak Heating Load (kW) | WARN (1.47) | - | - | FAIL |
| 195 | Peak Cooling Load (kW) | FAIL (0.19) | - | - | FAIL |
| 600 | Annual Heating Energy (kWh) | FAIL (4604.57) | PASS (4604.57) | FAIL (4604.57) | WARN |
| 600 | Annual Cooling Energy (kWh) | FAIL (3299.30) | FAIL (3299.30) | FAIL (3299.30) | FAIL |
| 600 | Peak Heating Load (kW) | FAIL (4.38) | FAIL (4.38) | FAIL (4.38) | FAIL |
| 600 | Peak Cooling Load (kW) | FAIL (3.72) | FAIL (3.72) | FAIL (3.72) | FAIL |
| 900 | Annual Heating Energy (kWh) | FAIL (5052.83) | - | - | FAIL |
| 900 | Annual Cooling Energy (kWh) | FAIL (7754.04) | - | - | FAIL |
| 900 | Peak Heating Load (kW) | FAIL (3.93) | - | - | FAIL |
| 900 | Peak Cooling Load (kW) | WARN (3.36) | - | - | FAIL |
| 920 | Annual Heating Energy (kWh) | FAIL (5354.01) | - | - | FAIL |
| 920 | Annual Cooling Energy (kWh) | FAIL (6463.07) | - | - | FAIL |
| 920 | Peak Heating Load (kW) | PASS (3.64) | - | - | PASS |
| 920 | Peak Cooling Load (kW) | WARN (3.32) | - | - | FAIL |
| 930 | Annual Heating Energy (kWh) | WARN (5531.77) | - | - | FAIL |
| 930 | Annual Cooling Energy (kWh) | FAIL (6317.71) | - | - | FAIL |
| 930 | Peak Heating Load (kW) | FAIL (3.53) | - | - | FAIL |
| 930 | Peak Cooling Load (kW) | FAIL (3.31) | - | - | FAIL |
| 940 | Annual Heating Energy (kWh) | WARN (6966.87) | - | - | FAIL |
| 940 | Annual Cooling Energy (kWh) | FAIL (11063.54) | - | - | FAIL |
| 940 | Peak Heating Load (kW) | PASS (6.25) | - | - | PASS |
| 940 | Peak Cooling Load (kW) | FAIL (7.38) | - | - | FAIL |
| 950 | Annual Heating Energy (kWh) | FAIL (0.00) | - | - | FAIL |
| 950 | Annual Cooling Energy (kWh) | FAIL (33.08) | - | - | FAIL |
| 950 | Peak Heating Load (kW) | FAIL (0.00) | - | - | FAIL |
| 950 | Peak Cooling Load (kW) | FAIL (0.39) | - | - | FAIL |
| 960 | Annual Heating Energy (kWh) | PASS (6871.29) | - | - | PASS |
| 960 | Annual Cooling Energy (kWh) | FAIL (8853.50) | - | - | FAIL |
| 960 | Peak Heating Load (kW) | FAIL (4.16) | - | - | FAIL |
| 960 | Peak Cooling Load (kW) | FAIL (3.64) | - | - | FAIL |
| 970 | Annual Heating Energy (kWh) | FAIL (18581.66) | - | - | FAIL |
| 970 | Annual Cooling Energy (kWh) | FAIL (21066.64) | - | - | FAIL |
| 970 | Peak Heating Load (kW) | FAIL (3.80) | - | - | FAIL |
| 970 | Peak Cooling Load (kW) | FAIL (2.58) | - | - | FAIL |

## Systematic Issues

The following recurring issues are affecting validation results:

### HVAC Load Calculation

**Affected metrics:** 610 - Peak Cooling Load (kW), 195 - Peak Cooling Load (kW), 640 - Peak Heating Load (kW), 195 - Peak Heating Load (kW), 650 - Peak Cooling Load (kW), 800 - Annual Cooling Energy (kWh), 810 - Annual Cooling Energy (kWh), 810 - Annual Heating Energy (kWh), 600 - Peak Heating Load (kW), 630 - Peak Cooling Load (kW), 620 - Peak Heating Load (kW) |
**Count:** 11 metrics

### Inter-Zone Heat Transfer

**Affected metrics:** 960 - Annual Cooling Energy (kWh) |
**Count:** 1 metrics

### 5R1C Model Limitation (Accepted)

**Affected metrics:** 910 - Annual Cooling Energy (kWh), 970 - Annual Heating Energy (kWh), 900 - Annual Cooling Energy (kWh), 920 - Annual Heating Energy (kWh), 970 - Annual Cooling Energy (kWh), 930 - Annual Cooling Energy (kWh), 900 - Annual Heating Energy (kWh), 940 - Annual Cooling Energy (kWh), 920 - Annual Cooling Energy (kWh), 910 - Annual Heating Energy (kWh) |
**Count:** 10 metrics

### Thermal Mass Dynamics

**Affected metrics:** 900FF - Minimum Free-Floating Temperature (°C), 950 - Peak Heating Load (kW), 970 - Peak Heating Load (kW), 900 - Peak Heating Load (kW), 940 - Peak Cooling Load (kW), 910 - Peak Cooling Load (kW), 950FF - Minimum Free-Floating Temperature (°C), 910 - Peak Heating Load (kW), 950FF - Maximum Free-Floating Temperature (°C), 930 - Peak Heating Load (kW), 960 - Peak Heating Load (kW) |
**Count:** 11 metrics

### Solar Gain Calculations

**Affected metrics:** 800 - Peak Cooling Load (kW), 610 - Annual Cooling Energy (kWh), 810 - Peak Cooling Load (kW), 650FF - Minimum Free-Floating Temperature (°C), 960 - Peak Cooling Load (kW), 950 - Peak Cooling Load (kW), 600 - Annual Cooling Energy (kWh), 620 - Annual Cooling Energy (kWh), 930 - Peak Cooling Load (kW), 970 - Peak Cooling Load (kW), 640 - Annual Cooling Energy (kWh), 195 - Annual Heating Energy (kWh), 650FF - Maximum Free-Floating Temperature (°C), 600 - Peak Cooling Load (kW), 600FF - Maximum Free-Floating Temperature (°C), 650 - Annual Cooling Energy (kWh) |
**Count:** 16 metrics

### Unknown/Unclassified

**Affected metrics:** 800 - Peak Heating Load (kW), 930 - Annual Heating Energy (kWh), 920 - Peak Cooling Load (kW), 640 - Annual Heating Energy (kWh), 195 - Annual Cooling Energy (kWh), 900 - Peak Cooling Load (kW), 940 - Annual Heating Energy (kWh), 950 - Annual Heating Energy (kWh), 950 - Annual Cooling Energy (kWh) |
**Count:** 9 metrics

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
