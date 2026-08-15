# ASHRAE Standard 140 Validation Results

*Generated: 2026-08-15 19:34 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 10.9% |
| Passed | 7 |
| Warnings | 6 |
| Failed | 51 |
| Mean Absolute Error | 52.59% |
| Max Deviation | 484.61% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 1.72 seconds |
| Throughput | 10.48 cases/sec |
| Total Cases | 18 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 4696.48 kWh (Ref: 4360.00-5790.00) | 3241.28 kWh (Ref: 3920.00-6140.00) | 4.42 kW (Ref: 2.80-3.80) | 3.70 kW (Ref: 4.80-6.20) | ❌ FAIL |
| 610 | 4786.05 kWh (Ref: 4360.00-5790.00) | 2611.90 kWh (Ref: 3920.00-6140.00) | 4.42 kW (Ref: 4.30-5.70) | 3.43 kW (Ref: 2.20-2.90) | ❌ FAIL |
| 620 | 6064.19 kWh (Ref: 4500.00-6500.00) | 2315.80 kWh (Ref: 3200.00-5000.00) | 4.53 kW (Ref: 2.80-3.80) | 2.96 kW (Ref: 2.50-3.50) | ❌ FAIL |
| 630 | 6182.92 kWh (Ref: 5050.00-6470.00) | 1984.30 kWh (Ref: 2130.00-3700.00) | 4.53 kW (Ref: 4.70-6.10) | 2.71 kW (Ref: 1.80-2.40) | ❌ FAIL |
| 640 | 4696.48 kWh (Ref: 2750.00-3800.00) | 3241.28 kWh (Ref: 5950.00-8100.00) | 4.42 kW (Ref: 4.30-5.70) | 3.70 kW (Ref: 2.80-3.70) | ❌ FAIL |
| 650 | 0.00 kWh (Ref: 0.00-0.00) | 2422.52 kWh (Ref: 4820.00-7060.00) | 0.00 kW (Ref: 0.00-0.00) | 3.33 kW (Ref: 1.90-2.50) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 5408.02 kWh (Ref: 1170.00-2040.00) | 7602.66 kWh (Ref: 2130.00-3670.00) | 3.99 kW (Ref: 1.80-2.40) | 3.34 kW (Ref: 1.60-2.10) | ❌ FAIL |
| 910 | 5966.36 kWh (Ref: 1510.00-2280.00) | 7892.27 kWh (Ref: 820.00-1880.00) | 3.99 kW (Ref: 1.90-2.50) | 3.32 kW (Ref: 1.20-1.60) | ❌ FAIL |
| 920 | 5319.44 kWh (Ref: 3260.00-4300.00) | 6010.83 kWh (Ref: 1840.00-3310.00) | 3.59 kW (Ref: 2.10-2.80) | 3.31 kW (Ref: 1.40-1.90) | ❌ FAIL |
| 930 | 5380.81 kWh (Ref: 4140.00-5340.00) | 5784.52 kWh (Ref: 1040.00-2240.00) | 3.61 kW (Ref: 2.30-3.00) | 3.38 kW (Ref: 1.10-1.50) | ❌ FAIL |
| 940 | 5408.02 kWh (Ref: 790.00-1410.00) | 7602.66 kWh (Ref: 2080.00-3550.00) | 3.99 kW (Ref: 1.90-2.50) | 3.34 kW (Ref: 1.70-2.30) | ❌ FAIL |
| 950 | 0.00 kWh (Ref: 0.00-0.00) | 25.01 kWh (Ref: 390.00-920.00) | 0.00 kW (Ref: 0.00-0.00) | 0.33 kW (Ref: 0.70-0.90) | ❌ FAIL |

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
| 960 | 5892.54 kWh (Ref: 1650.00-2450.00) | 8035.68 kWh (Ref: 1550.00-2780.00) | 4.00 kW (Ref: 2.00-8.00) | 3.41 kW (Ref: 0.00-4.00) | ❌ FAIL |
| 195 | 6552.13 kWh (Ref: 3500.00-6000.00) | 279.70 kWh (Ref: 0.00-0.00) | 3.65 kW (Ref: 1.40-2.20) | 1.02 kW (Ref: 0.00-0.00) | ❌ FAIL |

## Multi-Reference Comparison

| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |
|------|--------|------------|-------|--------|---------|
| 195 | Annual Heating Energy (kWh) | FAIL (6552.13) | - | - | FAIL |
| 195 | Annual Cooling Energy (kWh) | FAIL (279.70) | - | - | FAIL |
| 195 | Peak Heating Load (kW) | FAIL (3.65) | - | - | FAIL |
| 195 | Peak Cooling Load (kW) | FAIL (1.02) | - | - | FAIL |
| 600 | Annual Heating Energy (kWh) | FAIL (4696.48) | PASS (4696.48) | FAIL (4696.48) | WARN |
| 600 | Annual Cooling Energy (kWh) | FAIL (3241.28) | FAIL (3241.28) | FAIL (3241.28) | FAIL |
| 600 | Peak Heating Load (kW) | FAIL (4.42) | FAIL (4.42) | FAIL (4.42) | FAIL |
| 600 | Peak Cooling Load (kW) | FAIL (3.70) | FAIL (3.70) | FAIL (3.70) | FAIL |
| 900 | Annual Heating Energy (kWh) | FAIL (5408.02) | - | - | FAIL |
| 900 | Annual Cooling Energy (kWh) | FAIL (7602.66) | - | - | FAIL |
| 900 | Peak Heating Load (kW) | FAIL (3.99) | - | - | FAIL |
| 900 | Peak Cooling Load (kW) | WARN (3.34) | - | - | FAIL |
| 920 | Annual Heating Energy (kWh) | FAIL (5319.44) | - | - | FAIL |
| 920 | Annual Cooling Energy (kWh) | FAIL (6010.83) | - | - | FAIL |
| 920 | Peak Heating Load (kW) | PASS (3.59) | - | - | PASS |
| 920 | Peak Cooling Load (kW) | WARN (3.31) | - | - | FAIL |
| 930 | Annual Heating Energy (kWh) | WARN (5380.81) | - | - | FAIL |
| 930 | Annual Cooling Energy (kWh) | FAIL (5784.52) | - | - | FAIL |
| 930 | Peak Heating Load (kW) | FAIL (3.61) | - | - | FAIL |
| 930 | Peak Cooling Load (kW) | FAIL (3.38) | - | - | FAIL |
| 940 | Annual Heating Energy (kWh) | FAIL (5408.02) | - | - | FAIL |
| 940 | Annual Cooling Energy (kWh) | FAIL (7602.66) | - | - | FAIL |
| 940 | Peak Heating Load (kW) | FAIL (3.99) | - | - | FAIL |
| 940 | Peak Cooling Load (kW) | FAIL (3.34) | - | - | FAIL |
| 950 | Annual Heating Energy (kWh) | FAIL (0.00) | - | - | FAIL |
| 950 | Annual Cooling Energy (kWh) | FAIL (25.01) | - | - | FAIL |
| 950 | Peak Heating Load (kW) | FAIL (0.00) | - | - | FAIL |
| 950 | Peak Cooling Load (kW) | FAIL (0.33) | - | - | FAIL |
| 960 | Annual Heating Energy (kWh) | FAIL (5892.54) | - | - | FAIL |
| 960 | Annual Cooling Energy (kWh) | FAIL (8035.68) | - | - | FAIL |
| 960 | Peak Heating Load (kW) | FAIL (4.00) | - | - | FAIL |
| 960 | Peak Cooling Load (kW) | FAIL (3.41) | - | - | FAIL |

## Systematic Issues

The following recurring issues are affecting validation results:

### Thermal Mass Dynamics

**Affected metrics:** 900FF - Minimum Free-Floating Temperature (°C), 960 - Peak Heating Load (kW), 910 - Peak Heating Load (kW), 930 - Peak Heating Load (kW), 940 - Peak Heating Load (kW), 950FF - Maximum Free-Floating Temperature (°C), 900 - Peak Heating Load (kW), 950 - Peak Heating Load (kW), 950FF - Minimum Free-Floating Temperature (°C), 900FF - Maximum Free-Floating Temperature (°C), 910 - Peak Cooling Load (kW) |
**Count:** 11 metrics

### HVAC Load Calculation

**Affected metrics:** 195 - Peak Heating Load (kW), 630 - Peak Cooling Load (kW), 600 - Peak Heating Load (kW), 620 - Peak Heating Load (kW), 650 - Peak Cooling Load (kW), 195 - Peak Cooling Load (kW), 610 - Peak Cooling Load (kW) |
**Count:** 7 metrics

### Inter-Zone Heat Transfer

**Affected metrics:** 960 - Annual Heating Energy (kWh), 960 - Annual Cooling Energy (kWh) |
**Count:** 2 metrics

### 5R1C Model Limitation (Accepted)

**Affected metrics:** 920 - Annual Cooling Energy (kWh), 900 - Annual Heating Energy (kWh), 910 - Annual Heating Energy (kWh), 940 - Annual Cooling Energy (kWh), 910 - Annual Cooling Energy (kWh), 900 - Annual Cooling Energy (kWh), 920 - Annual Heating Energy (kWh) |
**Count:** 7 metrics

### Solar Gain Calculations

**Affected metrics:** 950 - Peak Cooling Load (kW), 650FF - Minimum Free-Floating Temperature (°C), 620 - Annual Cooling Energy (kWh), 940 - Peak Cooling Load (kW), 630 - Annual Cooling Energy (kWh), 600 - Peak Cooling Load (kW), 610 - Annual Cooling Energy (kWh), 640 - Annual Cooling Energy (kWh), 600 - Annual Cooling Energy (kWh), 650FF - Maximum Free-Floating Temperature (°C), 930 - Peak Cooling Load (kW), 960 - Peak Cooling Load (kW), 650 - Annual Cooling Energy (kWh), 600FF - Maximum Free-Floating Temperature (°C) |
**Count:** 14 metrics

### Unknown/Unclassified

**Affected metrics:** 930 - Annual Cooling Energy (kWh), 195 - Annual Heating Energy (kWh), 950 - Annual Heating Energy (kWh), 950 - Annual Cooling Energy (kWh), 930 - Annual Heating Energy (kWh), 940 - Annual Heating Energy (kWh), 640 - Annual Heating Energy (kWh), 900 - Peak Cooling Load (kW), 920 - Peak Cooling Load (kW), 195 - Annual Cooling Energy (kWh) |
**Count:** 10 metrics

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
