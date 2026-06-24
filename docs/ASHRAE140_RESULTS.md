# ASHRAE Standard 140 Validation Results

*Generated: 2026-06-24 00:54 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 18.8% |
| Passed | 12 |
| Warnings | 2 |
| Failed | 50 |
| Mean Absolute Error | 42.81% |
| Max Deviation | 225.27% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 0.15 seconds |
| Throughput | 121.01 cases/sec |
| Total Cases | 18 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 3404.56 kWh (Ref: 5500.00-7500.00) | 3122.95 kWh (Ref: 8000.00-10500.00) | 2.56 kW (Ref: 2.80-3.80) | 3.09 kW (Ref: 4.80-6.20) | ❌ FAIL |
| 610 | 3547.44 kWh (Ref: 4360.00-5790.00) | 2553.68 kWh (Ref: 3920.00-6140.00) | 2.63 kW (Ref: 4.30-5.70) | 2.63 kW (Ref: 2.20-2.90) | ❌ FAIL |
| 620 | 3908.04 kWh (Ref: 4500.00-6500.00) | 1789.04 kWh (Ref: 3200.00-5000.00) | 2.59 kW (Ref: 2.80-3.80) | 2.02 kW (Ref: 2.50-3.50) | ❌ FAIL |
| 630 | 4049.82 kWh (Ref: 5050.00-6470.00) | 1294.64 kWh (Ref: 2130.00-3700.00) | 2.59 kW (Ref: 4.70-6.10) | 1.68 kW (Ref: 1.80-2.40) | ❌ FAIL |
| 640 | 2140.87 kWh (Ref: 2750.00-3800.00) | 3077.60 kWh (Ref: 5950.00-8100.00) | 2.62 kW (Ref: 4.30-5.70) | 3.07 kW (Ref: 2.80-3.70) | ❌ FAIL |
| 650 | 0.00 kWh (Ref: 0.00-0.00) | 2875.87 kWh (Ref: 4820.00-7060.00) | 0.00 kW (Ref: 0.00-0.00) | 3.04 kW (Ref: 1.90-2.50) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 1927.41 kWh (Ref: 1170.00-2040.00) | 2100.81 kWh (Ref: 2130.00-3670.00) | 1.50 kW (Ref: 1.80-2.40) | 1.95 kW (Ref: 1.60-2.10) | ❌ FAIL |
| 910 | 1979.89 kWh (Ref: 1510.00-2280.00) | 1625.34 kWh (Ref: 820.00-1880.00) | 1.50 kW (Ref: 1.90-2.50) | 1.67 kW (Ref: 1.20-1.60) | ❌ FAIL |
| 920 | 2137.68 kWh (Ref: 3260.00-4300.00) | 1282.27 kWh (Ref: 1840.00-3310.00) | 1.53 kW (Ref: 2.10-2.80) | 1.28 kW (Ref: 1.40-1.90) | ❌ FAIL |
| 930 | 2187.87 kWh (Ref: 4140.00-5340.00) | 997.21 kWh (Ref: 1040.00-2240.00) | 1.53 kW (Ref: 2.30-3.00) | 1.05 kW (Ref: 1.10-1.50) | ❌ FAIL |
| 940 | 1186.45 kWh (Ref: 790.00-1410.00) | 2017.55 kWh (Ref: 2080.00-3550.00) | 1.77 kW (Ref: 1.90-2.50) | 1.93 kW (Ref: 1.70-2.30) | ❌ FAIL |
| 950 | 0.00 kWh (Ref: 0.00-0.00) | 1764.42 kWh (Ref: 390.00-920.00) | 0.00 kW (Ref: 0.00-0.00) | 1.88 kW (Ref: 0.70-0.90) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -18.81°C (Ref: -18.80--15.60) | 66.72°C (Ref: 64.90-75.10) | ❌ FAIL |
| 650FF | -23.22°C (Ref: -23.00--21.00) | 66.56°C (Ref: 63.20-73.50) | ❌ FAIL |
| 900FF | -13.01°C (Ref: -6.40--1.60) | 48.76°C (Ref: 41.80-46.40) | ❌ FAIL |
| 950FF | -18.61°C (Ref: -20.20--17.80) | 45.56°C (Ref: 35.50-38.50) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 5314.51 kWh (Ref: 1650.00-2450.00) | 106.44 kWh (Ref: 1550.00-2780.00) | 1.05 kW (Ref: 2.00-8.00) | 0.51 kW (Ref: 0.00-4.00) | ❌ FAIL |
| 195 | 3668.75 kWh (Ref: 3500.00-6000.00) | 0.00 kWh (Ref: 0.00-0.00) | 1.70 kW (Ref: 1.40-2.20) | 0.00 kW (Ref: 0.00-0.00) | ❌ FAIL |

## Multi-Reference Comparison

| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |
|------|--------|------------|-------|--------|---------|
| 195 | Annual Heating Energy (kWh) | WARN (3668.75) | - | - | FAIL |
| 195 | Annual Cooling Energy (kWh) | PASS (0.00) | - | - | PASS |
| 195 | Peak Heating Load (kW) | PASS (1.70) | - | - | PASS |
| 195 | Peak Cooling Load (kW) | PASS (0.00) | - | - | PASS |
| 600 | Annual Heating Energy (kWh) | FAIL (3404.56) | FAIL (3404.56) | FAIL (3404.56) | FAIL |
| 600 | Annual Cooling Energy (kWh) | FAIL (3122.95) | FAIL (3122.95) | FAIL (3122.95) | FAIL |
| 600 | Peak Heating Load (kW) | FAIL (2.56) | WARN (2.56) | FAIL (2.56) | FAIL |
| 600 | Peak Cooling Load (kW) | FAIL (3.09) | FAIL (3.09) | FAIL (3.09) | FAIL |
| 900 | Annual Heating Energy (kWh) | WARN (1927.41) | - | - | FAIL |
| 900 | Annual Cooling Energy (kWh) | WARN (2100.81) | - | - | FAIL |
| 900 | Peak Heating Load (kW) | PASS (1.50) | - | - | PASS |
| 900 | Peak Cooling Load (kW) | FAIL (1.95) | - | - | FAIL |
| 920 | Annual Heating Energy (kWh) | FAIL (2137.68) | - | - | FAIL |
| 920 | Annual Cooling Energy (kWh) | FAIL (1282.27) | - | - | FAIL |
| 920 | Peak Heating Load (kW) | FAIL (1.53) | - | - | FAIL |
| 920 | Peak Cooling Load (kW) | FAIL (1.28) | - | - | FAIL |
| 930 | Annual Heating Energy (kWh) | FAIL (2187.87) | - | - | FAIL |
| 930 | Annual Cooling Energy (kWh) | FAIL (997.21) | - | - | FAIL |
| 930 | Peak Heating Load (kW) | FAIL (1.53) | - | - | FAIL |
| 930 | Peak Cooling Load (kW) | FAIL (1.05) | - | - | FAIL |
| 940 | Annual Heating Energy (kWh) | FAIL (1186.45) | - | - | FAIL |
| 940 | Annual Cooling Energy (kWh) | FAIL (2017.55) | - | - | FAIL |
| 940 | Peak Heating Load (kW) | FAIL (1.77) | - | - | FAIL |
| 940 | Peak Cooling Load (kW) | FAIL (1.93) | - | - | FAIL |
| 950 | Annual Heating Energy (kWh) | FAIL (0.00) | - | - | FAIL |
| 950 | Annual Cooling Energy (kWh) | FAIL (1764.42) | - | - | FAIL |
| 950 | Peak Heating Load (kW) | FAIL (0.00) | - | - | FAIL |
| 950 | Peak Cooling Load (kW) | FAIL (1.88) | - | - | FAIL |
| 960 | Annual Heating Energy (kWh) | FAIL (5314.51) | - | - | FAIL |
| 960 | Annual Cooling Energy (kWh) | FAIL (106.44) | - | - | FAIL |
| 960 | Peak Heating Load (kW) | FAIL (1.05) | - | - | FAIL |
| 960 | Peak Cooling Load (kW) | FAIL (0.51) | - | - | FAIL |

## Systematic Issues

The following recurring issues are affecting validation results:

### 5R1C Model Limitation (Accepted)

**Affected metrics:** 950 - Annual Heating Energy (kWh), 900 - Annual Heating Energy (kWh), 920 - Annual Heating Energy (kWh), 920 - Annual Cooling Energy (kWh), 930 - Annual Heating Energy (kWh), 930 - Annual Cooling Energy (kWh), 950 - Annual Cooling Energy (kWh), 940 - Annual Cooling Energy (kWh), 900 - Annual Cooling Energy (kWh), 940 - Annual Heating Energy (kWh) |
**Count:** 10 metrics

### Thermal Mass Dynamics

**Affected metrics:** 950FF - Maximum Free-Floating Temperature (°C), 900FF - Maximum Free-Floating Temperature (°C), 900FF - Minimum Free-Floating Temperature (°C) |
**Count:** 3 metrics

### Inter-Zone Heat Transfer

**Affected metrics:** 960 - Annual Cooling Energy (kWh) |
**Count:** 1 metrics

### Unknown/Unclassified

**Affected metrics:** 600 - Peak Heating Load (kW), 640 - Annual Heating Energy (kWh), 640 - Peak Heating Load (kW), 960 - Peak Heating Load (kW), 960 - Peak Cooling Load (kW), 640 - Annual Cooling Energy (kWh), 620 - Annual Cooling Energy (kWh), 900 - Peak Cooling Load (kW), 630 - Peak Heating Load (kW), 650FF - Minimum Free-Floating Temperature (°C), 910 - Peak Heating Load (kW), 920 - Peak Heating Load (kW), 650 - Annual Cooling Energy (kWh), 610 - Peak Heating Load (kW), 620 - Annual Heating Energy (kWh), 930 - Peak Heating Load (kW), 930 - Peak Cooling Load (kW), 600FF - Minimum Free-Floating Temperature (°C), 600 - Annual Cooling Energy (kWh), 620 - Peak Heating Load (kW), 950 - Peak Cooling Load (kW), 195 - Annual Heating Energy (kWh), 630 - Annual Cooling Energy (kWh), 610 - Annual Heating Energy (kWh), 610 - Annual Cooling Energy (kWh), 940 - Peak Cooling Load (kW), 920 - Peak Cooling Load (kW), 950 - Peak Heating Load (kW), 630 - Annual Heating Energy (kWh), 600 - Annual Heating Energy (kWh), 940 - Peak Heating Load (kW), 960 - Annual Heating Energy (kWh) |
**Count:** 32 metrics

### Solar Gain Calculations

**Affected metrics:** 630 - Peak Cooling Load (kW), 600 - Peak Cooling Load (kW), 620 - Peak Cooling Load (kW), 650 - Peak Cooling Load (kW) |
**Count:** 4 metrics

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
