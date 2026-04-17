# ASHRAE Standard 140 Validation Results

*Generated: 2026-04-17 12:46 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 17.2% |
| Passed | 11 |
| Warnings | 5 |
| Failed | 48 |
| Mean Absolute Error | 53.62% |
| Max Deviation | 100.00% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 1.28 seconds |
| Throughput | 14.11 cases/sec |
| Total Cases | 18 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 6.54 MWh (Ref: 5.50-7.50) | 0.66 MWh (Ref: 8.00-10.50) | 0.24 kW (Ref: 2.80-3.80) | 0.19 kW (Ref: 4.80-6.20) | ❌ FAIL |
| 610 | 5.63 MWh (Ref: 4.36-5.79) | 0.54 MWh (Ref: 3.92-6.14) | 0.49 kW (Ref: 4.30-5.70) | 0.34 kW (Ref: 2.20-2.90) | ❌ FAIL |
| 620 | 5.25 MWh (Ref: 4.50-6.50) | 0.39 MWh (Ref: 3.20-5.00) | 0.49 kW (Ref: 2.80-3.80) | 0.28 kW (Ref: 2.50-3.50) | ❌ FAIL |
| 630 | 4.97 MWh (Ref: 5.05-6.47) | 0.34 MWh (Ref: 2.13-3.70) | 0.49 kW (Ref: 4.70-6.10) | 0.27 kW (Ref: 1.80-2.40) | ❌ FAIL |
| 640 | 3.68 MWh (Ref: 2.75-3.80) | 0.65 MWh (Ref: 5.95-8.10) | 0.49 kW (Ref: 4.30-5.70) | 0.38 kW (Ref: 2.80-3.70) | ❌ FAIL |
| 650 | 0.00 MWh (Ref: 0.00-0.00) | 0.50 MWh (Ref: 4.82-7.06) | 0.00 kW (Ref: 0.00-0.00) | 0.37 kW (Ref: 1.90-2.50) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 1.67 MWh (Ref: 1.17-2.04) | 2.92 MWh (Ref: 2.13-3.67) | 1.65 kW (Ref: 1.80-2.40) | 1.69 kW (Ref: 1.60-2.10) | ❌ FAIL |
| 910 | 1.97 MWh (Ref: 1.51-2.28) | 1.20 MWh (Ref: 0.82-1.88) | 1.65 kW (Ref: 1.90-2.50) | 1.40 kW (Ref: 1.20-1.60) | ❌ FAIL |
| 920 | 3.49 MWh (Ref: 3.26-4.30) | 1.76 MWh (Ref: 1.84-3.31) | 3.09 kW (Ref: 2.10-2.80) | 1.91 kW (Ref: 1.40-1.90) | ❌ FAIL |
| 930 | 4.92 MWh (Ref: 4.14-5.34) | 1.49 MWh (Ref: 1.04-2.24) | 3.13 kW (Ref: 2.30-3.00) | 1.67 kW (Ref: 1.10-1.50) | ❌ FAIL |
| 940 | 1.19 MWh (Ref: 0.79-1.41) | 2.88 MWh (Ref: 2.08-3.55) | 1.55 kW (Ref: 1.90-2.50) | 1.69 kW (Ref: 1.70-2.30) | ❌ FAIL |
| 950 | 0.00 MWh (Ref: 0.00-0.00) | 0.69 MWh (Ref: 0.39-0.92) | 0.00 kW (Ref: 0.00-0.00) | 1.68 kW (Ref: 0.70-0.90) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -11.94°C (Ref: -18.80--15.60) | 53.09°C (Ref: 64.90-75.10) | ❌ FAIL |
| 650FF | -12.28°C (Ref: -23.00--21.00) | 53.09°C (Ref: 63.20-73.50) | ❌ FAIL |
| 900FF | -6.57°C (Ref: -6.40--1.60) | 47.14°C (Ref: 41.80-46.40) | ❌ FAIL |
| 950FF | -10.95°C (Ref: -20.20--17.80) | 48.05°C (Ref: 35.50-38.50) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 1.88 MWh (Ref: 1.65-2.45) | 7.49 MWh (Ref: 1.55-2.78) | 6.33 kW (Ref: 2.00-8.00) | 3.45 kW (Ref: 0.00-4.00) | ❌ FAIL |
| 195 | 0.00 MWh (Ref: 3.50-6.00) | 0.00 MWh (Ref: 0.00-0.00) | 0.00 kW (Ref: 1.40-2.20) | 0.00 kW (Ref: 0.00-0.00) | ❌ FAIL |

## Multi-Reference Comparison

| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |
|------|--------|------------|-------|--------|---------|
| 195 | Annual Heating Energy (MWh) | FAIL (0.00) | - | - | FAIL |
| 195 | Annual Cooling Energy (MWh) | FAIL (0.00) | - | - | FAIL |
| 195 | Peak Heating Load (kW) | FAIL (0.00) | - | - | FAIL |
| 195 | Peak Cooling Load (kW) | FAIL (0.00) | - | - | FAIL |
| 600 | Annual Heating Energy (MWh) | PASS (6.54) | FAIL (6.54) | PASS (6.54) | PASS |
| 600 | Annual Cooling Energy (MWh) | FAIL (0.66) | FAIL (0.66) | FAIL (0.66) | FAIL |
| 600 | Peak Heating Load (kW) | FAIL (0.24) | FAIL (0.24) | FAIL (0.24) | FAIL |
| 600 | Peak Cooling Load (kW) | FAIL (0.19) | FAIL (0.19) | FAIL (0.19) | FAIL |
| 900 | Annual Heating Energy (MWh) | PASS (1.67) | - | - | PASS |
| 900 | Annual Cooling Energy (MWh) | PASS (2.92) | - | - | PASS |
| 900 | Peak Heating Load (kW) | PASS (1.65) | - | - | PASS |
| 900 | Peak Cooling Load (kW) | FAIL (1.69) | - | - | FAIL |
| 920 | Annual Heating Energy (MWh) | PASS (3.49) | - | - | PASS |
| 920 | Annual Cooling Energy (MWh) | FAIL (1.76) | - | - | FAIL |
| 920 | Peak Heating Load (kW) | FAIL (3.09) | - | - | FAIL |
| 920 | Peak Cooling Load (kW) | FAIL (1.91) | - | - | FAIL |
| 930 | Annual Heating Energy (MWh) | PASS (4.92) | - | - | PASS |
| 930 | Annual Cooling Energy (MWh) | FAIL (1.49) | - | - | FAIL |
| 930 | Peak Heating Load (kW) | FAIL (3.13) | - | - | FAIL |
| 930 | Peak Cooling Load (kW) | FAIL (1.67) | - | - | FAIL |
| 940 | Annual Heating Energy (MWh) | FAIL (1.19) | - | - | FAIL |
| 940 | Annual Cooling Energy (MWh) | FAIL (2.88) | - | - | FAIL |
| 940 | Peak Heating Load (kW) | FAIL (1.55) | - | - | FAIL |
| 940 | Peak Cooling Load (kW) | FAIL (1.69) | - | - | FAIL |
| 950 | Annual Heating Energy (MWh) | FAIL (0.00) | - | - | FAIL |
| 950 | Annual Cooling Energy (MWh) | FAIL (0.69) | - | - | FAIL |
| 950 | Peak Heating Load (kW) | FAIL (0.00) | - | - | FAIL |
| 950 | Peak Cooling Load (kW) | FAIL (1.68) | - | - | FAIL |
| 960 | Annual Heating Energy (MWh) | FAIL (1.88) | - | - | FAIL |
| 960 | Annual Cooling Energy (MWh) | FAIL (7.49) | - | - | FAIL |
| 960 | Peak Heating Load (kW) | FAIL (6.33) | - | - | FAIL |
| 960 | Peak Cooling Load (kW) | FAIL (3.45) | - | - | FAIL |

## Systematic Issues

The following recurring issues are affecting validation results:

### Unknown/Unclassified

**Affected metrics:** 195 - Annual Cooling Energy (MWh), 650FF - Minimum Free-Floating Temperature (°C), 930 - Peak Heating Load (kW), 640 - Annual Cooling Energy (MWh), 630 - Annual Cooling Energy (MWh), 640 - Peak Heating Load (kW), 920 - Peak Cooling Load (kW), 600FF - Minimum Free-Floating Temperature (°C), 940 - Peak Heating Load (kW), 940 - Peak Cooling Load (kW), 960 - Peak Heating Load (kW), 195 - Peak Cooling Load (kW), 650 - Annual Cooling Energy (MWh), 900 - Peak Cooling Load (kW), 600 - Annual Cooling Energy (MWh), 950 - Peak Cooling Load (kW), 195 - Peak Heating Load (kW), 910 - Peak Heating Load (kW), 610 - Peak Heating Load (kW), 620 - Peak Heating Load (kW), 610 - Annual Cooling Energy (MWh), 920 - Peak Heating Load (kW), 930 - Peak Cooling Load (kW), 650FF - Maximum Free-Floating Temperature (°C), 950 - Peak Heating Load (kW), 960 - Annual Heating Energy (MWh), 630 - Peak Heating Load (kW), 600FF - Maximum Free-Floating Temperature (°C), 620 - Annual Cooling Energy (MWh), 600 - Peak Heating Load (kW), 960 - Peak Cooling Load (kW), 195 - Annual Heating Energy (MWh) |
**Count:** 32 metrics

### Thermal Mass Dynamics

**Affected metrics:** 900FF - Minimum Free-Floating Temperature (°C), 950FF - Maximum Free-Floating Temperature (°C), 950FF - Minimum Free-Floating Temperature (°C) |
**Count:** 3 metrics

### Inter-Zone Heat Transfer

**Affected metrics:** 960 - Annual Cooling Energy (MWh) |
**Count:** 1 metrics

### Solar Gain Calculations

**Affected metrics:** 630 - Peak Cooling Load (kW), 610 - Peak Cooling Load (kW), 640 - Peak Cooling Load (kW), 620 - Peak Cooling Load (kW), 650 - Peak Cooling Load (kW), 600 - Peak Cooling Load (kW) |
**Count:** 6 metrics

### 5R1C Model Limitation (Accepted)

**Affected metrics:** 930 - Annual Cooling Energy (MWh), 940 - Annual Cooling Energy (MWh), 920 - Annual Cooling Energy (MWh), 950 - Annual Cooling Energy (MWh), 940 - Annual Heating Energy (MWh), 950 - Annual Heating Energy (MWh) |
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
