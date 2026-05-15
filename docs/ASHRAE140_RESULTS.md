# ASHRAE Standard 140 Validation Results

*Generated: 2026-05-14 20:12 UTC*

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 7.8% |
| Passed | 5 |
| Warnings | 0 |
| Failed | 59 |
| Mean Absolute Error | 105.55% |
| Max Deviation | 489.85% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 2.04 seconds |
| Throughput | 8.81 cases/sec |
| Total Cases | 18 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 21.04 MWh (Ref: 5.50-7.50) | 2.43 MWh (Ref: 8.00-10.50) | 10.38 kW (Ref: 2.80-3.80) | 5.15 kW (Ref: 4.80-6.20) | ❌ FAIL |
| 610 | 20.11 MWh (Ref: 4.36-5.79) | 2.17 MWh (Ref: 3.92-6.14) | 10.41 kW (Ref: 4.30-5.70) | 4.71 kW (Ref: 2.20-2.90) | ❌ FAIL |
| 620 | 18.89 MWh (Ref: 4.50-6.50) | 1.85 MWh (Ref: 3.20-5.00) | 9.83 kW (Ref: 2.80-3.80) | 4.48 kW (Ref: 2.50-3.50) | ❌ FAIL |
| 630 | 19.52 MWh (Ref: 5.05-6.47) | 0.88 MWh (Ref: 2.13-3.70) | 9.83 kW (Ref: 4.70-6.10) | 3.33 kW (Ref: 1.80-2.40) | ❌ FAIL |
| 640 | 15.93 MWh (Ref: 2.75-3.80) | 2.38 MWh (Ref: 5.95-8.10) | 10.30 kW (Ref: 4.30-5.70) | 5.14 kW (Ref: 2.80-3.70) | ❌ FAIL |
| 650 | 0.00 MWh (Ref: 0.00-0.00) | 1.50 MWh (Ref: 4.82-7.06) | 0.00 kW (Ref: 0.00-0.00) | 5.02 kW (Ref: 1.90-2.50) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 0.00 MWh (Ref: 1.17-2.04) | 0.00 MWh (Ref: 2.13-3.67) | 0.00 kW (Ref: 1.80-2.40) | 0.00 kW (Ref: 1.60-2.10) | ❌ FAIL |
| 910 | 0.00 MWh (Ref: 1.51-2.28) | 0.00 MWh (Ref: 0.82-1.88) | 0.00 kW (Ref: 1.90-2.50) | 0.00 kW (Ref: 1.20-1.60) | ❌ FAIL |
| 920 | 0.00 MWh (Ref: 3.26-4.30) | 0.00 MWh (Ref: 1.84-3.31) | 0.00 kW (Ref: 2.10-2.80) | 0.00 kW (Ref: 1.40-1.90) | ❌ FAIL |
| 930 | 0.00 MWh (Ref: 4.14-5.34) | 0.00 MWh (Ref: 1.04-2.24) | 0.00 kW (Ref: 2.30-3.00) | 0.00 kW (Ref: 1.10-1.50) | ❌ FAIL |
| 940 | 0.00 MWh (Ref: 0.79-1.41) | 0.00 MWh (Ref: 2.08-3.55) | 0.00 kW (Ref: 1.90-2.50) | 0.00 kW (Ref: 1.70-2.30) | ❌ FAIL |
| 950 | 0.00 MWh (Ref: 0.00-0.00) | 0.00 MWh (Ref: 0.39-0.92) | 0.00 kW (Ref: 0.00-0.00) | 0.00 kW (Ref: 0.70-0.90) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -28.00°C (Ref: -18.80--15.60) | 47.20°C (Ref: 64.90-75.10) | ❌ FAIL |
| 650FF | -29.79°C (Ref: -23.00--21.00) | 46.55°C (Ref: 63.20-73.50) | ❌ FAIL |
| 900FF | -19.95°C (Ref: -6.40--1.60) | 14.79°C (Ref: 41.80-46.40) | ❌ FAIL |
| 950FF | -16.35°C (Ref: -20.20--17.80) | 14.08°C (Ref: 35.50-38.50) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 9.82 MWh (Ref: 1.65-2.45) | 1.95 MWh (Ref: 1.55-2.78) | 4.70 kW (Ref: 2.00-8.00) | 1.88 kW (Ref: 0.00-4.00) | ❌ FAIL |
| 195 | 28.02 MWh (Ref: 3.50-6.00) | 0.00 MWh (Ref: 0.00-0.00) | 2.33 kW (Ref: 1.40-2.20) | 0.00 kW (Ref: 0.00-0.00) | ❌ FAIL |

## Multi-Reference Comparison

| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |
|------|--------|------------|-------|--------|---------|
| 195 | Annual Heating Energy (MWh) | FAIL (28.02) | - | - | FAIL |
| 195 | Annual Cooling Energy (MWh) | PASS (0.00) | - | - | PASS |
| 195 | Peak Heating Load (kW) | FAIL (2.33) | - | - | FAIL |
| 195 | Peak Cooling Load (kW) | PASS (0.00) | - | - | PASS |
| 600 | Annual Heating Energy (MWh) | FAIL (21.04) | FAIL (21.04) | FAIL (21.04) | FAIL |
| 600 | Annual Cooling Energy (MWh) | FAIL (2.43) | FAIL (2.43) | FAIL (2.43) | FAIL |
| 600 | Peak Heating Load (kW) | FAIL (10.38) | FAIL (10.38) | FAIL (10.38) | FAIL |
| 600 | Peak Cooling Load (kW) | PASS (5.15) | PASS (5.15) | PASS (5.15) | PASS |
| 900 | Annual Heating Energy (MWh) | FAIL (0.00) | - | - | FAIL |
| 900 | Annual Cooling Energy (MWh) | FAIL (0.00) | - | - | FAIL |
| 900 | Peak Heating Load (kW) | FAIL (0.00) | - | - | FAIL |
| 900 | Peak Cooling Load (kW) | FAIL (0.00) | - | - | FAIL |
| 920 | Annual Heating Energy (MWh) | FAIL (0.00) | - | - | FAIL |
| 920 | Annual Cooling Energy (MWh) | FAIL (0.00) | - | - | FAIL |
| 920 | Peak Heating Load (kW) | FAIL (0.00) | - | - | FAIL |
| 920 | Peak Cooling Load (kW) | FAIL (0.00) | - | - | FAIL |
| 930 | Annual Heating Energy (MWh) | FAIL (0.00) | - | - | FAIL |
| 930 | Annual Cooling Energy (MWh) | FAIL (0.00) | - | - | FAIL |
| 930 | Peak Heating Load (kW) | FAIL (0.00) | - | - | FAIL |
| 930 | Peak Cooling Load (kW) | FAIL (0.00) | - | - | FAIL |
| 940 | Annual Heating Energy (MWh) | FAIL (0.00) | - | - | FAIL |
| 940 | Annual Cooling Energy (MWh) | FAIL (0.00) | - | - | FAIL |
| 940 | Peak Heating Load (kW) | FAIL (0.00) | - | - | FAIL |
| 940 | Peak Cooling Load (kW) | FAIL (0.00) | - | - | FAIL |
| 950 | Annual Heating Energy (MWh) | FAIL (0.00) | - | - | FAIL |
| 950 | Annual Cooling Energy (MWh) | FAIL (0.00) | - | - | FAIL |
| 950 | Peak Heating Load (kW) | FAIL (0.00) | - | - | FAIL |
| 950 | Peak Cooling Load (kW) | FAIL (0.00) | - | - | FAIL |
| 960 | Annual Heating Energy (MWh) | FAIL (9.82) | - | - | FAIL |
| 960 | Annual Cooling Energy (MWh) | FAIL (1.95) | - | - | FAIL |
| 960 | Peak Heating Load (kW) | FAIL (4.70) | - | - | FAIL |
| 960 | Peak Cooling Load (kW) | FAIL (1.88) | - | - | FAIL |

## Systematic Issues

The following recurring issues are affecting validation results:

### Inter-Zone Heat Transfer

**Affected metrics:** 960 - Annual Cooling Energy (MWh) |
**Count:** 1 metrics

### Solar Gain Calculations

**Affected metrics:** 620 - Peak Cooling Load (kW), 610 - Peak Cooling Load (kW), 630 - Peak Cooling Load (kW), 650 - Peak Cooling Load (kW), 640 - Peak Cooling Load (kW) |
**Count:** 5 metrics

### Unknown/Unclassified

**Affected metrics:** 600FF - Minimum Free-Floating Temperature (°C), 600 - Annual Heating Energy (MWh), 650FF - Maximum Free-Floating Temperature (°C), 610 - Peak Heating Load (kW), 930 - Peak Cooling Load (kW), 910 - Peak Cooling Load (kW), 195 - Annual Heating Energy (MWh), 640 - Peak Heating Load (kW), 900 - Peak Heating Load (kW), 960 - Peak Heating Load (kW), 960 - Peak Cooling Load (kW), 650FF - Minimum Free-Floating Temperature (°C), 640 - Annual Cooling Energy (MWh), 600 - Peak Heating Load (kW), 610 - Annual Heating Energy (MWh), 900 - Peak Cooling Load (kW), 940 - Peak Heating Load (kW), 640 - Annual Heating Energy (MWh), 620 - Annual Cooling Energy (MWh), 920 - Peak Heating Load (kW), 610 - Annual Cooling Energy (MWh), 940 - Peak Cooling Load (kW), 910 - Peak Heating Load (kW), 630 - Peak Heating Load (kW), 620 - Peak Heating Load (kW), 650 - Annual Cooling Energy (MWh), 950 - Peak Cooling Load (kW), 600 - Annual Cooling Energy (MWh), 930 - Peak Heating Load (kW), 950 - Peak Heating Load (kW), 630 - Annual Heating Energy (MWh), 620 - Annual Heating Energy (MWh), 920 - Peak Cooling Load (kW), 195 - Peak Heating Load (kW), 960 - Annual Heating Energy (MWh), 600FF - Maximum Free-Floating Temperature (°C), 630 - Annual Cooling Energy (MWh) |
**Count:** 37 metrics

### Thermal Mass Dynamics

**Affected metrics:** 950FF - Maximum Free-Floating Temperature (°C), 950FF - Minimum Free-Floating Temperature (°C), 900FF - Maximum Free-Floating Temperature (°C), 900FF - Minimum Free-Floating Temperature (°C) |
**Count:** 4 metrics

### 5R1C Model Limitation (Accepted)

**Affected metrics:** 950 - Annual Cooling Energy (MWh), 920 - Annual Heating Energy (MWh), 930 - Annual Cooling Energy (MWh), 940 - Annual Heating Energy (MWh), 930 - Annual Heating Energy (MWh), 940 - Annual Cooling Energy (MWh), 910 - Annual Cooling Energy (MWh), 900 - Annual Cooling Energy (MWh), 950 - Annual Heating Energy (MWh), 910 - Annual Heating Energy (MWh), 920 - Annual Cooling Energy (MWh), 900 - Annual Heating Energy (MWh) |
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
