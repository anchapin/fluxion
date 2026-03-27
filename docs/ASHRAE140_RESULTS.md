# ASHRAE Standard 140 Validation Results

*Generated: 2026-03-27 11:30 UTC*

## Session 33: Empirical Factor Removal - Baseline Revealed

**IMPORTANT**: This validation was run AFTER removing empirical corrections. The model now
shows its baseline physics without manual adjustments. Results are intentionally high
because the underlying thermal model needs physics-based fixes, not empirical patches.

### PASS RATE: 1.6% (1/64) - BASELINE REVEALED

| 9 Empirical Factors Removed This Session |
|---|
| 1. Case 960 COP=3.0 correction |
| 2. Case 960 heating_efficiency=0.9 correction |
| 3. Case 900 heating 4.0x correction |
| 4. Case 900 cooling 0.50x correction |
| 5. Case 910 heating 2.5x correction |
| 6. Case 910 cooling 0.35x correction |
| 7. Case 940 heating 2.7x correction |
| 8. Case 940/950 cooling corrections |
| 9. Engine h_tr_em (0.15, 1.05) → (1.0, 1.0) |
| 10. Engine sensitivity_correction 4.0x → 1.0 |

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 64 |
| Pass Rate | 1.6% |
| Passed | 1 |
| Warnings | 1 |
| Failed | 62 |
| Mean Absolute Error | 6.23% |
| Max Deviation | 61.06% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 0.93 seconds |
| Throughput | 19.28 cases/sec |
| Total Cases | 18 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 8.65 MWh (Ref: 5.50-7.50) | 6.53 MWh (Ref: 8.00-10.50) | 4.43 kW (Ref: 2.80-3.80) | 5.04 kW (Ref: 4.80-6.20) | ❌ FAIL |
| 610 | 9.08 MWh (Ref: 4.36-5.79) | 4.56 MWh (Ref: 3.92-6.14) | 4.43 kW (Ref: 4.30-5.70) | 4.10 kW (Ref: 2.20-2.90) | ❌ FAIL |
| 620 | 7.90 MWh (Ref: 4.50-6.50) | 2.29 MWh (Ref: 3.20-5.00) | 4.38 kW (Ref: 2.80-3.80) | 2.71 kW (Ref: 2.50-3.50) | ❌ FAIL |
| 630 | 9.04 MWh (Ref: 5.05-6.47) | 1.12 MWh (Ref: 2.13-3.70) | 4.39 kW (Ref: 4.70-6.10) | 1.80 kW (Ref: 1.80-2.40) | ❌ FAIL |
| 640 | 6.49 MWh (Ref: 2.75-3.80) | 6.41 MWh (Ref: 5.95-8.10) | 6.96 kW (Ref: 4.30-5.70) | 5.04 kW (Ref: 2.80-3.70) | ❌ FAIL |
| 650 | 0.00 MWh (Ref: 0.00-0.00) | 4.65 MWh (Ref: 4.82-7.06) | 0.00 kW (Ref: 0.00-0.00) | 6.49 kW (Ref: 1.90-2.50) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 4.75 MWh (Ref: 1.17-2.04) | 6.95 MWh (Ref: 2.13-3.67) | 2.63 kW (Ref: 1.80-2.40) | 3.47 kW (Ref: 1.60-2.10) | ❌ FAIL |
| 910 | 5.23 MWh (Ref: 1.51-2.28) | 4.83 MWh (Ref: 0.82-1.88) | 2.65 kW (Ref: 1.90-2.50) | 2.72 kW (Ref: 1.20-1.60) | ❌ FAIL |
| 920 | 4.07 MWh (Ref: 3.26-4.30) | 2.42 MWh (Ref: 1.84-3.31) | 2.32 kW (Ref: 2.10-2.80) | 1.70 kW (Ref: 1.40-1.90) | ❌ FAIL |
| 930 | 5.26 MWh (Ref: 4.14-5.34) | 1.04 MWh (Ref: 1.04-2.24) | 2.42 kW (Ref: 2.30-3.00) | 1.06 kW (Ref: 1.10-1.50) | ❌ FAIL |
| 940 | 4.14 MWh (Ref: 0.79-1.41) | 6.95 MWh (Ref: 2.08-3.55) | 3.88 kW (Ref: 1.90-2.50) | 3.47 kW (Ref: 1.70-2.30) | ❌ FAIL |
| 950 | 0.00 MWh (Ref: 0.00-0.00) | 2.73 MWh (Ref: 0.39-0.92) | 0.00 kW (Ref: 0.00-0.00) | 5.14 kW (Ref: 0.70-0.90) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -6.70°C (Ref: -18.80--15.60) | 38.88°C (Ref: 64.90-75.10) | ❌ FAIL |
| 650FF | -10.85°C (Ref: -23.00--21.00) | 37.11°C (Ref: 63.20-73.50) | ❌ FAIL |
| 900FF | -3.51°C (Ref: -6.40--1.60) | 38.03°C (Ref: 41.80-46.40) | ❌ FAIL |
| 950FF | -9.46°C (Ref: -20.20--17.80) | 31.75°C (Ref: 35.50-38.50) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 0.91 MWh (Ref: 5.00-15.00) | 4.22 MWh (Ref: 1.00-3.50) | 100.00 kW (Ref: 2.00-8.00) | 3.61 kW (Ref: 0.00-4.00) | ❌ FAIL |
| 195 | 4.85 MWh (Ref: 3.50-6.00) | 0.00 MWh (Ref: 0.00-0.00) | 1.64 kW (Ref: 1.40-2.20) | 0.00 kW (Ref: 0.00-0.00) | ❌ FAIL |

## Multi-Reference Comparison

| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |
|------|--------|------------|-------|--------|---------|
| 600 | Annual Heating (MWh) | FAIL (8.65) | - | - | FAIL |
| 600 | Annual Cooling (MWh) | FAIL (6.53) | - | - | FAIL |
| 600 | Peak Heating (kW) | FAIL (4.43) | - | - | FAIL |
| 600 | Peak Cooling (kW) | PASS (5.04) | - | - | PASS |

## Systematic Issues

The following recurring issues are affecting validation results:

### Thermal Mass Dynamics

**Affected metrics:** 950FF - Min Free-Float Temp (°C), 950FF - Max Free-Float Temp (°C), 900FF - Max Free-Float Temp (°C) |
**Count:** 3 metrics

### Inter-Zone Heat Transfer

**Affected metrics:** 960 - Annual Cooling (MWh) |
**Count:** 1 metrics

### Solar Gain Calculations

**Affected metrics:** 610 - Peak Cooling (kW), 650 - Peak Cooling (kW), 620 - Peak Cooling (kW), 640 - Peak Cooling (kW), 630 - Peak Cooling (kW) |
**Count:** 5 metrics

### Unknown/Unclassified

**Affected metrics:** 960 - Annual Heating (MWh), 630 - Annual Cooling (MWh), 650 - Annual Cooling (MWh), 910 - Peak Cooling (kW), 600 - Annual Cooling (MWh), 910 - Peak Heating (kW), 195 - Peak Heating (kW), 600FF - Min Free-Float Temp (°C), 640 - Peak Heating (kW), 940 - Peak Cooling (kW), 650 - Peak Heating (kW), 940 - Peak Heating (kW), 950 - Peak Cooling (kW), 630 - Annual Heating (MWh), 620 - Peak Heating (kW), 950 - Peak Heating (kW), 600FF - Max Free-Float Temp (°C), 620 - Annual Heating (MWh), 640 - Annual Heating (MWh), 195 - Annual Cooling (MWh), 640 - Annual Cooling (MWh), 650FF - Min Free-Float Temp (°C), 610 - Annual Heating (MWh), 930 - Peak Cooling (kW), 960 - Peak Heating (kW), 600 - Annual Heating (MWh), 930 - Peak Heating (kW), 920 - Peak Heating (kW), 630 - Peak Heating (kW), 620 - Annual Cooling (MWh), 900 - Peak Heating (kW), 610 - Peak Heating (kW), 650 - Annual Heating (MWh), 960 - Peak Cooling (kW), 650FF - Max Free-Float Temp (°C), 900 - Peak Cooling (kW), 920 - Peak Cooling (kW), 195 - Peak Cooling (kW), 195 - Annual Heating (MWh), 600 - Peak Heating (kW), 610 - Annual Cooling (MWh) |
**Count:** 41 metrics

### 5R1C Model Limitation (Accepted)

**Affected metrics:** 920 - Annual Heating (MWh), 920 - Annual Cooling (MWh), 930 - Annual Cooling (MWh), 940 - Annual Heating (MWh), 900 - Annual Cooling (MWh), 910 - Annual Heating (MWh), 940 - Annual Cooling (MWh), 900 - Annual Heating (MWh), 930 - Annual Heating (MWh), 950 - Annual Heating (MWh), 950 - Annual Cooling (MWh), 910 - Annual Cooling (MWh) |
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
