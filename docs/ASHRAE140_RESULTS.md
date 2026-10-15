# ASHRAE Standard 140 Validation Results

*Generated: 2026-10-15 21:37 UTC* (live `fluxion validate` sweep at develop `ad88bfa9`, round 37 regeneration)

> **Context (2026-10-15):** this document is now the live validator sweep, replacing the stored 2026-09-26 snapshot (9.78% pass / 45.27% MAE). The metric-level movement (6.0% pass / 78.50% MAE) reflects the round-26+ fidelity-first engine changes measured against the un-tuned ASHRAE 140 reference bands; the strict ±15% annual-energy gate at the same tip is 9 PASS / 33 KNOWN-FAIL / 0 REGRESSION with zero regressions vs `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`. Per the standing fidelity direction, no validator band or engine constant was adjusted to move these figures.

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 84 |
| Pass Rate | 6.0% |
| Passed | 5 |
| Warnings | 9 |
| Failed | 70 |
| Mean Absolute Error | 78.50% |
| Max Deviation | 964.17% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 0.23 seconds |
| Throughput | 90.76 cases/sec |
| Total Cases | 21 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 6358.76 kWh (Ref: 3993.00-4504.00) | 4885.93 kWh (Ref: 5432.00-6162.00) | 5.85 kW (Ref: 3.02-3.36) | 4.44 kW (Ref: 5.42-6.48) | ❌ FAIL |
| 610 | 7063.31 kWh (Ref: 4066.00-4592.00) | 2440.52 kWh (Ref: 4117.00-4382.00) | 5.88 kW (Ref: 3.02-3.36) | 2.89 kW (Ref: 5.33-6.43) | ❌ FAIL |
| 620 | 7715.29 kWh (Ref: 4094.00-4719.00) | 3672.41 kWh (Ref: 3841.00-4404.00) | 5.93 kW (Ref: 3.04-3.38) | 4.23 kW (Ref: 3.96-4.80) | ❌ FAIL |
| 630 | 8484.08 kWh (Ref: 4356.00-5139.00) | 2458.99 kWh (Ref: 2573.00-3074.00) | 5.94 kW (Ref: 3.04-3.39) | 3.39 kW (Ref: 3.53-4.21) | ❌ FAIL |
| 640 | 3747.00 kWh (Ref: 2403.00-2682.00) | 4869.05 kWh (Ref: 5237.00-5893.00) | 6.30 kW (Ref: 4.04-4.66) | 4.43 kW (Ref: 5.37-6.43) | ❌ FAIL |
| 650 | 0.00 kWh (Ref: 0.00-0.00) | 3954.98 kWh (Ref: 4186.00-4945.00) | 0.00 kW (Ref: 0.00-0.00) | 4.20 kW (Ref: 5.04-6.29) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 1183.76 kWh (Ref: 1379.00-1814.00) | 1996.20 kWh (Ref: 2267.00-2714.00) | 2.42 kW (Ref: 2.44-2.78) | 1.64 kW (Ref: 2.56-3.38) | ❌ FAIL |
| 910 | 1702.83 kWh (Ref: 1648.00-2163.00) | 1160.10 kWh (Ref: 1191.00-1490.00) | 2.50 kW (Ref: 2.47-2.80) | 1.23 kW (Ref: 2.08-2.72) | ❌ FAIL |
| 920 | 2537.51 kWh (Ref: 2956.00-3607.00) | 2132.09 kWh (Ref: 2549.00-3128.00) | 2.70 kW (Ref: 2.51-2.90) | 1.74 kW (Ref: 2.71-3.48) | ❌ FAIL |
| 930 | 2944.23 kWh (Ref: 3524.00-4384.00) | 1537.26 kWh (Ref: 1654.00-2161.00) | 2.73 kW (Ref: 2.54-2.97) | 1.48 kW (Ref: 2.33-3.05) | ❌ FAIL |
| 940 | 742.31 kWh (Ref: 863.00-1389.00) | 1983.53 kWh (Ref: 2203.00-2613.00) | 2.63 kW (Ref: 3.05-3.88) | 1.64 kW (Ref: 2.56-3.38) | ❌ FAIL |
| 950 | 0.00 kWh (Ref: 0.00-0.00) | 778.11 kWh (Ref: 586.00-707.00) | 0.00 kW (Ref: 0.00-0.00) | 1.51 kW (Ref: 2.05-2.39) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -19.90°C (Ref: -13.80--9.90) | 54.10°C (Ref: 62.40-68.40) | ❌ FAIL |
| 650FF | -23.71°C (Ref: -17.80--16.70) | 52.76°C (Ref: 61.10-66.80) | ❌ FAIL |
| 900FF | -6.65°C (Ref: 0.60-2.20) | 42.71°C (Ref: 43.30-46.00) | ❌ FAIL |
| 950FF | -21.13°C (Ref: -13.40--12.50) | 38.63°C (Ref: 36.10-37.10) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 2329.96 kWh (Ref: 2522.00-2860.00) | 1192.19 kWh (Ref: 789.00-950.00) | 2.11 kW (Ref: 2.08-2.30) | 1.37 kW (Ref: 1.34-1.48) | ❌ FAIL |
| 195 | 1858.18 kWh (Ref: 3951.00-4217.00) | 767.63 kWh (Ref: 592.00-712.00) | 1.49 kW (Ref: 1.79-1.80) | 1.19 kW (Ref: 0.94-1.12) | ❌ FAIL |

## Multi-Reference Comparison

| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |
|------|--------|------------|-------|--------|---------|
| 195 | Annual Heating Energy (kWh) | FAIL (1858.18) | - | - | FAIL |
| 195 | Annual Cooling Energy (kWh) | FAIL (767.63) | - | - | FAIL |
| 195 | Peak Heating Load (kW) | WARN (1.49) | - | - | FAIL |
| 195 | Peak Cooling Load (kW) | FAIL (1.19) | - | - | FAIL |
| 600 | Annual Heating Energy (kWh) | PASS (6358.76) | FAIL (6358.76) | PASS (6358.76) | PASS |
| 600 | Annual Cooling Energy (kWh) | FAIL (4885.93) | FAIL (4885.93) | FAIL (4885.93) | FAIL |
| 600 | Peak Heating Load (kW) | FAIL (5.85) | FAIL (5.85) | FAIL (5.85) | FAIL |
| 600 | Peak Cooling Load (kW) | FAIL (4.44) | WARN (4.44) | FAIL (4.44) | FAIL |
| 900 | Annual Heating Energy (kWh) | WARN (1183.76) | - | - | FAIL |
| 900 | Annual Cooling Energy (kWh) | FAIL (1996.20) | - | - | FAIL |
| 900 | Peak Heating Load (kW) | FAIL (2.42) | - | - | FAIL |
| 900 | Peak Cooling Load (kW) | FAIL (1.64) | - | - | FAIL |
| 920 | Annual Heating Energy (kWh) | FAIL (2537.51) | - | - | FAIL |
| 920 | Annual Cooling Energy (kWh) | FAIL (2132.09) | - | - | FAIL |
| 920 | Peak Heating Load (kW) | FAIL (2.70) | - | - | FAIL |
| 920 | Peak Cooling Load (kW) | FAIL (1.74) | - | - | FAIL |
| 930 | Annual Heating Energy (kWh) | FAIL (2944.23) | - | - | FAIL |
| 930 | Annual Cooling Energy (kWh) | FAIL (1537.26) | - | - | FAIL |
| 930 | Peak Heating Load (kW) | FAIL (2.73) | - | - | FAIL |
| 930 | Peak Cooling Load (kW) | FAIL (1.48) | - | - | FAIL |
| 940 | Annual Heating Energy (kWh) | FAIL (742.31) | - | - | FAIL |
| 940 | Annual Cooling Energy (kWh) | FAIL (1983.53) | - | - | FAIL |
| 940 | Peak Heating Load (kW) | FAIL (2.63) | - | - | FAIL |
| 940 | Peak Cooling Load (kW) | FAIL (1.64) | - | - | FAIL |
| 950 | Annual Heating Energy (kWh) | FAIL (0.00) | - | - | FAIL |
| 950 | Annual Cooling Energy (kWh) | FAIL (778.11) | - | - | FAIL |
| 950 | Peak Heating Load (kW) | FAIL (0.00) | - | - | FAIL |
| 950 | Peak Cooling Load (kW) | FAIL (1.51) | - | - | FAIL |
| 960 | Annual Heating Energy (kWh) | FAIL (2329.96) | - | - | FAIL |
| 960 | Annual Cooling Energy (kWh) | FAIL (1192.19) | - | - | FAIL |
| 960 | Peak Heating Load (kW) | FAIL (2.11) | - | - | FAIL |
| 960 | Peak Cooling Load (kW) | FAIL (1.37) | - | - | FAIL |
| 970 | Annual Heating Energy (kWh) | FAIL (1642.95) | - | - | FAIL |
| 970 | Annual Cooling Energy (kWh) | FAIL (1046.19) | - | - | FAIL |
| 970 | Peak Heating Load (kW) | FAIL (2.19) | - | - | FAIL |
| 970 | Peak Cooling Load (kW) | FAIL (1.12) | - | - | FAIL |

## Systematic Issues

The following recurring issues are affecting validation results:

### HVAC Load Calculation

**Affected metrics:** 610 - Peak Heating Load (kW), 195 - Peak Heating Load (kW), 800 - Annual Cooling Energy (kWh), 600 - Peak Heating Load (kW), 810 - Annual Heating Energy (kWh), 800 - Annual Heating Energy (kWh), 640 - Peak Heating Load (kW), 810 - Annual Cooling Energy (kWh), 630 - Peak Heating Load (kW), 195 - Peak Cooling Load (kW), 620 - Peak Heating Load (kW) |
**Count:** 11 metrics

### Solar Gain Calculations

**Affected metrics:** 600 - Peak Cooling Load (kW), 640 - Peak Cooling Load (kW), 920 - Peak Cooling Load (kW), 610 - Annual Cooling Energy (kWh), 960 - Peak Cooling Load (kW), 600FF - Maximum Free-Floating Temperature (°C), 940 - Peak Cooling Load (kW), 950 - Peak Cooling Load (kW), 900 - Peak Cooling Load (kW), 910 - Peak Cooling Load (kW), 650 - Peak Cooling Load (kW), 650FF - Maximum Free-Floating Temperature (°C), 600 - Annual Cooling Energy (kWh), 600FF - Minimum Free-Floating Temperature (°C), 610 - Peak Cooling Load (kW), 810 - Peak Cooling Load (kW), 970 - Peak Cooling Load (kW), 195 - Annual Heating Energy (kWh), 650FF - Minimum Free-Floating Temperature (°C), 930 - Peak Cooling Load (kW) |
**Count:** 20 metrics

### Thermal Mass Dynamics

**Affected metrics:** 940 - Peak Heating Load (kW), 900FF - Minimum Free-Floating Temperature (°C), 960 - Peak Heating Load (kW), 950 - Peak Heating Load (kW), 950FF - Minimum Free-Floating Temperature (°C), 900 - Peak Heating Load (kW), 920 - Peak Heating Load (kW), 970 - Peak Heating Load (kW), 930 - Peak Heating Load (kW) |
**Count:** 9 metrics

### Inter-Zone Heat Transfer

**Affected metrics:** 960 - Annual Heating Energy (kWh), 960 - Annual Cooling Energy (kWh) |
**Count:** 2 metrics

### Unknown/Unclassified

**Affected metrics:** 940 - Annual Cooling Energy (kWh), 800 - Peak Heating Load (kW), 900 - Annual Heating Energy (kWh), 800 - Peak Cooling Load (kW), 195 - Annual Cooling Energy (kWh), 950 - Annual Cooling Energy (kWh), 970 - Annual Heating Energy (kWh), 640 - Annual Cooling Energy (kWh), 650 - Annual Cooling Energy (kWh), 940 - Annual Heating Energy (kWh), 640 - Annual Heating Energy (kWh), 620 - Annual Heating Energy (kWh), 970 - Annual Cooling Energy (kWh), 630 - Annual Heating Energy (kWh), 920 - Annual Heating Energy (kWh), 900 - Annual Cooling Energy (kWh), 930 - Annual Heating Energy (kWh), 920 - Annual Cooling Energy (kWh), 610 - Annual Heating Energy (kWh), 930 - Annual Cooling Energy (kWh), 950 - Annual Heating Energy (kWh) |
**Count:** 21 metrics

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
