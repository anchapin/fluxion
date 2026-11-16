# ASHRAE Standard 140 Validation Results

> **Scope (ADR-0026 / Issue #4275):** every figure in this document is **ASHRAE 140 §7 (Class I)** — Building Thermal Envelope and Fabric Load Tests — the only section of the standard fluxion's committed validation suite implements today. §6 is the weather-interpretation section and §12 is the envelope Class II re-run (in scope, blocked on §7); neither is represented here. Full per-section coverage: [`docs/compliance/section_coverage.md`](compliance/section_coverage.md).

*Generated: 2026-11-16 03:27 UTC* (live `fluxion validate` sweep at worker/r151-ci-hygiene `ca7d33d2`, engine = develop `d3dc30a9`, loop round 151 regeneration)

> **Context (2026-11-16):** this document is now the live validator sweep, replacing the round-37 snapshot (6.0% pass / 78.50% MAE at develop `ad88bfa9`, 2026-10-15) and re-deriving the drift baseline off its stale v1.3-era capture (14.3% pass / 51.03 MAE, 2026-08-16, 14 cases). The metric-level movement (8.3% pass / 77.47% MAE over the full 84-metric sweep, which now includes the 800/810/970 annual cases the old capture lacked) reflects the round-38..150 fidelity-first engine changes (canonical-weather binding round 41, the t_s surface advance rounds 70-71, the coupled single-N flip rounds 77-78, the rank-5 t_s sub-state round 144, among the recorded program) measured against the un-tuned ASHRAE 140 reference bands; the drift gate's own case-level criterion records zero pass-to-fail / fail-to-pass transitions across that span. The strict ±15% annual-energy gate remains the regression arbiter at its own re-recorded baseline. Per the standing fidelity direction, no validator band or engine constant was adjusted to move these figures.

## Summary

| Metric | Value |
|--------|-------|
| Total Results | 84 |
| Pass Rate | 8.3% |
| Passed | 7 |
| Warnings | 1 |
| Failed | 76 |
| Mean Absolute Error | 77.47% |
| Max Deviation | 914.15% |

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Validation Duration | 4.52 seconds |
| Throughput | 4.65 cases/sec |
| Total Cases | 21 |

## Detailed Results

### Baseline Cases (600 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 600 | 5383.44 kWh (Ref: 3993.00-4504.00) | 3828.42 kWh (Ref: 5432.00-6162.00) | 5.59 kW (Ref: 3.02-3.36) | 3.61 kW (Ref: 5.42-6.48) | ❌ FAIL |
| 610 | 6202.24 kWh (Ref: 4066.00-4592.00) | 1936.29 kWh (Ref: 4117.00-4382.00) | 5.62 kW (Ref: 3.02-3.36) | 2.83 kW (Ref: 5.33-6.43) | ❌ FAIL |
| 620 | 7443.84 kWh (Ref: 4094.00-4719.00) | 3348.23 kWh (Ref: 3841.00-4404.00) | 5.83 kW (Ref: 3.04-3.38) | 3.69 kW (Ref: 3.96-4.80) | ❌ FAIL |
| 630 | 8570.21 kWh (Ref: 4356.00-5139.00) | 2095.11 kWh (Ref: 2573.00-3074.00) | 5.88 kW (Ref: 3.04-3.39) | 3.02 kW (Ref: 3.53-4.21) | ❌ FAIL |
| 640 | 3228.99 kWh (Ref: 2403.00-2682.00) | 3828.42 kWh (Ref: 5237.00-5893.00) | 6.03 kW (Ref: 4.04-4.66) | 3.61 kW (Ref: 5.37-6.43) | ❌ FAIL |
| 650 | 0.00 kWh (Ref: 0.00-0.00) | 2605.88 kWh (Ref: 4186.00-4945.00) | 0.00 kW (Ref: 0.00-0.00) | 3.46 kW (Ref: 5.04-6.29) | ❌ FAIL |

### High-Mass Cases (900 Series)

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 900 | 777.27 kWh (Ref: 1379.00-1814.00) | 2364.83 kWh (Ref: 2267.00-2714.00) | 2.11 kW (Ref: 2.44-2.78) | 1.63 kW (Ref: 2.56-3.38) | ❌ FAIL |
| 910 | 1025.18 kWh (Ref: 1648.00-2163.00) | 1207.73 kWh (Ref: 1191.00-1490.00) | 2.14 kW (Ref: 2.47-2.80) | 1.21 kW (Ref: 2.08-2.72) | ❌ FAIL |
| 920 | 2033.90 kWh (Ref: 2956.00-3607.00) | 2423.30 kWh (Ref: 2549.00-3128.00) | 2.40 kW (Ref: 2.51-2.90) | 1.74 kW (Ref: 2.71-3.48) | ❌ FAIL |
| 930 | 2519.74 kWh (Ref: 3524.00-4384.00) | 1626.43 kWh (Ref: 1654.00-2161.00) | 2.46 kW (Ref: 2.54-2.97) | 1.45 kW (Ref: 2.33-3.05) | ❌ FAIL |
| 940 | 485.84 kWh (Ref: 863.00-1389.00) | 2352.37 kWh (Ref: 2203.00-2613.00) | 2.31 kW (Ref: 3.05-3.88) | 1.63 kW (Ref: 2.56-3.38) | ❌ FAIL |
| 950 | 0.00 kWh (Ref: 0.00-0.00) | 880.25 kWh (Ref: 586.00-707.00) | 0.00 kW (Ref: 0.00-0.00) | 1.52 kW (Ref: 2.05-2.39) | ❌ FAIL |

### Free-Floating Cases

| Case | Min Temperature | Max Temperature | Status |
|------|-----------------|-----------------|--------|
| 600FF | -18.45°C (Ref: -13.80--9.90) | 49.13°C (Ref: 62.40-68.40) | ❌ FAIL |
| 650FF | -22.73°C (Ref: -17.80--16.70) | 48.31°C (Ref: 61.10-66.80) | ❌ FAIL |
| 900FF | -4.29°C (Ref: 0.60-2.20) | 43.93°C (Ref: 43.30-46.00) | ❌ FAIL |
| 950FF | -21.16°C (Ref: -13.40--12.50) | 39.31°C (Ref: 36.10-37.10) | ❌ FAIL |

### Special Cases

| Case | Annual Heating | Annual Cooling | Peak Heating | Peak Cooling | Status |
|------|----------------|----------------|--------------|--------------|--------|
| 960 | 1677.24 kWh (Ref: 2522.00-2860.00) | 1427.56 kWh (Ref: 789.00-950.00) | 1.90 kW (Ref: 2.08-2.30) | 1.40 kW (Ref: 1.34-1.48) | ❌ FAIL |
| 195 | 4489.94 kWh (Ref: 3951.00-4217.00) | 1528.65 kWh (Ref: 592.00-712.00) | 3.56 kW (Ref: 1.79-1.80) | 1.91 kW (Ref: 0.94-1.12) | ❌ FAIL |

## Multi-Reference Comparison

| Case | Metric | EnergyPlus | ESP-r | TRNSYS | Overall |
|------|--------|------------|-------|--------|---------|
| 195 | Annual Heating Energy (kWh) | PASS (4489.94) | - | - | PASS |
| 195 | Annual Cooling Energy (kWh) | FAIL (1528.65) | - | - | FAIL |
| 195 | Peak Heating Load (kW) | FAIL (3.56) | - | - | FAIL |
| 195 | Peak Cooling Load (kW) | FAIL (1.91) | - | - | FAIL |
| 600 | Annual Heating Energy (kWh) | WARN (5383.44) | PASS (5383.44) | WARN (5383.44) | WARN |
| 600 | Annual Cooling Energy (kWh) | FAIL (3828.42) | FAIL (3828.42) | FAIL (3828.42) | FAIL |
| 600 | Peak Heating Load (kW) | FAIL (5.59) | FAIL (5.59) | FAIL (5.59) | FAIL |
| 600 | Peak Cooling Load (kW) | FAIL (3.61) | FAIL (3.61) | FAIL (3.61) | FAIL |
| 900 | Annual Heating Energy (kWh) | FAIL (777.27) | - | - | FAIL |
| 900 | Annual Cooling Energy (kWh) | WARN (2364.83) | - | - | FAIL |
| 900 | Peak Heating Load (kW) | WARN (2.11) | - | - | FAIL |
| 900 | Peak Cooling Load (kW) | FAIL (1.63) | - | - | FAIL |
| 920 | Annual Heating Energy (kWh) | FAIL (2033.90) | - | - | FAIL |
| 920 | Annual Cooling Energy (kWh) | FAIL (2423.30) | - | - | FAIL |
| 920 | Peak Heating Load (kW) | FAIL (2.40) | - | - | FAIL |
| 920 | Peak Cooling Load (kW) | FAIL (1.74) | - | - | FAIL |
| 930 | Annual Heating Energy (kWh) | FAIL (2519.74) | - | - | FAIL |
| 930 | Annual Cooling Energy (kWh) | FAIL (1626.43) | - | - | FAIL |
| 930 | Peak Heating Load (kW) | FAIL (2.46) | - | - | FAIL |
| 930 | Peak Cooling Load (kW) | FAIL (1.45) | - | - | FAIL |
| 940 | Annual Heating Energy (kWh) | FAIL (485.84) | - | - | FAIL |
| 940 | Annual Cooling Energy (kWh) | FAIL (2352.37) | - | - | FAIL |
| 940 | Peak Heating Load (kW) | FAIL (2.31) | - | - | FAIL |
| 940 | Peak Cooling Load (kW) | FAIL (1.63) | - | - | FAIL |
| 950 | Annual Heating Energy (kWh) | FAIL (0.00) | - | - | FAIL |
| 950 | Annual Cooling Energy (kWh) | FAIL (880.25) | - | - | FAIL |
| 950 | Peak Heating Load (kW) | FAIL (0.00) | - | - | FAIL |
| 950 | Peak Cooling Load (kW) | FAIL (1.52) | - | - | FAIL |
| 960 | Annual Heating Energy (kWh) | FAIL (1677.24) | - | - | FAIL |
| 960 | Annual Cooling Energy (kWh) | FAIL (1427.56) | - | - | FAIL |
| 960 | Peak Heating Load (kW) | FAIL (1.90) | - | - | FAIL |
| 960 | Peak Cooling Load (kW) | FAIL (1.40) | - | - | FAIL |
| 970 | Annual Heating Energy (kWh) | FAIL (1281.49) | - | - | FAIL |
| 970 | Annual Cooling Energy (kWh) | FAIL (1097.65) | - | - | FAIL |
| 970 | Peak Heating Load (kW) | FAIL (1.93) | - | - | FAIL |
| 970 | Peak Cooling Load (kW) | FAIL (1.05) | - | - | FAIL |

## Systematic Issues

The following recurring issues are affecting validation results:

### Inter-Zone Heat Transfer

**Affected metrics:** 960 - Annual Heating Energy (kWh), 960 - Annual Cooling Energy (kWh) |
**Count:** 2 metrics

### HVAC Load Calculation

**Affected metrics:** 620 - Peak Heating Load (kW), 195 - Peak Heating Load (kW), 800 - Annual Cooling Energy (kWh), 630 - Peak Heating Load (kW), 810 - Annual Heating Energy (kWh), 810 - Annual Cooling Energy (kWh), 640 - Peak Heating Load (kW), 610 - Peak Heating Load (kW), 600 - Peak Heating Load (kW), 195 - Peak Cooling Load (kW) |
**Count:** 10 metrics

### Solar Gain Calculations

**Affected metrics:** 600FF - Minimum Free-Floating Temperature (°C), 910 - Peak Cooling Load (kW), 620 - Peak Cooling Load (kW), 970 - Peak Cooling Load (kW), 810 - Peak Cooling Load (kW), 640 - Peak Cooling Load (kW), 940 - Peak Cooling Load (kW), 960 - Peak Cooling Load (kW), 610 - Annual Cooling Energy (kWh), 600 - Annual Cooling Energy (kWh), 630 - Peak Cooling Load (kW), 650FF - Minimum Free-Floating Temperature (°C), 900 - Peak Cooling Load (kW), 650FF - Maximum Free-Floating Temperature (°C), 930 - Peak Cooling Load (kW), 650 - Annual Cooling Energy (kWh), 600FF - Maximum Free-Floating Temperature (°C), 610 - Peak Cooling Load (kW), 920 - Peak Cooling Load (kW), 950 - Peak Cooling Load (kW), 640 - Annual Cooling Energy (kWh), 600 - Peak Cooling Load (kW), 650 - Peak Cooling Load (kW) |
**Count:** 23 metrics

### Thermal Mass Dynamics

**Affected metrics:** 900 - Peak Heating Load (kW), 970 - Peak Heating Load (kW), 960 - Peak Heating Load (kW), 900FF - Minimum Free-Floating Temperature (°C), 950FF - Maximum Free-Floating Temperature (°C), 910 - Peak Heating Load (kW), 950FF - Minimum Free-Floating Temperature (°C), 930 - Peak Heating Load (kW), 950 - Peak Heating Load (kW), 940 - Peak Heating Load (kW), 920 - Peak Heating Load (kW) |
**Count:** 11 metrics

### Unknown/Unclassified

**Affected metrics:** 900 - Annual Cooling Energy (kWh), 930 - Annual Heating Energy (kWh), 970 - Annual Cooling Energy (kWh), 900 - Annual Heating Energy (kWh), 630 - Annual Heating Energy (kWh), 950 - Annual Heating Energy (kWh), 950 - Annual Cooling Energy (kWh), 640 - Annual Heating Energy (kWh), 610 - Annual Heating Energy (kWh), 800 - Peak Cooling Load (kW), 910 - Annual Heating Energy (kWh), 970 - Annual Heating Energy (kWh), 940 - Annual Cooling Energy (kWh), 940 - Annual Heating Energy (kWh), 620 - Annual Heating Energy (kWh), 800 - Peak Heating Load (kW), 810 - Peak Heating Load (kW), 195 - Annual Cooling Energy (kWh), 920 - Annual Heating Energy (kWh), 920 - Annual Cooling Energy (kWh), 930 - Annual Cooling Energy (kWh), 630 - Annual Cooling Energy (kWh), 620 - Annual Cooling Energy (kWh) |
**Count:** 23 metrics

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

