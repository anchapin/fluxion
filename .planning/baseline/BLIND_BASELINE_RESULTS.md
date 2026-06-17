# ASHRAE 140 Blind Validation Baseline Results

**Phase:** A.2 - Measure True Baseline Failure State
**Plan:** A-02
**Date:** 2025-05-05
**Mode:** All corrections disabled (raw simulation only)

## Executive Summary

When all empirical corrections are disabled, the model produces significantly worse results than the ASHRAE 140 reference values. This confirms that **current "good" pass rates (~9-12%) depend entirely on post-simulation correction factors**.

| Metric | Value |
|--------|-------|
| Total metrics evaluated | 58 |
| Passed | 7 (12.07%) |
| Failed | 51 (87.93%) |
| Mean Absolute Error | 162.58% |

## Per-Case Results

### Low Mass Cases (600 Series)

| Case | Metric | Simulated | Reference Range | % Error | Status |
|------|--------|-----------|-----------------|---------|--------|
| 600 | AnnualHeating | 7.6069 MWh | 5.50-7.50 | 38.51% | FAIL |
| 600 | AnnualCooling | 10.7829 MWh | 8.00-10.50 | 16.50% | FAIL |
| 600 | PeakHeating | 3.4283 kW | 2.80-3.80 | 21.77% | FAIL |
| 600 | PeakCooling | 6.0277 kW | 4.80-6.20 | 24.51% | FAIL |
| 610 | AnnualHeating | 6.6399 MWh | 4.36-5.79 | 51.21% | FAIL |
| 610 | AnnualCooling | 8.1974 MWh | 3.92-6.14 | 63.80% | FAIL |
| 610 | PeakHeating | 3.4283 kW | 4.30-5.70 | 48.02% | FAIL |
| 610 | PeakCooling | 6.0277 kW | 2.20-2.90 | 174.25% | FAIL |
| 620 | AnnualHeating | 7.6069 MWh | 4.50-6.50 | 44.74% | FAIL |
| 620 | AnnualCooling | 10.7829 MWh | 3.20-5.00 | 231.63% | FAIL |
| 620 | PeakHeating | 3.4283 kW | 2.80-3.80 | 21.77% | FAIL |
| 620 | PeakCooling | 6.0277 kW | 2.50-3.50 | 151.11% | FAIL |
| 630 | AnnualHeating | 9.1554 MWh | 5.05-6.47 | 58.95% | FAIL |
| 630 | AnnualCooling | 5.3288 MWh | 2.13-3.70 | 82.81% | FAIL |
| 630 | PeakHeating | 4.0186 kW | 4.70-6.10 | 25.58% | FAIL |
| 630 | PeakCooling | 5.9960 kW | 1.80-2.40 | 185.52% | FAIL |
| 640 | AnnualHeating | 6.3292 MWh | 2.75-3.80 | 93.26% | FAIL |
| 640 | AnnualCooling | 8.0122 MWh | 5.95-8.10 | 14.05% | PASS |
| 640 | PeakHeating | 4.0234 kW | 4.30-5.70 | 19.53% | FAIL |
| 640 | PeakCooling | 6.8480 kW | 2.80-3.70 | 110.71% | FAIL |
| 650 | AnnualCooling | 7.2674 MWh | 4.82-7.06 | 22.35% | FAIL |
| 650 | PeakCooling | 6.7750 kW | 1.90-2.50 | 207.95% | FAIL |
| 600FF | MinFreeFloat | -9.10°C | -18.80 to -15.60 | 47.08% | FAIL |
| 600FF | MaxFreeFloat | 71.61°C | 64.90-75.10 | 2.30% | PASS |
| 650FF | MinFreeFloat | -11.76°C | -23.00 to -21.00 | 46.53% | FAIL |
| 650FF | MaxFreeFloat | 70.76°C | 63.20-73.50 | 3.53% | PASS |

### High Mass Cases (900 Series)

| Case | Metric | Simulated | Reference Range | % Error | Status |
|------|--------|-----------|-----------------|---------|--------|
| 900 | AnnualHeating | 8.8575 MWh | 1.17-2.04 | 451.87% | FAIL |
| 900 | AnnualCooling | 9.9533 MWh | 2.13-3.67 | 243.22% | FAIL |
| 900 | PeakHeating | 3.8674 kW | 1.80-2.40 | 84.16% | FAIL |
| 900 | PeakCooling | 8.2501 kW | 1.60-2.10 | 345.95% | FAIL |
| 910 | AnnualHeating | 9.3311 MWh | 1.51-2.28 | 392.41% | FAIL |
| 910 | AnnualCooling | 6.5006 MWh | 0.82-1.88 | 381.53% | FAIL |
| 910 | PeakHeating | 3.8758 kW | 1.90-2.50 | 76.17% | FAIL |
| 910 | PeakCooling | 5.0332 kW | 1.20-1.60 | 259.52% | FAIL |
| 920 | AnnualHeating | 8.8219 MWh | 3.26-4.30 | 133.38% | FAIL |
| 920 | AnnualCooling | 9.4196 MWh | 1.84-3.31 | 265.81% | FAIL |
| 920 | PeakHeating | 3.8660 kW | 2.10-2.80 | 57.80% | FAIL |
| 920 | PeakCooling | 7.7603 kW | 1.40-1.90 | 370.32% | FAIL |
| 930 | AnnualHeating | 8.9648 MWh | 4.14-5.34 | 89.13% | FAIL |
| 930 | AnnualCooling | 6.5775 MWh | 1.04-2.24 | 301.07% | FAIL |
| 930 | PeakHeating | 3.8680 kW | 2.30-3.00 | 45.96% | FAIL |
| 930 | PeakCooling | 7.6742 kW | 1.10-1.50 | 490.33% | FAIL |
| 940 | AnnualHeating | 6.2265 MWh | 0.79-1.41 | 466.04% | FAIL |
| 940 | AnnualCooling | 9.8293 MWh | 2.08-3.55 | 249.17% | FAIL |
| 940 | PeakHeating | 3.7241 kW | 1.90-2.50 | 69.28% | FAIL |
| 940 | PeakCooling | 8.2501 kW | 1.70-2.30 | 312.51% | FAIL |
| 950 | AnnualCooling | 9.4635 MWh | 0.39-0.92 | 1344.81% | FAIL |
| 950 | PeakCooling | 8.2238 kW | 0.70-0.90 | 927.98% | FAIL |
| 900FF | MinFreeFloat | -8.10°C | -6.40 to -1.60 | 102.58% | FAIL |
| 900FF | MaxFreeFloat | 81.27°C | 41.80-46.40 | 84.28% | FAIL |
| 950FF | MinFreeFloat | -11.08°C | -20.20 to -17.80 | 41.69% | FAIL |
| 950FF | MaxFreeFloat | 80.52°C | 35.50-38.50 | 117.63% | FAIL |

### Special Cases

| Case | Metric | Simulated | Reference Range | % Error | Status |
|------|--------|-----------|-----------------|---------|--------|
| 960 | AnnualHeating | 2.8859 MWh | 1.65-2.45 | 40.78% | FAIL |
| 960 | AnnualCooling | 2.9216 MWh | 1.55-2.78 | 34.95% | FAIL |
| 960 | PeakHeating | 0.9866 kW | 2.00-8.00 | 80.27% | FAIL |
| 960 | PeakCooling | 1.7391 kW | 0.00-4.00 | 13.04% | PASS |
| 195 | AnnualHeating | 7.0022 MWh | 3.50-6.00 | 47.41% | FAIL |
| 195 | PeakHeating | 1.4521 kW | 1.40-2.20 | 19.33% | PASS |

## Failure Magnitude Analysis

### Worst Offenders (by category)

**Heating Over-Prediction:**
- Case 900 Annual Heating: 451.87% error (simulated: 8.86 MWh, ref: 1.17-2.04 MWh)
- Case 940 Annual Heating: 466.04% error
- Case 910 Annual Heating: 392.41% error

**Cooling Over-Prediction:**
- Case 950 Annual Cooling: 1344.81% error (simulated: 9.46 MWh, ref: 0.39-0.92 MWh)
- Case 950 Peak Cooling: 927.98% error
- Case 930 Peak Cooling: 490.33% error

**Free-Float Temperature:**
- Case 900FF Max: 84.28% error (81°C vs 41.8-46.4°C reference)
- Case 950FF Max: 117.63% error
- Case 900FF Min: 102.58% error

### Failure Patterns by Category

| Category | Failures | Key Issues |
|----------|----------|------------|
| low-mass | 19 | Systematic heating/cooling over-prediction |
| high-mass | 25 | Severe over-prediction in both heating and cooling |
| free-float | 6 | Max temp way too high (thermal mass coupling issue) |
| special | 1 | Case 195 baseline heating deviation |

## Comparison: With Corrections vs Without

The issue notes indicate the "WITH corrections" state shows:
- Pass rate: 9.4% (6/64)
- Mean Absolute Error: 153.37%
- Max Deviation: 803.33%

The blind validation (no corrections) shows:
- Pass rate: 12.07% (7/58)
- Mean Absolute Error: 162.58%

The similar MAE suggests the corrections are not significantly improving overall metrics - they may be masking issues rather than fixing them. The corrections need investigation.

## Root Cause Indicators

The TODO-BLIND-VALIDATION comments in the code point to:
1. **Thermal mass coupling conductances need calibration**
2. **Solar gain distribution to thermal mass incomplete**
3. **CTF zone air coupling solver integration pending**
4. **Night ventilation modeling incomplete**

## Next Steps

Phase A.3 should focus on identifying which correction factors have the largest impact and determining if any can be replaced with physics-based fixes.

## Test Command

```bash
cargo test --test ashrae_140_blind_validation -- --nocapture
```
