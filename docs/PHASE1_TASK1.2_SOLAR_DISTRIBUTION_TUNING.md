# Phase 1 Task 1.2: Solar Distribution Tuning for Low-Mass

**Date:** 2026-03-29
**Status:** Complete
**Objective:** Test different `solar_distribution_to_air` values for Case 600 and measure impact on cooling energy

---

## Executive Summary

Solar distribution tuning shows **minimal impact** on energy results. The heating energy is ~6x higher than reference regardless of solar distribution value. This confirms that the root cause is **NOT solar distribution**, but rather the conductance/thermal mass issue identified in Task 1.1.

---

## Methodology

Tested `solar_distribution_to_air` values from 0.0 (all radiative gains to mass) to 0.5 (half to air) for Case 600:

- 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50

---

## Results

| Solar to Air | Heating (MWh) | Cooling (MWh) | Total (MWh) | % Heating Error | % Cooling Error |
|-------------|----------------|----------------|--------------|-----------------|----------------|
| 0.00 | 32.535 | 3.736 | 36.271 | +550.1% | -48.8% |
| 0.05 | 32.530 | 3.738 | 36.268 | +549.9% | -48.8% |
| 0.10 | 32.524 | 3.740 | 36.265 | +549.8% | -48.7% |
| 0.15 | 32.519 | 3.742 | 36.261 | +549.7% | -48.7% |
| 0.20 | 32.514 | 3.744 | 36.258 | +549.6% | -48.7% |
| 0.25 | 32.508 | 3.746 | 36.254 | +549.5% | -48.7% |
| 0.30 | 32.503 | 3.748 | 36.251 | +549.4% | -48.6% |
| 0.40 | 32.492 | 3.752 | 36.244 | +549.2% | -48.6% |
| 0.50 | 32.482 | 3.756 | 36.237 | +549.0% | -48.5% |

**Reference Values (ASHRAE 140 Case 600):**
- Heating: 4.30-5.71 MWh (midpoint: 5.0 MWh)
- Cooling: 6.14-8.45 MWh (midpoint: 7.3 MWh)

---

## Key Findings

### 1. Minimal Impact of Solar Distribution

Changing `solar_distribution_to_air` from 0.0 to 0.5:
- **Heating change:** 0.05 MWh (0.15% change)
- **Cooling change:** 0.02 MWh (0.5% change)

This is negligible compared to the 28-32 MWh heating error.

### 2. Heating Energy is ~6x Too High

All solar distribution values produce heating ~32.5 MWh:
- Reference: 4.30-5.71 MWh
- Simulated: 32.48-32.54 MWh
- **Error: +550% (6.5x too high)**

### 3. Cooling Energy is Too Low

All solar distribution values produce cooling ~3.74 MWh:
- Reference: 6.14-8.45 MWh
- Simulated: 3.73-3.76 MWh
- **Error: -48% (underprediction)**

---

## Optimal Value Analysis

| Metric | Optimal Value | Error |
|--------|----------------|--------|
| Heating | 0.50 | +549.0% |
| Cooling | 0.50 | -48.5% |
| Total | 0.50 | 597.5% |

**Conclusion:** No trade-off detected - all values produce similar results.

---

## Trend Analysis

| Solar Range | Average Heating (MWh) | Average Cooling (MWh) |
|-------------|----------------------|----------------------|
| Low (0.0-0.1) | 32.53 | 3.74 |
| Medium (0.15-0.3) | 32.51 | 3.74 |
| High (0.4-0.5) | 32.49 | 3.75 |

**No significant trend** - values are essentially constant.

---

## Physics-Based Explanation

### Hypothesis (From Task Plan)

For low-mass buildings:
- Small thermal capacitance (C_m = 2396.2 kJ/K)
- Fast thermal time constant (τ = 0.61 hours)
- Limited ability to buffer solar gains in thermal mass

**Higher `solar_distribution_to_air`** was hypothesized to:
- Send more solar gains directly to air
- Store less energy in mass
- More readily reject by cooling

### Actual Behavior

Solar distribution has minimal effect because:
1. **Heating error dominates** - the 550% heating error dwarfs any solar distribution effect
2. **Root cause is conductance** - h_tr_ms = 1092 W/K is too high (from Task 1.1)
3. **Cooling is being "overwhelmed"** - low cooling energy suggests system is fighting constant high heating demand

---

## Conclusion

**Solar distribution is NOT the root cause** of Case 600 validation failures.

### Evidence:
1. Changing solar_distribution_to_air from 0.0 to 0.5 changes results by <0.5%
2. Heating energy is 6x higher than reference regardless of solar distribution
3. This pattern matches the h_tr_ms issue identified in Task 1.1

### Root Cause (From Task 1.1):
- **h_tr_ms = 1092 W/K** is an order of magnitude too high
- **Expected: 10-100 W/K** for realistic thermal lag
- **Impact:** Thermal mass responds too fast (τ = 0.61 hours instead of 1-4 hours)

---

## Recommendations

### 1. Focus on Conductance Fix
Solar distribution tuning will not improve results until the conductance issue is resolved.

### 2. Revisit After h_tr_ms Fix
Once h_tr_ms is corrected to realistic values (10-100 W/K), solar distribution may have more meaningful impact.

### 3. Default Value
For now, keep `solar_distribution_to_air = 0.1` (current default) as it produces results similar to other tested values.

---

## Next Steps (Phase 1 Task 1.3)

**Investigate 600-series cases collectively:**

The solar distribution analysis confirms the issue is NOT solar-related. Task 1.3 will:
1. Run validation on all 600-series cases (600, 610, 620, 630, 640, 650)
2. Identify common failure modes across the series
3. Correlate findings with Task 1.1 and 1.2 results

---

## Files Modified/Created

1. **`src/bin/diagnose_solar_distribution.rs`** - Solar distribution tuning tool
2. **`docs/PHASE1_TASK1.2_SOLAR_DISTRIBUTION_TUNING.md`** - This report
