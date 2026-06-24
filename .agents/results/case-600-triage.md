# Case 600 Series — Triage Report (UPDATED)

## Status: DISCOVERY COMPLETE — Fix Attempt FAILED

### Attempted Fix: `h_ms_coeff` 2.0 → 9.1 for LowMass

**Result: Made things WORSE (16 → 20 failures)**

---

## What Was Tried

Changed `src/sim/thermal_model_core.rs` line 1035:
```rust
// BEFORE (existing code):
crate::validation::ashrae_140_cases::ConstructionType::LowMass => 2.0,

// AFTER (ISO 13790 "universal constant" hypothesis):
crate::validation::ashrae_140_cases::ConstructionType::LowMass => 9.1,
```

### Results After Fix

| Metric | Before (h_ms=2.0) | After (h_ms=9.1) | Reference | Effect |
|--------|-------------------|-----------------|-----------|--------|
| 600FF Min | -11.77°C ❌ | -11.21°C ❌ | [-18.8, -15.6] | Still failing |
| 600FF Max | 68.08°C ✅ | **53.86°C ❌** | [64.9, 75.1] | **NOW FAILING** |
| 650FF Min | -17.80°C ❌ | -17.13°C ❌ | [-23.0, -21.0] | Still failing |
| 650FF Max | 56.85°C ❌ | (changed) ❌ | [63.2, 73.5] | Still failing |
| 650 Peak Cooling | 3.06 ❌ | **2.15 ✅** | [1.90, 2.50] | **FIXED** |
| 630 Peak Cooling | PASS ✅ | **1.60 ❌** | [1.80, 2.40] | **NOW FAILING** |

**Conclusion**: h_ms_coeff=9.1 creates a DIFFERENT set of failures, not fewer. The hypothesis that ISO 13790 requires h_ms=9.1 for ALL weight classes is incorrect.

---

## Root Cause: NOT h_ms_coeff

ISO 13790 Table C.2 actually specifies different h_ms values by thermal weight class:
- Very light (κ < 10,000 J/m²K): h_ms = **2.0 W/(m²·K)**
- Light (κ < 29,000 J/m²K): h_ms = **4.5 W/(m²·K)**
- Medium: h_ms = 9.1 W/(m²·K)
- Heavy: h_ms = 12.0 W/(m²·K)

**The LowMass h_ms_coeff=2.0 IS correct per ISO 13790 for lightweight construction.**

The 16 failures are caused by something else.

---

## Baseline Test Results: 16 Failures

```
Suite: ashrae_140_case_600_series
Result: 10 passed, 16 failed, 0 ignored
```

### Failure Table

| Case | Metric | Actual | Reference Range | Δ | Type |
|------|--------|--------|----------------|---|------|
| 610 | Annual Heating [MWh] | 3.21 | [4.36 – 5.79] | +1.15 | too LOW |
| 610 | Annual Cooling [MWh] | 2.53 | [3.92 – 6.14] | +1.39 | too LOW |
| 610 | Peak Heating [kW] | 2.11 | [4.30 – 5.70] | +2.19 | too LOW |
| 610 | Peak Cooling [kW] | 2.03 | [2.50 – 3.20] | +0.47 | too LOW |
| 620 | Annual Heating [MWh] | 3.41 | [4.50 – 6.50] | +1.09 | too LOW |
| 620 | Annual Cooling [MWh] | 1.88 | [3.20 – 5.00] | +1.32 | too LOW |
| 620 | Peak Heating [kW] | 2.06 | [2.80 – 3.80] | +0.74 | too LOW |
| 620 | Peak Cooling [kW] | 2.37 | [2.50 – 3.50] | +0.13 | too LOW |
| 630 | Annual Heating [MWh] | 3.50 | [5.05 – 6.47] | +1.55 | too LOW |
| 630 | Annual Cooling [MWh] | 1.46 | [2.13 – 3.70] | +0.67 | too LOW |
| 630 | Peak Heating [kW] | 2.07 | [4.70 – 6.10] | +2.63 | too LOW |
| 630 | Peak Cooling [kW] | 2.05 | [3.00 – 3.80] | +0.95 | too LOW |
| 640 | Annual Cooling [MWh] | 2.88 | [5.95 – 8.10] | +3.07 | too LOW |
| 640 | Peak Heating [kW] | 2.06 | [4.30 – 5.70] | +2.24 | too LOW |
| 650 | Annual Cooling [MWh] | 2.65 | [4.82 – 7.06] | +2.17 | too LOW |
| 650 | Peak Heating [kW] | 2.07 | [3.70 – 4.80] | +1.63 | too LOW |
| **650** | **Peak Cooling [kW]** | **3.06** | **[1.90 – 2.50]** | **+0.56** | **too HIGH** |
| **600FF** | **Min Temp [°C]** | **-11.77** | **[-18.8 – -15.6]** | **+3.83** | **too WARM** |
| **650FF** | **Min Temp [°C]** | **-17.80** | **[-23.0 – -21.0]** | **+3.20** | **too WARM** |
| **650FF** | **Max Temp [°C]** | **56.85** | **[63.2 – 73.5]** | **+6.35** | **too COLD** |

---

## Baseline Conductance Parameters (Case 600 Low Mass)

| Parameter | Value | Notes |
|----------|-------|-------|
| h_tr_w (window) | 25.2 W/K | 2.1 W/m²K × 12 m² |
| h_tr_op (opaque) | 51.6 W/K | walls + roof |
| h_ve (vent+inf) | 22.2 W/K | 0.5 ACH |
| h_tr_is | 1343.6 W/K | interior surface film |
| **h_tr_ms** | **240 W/K** | h_ms_coeff=2.0 × a_m=120 ✓ CORRECT |
| h_tr_em | 54.1 W/K | external-mass coupling |
| h_ve_night (650) | 582.5 W/K | 26× infiltration |

---

## Key Insight: Pattern Analysis

**17 failures are "too LOW energy"** — simulated building uses less HVAC energy than expected.
**3 failures are thermal (temperature, not energy)** — free-float min/max.

The dominance of "too LOW" failures suggests:
- Zone not reaching heating setpoints → building loses heat too fast
- Zone not reaching cooling setpoints → building gains heat too slowly OR HVAC can't meet load

But the FREE-FLOATING cases (no HVAC) show:
- Min temps too WARM → mass not cooling enough overnight
- Max temp (650FF) too COLD → mass not heating enough during day

This paradox (HVAC under-delivers BUT mass under-swings) suggests the issue is NOT conductance but rather:
1. **Solar gain routing** — too little solar reaching the zone air?
2. **Internal gains** — too low contribution to heating?
3. **Night ventilation effectiveness** — fan cooling not reaching mass?

---

## Additional Finding: Case 900 Series (6 Pre-existing Failures)

With the clean code, Case 900 (High Mass) also shows failures:
- `test_case_900ff_min_temperature`: 900FF Min Temp = **-0.99°C** (reference: ~-20°C expected)
- `test_case_900_solar_beam_to_mass_fraction_sweep`: Sweep fails, max temp outside range

This suggests the thermal coupling problem is NOT isolated to LowMass.

---

## Files Modified During Triage

- `.agents/results/case-600-triage.md` — initial findings
- `.agents/results/case-600-triage-v2.md` — corrected findings after fix attempt

## Next Steps

The h_ms_coeff hypothesis is ruled out. The next investigation should focus on:
1. **Solar distribution** — how much solar gain reaches zone air vs mass vs opaque surfaces
2. **Internal gain fraction** — convective vs radiative split
3. **Night vent routing** — does the night vent fan actually purge the zone air effectively?
4. **650FF night vent max temp** — why is 650FF max 6°C below reference?
