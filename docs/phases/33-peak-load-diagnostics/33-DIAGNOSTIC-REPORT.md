# Phase 33 Diagnostic Report: Peak Load Overestimation Analysis

## 1. Objective
Identify the root cause of the significant peak load overestimation in high-mass buildings (ASHRAE 140 Case 900-950) observed in Fluxion v0.7.0.

## 2. Quantitative Baseline (Case 900)
From Plan 33-01 and 33-02 diagnostic results:

| Metric | Fluxion (Current) | Reference (EnergyPlus/ASHRAE) | Error (%) |
| --- | --- | --- | --- |
| Peak Heating | 4437 W | 2687 W | +65.1% |
| Peak Cooling | 3415 W | 3041 W | +12.3% |
| Min Air Temp (FF) | -9.59°C | -6.40 to -1.60°C | -3.19°C (Too Cold) |
| Max Air Temp (FF) | 46.88°C | 41.80 to 46.40°C | +0.48°C (Slightly High) |
| Avg Diurnal Swing (FF) | 22.85°C | ~19.6°C | +16.6% (Over-swing) |

## 3. Component Audit & Root Cause Analysis

### 3.1 Under-estimation of Thermal Capacitance ($C_m$)
The investigation revealed a critical bug in `src/sim/engine.rs` (lines 1128-1133) where only the wall capacitance is added to the zone's thermal capacitance, intentionally excluding floor and roof mass.
- **Impact:** For Case 900, this excludes ~60% of the total thermal mass (the 150mm concrete floor and 100mm concrete roof).
- **Result:** The building behaves like a medium-mass building ($C_m \approx 1.62 \times 10^5$ J/m²K floor area) instead of a high-mass building ($C_m \approx 4.08 \times 10^5$ J/m²K floor area). This directly causes the over-swing in free-floating temperatures and higher heating peaks due to insufficient nighttime buffering.

### 3.2 Under-estimation of Mass-Surface Coupling ($H_{ms}$)
The `h_tr_ms` calculation (line 1039) only uses the `opaque_area` of walls, ignoring the roof and floor areas.
- **Impact:** Weaker coupling ($H_{ms} \approx 718$ W/K vs expected $\approx 1800$ W/K) means the interior air node cannot effectively "sink" heat into or "source" heat from the thermal mass.
- **Result:** The air temperature responds too quickly to external and solar gains, further contributing to under-damping.

### 3.3 Missing Roof Conduction in $H_{em}$
The `h_tr_em` calculation (line 1083) also excludes the roof area, meaning the conduction path from the exterior to the mass node is physically incomplete. While this technically reduces heat loss in the current implementation, it contributes to an incorrect time constant ($\tau$).

### 3.4 Short Time Constant ($\tau$)
The current model's time constant for Case 900 is $\approx 4.5$ hours.
- **Physical Reality:** High-mass concrete buildings typically have time constants $\tau > 50$ hours.
- **Diagnosis:** The 5R1C network structure, even when corrected, may be fundamentally limited in capturing the full $\tau$ of thick concrete layers due to single-node discretization. However, the current under-estimation by 10x is primarily due to the missing floor/roof mass and surface area.

## 4. Proposed Physics Correction (Phase 34)
To resolve these issues, the following corrections are recommended for Phase 34:

1.  **Correct $C_m$ Calculation:** Include all mass elements (walls, roof, floor, partitions) in the total thermal capacitance.
2.  **Unify Envelope Conduction:** Update $H_{ms}$ and $H_{em}$ to include all opaque envelope areas (walls + roof).
3.  **Refine Surface-to-Air Coupling ($H_{is}$):** Align $H_{is}$ with the ISO 13790 standard value (approx. 3.45 W/m²K per area) instead of raw film coefficients to prevent air from over-tracking surface temperatures.
4.  **Re-evaluate Time Constant:** After fixing the mass and coupling, verify if the corrected $\tau$ significantly improves the peak load accuracy. If it remains off, consider multi-node RC (6R2C/8R3C) or CTF-based damping.

## 5. Impact on Annual Energy
Annual energy was 100% compliant in v0.7.0, but this was achieved using a high `time_constant_sensitivity_correction` (up to 10.0 for Case 900 heating). This correction factor was likely compensating for the under-estimation of mass and the resulting over-estimation of heating demand.
- **Prediction:** After fixing the physics, the empirical correction factors can likely be reduced or eliminated, leading to a more robust and physically-grounded model.

**Status:** Ready for Phase 34 (Physics Fix).
