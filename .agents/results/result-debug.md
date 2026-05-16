# Debug Result: Case 600 Reference Comparison — Root Cause Analysis

## Status: COMPLETE

## Summary

Identified root cause of 600-series low-mass heating overestimation (~2.6x reference). The 5R1C sensitivity-based HVAC superposition formula produces an effective conductance of 672 W/K instead of the correct ~108 W/K for Case 600, causing HVAC power to be overestimated by 6.1x. The free-floating temperature partially compensates, yielding the observed 2.6x heating excess.

**Root cause confirmed: H1 (Sensitivity miscalibration for low-mass constructions).**

## Files Changed

| File | Action |
|------|--------|
| `.planning/debug/case600-comparison-report.md` | Created — Fluxion vs ASHRAE 140 comparison |
| `.planning/debug/case600-root-cause.md` | Created — Detailed root cause with source locations and fix direction |
| `.planning/debug/case600-components.csv` | Created — Component energy breakdown from Fluxion |

No source code changes were made (diagnosis only).

## Acceptance Criteria Checklist

- [x] `.planning/debug/case600-components.csv` exists with energy breakdown
- [x] ASHRAE 140 reference ranges documented (from `data/ashrae140_reference.json`)
- [x] `.planning/debug/case600-comparison-report.md` exists with comparison tables
- [x] `.planning/debug/case600-root-cause.md` exists with identified divergent term
- [x] Root cause maps to H1 (sensitivity miscalibration)
- [x] Specific source code lines and formula discrepancy documented

## Key Findings

### Metric Comparison

| Metric | Fluxion | ASHRAE 140 Ref (mean) | Ratio |
|--------|---------|----------------------|-------|
| Heating | 11.104 MWh | 4.213 MWh | **2.64x** |
| Cooling | 4.394 MWh | 5.856 MWh | **0.75x** |
| 600FF Max Temp | 50.3°C | 64.6°C | 0.78x |
| 600FF Min Temp | -8.25°C | -11.8°C | 0.70x |

### Root Cause

The sensitivity formula at `thermal_model_solvers.rs:152-153`:
```rust
self.0.derived_sensitivity = self.0.derived_term_rest_1 / self.0.derived_den;
```

For low-mass Case 600:
- `h_tr_ms = 1092 W/K` (very high — mass is tightly coupled to air)
- `h_tr_is = 1251 W/K`
- `sensitivity = 0.001489 K/W` → `1/sensitivity = 672 W/K`
- Correct value should be ~`1/108 = 0.009266 K/W`

The superposition `t_i_actual = t_i_free + sens × phi_HVAC` assumes thermal mass temperature is unchanged by HVAC. For low-mass buildings with high h_tr_ms, this assumption is invalid because HVAC heat flows easily to the mass node, changing its temperature.

### Source Code Locations

1. **`src/sim/thermal_model_solvers.rs:152-153`** — Sensitivity calculation
2. **`src/sim/thermal_model_physics.rs:43-76`** — HVAC power using `(deficit / sensitivity)`
3. **`src/sim/thermal_model_physics.rs:1203-1223`** — Temperature superposition
4. **`src/sim/thermal_model_core.rs:945-994`** — h_tr_ms = 9.1 × A_m (correct per ISO 13790)

### Recommended Fix

Use the existing `hvac_demand_from_ideal_loads()` (thermodynamic `mass_flow × cp × delta_T` formula) for all cases instead of the sensitivity-based formula. This path already exists in the code but is only activated when `ideal_loads_system` is initialized (currently only for 900-series).

## Out-of-Scope Findings

- EnergyPlus reference JSON (`600_reference.json`) has unrealistic annual totals (15M+ kWh) — likely corrupted or in Joules, not kWh
- Energy balance residual of -1.33 MWh suggests additional accounting issues
- `validate-case 600` CLI command doesn't work (600-series not in case range)
