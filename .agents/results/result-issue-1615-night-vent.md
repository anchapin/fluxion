# Issue #1615: Case 950 Night Ventilation Peak Cooling Over-Prediction

## Status: ACTIVE - Root Cause Identification

## Summary
Case 950 night ventilation peak cooling is over-predicted by 153-226% (fluxion 2.284 kW vs reference 0.70-0.90 kW). The night ventilation physics is **confirmed correct** (h_ve_night=567.7 W/K). The root cause lies in the **multi-node 9R4C thermal model's mass coupling to night ventilation**, not in the ventilation configuration itself.

## Key Finding: Night Vent Physics Is NOT the Problem

Verification confirms:
- NightVentilation ACH = 13.14 (1703.16 m³/h ÷ 129.6 m³) ✓
- h_ve_night = 567.7 W/K (correctly computed) ✓
- Night vent removes 5.67 kW at ΔT=10°C (meets ≥0.5 kW criterion) ✓
- Night vent schedule 18:00-07:00 is correctly implemented ✓

The over-prediction (2.284 kW vs 1.414 kW E+ reference, or 2.284 kW vs 0.70-0.90 kW band) occurs DESPITE correct night ventilation.

## Engineering Calculations

### Calc 1: Night Vent ACH and Conductance
```
ACH_night = 1703.16 m³/h ÷ 129.6 m³ = 13.14 ACH
h_ve_night = 1703.16 × 1.2 × 1005 ÷ 3600 = 567.7 W/K
Baseline h_ve (0.5 ACH) = 21.7 W/K
Night vent is 26x larger than baseline infiltration
```

### Calc 2: Steady-State Free-Float Temperature
```
h_ext_base = h_ve + h_tr_w + h_tr_em = 21.7 + 25.2 + 49.6 = 96.5 W/K
h_ext_night = h_ve_night + h_tr_w + h_tr_em = 567.7 + 25.2 + 49.6 = 642.5 W/K

Without night vent (T_out=15°C): t_i_free ≈ 24.7°C
With night vent (T_out=15°C): t_i_free ≈ 18.6°C
Temperature reduction: 6.1°C
```

### Calc 3: HVAC Demand Formula
```
Q_HVAC = h_coeff × (T_free - T_cool_sp)
h_coeff = h_tr_1 + h_tr_w = 150.8 + 25.2 = 176 W/K
T_cool_sp = 27°C

At T_free=35°C (E+ peak): Q = 176 × (35-27) = 1408 W (1.41 kW) ✓ matches E+
At T_free=40°C (fluxion): Q = 176 × (40-27) = 2288 W (2.29 kW) ✓ matches fluxion
```

**The 5°C difference in T_free (35°C vs 40°C) explains the entire gap.**

## Root Cause Analysis

The multi-node 9R4C thermal model's mass coupling does NOT properly transfer the night ventilation cooling effect to the thermal mass nodes. The chain:

1. Night vent → h_ve_night added to air balance → lowers T_air ✓ (implemented)
2. T_air lowered → should cool mass through h_tr_is → mass temperature drops
3. Mass temperature at 07:00 determines starting conditions for peak day ← **BROKEN**

### Evidence from Code Analysis

**Issue #821 history (physics_impl.rs lines 396-404):**
- Legacy implementation routed 30% of night-vent flow directly to mass node
- This was disabled because it "double-counted air-side cooling"
- The air-side path was "never actually wired in step_physics_5r1c"

**Issue #1422 fix (physics_impl.rs lines 2330-2362):**
- Adds h_ve_night to h_ext during night-vent active hours
- Recomputes den dynamically for free-float temperature
- BUT: does NOT wire night-vent into phi_m (mass node source term)

**Key issue (physics_impl.rs line 2699):**
```rust
solver.step_with_gains(dt, gains_wall, gains_roof, gains_floor, gains_internal);
```
The gains (phi_m) do NOT include any night-vent effect on the mass.

### The Broken Chain

During night vent (18:00-07:00):
1. h_ve_night IS added to h_ext → T_i_free_5r1c drops ✓
2. T_air_mn IS computed with h_ve_night ✓
3. Mass node temperatures are updated via step_with_gains ✗
   - step_with_gains uses PREVIOUS surface_temperature (line 925, 942, etc.)
   - Surface temperature = zone_temp_prev - 0.5°C (line 2564)
   - This is a LAGGED approximation, not the actual multi-node surface temp

The mass node update does NOT see the cool air temperature from the night vent until the NEXT timestep. By then, the night vent may have ended (at 07:00), and the mass is still warm from the previous day's peak.

### 9R4C Mass Time Constant vs Night Vent Duration

For 200mm concrete (Case 950 high-mass):
- τ_mass ≈ C_mass / h_total ≈ 6 days (typical for heavy concrete)
- Night vent period: 11 hours (18:00-07:00)

The mass can only cool a FRACTION of its heat content during the 11-hour night vent period. But in E+, the night vent IS effective (peak=1.414 kW). This suggests E+ models the night vent effect on mass differently.

## What E+ Shows

From case_950_energy_hourly.csv (hour 5201 peak):
- T_outdoor: 32°C
- T_zone: 27°C (maintained by HVAC)
- Q_cool: 1414 W
- T_free (implied): 35°C

The zone is maintained at 27°C. The HVAC removes 1414 W to compensate for heat gains.

## Fluxion vs E+ Comparison

| Metric | E+ Reference | Fluxion | Ratio |
|--------|-------------|---------|-------|
| Peak cooling | 1.414 kW | 2.284 kW | 1.61x |
| Annual cooling | 0.39-0.92 MWh | 3.112 MWh | 3.4-8x |
| Implied T_free at peak | 35°C | 40°C | +5°C |

## Hypothesis

The night ventilation effect on the 9R4C mass node is NOT properly carried through because:

1. **Lagged surface temperature**: The mass update uses surface_temperature from PREVIOUS timestep, not the current multi-node computed value

2. **Weak air-mass coupling**: h_tr_is = 165.6 W/K creates a bottleneck for heat transfer from mass to air

3. **Mass node not directly cooled**: Unlike the legacy 30% path that "directly cooled mass", the current implementation only cools through the air

## Next Steps

1. **Verify morning starting temperature**: Add debug output at hour 5191 (07:00) to see if mass temperatures match between fluxion and E+

2. **Compare hourly zone temperatures**: Add hourly T_zone, T_mass output to simulation and compare with E+ CSV

3. **Check if Issue #821 "direct mass cooling" was correct**: The legacy 30% path may have been closer to E+ behavior

4. **Consider adding phi_m modification during night vent**: If the night vent IS supposed to directly cool mass (per Issue #1615 original issue), this needs to be re-introduced

## Files Analyzed

- `fluxion-core/src/ashrae_cases.rs:635-640` - NightVentilation::case_650() config ✓
- `src/sim/thermal_model_physics/physics_impl.rs:2190-2379` - Issue #1422 fix (h_ve_night in h_ext, den recomputed)
- `src/sim/thermal_model_physics/physics_impl.rs:2550-2702` - 9R4C mass update with lagged surface temperature
- `src/sim/thermal_model_physics/physics_impl.rs:2876-2896` - HVAC demand using t_i_free_mn
- `src/physics/multi_node_solver.rs:892-960` - step_backward_euler_with_gains (mass update)
- `tests/reference_data/zone_balance/case_950_energy_hourly.csv` - E+ hourly reference
