# ISO 13790 5R1C HVAC Demand Calculation — Research Findings

## 1. The 5R1C Thermal Network

The ISO 13790 simple hourly method uses a **5R1C network** with 5 thermal resistances and 1 capacitance:

```
        H_tr_em        H_tr_ms
T_ext ─┤████├─ T_m ─┤████├─ T_s ─┬─ T_air
                            │        │  │
                            │        │  │
                          H_tr_w   H_tr_is  H_ve_adj
                            │        │  │
                          T_ext    T_air  T_supply
```

### Nodes:
- **T_m** (thermal mass node) — the single capacitance C_m
- **T_s** (internal surface temperature node)
- **T_air** (indoor air node) — where HVAC input φHC,nd is applied

### The 5 Resistances:
| Symbol | Meaning | Formula |
|--------|---------|---------|
| H_tr_em | Opaque envelope → mass | U_walls × A_walls |
| H_tr_ms | Mass → surface | 9.1 × A_m (mass area) |
| H_tr_w | Windows → surface | U_windows × A_windows |
| H_tr_is | Surface → air | h_as × A_t (typically 3.45 W/m²K × total internal area) |
| H_ve_adj | Ventilation | 1200 × b_ek × V × (ACH/3600) |

### Combined Conductances (ISO 13790 §C.6–C.8):
```
H_tr_1 = 1 / (1/H_ve_adj + 1/H_tr_is)     — Eq C.6
H_tr_2 = H_tr_1 + H_tr_w                    — Eq C.7
H_tr_3 = 1 / (1/H_tr_2 + 1/H_tr_ms)        — Eq C.8
```

## 2. Heat Flow Splitting (ISO 13790 §C.1–C.3)

Internal gains (φint) and solar gains (φsol) are split to the three nodes:

```
φ_ia = 0.5 × φ_int                                    — Eq C.1 (to air node)

φ_st = (1 - A_m/A_t - H_tr_w/(9.1×A_t)) × (0.5×φ_int + φ_sol)  — Eq C.2 (to surface)

φ_m  = (A_m/A_t) × (0.5×φ_int + φ_sol)               — Eq C.3 (to mass)
```

**CRITICAL**: The HVAC energy demand φHC,nd is added to φ_ia (for the default "air conditioning" emission system):
```
φ_ia += energy_demand   (i.e., φHC,nd goes entirely to the AIR NODE)
```

## 3. The Core Crank-Nicolson Equation (ISO 13790 §C.4)

### Step A: Compute φ_m,tot (Eq C.5)
```
φ_m,tot = φ_m + H_tr_em × T_ext
         + H_tr_3 × (φ_st + H_tr_w × T_ext + H_tr_1 × (φ_ia/H_ve_adj + T_supply)) / H_tr_2
```
where T_supply = T_ext (assumption per §9.3.2)

### Step B: Mass temperature update (Eq C.4) — Crank-Nicolson
```
T_m,next = [T_m,prev × (C_m/Δt - 0.5×(H_tr_3 + H_tr_em)) + φ_m,tot]
           / [C_m/Δt + 0.5×(H_tr_3 + H_tr_em)]
```

### Step C: Average mass temperature (Eq C.9)
```
T_m = (T_m,next + T_m,prev) / 2
```

### Step D: Surface temperature (Eq C.10)
```
T_s = (H_tr_ms × T_m + φ_st + H_tr_w × T_ext + H_tr_1 × (T_supply + φ_ia/H_ve_adj))
      / (H_tr_ms + H_tr_w + H_tr_1)
```

### Step E: Air temperature (Eq C.11)
```
T_air = (H_tr_is × T_s + H_ve_adj × T_supply + φ_ia) / (H_tr_is + H_ve_adj)
```

## 4. HVAC Demand Calculation (ISO 13790 §C.4.2)

The standard uses a **linear interpolation (Thales Intercept Theorem)** approach:

### Step 1: Free-floating check
Set φHC,nd = 0, compute T_air,0 (free-floating air temperature).

```
if T_air,0 < T_set,heating → heating needed
if T_air,0 > T_set,cooling → cooling needed
else → no demand
```

### Step 2: Linear interpolation for unrestricted demand (Eq C.13)
```
φ_ref = 10 × A_floor   (a reference power of 10 W/m²)

Compute T_air,10 = air temperature with φHC,nd = φ_ref

φHC,nd,unrestricted = φ_ref × (T_air,set - T_air,0) / (T_air,10 - T_air,0)
```

This is **linear interpolation** between the (0, T_air,0) and (φ_ref, T_air,10) points.

### Step 3/4: Apply capacity limits
```
if φHC,nd,max,cool ≤ φHC,nd,unrestricted ≤ φHC,nd,max,heat:
    φHC,nd = φHC,nd,unrestricted
elif φHC,nd,unrestricted > φHC,nd,max,heat:
    φHC,nd = φHC,nd,max,heat
elif φHC,nd,unrestricted < φHC,nd,max,cool:
    φHC,nd = φHC,nd,max,cool
```

## 5. KEY INSIGHT: How HVAC Demand Enters the Network

**The HVAC demand φHC,nd enters ONLY through φ_ia (the air node heat flow).**

For the default emission system (air conditioning / convective):
```
φ_ia = 0.5 × φ_int + φHC,nd
```

This means φHC,nd propagates through the network via:
1. φ_ia → φ_m,tot (through H_tr_1, H_tr_2, H_tr_3 path)
2. φ_ia → T_air (directly, via Eq C.11)

The "effective heat transfer coefficient" for the HVAC demand is **NOT** simply H_tot = H_tr + H_ve.

Instead, the HVAC demand influences the mass temperature through the full network path:
```
φHC,nd → φ_ia → φ_m,tot → T_m,next
```

The sensitivity of T_air to φHC,nd comes from the combined effect of the linear interpolation and the network equations. The effective coefficient is approximately:
```
H_eff ≈ (H_tr_is + H_ve_adj) × (H_tr_3 × H_tr_1) / (H_tr_2 × (H_tr_is + H_ve_adj))
```
But in practice, the linear interpolation handles this automatically.

## 6. CRITICAL DIFFERENCE vs. Your Current Implementation

### Your current (problematic) approach:
```python
ideal_loads = m_dot * cp * (T_supply - T_zone)   # Q = 43 W/K × ΔT
t_i_act = t_i_free + Q / h_tr_is                   # Air temp correction
# Mass evolves using t_i_act ≈ t_i_free (Q is tiny)
```

### What's wrong:
1. **φHC,nd = 43 W/K is FAR too small.** The ISO 13790 approach uses UNLIMITED power to achieve setpoint. The effective conductance should be much larger — essentially infinite (perfect control).

2. **The mass temperature uses t_i_act ≈ t_i_free**, which means the HVAC barely affects the thermal mass. This produces ~6.88 MWh because the HVAC has negligible feedback.

3. **The HVAC coupling coefficient (43 W/K) is wrong.** In ISO 13790, the HVAC can deliver ANY power needed to hit setpoint — there's no small "ideal loads" coefficient.

### The CORRECT approach (ISO 13790):

**Step 1**: Compute free-floating T_air (φHC,nd = 0)
**Step 2**: If T_air,free < T_set (need heating):
   - Compute T_air with a reference power φ_ref
   - Linear interpolate to find the φHC,nd that gives T_air = T_set
**Step 3**: Use THAT φHC,nd to update the mass temperature

The mass temperature ALWAYS evolves with the ACTUAL HVAC power applied. When HVAC is active, φHC,nd is large enough to bring T_air to T_set, and this power feeds back through the network to warm/cool the mass.

## 7. Recommended Approach for Your 9R4C Model

### Option A: Full ISO 13790 approach (recommended)
Extend the linear interpolation to your 9R4C model:

```python
def compute_hvac_demand_9r4c(self, gains, t_ext, t_m_prev_vector):
    # Step 1: Free-floating (no HVAC)
    t_air_free = self.solve_network(energy_demand=0, gains, t_ext, t_m_prev_vector)

    if t_air_free < self.t_set_heating:
        t_air_target = self.t_set_heating
    elif t_air_free > self.t_set_cooling:
        t_air_target = self.t_set_cooling
    else:
        return 0  # No demand

    # Step 2: Reference power
    phi_ref = 10 * self.floor_area  # 10 W/m² as per ISO 13790

    # Step 3: Solve with reference power
    t_air_ref = self.solve_network(energy_demand=phi_ref, gains, t_ext, t_m_prev_vector)

    # Step 4: Linear interpolation (Thales theorem)
    phi_hc_nd = phi_ref * (t_air_target - t_air_free) / (t_air_ref - t_air_free)

    return phi_hc_nd
```

### Option B: Direct analytical solution
If you can derive a closed-form expression for T_air as a function of φHC,nd in your 9R4C model, you can solve directly:
```
φHC,nd = (T_set × H_eff - Σ(other terms)) / coefficient_of_φHC,nd_in_T_air
```

### Key principles:
1. **HVAC power feeds into the air node** (φ_ia), not via a small conductance
2. **Mass temperature evolves with ACTUAL HVAC power** — not with free-floating temperature
3. **The linear interpolation** automatically accounts for the network coupling
4. **No small "ideal loads" coefficient** — the HVAC provides whatever power is needed to hit setpoint (within capacity limits)

## 8. Code References

### Verified implementations:
1. **RC_BuildingSimulator** (ETH Zurich): https://github.com/architecture-building-systems/RC_BuildingSimulator
   - `building_physics.py` — full ISO 13790 Annex C implementation

2. **DIBS** (IWU Germany): https://github.com/IWUGERMANY/DIBS---Dynamic-ISO-Building-Simulator
   - `building_physics.py` + `emission_system.py` — validated against BESTEST

3. **Modelica Buildings Library** (LBNL): https://simulationresearch.lbl.gov/modelica/
   - `Buildings.ThermalZones.ISO13790.Zone5R1C` — Modelica implementation

4. **simple-simple**: https://github.com/timtroendle/simple-simple
   - Simplified 1R1C version for educational purposes

### Key equation references in ISO 13790:2008:
- §C.1–C.3: Heat flow splitting (φ_ia, φ_st, φ_m)
- §C.4: Mass temperature Crank-Nicolson update
- §C.5: φ_m,tot definition
- §C.6–C.8: Combined conductances (H_tr_1, H_tr_2, H_tr_3)
- §C.9: Average mass temperature
- §C.10: Surface temperature
- §C.11: Air temperature
- §C.13: Energy demand by Thales interpolation
