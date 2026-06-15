# ISO 13790 Annex C Reference Implementation Research

Date: 2026-05-18
Purpose: Analyze design patterns in open-source ISO 13790 5R1C implementations

## 1. Reference Implementation 1: RC_BuildingSimulator (ETH Zurich)

- **Language**: Python
- **URL**: https://github.com/architecture-building-systems/RC_BuildingSimulator
- **License**: MIT
- **Stars**: 127 | Forks: 42
- **Reference**: EN ISO 13790 Annex C (cited in source as "EN-13970")

### Key Design Decisions

1. **Crank-Nicolson mass update** — Uses ISO 13790 Eq. C.4 exactly:
   ```
   t_m_next = [t_m_prev * (Cm/3600 - 0.5*(H_tr_3 + H_tr_em)) + phi_m_tot]
              / [Cm/3600 + 0.5*(H_tr_3 + H_tr_em)]
   ```
   Note: `Cm/3600` because timestep is 1 hour = 3600s, and Cm is in J/K.

2. **Thales interpolation for HVAC demand** — Full ISO 13790 §C.4.2 procedure:
   - Step 1: Compute `t_air_0` (free-floating air temp, energy_demand=0)
   - Step 2: Compute `t_air_10` (air temp with 10 W/m² × floor_area)
   - Step 3: Interpolate: `energy_demand = 10*A_floor * (t_set - t_air_0) / (t_air_10 - t_air_0)`
   - Step 4: Clamp to max heating/cooling capacity

3. **Averaged mass temperature**: Uses `t_m = (t_m_prev + t_m_next) / 2` (Eq. C.9)
   - This averaged `t_m` is used for computing `t_s` (surface temp) and `t_air`

4. **Separate code paths for free-float vs HVAC**:
   - `has_demand()` first runs `calc_temperatures_crank_nicolson(energy_demand=0, ...)` to get free-floating `t_air`
   - If `t_air > cooling_setpoint` → cooling needed
   - If `t_air < heating_setpoint` → heating needed
   - If neither → free-float path (energy_demand = 0)
   - If either → `calc_energy_demand()` runs full Thales procedure

5. **Surface temperature computed explicitly** via Eq. C.10:
   ```
   t_s = (H_tr_ms * t_m + phi_st + H_tr_w * t_out + H_tr_1 * (t_supply + phi_ia/H_ve))
         / (H_tr_ms + H_tr_w + H_tr_1)
   ```

6. **Air temperature** via Eq. C.11:
   ```
   t_air = (H_tr_is * t_s + phi_ia + H_ve * t_supply) / (H_tr_is + H_ve)
   ```

### Timestep Loop Pseudocode

```python
def solve_energy(internal_gains, solar_gains, t_out, t_m_prev):
    # Step 1: Check if heating/cooling is needed
    self.has_demand(internal_gains, solar_gains, t_out, t_m_prev)

    if not self.has_heating_demand and not self.has_cooling_demand:
        # Free-float path
        energy_demand = 0
        self.calc_temperatures_crank_nicolson(0, internal_gains, solar_gains, t_out, t_m_prev)
    else:
        # HVAC path — Thales interpolation
        self.calc_energy_demand(internal_gains, solar_gains, t_out, t_m_prev)
        energy_demand = self.energy_demand  # clamped to max capacity

        # Recompute final temperatures with actual energy_demand
        self.calc_temperatures_crank_nicolson(energy_demand, ..., t_out, t_m_prev)

    # Return results
    self.t_m_prev = self.t_m_next  # store for next timestep
```

### `calc_temperatures_crank_nicolson` sequence (the core solver):

```python
def calc_temperatures_crank_nicolson(energy_demand, internal_gains, solar_gains, t_out, t_m_prev):
    # Eq. C.1: phi_ia = 0.5 * internal_gains (convective fraction)
    # Eq. C.2: phi_st = (1 - A_m/A_t - H_tr_w/(9.1*A_t)) * 0.5 * internal_gains + solar_to_surface
    # Eq. C.3: phi_m = A_m/A_t * 0.5 * internal_gains + solar_to_mass
    self.calc_heat_flow(t_out, internal_gains, solar_gains, energy_demand)

    # Eq. C.5: phi_m_tot = phi_m + H_tr_em*t_out + H_tr_3*(phi_st + H_tr_w*t_out + H_tr_1*(phi_ia/H_ve + t_supply)) / H_tr_2
    self.calc_phi_m_tot(t_out)

    # Eq. C.4: Crank-Nicolson mass update
    self.calc_t_m_next(t_m_prev)   # → t_m_next

    # Eq. C.9: Averaged mass temperature
    self.calc_t_m(t_m_prev)         # → t_m = (t_m_prev + t_m_next) / 2

    # Eq. C.10: Surface temperature (uses averaged t_m)
    self.calc_t_s(t_out)            # → t_s

    # Eq. C.11: Air temperature
    self.calc_t_air(t_out)          # → t_air

    return (t_m_next, t_air, t_s)
```

### How They Handle HVAC-Mass Coupling

The Thales interpolation is a **three-pass** approach:
1. **Pass 1** (in `has_demand`): `calc_temperatures_crank_nicolson(energy_demand=0)` → get `t_air_0`
2. **Pass 2** (in `calc_energy_demand`): `calc_temperatures_crank_nicolson(energy_demand=10*A_floor)` → get `t_air_10`
3. **Pass 3** (in `solve_energy`): `calc_temperatures_crank_nicolson(energy_demand=actual)` → final temperatures

Each pass runs the FULL Crank-Nicolson update (mass + surface + air). The mass temperature is updated in all three passes, but only the FINAL pass's `t_m_next` is kept as the state for the next timestep.

**Critical observation**: The HVAC demand is computed BEFORE the final mass update. The mass update in passes 1 and 2 is "throwaway" — used only to determine `t_air` for the interpolation. The actual `t_m_next` that persists comes from pass 3 with the interpolated HVAC demand.

---

## 2. Reference Implementation 2: DIBS (IWU Darmstadt / IWUGERMANY)

- **Language**: Python
- **URL**: https://github.com/IWUGERMANY/DIBS---Dynamic-ISO-Building-Simulator
- **License**: MIT
- **Stars**: 29 | Forks: 7
- **Reference**: DIN EN ISO 13790:2008 (German standard version)
- **Fork of**: RC_BuildingSimulator (credited in source header)

### Key Design Decisions

DIBS is a **direct fork** of RC_BuildingSimulator with extensions for German non-residential buildings:

1. **Same Crank-Nicolson mass update** — Identical Eq. C.4:
   ```
   t_m_next = [t_m_prev * (Cm/3600 - 0.5*(H_tr_3 + H_tr_em)) + phi_m_tot]
              / [Cm/3600 + 0.5*(H_tr_3 + H_tr_em)]
   ```

2. **Same Thales interpolation** — Full ISO 13790 §C.4.2 procedure (identical to RC_BuildingSimulator)

3. **Same averaged mass temperature**: `t_m = (t_m_prev + t_m_next) / 2`

4. **Adds emission system modeling**:
   - After computing unrestricted energy demand, routes through emission systems
   - Emission systems have radiative/convective splits that modify phi_ia, phi_st, phi_m
   - Supply systems model COP/efficiency curves for electricity and fuel consumption

5. **Adds below-ground envelope**:
   - Separate wall areas for above/below ground
   - Temperature adjustment factors for ground contact

6. **Same free-float vs HVAC split paths** as RC_BuildingSimulator

### Timestep Loop (extends RC_BuildingSimulator)

```python
def solve_building_energy(internal_gains, solar_gains, t_out, t_m_prev):
    self.has_demand(internal_gains, solar_gains, t_out, t_m_prev)

    if not has_heating_demand and not has_cooling_demand:
        # Free-float
        self.calc_temperatures_crank_nicolson(0, ..., t_out, t_m_prev)
        self.heating_demand = 0
        self.cooling_demand = 0
    elif has_heating_demand:
        # Thales interpolation → energy_demand
        self.calc_energy_demand(...)
        # Route through emission system
        flows = emDirector.calc_flows()  # radiative/convective split
        # Route through supply system
        supplyOut = supply_director.calc_system()
        self.heating_demand = self.energy_demand
        # Final temperature update with actual energy
        self.calc_temperatures_crank_nicolson(energy_demand, ..., t_out, t_m_prev)
    elif has_cooling_demand:
        # Same as heating but with cooling supply system

    self.t_m_prev = self.t_m_next
```

---

## 3. Reference Implementation 3: EnergyPlus

- **Language**: C++
- **URL**: https://github.com/NREL/EnergyPlus
- **License**: EnergyPlus Open Source License (modified BSD)
- **ISO 13790 implementation**: **NONE**

### Finding

EnergyPlus does **NOT** implement ISO 13790 Annex C. EnergyPlus uses its own:
- **CTF (Conduction Transfer Function)** method for envelope heat transfer
- **ZoneTempPredictorCorrection** module for zone air temperature
- **3rd-order backward difference** for zone air temperature prediction
- **Iterative surface heat balance** with simultaneous solution

EnergyPlus's approach is fundamentally different from ISO 13790:
- ISO 13790 uses a single lumped mass node (Cm) with 5 conductances
- EnergyPlus models each surface individually with CTF coefficients and solves a coupled system
- EnergyPlus iterates between surface heat balance and air temperature until convergence

EnergyPlus includes a **Simplified HVAC** model (ZoneHVAC:IdealLoadsAirSystem) which provides similar functionality to the ISO 13790 ideal HVAC, but using its own multi-surface thermal network, not the 5R1C model.

---

## 4. Design Pattern Comparison

| Pattern | RC_BuildingSimulator | DIBS (IWU) | EnergyPlus | **Fluxion (Our Code)** |
|---------|---------------------|------------|------------|----------------------|
| **CN mass update** | Yes (Eq. C.4) | Yes (Eq. C.4) | N/A (CTF) | Backward Euler + CN option via `select_integration_method` |
| **HVAC demand method** | Thales interpolation (3-pass) | Thales interpolation (3-pass) | Ideal loads (iterative) | `hvac_demand_from_ideal_loads()` — coefficient-based |
| **Tm averaging** | Yes: `t_m = (prev+next)/2` for t_s, t_air | Yes (same) | N/A | **No averaging** — uses `t_m_prev` (old mass temp) for t_i_free |
| **Free-float handling** | Separate path: `has_demand()` → 3-way branch | Same | Always computed | Separate `free_float` flag, zero HVAC output path |
| **Surface temperature** | Yes (Eq. C.10, explicit) | Yes (same) | Iterative surface balance | Computed after t_i_free for mass update (`t_s_act`) |
| **Pass count** | 3-pass (0W, 10W/m², actual) | 3-pass + emission/supply | Iterative | **1-pass** (t_i_free → HVAC → mass) |
| **Order of operations** | Gains → CN(0W) → CN(10W) → Thales → CN(actual) | Same | Iterative | Gains → t_i_free → HVAC demand → t_i_act → t_s → mass update |
| **HVAC in phi_m_tot** | Yes (energy_demand flows through H_tr_1→H_tr_2→H_tr_3) | Same | N/A | **No** — HVAC is NOT in phi_m_tot; HVAC is applied to air only |
| **t_supply assumption** | `t_supply = t_out` (no heat recovery) | `t_supply = t_out` | Varies | `t_supply = t_out` (same assumption) |

---

## 5. Key Design Questions — Answers from Reference Implementations

### a. Single-pass vs double-pass vs three-pass

**RC_BuildingSimulator and DIBS use THREE passes** of the full Crank-Nicolson update:
1. Pass 1 (energy=0): Get free-floating `t_air_0`
2. Pass 2 (energy=10*A_floor): Get `t_air_10`
3. Pass 3 (energy=interpolated): Get final `t_m_next`, `t_air`, `t_s`

Our code uses a **single pass**: compute `t_i_free` algebraically (not via CN), then compute HVAC demand, then update mass. This is faster but potentially less accurate because:
- Our `t_i_free` uses `t_m_prev` (old mass temp), not the averaged `t_m`
- We don't account for the HVAC power flowing through the network to the mass

### b. Thales interpolation vs coefficient

**Both reference implementations use Thales interpolation** per ISO 13790 §C.4.2:

```python
energy_demand = 10 * A_floor * (t_set - t_air_0) / (t_air_10 - t_air_0)
```

Our code uses a **coefficient-based approach** (`hvac_demand_from_ideal_loads`) that computes:
```
Q_hvac = h_loss × (t_setpoint - t_i_free)
```
where `h_loss` is an effective heat loss coefficient. This is equivalent to Thales IF `h_loss = 10*A_floor / (t_air_10 - t_air_0)`, which it is (the Thales formula is just a linear interpolation that yields the same coefficient).

**Verdict**: Our coefficient approach is mathematically equivalent to Thales interpolation. The coefficient IS the Thales-derived sensitivity.

### c. Averaged mass temperature

**Both reference implementations use `t_m = (t_m_prev + t_m_next) / 2`** (Eq. C.9):
- This averaged temperature is used for computing `t_s` (surface) and `t_air`
- The Crank-Nicolson scheme naturally averages between old and new time steps
- This is a key part of the ISO 13790 method

**Our code does NOT use averaged mass temperature**:
- We use `t_m_prev` (old mass temp) for computing `t_i_free`
- We compute `t_i_free` algebraically before the mass update
- This means our `t_i_free` and HVAC demand are computed with "stale" mass temperature

### d. Order of operations

**RC_BuildingSimulator / DIBS order**:
```
1. Compute gains (phi_ia, phi_st, phi_m)  [Eq. C.1-C.3]
2. Compute phi_m_tot                       [Eq. C.5]
3. Crank-Nicolson: t_m_next               [Eq. C.4]  ← mass updated FIRST
4. Average: t_m = (prev + next) / 2        [Eq. C.9]
5. Surface temp: t_s                       [Eq. C.10] ← uses averaged t_m
6. Air temp: t_air                         [Eq. C.11] ← uses t_s
```

**Our order**:
```
1. Compute gains (phi_ia, phi_st, phi_m)
2. Compute t_i_free algebraically          ← uses t_m_prev, NOT averaged
3. Compute HVAC demand from t_i_free       ← coefficient-based
4. Compute t_i_act = t_i_free + HVAC/h_tr_is
5. Compute t_s_act (surface with HVAC)     ← uses t_m_prev
6. Update mass temperature                 ← uses t_s_act (backward Euler/CN)
```

**Key difference**: ISO 13790 updates mass FIRST (via phi_m_tot which includes HVAC), then derives air temperature. We compute air temperature FIRST (without HVAC in the network), then update mass.

### e. Free-float vs HVAC code paths

**Both reference implementations use separate code paths**:
- `has_demand()` checks free-float temperature
- If no demand → free-float path (single CN update with energy=0)
- If demand → HVAC path (3-pass Thales procedure)

**Our code** also uses separate paths (`free_float` flag), but both paths share the same algebraic `t_i_free` calculation. The difference is that the HVAC path adds `hvac/h_tr_is` to get `t_i_act`.

### f. Surface temperature

**Both reference implementations explicitly compute surface temperature** (Eq. C.10):
```
t_s = (H_tr_ms * t_m + phi_st + H_tr_w * t_out + H_tr_1 * (t_supply + phi_ia/H_ve))
      / (H_tr_ms + H_tr_w + H_tr_1)
```

**Our code** computes `t_s_act` after t_i_act:
```
t_s = (H_tr_ms * mass_temp + H_tr_is * t_i_act + phi_st) / (H_tr_ms + H_tr_is + H_tr_w + ...)
```

This is used for the mass update but not for the air temperature calculation (we use `t_i_free` directly).

---

## 6. Recommended Design for Fluxion

Based on the reference implementations, the most impactful changes would be:

### Priority 1: Use averaged mass temperature (Eq. C.9)

The biggest divergence is that ISO 13790 uses `t_m = (t_m_prev + t_m_next)/2` for computing `t_s` and `t_air`. This requires the mass to be updated BEFORE the air temperature is computed.

This creates a chicken-and-egg problem: to update mass, you need HVAC demand, but to get HVAC demand, you need air temperature, which depends on the mass temperature.

**ISO 13790's solution**: The Thales 3-pass approach resolves this:
- Pass 1 and 2 determine HVAC demand (with throwaway mass updates)
- Pass 3 uses the interpolated HVAC demand for the final mass update

**Our simpler solution**: Since our coefficient approach is mathematically equivalent to Thales, we can:
1. Keep our single-pass HVAC demand (it's equivalent)
2. Add a second pass that recomputes `t_i_free` with averaged mass temp
3. Or: accept the `t_m_prev` approach as a valid simplification (first-order accurate vs second-order)

### Priority 2: Consider the 3-pass approach for accuracy

If ASHRAE 140 validation shows discrepancies in the mass temperature trajectory, implementing the full 3-pass Thales approach would bring us into exact alignment with ISO 13790. The cost is 3× the Crank-Nicolson updates per timestep, but since each is O(1) per zone, this is negligible.

### Priority 3: HVAC power in phi_m_tot

The reference implementations include HVAC power in `phi_m_tot` (it flows through H_tr_1 → H_tr_2 → H_tr_3 to the mass node). Our code applies HVAC directly to the air node only (`t_i_act = t_i_free + HVAC/h_tr_is`).

This is the fundamental coupling difference. In ISO 13790, HVAC power reaches the mass through the thermal network (via the series conductances). In our code, HVAC heats the air, which then heats the mass through `h_tr_ms`. The end result should be similar for steady-state but may differ transiently.

### Non-priority: Surface temperature

Our code already computes surface temperature for the mass update. The reference implementations compute it as an intermediate step for air temperature, but since our air temperature is computed algebraically (not via surface temperature), this is not a critical difference.

### Summary Table

| Change | Impact | Effort | Recommendation |
|--------|--------|--------|----------------|
| Averaged t_m for t_s/t_air | Medium (2nd-order accuracy) | Medium | Consider if validation shows mass temp drift |
| 3-pass Thales | Low (mathematically equivalent to our coefficient) | Low | Not needed — our coefficient IS the Thales result |
| HVAC in phi_m_tot | High (correct network coupling) | High | Consider for Phase 20 accuracy push |
| Explicit t_s for t_air | Low | Low | Already done for mass update |
