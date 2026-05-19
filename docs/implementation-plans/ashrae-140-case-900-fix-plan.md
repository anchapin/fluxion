# ASHRAE 140 Case 900 Fix Plan

**Status**: DRAFT — Architecture Review Complete
**Date**: 2026-05-17
**Target**: All 5 ASHRAE 140 Case 900 metrics within reference ranges
**Scope**: `step_physics_9r4c` path only; MUST NOT regress Case 600/610/620/650

## 1. Current Metric Status

| Metric | Value | Reference Range | Status |
|--------|-------|----------------|--------|
| Annual heating | 1.91 MWh | 1.17–2.04 MWh | ✅ PASS |
| Annual cooling | 0.04 MWh | 2.13–3.67 MWh | ❌ 50x too low |
| Peak heating | 0.42 kW | 1.10–2.10 kW | ❌ 3x too low |
| Peak cooling | 0.15 kW | 1.50–3.50 kW | ❌ 10x too low |
| 900FF max temp | 27.73°C | 41.8–46.4°C | ❌ 15°C too low |

---

## 2. Root Cause Analysis

### 2.1 The Fundamental Architectural Flaw

`step_physics_9r4c` is a **5R1C computation with a multi-node side-car**. The multi-node solver's output does not drive zone temperature or HVAC energy. The causal chain is:

```
5R1C computes t_i_free → feeds to multi-node as boundary → multi-node temps averaged → overwrite mass_temperatures → next timestep's 5R1C uses corrupted mass_temperatures
```

This creates a destructive feedback loop: the multi-node solver produces different (typically lower) mass temperatures than the 5R1C expects, dragging down zone temperatures in subsequent steps.

### 2.2 Why Each Metric Fails

#### Cooling 50x Too Low (0.04 vs 2.13–3.67 MWh)

The 5R1C free-floating temperature is **too cool** during summer because the multi-node mass temperature feedback corrupts the lumped mass temperature. When `t_i_free` is artificially low, the zone spends less time above the cooling setpoint, so the HVAC system requests almost no cooling. The 5R1C's lumped conductance `h_tr_ms = 9.1 × A_m` (ISO 13790 Annex D) bundles radiative exchange and mass distribution differently than the multi-node solver's per-surface decomposition, creating an impedance mismatch when the two are coupled.

#### Peak Heating 3x Too Low (0.42 vs 1.10–2.10 kW)

Peak heating occurs on cold winter mornings when the zone needs maximum power to maintain setpoint. The corrupted mass temperatures drag `t_i_free` toward the mass temperature, reducing the apparent temperature deficit. The HVAC system sees a zone that's warmer than it actually is, so it requests less heating power.

#### Peak Cooling 10x Too Low (0.15 vs 1.50–3.50 kW)

Same mechanism as heating but amplified. Peak cooling occurs during peak solar gain. The 5R1C's `phi_st` term (solar to surface node) is computed from the lumped mass temperature, which has been dragged down by the multi-node feedback. The resulting `t_i_free` doesn't rise enough to trigger significant cooling demand.

#### 900FF Max Temp 15°C Too Low (27.73 vs 41.8–46.4°C)

This is the **smoking gun**. Before multi-node feedback was added, 900FF max was 42.87°C (passing). Now it's 27.73°C. The free-floating case has NO HVAC, so the temperature should be purely determined by the thermal network. The multi-node solver's mass temperatures are being averaged (`(wall + roof + floor + internal) / 4`) and fed back as `mass_temperatures`, which then drives the next timestep's `num_tm` term:

```rust
let num_tm = self.0.derived_h_ms_is_prod
    .zip_with(&self.0.mass_temperatures, |a, b| a * b);
```

The multi-node solver converges to different steady-state temperatures than the 5R1C's lumped mass. For a heavy-mass building with high solar gain, the floor node (coupled to ground at ~10°C) and the equal-weighting average drag the effective mass temperature far below what the 5R1C thermal resistance network expects. This suppresses `t_i_free` by 15°C.

### 2.3 Why Annual Heating Passes

Heating passes because it's dominated by winter conditions when outdoor temperatures are well below setpoint. The HVAC demand is `h_tr_is × (setpoint - t_i_free)`, and even with corrupted mass temperatures, the outdoor-to-indoor temperature difference is so large that heating demand falls within the reference range. The corruption's effect is proportionally smaller during winter.

### 2.4 The MultiNodeHvacRunner Dead Code

`MultiNodeHvacRunner` (#865) was designed to solve exactly this problem: it wraps `MultiNodeSolver` with HVAC control and energy accumulation. But it is **never called** from `step_physics_9r4c`. Its `step()` method computes its own zone temperature estimate:

```rust
let zone_temp = (self.h_tr_w * outdoor_temp
    + self.h_ve * outdoor_temp
    + solar_gain
    + internal_gain
    + h_tr_is * t_surface)
    / (self.h_tr_w + self.h_ve + h_tr_is);
```

This estimate is a simple steady-state balance — no mass dynamics, no proper coupling to the multi-node solver's thermal capacitances. It's not usable in its current form.

---

## 3. Architectural Solution: Option C — Coupled Multi-Node Air Balance

### 3.1 Why Not Options A, B, or D?

**Option A (Replace 5R1C entirely with multi-node):** The 5R1C's `h_tr_ms = 9.1 × A_m` regression constant encodes decades of empirical validation. The multi-node solver has no equivalent calibration for the lumped zone air temperature. We'd need to solve 4 surface nodes + 1 air node simultaneously, which is a different physical model entirely.

**Option B (Fix feedback only):** This is what the current code attempted and failed. The fundamental problem is that the 5R1C's lumped mass temperature and the multi-node solver's per-surface temperatures cannot be simply averaged. They're solutions to different thermal networks. Any averaging creates an energy imbalance.

**Option C (Coupled multi-node air balance):** Solve the multi-node solver's per-surface heat balance simultaneously with the zone air energy balance. This is what EnergyPlus and TRNSYS actually do. The multi-node solver becomes the PRIMARY thermal engine for the 9R4C path, and the 5R1C sensitivity formula is retired for this path.

### 3.2 The Physics

For Case 900 (9R4C path), at each timestep, solve simultaneously:

**Surface node balance** (for each envelope node k ∈ {wall, roof, floor}):
```
C_k/dt × (T_k^new - T_k^old) = h_tr_em_k × (T_ext_k - T_k^new) + h_tr_ms_k × (T_s - T_k^new)
```

**Internal mass node:**
```
C_int/dt × (T_int^new - T_int^old) = h_tr_is × (T_air - T_int^new) + h_tr_me × (T_env_avg - T_int^new)
```

**Zone air energy balance** (the critical missing piece):
```
0 = h_tr_is × (T_s - T_air) + h_ve × (T_out - T_air) + phi_ia + Q_hvac
```

Where `T_s` (surface node temperature) is derived from the envelope node temperatures via their `h_tr_ms` conductances:
```
T_s = Σ(h_tr_ms_k × T_k) / Σ(h_tr_ms_k)
```

And the free-floating temperature `T_air` is the solution of the air balance with `Q_hvac = 0`:
```
T_air_free = (h_tr_is × T_s + h_ve × T_out + phi_ia) / (h_tr_is + h_ve)
```

This is a **sequential solve** (not fully coupled), which is standard practice:
1. Solve envelope node temperatures using backward Euler (existing `step_backward_euler`)
2. Compute `T_s` from updated envelope node temps
3. Compute `T_air` from air energy balance
4. Update internal mass node with `T_air`

### 3.3 Why This Fixes All Five Metrics

- **900FF max temp**: `T_air_free` is now computed from the multi-node solver's surface temperatures, not from the 5R1C's corrupted lumped mass. The surface temperatures respond to solar gain via sol-air temps, producing correct summer peaks.
- **Cooling**: `T_air_free` rises correctly during summer → cooling demand is proportional to `(T_air_free - cooling_setpoint)`.
- **Peak cooling**: At peak solar, `T_s` rises from solar-driven envelope nodes → `T_air_free` rises → peak cooling demand matches EnergyPlus/TRNSYS.
- **Peak heating**: At peak cold, envelope nodes are cold → `T_s` is low → `T_air_free` is low → heating demand is high.
- **Annual heating**: Should remain within range because the winter balance is dominated by `h_ve × T_out`, which hasn't changed.

---

## 4. Specific Code Changes

### 4.1 `src/physics/multi_node_solver.rs` — Add Zone Air Temperature Computation

**Add a new method** that computes the zone air temperature from the current surface and envelope node temperatures, after the backward Euler step:

```rust
impl MultiNodeSolver {
    /// Compute zone air temperature from the multi-node thermal balance.
    ///
    /// This replaces the 5R1C sensitivity formula for the 9R4C path.
    /// Must be called AFTER `step()` has updated mass node temperatures.
    ///
    /// # Arguments
    /// * `t_outdoor` - Outdoor air temperature [°C]
    /// * `h_ve` - Ventilation/infiltration conductance [W/K]
    /// * `phi_ia` - Internal convective + solar-to-air gains [W]
    ///
    /// # Returns
    /// Free-floating zone air temperature [°C]
    pub fn compute_zone_air_temperature(
        &self,
        t_outdoor: f64,
        h_ve: f64,
        phi_ia: f64,
    ) -> f64 {
        // Surface node temperature: weighted average of envelope node temps
        // T_s = Σ(h_tr_ms_k × T_k) / Σ(h_tr_ms_k)  for k ∈ {wall, roof, floor}
        let h_ms_wall = self.mass.wall.h_tr_ms;
        let h_ms_roof = self.mass.roof.h_tr_ms;
        let h_ms_floor = self.mass.floor.h_tr_ms;
        let h_ms_total = h_ms_wall + h_ms_roof + h_ms_floor;

        let t_surface = if h_ms_total > 0.0 {
            (h_ms_wall * self.mass.wall.temperature
                + h_ms_roof * self.mass.roof.temperature
                + h_ms_floor * self.mass.floor.temperature)
                / h_ms_total
        } else {
            self.surface_temperature
        };

        // Zone air energy balance (free-floating, Q_hvac = 0):
        // 0 = h_tr_is × (T_s - T_air) + h_ve × (T_out - T_air) + phi_ia
        // T_air = (h_tr_is × T_s + h_ve × T_out + phi_ia) / (h_tr_is + h_ve)
        let h_is = self.h_tr_is;
        let denom = h_is + h_ve;

        if denom > 0.0 {
            (h_is * t_surface + h_ve * t_outdoor + phi_ia) / denom
        } else {
            t_surface // Fallback: no ventilation and no internal coupling
        }
    }

    /// Compute HVAC power demand to maintain setpoints.
    ///
    /// Uses the multi-node air temperature as the basis.
    ///
    /// # Arguments
    /// * `t_air_free` - Free-floating zone air temperature [°C]
    /// * `heating_setpoint` - Heating setpoint [°C]
    /// * `cooling_setpoint` - Cooling setpoint [°C]
    /// * `t_outdoor` - Outdoor air temperature [°C]
    /// * `h_ve` - Ventilation conductance [W/K]
    /// * `phi_ia` - Internal convective gains [W]
    ///
    /// # Returns
    /// HVAC power demand [W]. Positive = heating, negative = cooling.
    pub fn compute_hvac_demand(
        &self,
        t_air_free: f64,
        heating_setpoint: f64,
        cooling_setpoint: f64,
        t_outdoor: f64,
        h_ve: f64,
        phi_ia: f64,
    ) -> f64 {
        let h_is = self.h_tr_is;

        // Surface node temperature (same as in compute_zone_air_temperature)
        let h_ms_wall = self.mass.wall.h_tr_ms;
        let h_ms_roof = self.mass.roof.h_tr_ms;
        let h_ms_floor = self.mass.floor.h_tr_ms;
        let h_ms_total = h_ms_wall + h_ms_roof + h_ms_floor;

        let t_surface = if h_ms_total > 0.0 {
            (h_ms_wall * self.mass.wall.temperature
                + h_ms_roof * self.mass.roof.temperature
                + h_ms_floor * self.mass.floor.temperature)
                / h_ms_total
        } else {
            self.surface_temperature
        };

        let total_conductance = h_is + h_ve;

        if t_air_free < heating_setpoint {
            // Heating: Q_hvac = total_conductance × (T_setpoint_heating - T_air_free)
            total_conductance * (heating_setpoint - t_air_free)
        } else if t_air_free > cooling_setpoint {
            // Cooling: Q_hvac = -total_conductance × (T_air_free - T_setpoint_cooling)
            -(total_conductance * (t_air_free - cooling_setpoint))
        } else {
            0.0
        }
    }
}
```

**Why this is thermodynamically correct:**
- The surface node `T_s` is a conductance-weighted average of envelope node temperatures, which is the standard lumped-surface approach in ISO 13790 §7.2.
- The air balance `T_air = (h_is × T_s + h_ve × T_out + phi_ia) / (h_is + h_ve)` is the steady-state energy balance at the air node, which is valid for a 1-hour timestep when the air thermal capacitance is negligible (air has ~0.3 Wh/K·m³ for a typical zone — tiny compared to envelope mass).
- HVAC demand `Q = (h_is + h_ve) × (T_setpoint - T_air_free)` is the power needed to shift the air temperature to setpoint, which is the standard ideal loads equation from ASHRAE 140.

### 4.2 `src/sim/thermal_model_physics.rs` — Rewrite `step_physics_9r4c`

The 9R4C path must be restructured to use the multi-node solver as the primary thermal engine. Here is the new algorithm:

```
fn step_physics_9r4c(timestep, outdoor_temp, dt):
    // 1. Gather inputs (same as current)
    t_g = ground_temperature(timestep)
    sky_temp = weather.sky_temperature()
    (sol_air_data, ctf_flux, fd_flux, ctf_surface_temps) = prepare_solvers_and_sol_air(...)

    // 2. Compute gain distribution (same as current)
    phi_ia, phi_st, phi_m = distribute_gains(...)

    // 3. Compute per-surface sol-air temperatures (same as current)
    surface_ext_temps = compute_surface_exterior_temperatures(...)

    // 4. Distribute gains to multi-node solver surfaces
    for each zone:
        solver = multi_node_solvers[zone]
        solver.set_surface_exterior_temperatures(surface_ext_temps)

        // Distribute phi_st to envelope nodes (proportional to h_tr_ms)
        // Distribute phi_m to internal mass node
        // phi_ia stays at air node (used in air balance)

        // 5. Step the multi-node solver (backward Euler for mass nodes)
        solver.step_with_gains(dt, phi_st_per_node, phi_m, phi_ia)

    // 6. Compute zone air temperature from multi-node results
    for each zone:
        solver = multi_node_solvers[zone]
        t_air_free = solver.compute_zone_air_temperature(
            outdoor_temp, h_ve, phi_ia
        )

    // 7. Compute HVAC demand from multi-node air temperature
    if free_float:
        t_air_act = t_air_free
        q_hvac = 0.0
    else:
        q_hvac = solver.compute_hvac_demand(
            t_air_free, heating_setpoint, cooling_setpoint,
            outdoor_temp, h_ve, phi_ia
        )
        // Corrected air temperature with HVAC
        t_air_act = t_air_free + q_hvac / (h_tr_is + h_ve)

    // 8. Update zone temperatures and accumulate energy (same structure)
    self.temperatures = t_air_act
    accumulate_annual_energy(q_hvac, dt)
    track_peak_power(q_hvac)

    // 9. Update mass_temperatures for diagnostics/inter-zone coupling
    //    Use conductance-weighted average, NOT simple average
    mass_temperatures = conductance_weighted_average(solver.temperatures)

    return q_hvac * dt / 3.6e6  // kWh
```

### 4.3 Specific Changes to `step_physics_9r4c`

**Lines ~2391-2522 (the 5R1C free-floating temp + multi-node update section):**

DELETE the 5R1C sensitivity computation for the 9R4C path. Replace with:

```rust
// === Multi-Node Solver as Primary Thermal Engine ===
// The 9R4C path uses the multi-node solver for zone temperature,
// NOT the 5R1C sensitivity formula.

for zone_idx in 0..self.0.num_zones {
    if zone_idx >= self.0.multi_node_solvers.len() {
        continue;
    }

    let solver = &mut self.0.multi_node_solvers[zone_idx];

    // Set per-surface exterior temperatures (sol-air) — same as current
    solver.set_surface_exterior_temperatures(surface_ext_temps);

    // Set surface temperature estimate from previous envelope temps
    // (conductance-weighted, not hardcoded -0.5)
    let h_ms_w = solver.mass.wall.h_tr_ms;
    let h_ms_r = solver.mass.roof.h_tr_ms;
    let h_ms_f = solver.mass.floor.h_tr_ms;
    let h_ms_total = h_ms_w + h_ms_r + h_ms_f;
    let t_surface = if h_ms_total > 0.0 {
        (h_ms_w * solver.mass.wall.temperature
            + h_ms_r * solver.mass.roof.temperature
            + h_ms_f * solver.mass.floor.temperature) / h_ms_total
    } else {
        solver.surface_temperature
    };
    solver.set_surface_temperature(t_surface);

    // Step the multi-node solver
    solver.step(dt);
}

// Compute zone air temperatures from multi-node results
let mut t_i_free_data = Vec::with_capacity(self.0.num_zones);
for zone_idx in 0..self.0.num_zones {
    if zone_idx < self.0.multi_node_solvers.len() {
        let solver = &self.0.multi_node_solvers[zone_idx];
        let phi_ia_w = phi_ia.as_ref()[zone_idx];
        let h_ve_val = /* current ventilation conductance for this zone */;

        let t_air = solver.compute_zone_air_temperature(
            outdoor_temp, h_ve_val, phi_ia_w
        );
        t_i_free_data.push(t_air);
    } else {
        t_i_free_data.push(outdoor_temp); // Fallback
    }
}
let t_i_free = T::from(VectorField::new(t_i_free_data));
```

**Lines ~2494-2522 (mass temperature feedback):**

REPLACE the simple average with a conductance-weighted average:

```rust
if !self.0.multi_node_solvers.is_empty() {
    let mut mass_temps = Vec::with_capacity(self.0.num_zones);
    let mut env_temps = Vec::with_capacity(self.0.num_zones);
    let mut int_temps = Vec::with_capacity(self.0.num_zones);

    for solver in &self.0.multi_node_solvers {
        env_temps.push(solver.envelope_temperature());
        int_temps.push(solver.internal_temperature());

        // Conductance-weighted mass temperature
        // This respects the thermal resistance network structure
        let h_ms_w = solver.mass.wall.h_tr_ms;
        let h_ms_r = solver.mass.roof.h_tr_ms;
        let h_ms_f = solver.mass.floor.h_tr_ms;
        let h_is = solver.h_tr_is;

        let h_env = h_ms_w + h_ms_r + h_ms_f;
        let t_env = solver.envelope_temperature();
        let t_int = solver.internal_temperature();

        let weighted_mass = if h_env + h_is > 0.0 {
            (h_env * t_env + h_is * t_int) / (h_env + h_is)
        } else {
            (t_env + t_int) / 2.0
        };
        mass_temps.push(weighted_mass);
    }

    self.0.envelope_mass_temperatures = T::from(VectorField::new(env_temps));
    self.0.internal_mass_temperatures = T::from(VectorField::new(int_temps));
    self.0.mass_temperatures = T::from(VectorField::new(mass_temps));
}
```

**Lines ~2524-2646 (HVAC demand and energy accumulation):**

Keep the same structure but compute HVAC from the multi-node `t_i_free`:

```rust
// HVAC demand from multi-node free-floating temperature
if self.0.free_float {
    // Free-floating: no HVAC
    let temps_slice = self.0.temperatures.as_mut();
    for (i, t_val) in t_i_free.as_ref().iter().enumerate() {
        if i < temps_slice.len() {
            temps_slice[i] = *t_val;
        }
    }
    return 0.0;
}

// Compute HVAC from multi-node solver
let zone_idx = 0; // Single-zone for ASHRAE 140
let solver = &self.0.multi_node_solvers[zone_idx];
let t_air_free = t_i_free.as_ref()[zone_idx];
let h_ve_val = /* current ventilation */;

let q_hvac = solver.compute_hvac_demand(
    t_air_free,
    self.0.heating_setpoint,
    self.0.cooling_setpoint,
    outdoor_temp,
    h_ve_val,
    phi_ia.as_ref()[zone_idx],
);

// Corrected zone temperature
let t_air_act = t_air_free + q_hvac / (solver.h_tr_is + h_ve_val);

// Update zone temperatures
self.0.temperatures.as_mut()[zone_idx] = t_air_act;

// Energy accumulation (same structure as current)
let heating_energy_j = if q_hvac > 0.0 { q_hvac * dt } else { 0.0 };
let cooling_energy_j = if q_hvac < 0.0 { -q_hvac * dt } else { 0.0 };
self.0.annual_heating_energy += heating_energy_j / 3.6e6;
self.0.annual_cooling_energy += cooling_energy_j / 3.6e6;

if q_hvac > 0.0 {
    self.0.peak_power_heating = self.0.peak_power_heating.max(q_hvac);
} else if q_hvac < 0.0 {
    self.0.peak_power_cooling = self.0.peak_power_cooling.max(-q_hvac);
}

q_hvac * dt / 3.6e6 // kWh
```

### 4.4 `src/physics/multi_node_solver.rs` — Add `step_with_gains` Method

The current `step_backward_euler` ignores solar/internal gains distributed to each node. Add gain injection:

```rust
/// Advance the solver by one timestep with distributed gains per node.
///
/// # Arguments
/// * `dt` - Timestep duration [seconds]
/// * `gains_wall` - Heat gain to wall node [W]
/// * `gains_roof` - Heat gain to roof node [W]
/// * `gains_floor` - Heat gain to floor node [W]
/// * `gains_internal` - Heat gain to internal mass node [W]
pub fn step_with_gains(
    &mut self,
    dt: f64,
    gains_wall: f64,
    gains_roof: f64,
    gains_floor: f64,
    gains_internal: f64,
) -> &MultiNodeThermalMass {
    self.timestep_seconds = dt;
    self.step_backward_euler_with_gains(gains_wall, gains_roof, gains_floor, gains_internal);
    &self.mass
}

fn step_backward_euler_with_gains(
    &mut self,
    gains_wall: f64,
    gains_roof: f64,
    gains_floor: f64,
    gains_internal: f64,
) {
    let dt = self.timestep_seconds;
    let h_is = self.h_tr_is;

    let t_ext_wall = self.exterior_temperatures.t_ext_wall;
    let t_ext_roof = self.exterior_temperatures.t_ext_roof;
    let t_ext_floor = self.exterior_temperatures.t_ext_floor;

    let m = &mut self.mass;

    // Wall node
    {
        let node = &mut m.wall;
        let h_em = node.h_tr_em;
        let h_ms = node.h_tr_ms;
        let denom = node.capacitance / dt + h_em + h_ms;
        let numer = node.capacitance / dt * node.temperature
            + h_em * t_ext_wall
            + h_ms * self.surface_temperature
            + gains_wall;
        node.temperature = numer / denom;
    }

    // Roof node
    {
        let node = &mut m.roof;
        let h_em = node.h_tr_em;
        let h_ms = node.h_tr_ms;
        let denom = node.capacitance / dt + h_em + h_ms;
        let numer = node.capacitance / dt * node.temperature
            + h_em * t_ext_roof
            + h_ms * self.surface_temperature
            + gains_roof;
        node.temperature = numer / denom;
    }

    // Floor node
    {
        let node = &mut m.floor;
        let h_em = node.h_tr_em;
        let h_ms = node.h_tr_ms;
        let denom = node.capacitance / dt + h_em + h_ms;
        let numer = node.capacitance / dt * node.temperature
            + h_em * t_ext_floor
            + h_ms * self.surface_temperature
            + gains_floor;
        node.temperature = numer / denom;
    }

    // Internal mass node
    {
        let node = &mut m.internal;
        let t_env_avg = (m.wall.temperature + m.roof.temperature + m.floor.temperature) / 3.0;
        let h_me = node.h_tr_me;
        let denom = node.capacitance / dt + h_is + h_me;
        let numer = node.capacitance / dt * node.temperature
            + h_is * self.zone_temperature
            + h_me * t_env_avg
            + gains_internal;
        node.temperature = numer / denom;
    }
}
```

### 4.5 `src/sim/multi_node_hvac_runner.rs` — Remove or Deprecate

After the changes above, `MultiNodeHvacRunner` is dead code. Options:

1. **Remove it** — its functionality is now in `step_physics_9r4c` directly.
2. **Deprecate with `#[deprecated]`** — keep for reference, mark for removal.
3. **Repurpose as warm-up wrapper** — if warm-up is needed for the multi-node solver, wrap the new `compute_zone_air_temperature` + `compute_hvac_demand` in the existing warm-up logic.

Recommendation: **Option 2 (deprecate)** for this PR. Clean removal in a follow-up.

### 4.6 Gain Distribution for Multi-Node Solver

In `step_physics_9r4c`, the gains must be distributed to the correct multi-node solver nodes. The current ISO 13790 distribution splits gains into `phi_ia` (air), `phi_st` (surface), `phi_m` (mass). For the multi-node solver:

- `phi_ia` → air node (used in `compute_zone_air_temperature`)
- `phi_st` → distributed to envelope nodes proportional to their `h_tr_ms`
- `phi_m` → internal mass node

```rust
// Distribute phi_st to envelope nodes proportional to h_tr_ms
let h_ms_w = solver.mass.wall.h_tr_ms;
let h_ms_r = solver.mass.roof.h_tr_ms;
let h_ms_f = solver.mass.floor.h_tr_ms;
let h_ms_total = h_ms_w + h_ms_r + h_ms_f;

let phi_st_w = phi_st * h_ms_w / h_ms_total;
let phi_st_r = phi_st * h_ms_r / h_ms_total;
let phi_st_f = phi_st * h_ms_f / h_ms_total;

solver.step_with_gains(dt, phi_st_w, phi_st_r, phi_st_f, phi_m);
```

---

## 5. Physics Validation

### 5.1 Energy Balance Check

At each timestep, the following energy balance must hold:

```
Q_stored = Q_gains - Q_losses
         = (phi_ia + phi_st + phi_m) - (h_ve × (T_air - T_out) + h_tr_em × (T_node - T_ext))
```

Verify by checking that the sum of all heat fluxes through the multi-node solver equals the change in stored energy:

```
ΔE_stored = Σ(C_k × ΔT_k) for k ∈ {wall, roof, floor, internal}
```

### 5.2 Heat Transfer Direction Check

- When `T_air > T_out`: heat flows OUT through ventilation → `h_ve × (T_air - T_out) > 0`
- When `T_ext > T_node`: heat flows IN through envelope → `h_tr_em × (T_ext - T_node) > 0`
- When `T_air > T_surface`: heat flows from air to mass → `h_tr_is × (T_air - T_surface) > 0`

### 5.3 ASHRAE 140 Compliance

The proposed approach is consistent with ASHRAE 140 §5.2.1, which specifies:
- Zone air temperature is the primary output
- HVAC energy is computed from the zone air temperature vs setpoints
- Surface temperatures are computed from conduction/convection balance
- The multi-node approach is exactly what EnergyPlus, TRNSYS, and DOE-2 use internally

### 5.4 900FF Specific Check

For 900FF (no HVAC), the zone temperature should be:
```
T_air = (h_tr_is × T_s + h_ve × T_out + phi_ia) / (h_tr_is + h_ve)
```

Where `T_s` is the conductance-weighted surface temperature. During summer peak:
- `T_ext_roof` ≈ 60-70°C (sol-air temp with solar on dark roof)
- `T_ext_wall` ≈ 40-50°C (sol-air temp with solar on south wall)
- `T_ext_floor` ≈ 10°C (ground)
- Roof node temp rises → `T_s` rises → `T_air` rises → max temp should be 41-46°C

The existing multi-node solver correctly computes high roof temperatures when given proper sol-air temps. The problem is that `compute_zone_air_temperature` was never connected. Once connected, 900FF should pass.

---

## 6. Implementation Order

### Phase 1: Foundation (non-breaking)

| Step | File | Change | Risk |
|------|------|--------|------|
| 1.1 | `multi_node_solver.rs` | Add `compute_zone_air_temperature()` method | Zero — additive |
| 1.2 | `multi_node_solver.rs` | Add `compute_hvac_demand()` method | Zero — additive |
| 1.3 | `multi_node_solver.rs` | Add `step_with_gains()` method | Zero — additive |
| 1.4 | `multi_node_solver.rs` | Add tests for new methods | Zero — additive |

### Phase 2: Integration (changes 9R4C path only)

| Step | File | Change | Risk |
|------|------|--------|------|
| 2.1 | `thermal_model_physics.rs` | Replace 5R1C `t_i_free` computation with multi-node `compute_zone_air_temperature()` | **Medium** — only affects 9R4C path |
| 2.2 | `thermal_model_physics.rs` | Replace HVAC demand with multi-node `compute_hvac_demand()` | **Medium** — only affects 9R4C path |
| 2.3 | `thermal_model_physics.rs` | Replace gain distribution with per-node `step_with_gains()` | **Medium** |
| 2.4 | `thermal_model_physics.rs` | Fix mass_temperature feedback to conductance-weighted average | **Low** — only diagnostics/inter-zone |

### Phase 3: Cleanup

| Step | File | Change | Risk |
|------|------|--------|------|
| 3.1 | `multi_node_hvac_runner.rs` | Deprecate with `#[deprecated]` | Zero |
| 3.2 | `thermal_model_physics.rs` | Remove dead 5R1C code from `step_physics_9r4c` | Low |
| 3.3 | Tests | Run full ASHRAE 140 suite, verify all cases pass | — |

### Dependencies

```
Phase 1 (all parallel) → Phase 2.1-2.2 (sequential) → Phase 2.3 → Phase 2.4 → Phase 3
```

Steps 2.1 and 2.2 must be done together because `t_i_free` and HVAC demand are coupled. Step 2.3 (gain distribution) can be done separately but should precede 2.4.

---

## 7. Risk Assessment

### 7.1 Regression Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Case 600/610/620/650 regression | **Very Low** — these use `step_physics_5r1c`, not `step_physics_9r4c` | High | Run full test suite after each phase |
| Annual heating regression | **Low** — winter conditions dominated by ventilation losses | Medium | Verify heating stays in 1.17–2.04 MWh range |
| 900FF regression in wrong direction | **Low** — physics is correct, but numerical stability could be an issue | High | Add convergence check on multi-node solver |
| Inter-zone coupling break | **Low** — ASHRAE 140 is single-zone | Low | Keep `mass_temperatures` update for multi-zone scenarios |

### 7.2 Numerical Stability

The multi-node solver uses backward Euler, which is unconditionally stable. However, the air temperature computation is an explicit algebraic equation. If `h_tr_is + h_ve` is very small (near-zero ventilation), the air temperature could be unstable. Mitigation:

```rust
let denom = h_is + h_ve;
if denom < 1e-6 {
    // Fallback: use surface temperature
    return t_surface;
}
```

### 7.3 Gain Distribution Accuracy

The per-node gain distribution (phi_st proportional to h_tr_ms) assumes uniform radiative exchange. For Case 900, this is reasonable because:
- Window solar gains hit the floor directly (high h_tr_ms for floor)
- Internal radiative gains distribute evenly to all surfaces
- The floor has the largest h_tr_ms in the typical Case 900 configuration

### 7.4 Verification Checklist

After implementation, verify:

- [ ] All 5 Case 900 metrics within reference ranges
- [ ] Case 600 annual heating: 0.94–1.96 MWh
- [ ] Case 600 annual cooling: 2.65–5.10 MWh
- [ ] Case 600 peak heating: 1.30–2.80 kW
- [ ] Case 600 peak cooling: 1.70–3.60 kW
- [ ] Case 610 metrics within range
- [ ] Case 620 metrics within range
- [ ] Case 650 metrics within range
- [ ] 900FF max temp: 41.8–46.4°C
- [ ] 900FF min temp: 18.1–22.6°C
- [ ] Energy conservation test passes (energy in = energy out + stored)
- [ ] All existing unit tests pass
- [ ] No regressions in `cargo test`

---

## 8. Why Previous PRs Didn't Fix This

| PR | What It Did | Why It Didn't Fix the Root Cause |
|-----|-------------|------|
| #863 (sol-air temps) | Added per-surface sol-air temps to multi-node solver | Correct physics but multi-node solver output was never used for zone temp |
| #864 (gain distribution) | Added per-surface gain distribution functions | Correct physics but gains never reached the multi-node solver's backward Euler step |
| #865 (warm-up period) | Added MultiNodeHvacRunner with warm-up | Runner was never called from step_physics_9r4c |
| #866 (energy override) | Added energy accumulation to 9R4C path | Accumulation is correct but the HVAC demand was still from 5R1C |

Each PR added correct infrastructure but none connected it to the primary thermal calculation. This plan connects all the pieces.

---

## 9. Summary of Changes by File

| File | Lines Changed | Nature |
|------|--------------|--------|
| `src/physics/multi_node_solver.rs` | ~80 lines added | New methods: `compute_zone_air_temperature`, `compute_hvac_demand`, `step_with_gains`, `step_backward_euler_with_gains` |
| `src/sim/thermal_model_physics.rs` | ~130 lines changed in `step_physics_9r4c` | Replace 5R1C sensitivity with multi-node air balance; fix mass temp feedback |
| `src/sim/multi_node_hvac_runner.rs` | ~5 lines added | Deprecation annotation |
| Tests | ~50 lines added | Unit tests for new methods |

**Total estimated change**: ~265 lines across 3 files. No new files. No changes to 5R1C, 6R2C, or 8R3C paths.
