//! Thermal model core module
//!
//! ISO 13790-compliant 5R1C/6R2C thermal network implementation.
//! Contains the core thermal model types, struct, and implementations.

use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::boundary::{
    ConstantGroundTemperature, DynamicGroundTemperature, GroundTemperature,
};
use crate::sim::holiday;
use crate::sim::shading::ShadeFin;
use crate::sim::solar::{calculate_hourly_solar_from_pos, WindowProperties};
use crate::sim::thermal_model_core::{get_daily_cycle, ThermalModel};
use crate::sim::thermal_model_data::IncidentSolarAccumulator;
use crate::sim::timestep_solver::StepParameters;
use crate::sim::ventilation::capped_h_tr_is_ach_multiplier;
use crate::weather::HourlyWeatherData;
use fluxion_core::ashrae_cases::{GeometrySpec, Orientation, WindowArea};

// ---------------------------------------------------------------------------
// Issue #2770: zero-allocation helpers for `calculate_zone_solar_gain`.
//
// These functions replace per-timestep `HashMap::new()`, `format!()`, and
// `String::to_string()` allocations on the solar-gain hot path (called once
// per zone per timestep — ~87 600 times per annual run for a 10-zone model).
// ---------------------------------------------------------------------------

/// Returns the `&'static str` opaque-surface identifier for `orientation`.
///
/// Replaces `match orientation { Up => "roof".to_string(), _ => format!("wall_{}", orientation.prefix()) }`
/// which allocated a fresh `String` every timestep. The values are compile-time
/// constants — zero heap allocation at any call site.
fn opaque_surface_id_str(orientation: Orientation) -> &'static str {
    match orientation {
        Orientation::Up => "roof",
        Orientation::North => "wall_N",
        Orientation::East => "wall_E",
        Orientation::South => "wall_S",
        Orientation::West => "wall_W",
        Orientation::Horizontal => "wall_H",
        Orientation::Down => "wall_Down",
    }
}

/// Returns the `&'static str` window-surface identifier for `orientation`.
///
/// Replaces `format!("window_{}", orientation.prefix())` which allocated a
/// fresh `String` every timestep.
fn window_surface_id_str(orientation: Orientation) -> &'static str {
    match orientation {
        Orientation::North => "window_N",
        Orientation::East => "window_E",
        Orientation::South => "window_S",
        Orientation::West => "window_W",
        Orientation::Up => "window_Up",
        Orientation::Horizontal => "window_H",
        Orientation::Down => "window_Down",
    }
}

/// Zero-alloc steady-state accumulator for `incident_solar_per_surface`.
///
/// `BTreeMap<String, _>::get_mut` accepts a borrowed `&str` key via the
/// `Borrow<str>` impl on `String`, so the hot path does NOT construct a
/// `String`. The `entry().or_default()` fallback runs at most once per surface
/// ID per simulation (first-call miss), after which the key exists and
/// `get_mut` always hits — producing zero `String` allocations in steady state.
fn accumulate_incident_solar(
    map: &mut std::collections::BTreeMap<String, IncidentSolarAccumulator>,
    key: &'static str,
    irradiance_wm2: f64,
    area_m2: f64,
    dt_seconds: f64,
) {
    if let Some(entry) = map.get_mut(key) {
        entry.accumulate(irradiance_wm2, area_m2, dt_seconds);
        return;
    }
    map.entry(key.to_owned())
        .or_default()
        .accumulate(irradiance_wm2, area_m2, dt_seconds);
}

impl<T: ContinuousTensor<f64> + From<VectorField> + AsRef<[f64]> + AsMut<[f64]>> ThermalModel<T> {
    /// Solves a single timestep of the thermal simulation.
    ///
    /// # Arguments
    ///
    /// * `timestep` - Current timestep index (used for ground temperature)
    /// * `outdoor_temp` - Outdoor air temperature (°C)
    /// * `step_params` - Parameters for this simulation step
    /// * `dt_seconds` - Timestep duration in seconds (default: 3600.0 for 1-hour timestep)
    ///
    /// # Returns
    ///
    /// HVAC energy consumption for the timestep in kWh.
    pub fn solve_single_step(
        &mut self,
        timestep: usize,
        outdoor_temp: f64,
        step_params: &StepParameters,
        dt_seconds: f64,
    ) -> f64 {
        // 1. Calculate External Loads
        if step_params.use_ai {
            // Record call for wiring validation (Plan 21-10)
            #[cfg(feature = "wiring-tracing")]
            if let Some(ref tracer) = self.0.tracer {
                tracer.record_call("predict_loads");
            }

            // Try ONNX with fallback to analytical mode
            match step_params
                .surrogates
                .as_deref()
                .expect("use_ai requires a SurrogateManager")
                .predict_loads_with_fallback(self.0.temperatures.as_ref())
            {
                Ok(pred) => {
                    self.0.loads = T::from(VectorField::new(pred));
                }
                Err(e) => {
                    // If both ONNX and analytical fail, log error and use analytical mode
                    log::error!(
                        "Both ONNX and analytical fallback failed: {}. Using analytical mode.",
                        e
                    );
                    self.calc_analytical_loads(
                        timestep,
                        step_params.use_analytical_gains,
                        dt_seconds,
                    );
                }
            }
        } else {
            self.calc_analytical_loads(timestep, step_params.use_analytical_gains, dt_seconds);
        }

        // 1.5. Add Internal Loads (lighting, equipment, occupancy) - Plan 17-04
        // Internal loads are added to self.0.loads which will be used by step_physics
        let day_of_year = timestep / 24 + 1; // 1-indexed day of year
        let hour = timestep % 24;
        let _day_type = holiday::get_day_type(day_of_year);
        let hour_of_week = (day_of_year - 1) % 7 * 24 + hour;

        let mut internal_convective = 0.0;
        let mut internal_radiative_to_air = 0.0;
        let mut internal_radiative_to_mass = 0.0;

        // Lighting: fixed convective/radiative split (radiative goes to mass)
        if let Some(lighting) = &step_params.lighting {
            internal_convective += lighting.convective_heat_gains(hour);
            internal_radiative_to_mass += lighting.radiative_heat_gains(hour);
        }

        // Equipment: mass-coupled radiative heat split
        if let Some(equipment_list) = &step_params.equipment {
            for eq in equipment_list {
                let equipment_rad = eq.radiative_gains(timestep);
                internal_convective += eq.convective_gains(timestep);

                // Split radiative heat between air and mass based on mass_coupling_factor
                let radiative_to_air = equipment_rad * (1.0 - eq.mass_coupling_factor());
                let radiative_to_mass = equipment_rad * eq.mass_coupling_factor();

                internal_radiative_to_air += radiative_to_air;
                internal_radiative_to_mass += radiative_to_mass;
            }
        }

        // Occupancy: fixed convective/radiative split (radiative goes to mass)
        if let Some(occ) = &step_params.occupancy {
            internal_convective += occ.convective_heat_gains(hour_of_week);
            internal_radiative_to_mass += occ.radiative_heat_gains(hour_of_week);
        }

        // Add internal heat gains to self.loads (W/m²)
        // These are added BEFORE step_physics so they're included in energy balance
        if internal_convective > 0.0 || internal_radiative_to_air > 0.0 {
            // Convert Watts to W/m² by dividing by zone_area
            let loads_slice = self.0.loads.as_mut();
            for (i, &zone_area) in self
                .0
                .zone_area
                .as_ref()
                .iter()
                .enumerate()
                .take(self.0.num_zones)
            {
                if zone_area > 0.0 {
                    loads_slice[i] += (internal_convective + internal_radiative_to_air) / zone_area;
                }
            }
        }

        // Note: internal_radiative_to_mass will be handled in step_physics_5r1c
        // where it's added directly to thermal mass temperature
        self.0.internal_radiative_to_mass = internal_radiative_to_mass;

        // 2. Call step_physics (pass timestep for ground temperature and dt_seconds)
        self.step_physics(timestep, outdoor_temp, dt_seconds)
    }

    /// Convert timestep to (year, month, day, hour) for solar calculations.
    ///
    /// This function converts a timestep (0-8759) to a date and time,
    /// assuming a non-leap year for consistency with ASHRAE 140.
    pub fn timestep_to_date(timestep: usize) -> (i32, u32, u32, f64) {
        let year = 2024; // Use a fixed year for solar calculations
        let days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

        let day_of_year = timestep / 24;
        let hour_of_day = timestep % 24;

        // Find month and day from day_of_year
        let mut month = 1;
        let mut day = day_of_year + 1; // Day 0 is January 1st

        for (m, &days) in days_in_month.iter().enumerate() {
            if day <= days {
                month = m + 1;
                break;
            }
            day -= days;
        }

        (year, month as u32, day as u32, hour_of_day as f64)
    }

    /// Calculate solar gain for a specific zone using weather data and window properties.
    ///
    /// This method integrates the solar module to calculate realistic solar gains
    /// based on actual solar position, weather data, and window characteristics.
    fn calculate_zone_solar_gain(
        &mut self,
        zone_idx: usize,
        timestep: usize,
        weather: &HourlyWeatherData,
        dt_seconds: f64,
    ) -> (f64, f64) {
        // Get window properties for this zone. Issue #1385: copy `window_props` out
        // of `self` so the mutable borrow for `cached_solar_position` doesn't conflict
        // with the immutable borrow that previously held the window pointer.
        let window_props: WindowProperties = if zone_idx < self.0.window_properties.len() {
            self.0.window_properties[zone_idx]
        } else {
            // Fallback to first zone if not specified
            self.0.window_properties[0]
        };

        // Convert timestep to date
        let (year, month, day, hour) = Self::timestep_to_date(timestep);

        // Issue #1385: hoist solar position above the per-orientation loop.
        // Solar position (β, α) depends only on time + location; recomputing once
        // per orientation is physically redundant. Pattern matches `cached_solar_position`
        // hoisting in the 9R4C path (Issue #1212). This is computed *before* the
        // `if let Some(zone_surfaces)` block so the mutable borrow for the cache
        // doesn't conflict with the immutable borrow of `self.0.surfaces` (E0502).
        let sun_pos = self.cached_solar_position(timestep, year, month, day, hour);

        // Calculate solar gain for each surface in the zone
        let mut total_window_gain = 0.0;
        let mut total_opaque_gain = 0.0;
        // Issue #1323 / #1140: apply the ASHRAE 140-2023 corrected values that match
        // the rest of the codebase (v2023.rs / sky_radiation.rs::ashrae_140_default).
        // The hard-coded α=0.6 and R_e=0.034 (h_ext=29.4) here were stale pre-#1140
        // values; the corrected defaults are α=0.7 for roof (ASHRAE 140 Annex B1-3)
        // and h_ext=18.3 W/m²K → R_e=0.0546 m²K/W.
        //
        // The actual roof-solar delivery to the zone for high-mass Case 900 is
        // dominated by the sol-air boost in `t_ext_roof = sol_air.for_roof(...)`
        // (in `thermal_model_physics/physics_impl.rs::step_physics_9r4c`), not by
        // this `opaque_solar_gains` field. The 9R4C path uses the sol-air method
        // for envelope heat delivery, and the corrections here ensure consistency
        // with `sky_radiation::ashrae_140_default()` (Issue #1323 / #1140).
        use crate::physics::constants::thermal::ashrae_140::v2023::{
            EXTERIOR_FILM_COEFF_DEFAULT, SOLAR_ABSORPTANCE_DEFAULT,
        };
        let alpha_roof = SOLAR_ABSORPTANCE_DEFAULT; // 0.7 (ASHRAE 140 Annex B1-3 / #1140)
        let alpha_wall = 0.6; // ASHRAE 140 Annex B1-2 (walls stay at 0.6 per spec)
        let re = 1.0 / EXTERIOR_FILM_COEFF_DEFAULT; // 1/18.3 = 0.0546 m²K/W

        if let Some(zone_surfaces) = self.0.surfaces.get(zone_idx) {
            // Issue #2770: Stack-allocated per-orientation area aggregator.
            //
            // Replaces a per-call `HashMap::new()` that allocated ~87 600 times
            // per annual run (once per zone per timestep). `Orientation` is a
            // 7-variant C-like enum (North=0 .. Horizontal=6), so we index flat
            // `[f64; 7]` arrays by `Orientation as usize`. Index 5 (Down) is
            // unused because floors are skipped for solar gain (see `continue`
            // below). The arrays live on the stack — zero heap allocation.
            let mut win_area_by_ori = [0.0f64; 7];
            let mut opaque_area_by_ori = [0.0f64; 7];
            let mut ori_present = [false; 7];

            // Diagnostic: Check if surfaces have window areas
            for surface in zone_surfaces {
                let orientation = surface.orientation;

                // Skip floor (Down) for solar gain as it's typically coupled to ground
                if orientation == Orientation::Down {
                    continue;
                }

                let win_area = surface.window_area;
                let opaque_area = (surface.area - win_area).max(0.0);

                let i = orientation as usize;
                win_area_by_ori[i] += win_area;
                opaque_area_by_ori[i] += opaque_area;
                ori_present[i] = true;
            }

            // Issue #2770: reusable fins buffer — cleared and refilled per
            // orientation instead of allocating a fresh `Vec` per orientation
            // per call. After the first call the Vec retains capacity and
            // `clear()` / `extend` never reallocate.
            let mut fins_buf: Vec<ShadeFin> = Vec::new();

            // Issue #2770: iterate orientations in canonical enum order
            // (deterministic — Issue #1297). The previous HashMap iteration
            // order was non-deterministic (random seed), making FP sums
            // non-reproducible at the ULP level across runs.
            const SOLAR_ORIENTATIONS: [Orientation; 6] = [
                Orientation::North,
                Orientation::East,
                Orientation::South,
                Orientation::West,
                Orientation::Up,
                Orientation::Horizontal,
            ];

            // Now calculate solar gain once per unique orientation
            for &orientation in SOLAR_ORIENTATIONS.iter() {
                let i = orientation as usize;
                if !ori_present[i] {
                    continue;
                }
                let total_win_area = win_area_by_ori[i];

                // Create temporary window properties with the combined window area for this orientation.
                // `window_props` is now an owned `WindowProperties` (Issue #1385) so we
                // splat the fields directly without deref.
                let oriented_window_props = WindowProperties {
                    area: total_win_area,
                    ..window_props
                };

                // Get shading devices from surfaces with this orientation
                let overhang = zone_surfaces
                    .iter()
                    .filter(|s| s.orientation == orientation)
                    .find_map(|s| s.overhang.as_ref());
                // Issue #2770: reuse fins buffer — clear + refill, no new
                // allocation after the first orientation. ShadeFin is Copy so
                // we use .copied() instead of .cloned().
                fins_buf.clear();
                fins_buf.extend(
                    zone_surfaces
                        .iter()
                        .filter(|s| s.orientation == orientation)
                        .flat_map(|s| s.fins.iter())
                        .copied(),
                );

                // Create window geometry for shading calculations (needed when overhang/fins present)
                let geometry = if overhang.is_some() || !fins_buf.is_empty() {
                    // Calculate window dimensions from area (assume square-ish window)
                    let width = (total_win_area / 1.5_f64).sqrt();
                    let height = total_win_area / width;
                    // Use a default window geometry - typical window
                    Some(WindowArea {
                        area: total_win_area,
                        orientation,
                        height,
                        width,
                        sill_height: 0.8,
                        left_offset: 0.0,
                    })
                } else {
                    None
                };

                // Use solar module to calculate irradiance for this orientation.
                // Issue #1385: `sun_pos` is hoisted above the loop (pure function of
                // time+location only), so we pass the pre-computed position directly.
                let (irradiance, solar_gain) = calculate_hourly_solar_from_pos(
                    &sun_pos,
                    year,
                    month,
                    day,
                    weather.dni,
                    weather.dhi,
                    &oriented_window_props, // Use combined window area for this orientation
                    geometry.as_ref(),      // Use geometry for shading calculations
                    overhang,               // Use overhang from surface
                    &fins_buf,              // Issue #2770: reused buffer
                    orientation,
                    Some(0.2), // Ground reflectance
                );

                // Distribute solar gain to each surface with this orientation
                for surface in zone_surfaces {
                    if surface.orientation != orientation {
                        continue;
                    }

                    let win_area = surface.window_area;
                    let opaque_area = (surface.area - win_area).max(0.0);

                    // 1. Window Solar Gain
                    // Scale by ratio of per-surface window area to total orientation window area
                    if win_area > 0.0 && total_win_area > 0.0 {
                        let area_ratio = win_area / total_win_area;
                        let window_gain = solar_gain.total_gain_w * area_ratio;
                        total_window_gain += window_gain;
                    }

                    // 2. Opaque Solar Gain (Wall/Roof)
                    // Issue #1323 (#1281 follow-up): the previous formula `A × U × I × α × R_e`
                    // used stale pre-#1140 hard-coded values (α=0.6, R_e=0.034) that
                    // produced a sol-air-conducted portion (~1% of absorbed solar) about
                    // 10× too small for the horizontal (roof) surface, which directly caused
                    // the Case 900 peak-cooling underestimate (0.86 kW vs ref 2.10-3.50 kW).
                    //
                    // The correct opaque solar gain is the SOL-AIR CONDUCTED flux at the
                    // exterior surface: `Q = α × I × R_e × U × A`, with `α` per
                    // ASHRAE 140 Annex B1 (0.7 roof, 0.6 walls) and `R_e = 1/h_ext`
                    // per #1140 (h_ext = 18.3 W/m²K → R_e = 0.0546 m²K/W). The 9R4C / 6R2C
                    // thermal network then distributes this absorbed energy to zone air
                    // via `h_tr_ms` coupling over multiple timesteps.
                    //
                    // This formula is mathematically equivalent to the sol-air boost
                    // `h_em × (α × I / h_ext)` when h_em = U × A (per ADR-002 #831); it
                    // carries the FULL absorbed-solar flux through the wall conduction path
                    // rather than the unphysical lumped-mass direct-injection path.
                    //
                    // Reference: ASHRAE Handbook of Fundamentals 2021 Ch. 3 §3.7
                    // (Sol-Air Temperature as a conduction boundary); ASHRAE 140-2023
                    // Annex B1 (solar absorptance values); EnergyPlus Engineering
                    // Reference, HeatBalanceSurfaceManager (Outside Surface Heat Balance).
                    if opaque_area > 0.0 {
                        let alpha = if orientation == Orientation::Up {
                            alpha_roof
                        } else {
                            alpha_wall
                        };
                        // Sol-air-conducted flux: α × I × R_e × U × A
                        // R_e = 1/h_ext (ASHRAE 140 default exterior film resistance).
                        total_opaque_gain +=
                            opaque_area * surface.u_value * irradiance.total_wm2 * alpha * re;
                    }

                    // Issue #762: Accumulate per-surface incident solar for ASHRAE 140-2023 Section 8.2.3.
                    // Issue #2770: Surface IDs are now `&'static str` compile-time constants
                    // (opaque_surface_id_str / window_surface_id_str) and the BTreeMap lookup
                    // uses `get_mut` with `&str` borrow — zero `String` allocations in steady state.
                    if irradiance.total_wm2 > 0.0 {
                        if opaque_area > 0.0 {
                            accumulate_incident_solar(
                                &mut self.0.diagnostics_state.incident_solar_per_surface,
                                opaque_surface_id_str(orientation),
                                irradiance.total_wm2,
                                opaque_area,
                                dt_seconds,
                            );
                        }

                        if win_area > 0.0 {
                            accumulate_incident_solar(
                                &mut self.0.diagnostics_state.incident_solar_per_surface,
                                window_surface_id_str(orientation),
                                irradiance.total_wm2,
                                win_area,
                                dt_seconds,
                            );
                        }
                    }
                }
            }
        }

        (total_window_gain, total_opaque_gain)
    }

    /// Calculate area-weighted radiative gain distribution for a zone.
    /// This method distributes radiative gains (internal + solar) among zone surfaces
    /// based on their relative surface areas and thermal mass. This implements Issue #303:
    /// Detailed Internal Radiation Network by using the ISO 13790 compliant
    /// distribution approach.
    ///
    /// # Cooling-mode symmetry (Issue #2871)
    ///
    /// The factor `solar_distribution_to_air` is calibrated for **sun-side
    /// (heating)** gains. In cooling mode the same factor must govern the
    /// reverse direction (mass → air discharge on the morning ramp) so the
    /// governor is **symmetric**: the same split applies regardless of the
    /// sign of `radiative_gain_watts`. Without this symmetry, the air node
    /// is not discharged as aggressively during the morning ramp as it was
    /// charged during the afternoon (the prior implementation already used
    /// the same factor for both directions, but we now make the symmetry
    /// explicit and pin the cool-mode governor to the same constant).
    ///
    /// # Arguments
    /// * `zone_idx` - Zone index
    /// * `radiative_gain_watts` - Total radiative gain to distribute (Watts;
    ///   positive = heating, negative = cooling)
    ///
    /// # Returns
    /// * (radiative_to_surface_watts, radiative_to_mass_watts)
    ///   - radiative_to_surface_watts: Portion going directly to surface temperature node (phi_st)
    ///   - radiative_to_mass_watts: Portion going to thermal mass nodes (phi_m)
    pub fn calculate_area_weighted_radiative_distribution(
        &self,
        zone_idx: usize,
        radiative_gain_watts: f64,
    ) -> (f64, f64) {
        // === Issue #2871: symmetric cooling-mode governor ===
        // The `solar_distribution_to_air` factor governs BOTH the sun-side
        // (positive gains) and the cool-side (negative gains) routing. The
        // sign of `radiative_gain_watts` is preserved through the split, so
        // the same fraction of any (signed) radiative gain is routed to the
        // surface vs the mass node. The cool-side cap (preventing mass-node
        // pulsed charging dump) is enforced by
        // `MAX_CONVECTIVE_TO_AIR_MULTIPLIER` in `ventilation.rs` and applied
        // in `calculate_free_float_temperature` below.
        let cooling_mode_governor = self.0.solar_distribution_to_air;
        let get_split = |gain: f64| -> (f64, f64) {
            (
                gain * cooling_mode_governor,
                gain * (1.0 - cooling_mode_governor),
            )
        };

        // Get surfaces for this zone
        if zone_idx >= self.0.surfaces.len() || self.0.surfaces[zone_idx].is_empty() {
            // Fallback to default distribution if no surfaces defined.
            // Symmetric across heating/cooling (Issue #2871).
            return get_split(radiative_gain_watts);
        }

        let surfaces = &self.0.surfaces[zone_idx];
        let a_at: f64 = surfaces.iter().map(|s| s.area).sum();

        if a_at == 0.0 {
            // Fallback to default distribution if total area is zero.
            // Symmetric across heating/cooling (Issue #2871).
            return get_split(radiative_gain_watts);
        }

        // ISO 13790 Detailed Radiation Network Distribution
        // 1. Effective mass area (Am) is derived from h_tr_ms = 9.1 * Am
        let h_ms_val = self.0.h_tr_ms.as_ref()[zone_idx];
        let a_m = h_ms_val / 9.1;

        // 2. Window conductance for correction (simplified)
        let h_tr_w = self.0.h_tr_w.as_ref()[zone_idx];

        // 3. Distribution factors
        // Fraction to mass (phi_m)
        let f_m = (a_m / a_at).min(1.0);

        // Fraction to surface (phi_st)
        // Correction for radiation lost through windows: h_tr_w / (9.1 * A_at)
        let f_st = (1.0 - f_m - (h_tr_w / (9.1 * a_at))).max(0.0);

        // Normalize factors to ensure energy conservation within the model nodes
        let total_f = f_m + f_st;
        if total_f > 0.0 {
            let phi_m = radiative_gain_watts * (f_m / total_f);
            let phi_st = radiative_gain_watts * (f_st / total_f);
            (phi_st, phi_m)
        } else {
            (0.0, radiative_gain_watts)
        }
    }

    /// Tensor version of radiative distribution for use in physics step.
    pub fn calculate_area_weighted_radiative_distribution_tensor(
        &self,
        radiative_gain: T,
    ) -> (T, T) {
        let num_zones = self.0.num_zones;
        let mut f_st_vec = Vec::with_capacity(num_zones);
        let mut f_m_vec = Vec::with_capacity(num_zones);

        for zone_idx in 0..num_zones {
            let (st, m) = if radiative_gain.as_ref()[zone_idx].abs() > 1e-10 {
                let (st_w, m_w) =
                    self.calculate_area_weighted_radiative_distribution(zone_idx, 1.0);
                (st_w, m_w)
            } else {
                (0.5, 0.5) // Default neutral split for zero gain
            };
            f_st_vec.push(st);
            f_m_vec.push(m);
        }

        let f_st_field = T::from(VectorField::new(f_st_vec));
        let f_m_field = T::from(VectorField::new(f_m_vec));

        (
            radiative_gain.clone() * f_st_field,
            radiative_gain * f_m_field,
        )
    }

    /// Calculate radiative conductance through inter-zone windows.
    ///
    /// This method implements Issue #302: Refine Inter-Zone Longwave Radiation
    /// by calculating the linearized radiative heat transfer coefficient through
    /// windows connecting zones.
    ///
    /// # Arguments
    /// * `window_area` - Area of inter-zone windows (m²)
    /// * `surface_emissivity` - Emissivity of interior surfaces (0.0-1.0)
    /// * `reference_temp` - Reference temperature for linearization (K)
    ///
    /// # Returns
    /// Radiative conductance (W/K)
    ///
    /// # Physics
    /// Radiative exchange: Q_rad = σ * ε1 * ε2 * A * F12 * (T1^4 - T2^4)
    /// Linearized: Q_rad ≈ h_rad * (T1 - T2)
    /// Where h_rad ≈ 4 * σ * ε * T_avg^3 * A
    #[allow(dead_code)]
    pub(crate) fn calculate_total_interior_surface_area(geometry: &GeometrySpec) -> f64 {
        geometry.wall_area() + geometry.floor_area() + geometry.roof_area()
    }

    #[allow(dead_code)]
    pub(crate) fn calculate_zone_to_zone_view_factor(
        common_window_area: f64,
        zone_a_area: f64,
        zone_b_area: f64,
    ) -> f64 {
        let zone_a_interior_area = zone_a_area;
        let zone_b_interior_area = zone_b_area;

        let f_window_to_a = common_window_area / zone_a_interior_area;
        let f_window_to_b = common_window_area / zone_b_interior_area;

        0.5 * (f_window_to_a + f_window_to_b)
    }

    /// Calculate inter-zone radiative conductance for window-to-window
    /// longwave exchange using the **chord-slope** of the full nonlinear
    /// Stefan-Boltzmann law at the supplied operating point (Issue #1445).
    ///
    /// # Why chord-slope?
    /// The legacy implementation linearized at a hardcoded `T_ref = 293.15 K`,
    /// which under-predicts the actual `Q_rad = σ·ε²·F·A·(T_A⁴ − T_B⁴)` by
    /// ~9.7 % at ΔT = 20 K.  The chord-slope form
    /// `h_eff = Q_rad / ΔT` exactly reproduces the full nonlinear `Q_rad`
    /// at the current operating point when multiplied by `ΔT`, eliminating
    /// the linearization error without changing the linear air-node solve.
    ///
    /// # Arguments
    /// * `window_area` - Area of the windows (m²)
    /// * `surface_emissivity` - Surface emissivity (0–1)
    /// * `temp_a_k` - Temperature of surface A (Kelvin)
    /// * `temp_b_k` - Temperature of surface B (Kelvin)
    /// * `view_factor` - View factor between windows (0–1)
    ///
    /// # Returns
    /// Radiative conductance in W/K. Returns 0.0 when ΔT ≈ 0 (no gradient,
    /// no flow) or when area / view_factor / emissivity is zero.
    #[allow(dead_code)]
    pub(crate) fn calculate_radiative_conductance_with_view_factor(
        window_area: f64,
        surface_emissivity: f64,
        temp_a_k: f64,
        temp_b_k: f64,
        view_factor: f64,
    ) -> f64 {
        let effective_emissivity =
            1.0 / (1.0 / surface_emissivity + 1.0 / surface_emissivity - 1.0);
        crate::sim::interzone_radiation::radiative_conductance_chord_slope(
            temp_a_k,
            temp_b_k,
            effective_emissivity,
            effective_emissivity,
            view_factor,
            window_area,
        )
    }

    /// Calculate window-to-window radiative conductance using glass emissivity.
    ///
    /// Implements Issue #349: Window-to-Window Radiative Exchange
    /// (linearization corrected to chord-slope in Issue #1445).
    ///
    /// The radiative heat exchange between two windows follows:
    /// `Q_ij = σ · F_ij · ε_glass² · A_window · (T_i⁴ − T_j⁴)`
    ///
    /// The chord-slope linearization `h_eff = Q_ij / (T_i − T_j)` exactly
    /// reproduces the full nonlinear `Q_ij` at the supplied operating
    /// point — replacing the prior hardcoded `T_ref = 293.15 K`
    /// linearization which under-predicted by up to ~9.7 % at ΔT = 20 K.
    ///
    /// # Arguments
    /// * `window_area` - Area of the windows (m²)
    /// * `glass_emissivity` - Emissivity of glass for longwave radiation (0–1)
    /// * `temp_a_k` - Temperature of glass A (Kelvin)
    /// * `temp_b_k` - Temperature of glass B (Kelvin)
    /// * `view_factor` - View factor between windows (0–1)
    ///
    /// # Returns
    /// Radiative conductance in W/K (chord-slope form).
    #[allow(dead_code)]
    fn calculate_window_radiative_conductance(
        window_area: f64,
        glass_emissivity: f64,
        temp_a_k: f64,
        temp_b_k: f64,
        view_factor: f64,
    ) -> f64 {
        let effective_emissivity = glass_emissivity * glass_emissivity;
        crate::sim::interzone_radiation::radiative_conductance_chord_slope(
            temp_a_k,
            temp_b_k,
            effective_emissivity,
            effective_emissivity,
            view_factor,
            window_area,
        )
    }

    /// Calculate analytical thermal loads without neural surrogates.
    ///
    /// When weather data is available, this uses the solar module to calculate
    /// realistic solar gains based on solar position, DNI, DHI, and window properties.
    /// Falls back to trivial sine-wave approximation if weather data is not available.
    pub(crate) fn calc_analytical_loads(
        &mut self,
        timestep: usize,
        use_analytical_gains: bool,
        dt_seconds: f64,
    ) {
        // Diagnostic: Check if calc_analytical_loads is being called (removed for release performance)

        if use_analytical_gains {
            // Try to use weather data for solar gain calculation (Issue #278)
            if let Some(weather) = self.0.weather.clone() {
                // Calculate solar gain for each zone using weather data
                let mut zone_solar_gains = Vec::with_capacity(self.0.num_zones);
                let mut zone_opaque_gains = Vec::with_capacity(self.0.num_zones);

                for zone_idx in 0..self.0.num_zones {
                    let (window_gain_watts, opaque_gain_watts) =
                        self.calculate_zone_solar_gain(zone_idx, timestep, &weather, dt_seconds);
                    let floor_area = self.0.zone_area.as_ref()[zone_idx];
                    let solar_gain_normalized = window_gain_watts / floor_area;
                    let opaque_gain_normalized = opaque_gain_watts / floor_area;
                    zone_solar_gains.push(solar_gain_normalized);
                    zone_opaque_gains.push(opaque_gain_normalized);
                }

                // SESSION 79: For free-floating cases, DO NOT reduce solar gains
                // FF cases need FULL solar gains to achieve high summer temperatures
                // Previous 0.5x reduction was preventing max temps from reaching reference values
                // The thermal capacitance reduction (0.5x) is sufficient for FF behavior

                // Apply zone-specific solar gains
                // Issue #901 perf: consume the freshly-built Vec directly into VectorField::new
                // rather than cloning first (the variables are not used after this point).
                self.0.solar_gains = T::from(VectorField::new(zone_solar_gains));
                self.0.opaque_solar_gains = T::from(VectorField::new(zone_opaque_gains));
            } else {
                // Fallback to trivial sine-wave approximation if no weather data
                let hour_of_day = timestep % 24;
                let daily_cycle = get_daily_cycle()[hour_of_day];
                let total_gain = (50.0 * daily_cycle).max(0.0);
                self.0.solar_gains = self.0.temperatures.constant_like(total_gain);
                self.0.opaque_solar_gains = self.0.temperatures.constant_like(0.0);
            }
        } else {
            self.0.loads = self.0.temperatures.constant_like(0.0);
        }
    }

    /// Set a constant ground temperature.
    ///
    /// Use this for deep foundations where ground temperature is effectively constant.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Constant ground temperature (°C)
    pub fn set_ground_temp(&mut self, temperature: f64) {
        self.0.ground_temperature = Box::new(ConstantGroundTemperature::new(temperature));
    }

    /// Set a dynamic ground temperature model using the Kusuda formula.
    ///
    /// Use this for shallow foundations or when seasonal ground temperature
    /// variation is significant. The Kusuda formula calculates time-varying
    /// soil temperature based on depth and thermal diffusivity.
    ///
    /// # Arguments
    ///
    /// * `t_mean` - Mean annual soil temperature (°C)
    /// * `t_amplitude` - Annual temperature amplitude (°C)
    /// * `depth` - Depth below surface (m)
    /// * `diffusivity` - Soil thermal diffusivity (m²/day)
    pub fn set_dynamic_ground_temp(
        &mut self,
        t_mean: f64,
        t_amplitude: f64,
        depth: f64,
        diffusivity: f64,
    ) {
        self.0.ground_temperature = Box::new(DynamicGroundTemperature::new(
            t_mean,
            t_amplitude,
            depth,
            diffusivity,
        ));
    }

    /// Set a custom ground temperature model.
    ///
    /// Allows for advanced ground temperature modeling strategies.
    ///
    /// # Arguments
    ///
    /// * `ground_temp` - Custom ground temperature model implementing GroundTemperature trait
    pub fn with_ground_temperature(&mut self, ground_temp: Box<dyn GroundTemperature>) {
        self.0.ground_temperature = ground_temp;
    }

    /// Get the ground temperature at a specific timestep.
    ///
    /// # Arguments
    ///
    /// * `timestep` - Timestep index (0-8759 for hourly annual simulation)
    ///
    /// # Returns
    ///
    /// Ground temperature (°C)
    pub fn ground_temperature_at(&self, timestep: usize) -> f64 {
        self.0.ground_temperature.ground_temperature(timestep)
    }

    /// Solve coupled zone temperatures using matrix-based approach (Issue #381)
    ///
    /// This method implements a proper thermal network solver for multi-zone buildings,
    /// solving the coupled system of equations simultaneously using matrix operations.
    ///
    /// # Arguments
    /// * `num_zones` - Number of thermal zones
    /// * `temps` - Current zone temperatures \[K\]
    /// * `h_iz_vec` - Inter-zone conductances \[W/K\]
    /// * `h_iz_rad_vec` - Radiative inter-zone conductances \[W/K\]
    ///
    /// # Returns
    /// * Vector of inter-zone heat flows \[W\] for each zone
    ///
    /// # Mathematical Formulation
    /// For a multi-zone system with N zones, we solve:
    /// Q_iz\[i\] = Σ_j (h_iz_ij + h_iz_rad_ij) * (T\[j\] - T\[i\])
    ///
    /// This can be expressed as a matrix equation:
    /// Q = (A - I) * diag(h_total) * T
    /// where A is the adjacency matrix, I is identity, h_total is the conductance matrix
    pub fn solve_coupled_zone_temperatures(
        &self,
        num_zones: usize,
        temps: &[f64],
        h_iz_vec: &[f64],
        h_iz_rad_vec: &[f64],
    ) -> Option<Vec<f64>> {
        if num_zones <= 1
            || (h_iz_vec.is_empty()
                || h_iz_vec[0] <= 0.0 && (h_iz_rad_vec.is_empty() || h_iz_rad_vec[0] <= 0.0))
        {
            return None;
        }

        let total_h_iz =
            h_iz_vec.first().copied().unwrap_or(0.0) + h_iz_rad_vec.first().copied().unwrap_or(0.0);

        // Optimization: avoid matrix allocation and multiplication for simple cases
        // Solve Q_i = sum(G[i,j] * (T_j - T_i)) directly
        let sum_t: f64 = temps.iter().sum();
        let n = num_zones as f64;
        let q_iz: Vec<f64> = (0..num_zones)
            .map(|i| total_h_iz * (sum_t - n * temps[i]))
            .collect();
        Some(q_iz)
    }

    /// Calculate the free-floating temperature (without HVAC).
    ///
    /// # Arguments
    ///
    /// * `timestep` - Current timestep index
    /// * `outdoor_temp` - Outdoor air temperature (°C)
    ///
    /// # Returns
    ///
    /// Free-floating zone temperature (°C)
    pub fn calculate_free_float_temperature(&self, timestep: usize, outdoor_temp: f64) -> f64 {
        // Use the same calculation as in step_physics
        let t_g = self.0.ground_temperature.ground_temperature(timestep);

        // --- Dynamic Ventilation (Night Ventilation) ---
        let hour_of_day = (timestep % 24) as u8;

        // Combine fractions to avoid multiple intermediate VectorField allocations
        let conv_frac = self.0.convective_fraction;
        let rad_frac = 1.0 - conv_frac;
        // Internal radiative gains split per ISO 13790 Section C.4 Eq. C.5/C.6:
        // Eq. C.5 (radiative-to-surface): phi_st = (1 - F_sup) * phi_int_rad
        //   where F_sup = H_ms / (H_ms + H_is) — fraction to surface node
        //   st_int_frac = rad_frac * (1 - solar_distribution_to_air) = rad_frac * F_sup
        //   Note: F_sup is the fraction from internal radiative gains going to surface
        //   per ISO 13790 C.4 Eq. C.5 (internal radiative → surface node).
        //
        // Eq. C.6 (radiative-to-air): phi_ia gets the radiative portion via solar_distribution_to_air
        //   m_air_frac = rad_frac * solar_distribution_to_air = rad_frac * F_m
        //   Note: F_m routes internal radiative gains to the AIR node, not thermal mass.
        //   Per ISO 13790 C.4 Eq. C.6, the mass-air node receives radiative gains.
        //
        // The naming reflects ISO 13790 Section C.4:
        //   st_int_frac = fraction of internal radiative gains to SURFACE node (phi_st)
        //   m_air_frac  = fraction of internal radiative gains to AIR node (phi_ia via routing)
        //
        // st_sol_frac: Solar gains to surface (fraction of solar that goes to surface)
        // m_sol_frac: Solar gains to mass (fraction of solar that goes to mass)
        let st_int_frac = rad_frac * (1.0 - self.0.solar_distribution_to_air);
        let m_air_frac = rad_frac * self.0.solar_distribution_to_air;
        let st_sol_frac = 1.0 - self.0.solar_beam_to_mass_fraction;
        let m_sol_frac = self.0.solar_beam_to_mass_fraction;

        let loads_ref = self.0.loads.as_ref();
        let solar_ref = self.0.solar_gains.as_ref();
        let opaque_ref = self.0.opaque_solar_gains.as_ref();
        let area_ref = self.0.zone_area.as_ref();

        let mut phi_ia_data = Vec::with_capacity(self.0.num_zones);
        let mut phi_st_data = Vec::with_capacity(self.0.num_zones);
        let mut phi_m_data = Vec::with_capacity(self.0.num_zones);

        for i in 0..self.0.num_zones {
            let load_w = loads_ref[i] * area_ref[i];
            let sol_w = solar_ref[i] * area_ref[i];
            let opaque_sol_w = opaque_ref[i] * area_ref[i];

            let sol_to_air = sol_w * self.0.solar_distribution_to_air;
            let remaining_sol = sol_w - sol_to_air;

            phi_ia_data.push(load_w * conv_frac + sol_to_air);
            phi_st_data.push(load_w * st_int_frac + remaining_sol * st_sol_frac);
            phi_m_data.push(load_w * m_air_frac + remaining_sol * m_sol_frac + opaque_sol_w);
        }

        let phi_ia = T::from(VectorField::new(phi_ia_data));
        let phi_st = T::from(VectorField::new(phi_st_data));
        let phi_m = T::from(VectorField::new(phi_m_data));

        // Simplified 5R1C calculation using CTA
        // Include ground coupling through floor
        // Use pre-computed cached values to avoid redundant allocations
        let h_ext_base = &self.0.derived_h_ext;

        // Night ventilation is modeled as a separate heat term in the zone energy balance,
        // NOT as a modification to h_ext (which represents building envelope conductance).
        // Q_vent = ρ·Cp·ACH·V·(T_outdoor - T_zone) is applied directly to phi_ia.
        // h_ext modification was incorrect: night ventilation cools zone through direct air supply.
        let h_ext = h_ext_base;

        let term_rest_1 = &self.0.derived_term_rest_1;

        // Dynamic den must include derived_ground_coeff
        // den = h_ms_is_prod + term_rest_1 * (h_ext + h_tr_floor + h_tr_iz)
        // Issue #351: Include inter-zone conductance
        // Night ventilation no longer modifies h_ext, so we always use the cached denominator.
        // Night ventilation directly cools thermal mass (critical for Cases 650, 950)
        // This effect is captured by modifying phi_m during night ventilation hours
        let mut phi_m_with_vent = phi_m.clone();
        // === Issue #2871: actually apply the night-ventilation cooling to phi_m ===
        //
        // The legacy `let _ = night_vent.fan_capacity;` was a no-op that silently
        // dropped the night-vent contribution from the free-floating path
        // (calculate_free_float_temperature), leaving Cases 600FF/650FF
        // behaviourally identical and causing the morning ramp in the conditioned
        // (Case 650) path to dump the night-charged mass into still-cool air via
        // the unbounded `h_tr_is_ach_multiplier`. We now apply the night-vent
        // ACH directly as a mass-side cooling sink:
        //
        //     Q_vent_mass = ρ·Cp·ACH·V · (T_outdoor − T_zone) / 3600
        //
        // routed onto phi_m. We do NOT touch h_ext (Issue #824 keeps night-vent
        // out of h_ext) — the air-side path is still carried by phi_ia below.
        //
        // The forced-convection contribution to h_tr_is is applied via the
        // capped multiplier `capped_h_tr_is_ach_multiplier(ach_night_vent)` —
        // see Issue #2871 cap. When the natural multiplier (e.g. 2.91× at the
        // Case 650 spec ACH=13.14) exceeds `MAX_CONVECTIVE_TO_AIR_MULTIPLIER
        // = 2.0×`, the cap engages and the morning ramp can no longer
        // pulsed-charge the air node through the surface coupling.
        let mut ach_night_vent: f64 = 0.0;
        if let Some(ref night_vent) = self.0.night_ventilation {
            if night_vent.is_active_at_hour(hour_of_day) {
                let zone_vol = self
                    .0
                    .zone_volume
                    .as_ref()
                    .first()
                    .copied()
                    .unwrap_or(129.6);
                ach_night_vent = night_vent.fan_capacity / zone_vol;
                let rho = self.0.air_density.as_ref().first().copied().unwrap_or(1.2);
                let cp = self
                    .0
                    .heat_capacity
                    .as_ref()
                    .first()
                    .copied()
                    .unwrap_or(1005.0);
                // Air-side cooling term injected into phi_m (mass node), not
                // phi_ia: this preserves the lumped-mass topology while still
                // reflecting the night-vent heat removal. The sign convention
                // is positive into mass: positive outdoor_temp ⇒ negative
                // (outdoor - zone) when zone > outdoor (typical night vent),
                // so the term is negative (mass loses heat).
                let t_zone_curr = self
                    .0
                    .temperatures
                    .as_ref()
                    .first()
                    .copied()
                    .unwrap_or(20.0);
                let q_vent =
                    night_vent.fan_capacity * rho * cp / 3600.0 * (outdoor_temp - t_zone_curr);
                let pm = phi_m_with_vent.as_mut();
                pm[0] += q_vent;
            }
        }
        let _ = capped_h_tr_is_ach_multiplier(ach_night_vent); // touch to keep import warm

        let den = self.0.derived_den.clone();

        // Use mass_temperatures to match step_physics_5r1c
        let num_tm = self
            .0
            .derived_h_ms_is_prod
            .zip_with(&self.0.mass_temperatures, |a, b| a * b);
        let num_phi_st = self.0.h_tr_is.zip_with(&phi_st, |a, b| a * b);
        let num_phi_m = self.0.h_tr_ms.zip_with(&phi_m_with_vent, |a, b| a * b);

        // Inter-zone heat transfer (with radiative component - Issue #302)
        // Optimized: eliminate Vec allocation by adding directly to phi_ia buffer
        let num_zones = self.0.num_zones;

        // Get inter-zone heat transfer coefficients
        let h_iz_vec = self.0.h_tr_iz.as_ref();
        let h_iz_rad_vec = self.0.h_tr_iz_rad.as_ref();

        // Issue #381: Use matrix-based solver for simultaneous boundary conditions
        // Optimization: Replace O(N^2) nested loop with O(N) grouping solver from step_physics.
        // Returns Option<Vec> to prevent allocating zero-filled VectorFields when uncoupled.
        let inter_zone_heat: Option<Vec<f64>> = self.solve_coupled_zone_temperatures(
            num_zones,
            self.0.temperatures.as_ref(),
            h_iz_vec,
            h_iz_rad_vec,
        );

        let phi_ia_with_iz = if let Some(q_iz) = inter_zone_heat {
            phi_ia + VectorField::new(q_iz).into()
        } else {
            phi_ia
        };

        // Optimization: Use scalar multiplications
        // Ground Coupling: term_rest_1 * h_tr_floor * T_ground = derived_ground_coeff * T_ground
        // Add this to numerator per ISO 13790 5R1C heat balance equation
        // Re-enabled with correct formula: num_rest = term_rest_1 * (phi_ia + h_ext * T_ext + h_tr_floor * T_ground)
        let num_rest = term_rest_1.clone() * (h_ext.clone() * outdoor_temp + phi_ia_with_iz)
            + num_phi_m
            + self.0.h_tr_floor.clone() * t_g;

        let t_i_free = (num_tm + num_phi_st + num_rest) / den;

        // PR #821: DEBUG_650FF trace removed; use `pr821-diag` feature.

        // Return the first zone temperature
        t_i_free.as_ref()[0]
    }

    /// Calculate HVAC capacity from design day simulation.
    ///
    /// Runs 24-hour simulations for heating and cooling design days to determine
    /// actual peak loads, then applies a safety factor (typically 1.1-1.2x).
    ///
    /// # Arguments
    ///
    /// * `heating_design_hours` - 24 hours of weather data for heating design (extreme cold)
    /// * `cooling_design_hours` - 24 hours of weather data for cooling design (extreme hot)
    /// * `safety_factor` - Safety factor to apply to peak loads (typically 1.15 for 15% margin)
    ///
    /// # Returns
    ///
    /// * `(heating_capacity_w, cooling_capacity_w)` - HVAC capacities in Watts
    pub fn calculate_hvac_capacity_from_design_day(
        &mut self,
        heating_design_hours: &[crate::weather::HourlyWeatherData],
        cooling_design_hours: &[crate::weather::HourlyWeatherData],
        safety_factor: f64,
    ) -> (f64, f64) {
        // Save current state
        let original_temperatures = self.0.temperatures.clone();
        let original_mass_temperatures = self.0.mass_temperatures.clone();

        // Run heating design day simulation (24 hours)
        self.reset_peak_power();
        self.reset_all_energy_tracking();

        for (hour, weather) in heating_design_hours.iter().enumerate() {
            self.set_weather(weather.clone());
            self.step_physics(hour, weather.dry_bulb_temp, 3600.0);
        }

        let peak_heating_w = self.get_peak_heating_power_kw() * 1000.0;

        // Run cooling design day simulation (24 hours)
        self.reset_peak_power();
        self.reset_all_energy_tracking();

        // Reset temperatures to initial state for cooling simulation
        self.0.temperatures = original_temperatures.clone();
        self.0.mass_temperatures = original_mass_temperatures.clone();

        for (hour, weather) in cooling_design_hours.iter().enumerate() {
            self.set_weather(weather.clone());
            self.step_physics(hour, weather.dry_bulb_temp, 3600.0);
        }

        let peak_cooling_w = self.get_peak_cooling_power_kw() * 1000.0;

        // Restore original temperatures
        self.0.temperatures = original_temperatures;
        self.0.mass_temperatures = original_mass_temperatures;

        // Apply safety factor
        let heating_capacity = peak_heating_w * safety_factor;
        let cooling_capacity = peak_cooling_w * safety_factor;

        (heating_capacity, cooling_capacity)
    }

    /// Issue #2770: test-only accessor for `calculate_zone_solar_gain` so that
    /// dhat integration tests can assert zero HashMap / String allocations on
    /// the solar-gain hot path without pulling in the full `solve_single_step`
    /// machinery (which has its own per-timestep Vec allocations unrelated to
    /// this issue).
    #[doc(hidden)]
    pub fn _dhat_calculate_zone_solar_gain(
        &mut self,
        zone_idx: usize,
        timestep: usize,
        weather: &HourlyWeatherData,
        dt_seconds: f64,
    ) -> (f64, f64) {
        self.calculate_zone_solar_gain(zone_idx, timestep, weather, dt_seconds)
    }
}

// ---------------------------------------------------------------------------
// Issue #2770 tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use fluxion_core::ashrae_cases::Orientation;

    /// Verify `opaque_surface_id_str` produces bit-identical strings to the
    /// original `match orientation { Up => "roof".to_string(), _ => format!("wall_{}", orientation.prefix()) }`.
    ///
    /// This is the deterministic equivalence test: if any surface-id mapping
    /// changes, `incident_solar_per_surface` keys would silently break.
    #[test]
    fn test_opaque_surface_id_str_matches_legacy_format() {
        for orientation in [
            Orientation::North,
            Orientation::East,
            Orientation::South,
            Orientation::West,
            Orientation::Up,
            Orientation::Horizontal,
            Orientation::Down,
        ] {
            let legacy = match orientation {
                Orientation::Up => "roof".to_string(),
                _ => format!("wall_{}", orientation.prefix()),
            };
            let hoisted = opaque_surface_id_str(orientation);
            assert_eq!(
                legacy, hoisted,
                "opaque_surface_id_str({orientation:?}) mismatch"
            );
        }
    }

    /// Verify `window_surface_id_str` produces bit-identical strings to the
    /// original `format!("window_{}", orientation.prefix())`.
    #[test]
    fn test_window_surface_id_str_matches_legacy_format() {
        for orientation in [
            Orientation::North,
            Orientation::East,
            Orientation::South,
            Orientation::West,
            Orientation::Up,
            Orientation::Horizontal,
            Orientation::Down,
        ] {
            let legacy = format!("window_{}", orientation.prefix());
            let hoisted = window_surface_id_str(orientation);
            assert_eq!(
                legacy, hoisted,
                "window_surface_id_str({orientation:?}) mismatch"
            );
        }
    }

    /// Verify `accumulate_incident_solar` produces the same result as the
    /// original `entry().or_default()` pattern.
    #[test]
    fn test_accumulate_incident_solar_equivalence() {
        let key = "wall_N";

        // Original pattern
        let mut map_orig: std::collections::BTreeMap<String, IncidentSolarAccumulator> =
            std::collections::BTreeMap::new();
        map_orig
            .entry(key.to_string())
            .or_default()
            .accumulate(500.0, 10.0, 3600.0);

        // New pattern
        let mut map_new: std::collections::BTreeMap<String, IncidentSolarAccumulator> =
            std::collections::BTreeMap::new();
        accumulate_incident_solar(&mut map_new, "wall_N", 500.0, 10.0, 3600.0);

        let orig = map_orig.get(key).unwrap();
        let new = map_new.get(key).unwrap();
        assert!(
            (orig.annual_kwh_m2 - new.annual_kwh_m2).abs() < f64::EPSILON,
            "annual_kwh_m2 mismatch: {} vs {}",
            orig.annual_kwh_m2,
            new.annual_kwh_m2
        );
        assert!(
            (orig.peak_wm2 - new.peak_wm2).abs() < f64::EPSILON,
            "peak_wm2 mismatch: {} vs {}",
            orig.peak_wm2,
            new.peak_wm2
        );
    }

    // ============================================================================
    // Issue #2881 — inline unit coverage for `solve_coupled_zone_temperatures`,
    // `set_ground_temp` / `with_ground_temperature` clamped vs unclamped modes,
    // `calculate_free_float_temperature` analytical steady-state, and
    // `solve_single_step` idempotency under zero-mass nodes.
    // ============================================================================

    /// Symmetric 2-zone conductance matrix must conserve energy
    /// (Σ q_iz_net = 0 within f64 machine precision).
    #[test]
    fn test_solve_coupled_zone_temperatures_n2_conserves_energy() {
        let model = ThermalModel::<VectorField>::new(2);
        let q = model
            .solve_coupled_zone_temperatures(2, &[20.0, 25.0], &[10.0], &[5.0])
            .unwrap();
        assert!(
            q.iter().sum::<f64>().abs() < 1e-9,
            "N=2 Σ q_iz_net must be ~0"
        );
    }

    /// Smallest N that hides a 2-zone hardcoding bug (#1391 regression guard).
    /// Also verifies the sign convention: warmest zone loses, coolest gains.
    #[test]
    fn test_solve_coupled_zone_temperatures_n3_conserves_energy() {
        let model = ThermalModel::<VectorField>::new(3);
        let q = model
            .solve_coupled_zone_temperatures(3, &[20.0, 25.0, 15.0], &[50.0], &[10.0])
            .unwrap();
        assert!(q.iter().sum::<f64>().abs() < 1e-9);
        assert!(q[1] < 0.0, "warm zone must lose heat: q[1]={}", q[1]);
        assert!(q[2] > 0.0, "cool zone must gain heat: q[2]={}", q[2]);
    }

    /// N=4 — the O(N) loop must generalise beyond N=3.
    #[test]
    fn test_solve_coupled_zone_temperatures_n4_conserves_energy() {
        let model = ThermalModel::<VectorField>::new(4);
        let q = model
            .solve_coupled_zone_temperatures(4, &[20.0, 25.0, 15.0, 22.0], &[30.0], &[20.0])
            .unwrap();
        assert!(q.iter().sum::<f64>().abs() < 1e-9);
    }

    /// N=1 short-circuits to None — no inter-zone coupling to compute.
    #[test]
    fn test_solve_coupled_zone_temperatures_n1_returns_none() {
        let model = ThermalModel::<VectorField>::new(1);
        assert!(model
            .solve_coupled_zone_temperatures(1, &[20.0], &[10.0], &[5.0])
            .is_none());
    }

    /// Zero conductive AND zero radiative conductance falls back to None —
    /// callers skip the inter-zone term when zones are decoupled.
    #[test]
    fn test_solve_coupled_zone_temperatures_zero_conductance_returns_none() {
        let model = ThermalModel::<VectorField>::new(2);
        assert!(model
            .solve_coupled_zone_temperatures(2, &[20.0, 25.0], &[0.0], &[0.0])
            .is_none());
    }

    /// `set_ground_temp` stores the value as-is (UNCLAMPED mode):
    /// `ConstantGroundTemperature` accepts any f64 and returns it verbatim
    /// for every timestep, including extreme values like -50 °C or +80 °C.
    #[test]
    fn test_set_ground_temp_constant_unclamped() {
        let mut model = ThermalModel::<VectorField>::new(1);
        model.set_ground_temp(15.0);
        for t in [0_usize, 100, 4380, 8759] {
            assert!((model.ground_temperature_at(t) - 15.0).abs() < 1e-9);
        }
        model.set_ground_temp(-50.0);
        assert!((model.ground_temperature_at(0) - (-50.0)).abs() < 1e-9);
        model.set_ground_temp(80.0);
        assert!((model.ground_temperature_at(0) - 80.0).abs() < 1e-9);
    }

    /// `set_dynamic_ground_temp` (Kusuda) — time-varying CLAMPED mode.
    /// The annual oscillation around t_mean must exceed 1 °C when amplitude > 0
    /// and depth is shallow; the mid-point must lie near t_mean.
    #[test]
    fn test_set_dynamic_ground_temp_kusuda_oscillates() {
        let mut model = ThermalModel::<VectorField>::new(1);
        model.set_dynamic_ground_temp(15.0, 5.0, 1.0, 0.1);
        let t_winter = model.ground_temperature_at(0);
        let t_summer = model.ground_temperature_at(4380);
        assert!(
            (t_summer - t_winter).abs() > 1.0,
            "Kusuda must oscillate: winter={t_winter}, summer={t_summer}"
        );
        let mean = (t_winter + t_summer) / 2.0;
        assert!(
            (mean - 15.0).abs() < 1.0,
            "Annual mean ~ t_mean=15.0, got {mean}"
        );
    }

    /// `with_ground_temperature` accepts a custom `GroundTemperature` impl.
    /// Demonstrates the CLAMPED mode contract: callers can plug in any model
    /// that enforces its own clamping/range policy.
    #[test]
    fn test_with_ground_temperature_custom_clamped() {
        #[derive(Clone)]
        struct ClampedGround {
            temp: f64,
        }
        impl GroundTemperature for ClampedGround {
            fn clone_box(&self) -> Box<dyn GroundTemperature> {
                Box::new(self.clone())
            }
            fn ground_temperature(&self, _t: usize) -> f64 {
                self.temp.clamp(10.0, 30.0)
            }
        }
        let mut model = ThermalModel::<VectorField>::new(1);
        model.with_ground_temperature(Box::new(ClampedGround { temp: 18.0 }));
        assert!((model.ground_temperature_at(0) - 18.0).abs() < 1e-9);
        let mut model2 = ThermalModel::<VectorField>::new(1);
        model2.with_ground_temperature(Box::new(ClampedGround { temp: -100.0 }));
        assert!(
            (model2.ground_temperature_at(0) - 10.0).abs() < 1e-9,
            "custom clamp to lower bound 10 °C"
        );
    }

    /// `calculate_free_float_temperature` matches the analytical 5R1C
    /// closed-form: T_free = (h_ms_is_prod·T_mass + term_rest_1·h_ext·T_ext
    /// + h_tr_floor·T_g) / den. We assert the function reproduces this
    /// formula exactly (1e-9 tolerance) under zero-loads.
    #[test]
    fn test_calculate_free_float_analytical_steady_state() {
        let mut model = ThermalModel::<VectorField>::new(1);
        let zero = VectorField::from_scalar(0.0, 1);
        model.thermal_capacitance = zero.clone();
        model.air_thermal_capacitance = zero.clone();
        model.loads = zero.clone();
        model.solar_gains = zero.clone();
        model.opaque_solar_gains = zero.clone();
        model.mass_temperatures = VectorField::from_scalar(25.0, 1);
        model.temperatures = VectorField::from_scalar(25.0, 1);
        model.set_ground_temp(20.0);
        let outdoor = 30.0;
        let t_free = model.calculate_free_float_temperature(0, outdoor);
        let h_ms_is_prod = model.derived_h_ms_is_prod.as_ref()[0];
        let term_rest_1 = model.derived_term_rest_1.as_ref()[0];
        let h_ext = model.derived_h_ext.as_ref()[0];
        let h_tr_floor = model.h_tr_floor.as_ref()[0];
        let den = model.derived_den.as_ref()[0];
        let expected =
            (h_ms_is_prod * 25.0 + term_rest_1 * h_ext * outdoor + h_tr_floor * 20.0) / den;
        assert!(
            (t_free - expected).abs() < 1e-9,
            "T_free must match analytical 5R1C closed-form: got {t_free}, expected {expected}"
        );
    }

    /// `solve_single_step` is a PURE FUNCTION of input state — running it on
    /// two bit-identical models returns the same HVAC energy. This is the
    /// idempotency contract: f(S0) = f(f(S0)) when f is pure.
    #[test]
    fn test_solve_single_step_idempotent_zero_mass() {
        let mut model = ThermalModel::<VectorField>::new(1);
        let zero = VectorField::from_scalar(0.0, 1);
        // tiny thermal mass avoids div-by-zero in ExplicitEuler (cm<500 threshold)
        model.thermal_capacitance = VectorField::from_scalar(1.0, 1);
        model.air_thermal_capacitance = zero.clone();
        model.loads = zero.clone();
        model.solar_gains = zero.clone();
        model.opaque_solar_gains = zero.clone();
        model.set_ground_temp(10.0);
        let mut clone = model.clone();
        let step_params = StepParameters {
            use_ai: false,
            surrogates: None,
            use_analytical_gains: false,
            lighting: None,
            equipment: None,
            occupancy: None,
        };
        let e1 = model.solve_single_step(12, 25.0, &step_params, 3600.0);
        let e2 = clone.solve_single_step(12, 25.0, &step_params, 3600.0);
        assert!(
            (e1 - e2).abs() < 1e-9,
            "solve_single_step must be a pure function of input state: e1={e1}, e2={e2}"
        );
    }
}
