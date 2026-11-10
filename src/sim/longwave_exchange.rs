//! Inter-surface longwave (LW) radiation exchange network for the zone interior.
//!
//! This module implements the explicit floor-ceiling-wall longwave radiation
//! exchange network for the 5R1C / 9R4C lumped-mass thermal model. Each
//! interior surface (floor, ceiling, walls) carries its own surface
//! temperature state that evolves via the envelope conduction pathway, and
//! the three surface temperatures exchange LW radiation through the
//! Stefan-Boltzmann law using the closed-form Hottel view factors for
//! parallel plates (floor/ceiling) and parallel-plate-vs-wall-ring
//! (floor/ceiling ↔ walls).
//!
//! # Issue #2890
//!
//! The 5R1C / 9R4C lumped-mass architecture in `src/sim/thermal_model.rs`
//! previously did not model the explicit floor-ceiling-wall longwave
//! radiation exchange network: `src/physics/ctf_zone_coupling.rs:269`
//! documents the network but it was not active in the production 5R1C path.
//! This caused the free-floating diurnal swing to be too large (Case 900FF
//! max off by ~4 °C, Case 950FF max off by ~4 °C, Cases 650FF / 900FF / 950FF
//! min off by 0.7–6 °C). This module provides the network that closes the
//! gap.
//!
//! # View factors
//!
//! For a rectangular zone of width `W`, depth `D`, height `H`:
//!
//! - `A_floor = A_ceiling = W · D`
//! - `A_wall = 2 · (W + D) · H`
//! - `F_floor↔ceiling = 1.0` (parallel, equal-area plates)
//! - `F_floor→walls = A_wall / (A_floor + A_wall)` (the wall ring sees a
//!   fraction of the floor emission equal to the wall-to-total-area ratio)
//! - `F_ceiling→walls = A_wall / (A_ceiling + A_wall)` (same, by symmetry)
//!
//! These are the standard closed-form Hottel view factors for a rectangular
//! enclosure; the same geometry underlies `src/sim/view_factors.rs:53`. The
//! floor ceiling pair is treated as parallel equal-area plates (F = 1.0);
//! the floor/ceiling and wall-ring coupling is derived from the
//! 2-D enclosure identity that the integrated view factor from a planar
//! surface to the surrounding wall ring equals the wall-area-to-total-area
//! ratio (the planar surface "sees" the wall ring around its perimeter).
//!
//! # Energy bookkeeping
//!
//! Each surface's interior heat balance is
//!
//! ```text
//! C_s · dT_s/dt = h_ri · (T_mrt − T_s) + h_ci · (T_zone − T_s)
//!               + net_longwave_exchange_with_other_surfaces
//! ```
//!
//! where the net LW exchange is the sum of pairwise Stefan-Boltzmann
//! exchanges, **`T_mrt` is the area-weighted mean radiant temperature of the
//! interior surfaces**, and `h_ci + h_ri = h_is`. The interior film
//! coefficients are unchanged from the existing 5R1C model (ISO 13790
//! §12.2.2: `h_ci = 3.0 W/m²·K`, `h_ri = 5.0 W/m²·K`).
//!
//! The air node receives the convective portion of the surface heat balance
//! (the existing `h_ci · (T_s − T_zone)` term), already computed in the
//! 5R1C / 9R4C paths. The LW radiation exchange redistributes heat between
//! interior surfaces; the air node sees the *net* effect through the surface
//! heat balance. The end-of-timestep `T_s` values are persisted in the
//! three new `surface_temp_*` fields on `ThermalModelData`.
//!
//! # Numerical treatment
//!
//! The full nonlinear Stefan-Boltzmann law `Q = ε · σ · A · (T_a⁴ − T_b⁴)`
//! is used (no linearization error). The Kelvin conversion is applied
//! internally. Per-pair reciprocity `F_ab · A_a = F_ba · A_b` is enforced
//! by the view factor construction.
//!
//! # References
//!
//! - Hottel's crossed-string method for radiation exchange in enclosures
//! - ASHRAE Handbook — Fundamentals (Chapter 4): interior surface heat balance
//! - ISO 13790 §12.2.2: interior film coefficients
//! - `src/sim/view_factors.rs::F_AB` — primitives for parallel rectangles
//! - `src/physics/ctf_zone_coupling.rs:269` — network documentation

use crate::physics::fp_algebraic::{algebraic_add, algebraic_mul};
use fluxion_core::physics_constants::STEFAN_BOLTZMANN;

/// Interior surface identifier.
///
/// Tagged union of the three interior surface types that participate in the
/// longwave radiation exchange network. The 5R1C / 9R4C lumped-mass model
/// tracks each surface type separately at the zone granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InteriorSurfaceKind {
    /// Floor surface (ground-coupled interior).
    Floor,
    /// Ceiling surface (roof-coupled interior).
    Ceiling,
    /// Wall surface (vertical envelope interior).
    Wall,
}

/// View factor parameters for a single rectangular zone.
///
/// Holds the per-surface-type areas and the per-pair view factors derived
/// from the rectangular-enclosure geometry. View factors for the three
/// surface pairs are computed at construction time and cached.
#[derive(Debug, Clone, Copy)]
pub struct InteriorSurfaceNetwork {
    /// Floor area (m²). Equals ceiling area for a rectangular zone.
    pub floor_area: f64,
    /// Ceiling area (m²). Equals floor area for a rectangular zone.
    pub ceiling_area: f64,
    /// Gross wall area (m²). Includes windows but the LW network treats
    /// all walls as one opaque ring.
    pub wall_area: f64,
    /// Interior surface emissivity for the LW radiation network (0–1).
    /// Default 0.9 (most building materials).
    pub emissivity: f64,
    /// View factor from floor to ceiling (parallel equal plates, F = 1.0).
    f_floor_ceiling: f64,
    /// View factor from floor to walls (wall ring).
    f_floor_wall: f64,
    /// View factor from ceiling to walls (wall ring, equals floor by symmetry).
    f_ceiling_wall: f64,
}

impl InteriorSurfaceNetwork {
    /// Construct the network for a rectangular zone.
    ///
    /// # Arguments
    /// * `width` — Zone width (m)
    /// * `depth` — Zone depth (m)
    /// * `height` — Zone height (m)
    /// * `emissivity` — Interior surface emissivity (0–1). Default 0.9.
    pub fn from_rect_zone(width: f64, depth: f64, height: f64, emissivity: f64) -> Self {
        let floor_area = width * depth;
        let ceiling_area = floor_area;
        let wall_area = 2.0 * (width + depth) * height;
        Self::from_areas(floor_area, ceiling_area, wall_area, emissivity)
    }

    /// Construct from precomputed area fields.
    ///
    /// # Arguments
    /// * `floor_area` — Floor area (m²)
    /// * `ceiling_area` — Ceiling area (m²)
    /// * `wall_area` — Gross wall area (m²)
    /// * `emissivity` — Interior surface emissivity (0–1)
    pub fn from_areas(floor_area: f64, ceiling_area: f64, wall_area: f64, emissivity: f64) -> Self {
        let f_floor_wall = if floor_area + wall_area > 0.0 {
            wall_area / (floor_area + wall_area)
        } else {
            0.0
        };
        let f_ceiling_wall = if ceiling_area + wall_area > 0.0 {
            wall_area / (ceiling_area + wall_area)
        } else {
            0.0
        };
        Self {
            floor_area,
            ceiling_area,
            wall_area,
            emissivity: emissivity.clamp(0.0, 1.0),
            f_floor_ceiling: 1.0,
            f_floor_wall,
            f_ceiling_wall,
        }
    }

    /// Construct a degenerate network (zero area, no exchange).
    /// Used as a safe default when zone geometry is absent.
    pub fn degenerate() -> Self {
        Self {
            floor_area: 0.0,
            ceiling_area: 0.0,
            wall_area: 0.0,
            emissivity: 0.0,
            f_floor_ceiling: 0.0,
            f_floor_wall: 0.0,
            f_ceiling_wall: 0.0,
        }
    }

    /// View factor from floor to ceiling (parallel equal plates).
    pub fn view_factor_floor_ceiling(&self) -> f64 {
        self.f_floor_ceiling
    }

    /// View factor from floor to walls.
    pub fn view_factor_floor_wall(&self) -> f64 {
        self.f_floor_wall
    }

    /// View factor from ceiling to walls.
    pub fn view_factor_ceiling_wall(&self) -> f64 {
        self.f_ceiling_wall
    }

    /// Total interior surface area (m²).
    ///
    /// 3-term floor/ceiling/wall reduction routed through `algebraic_add`
    /// (Issue #3324) so the compiler can fold the sum with surrounding
    /// area-weighted products under `--features fast-math`. Default
    /// features stay bit-identical.
    pub fn total_area(&self) -> f64 {
        algebraic_add(
            algebraic_add(self.floor_area, self.ceiling_area),
            self.wall_area,
        )
    }

    /// Net LW heat gain for the floor surface [W] (positive = heat into floor).
    ///
    /// Computed from the full nonlinear Stefan-Boltzmann law:
    ///
    /// ```text
    /// Q_floor = ε · σ · F_fc · A_floor · (T_c⁴ − T_f⁴)
    ///        + ε · σ · F_fw · A_floor · (T_w⁴ − T_f⁴)
    /// ```
    ///
    /// The two terms are the LW exchange with the ceiling and the wall ring.
    /// Kelvin conversion is applied internally.
    ///
    /// **Reciprocity:** The wall-ring view factor used here is the
    /// *floor-to-wall* one (`F_floor→wall = A_w / (A_f + A_w)`). The
    /// reverse-direction *wall-to-floor* view factor is derived by
    /// reciprocity `F_wall→floor = F_floor→wall · A_floor / A_wall` and
    /// applied on the wall side in [`Self::net_lw_wall`]. This preserves
    /// the per-pair identity `F_AB · A_A = F_BA · A_B` that
    /// `view_factors.rs::hottels_rectangular_view_factor_pair` enforces.
    pub fn net_lw_floor(&self, t_floor_c: f64, t_ceiling_c: f64, t_wall_c: f64) -> f64 {
        if self.floor_area <= 0.0 || self.emissivity <= 0.0 {
            return 0.0;
        }
        let t_f_k = t_floor_c + 273.15;
        let t_c_k = t_ceiling_c + 273.15;
        let t_w_k = t_wall_c + 273.15;
        let q_fc = STEFAN_BOLTZMANN
            * self.emissivity
            * self.f_floor_ceiling
            * self.floor_area
            * (t_c_k.powi(4) - t_f_k.powi(4));
        // F_floor→wall · A_floor (directional, not symmetric)
        let q_fw = STEFAN_BOLTZMANN
            * self.emissivity
            * self.f_floor_wall
            * self.floor_area
            * (t_w_k.powi(4) - t_f_k.powi(4));
        // 2-term `(q_floor↔ceiling) + (q_floor↔wall)` reduction routed
        // through `algebraic_add` (Issue #3324). Default-feature builds
        // stay bit-identical; under `--features fast-math` the surrounding
        // Stefan-Boltzmann products can be FMA-contracted.
        algebraic_add(q_fc, q_fw)
    }

    /// Net LW heat gain for the ceiling surface [W] (positive = heat into ceiling).
    ///
    /// Symmetric to [`Self::net_lw_floor`]: receives from floor and walls.
    pub fn net_lw_ceiling(&self, t_floor_c: f64, t_ceiling_c: f64, t_wall_c: f64) -> f64 {
        if self.ceiling_area <= 0.0 || self.emissivity <= 0.0 {
            return 0.0;
        }
        let t_f_k = t_floor_c + 273.15;
        let t_c_k = t_ceiling_c + 273.15;
        let t_w_k = t_wall_c + 273.15;
        let q_cf = STEFAN_BOLTZMANN
            * self.emissivity
            * self.f_floor_ceiling
            * self.ceiling_area
            * (t_f_k.powi(4) - t_c_k.powi(4));
        // F_ceiling→wall · A_ceiling (directional, not symmetric)
        let q_cw = STEFAN_BOLTZMANN
            * self.emissivity
            * self.f_ceiling_wall
            * self.ceiling_area
            * (t_w_k.powi(4) - t_c_k.powi(4));
        // 2-term ceiling-pair reduction (Issue #3324). Default-feature
        // builds stay bit-identical.
        algebraic_add(q_cf, q_cw)
    }

    /// Net LW heat gain for the wall surface [W] (positive = heat into wall).
    ///
    /// Wall receives from floor and ceiling. The reverse-direction view
    /// factors `F_wall→floor = F_floor→wall · A_floor / A_wall` and
    /// `F_wall→ceiling = F_ceiling→wall · A_ceiling / A_wall` are derived
    /// from the per-pair reciprocity identity enforced by
    /// `view_factors.rs::hottels_rectangular_view_factor_pair`.
    pub fn net_lw_wall(&self, t_floor_c: f64, t_ceiling_c: f64, t_wall_c: f64) -> f64 {
        if self.wall_area <= 0.0 || self.emissivity <= 0.0 {
            return 0.0;
        }
        let t_f_k = t_floor_c + 273.15;
        let t_c_k = t_ceiling_c + 273.15;
        let t_w_k = t_wall_c + 273.15;
        // F_wall→floor = F_floor→wall · A_floor / A_wall
        let f_wall_floor = if self.wall_area > 0.0 {
            self.f_floor_wall * self.floor_area / self.wall_area
        } else {
            0.0
        };
        // F_wall→ceiling = F_ceiling→wall · A_ceiling / A_wall
        let f_wall_ceiling = if self.wall_area > 0.0 {
            self.f_ceiling_wall * self.ceiling_area / self.wall_area
        } else {
            0.0
        };
        let q_wf = STEFAN_BOLTZMANN
            * self.emissivity
            * f_wall_floor
            * self.wall_area
            * (t_f_k.powi(4) - t_w_k.powi(4));
        let q_wc = STEFAN_BOLTZMANN
            * self.emissivity
            * f_wall_ceiling
            * self.wall_area
            * (t_c_k.powi(4) - t_w_k.powi(4));
        // 2-term wall-pair reduction (Issue #3324). Default-feature
        // builds stay bit-identical.
        algebraic_add(q_wf, q_wc)
    }

    /// Net LW heat exchanged among the three surfaces — sanity check.
    ///
    /// By energy conservation, the sum of `net_lw_floor + net_lw_ceiling +
    /// net_lw_wall` is zero when the view factors satisfy reciprocity
    /// (which the rectangular construction enforces). Use this in tests.
    ///
    /// 3-term cross-surface reduction routed through `algebraic_add`
    /// (Issue #3324). Default-feature builds stay bit-identical.
    pub fn total_net_lw(&self, t_floor_c: f64, t_ceiling_c: f64, t_wall_c: f64) -> f64 {
        algebraic_add(
            algebraic_add(
                self.net_lw_floor(t_floor_c, t_ceiling_c, t_wall_c),
                self.net_lw_ceiling(t_floor_c, t_ceiling_c, t_wall_c),
            ),
            self.net_lw_wall(t_floor_c, t_ceiling_c, t_wall_c),
        )
    }
}

/// Per-zone interior surface state (Issue #2890).
///
/// Holds the three surface temperatures — floor, ceiling, wall — for a
/// single zone. The temperatures are evolved via the envelope conduction
/// pathway in [`step_interior_surface`] and the LW exchange is computed in
/// [`Self::net_lw_split`].
#[derive(Debug, Clone, Copy)]
pub struct InteriorSurfaceState {
    /// Floor interior surface temperature (°C).
    pub t_floor: f64,
    /// Ceiling interior surface temperature (°C).
    pub t_ceiling: f64,
    /// Wall interior surface temperature (°C).
    pub t_wall: f64,
}

impl InteriorSurfaceState {
    /// Construct with uniform initial temperature (default 20°C).
    pub fn uniform(t: f64) -> Self {
        Self {
            t_floor: t,
            t_ceiling: t,
            t_wall: t,
        }
    }

    /// Split the net LW exchange per surface [W] for diagnostic / logging.
    ///
    /// Returns `(floor, ceiling, wall)` net LW heat gains (sum = 0 by
    /// reciprocity).
    pub fn net_lw_split(&self, network: &InteriorSurfaceNetwork) -> (f64, f64, f64) {
        let floor = network.net_lw_floor(self.t_floor, self.t_ceiling, self.t_wall);
        let ceiling = network.net_lw_ceiling(self.t_floor, self.t_ceiling, self.t_wall);
        let wall = network.net_lw_wall(self.t_floor, self.t_ceiling, self.t_wall);
        (floor, ceiling, wall)
    }
}

/// Bundle of per-zone interior surface temperatures and the per-timestep
/// environment drivers (Issue #2890).
///
/// The 5R1C / 9R4C physics step fills in this bundle, calls
/// [`step_interior_surface_with_lw`], and reads the updated
/// `surface_temp_*` fields to drive subsequent air-node / mass-node
/// numerators.
#[derive(Debug, Clone, Copy)]
pub struct InteriorSurfaceStep {
    /// Float state: floor / ceiling / wall interior surface temperatures.
    pub state: InteriorSurfaceState,
    /// Floor interior env temperature (°C) — typically ground temperature.
    pub t_env_floor: f64,
    /// Ceiling interior env temperature (°C) — typically roof sol-air.
    pub t_env_ceiling: f64,
    /// Wall interior env temperature (°C) — typically wall sol-air.
    pub t_env_wall: f64,
}

/// Step the three interior surface temperatures forward by one timestep
/// (Issue #2890 — full partitioned surface).
///
/// The surface ODE is:
///
/// ```text
/// τ_s = C_s / (h_is + h_ms_s)
/// T_s_eq = (T_zone · h_is + T_env_s · h_ms_s) / (h_is + h_ms_s)
/// T_s_new = T_s_eq + (T_s_old − T_s_eq) · exp(−dt / τ_s)
/// ```
///
/// Where `h_is = h_ci + h_ri` (combined interior film coefficient,
/// ISO 13790 §12.2.2 default `8.0 W/m²·K`), `h_ms_s` is the surface-to-mass
/// conductance for the surface type, and `C_s` is the surface's effective
/// thermal capacitance (J/K).
///
/// After the ODE step, the longwave radiation exchange redistributes heat
/// between the three surfaces via the network's Stefan-Boltzmann coupling.
/// The net air-node damping term is the additional convective heat flow
/// induced by the surface-temperature asymmetry:
///
/// ```text
/// ΔQ_air = h_ci · (A_f · δT_f + A_c · δT_c + A_w · δT_w)
/// ```
///
/// where `δT_s = T_s_new − T_s_eq` is the surface temperature deviation from
/// its pre-network equilibrium (positive = above equilibrium). The damping
/// term is positive when the LW network has warmed the surface above
/// equilibrium (convective heat to air) and negative when it has cooled it
/// below equilibrium (convective heat from air).
///
/// # Arguments
///
/// * `dt` — Timestep duration (seconds)
/// * `h_ci` — Interior convective film coefficient (W/m²·K)
/// * `h_ri` — Interior radiative film coefficient (W/m²·K)
/// * `h_ms_floor`, `h_ms_ceiling`, `h_ms_wall` — Surface-to-mass conductances
///   for each surface type (W/m²·K). For the 5R1C lumped-mass model, all
///   three equal `h_tr_ms / A_total_eff` (the lumped mass-node conductance).
/// * `t_zone` — Zone air temperature (°C)
/// * `t_env_floor`, `t_env_ceiling`, `t_env_wall` — See [`InteriorSurfaceStep`]
/// * `c_floor`, `c_ceiling`, `c_wall` — Surface thermal capacitances (J/K)
/// * `network` — Interior surface geometry / view factors
/// * `step` — Mutable input state (in: pre-step temperatures, env drivers;
///   out: updated surface temperatures and the net air-node damping term)
///
/// # Returns
///
/// Net air-node damping term `ΔQ_air` [W] (positive = heat into air).
/// The caller adds this to the air-node numerator / heat balance.
#[allow(clippy::too_many_arguments)]
pub fn step_interior_surface_with_lw(
    dt: f64,
    h_ci: f64,
    h_ri: f64,
    h_ms_floor: f64,
    h_ms_ceiling: f64,
    h_ms_wall: f64,
    t_zone: f64,
    t_env_floor: f64,
    t_env_ceiling: f64,
    t_env_wall: f64,
    c_floor: f64,
    c_ceiling: f64,
    c_wall: f64,
    network: &InteriorSurfaceNetwork,
    step: &mut InteriorSurfaceStep,
) -> f64 {
    let h_is = h_ci + h_ri;

    // Save pre-step temperatures for the LW-induced delta computation.
    let t_f_old = step.state.t_floor;
    let t_c_old = step.state.t_ceiling;
    let t_w_old = step.state.t_wall;

    // Step the three surface temperatures via the surface ODE.
    step_interior_surface(
        dt,
        h_is,
        h_ms_floor,
        h_ms_ceiling,
        h_ms_wall,
        t_zone,
        t_env_floor,
        t_env_ceiling,
        t_env_wall,
        c_floor,
        c_ceiling,
        c_wall,
        &mut step.state,
    );

    // Longwave radiation exchange: redistribute heat between the three
    // surfaces via Stefan-Boltzmann. The LW network absorbs/emits heat
    // *between surfaces* only — the air node sees the *side-effect* of the
    // surface temperature asymmetry through the convective heat balance.
    //
    // The LW heat each surface *receives* (positive = heat in):
    let q_floor = network.net_lw_floor(step.state.t_floor, step.state.t_ceiling, step.state.t_wall);
    let q_ceiling =
        network.net_lw_ceiling(step.state.t_floor, step.state.t_ceiling, step.state.t_wall);
    let q_wall = network.net_lw_wall(step.state.t_floor, step.state.t_ceiling, step.state.t_wall);

    // Apply the LW heat as an instantaneous energy gain to each surface.
    // The resulting temperature change is
    //   ΔT_s = Q_lw · dt / C_s
    // which is the first-order approximation for small dt·gains. For an
    // explicit Euler step (acceptable here since the LW gain is small
    // relative to the surface heat flux), the new surface temperature is
    //   T_s_new = T_s_old + Q_lw · dt / C_s
    //
    // If C_s is zero (degenerate), the gain is absorbed by the next ODE
    // step (which clamps to the equilibrium value).
    apply_lw_heat_to_surface(dt, q_floor, c_floor, &mut step.state.t_floor, t_zone);
    apply_lw_heat_to_surface(dt, q_ceiling, c_ceiling, &mut step.state.t_ceiling, t_zone);
    apply_lw_heat_to_surface(dt, q_wall, c_wall, &mut step.state.t_wall, t_zone);

    // Compute the net air-node damping term from the surface temperature
    // asymmetry. The LW network has redistributed heat between surfaces,
    // creating a temperature difference vs. the pre-step state. The
    // convective heat from each surface to the air is:
    //   Q_conv_s = h_ci · (T_s_new − T_zone) · A_s
    // The total convective heat is summed over the three surfaces. The
    // *net air-node damping* is the difference between this and the
    // baseline (uniform pre-step surface temperature), which we
    // approximate as the convective heat from the *delta* surface
    // temperature (the change induced by the LW network):
    //   ΔQ_air = h_ci · (A_f · δT_f + A_c · δT_c + A_w · δT_w)
    // where δT_s = T_s_new − T_s_old.
    let delta_t_f = step.state.t_floor - t_f_old;
    let delta_t_c = step.state.t_ceiling - t_c_old;
    let delta_t_w = step.state.t_wall - t_w_old;
    // 3-term area-weighted ΔT reduction (Issue #3324): the per-timestep
    // air-node damping loop body. Algebraic reassociation under
    // `--features fast-math` lets the surrounding `h_ci * (...)` fold
    // into a single FMA chain instead of three sequential FMAs gated by
    // strict IEEE add ordering.
    algebraic_mul(
        h_ci,
        algebraic_add(
            algebraic_add(
                algebraic_mul(network.floor_area, delta_t_f),
                algebraic_mul(network.ceiling_area, delta_t_c),
            ),
            algebraic_mul(network.wall_area, delta_t_w),
        ),
    )
}

/// Apply the LW radiation exchange heat to a surface, capped to the
/// equilibrium that the air zone would impose (degenerate guard).
#[inline]
fn apply_lw_heat_to_surface(dt: f64, q_lw: f64, c_s: f64, t_s: &mut f64, t_zone: f64) {
    if c_s > 0.0 && dt > 0.0 && q_lw.is_finite() {
        let delta_t = q_lw * dt / c_s;
        // Cap the delta to ±50 K to avoid runaway from upstream
        // arithmetic errors. The convective heat balance will absorb the
        // remainder on the next step.
        *t_s += delta_t.clamp(-50.0, 50.0);
    }
    // Clamp to a finite range.
    if !t_s.is_finite() {
        *t_s = t_zone;
    }
}

/// Interior surface temperature step (Issue #2890 step 1 / network).
///
/// Evolves the three surface temperatures via the surface-node ODE
/// (exponential solution of the first-order linear ODE):
///
/// ```text
/// τ_s = C_s / (h_is + h_ms_s)
/// T_s_eq = (T_zone · h_is + T_env_s · h_ms_s) / (h_is + h_ms_s)
/// T_s_new = T_s_eq + (T_s_old − T_s_eq) · exp(−dt / τ_s)
/// ```
///
/// where:
/// - `h_is = h_ci + h_ri` is the combined interior film coefficient
/// - `h_ms_s` is the surface-to-mass conductance (≈ 9.1 W/m²K for the
///   ISO 13790 lumped-mass assumption, scaled by the floor/ceiling/wall
///   area weighting)
/// - `T_env_s` is the surface-type-specific exterior boundary temperature
///   (sol-air for wall/ceiling, ground for floor)
/// - `C_s` is the surface's lumped thermal capacitance (J/K) — scaled
///   from the zone's total capacitance by the area-weighted fraction
///
/// On degenerate inputs (`h_is + h_ms_s <= 0.0` or `dt <= 0.0`) the legacy
/// lumped path is preserved exactly: `T_s` is left at its previous value
/// and the equilibrium drive is returned for diagnostic logging only.
#[allow(clippy::too_many_arguments)]
pub fn step_interior_surface(
    dt: f64,
    h_is: f64,
    h_ms_floor: f64,
    h_ms_ceiling: f64,
    h_ms_wall: f64,
    t_zone: f64,
    t_env_floor: f64,
    t_env_ceiling: f64,
    t_env_wall: f64,
    c_floor: f64,
    c_ceiling: f64,
    c_wall: f64,
    state: &mut InteriorSurfaceState,
) {
    // Floor
    step_one_surface(
        dt,
        h_is,
        h_ms_floor,
        t_zone,
        t_env_floor,
        c_floor,
        &mut state.t_floor,
    );
    // Ceiling
    step_one_surface(
        dt,
        h_is,
        h_ms_ceiling,
        t_zone,
        t_env_ceiling,
        c_ceiling,
        &mut state.t_ceiling,
    );
    // Wall
    step_one_surface(
        dt,
        h_is,
        h_ms_wall,
        t_zone,
        t_env_wall,
        c_wall,
        &mut state.t_wall,
    );
}

#[inline]
fn step_one_surface(dt: f64, h_is: f64, h_ms: f64, t_zone: f64, t_env: f64, c: f64, t_s: &mut f64) {
    // Degenerate guards: if h_is is zero, the surface has no air coupling
    // and the function must leave the state unchanged (matches the legacy
    // 5R1C behavior — see test_step_degenerate_holds_state). If h_total
    // is zero or non-finite, the surface ODE is undefined and we leave
    // the state unchanged.
    if h_is <= 0.0 || !h_is.is_finite() {
        return;
    }
    let h_total = h_is + h_ms;
    if h_total <= 0.0 || !h_total.is_finite() {
        return;
    }
    let t_eq = (t_zone * h_is + t_env * h_ms) / h_total;
    if !t_eq.is_finite() {
        return;
    }
    let tau = if c > 0.0 && h_total > 0.0 {
        c / h_total
    } else {
        f64::INFINITY
    };
    *t_s = if tau.is_finite() && dt > 0.0 {
        t_eq + (*t_s - t_eq) * (-dt / tau).exp()
    } else {
        t_eq
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Typical ASHRAE 140 Case 600/900 geometry: 8.0 m × 6.0 m × 2.7 m.
    const W: f64 = 8.0;
    const D: f64 = 6.0;
    const H: f64 = 2.7;

    /// Network with default emissivity 0.9.
    fn case_600_network() -> InteriorSurfaceNetwork {
        InteriorSurfaceNetwork::from_rect_zone(W, D, H, 0.9)
    }

    #[test]
    fn test_rect_zone_areas() {
        let n = case_600_network();
        assert!((n.floor_area - 48.0).abs() < 1e-9);
        assert!((n.ceiling_area - 48.0).abs() < 1e-9);
        assert!((n.wall_area - 75.6).abs() < 1e-9);
    }

    #[test]
    fn test_view_factor_floor_ceiling_is_unity() {
        let n = case_600_network();
        assert!(
            (n.view_factor_floor_ceiling() - 1.0).abs() < 1e-9,
            "parallel equal-area plates: F_floor→ceiling = 1.0"
        );
    }

    #[test]
    fn test_view_factor_floor_wall_ratio() {
        let n = case_600_network();
        let expected = 75.6 / (48.0 + 75.6);
        assert!(
            (n.view_factor_floor_wall() - expected).abs() < 1e-9,
            "F_floor→walls = A_wall / (A_floor + A_wall)"
        );
    }

    #[test]
    fn test_view_factor_ceiling_wall_equals_floor_wall() {
        let n = case_600_network();
        assert!(
            (n.view_factor_ceiling_wall() - n.view_factor_floor_wall()).abs() < 1e-9,
            "ceiling = floor by rectangular symmetry"
        );
    }

    #[test]
    fn test_reciprocity_zero_conservation() {
        // At uniform temperature, net LW exchange is zero.
        let n = case_600_network();
        let total = n.total_net_lw(20.0, 20.0, 20.0);
        assert!(
            total.abs() < 1e-6,
            "uniform temperature: total net LW = 0, got {total:.6e}"
        );
    }

    #[test]
    fn test_reciprocity_pairwise_conservation() {
        // Per-pair reciprocity: F_fc · A_f = F_cf · A_c (true here because
        // F_fc = 1.0 and A_f = A_c). The net exchange between floor and
        // ceiling = -net_exchange_ceiling→floor by the energy conservation
        // identity.
        let n = case_600_network();
        let t_f = 15.0;
        let t_c = 35.0;
        let t_w = 20.0;
        let q_floor = n.net_lw_floor(t_f, t_c, t_w);
        let q_ceiling = n.net_lw_ceiling(t_f, t_c, t_w);
        let q_wall = n.net_lw_wall(t_f, t_c, t_w);
        let sum = q_floor + q_ceiling + q_wall;
        // Per-pair reciprocity: F_fc * A_f = F_cf * A_c means the
        // floor-ceiling exchange is sign-symmetric and the total sum
        // accounts for the wall ring's contribution.
        assert!(
            sum.abs() < 1.0,
            "total net LW ≈ 0 (W), got {sum:.6e} (small imbalance absorbed by wall term)"
        );
    }

    #[test]
    fn test_hot_ceiling_warms_floor_and_walls() {
        // Hot ceiling (50°C) over cool floor (20°C) and walls (20°C):
        // the ceiling must radiate heat DOWN to the floor and walls.
        let n = case_600_network();
        let q_floor = n.net_lw_floor(20.0, 50.0, 20.0);
        let q_wall = n.net_lw_wall(20.0, 50.0, 20.0);
        assert!(
            q_floor > 0.0,
            "hot ceiling must warm cool floor: q_floor = {q_floor:.2} W"
        );
        assert!(
            q_wall > 0.0,
            "hot ceiling must warm cool walls: q_wall = {q_wall:.2} W"
        );
    }

    #[test]
    fn test_lw_magnitude_physically_reasonable() {
        // Verify the LW exchange magnitude is in the expected W range for
        // a 30 K temperature difference. The full nonlinear Stefan-Boltzmann
        // law for a 30 K ΔT (T_c = 50 °C, T_f = 20 °C) gives:
        //   Q_floor = ε · σ · F_fc · A_floor · (T_c⁴ − T_f⁴)
        //          = 0.9 · 5.67e-8 · 1.0 · 48 · (323.15⁴ − 293.15⁴)
        //          ≈ 8622 W
        // (the floor↔ceiling pair dominates because the ceiling is at
        // F = 1.0 and the floor area is 48 m²; the wall term is zero
        // when T_w = T_f). The Python check confirms this magnitude.
        let n = case_600_network();
        let q_floor = n.net_lw_floor(20.0, 50.0, 20.0);
        assert!(
            q_floor > 8000.0 && q_floor < 9000.0,
            "Q_floor @ ΔT=30K should be in [8000, 9000] W, got {q_floor:.2}"
        );
    }

    #[test]
    fn test_degenerate_network_zero_exchange() {
        let n = InteriorSurfaceNetwork::degenerate();
        let q = n.net_lw_floor(20.0, 50.0, 20.0);
        assert_eq!(q, 0.0, "degenerate network has no exchange");
    }

    #[test]
    fn test_zero_emissivity_zero_exchange() {
        let mut n = case_600_network();
        n.emissivity = 0.0;
        let q = n.net_lw_floor(20.0, 50.0, 20.0);
        assert_eq!(q, 0.0, "zero emissivity has no exchange");
    }

    #[test]
    fn test_step_interior_surface_equilibrium_recovery() {
        // Run step_interior_surface to equilibrium and verify that the
        // surface temperature converges to (T_zone · h_is + T_env · h_ms) / (h_is + h_ms).
        let mut state = InteriorSurfaceState::uniform(10.0);
        let h_is = 8.0;
        let h_ms = 1000.0;
        let t_zone = 20.0;
        let t_env = 50.0;
        let c = 1.0e6;
        let dt = 3600.0;
        let t_eq = (t_zone * h_is + t_env * h_ms) / (h_is + h_ms);
        for _ in 0..2000 {
            step_interior_surface(
                dt, h_is, h_ms, h_ms, h_ms, t_zone, t_env, t_env, t_env, c, c, c, &mut state,
            );
        }
        let tol = 0.01;
        assert!(
            (state.t_floor - t_eq).abs() < tol,
            "t_floor should converge to {t_eq} (got {t_f})",
            t_f = state.t_floor
        );
        assert!(
            (state.t_ceiling - t_eq).abs() < tol,
            "t_ceiling should converge to {t_eq} (got {t_c})",
            t_c = state.t_ceiling
        );
        assert!(
            (state.t_wall - t_eq).abs() < tol,
            "t_wall should converge to {t_eq} (got {t_w})",
            t_w = state.t_wall
        );
    }

    #[test]
    fn test_step_degenerate_holds_state() {
        let mut state = InteriorSurfaceState::uniform(15.0);
        step_interior_surface(
            3600.0, 0.0, // h_is = 0 (degenerate)
            1000.0, 1000.0, 1000.0, 20.0, 30.0, 30.0, 30.0, 1.0e6, 1.0e6, 1.0e6, &mut state,
        );
        // Degenerate h_is=0 should leave state unchanged.
        assert!((state.t_floor - 15.0).abs() < 1e-9);
        assert!((state.t_ceiling - 15.0).abs() < 1e-9);
        assert!((state.t_wall - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_uniform_state_with_uniform_environment_stays_uniform() {
        // A uniform state with uniform environment should remain uniform
        // (no LW exchange, no gradient).
        let mut state = InteriorSurfaceState::uniform(20.0);
        for _ in 0..100 {
            step_interior_surface(
                3600.0, 8.0, 1000.0, 1000.0, 1000.0, 20.0, 20.0, 20.0, 20.0, 1.0e6, 1.0e6, 1.0e6,
                &mut state,
            );
        }
        assert!((state.t_floor - 20.0).abs() < 1e-9);
        assert!((state.t_ceiling - 20.0).abs() < 1e-9);
        assert!((state.t_wall - 20.0).abs() < 1e-9);
    }
}
