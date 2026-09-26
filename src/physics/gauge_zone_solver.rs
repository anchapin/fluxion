//! Zone-level heat balance aggregator using per-surface GaugeSolvers.
//!
//! This module implements Phase 2 of the GaugeSolver elevation: replacing the
//! lumped 5R1C zone network with a geometrically-accurate assembly of 1D
//! per-surface GaugeSolvers that conserve energy at the zone air node.
//!
//! ## N-Zone Support
//!
//! This module supports both single-zone and multi-zone configurations:
//!
//! - **Single-zone**: Classic ASHRAE 140 case with multiple surfaces
//! - **Multi-zone**: N zones with inter-zone coupling via shared walls/floors
//!
//! Inter-zone coupling is defined over an adjacency graph where each edge
//! represents a shared boundary with thermal conductance. No pairwise
//! special-casing is used — all coupling is computed generically from
//! the adjacency structure.
//!
//! ## Mathematical Model
//!
//! The zone is modeled as a well-mixed air node with thermal capacitance
//! surrounded by N discrete surfaces. For each surface i:
//!
//! q_i = (T_ext,i - T_air) / R_i
//!
//! where T_ext,i = T_exterior + G_solar,i / h_exterior (sol-air temperature).
//!
//! For multi-zone configurations, inter-zone heat transfer is:
//!
//! Q_zone_j = sum_over_adjacent_zones(g_ij * (T_zone_i - T_zone_j))
//!
//! where g_ij is the inter-zone conductance.
//!
//! The zone air energy balance becomes:
//!
//! rho_air * V_zone * c_air * dT_air/dt = sum_i(q_i * A_i) + Q_int + Q_inf + Q_zone_coupling
//!
//! Using implicit Euler (unconditionally stable):
//!
//! T_air_new = (C_air * T_air_old + (Q_net + Q_zone_coupling) * dt) / (C_air + h_total * dt)
//!
//! ## Surface Types
//!
//! - **Opaque**: Walls, roof, floor - full resistance path
//! - **Window**: Simplified glazing model (future: multi-layer)
//! - **Ground**: Fixed ground temperature boundary
//! - **InternalMass**: Furniture, partitions
//! - **InterZone**: Shared boundary with adjacent zone

use crate::physics::gauge_solver::{GaugeBoundaryConditions, GaugeSolver};
use crate::physics::multi_node_solver::air_sky_conductance;
use crate::physics::solver_trait::{HeatConductionSolver, SolverError};
use crate::physics::units::FromF64;
use crate::physics::units::{HeatFlux, HeatTransferCoefficient, Temperature, Time, ToF64};
use crate::physics::wall_spec::WallSpec;
use std::collections::HashMap;

/// Surface classification for zone modeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceType {
    /// Opaque exterior wall
    Wall,
    /// Window/glazing
    Window,
    /// Roof/ceiling
    Roof,
    /// Floor slab
    Floor,
    /// Fixed-temperature ground (ASHRAE 140 B3.3)
    Ground,
    /// Internal mass (furniture, partitions)
    InternalMass,
    /// Inter-zone boundary (shared wall/floor between zones)
    InterZone,
}

impl SurfaceType {
    /// Returns the solar gain fraction for this surface type.
    /// Windows receive solar; opaque surfaces conduct it away.
    pub fn solar_fraction(&self) -> f64 {
        match self {
            SurfaceType::Window => 1.0,
            SurfaceType::Wall => 0.0,
            SurfaceType::Roof => 0.0,
            SurfaceType::Floor => 0.0,
            SurfaceType::Ground => 0.0,
            SurfaceType::InternalMass => 0.0,
            SurfaceType::InterZone => 0.0,
        }
    }

    /// Returns true if this surface type represents an inter-zone boundary.
    pub fn is_inter_zone(&self) -> bool {
        matches!(self, SurfaceType::InterZone)
    }
}

/// Inter-zone coupling conductance.
///
/// Represents the thermal conductance between two adjacent zones
/// through a shared boundary (wall, floor, etc.).
#[derive(Debug, Clone, Copy)]
pub struct ZoneCoupling {
    /// Conductance between zones (W/K)
    pub conductance: f64,
    /// Area of shared boundary (m²)
    pub area_m2: f64,
    /// Zone ID of the adjacent zone
    pub adjacent_zone_id: usize,
}

impl ZoneCoupling {
    /// Create a new zone coupling.
    pub fn new(conductance: f64, area_m2: f64, adjacent_zone_id: usize) -> Self {
        Self {
            conductance,
            area_m2,
            adjacent_zone_id,
        }
    }

    /// Compute heat transfer rate given temperature difference.
    pub fn heat_transfer(&self, delta_t: f64) -> f64 {
        self.conductance * delta_t
    }
}

/// Boundary conditions for a zone in a multi-zone configuration.
///
/// This extends the basic zone conditions with inter-zone coupling.
#[derive(Debug, Clone)]
pub struct ZoneBoundaryConditions {
    /// Exterior air temperature (°C)
    pub T_exterior: Temperature,
    /// Exterior film coefficient (W/m²·K)
    pub h_exterior: HeatTransferCoefficient,
    /// Solar irradiance on horizontal plane (W/m²)
    pub solar_irradiance_wm2: f64,
    /// Internal heat gains (W)
    pub Q_internal_w: f64,
    /// Infiltration heat gain/loss (W)
    pub Q_infiltration_w: f64,
    /// Air changes per hour for infiltration
    pub infiltration_ach: f64,
    /// Ventilation ACH from night_ventilation schedule (Issue #3904)
    pub ventilation_ach: f64,
    /// Coupled heat from adjacent zones (W)
    pub inter_zone_heat: f64,
    /// Night-sky radiative temperature (°C)
    pub t_sky: f64,
    /// Linearized sky-radiative conductance [W/m²·K]
    pub h_rad_sky: f64,
    /// Fraction of window solar that goes directly to zone air (ASHRAE 140 solar_distribution_to_air).
    /// 0.30 for LowMass (30% instant), 0.0 for HighMass per Issue #3911 / LIMIT-21.
    pub solar_distribution_to_air: f64,
    // Issue #3918: Threading parameters for solar lag correction
    /// Combined air-to-mass conductance [W/K]
    pub h_tr_3: f64,
    /// Zone thermal capacitance [J/K]
    pub cm: f64,
    /// Interior surface-to-air conductance [W/K]
    pub h_tr_is: f64,
    /// term_rest_1 = h_tr_ms + h_tr_is [W/K]
    pub term_rest_1: f64,
    /// Convective fraction of internal gains (Issue #3918)
    pub convective_fraction: f64,
    /// Solar beam-to-mass fraction (Issue #3918)
    pub solar_beam_to_mass_fraction: f64,
}

impl Default for ZoneBoundaryConditions {
    fn default() -> Self {
        Self {
            T_exterior: Temperature::from_value(20.0),
            h_exterior: HeatTransferCoefficient::from_value(25.0),
            solar_irradiance_wm2: 0.0,
            Q_internal_w: 0.0,
            Q_infiltration_w: 0.0,
            infiltration_ach: 0.5,
            ventilation_ach: 0.0, // Issue #3904: default no ventilation
            inter_zone_heat: 0.0,
            t_sky: 0.0,
            h_rad_sky: 0.0,
            solar_distribution_to_air: 0.0,
            // Issue #3918: defaults for lag parameters (0 = disabled if not threaded)
            h_tr_3: 0.0,
            cm: 0.0,
            h_tr_is: 0.0,
            term_rest_1: 0.0,
            convective_fraction: 0.0,
            solar_beam_to_mass_fraction: 0.0,
        }
    }
}

impl ZoneBoundaryConditions {
    /// Create new boundary conditions.
    pub fn new(
        T_exterior: Temperature,
        h_exterior: HeatTransferCoefficient,
        solar_irradiance_wm2: f64,
        solar_distribution_to_air: f64,
    ) -> Self {
        Self {
            T_exterior,
            h_exterior,
            solar_irradiance_wm2,
            Q_internal_w: 0.0,
            Q_infiltration_w: 0.0,
            infiltration_ach: 0.5,
            ventilation_ach: 0.0, // Issue #3904: default no ventilation
            inter_zone_heat: 0.0,
            t_sky: 0.0,
            h_rad_sky: 0.0,
            solar_distribution_to_air,
            h_tr_3: 0.0,
            cm: 0.0,
            h_tr_is: 0.0,
            term_rest_1: 0.0,
            convective_fraction: 0.0,
            solar_beam_to_mass_fraction: 0.0,
        }
    }

    /// Set sky radiative parameters.
    pub fn with_sky_radiation(mut self, t_sky: f64, h_rad_sky: f64) -> Self {
        self.t_sky = t_sky;
        self.h_rad_sky = h_rad_sky;
        self
    }

    /// Set internal heat gains.
    pub fn with_internal_gains(mut self, Q_internal_w: f64) -> Self {
        self.Q_internal_w = Q_internal_w;
        self
    }

    /// Set infiltration parameters.
    pub fn with_infiltration(mut self, Q_infiltration_w: f64, infiltration_ach: f64) -> Self {
        self.Q_infiltration_w = Q_infiltration_w;
        self.infiltration_ach = infiltration_ach;
        self
    }

    /// Set coupled heat from adjacent zones.
    pub fn with_inter_zone_heat(mut self, inter_zone_heat: f64) -> Self {
        self.inter_zone_heat = inter_zone_heat;
        self
    }
}

/// Per-surface gauge solver with geometric and type metadata.
///
/// # Clone semantics (Issue #3729)
///
/// `SurfaceGaugeSolver` does not derive `Clone`: the per-surface `GaugeSolver`
/// carries per-step internal state (`prev_T_interior`, `q_flux`,
/// `energy_storage_rate`, `initialized`) that must NOT round-trip across
/// `BatchOracle::evaluate_population`'s per-candidate clones — a clone of a
/// mid-solve surface would inherit the post-step interior temperature and
/// diverge from a fresh-from-scratch solver. The hand-rolled `Clone` impl
/// below preserves geometry (`area_m2`, `surface_type`, azimuth/tilt,
/// `wall_spec`) and re-initializes the per-surface gauge from `wall_spec`
/// when available (the common path — `add_opaque_surface` always sets
/// `wall_spec`). When `wall_spec` is `None` (the rare direct-`add_surface`
/// path) the cloned gauge retains its prior state and callers are
/// responsible for re-initializing.
pub(crate) struct SurfaceGaugeSolver {
    /// The 1D gauge solver for this surface
    gauge: GaugeSolver,
    /// Surface area in m²
    area_m2: f64,
    /// Surface type for solar distribution
    surface_type: SurfaceType,
    /// Surface azimuth (degrees, 0=South, 90=West, -90=East)
    _azimuth_deg: f64,
    /// Surface tilt from horizontal (degrees, 90=vertical wall, 0=roof)
    _tilt_deg: f64,
    /// Wall spec for initialization (stored for re-initialization if needed)
    wall_spec: Option<WallSpec>,
    /// Issue #3983 — solar absorbed by this surface's interior node during
    /// the most recent step [W]. Solar-only attribution of the interior
    /// network pools; `Σ_i + Φ_sol·to_air = Φ_sol` must hold exactly.
    last_solar_absorbed_w: f64,
    /// Issue #3918 follow-up (daytime solar gap): interior surface-node
    /// temperature T_s [°C]. Tracks this surface's response to interior
    /// absorbed gains (transmitted solar + radiative internal gains) and
    /// couples back to the zone air through
    /// h_tr_is,i = A_i · h_tr_is_coeff(tilt_i) inside
    /// `GaugeZoneSolver::step_interior_surface_network`.
    T_surface: f64,
}

/// Issue #3918 follow-up: does this surface participate in the interior
/// absorbed-gain network? Inter-zone partitions exchange through the explicit
/// coupling conductances (they carry no independent interior node), so they
/// are excluded; every other surface — including slab-on-grade floors and
/// internal mass — has an interior film and participates, matching the
/// ISO 13790 A_tot convention used by `compute_h_tr_is`.
fn surface_is_absorbing(st: &SurfaceType) -> bool {
    !st.is_inter_zone()
}

/// Issue #3928 / #3918 follow-up: tilt-dependent interior surface film
/// coefficient [W/m²K].
/// - Horizontal (tilt ≈ 0° or 180°): 2.3 W/m²K
/// - Vertical (tilt ≈ ±90°): 8.3 W/m²K
/// - Intermediate tilts: sin(tilt)-interpolated
fn h_tr_is_coeff(tilt_deg: f64) -> f64 {
    if tilt_deg.abs() < 1.0 || (tilt_deg - 180.0).abs() < 1.0 {
        2.3 // horizontal
    } else if (tilt_deg - 90.0).abs() < 1.0 || (tilt_deg + 90.0).abs() < 1.0 {
        8.3 // vertical
    } else {
        let tilt_rad = tilt_deg.to_radians();
        let ratio = tilt_rad.sin().abs();
        2.3 + ratio * (8.3 - 2.3)
    }
}

/// Parameters for computing the surface boundary conditions.
/// Groups the solar irradiance, sky temperature, and linearized sky conductance.
pub(crate) struct SurfaceBoundaryInput {
    /// Effective solar irradiance on this surface [W/m²]
    pub solar_irradiance_wm2: f64,
    /// Night-sky radiative temperature [°C]
    pub t_sky: f64,
    /// Linearized sky-radiative conductance [W/m²·K]
    pub h_rad_sky: f64,
}

impl SurfaceGaugeSolver {
    /// Create a new surface gauge solver.
    fn new(
        gauge: GaugeSolver,
        area_m2: f64,
        surface_type: SurfaceType,
        _azimuth_deg: f64,
        _tilt_deg: f64,
    ) -> Self {
        Self {
            gauge,
            area_m2,
            surface_type,
            _azimuth_deg,
            _tilt_deg,
            wall_spec: None,
            T_surface: 20.0,
            last_solar_absorbed_w: 0.0,
        }
    }

    /// Issue #3918 follow-up: interior surface-node temperature T_s [°C]
    /// from the most recent interior absorbed-gain network step.
    pub(crate) fn t_surface(&self) -> f64 {
        self.T_surface
    }

    /// Compute heat flux through this surface.
    #[allow(clippy::too_many_arguments)]
    fn compute_flux(
        &mut self,
        timestep: Time,
        T_interior: Temperature,
        T_exterior: Temperature,
        h_exterior: HeatTransferCoefficient,
        boundary: SurfaceBoundaryInput,
    ) -> Result<HeatFlux, SolverError> {
        let boundary_conds = GaugeBoundaryConditions::new(
            boundary.solar_irradiance_wm2,
            T_exterior.to_value(),
            boundary.t_sky,
            boundary.h_rad_sky,
        );
        self.gauge
            .step_with_boundary_conditions(timestep, T_interior, h_exterior, boundary_conds)
    }

    /// Issue #3297 / LIMIT-21 Phase 5: sky view factor from surface tilt.
    ///
    /// ISO 13790 §12.3.2 sky radiation exchange factor for a tilted surface:
    /// `F_sky = (1 + cos(β)) / 2` where β is the tilt from horizontal.
    /// - Horizontal roof (β=0°): F_sky = 1.0 (full sky dome view)
    /// - Vertical wall (β=90°): F_sky = 0.5 (half sky dome view)
    /// - Floor (β=180°): F_sky = 0.0 (no sky view, radiates to ground)
    fn sky_view_factor(&self) -> f64 {
        let tilt_rad = self._tilt_deg.to_radians();
        ((1.0 + tilt_rad.cos()) / 2.0).clamp(0.0, 1.0)
    }

    /// Issue #3297 / LIMIT-21 Phase 5: linearized sky-radiative conductance [W/m²K].
    ///
    /// Computes per-unit-area sky radiative conductance from `air_sky_conductance`
    /// (returns W/K total). The gauge formula expects h_rad_sky in W/m²K for its
    /// `h_rad_sky / h_exterior` dimensionless ratio. Uses T_exterior as the "air"
    /// node temperature proxy (iteration on T_air would be second-order; the
    /// h_rad_sky temperature sensitivity is O(T³) so the error is negligible).
    ///
    /// Returns 0.0 for surfaces with no sky view (Floor) or degenerate inputs.
    ///
    /// Default emissivity of 0.9 follows ASHRAE 140 / ISO 13790 convention
    /// for opaque exterior surfaces (gypsum board interior, concrete, brick).
    fn h_rad_sky_for_gauge(&self, t_exterior_c: f64, t_sky_c: f64) -> f64 {
        let f_sky = self.sky_view_factor();
        if f_sky <= 0.0 || self.area_m2 <= 0.0 {
            return 0.0;
        }
        // ASHRAE 140 default exterior emissivity for opaque surfaces (Issue #3297)
        let emissivity = 0.9;
        // air_sky_conductance returns W/K (total for this surface)
        let h_total = air_sky_conductance(emissivity, f_sky, self.area_m2, t_exterior_c, t_sky_c);
        // Convert to per-unit-area for the gauge formula: W/K / m² = W/(K·m²)
        h_total / self.area_m2
    }
}

// Issue #3729 — hand-rolled `Clone` for `SurfaceGaugeSolver`.
//
// A `#[derive(Clone)]` would deep-copy the per-surface `GaugeSolver` slot,
// inheriting the post-step `prev_T_interior` / `q_flux` / `initialized`
// flags verbatim across `BatchOracle::evaluate_population`'s per-candidate
// clones. This impl instead preserves the geometry / metadata and
// re-initializes the per-surface gauge from `wall_spec` when available —
// the common path (`add_opaque_surface` always sets `wall_spec`).
impl Clone for SurfaceGaugeSolver {
    fn clone(&self) -> Self {
        let mut clone = Self {
            gauge: self.gauge.clone(),
            area_m2: self.area_m2,
            surface_type: self.surface_type,
            _azimuth_deg: self._azimuth_deg,
            _tilt_deg: self._tilt_deg,
            wall_spec: self.wall_spec.clone(),
            // Issue #3918 follow-up: runtime surface-node state RESET on
            // clone (topology preserved, state reset — Issue #3729 contract).
            T_surface: 20.0,
            last_solar_absorbed_w: 0.0,
        };
        // Reset per-surface gauge state to the freshly-initialized form
        // by re-running `GaugeSolver::initialize` from the stored
        // `wall_spec`. The original surface was successfully initialized
        // (otherwise `GaugeZoneSolver::initialize` would have errored at
        // construction time), so the wall is valid and the re-init
        // succeeds. If `wall_spec` is `None` (direct-`add_surface` path),
        // the cloned gauge retains its prior state — callers in this
        // category must re-initialize manually before solving.
        if let Some(ref wall) = clone.wall_spec {
            // Best-effort: the source was already initialized with this
            // wall, so re-init cannot fail barring a programming error.
            // `log::warn` keeps the contract permissive in the rare
            // `None`-wall_spec case while still surfacing a regression.
            if let Err(e) = clone.gauge.initialize(wall) {
                log::warn!(
                    "SurfaceGaugeSolver::clone: re-initialize from wall_spec failed ({}); \
                     cloned gauge retains prior state. Caller is responsible for re-initialization.",
                    e
                );
            }
        }
        clone
    }
}

/// Zone-level heat balance solver using per-surface GaugeSolvers.
///
/// This struct owns the collection of per-surface 1D solvers and manages
/// the zone air node thermal capacitance. Supports both single-zone
/// and multi-zone configurations with inter-zone coupling.
///
/// # Clone semantics (Issue #3729)
///
/// `GaugeZoneSolver` does not derive `Clone`: it carries per-step
/// state (`T_air`, `initialized`, and the per-surface `GaugeSolver` slot
/// state which `SurfaceGaugeSolver::clone` resets on its own). A
/// `#[derive(Clone)]` would deep-copy `T_air` and `initialized`, silently
/// leaking a mid-solve zone air temperature into a freshly-cloned
/// `BatchOracle` candidate. The hand-rolled `Clone` impl below follows the
/// [`HybridThermalModel`] precedent (ARCHITECTURE.md §"Clone semantics &
/// BatchOracle parallelism contract", Issue #2539): **topology preserved,
/// runtime state reset**. Specifically:
///
/// | Field | On clone | Why |
/// |---|---|---|
/// | `surfaces`, `C_air`, `zone_volume`, `floor_area`, `num_surfaces`, `zone_id`, `couplings`, `inter_zone_conductance` | Deep-cloned | Pure geometry / adjacency data; round-trips correctly. |
/// | `T_air` | **Reset to `20.0`** (the `new_with_id` default) | A cloned candidate must start from the same zone air temperature a freshly-constructed solver would, regardless of the parent's mid-solve `T_air`. |
/// | `initialized` | **Reset to `false`** | The clone is NOT pre-solved; the caller invokes `GaugeZoneSolver::initialize()` (or relies on the dispatcher's first-step path) before solving. Matches the slot-reset contract applied to `HybridThermalModel::conduction_solver`. |
/// | `previous_T_surface` | **Reset to `20.0`** | Issue #3920: Must start from a known thermal state like T_air. The h_tr_is coupling requires a valid previous T_surface for the first step. |
///
/// Per-surface `GaugeSolver` state is reset transitively by
/// [`SurfaceGaugeSolver::clone`] (re-initialized from `wall_spec` when
/// available — the common path; otherwise the caller is responsible).
///
/// **Independence guarantee.** Two clones are fully independent: mutating
/// `clone.T_air` (or stepping `clone`) does not affect `original.T_air`,
/// and vice versa. Pinned by `tests/all_tests/gauge_conduction_backend_clone.rs`.
pub struct GaugeZoneSolver {
    /// Per-surface gauge solvers (walls, windows, roof, floor)
    surfaces: Vec<SurfaceGaugeSolver>,
    /// Zone air thermal capacitance (J/K)
    /// C_air = rho * V * c_p ≈ 1.2 * (floor_area * height) * 1006 J/K
    C_air: f64,
    /// Current zone air temperature (°C)
    T_air: f64,
    /// Zone volume (m³)
    zone_volume: f64,
    /// Floor area (m²)
    floor_area: f64,
    /// Number of surfaces
    num_surfaces: usize,
    /// Solver initialized flag
    initialized: bool,
    /// Zone identifier (for multi-zone coupling)
    zone_id: usize,
    /// Inter-zone couplings to adjacent zones
    couplings: Vec<ZoneCoupling>,
    /// Pre-computed inter-zone conductance matrix (zone_id -> conductance)
    inter_zone_conductance: HashMap<usize, f64>,
    /// Number of sub-steps for the air-node update per timestep.
    /// At dt/τ_air ≈ 3.6 on a 1-hour timestep, splitting into N sub-steps
    /// reduces dt/τ to ≈ 3.6/N, within stability bounds when N ≥ 3.
    /// Default: 3 (matching 5R1C sub_hour_air_node_steps).
    sub_hour_air_node_steps: u32,
    // Issue #3920 / LIMIT-21: Area-weighted mean interior surface temperature for
    // h_tr_is coupling. Computed as T_air + Q_gauge_total / h_tr_is where
    // Q_gauge_total = sum of q_flux * area for all surfaces. Stored for use
    // in subsequent timesteps to provide proper thermal coupling.
    previous_T_surface: f64,
}

// Issue #3729 — hand-rolled `Clone` for `GaugeZoneSolver`.
//
// See the struct doc-comment for the full table of preserved-vs-reset
// fields. Summary: topology preserved (geometry, adjacency, surface
// metadata, `num_surfaces`, `zone_id`); runtime state reset (`T_air`
// to the `new_with_id` default of `20.0`, `initialized` to `false`).
// Per-surface `GaugeSolver` state is reset transitively by the
// `SurfaceGaugeSolver::clone` impl above.
impl Clone for GaugeZoneSolver {
    fn clone(&self) -> Self {
        Self {
            surfaces: self.surfaces.clone(),
            C_air: self.C_air,
            T_air: 20.0, // RESET — see struct doc-comment.
            zone_volume: self.zone_volume,
            floor_area: self.floor_area,
            num_surfaces: self.num_surfaces,
            initialized: false, // RESET — see struct doc-comment.
            zone_id: self.zone_id,
            couplings: self.couplings.clone(),
            inter_zone_conductance: self.inter_zone_conductance.clone(),
            sub_hour_air_node_steps: self.sub_hour_air_node_steps, // preserved on clone
            previous_T_surface: 0.0, // RESET — see struct doc-comment.
        }
    }
}

/// Physical constants for air
mod air_constants {
    /// Air density at standard conditions (kg/m³)
    pub const RHO_AIR: f64 = 1.2;

    /// Air specific heat at constant pressure (J/kg·K)
    pub const CP_AIR: f64 = 1006.0;

    /// Calculate zone air thermal capacitance (J/K)
    pub fn zone_air_capacitance(floor_area: f64, ceiling_height: f64) -> f64 {
        let volume = floor_area * ceiling_height;
        RHO_AIR * volume * CP_AIR
    }
}

impl GaugeZoneSolver {
    /// Create a new GaugeZoneSolver with the given zone geometry.
    pub fn new(floor_area: f64, ceiling_height: f64) -> Self {
        Self::new_with_id(0, floor_area, ceiling_height)
    }

    /// Create a new GaugeZoneSolver with a specific zone ID.
    pub fn new_with_id(zone_id: usize, floor_area: f64, ceiling_height: f64) -> Self {
        let zone_volume = floor_area * ceiling_height;
        let C_air = air_constants::zone_air_capacitance(floor_area, ceiling_height);

        Self {
            surfaces: Vec::new(),
            C_air,
            T_air: 20.0,
            zone_volume,
            floor_area,
            num_surfaces: 0,
            initialized: false,
            zone_id,
            couplings: Vec::new(),
            inter_zone_conductance: HashMap::new(),
            sub_hour_air_node_steps: 3, // default: 3 sub-steps per timestep (matching 5R1C)
            previous_T_surface: 0.0,    // Issue #3920: initialized to 0, computed each step
        }
    }

    /// Read-only accessor for the derived zone volume (m³).
    ///
    /// The dispatcher derives ventilation ACH from `setpoints.zone_volume`
    /// while `h_vent` is computed against `self.zone_volume` (= floor_area ×
    /// ceiling_height); a mismatch between the two scales `h_vent` by their
    /// ratio (Issue #3962 diagnostics).
    pub fn zone_volume_m3(&self) -> f64 {
        self.zone_volume
    }

    /// Add a surface with its gauge solver and geometric properties.
    pub fn add_surface(
        &mut self,
        gauge: GaugeSolver,
        area_m2: f64,
        surface_type: SurfaceType,
        azimuth_deg: f64,
        tilt_deg: f64,
    ) {
        self.surfaces.push(SurfaceGaugeSolver::new(
            gauge,
            area_m2,
            surface_type,
            azimuth_deg,
            tilt_deg,
        ));
        self.num_surfaces = self.surfaces.len();
    }

    /// Add an inter-zone coupling to an adjacent zone.
    ///
    /// This establishes thermal coupling between this zone and an adjacent zone
    /// through a shared boundary (e.g., interior wall, floor between levels).
    ///
    /// # Arguments
    /// * `adjacent_zone_id` - ID of the adjacent zone
    /// * `shared_area_m2` - Area of shared boundary (m²)
    /// * `R_value` - Thermal resistance of shared boundary (m²·K/W)
    pub fn add_zone_coupling(
        &mut self,
        adjacent_zone_id: usize,
        shared_area_m2: f64,
        R_value: f64,
    ) -> Result<(), SolverError> {
        if R_value <= 0.0 {
            return Err(SolverError::ConstructionError(
                "Inter-zone R_value must be positive".to_string(),
            ));
        }
        let conductance = shared_area_m2 / R_value;
        self.couplings.push(ZoneCoupling::new(
            conductance,
            shared_area_m2,
            adjacent_zone_id,
        ));
        *self
            .inter_zone_conductance
            .entry(adjacent_zone_id)
            .or_insert(0.0) += conductance;
        Ok(())
    }

    /// Get the zone ID.
    pub fn zone_id(&self) -> usize {
        self.zone_id
    }

    /// Get inter-zone conductance to a specific adjacent zone.
    pub fn inter_zone_conductance(&self, adjacent_zone_id: usize) -> f64 {
        self.inter_zone_conductance
            .get(&adjacent_zone_id)
            .copied()
            .unwrap_or(0.0)
    }

    /// Get all inter-zone couplings.
    pub fn couplings(&self) -> &[ZoneCoupling] {
        &self.couplings
    }

    /// Add an opaque surface from a WallSpec.
    pub fn add_opaque_surface(
        &mut self,
        wall: &WallSpec,
        area_m2: f64,
        surface_type: SurfaceType,
        azimuth_deg: f64,
        tilt_deg: f64,
    ) -> Result<(), SolverError> {
        let mut gauge = GaugeSolver::default();
        gauge.initialize(wall)?;

        let mut surface =
            SurfaceGaugeSolver::new(gauge, area_m2, surface_type, azimuth_deg, tilt_deg);
        surface.wall_spec = Some(wall.clone());

        self.surfaces.push(surface);
        self.num_surfaces = self.surfaces.len();
        Ok(())
    }

    /// Get current zone air temperature.
    pub fn T_air(&self) -> Temperature {
        Temperature::from_value(self.T_air)
    }

    /// Set zone air temperature (for test initialization).
    pub fn set_T_air(&mut self, temp: f64) {
        self.T_air = temp;
    }

    /// Set number of sub-steps per timestep for air-node update.
    /// At dt/τ_air ≈ 3.6 on a 1-hour timestep, use N ≥ 3 for stability.
    /// Default is 3 (matching 5R1C sub_hour_air_node_steps).
    pub fn set_sub_hour_air_node_steps(&mut self, steps: u32) {
        self.sub_hour_air_node_steps = steps;
    }

    /// Get zone air thermal capacitance.
    pub fn C_air(&self) -> f64 {
        self.C_air
    }

    /// Check if solver is initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized && !self.surfaces.is_empty()
    }

    /// Initialize all surface gauge solvers.
    pub fn initialize(&mut self) -> Result<(), SolverError> {
        if self.surfaces.is_empty() {
            return Err(SolverError::InvalidConfig(
                "GaugeZoneSolver has no surfaces. Add surfaces before initializing.".to_string(),
            ));
        }

        for surface in &mut self.surfaces {
            if let Some(ref wall) = surface.wall_spec {
                surface.gauge.initialize(wall)?;
            }
        }

        self.initialized = true;
        Ok(())
    }

    /// Compute steady-state heat flux (no thermal mass).
    pub fn steady_state_flux(
        &self,
        T_interior: Temperature,
        T_exterior: Temperature,
    ) -> Result<HeatFlux, SolverError> {
        if !self.is_initialized() {
            return Err(SolverError::InvalidConfig(
                "GaugeZoneSolver not initialized".to_string(),
            ));
        }

        let mut total_Q = 0.0;
        for surface in &self.surfaces {
            let q_flux = surface.gauge.steady_state_flux(T_interior, T_exterior)?;
            let q = q_flux.to_value();
            total_Q += q * surface.area_m2;
        }

        // Return flux per unit area (W/m² of floor area)
        let flux_wm2 = total_Q / self.floor_area;
        Ok(HeatFlux::from_value(flux_wm2))
    }

    /// Step the zone model forward by one timestep.
    ///
    /// Computes per-surface fluxes, aggregates at zone air node, and updates T_air.
    ///
    /// # Arguments
    /// * `timestep` - Current timestep index (used for diagnostics)
    /// * `dt_seconds` - Timestep duration in seconds
    /// * `T_exterior` - Exterior air temperature (°C)
    /// * `h_exterior` - Exterior film coefficient (W/m²·K)
    /// * `solar_irradiance_wm2` - Total solar irradiance on horizontal plane (W/m²)
    /// * `solar_distribution_to_air` - Fraction of window solar that goes directly to zone air
    /// * `Q_internal_w` - Internal heat gains (W)
    /// * `Q_infiltration_w` - Infiltration heat gain/loss (W)
    /// * `t_sky` - Night-sky radiative temperature (°C)
    /// * `h_rad_sky` - Linearized sky-radiative conductance [W/m²·K]
    /// * `h_tr_3` - Combined air-to-mass conductance for lag [W/K] (Issue #3918)
    /// * `cm` - Zone thermal capacitance for lag [J/K] (Issue #3918)
    /// * `h_tr_is` - Interior surface-to-air conductance [W/K] (Issue #3918)
    /// * `term_rest_1` - h_tr_ms + h_tr_is [W/K] (Issue #3918)
    /// * `convective_fraction` - Convective fraction of internal gains (Issue #3918)
    /// * `solar_beam_to_mass_fraction` - Solar beam-to-mass fraction (Issue #3918)
    ///
    /// # Returns
    /// Net zone load in kWh (positive = heating needed, negative = cooling needed)
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        _timestep: usize,
        dt_seconds: f64,
        T_exterior: Temperature,
        h_exterior: HeatTransferCoefficient,
        solar_irradiance_wm2: f64,
        solar_distribution_to_air: f64,
        Q_internal_w: f64,
        Q_infiltration_w: f64,
        t_sky: f64,
        // Issue #3297 Phase 5: per-surface h_rad_sky is now computed inside
        // the surface loop using air_sky_conductance(); this parameter is retained
        // for API compatibility but is ignored (prefixed _ to silence warning).
        _h_rad_sky: f64,
        // Issue #3904: Ventilation ACH from night_ventilation schedule
        ventilation_ach: f64,
        // Issue #3918 + follow-up: the former scalar lag-correction inputs
        // `h_tr_3`, `cm`, and `term_rest_1` are retained for API stability
        // but no longer used — the per-surface T_s network derives its time
        // constants (C_i/h_tr_is,i) from each surface's own gauge
        // capacitance. `h_tr_is` IS used: it is the zone star-node
        // conductance the network distributes per-surface by area share.
        _h_tr_3: f64,
        _cm: f64,
        h_tr_is: f64,
        _term_rest_1: f64,
        convective_fraction: f64,
        solar_beam_to_mass_fraction: f64,
    ) -> Result<f64, SolverError> {
        if !self.is_initialized() {
            return Err(SolverError::InvalidConfig(
                "GaugeZoneSolver not initialized".to_string(),
            ));
        }

        // Issue #3817 / LIMIT-05 / LIMIT-16: At dt/τ_air ≈ 3.6 on a 1-hour
        // timestep, the air node equilibrates ~98% each step, making the
        // conditionally-stable implicit Euler scheme diverge when H·dt > C_air
        // (Case 900FF heavyweight concrete). Sub-stepping mirrors the 5R1C
        // fix (Issue #2339): splitting into N sub-steps reduces dt/τ to
        // ≈ 3.6/N, within stability bounds when N ≥ 3.
        //
        // Surface fluxes are computed ONCE per outer timestep (at T_air_old)
        // and reused across all sub-steps. This is consistent with the 5R1C
        // approach where driving terms remain constant during sub-stepping.
        let T_int = Temperature::from_value(self.T_air);
        let mut net_power_watts = 0.0;

        // Sum heat flux from all surfaces (computed once at T_air_old).
        // Issue #3297 / LIMIT-21 Phase 5: per-surface h_rad_sky via
        // air_sky_conductance(), keyed to each surface's sky-view factor
        // derived from its tilt angle. Uses T_exterior as the air-node
        // temperature proxy (iteration on T_air would be second-order).
        //
        // Issue #3918 follow-up (daytime solar gap): transmitted window
        // solar NO LONGER enters the window gauge's exterior sol-air node.
        // The dispatcher's `solar_irradiance_wm2` is the zone solar-gain
        // density [W per m² of FLOOR area] — the same 5R1C `solar_gains[i]`
        // quantity that model multiplies by `zone_area[i]` (step_5r1c.rs:114)
        // — and injecting it as exterior absorbed irradiance shed most of
        // it back through the exterior film: the sol-air formulation
        // t_ext = T_out + G/h_ext driving q = (t_ext − T_in)/R_total delivers
        // only 1/(h_ext·R_total) ≈ 16% of G for a U=3.0 window, and the old
        // per-window-area scaling additionally starved it by A_win/A_floor
        // (12/48 for Case 600). Transmitted solar is instead absorbed by the
        // interior surfaces via the per-surface T_s network
        // (`step_interior_surface_network` below).
        let t_exterior_c = T_exterior.to_value();
        for surface in &mut self.surfaces {
            let h_rad_sky_surface = surface.h_rad_sky_for_gauge(t_exterior_c, t_sky);

            // Issue #3918: At night (solar_irradiance_wm2 == 0), neutralize the sky radiative
            // term by setting t_sky = t_exterior_c. This disables sky radiative forcing since
            // (t_sky - t_exterior) = 0, matching 5R1C behavior which doesn't include sky
            // radiation in the free-floating zone air temperature balance.
            let t_sky_effective = if solar_irradiance_wm2.abs() < 1e-6 {
                t_exterior_c
            } else {
                t_sky
            };

            let q_flux = surface.compute_flux(
                Time::from_value(dt_seconds),
                T_int,
                T_exterior,
                h_exterior,
                SurfaceBoundaryInput {
                    solar_irradiance_wm2: 0.0,
                    t_sky: t_sky_effective,
                    h_rad_sky: h_rad_sky_surface,
                },
            )?;

            let Q_surface = q_flux.to_value() * surface.area_m2;
            net_power_watts += Q_surface;
        }

        // Issue #3918 follow-up (daytime solar gap): route zone solar and
        // internal gains through the 5R1C three-way split (phi_ia / phi_st /
        // phi_m, step_5r1c.rs:96-130) computed on the FULL zone solar
        // Φ_sol = solar_irradiance_wm2 · floor_area [W].
        let phi_sol_total = solar_irradiance_wm2 * self.floor_area;
        let phi_ia_sol = phi_sol_total * solar_distribution_to_air;
        let remaining_sol = phi_sol_total - phi_ia_sol;
        let phi_ia_int = Q_internal_w * convective_fraction;
        let phi_int_rad = Q_internal_w - phi_ia_int;
        let st_sol_frac = 1.0 - solar_beam_to_mass_fraction;
        let surface_pool =
            remaining_sol * st_sol_frac + phi_int_rad * (1.0 - solar_distribution_to_air);
        let mass_pool =
            remaining_sol * solar_beam_to_mass_fraction + phi_int_rad * solar_distribution_to_air;
        // Solar-only pool portions for the per-surface absorption telemetry.
        let solar_surface_pool = remaining_sol * st_sol_frac;
        let solar_mass_pool = remaining_sol * solar_beam_to_mass_fraction;
        // Direct-to-air gains (5R1C phi_ia) enter the air balance instantly.
        net_power_watts += phi_ia_sol + phi_ia_int;
        // Interior absorbed-gain network: per-surface T_s tracking returning
        // φ_st = Σ h_tr_is,i·(T_s,i − T_air) as an energy-exact timestep
        // average (Issue #3918 architectural fix). `h_interior` is the
        // network's air-side conductance — it enters den_air below as the
        // implicit-conductance counterpart of φ_st.
        let (phi_st, h_interior) = self.step_interior_surface_network(
            dt_seconds,
            surface_pool,
            mass_pool,
            solar_surface_pool,
            solar_mass_pool,
            h_tr_is,
        );
        net_power_watts += phi_st;

        // Infiltration/ventilation coupling
        // Issue #3904: Ventilation ACH is now passed as a parameter instead of hardcoded 0.
        // infiltration_ach remains 0.5 for ASHRAE 140 Case 600 (baseline infiltration).
        let infiltration_ach = 0.5; // ASHRAE 140 Case 600
        let h_inf = air_constants::RHO_AIR
            * air_constants::CP_AIR
            * (infiltration_ach / 3600.0)
            * self.zone_volume;
        // Issue #3904: h_vent is computed from ventilation_ach parameter (night ventilation)
        let h_vent = air_constants::RHO_AIR
            * air_constants::CP_AIR
            * (ventilation_ach / 3600.0)
            * self.zone_volume;
        let h_total = h_vent + h_inf;

        // Issue #3918: envelope transmission conductance Σ_i A_i/R_i [W/K] over
        // all non-inter-zone surfaces — the gauge analog of the 5R1C air-node
        // denominator terms H_tr,1 + H_tr,w (opaque + window transmission).
        //
        // The per-surface fluxes accumulated in `net_power_watts` were
        // evaluated at T_air_old and are linear in T_air with slope −h_env
        // (q_flux = (t_ext − T_air)/R_total). Omitting h_env from the air-node
        // denominator therefore treats the envelope loss as a fixed heat
        // source divided by the ventilation-only conductance (~21.7 W/K vs
        // ~90 W/K of envelope coupling for Case 600), yielding the physically
        // impossible steady state −169 °C at ΔT = 38 K and the unstable
        // fixed-point map T_{n+1} = T_ext + Q_env(T_n)/h_ve with slope
        // ≈ −UA/h_ve ≈ −4 (observed oscillation 20 → −54 → 19 °C).
        let h_env = self.surface_to_air_conductance();
        // True air-node denominator — the exact analog of 5R1C's den_true
        // (step_5r1c.rs:963-967: den_true = den / term_rest_1 =
        // H_tr,1 + H_tr,w + H_ve + H_tr,floor + h_ms_is_prod/term_rest_1).
        //
        // Issue #3918 follow-up: `h_interior` (Σ h_tr_is,i of the interior
        // network) is the air-side conductance the T_s network presents to
        // the air node. φ_st inside `net_power_watts` is its frozen-flux
        // driving counterpart — the same implicit linearization the envelope
        // fluxes get through `h_env` — so the ultimate steady state stays
        // T_ext + Φ_gains/(h_env + h_total): the interior film circulates
        // heat but leaks nothing.
        let den_air = h_env + h_total + h_interior;

        let T_ext_val = T_exterior.to_value();
        // Quasi-steady-state temperature (constant over all sub-steps, like 5R1C's "steady")
        //
        // Issue #3918: exact analog of 5R1C `steady = num / den`
        // (step_5r1c.rs:926). The surface fluxes inside `net_power_watts`
        // are frozen at T_air_old, so linearize them implicitly around that
        // point:  Q_surface(T) ≈ Q_surface(T_air_old) + h_env·(T_air_old − T).
        // The quasi-steady state of
        //   C_air·dT/dt = Q_net(T_air_old) + h_env·(T_air_old − T) + h_total·(T_ext − T)
        // is therefore bounded:
        //   t_steady = T_air_old + [Q_net(T_air_old) + h_total·(T_ext − T_air_old)] / (h_env + h_total)
        // instead of the divergent `T_ext + Q_net / h_total` that produced
        // −169 °C driving temperatures and the 20 → −54 °C oscillation.
        // Issue #3918 follow-up: `net_power_watts` already carries the full
        // interior absorbed-gain response φ_st from the per-surface T_s
        // network, replacing the former scalar solar-lag temperature
        // increment.
        let T_air_old = self.T_air;
        let (t_steady, tau_air) = if den_air > 0.0 {
            (
                T_air_old + (net_power_watts + h_total * (T_ext_val - T_air_old)) / den_air,
                self.C_air / den_air,
            )
        } else {
            // No air-coupled conductance at all (no surfaces, no ventilation):
            // hold the air state (exp(−dt/∞) = 1).
            (T_air_old, f64::INFINITY)
        };
        // Sub-stepping for the exact exponential solution
        let steps = self.sub_hour_air_node_steps as usize;
        let dt_sub = dt_seconds / steps as f64;
        let mut T_air_current = self.T_air;

        for _ in 0..steps {
            // Exact exponential update over dt_sub
            let exponent = -dt_sub / tau_air;
            T_air_current = t_steady + (T_air_current - t_steady) * exponent.exp();
        }

        // Update T_air
        self.T_air = T_air_current;

        // Add infiltration heat contribution to net power for return value
        net_power_watts += Q_infiltration_w;

        // Return net energy in kWh
        // Convention: positive = heating needed, negative = cooling needed
        let energy_kwh = -(net_power_watts * dt_seconds) / 3_600_000.0;
        Ok(energy_kwh)
    }

    /// Access the per-surface solvers (for diagnostics).
    #[allow(private_interfaces)]
    pub fn surfaces(&self) -> &[SurfaceGaugeSolver] {
        &self.surfaces
    }

    /// Issue #3729 — public surface count accessor so external callers
    /// (e.g. integration tests under `tests/all_tests/`) can verify the
    /// topology-preservation contract without depending on the
    /// `pub(crate)` `SurfaceGaugeSolver` type. The returned length
    /// matches `self.surfaces().len()`.
    pub fn surface_count(&self) -> usize {
        self.surfaces.len()
    }

    /// Issue #3297 — per-surface interior temperature (the interior-most
    /// node state of each surface's 1D solve) at the most recent step.
    /// Length matches `self.surfaces.len()`. Read-only telemetry: used by
    /// the dispatch to compute a 5R1C-compatible mass-state proxy; the
    /// gauge integration never consumes these values.
    pub fn surface_interior_temperatures(&self) -> Vec<f64> {
        self.surfaces
            .iter()
            .map(|surface| surface.gauge.interior_temperature())
            .collect()
    }

    /// Issue #3297 — area-weighted mean of the per-surface interior
    /// temperatures for this zone. Falls back to the current zone air
    /// temperature when the zone has no surfaces (or zero total area).
    pub fn mean_interior_surface_temperature(&self) -> f64 {
        let mut weighted_sum = 0.0;
        let mut total_area = 0.0;
        for surface in &self.surfaces {
            weighted_sum += surface.gauge.interior_temperature() * surface.area_m2;
            total_area += surface.area_m2;
        }
        if total_area > 0.0 {
            weighted_sum / total_area
        } else {
            self.T_air
        }
    }

    /// Issue #3920 / LIMIT-21 — the previous-timestep area-weighted mean interior
    /// surface temperature (T_air + Q_gauge_total / h_tr_is), used for h_tr_is
    /// coupling in the thermal network. Returns 0.0 when h_tr_is <= 0 (not yet
    /// computed) or when the solver has not been stepped.
    pub fn previous_T_surface(&self) -> f64 {
        self.previous_T_surface
    }

    /// Step the zone model with inter-zone coupling.
    ///
    /// This extends the basic step with coupling to adjacent zones.
    /// The inter-zone heat is computed from the temperature difference
    /// between this zone and its neighbors.
    ///
    /// # Arguments
    /// * `dt_seconds` - Timestep duration in seconds
    /// * `bc` - Zone boundary conditions
    /// * `adjacent_temperatures` - Map of zone_id -> temperature for adjacent zones
    ///
    /// # Returns
    /// Net zone load in kWh (positive = heating needed, negative = cooling needed)
    pub fn step_with_coupling(
        &mut self,
        dt_seconds: f64,
        bc: &ZoneBoundaryConditions,
        adjacent_temperatures: &HashMap<usize, Temperature>,
    ) -> Result<f64, SolverError> {
        if !self.is_initialized() {
            return Err(SolverError::InvalidConfig(
                "GaugeZoneSolver not initialized".to_string(),
            ));
        }

        // Issue #3817 / LIMIT-05 / LIMIT-16: Sub-stepping for stability.
        // Same pattern as step(): surface fluxes and inter-zone terms are
        // computed once at T_air_old and reused across sub-steps.
        let T_int = Temperature::from_value(self.T_air);
        let mut net_power_watts = 0.0;
        // Issue #3889 — solar-aware exterior-surface flux sum.
        let mut q_surfaces_w = 0.0;

        // Sum heat flux from all exterior surfaces (computed once at T_air_old).
        // Issue #3297 / LIMIT-21 Phase 5: per-surface h_rad_sky via
        // air_sky_conductance(), keyed to each surface's sky-view factor
        // derived from its tilt angle.
        //
        // Issue #3918 follow-up (daytime solar gap): transmitted window
        // solar is absorbed by the interior surface network below (see the
        // full rationale in step()); the window gauge receives no exterior
        // solar injection.
        let t_exterior_c = bc.T_exterior.to_value();
        for surface in &mut self.surfaces {
            if surface.surface_type.is_inter_zone() {
                continue;
            }
            let h_rad_sky_surface = surface.h_rad_sky_for_gauge(t_exterior_c, bc.t_sky);

            // Issue #3918: At night (solar_irradiance_wm2 == 0), neutralize the sky radiative
            // term by setting t_sky = t_exterior_c. This disables sky radiative forcing since
            // (t_sky - t_exterior) = 0, matching 5R1C behavior which doesn't include sky
            // radiation in the free-floating zone air temperature balance.
            let t_sky_effective = if bc.solar_irradiance_wm2.abs() < 1e-6 {
                t_exterior_c
            } else {
                bc.t_sky
            };

            let q_flux = surface.compute_flux(
                Time::from_value(dt_seconds),
                T_int,
                bc.T_exterior,
                bc.h_exterior,
                SurfaceBoundaryInput {
                    solar_irradiance_wm2: 0.0,
                    t_sky: t_sky_effective,
                    h_rad_sky: h_rad_sky_surface,
                },
            )?;

            let Q_surface = q_flux.to_value() * surface.area_m2;
            net_power_watts += Q_surface;
            q_surfaces_w += Q_surface;
        }

        // Compute inter-zone heat transfer (evaluated at T_air_old)
        let mut inter_zone_heat = 0.0;
        for coupling in &self.couplings {
            if let Some(&T_adjacent) = adjacent_temperatures.get(&coupling.adjacent_zone_id) {
                let Q_transfer = coupling.heat_transfer(T_int.to_value() - T_adjacent.to_value());
                inter_zone_heat += Q_transfer;
            }
        }
        net_power_watts += inter_zone_heat + bc.inter_zone_heat;

        // Issue #3918 follow-up (daytime solar gap): route zone solar and
        // internal gains through the 5R1C three-way split (phi_ia / phi_st /
        // phi_m, step_5r1c.rs:96-130) computed on the FULL zone solar
        // Φ_sol = solar_irradiance_wm2 · floor_area [W]. `gains_w` carries
        // every non-surface-flux gain for the sub-step driving numerator —
        // including the direct-to-air solar that previously reached only
        // the returned energy, never the T_air update (latent multi-zone
        // routing bug fixed en passant).
        let phi_sol_total = bc.solar_irradiance_wm2 * self.floor_area;
        let phi_ia_sol = phi_sol_total * bc.solar_distribution_to_air;
        let remaining_sol = phi_sol_total - phi_ia_sol;
        let phi_ia_int = bc.Q_internal_w * bc.convective_fraction;
        let phi_int_rad = bc.Q_internal_w - phi_ia_int;
        let st_sol_frac = 1.0 - bc.solar_beam_to_mass_fraction;
        let surface_pool =
            remaining_sol * st_sol_frac + phi_int_rad * (1.0 - bc.solar_distribution_to_air);
        let mass_pool = remaining_sol * bc.solar_beam_to_mass_fraction
            + phi_int_rad * bc.solar_distribution_to_air;
        // Solar-only pool portions for the per-surface absorption telemetry.
        let solar_surface_pool = remaining_sol * st_sol_frac;
        let solar_mass_pool = remaining_sol * bc.solar_beam_to_mass_fraction;
        let (phi_st, h_interior) = self.step_interior_surface_network(
            dt_seconds,
            surface_pool,
            mass_pool,
            solar_surface_pool,
            solar_mass_pool,
            bc.h_tr_is,
        );
        let gains_w = phi_ia_sol + phi_ia_int + phi_st;
        net_power_watts += gains_w;

        // Infiltration/ventilation coupling
        // Issue #3904: Compute h_vent from ventilation_ach (night ventilation)
        let h_inf = air_constants::RHO_AIR
            * air_constants::CP_AIR
            * (bc.infiltration_ach / 3600.0)
            * self.zone_volume;
        let h_vent = air_constants::RHO_AIR
            * air_constants::CP_AIR
            * (bc.ventilation_ach / 3600.0)
            * self.zone_volume;
        let h_total = h_vent + h_inf;

        // Issue #3918: envelope transmission conductance Σ_i A_i/R_i [W/K]
        // (non-inter-zone surfaces) — must appear in the air-node denominator
        // exactly like 5R1C's den_true; see the step() fix above.
        let h_env = self.surface_to_air_conductance();

        let h_inter_zone_total: f64 = self.couplings.iter().map(|c| c.conductance).sum();

        // Σ_j H_ij · T_adj,j — inter-zone driving term evaluated at the
        // adjacent zones' temperatures (constant during sub-stepping).
        let inter_zone_drive: f64 = self
            .couplings
            .iter()
            .filter_map(|c| {
                let t_adj = adjacent_temperatures.get(&c.adjacent_zone_id)?.to_value();
                Some(c.conductance * t_adj)
            })
            .sum();

        let T_ext_val = bc.T_exterior.to_value();

        // Sub-stepping loop: N iterations with dt/N per sub-step
        let steps = self.sub_hour_air_node_steps as usize;
        let dt_sub = dt_seconds / steps as f64;
        let mut T_air_current = self.T_air;
        // Issue #3918: linearization reference — the frozen surface fluxes
        // (q_surfaces_w) were evaluated at T_air_old, so the implicit
        // conductance h_env must enter with that driving temperature.
        let T_air_old = self.T_air;
        // Total air-coupled conductance: envelope + ventilation + inter-zone
        // + interior network Σ h_tr_is,i (Issue #3918 follow-up, see step()).
        let den_air = h_env + h_total + h_inter_zone_total + h_interior;

        for _ in 0..steps {
            // Implicit Euler update with dt_sub
            // T_air_new = (C·T_old + dt·(h_env·T_air_old + Q_surfaces + h_total·T_ext + inter_zone_drive + gains_w + Q_inter_zone_fixed))
            //             / (C + dt·(h_env + h_total + h_inter_zone_total + mass_coupling))
            //
            // Issue #3918: `h_env·T_air_old` / `h_env` pair linearizes the
            // frozen per-surface fluxes implicitly (the envelope loss is not a
            // fixed heat source — it scales with T_air through Σ A_i/R_i),
            // and `bc.inter_zone_heat` (a fixed W input) is now included in
            // the driving numerator, matching its treatment in
            // `net_power_watts` above.
            //
            // Issue #3918 follow-up: `gains_w` (direct-to-air solar +
            // convective internal gains + the interior-network φ_st) replaces
            // the bare `bc.Q_internal_w` — the direct-to-air solar previously
            // reached only the returned energy, never the T_air update.
            // φ_st is a DEVIATION flux (Σ h_is·(T̄_s − T_air_old)); this
            // absolute-form update therefore adds back `h_interior·T_air_old`
            // so the network's implicit conductance carries the matching
            // absolute driving term h_interior·T̄_s (equilibrium check: at
            // T_s = T_air = T_ext the interior terms cancel and the zone
            // settles exactly at ambient, as pinned by
            // step_with_coupling_no_solar_settles_at_ambient).
            T_air_current = (self.C_air * T_air_current
                + dt_sub
                    * (h_env * T_air_old
                        + q_surfaces_w
                        + h_total * T_ext_val
                        + inter_zone_drive
                        + gains_w
                        + h_interior * T_air_old
                        + bc.inter_zone_heat))
                / (self.C_air + den_air * dt_sub);
        }
        self.T_air = T_air_current;

        // Return net energy in kWh
        let energy_kwh = -(net_power_watts * dt_seconds) / 3_600_000.0;
        Ok(energy_kwh)
    }

    /// Compute inter-zone coupling matrix contribution.
    ///
    /// Returns the heat exchange vector for all adjacent zones.
    /// Used by MultiZoneGaugeSolver to build the global coupling system.
    pub fn compute_zone_coupling_vector(
        &self,
        adjacent_temperatures: &HashMap<usize, Temperature>,
    ) -> HashMap<usize, f64> {
        let T_int = self.T_air;
        let mut coupling_vector = HashMap::new();

        for coupling in &self.couplings {
            if let Some(&T_adjacent) = adjacent_temperatures.get(&coupling.adjacent_zone_id) {
                let Q_transfer = coupling.heat_transfer(T_int - T_adjacent.to_value());
                *coupling_vector
                    .entry(coupling.adjacent_zone_id)
                    .or_insert(0.0) += Q_transfer;
            }
        }

        coupling_vector
    }

    /// Issue #3911 — Per-surface telemetry from each surface's 1D gauge solve.
    ///
    /// Returns a vector of per-surface diagnostic data for the LIMIT-21 Phase 6
    /// diagnostic (Case 640 annual cooling gap isolation). Each entry corresponds
    /// to one surface in the zone. Telemetry is captured at the state *after*
    /// the most recent `step()` or `step_with_coupling()` call.
    ///
    /// Telemetry per surface:
    /// - `area_m2` — surface area [m²]
    /// - `surface_type` — SurfaceType enum (Window, Wall, Roof, etc.)
    /// - `tilt_deg` — tilt from horizontal [degrees]
    /// - `azimuth_deg` — azimuth from north [degrees]
    /// - `q_flux_Wm2` — heat flux through surface [W/m²] (positive = into zone)
    /// - `r_total_m2K_W` — total surface resistance [m²K/W]
    /// - `c_mass_Jm2K` — surface thermal mass [J/m²K]
    /// - `solar_fraction` — fraction of horizontal GHI that reaches this surface
    /// - `sky_view_factor` — ISO 13790 sky view factor (0–1)
    pub fn per_surface_telemetry(&self) -> Vec<SurfaceTelemetry> {
        self.surfaces
            .iter()
            .map(|s| SurfaceTelemetry {
                area_m2: s.area_m2,
                surface_type: s.surface_type,
                tilt_deg: s._tilt_deg,
                azimuth_deg: s._azimuth_deg,
                q_flux_Wm2: s.gauge.q_flux(),
                r_total_m2K_W: s.gauge.r_total_for_test(),
                c_mass_Jm2K: s.gauge.c_mass_for_test(),
                solar_fraction: s.surface_type.solar_fraction(),
                sky_view_factor: s.sky_view_factor(),
                t_surface_deg: s.t_surface(),
                solar_absorbed_w: s.last_solar_absorbed_w,
            })
            .collect()
    }

    /// Issue #3911 — Infiltration conductance [W/K] from the most recent step.
    ///
    /// Computed as: ρ_air · c_p,air · (ACH / 3600) · V_zone
    pub fn infiltration_conductance_WK(&self, infiltration_ach: f64) -> f64 {
        air_constants::RHO_AIR
            * air_constants::CP_AIR
            * (infiltration_ach / 3600.0)
            * self.zone_volume
    }

    /// Issue #3911 — Total surface-to-air conductance [W/K] from the most recent step.
    ///
    /// Σ_i (A_i / R_total,i) for all non-inter-zone surfaces.
    /// This is the sum of window, wall, roof, and floor conductances that
    /// couple the zone air to the exterior / sky / ground.
    fn surface_to_air_conductance(&self) -> f64 {
        self.surfaces
            .iter()
            .filter(|s| !s.surface_type.is_inter_zone())
            .filter(|s| s.gauge.r_total_for_test() > 0.0)
            .map(|s| s.area_m2 / s.gauge.r_total_for_test())
            .sum()
    }

    /// Issue #3928 — Compute interior surface-to-air conductance [W/K] from surface geometry.
    ///
    /// h_tr_is = Σ A_surface × h_tr_is_coeff(tilt)
    ///
    /// h_tr_is_coeff varies with surface tilt:
    /// - Horizontal (tilt ≈ 0° or 180°): 2.3 W/m²K
    /// - Vertical (tilt ≈ 90°): 8.3 W/m²K
    /// - Intermediate tilts: linear interpolation using sin(tilt)
    ///
    /// This activates the solar lag correction in the zone energy balance.
    /// The solar lag correction is gated on h_tr_is > 0.0 && term_rest_1 > 0.0.
    ///
    /// NOTE: The tilt-linear interpolation formula (sin-based) needs validation
    /// against the actual 5R1C reference. This implementation follows the
    /// issue #3928 specification.
    pub fn compute_h_tr_is(&self) -> f64 {
        self.surfaces
            .iter()
            .filter(|s| surface_is_absorbing(&s.surface_type))
            .map(|s| s.area_m2 * h_tr_is_coeff(s._tilt_deg))
            .sum()
    }

    /// Issue #3918 follow-up (daytime solar gap): step the per-surface
    /// interior absorbed-gain network and return the surface→air heat flow.
    ///
    /// Each eligible surface ([`surface_is_absorbing`]: opaque envelope,
    /// window, internal mass — NOT inter-zone partitions or the fixed-
    /// temperature Ground boundary) carries an interior surface node T_s,i
    /// coupled to the zone air by h_tr_is,i = A_i · h_tr_is_coeff(tilt_i)
    /// [W/K] with capacitance C_i = A_i · c_mass,i [J/K] taken from the
    /// surface's own 1D gauge. This is the gauge analog of 5R1C's surface
    /// star node θ_s + mass node θ_m pair, resolved per surface.
    ///
    /// The absorbed pools mirror the 5R1C split (step_5r1c.rs:96-130):
    /// - `surface_pool_w` (5R1C phi_st): distributed conductance-weighted
    ///   (h_tr_is,i/Σ), giving a uniform steady-state ΔT_s across surfaces —
    ///   the ISO 13790 isothermal surface-node behavior.
    /// - `mass_pool_w` (5R1C phi_m): distributed capacitance-weighted
    ///   (C_i/ΣC), placing beam-like gains on the massive surfaces (the
    ///   floor slab in the ASHRAE 140 cases) where the beam physically lands
    ///   and releasing it on the mass time constant.
    ///
    /// Each node follows the exact exponential solution of
    /// `C_i·dT_s/dt = q_abs,i − h_tr_is,i·(T_s,i − T_air)`, and the emitted
    /// energy is taken from the energy balance
    /// `∫h·(T_s−T_air)dt = q_abs,i·dt − C_i·ΔT_s,i`, so per-surface and
    /// zone-level conservation hold by construction. With zero pools the
    /// surfaces relax toward T_air, releasing stored solar at night.
    ///
    /// Returns `(φ_st, Σ h_tr_is,i)`: the timestep-average surface→air heat
    /// flow [W] and the network's total air-side conductance [W/K]. The
    /// caller must thread Σ h_tr_is,i into the air-node denominator — it is
    /// the implicit-conductance counterpart of `φ_st` (frozen-flux
    /// linearization, same pattern as `h_env` for the envelope fluxes).
    /// Also updates `previous_T_surface` to the h_tr_is-weighted mean T_s.
    /// Fallback: when no eligible surface can couple (Σ h_tr_is,i ≤ 0) the
    /// pools are delivered instantly so the gains are never silently
    /// dropped.
    fn step_interior_surface_network(
        &mut self,
        dt_seconds: f64,
        surface_pool_w: f64,
        mass_pool_w: f64,
        solar_surface_pool_w: f64,
        solar_mass_pool_w: f64,
        h_tr_is_zone: f64,
    ) -> (f64, f64) {
        let t_air = self.T_air;

        // First pass: geometry sums. Per-surface air-side conductance is the
        // ZONE h_tr_is threaded from the thermal model (the same ISO 13790
        // star-node value the 5R1C reference solver uses, e.g. 3.45·A_tot),
        // distributed by area share. The Issue #3928 tilt coefficients
        // (2.3/8.3 W/m²K) are only a geometric fallback for callers that do
        // not thread h_tr_is — using them unconditionally made the interior
        // film ~2.2× stronger than the 5R1C star node and over-damped the
        // diurnal swing (Issue #3918 follow-up).
        let mut sum_area = 0.0;
        let mut sum_c = 0.0;
        for s in &self.surfaces {
            if !surface_is_absorbing(&s.surface_type) {
                continue;
            }
            sum_area += s.area_m2;
            sum_c += s.area_m2 * s.gauge.c_mass_for_test();
        }
        let per_surface_h = |s: &SurfaceGaugeSolver| -> f64 {
            if h_tr_is_zone > 0.0 && sum_area > 0.0 {
                h_tr_is_zone * s.area_m2 / sum_area
            } else {
                s.area_m2 * h_tr_is_coeff(s._tilt_deg)
            }
        };
        let mut sum_h = 0.0;
        for s in &self.surfaces {
            if surface_is_absorbing(&s.surface_type) {
                sum_h += per_surface_h(s);
            }
        }
        if sum_h <= 0.0 {
            // No interior surface can couple to the air node: deliver the
            // absorbed gains instantly (energy must not vanish).
            return (surface_pool_w + mass_pool_w, 0.0);
        }

        let mut emitted_j = 0.0;
        let mut h_weighted_ts = 0.0;
        for s in &mut self.surfaces {
            if !surface_is_absorbing(&s.surface_type) {
                continue;
            }
            let h_i = per_surface_h(s);
            let c_i = s.area_m2 * s.gauge.c_mass_for_test();
            let w_surface = h_i / sum_h;
            let w_mass = if sum_c > 0.0 {
                c_i / sum_c
            } else {
                h_i / sum_h
            };
            let q_abs_i = surface_pool_w * w_surface + mass_pool_w * w_mass;
            // Issue #3983 — solar-only attribution for the per-surface
            // absorption telemetry (Σ_i + Φ_sol·to_air = Φ_sol).
            s.last_solar_absorbed_w =
                solar_surface_pool_w * w_surface + solar_mass_pool_w * w_mass;
            let steady = t_air + q_abs_i / h_i;
            let new_ts = if c_i > 0.0 && dt_seconds > 0.0 {
                let tau_i = c_i / h_i;
                steady + (s.T_surface - steady) * (-dt_seconds / tau_i).exp()
            } else {
                steady
            };
            // Energy-exact emission over the step: absorbed − stored.
            emitted_j += q_abs_i * dt_seconds - c_i * (new_ts - s.T_surface);
            s.T_surface = new_ts;
            h_weighted_ts += h_i * new_ts;
        }
        self.previous_T_surface = h_weighted_ts / sum_h;

        if dt_seconds > 0.0 {
            (emitted_j / dt_seconds, sum_h)
        } else {
            (surface_pool_w + mass_pool_w, sum_h)
        }
    }

    /// Issue #3911 — Effective air-node time constant [seconds] from the most recent step.
    ///
    /// τ_air = C_air / (h_inf + h_surface_total + h_inter_zone)
    /// where C_air = ρ_air · c_p,air · V_zone.
    ///
    /// Matches the 5R1C τ_air = C_air / den_true formula, where den_true
    /// includes window conductance (H_tr,w), opaque surface conductance (H_tr,1),
    /// infiltration (H_ve), and inter-zone coupling.
    ///
    /// At τ_air ≈ 1000 s and dt = 3600 s: dt/τ_air ≈ 3.6 (5R1C regime).
    /// A larger τ_air means the air node responds more slowly.
    pub fn effective_time_constant_s(&self, infiltration_ach: f64) -> f64 {
        let h_inf = self.infiltration_conductance_WK(infiltration_ach);
        let h_surface = self.surface_to_air_conductance();
        let h_inter_zone: f64 = self.couplings.iter().map(|c| c.conductance).sum();
        let C_air = air_constants::RHO_AIR * air_constants::CP_AIR * self.zone_volume;
        C_air / (h_inf + h_surface + h_inter_zone)
    }

    /// Issue #3911 — Effective dt/τ_air ratio.
    ///
    /// At dt/τ_air ≈ 3.6, implicit Euler sub-stepping (N=3) still under-equilibrates
    /// the air node vs the analytical solution. This ratio is the key diagnostic
    /// for the "family-level lumped-mass damping" hypothesis.
    pub fn dt_over_tau(&self, dt_seconds: f64, infiltration_ach: f64) -> f64 {
        dt_seconds / self.effective_time_constant_s(infiltration_ach)
    }
}

/// Per-surface telemetry returned by [`GaugeZoneSolver::per_surface_telemetry()`].
#[derive(Debug, Clone)]
pub struct SurfaceTelemetry {
    /// Surface area [m²]
    pub area_m2: f64,
    /// Surface type (Window, Wall, Roof, etc.)
    pub surface_type: SurfaceType,
    /// Surface tilt from horizontal [degrees]
    pub tilt_deg: f64,
    /// Surface azimuth from north [degrees]
    pub azimuth_deg: f64,
    /// Heat flux through surface [W/m²] (positive = into zone)
    pub q_flux_Wm2: f64,
    /// Issue #3983 — solar absorbed by this surface's interior node during
    /// the most recent step [W]. Solar-only attribution; the zone-level
    /// closure Σ_i + Φ_sol·to_air = Φ_sol must hold exactly.
    pub solar_absorbed_w: f64,
    /// Total surface resistance [m²K/W]
    pub r_total_m2K_W: f64,
    /// Surface thermal mass [J/m²K]
    pub c_mass_Jm2K: f64,
    /// Fraction of horizontal GHI that reaches this surface
    pub solar_fraction: f64,
    /// ISO 13790 sky view factor (0–1)
    pub sky_view_factor: f64,
    /// Issue #3918 follow-up: interior surface-node temperature T_s [°C]
    /// from the interior absorbed-gain network (per-surface solar-lag state).
    pub t_surface_deg: f64,
}

/// Multi-zone gauge solver for N-zone thermal coupling.
///
/// Manages multiple zones with inter-zone coupling defined over
/// an adjacency graph. Each zone is solved independently but
/// coupling is handled through a global system solve.
///
/// # Example
///
/// ```ignore
/// let mut multi_zone = MultiZoneGaugeSolver::new();
/// multi_zone.add_zone(0, 48.0, 2.7); // Zone 0: 48m² floor, 2.7m height
/// multi_zone.add_zone(1, 36.0, 2.7); // Zone 1: 36m² floor, 2.7m height
/// multi_zone.add_zone_coupling(0, 1, 10.0, 0.5); // Shared wall: 10m², R=0.5
/// multi_zone.initialize().unwrap();
/// ```
///
/// # Clone semantics (Issue #3729)
///
/// `MultiZoneGaugeSolver` does not derive `Clone`. Per-zone state is
/// reset by [`GaugeZoneSolver::clone`] (the topology is preserved,
/// `T_air` is reset to the `new_with_id` default, and `initialized` is
/// reset to `false`). The aggregate `initialized` flag here is also
/// reset to `false`, mirroring the slot-reset contract applied to
/// `HybridThermalModel::conduction_solver` (ARCHITECTURE.md Issue #2539).
pub struct MultiZoneGaugeSolver {
    /// All zones in the system
    zones: Vec<GaugeZoneSolver>,
    /// Zone IDs for quick lookup
    zone_ids: Vec<usize>,
    /// Number of zones
    num_zones: usize,
    /// Solver initialized flag
    initialized: bool,
}

// Issue #3729 — hand-rolled `Clone` for `MultiZoneGaugeSolver`.
//
// Per-zone state (including the `T_air` / `initialized` reset) is handled
// by `GaugeZoneSolver::clone`; this impl additionally resets the
// aggregate `initialized` flag. Topology (zone list, IDs, count) is
// preserved verbatim. Two clones are fully independent: solving one does
// not perturb the other's per-zone `T_air`.
impl Clone for MultiZoneGaugeSolver {
    fn clone(&self) -> Self {
        Self {
            zones: self.zones.clone(),
            zone_ids: self.zone_ids.clone(),
            num_zones: self.num_zones,
            initialized: false, // RESET — see struct doc-comment.
        }
    }
}

impl MultiZoneGaugeSolver {
    /// Create a new empty multi-zone solver.
    pub fn new() -> Self {
        Self {
            zones: Vec::new(),
            zone_ids: Vec::new(),
            num_zones: 0,
            initialized: false,
        }
    }

    /// Add a zone to the system.
    ///
    /// # Arguments
    /// * `zone_id` - Unique identifier for this zone
    /// * `floor_area` - Floor area in m²
    /// * `ceiling_height` - Ceiling height in m
    pub fn add_zone(&mut self, zone_id: usize, floor_area: f64, ceiling_height: f64) {
        self.zones.push(GaugeZoneSolver::new_with_id(
            zone_id,
            floor_area,
            ceiling_height,
        ));
        self.zone_ids.push(zone_id);
        self.num_zones = self.zones.len();
    }

    /// Add a surface to a specific zone.
    pub fn add_surface_to_zone(
        &mut self,
        zone_id: usize,
        gauge: GaugeSolver,
        area_m2: f64,
        surface_type: SurfaceType,
        azimuth_deg: f64,
        tilt_deg: f64,
    ) -> Result<(), SolverError> {
        let zone = self
            .zones
            .iter_mut()
            .find(|z| z.zone_id == zone_id)
            .ok_or_else(|| SolverError::InvalidConfig(format!("Zone {} not found", zone_id)))?;
        zone.add_surface(gauge, area_m2, surface_type, azimuth_deg, tilt_deg);
        Ok(())
    }

    /// Add an opaque surface to a specific zone.
    pub fn add_opaque_surface_to_zone(
        &mut self,
        zone_id: usize,
        wall: &WallSpec,
        area_m2: f64,
        surface_type: SurfaceType,
        azimuth_deg: f64,
        tilt_deg: f64,
    ) -> Result<(), SolverError> {
        let zone = self
            .zones
            .iter_mut()
            .find(|z| z.zone_id == zone_id)
            .ok_or_else(|| SolverError::InvalidConfig(format!("Zone {} not found", zone_id)))?;
        zone.add_opaque_surface(wall, area_m2, surface_type, azimuth_deg, tilt_deg)
    }

    /// Add inter-zone coupling between two zones.
    ///
    /// This adds a symmetric coupling where heat flows between zones
    /// based on the temperature difference and shared boundary conductance.
    ///
    /// # Arguments
    /// * `zone_id_a` - First zone ID
    /// * `zone_id_b` - Second zone ID
    /// * `shared_area_m2` - Area of shared boundary (m²)
    /// * `R_value` - Thermal resistance of shared boundary (m²·K/W)
    pub fn add_zone_coupling(
        &mut self,
        zone_id_a: usize,
        zone_id_b: usize,
        shared_area_m2: f64,
        R_value: f64,
    ) -> Result<(), SolverError> {
        // Add coupling to zone A
        if let Some(zone) = self.zones.iter_mut().find(|z| z.zone_id == zone_id_a) {
            zone.add_zone_coupling(zone_id_b, shared_area_m2, R_value)?;
        }
        // Add symmetric coupling to zone B
        if let Some(zone) = self.zones.iter_mut().find(|z| z.zone_id == zone_id_b) {
            zone.add_zone_coupling(zone_id_a, shared_area_m2, R_value)?;
        }
        Ok(())
    }

    /// Initialize all zones.
    pub fn initialize(&mut self) -> Result<(), SolverError> {
        for zone in &mut self.zones {
            zone.initialize()?;
        }
        self.initialized = true;
        Ok(())
    }

    /// Check if solver is initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized && !self.zones.is_empty()
    }

    /// Get the number of zones.
    pub fn num_zones(&self) -> usize {
        self.num_zones
    }

    /// Get zone by ID.
    pub fn get_zone(&self, zone_id: usize) -> Option<&GaugeZoneSolver> {
        self.zones.iter().find(|z| z.zone_id == zone_id)
    }

    /// Get mutable zone by ID.
    pub fn get_zone_mut(&mut self, zone_id: usize) -> Option<&mut GaugeZoneSolver> {
        self.zones.iter_mut().find(|z| z.zone_id == zone_id)
    }

    /// Get all zone IDs.
    pub fn zone_ids(&self) -> &[usize] {
        &self.zone_ids
    }

    /// Issue #3297 — per-zone interior surface temperature proxy: for
    /// each zone, the area-weighted mean of that zone's surfaces'
    /// interior temperatures (see [`GaugeZoneSolver::mean_interior_surface_temperature`]).
    /// Length matches `self.num_zones()`. Read-only telemetry for the
    /// dispatch's 5R1C-compatible mass-state proxy.
    pub fn zone_interior_temperatures(&self) -> Vec<f64> {
        self.zones
            .iter()
            .map(|zone| zone.mean_interior_surface_temperature())
            .collect()
    }

    /// Step all zones with inter-zone coupling.
    ///
    /// This performs a single timestep for all zones, computing
    /// inter-zone heat transfer based on current temperatures.
    ///
    /// # Arguments
    /// * `dt_seconds` - Timestep duration in seconds
    /// * `boundary_conditions` - Map of zone_id -> boundary conditions
    ///
    /// # Returns
    /// Map of zone_id -> energy in kWh
    pub fn step(
        &mut self,
        dt_seconds: f64,
        boundary_conditions: &HashMap<usize, ZoneBoundaryConditions>,
    ) -> Result<HashMap<usize, f64>, SolverError> {
        if !self.is_initialized() {
            return Err(SolverError::InvalidConfig(
                "GaugeZoneSolver not initialized".to_string(),
            ));
        }

        // First: collect all zone temperatures for coupling calculations
        let zone_temps: HashMap<usize, f64> =
            self.zones.iter().map(|z| (z.zone_id, z.T_air)).collect();

        // Compute coupling vectors for each zone using collected temperatures
        let mut coupling_vectors: HashMap<usize, HashMap<usize, f64>> = HashMap::new();
        for zone in &self.zones {
            let adjacent_temps: HashMap<usize, Temperature> = zone
                .couplings
                .iter()
                .filter_map(|c| {
                    zone_temps
                        .get(&c.adjacent_zone_id)
                        .map(|&t| (c.adjacent_zone_id, Temperature::from_value(t)))
                })
                .collect();
            coupling_vectors.insert(
                zone.zone_id,
                zone.compute_zone_coupling_vector(&adjacent_temps),
            );
        }

        // Second pass: step each zone with its coupling contributions
        let mut results = HashMap::new();
        for zone in &mut self.zones {
            let zone_id = zone.zone_id;
            let bc = boundary_conditions
                .get(&zone_id)
                .cloned()
                .unwrap_or_default();

            // Add coupling contributions to boundary conditions
            // NOTE: inter-zone coupling is applied via adjacent_temps in step_with_coupling.
            // Zero bc.inter_zone_heat to avoid double-counting: the coupling vector
            // contribution is already accounted for through the adjacent_temps path.
            let mut bc_with_coupling = bc.clone();
            bc_with_coupling.inter_zone_heat = 0.0;

            // Get adjacent temperatures for inter-zone surfaces
            let adjacent_temps: HashMap<usize, Temperature> = zone
                .couplings
                .iter()
                .filter_map(|c| {
                    zone_temps
                        .get(&c.adjacent_zone_id)
                        .map(|&t| (c.adjacent_zone_id, Temperature::from_value(t)))
                })
                .collect();

            let energy = zone.step_with_coupling(dt_seconds, &bc_with_coupling, &adjacent_temps)?;
            results.insert(zone_id, energy);
        }

        Ok(results)
    }
}

impl Default for MultiZoneGaugeSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ============ Tests ============

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::wall_spec::WallSpec;

    /// Conductive stub wall — NOT the real ASHRAE 140 Case 600 construction
    /// (Issue #3893). R = 0.09 m²K/W (0.09 m / 1.0 W/mK), roughly 27× more
    /// conductive than the real Case 600 wood-frame wall (R ≈ 2.4, per
    /// `docs/contributing/ADDING_A_NEW_ASHRAE_140_CASE.md` reference data).
    ///
    /// **Stability footgun:** PR #3890 made the surface flux explicit in the
    /// multi-zone T_air update (mirroring the single-zone #3878 fix), so
    /// explicit-Euler stability requires `dt·H_surface < 2·C_air`. At hourly
    /// dt this stub's H ≈ 373 W/K gives `dt·H ≈ 1.34 MJ > 2·C_air ≈ 0.31 MJ`
    /// — a free-floating settle at hourly dt through the explicit-surface-flux
    /// formula DIVERGES to non-finite (this cost PR #3890's first tracer test
    /// a RED-cycle debug). Use `insulated_wall()` (R ≈ 2.4) for any test that
    /// settles at hourly dt; see the #3817 stability analysis in `step`.
    ///
    /// Kept (renamed from `case600_wall`, Issue #3893) for the existing
    /// short-step / steady-state tests calibrated against R = 0.09 — none of
    /// them drive an hourly-dt free-float settle (audited in #3893).
    fn conductive_stub_wall() -> WallSpec {
        WallSpec::single_layer("LightWeight", 0.09, 1.0, 50.0, 50.0)
    }

    #[test]
    fn test_zone_air_capacitance() {
        // Case 600 zone: 6m x 8m x 2.7m = 129.6 m³
        let C = air_constants::zone_air_capacitance(48.0, 2.7);
        let expected = 1.2 * 48.0 * 2.7 * 1006.0; // ~156,000 J/K
        assert!((C - expected).abs() < 1.0);
    }

    // Issue #3928 — Test that compute_h_tr_is() exercises the tilt-based coefficient
    // branches (horizontal ≈ 2.3, vertical ≈ 8.3, intermediate = sin-interpolated).
    // This test covers the new branches introduced by PR #3931 in the gauge solver's
    // solar lag correction path. Without this test, compute_h_tr_is() is only
    // invoked from step_dispatcher.rs (not exercised by lib tests), causing the
    // Code Coverage Gate to fail on the conduction_zone ratchet.
    #[test]
    fn test_compute_h_tr_is_tilt_branches() {
        let mut mz = MultiZoneGaugeSolver::new();
        mz.add_zone(0, 48.0, 2.7);

        // Use conductive_stub_wall for all surfaces; tilt drives the coefficient.
        let wall = conductive_stub_wall();

        // Roof (tilt ≈ 0°, horizontal): coeff = 2.3
        mz.add_opaque_surface_to_zone(0, &wall, 48.0, SurfaceType::Roof, 0.0, 0.0)
            .unwrap();
        // North wall (tilt = 90°, vertical): coeff = 8.3
        mz.add_opaque_surface_to_zone(0, &wall, 12.0, SurfaceType::Wall, 180.0, 90.0)
            .unwrap();
        // South wall (tilt = 90°, vertical): coeff = 8.3
        mz.add_opaque_surface_to_zone(0, &wall, 12.0, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        // Intermediate tilt (45°): coeff = 2.3 + sin(45°)*(8.3-2.3) ≈ 6.54
        mz.add_opaque_surface_to_zone(0, &wall, 10.0, SurfaceType::Wall, 45.0, 45.0)
            .unwrap();
        // Floor (tilt = 180°, horizontal): coeff = 2.3
        mz.add_opaque_surface_to_zone(0, &wall, 48.0, SurfaceType::Floor, 0.0, 180.0)
            .unwrap();

        mz.initialize().unwrap();

        let zone = mz.get_zone(0).unwrap();
        let h_tr_is = zone.compute_h_tr_is();

        // Expected: 48*2.3 + 12*8.3 + 12*8.3 + 10*6.54 + 48*2.3
        // = 110.4 + 99.6 + 99.6 + 65.4 + 110.4 = 485.4
        let expected = 48.0 * 2.3
            + 12.0 * 8.3
            + 12.0 * 8.3
            + 10.0 * (2.3 + 45.0_f64.to_radians().sin() * 6.0)
            + 48.0 * 2.3;
        assert!(
            (h_tr_is - expected).abs() < 1e-9,
            "h_tr_is = {h_tr_is}, expected {expected}"
        );
    }

    #[test]
    fn test_steady_state_no_solar() {
        let mut zone = GaugeZoneSolver::new(48.0, 2.7); // Case 600 floor area

        // Add 4 walls (simplified - each gets full wall R)
        let wall = conductive_stub_wall();

        // Case 600 dimensions: 8m x 6m x 2.7m
        // Wall heights are 2.7m
        let south_area = 8.0 * 2.7; // 21.6 m²
        let east_area = 6.0 * 2.7; // 16.2 m²

        // South wall (faces equator, gets most solar)
        zone.add_opaque_surface(&wall, south_area, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        // North wall
        zone.add_opaque_surface(&wall, south_area, SurfaceType::Wall, 180.0, 90.0)
            .unwrap();
        // East wall
        zone.add_opaque_surface(&wall, east_area, SurfaceType::Wall, 90.0, 90.0)
            .unwrap();
        // West wall
        zone.add_opaque_surface(&wall, east_area, SurfaceType::Wall, -90.0, 90.0)
            .unwrap();

        zone.initialize().unwrap();

        // Steady state: inside 20°C, outside 10°C, no solar
        let T_int = Temperature::from_value(20.0);
        let T_ext = Temperature::from_value(10.0);

        let flux = zone.steady_state_flux(T_int, T_ext).unwrap();
        // Net flux should be negative (heat leaving zone)
        assert!(
            flux.to_value() < 0.0,
            "Heat should flow from warm interior to cold exterior"
        );
    }

    #[test]
    fn test_multi_surface_aggregation() {
        // Test that adding multiple surfaces properly aggregates heat flows
        let mut zone = GaugeZoneSolver::new(48.0, 2.7);

        let wall = conductive_stub_wall();

        // Add 2 identical walls
        let wall_area = 10.0; // 10 m² each
        zone.add_opaque_surface(&wall, wall_area, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        zone.add_opaque_surface(&wall, wall_area, SurfaceType::Wall, 180.0, 90.0)
            .unwrap();

        zone.initialize().unwrap();

        // With T_int=20, T_ext=10, each wall has heat leaving
        // Total should be roughly 2x a single wall
        let T_int = Temperature::from_value(20.0);
        let T_ext = Temperature::from_value(10.0);

        // Get flux from zone with 2 walls
        let flux_2walls = zone.steady_state_flux(T_int, T_ext).unwrap();

        // Now add a 3rd identical wall
        zone.add_opaque_surface(&wall, wall_area, SurfaceType::Wall, 90.0, 90.0)
            .unwrap();
        // Re-initialize since we added a surface
        zone.initialize().unwrap();

        let flux_3walls = zone.steady_state_flux(T_int, T_ext).unwrap();

        // More walls = more heat loss = more negative flux
        // (flux is W/m² of floor area, so with more wall area conducting,
        // the net flux per m² should be more negative)
        assert!(
            flux_3walls.to_value() < flux_2walls.to_value(),
            "More wall area should result in more heat loss"
        );
    }

    #[test]
    fn test_step_updates_temperature() {
        let mut zone = GaugeZoneSolver::new(48.0, 2.7);

        let wall = conductive_stub_wall();
        zone.add_opaque_surface(&wall, 48.0, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();

        zone.initialize().unwrap();

        let T_before = zone.T_air();
        assert!((T_before.to_value() - 20.0).abs() < 0.1);

        // Step with large exterior temp difference
        let T_ext = Temperature::from_value(-10.0); // Cold outside
        let h_ext = HeatTransferCoefficient::from_value(25.0);

        // One hour timestep
        let energy = zone
            .step(
                0,      // timestep
                3600.0, // dt = 1 hour
                T_ext, h_ext, 0.0, // solar_irradiance_wm2
                0.0, // solar_distribution_to_air
                0.0, // Q_internal_w
                0.0, // Q_infiltration_w
                0.0, // t_sky
                0.0, // h_rad_sky
                0.0, // ventilation_ach (Issue #3904: 0 = no night vent)
                // Issue #3918: solar lag parameters (0.0 = lag disabled)
                0.0, // h_tr_3
                0.0, // cm
                0.0, // h_tr_is
                0.0, // term_rest_1
                0.0, // convective_fraction
                0.0, // solar_beam_to_mass_fraction
            )
            .unwrap();

        let T_after = zone.T_air();

        // Zone should have cooled (T_after < T_before)
        assert!(
            T_after.to_value() < T_before.to_value(),
            "Zone should cool when exterior is cold"
        );

        // Energy should be positive (heating required)
        assert!(energy > 0.0, "Heating energy should be positive");
    }

    #[test]
    fn test_zone_coupling_basic() {
        // Test that inter-zone coupling is properly stored
        let mut zone_a = GaugeZoneSolver::new(48.0, 2.7);
        let _zone_b = GaugeZoneSolver::new(36.0, 2.7);

        // Add coupling between zones (10m² shared wall, R=0.5 m²K/W)
        zone_a.add_zone_coupling(1, 10.0, 0.5).unwrap();

        assert_eq!(zone_a.zone_id(), 0);
        assert_eq!(zone_a.inter_zone_conductance(1), 20.0); // g = 10/0.5 = 20 W/K

        assert_eq!(zone_a.couplings().len(), 1);
        assert_eq!(zone_a.couplings()[0].adjacent_zone_id, 1);
        assert_eq!(zone_a.couplings()[0].conductance, 20.0);
    }

    #[test]
    fn test_multi_zone_solver_two_zones() {
        let mut multi_zone = MultiZoneGaugeSolver::new();

        // Zone 0: 48m² floor, 2.7m height (Zone A)
        multi_zone.add_zone(0, 48.0, 2.7);
        // Zone 1: 36m² floor, 2.7m height (Zone B)
        multi_zone.add_zone(1, 36.0, 2.7);

        let wall = conductive_stub_wall();

        // Add walls to Zone 0 (exterior walls only)
        multi_zone
            .add_opaque_surface_to_zone(0, &wall, 21.6, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        multi_zone
            .add_opaque_surface_to_zone(0, &wall, 16.2, SurfaceType::Wall, 90.0, 90.0)
            .unwrap();
        multi_zone
            .add_opaque_surface_to_zone(0, &wall, 16.2, SurfaceType::Wall, -90.0, 90.0)
            .unwrap();

        // Add walls to Zone 1 (exterior walls only)
        multi_zone
            .add_opaque_surface_to_zone(1, &wall, 18.0, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        multi_zone
            .add_opaque_surface_to_zone(1, &wall, 10.8, SurfaceType::Wall, 90.0, 90.0)
            .unwrap();

        // Add inter-zone coupling (shared interior wall: 10m², R=0.5)
        multi_zone.add_zone_coupling(0, 1, 10.0, 0.5).unwrap();

        multi_zone.initialize().unwrap();

        assert_eq!(multi_zone.num_zones(), 2);
        assert!(multi_zone.is_initialized());

        // Step both zones
        let mut bc = HashMap::new();
        bc.insert(
            0,
            ZoneBoundaryConditions::new(
                Temperature::from_value(5.0),
                HeatTransferCoefficient::from_value(25.0),
                0.0,
                0.0, // solar_distribution_to_air
            ),
        );
        bc.insert(
            1,
            ZoneBoundaryConditions::new(
                Temperature::from_value(5.0),
                HeatTransferCoefficient::from_value(25.0),
                0.0,
                0.0, // solar_distribution_to_air
            ),
        );

        let results = multi_zone.step(3600.0, &bc).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_inter_zone_heat_transfer() {
        // Test that heat flows between zones based on temperature difference
        let mut zone_a = GaugeZoneSolver::new(48.0, 2.7);
        let mut zone_b = GaugeZoneSolver::new(48.0, 2.7);

        let wall = conductive_stub_wall();
        zone_a
            .add_opaque_surface(&wall, 48.0, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        zone_b
            .add_opaque_surface(&wall, 48.0, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();

        // Add coupling: 20 W/K conductance
        zone_a.add_zone_coupling(1, 10.0, 0.5).unwrap();
        zone_b.add_zone_coupling(0, 10.0, 0.5).unwrap();

        zone_a.initialize().unwrap();
        zone_b.initialize().unwrap();

        // Set different initial temperatures
        zone_a.T_air = 25.0; // Warmer zone
        zone_b.T_air = 15.0; // Cooler zone

        // Compute coupling vector for zone A
        let mut adjacent_temps = HashMap::new();
        adjacent_temps.insert(1, Temperature::from_value(15.0));

        let coupling_vec = zone_a.compute_zone_coupling_vector(&adjacent_temps);

        // Heat should flow from A to B (positive in our convention means flowing to adjacent)
        // Q = g * (T_a - T_b) = 20 * (25 - 15) = 200 W
        assert!(*coupling_vec.get(&1).unwrap() > 0.0);
    }

    #[test]
    fn test_backward_compatibility_single_zone() {
        // Ensure existing single-zone usage still works
        let mut zone = GaugeZoneSolver::new(48.0, 2.7);
        let wall = conductive_stub_wall();

        zone.add_opaque_surface(&wall, 21.6, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        zone.add_opaque_surface(&wall, 16.2, SurfaceType::Wall, 90.0, 90.0)
            .unwrap();

        zone.initialize().unwrap();

        let T_int = Temperature::from_value(20.0);
        let T_ext = Temperature::from_value(10.0);

        let flux = zone.steady_state_flux(T_int, T_ext).unwrap();
        assert!(flux.to_value() < 0.0); // Heat flows out
    }

    /// Issue #3297 — `surface_interior_temperatures()` must return one
    /// entry per surface, each equal to the interior boundary
    /// temperature that surface's gauge integrated against at the most
    /// recent step (20 °C initialization before any step).
    #[test]
    fn test_surface_interior_temperatures_per_surface() {
        let mut zone = GaugeZoneSolver::new(48.0, 2.7);
        let wall = conductive_stub_wall();

        zone.add_opaque_surface(&wall, 21.6, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        zone.add_opaque_surface(&wall, 16.2, SurfaceType::Roof, 90.0, 0.0)
            .unwrap();
        zone.initialize().unwrap();

        let temps = zone.surface_interior_temperatures();
        assert_eq!(temps.len(), 2, "one entry per surface");
        // Before any step, the interior-most state is the 20 °C default.
        assert!(
            temps.iter().all(|&t| (t - 20.0).abs() < 1e-12),
            "unstepped surfaces report the 20 °C initial interior state, got {temps:?}"
        );

        // After a step with T_air = 25 °C, every surface integrated
        // against T_air = 25 °C as its interior boundary.
        zone.set_T_air(25.0);
        zone.step(
            0,
            3600.0,
            Temperature::from_value(5.0),
            HeatTransferCoefficient::from_value(25.0),
            0.0, // solar_irradiance_wm2
            0.0, // solar_distribution_to_air
            0.0, // Q_internal_w
            0.0, // Q_infiltration_w
            0.0, // t_sky
            0.0, // h_rad_sky
            0.0, // ventilation_ach (Issue #3904: 0 = no night vent)
            // Issue #3918: solar lag parameters (0.0 = lag disabled)
            0.0, // h_tr_3
            0.0, // cm
            0.0, // h_tr_is
            0.0, // term_rest_1
            0.0, // convective_fraction
            0.0, // solar_beam_to_mass_fraction
        )
        .unwrap();
        let temps = zone.surface_interior_temperatures();
        assert!(
            temps.iter().all(|&t| (t - 25.0).abs() < 1e-12),
            "stepped surfaces report the step's interior boundary temperature, got {temps:?}"
        );
    }

    /// Issue #3297 — `mean_interior_surface_temperature()` is the
    /// area-weighted mean, and `MultiZoneGaugeSolver::zone_interior_temperatures()`
    /// returns one area-weighted entry per zone.
    #[test]
    fn test_zone_interior_temperatures_area_weighted() {
        let mut zone = GaugeZoneSolver::new(48.0, 2.7);
        let wall = conductive_stub_wall();

        // Two surfaces with areas 30 m² and 10 m²: the weighted mean of
        // (20, 20) is 20 before any step.
        zone.add_opaque_surface(&wall, 30.0, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        zone.add_opaque_surface(&wall, 10.0, SurfaceType::Roof, 90.0, 0.0)
            .unwrap();
        zone.initialize().unwrap();
        assert!((zone.mean_interior_surface_temperature() - 20.0).abs() < 1e-12);

        let mut mz = MultiZoneGaugeSolver::new();
        mz.add_zone(0, 48.0, 2.7);
        mz.add_zone(1, 32.0, 2.7);
        mz.add_opaque_surface_to_zone(0, &wall, 21.6, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        mz.add_opaque_surface_to_zone(1, &wall, 12.0, SurfaceType::Wall, 180.0, 90.0)
            .unwrap();
        mz.initialize().unwrap();

        let per_zone = mz.zone_interior_temperatures();
        assert_eq!(per_zone.len(), mz.num_zones());
        assert!(
            per_zone.iter().all(|&t| (t - 20.0).abs() < 1e-12),
            "unstepped zones report the 20 °C initial interior state, got {per_zone:?}"
        );
    }

    /// Issue #3297 — `zone_interior_temperatures()` must return per-zone
    /// values that reflect each zone's individual post-step state. The
    /// dispatch's `write_gauge_mass_state_proxy` consumes this accessor
    /// once per zone, so per-zone differentiation is the read-back
    /// invariant the strict-energy-balance gate depends on. With two
    /// zones pinned to distinct interior boundary temperatures (15 °C and
    /// 25 °C) and a uniform exterior BC, the post-step proxy must reflect
    /// each zone's pinned T_air.
    #[test]
    fn test_zone_interior_temperatures_post_multi_zone_step() {
        let mut mz = MultiZoneGaugeSolver::new();
        mz.add_zone(0, 48.0, 2.7);
        mz.add_zone(1, 32.0, 2.7);
        let wall = conductive_stub_wall();
        mz.add_opaque_surface_to_zone(0, &wall, 21.6, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        mz.add_opaque_surface_to_zone(1, &wall, 12.0, SurfaceType::Wall, 180.0, 90.0)
            .unwrap();
        mz.initialize().unwrap();

        // Pin each zone to a distinct T_air before the step so the
        // per-surface gauge integrates against a per-zone boundary.
        mz.get_zone_mut(0).unwrap().set_T_air(15.0);
        mz.get_zone_mut(1).unwrap().set_T_air(25.0);

        // Uniform exterior BC across both zones (cold + sunny), so the
        // only thing differentiating the per-zone states is the pinned
        // interior boundary.
        let mut bc = HashMap::new();
        let cold = Temperature::from_value(5.0);
        let h_ext = HeatTransferCoefficient::from_value(25.0);
        bc.insert(0, ZoneBoundaryConditions::new(cold, h_ext, 300.0, 0.0));
        bc.insert(1, ZoneBoundaryConditions::new(cold, h_ext, 300.0, 0.0));

        mz.step(3600.0, &bc).unwrap();

        let per_zone = mz.zone_interior_temperatures();
        assert_eq!(per_zone.len(), mz.num_zones());
        // Each zone's accessor must reflect the T_air the gauge
        // integrated against (GaugeSolver writes T_int into
        // prev_T_interior at end-of-step, which `interior_temperature()`
        // returns).
        assert!(
            (per_zone[0] - 15.0).abs() < 1e-9,
            "zone 0 interior T must equal its pinned T_air=15 °C, got {per_zone:?}"
        );
        assert!(
            (per_zone[1] - 25.0).abs() < 1e-9,
            "zone 1 interior T must equal its pinned T_air=25 °C, got {per_zone:?}"
        );
    }

    // ============== Issue #3889 — solar-coupled T_air on the multi-zone path ==============
    //
    // The single-zone fix (#3878 / PR #3884) replaced the `h_eff · T_ext`
    // surface heat-flow proxy in `GaugeZoneSolver::step`'s T_air update with
    // the solar-aware surface flux sum. `step_with_coupling` kept the proxy,
    // so solar admitted through windows was invisible to the multi-zone air
    // update — a free-floating zone stayed pinned at outdoor air temperature
    // regardless of irradiance (the ~39.6 °C free-float ceiling on multi-zone
    // ASHRAE 140 configurations). The tests below verify the behavior through
    // the public `MultiZoneGaugeSolver::step` interface.

    /// Issue #3889 — realistic insulated wall (R ≈ 2.4 m²K/W) so the
    /// explicit surface flux stays inside the T_air update's stability
    /// envelope at hourly timesteps (the `conductive_stub_wall` stub is
    /// R = 0.09 — ~27× too conductive for the real Case 600 construction,
    /// Issue #3893 — whose H·dt > 2·C_air is unstable once surface flux is
    /// treated explicitly, matching the #3817 stability analysis in `step`;
    /// see that fixture's doc comment for the full divergence arithmetic).
    fn insulated_wall() -> WallSpec {
        WallSpec::single_layer("Insulated", 0.24, 0.1, 50.0, 50.0)
    }

    /// Build the Issue #3889 sunspace pair: zone 0 carries a 40 m² window
    /// (the only solar aperture), zone 1 is opaque, and the zones share a
    /// 10 m² / R=0.5 coupling wall. Both zones are free-floating.
    fn sunspace_pair() -> MultiZoneGaugeSolver {
        let mut mz = MultiZoneGaugeSolver::new();
        mz.add_zone(0, 48.0, 2.7);
        mz.add_zone(1, 48.0, 2.7);

        let wall = insulated_wall();
        mz.add_opaque_surface_to_zone(0, &wall, 40.0, SurfaceType::Window, 0.0, 90.0)
            .unwrap();
        mz.add_opaque_surface_to_zone(0, &wall, 21.6, SurfaceType::Wall, 180.0, 90.0)
            .unwrap();
        mz.add_opaque_surface_to_zone(1, &wall, 21.6, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        mz.add_zone_coupling(0, 1, 10.0, 0.5).unwrap();
        mz.initialize().unwrap();
        mz
    }

    /// Step the sunspace pair to steady state under constant irradiance at
    /// 20 °C outdoor air; return (zone 0 T_air, zone 1 T_air).
    ///
    /// Issue #3297 / LIMIT-21 Phase 5: `t_sky = T_exterior` neutralises the
    /// sky-radiative term so these tests isolate pure solar / conduction
    /// physics. The tests' names ("no solar settles at ambient", "solar lift
    /// scales linearly") reflect the old h_rad_sky=0 behaviour; neutralising
    /// the term here preserves the original test intent while allowing the
    /// per-surface h_rad_sky computation to run correctly in production.
    fn sunspace_settled_temps(solar_wm2: f64) -> (f64, f64) {
        let mut mz = sunspace_pair();
        let mut bc = HashMap::new();
        // Neutralise sky radiative term (t_sky = T_exterior → zero contribution)
        let entry = ZoneBoundaryConditions::new(
            Temperature::from_value(20.0),
            HeatTransferCoefficient::from_value(25.0),
            solar_wm2,
            0.0, // solar_distribution_to_air
        )
        .with_sky_radiation(20.0, 0.0); // t_sky = T_exterior, h_rad_sky = 0
                                        // Issue #3918 follow-up: solar_irradiance_wm2 is the zone-level solar
                                        // density (5R1C solar_gains semantics) — the opaque neighbor has no
                                        // aperture, so its density is zero. Zone 0 alone is sunlit.
        let mut entry_neighbor = entry.clone();
        entry_neighbor.solar_irradiance_wm2 = 0.0;
        bc.insert(0, entry);
        bc.insert(1, entry_neighbor);
        for _ in 0..240 {
            mz.step(3600.0, &bc).unwrap();
        }
        (
            mz.get_zone(0).unwrap().T_air().to_value(),
            mz.get_zone(1).unwrap().T_air().to_value(),
        )
    }

    #[test]
    fn step_with_coupling_solar_lifts_free_float_above_ambient() {
        let (t0, _) = sunspace_settled_temps(800.0);
        assert!(
            t0 > 25.0,
            "sunlit zone must float above T_ext + 5 °C under 800 W/m², got {t0:.2} °C"
        );
    }

    #[test]
    fn step_counts_internal_gains_once_in_t_air_update() {
        // Companion of the #3889 multi-zone fix: the single-zone step()
        // T_air update must count Q_internal exactly once. The formula
        // historically carried an explicit `+ Q_internal_w` term while
        // `net_power_watts` already accumulated it, so a free-floating
        // zone settled at T_ext + 2·Q/(H + h_inf) instead of
        // T_ext + Q/(H + h_inf) — a +9 °C error at Case 600 gains.
        let mut zone = GaugeZoneSolver::new(48.0, 2.7);
        let wall = insulated_wall();
        zone.add_opaque_surface(&wall, 40.0, SurfaceType::Window, 0.0, 90.0)
            .unwrap();
        zone.add_opaque_surface(&wall, 21.6, SurfaceType::Wall, 180.0, 90.0)
            .unwrap();
        zone.initialize().unwrap();

        // 480 W into a zone whose envelope + infiltration conductance is
        // ~47 W/K settles ~10 °C above ambient when counted once (≈30 °C),
        // ~20 °C above when double-counted (≈40 °C).
        for _ in 0..240 {
            zone.step(
                0,
                3600.0,
                Temperature::from_value(20.0),
                HeatTransferCoefficient::from_value(25.0),
                0.0,   // solar_irradiance_wm2
                0.0,   // solar_distribution_to_air
                480.0, // Q_internal_w
                0.0,   // Q_infiltration_w
                0.0,   // t_sky
                0.0,   // h_rad_sky
                0.0,   // ventilation_ach (Issue #3904: 0 = no night vent)
                // Issue #3918: solar lag parameters (0.0 = lag disabled)
                0.0, // h_tr_3
                0.0, // cm
                0.0, // h_tr_is
                0.0, // term_rest_1
                0.0, // convective_fraction
                0.0, // solar_beam_to_mass_fraction
            )
            .unwrap();
        }
        let t = zone.T_air().to_value();
        assert!(
            t > 25.0 && t < 35.0,
            "internal gains must count once: expected ≈30 °C, double-count lands ≈40 °C, got {t:.2} °C"
        );
    }

    // Issue #3983 (slice 1) — per-surface absorbed-solar accounting closes
    // the zone solar balance: Σ_i solar_absorbed_w,i + direct-to-air = Φ_sol.
    //
    // The interior surface network already computes per-surface absorption
    // internally (step_interior_surface_network); this exposes it as public
    // telemetry so the per-surface solar-distribution contract is observable.
    // Later slices (per-surface FD stacks, irradiance weighting) must keep
    // this closure exact — it is the energy gate for the #3983 rework.
    #[test]
    fn per_surface_solar_absorption_closes_zone_solar_balance() {
        let mut zone = GaugeZoneSolver::new(48.0, 2.7);
        let wall = insulated_wall();
        // Two equal-area opaque walls: equal shares under both the
        // conductance weight (h_i/Σh) and the capacitance weight (C_i/ΣC).
        zone.add_opaque_surface(&wall, 21.6, SurfaceType::Wall, 0.0, 90.0)
            .unwrap();
        zone.add_opaque_surface(&wall, 21.6, SurfaceType::Wall, 180.0, 90.0)
            .unwrap();
        zone.initialize().unwrap();

        let solar_density = 100.0; // W per m² floor
        let to_air = 0.30;
        let beam_to_mass = 0.30;
        zone.step(
            0,
            3600.0,
            Temperature::from_value(20.0),
            HeatTransferCoefficient::from_value(25.0),
            solar_density,
            to_air,
            0.0, // Q_internal_w
            0.0, // Q_infiltration_w
            0.0, // t_sky
            0.0, // h_rad_sky
            0.0, // ventilation_ach
            5.0,  // h_tr_3
            5000.0, // cm
            500.0, // h_tr_is (zone total)
            0.0,   // term_rest_1
            0.0,   // convective_fraction
            beam_to_mass,
        )
        .unwrap();

        let tel = zone.per_surface_telemetry();
        assert_eq!(tel.len(), 2, "both walls must report telemetry");
        let phi_sol = solar_density * 48.0; // 4800 W
        let direct_air = phi_sol * to_air; // 1440 W
        let absorbed: f64 = tel.iter().map(|t| t.solar_absorbed_w).sum();
        assert!(
            (absorbed + direct_air - phi_sol).abs() < 1e-9,
            "solar balance must close: Σ_absorbed ({absorbed:.3}) + to_air ({direct_air:.3}) \
             != Φ_sol ({phi_sol:.3})"
        );
        assert!(
            (tel[0].solar_absorbed_w - tel[1].solar_absorbed_w).abs() < 1e-9,
            "equal-area walls must absorb equal solar, got {} vs {}",
            tel[0].solar_absorbed_w,
            tel[1].solar_absorbed_w
        );
        // Each wall takes half of the non-air solar: 4800 · 0.7 / 2 = 1680 W.
        assert!(
            (tel[0].solar_absorbed_w - 1680.0).abs() < 1e-6,
            "expected 1680 W per wall, got {}",
            tel[0].solar_absorbed_w
        );
    }

    #[test]
    fn step_with_coupling_return_energy_convention() {
        // The returned net load keeps the documented convention (positive =
        // heating needed, negative = cooling needed) and still reflects the
        // full net power (surface + inter-zone + internal), independent of
        // the T_air update formula change.

        // Gaining heat: sunlit first step from T_air = T_ext.
        let mut mz = sunspace_pair();
        let mut bc = HashMap::new();
        let entry = ZoneBoundaryConditions::new(
            Temperature::from_value(20.0),
            HeatTransferCoefficient::from_value(25.0),
            800.0,
            0.0, // solar_distribution_to_air
        );
        bc.insert(0, entry.clone());
        bc.insert(1, entry);
        let results = mz.step(3600.0, &bc).unwrap();
        let e0 = *results.get(&0).unwrap();
        // Issue #3918 follow-up: the first step now admits ~the full zone
        // solar (800 W/m² · 48 m² floor ≈ 38 kW, mostly instant through the
        // low-capacitance surfaces) instead of the former ~1 kW sol-air
        // remnant. Bounds assert the SIGN convention and order of magnitude.
        assert!(
            e0 < -5.0 && e0 > -45.0,
            "sunlit gaining zone must report cooling load (negative kWh), got {e0:.3}"
        );

        // Losing heat: cold night, no solar.
        let mut mz = sunspace_pair();
        let mut bc = HashMap::new();
        let entry = ZoneBoundaryConditions::new(
            Temperature::from_value(0.0),
            HeatTransferCoefficient::from_value(25.0),
            0.0,
            0.0, // solar_distribution_to_air
        );
        bc.insert(0, entry.clone());
        bc.insert(1, entry);
        let results = mz.step(3600.0, &bc).unwrap();
        let e0 = *results.get(&0).unwrap();
        assert!(
            e0 > 0.1 && e0 < 2.0,
            "zone losing heat to the cold outdoors must report heating load (positive kWh), got {e0:.3}"
        );
    }

    #[test]
    fn step_with_coupling_solar_gain_spreads_to_coupled_neighbor() {
        // The #3889 fix must not disturb the #3817 inter-zone coupling:
        // solar admitted to zone 0 flows through the shared boundary, so
        // the opaque neighbor floats above ambient while staying cooler
        // than the sunlit zone (heat flows downhill).
        let (t0, t1) = sunspace_settled_temps(800.0);
        assert!(
            t1 > 21.0,
            "coupled neighbor must share the solar gain (float above T_ext), got {t1:.2} °C"
        );
        assert!(
            t1 < t0,
            "heat must flow downhill: neighbor {t1:.2} °C must stay below sunlit zone {t0:.2} °C"
        );
    }

    #[test]
    fn step_with_coupling_no_solar_settles_at_ambient() {
        // With zero irradiance and no internal gains, a free-floating pair
        // must settle at outdoor air temperature — solar coupling must not
        // over-heat the air update when the sun is absent.
        let (t0, t1) = sunspace_settled_temps(0.0);
        assert!(
            (t0 - 20.0).abs() < 0.5,
            "zone 0 must settle at T_ext without solar, got {t0:.2} °C"
        );
        assert!(
            (t1 - 20.0).abs() < 0.5,
            "zone 1 must settle at T_ext without solar, got {t1:.2} °C"
        );
    }

    #[test]
    fn step_with_coupling_solar_lift_scales_with_irradiance() {
        // The surface flux is linear in irradiance (sol-air film adds
        // solar / h_ext to the effective exterior temperature), so the
        // free-float lift must track irradiance proportionally. Guards
        // against solar entering through a saturating or capped path.
        let (t0_low, _) = sunspace_settled_temps(400.0);
        let (t0_high, _) = sunspace_settled_temps(800.0);
        let lift_low = t0_low - 20.0;
        let lift_high = t0_high - 20.0;
        assert!(
            lift_low > 0.0 && lift_high > lift_low,
            "lift must grow with irradiance: 400 W/m² -> {lift_low:.2} K, 800 W/m² -> {lift_high:.2} K"
        );
        let ratio = lift_high / lift_low;
        assert!(
            (ratio - 2.0).abs() < 0.1,
            "doubling irradiance must double the lift (linear sol-air physics), ratio {ratio:.3}"
        );
    }

    // ============== Issue #3729 — `GaugeZoneSolver` clone contract ==============
    //
    // The hand-rolled `Clone` impls in this module are pinned by the
    // tests below. They mirror the
    // `tests/all_tests/hybrid_clone_preserves_dispatch_counters.rs`
    // pattern: topology preserved, runtime state reset, candidates
    // independent.

    /// Build a Case 600-style initialized zone with two opaque surfaces.
    fn build_initialized_zone() -> GaugeZoneSolver {
        let wall = conductive_stub_wall();
        let mut zone = GaugeZoneSolver::new(48.0, 2.7);
        zone.add_opaque_surface(&wall, 48.0, SurfaceType::Wall, 180.0, 90.0)
            .expect("Case 600 wall must add");
        zone.add_opaque_surface(&wall, 48.0, SurfaceType::Roof, 180.0, 0.0)
            .expect("Case 600 roof must add");
        zone.initialize().expect("Case 600 zone must initialize");
        zone.set_T_air(35.0); // Pin a mid-solve zone air temperature.
        zone
    }

    /// `GaugeZoneSolver::clone` must reset `T_air` to the
    /// `new_with_id` default (20.0) regardless of the parent's mid-solve
    /// value. The clone starts from the same baseline a freshly-constructed
    /// solver would, matching the
    /// `HybridThermalModel::conduction_solver` slot-reset precedent.
    #[test]
    fn issue_3729_clone_resets_t_air_to_construction_default() {
        let original = build_initialized_zone();
        assert!(
            original.is_initialized(),
            "fixture: original must be initialized"
        );
        assert!(
            (original.T_air().to_value() - 35.0).abs() < 1e-9,
            "fixture: original T_air must be pinned to 35.0, got {}",
            original.T_air().to_value()
        );

        let clone = original.clone();

        assert!(
            (clone.T_air().to_value() - 20.0).abs() < 1e-9,
            "clone.T_air must reset to the new_with_id default (20.0), got {} \
             (Issue #3729: clone must NOT carry forward the parent's mid-solve T_air)",
            clone.T_air().to_value()
        );
        // Original must remain untouched (Clone reads `&self`).
        assert!(
            (original.T_air().to_value() - 35.0).abs() < 1e-9,
            "Clone must not perturb the original's T_air (Issue #3729 independence)"
        );
    }

    /// `GaugeZoneSolver::clone` must reset `initialized` to `false`.
    /// The clone is not pre-solved; callers must re-initialize before
    /// stepping the clone. Mirrors the slot-reset contract for
    /// `HybridThermalModel::conduction_solver` (ARCHITECTURE.md #2539).
    #[test]
    fn issue_3729_clone_resets_initialized_flag() {
        let original = build_initialized_zone();
        assert!(original.is_initialized());

        let clone = original.clone();

        assert!(
            !clone.is_initialized(),
            "clone.is_initialized must be false after Clone (Issue #3729 slot-reset contract); \
             got true"
        );
        assert!(
            original.is_initialized(),
            "Clone must not perturb the original's is_initialized (Issue #3729 independence)"
        );
    }

    /// `GaugeZoneSolver::clone` must preserve topology: surfaces (count
    /// and metadata), `zone_id`, `floor_area`, `zone_volume`, `C_air`,
    /// couplings, and the per-surface `wall_spec`. A candidate model
    /// built from this clone must have the same envelope and adjacency
    /// as the original — only the runtime state diverges.
    #[test]
    fn issue_3729_clone_preserves_topology() {
        let original = build_initialized_zone();
        let original_zone_id = original.zone_id();
        let original_surface_count = original.surface_count();
        let original_floor_area = original.floor_area;
        let original_zone_volume = original.zone_volume;
        let original_C_air = original.C_air();

        let clone = original.clone();

        assert_eq!(
            clone.zone_id(),
            original_zone_id,
            "clone must preserve zone_id"
        );
        assert_eq!(
            clone.surface_count(),
            original_surface_count,
            "clone must preserve surface count"
        );
        assert!(
            (clone.floor_area - original_floor_area).abs() < 1e-12,
            "clone must preserve floor_area"
        );
        assert!(
            (clone.zone_volume - original_zone_volume).abs() < 1e-12,
            "clone must preserve zone_volume"
        );
        assert!(
            (clone.C_air() - original_C_air).abs() < 1e-9,
            "clone must preserve C_air"
        );
        // Per-surface metadata (area, type, wall_spec) must round-trip.
        for (orig_surface, clone_surface) in original.surfaces().iter().zip(clone.surfaces().iter())
        {
            assert!(
                (orig_surface.area_m2 - clone_surface.area_m2).abs() < 1e-12,
                "clone must preserve per-surface area_m2"
            );
            assert_eq!(
                orig_surface.surface_type, clone_surface.surface_type,
                "clone must preserve per-surface surface_type"
            );
        }
    }

    /// Two clones are fully independent: mutating one's `T_air` (or
    /// stepping one) must not perturb the other. Pinned by the
    /// candidate-independence requirement in the BatchOracle
    /// `par_iter` hot loop — a shared-state bug would corrupt every
    /// per-config dispatch in `evaluate_population`.
    #[test]
    fn issue_3729_clones_are_independent() {
        let original = build_initialized_zone();
        let mut clone_a = original.clone();
        let mut clone_b = original.clone();

        clone_a.set_T_air(15.0);
        clone_b.set_T_air(30.0);

        // Each clone's T_air must reflect only its own mutation.
        assert!(
            (clone_a.T_air().to_value() - 15.0).abs() < 1e-9,
            "clone_a T_air must equal its own pin (15.0), got {}",
            clone_a.T_air().to_value()
        );
        assert!(
            (clone_b.T_air().to_value() - 30.0).abs() < 1e-9,
            "clone_b T_air must equal its own pin (30.0), got {}",
            clone_b.T_air().to_value()
        );
        // Original must remain untouched.
        assert!(
            (original.T_air().to_value() - 35.0).abs() < 1e-9,
            "original T_air must be untouched by subsequent clone mutations"
        );

        // Re-initialize each clone and step independently. The clone's
        // freshly-reset surface gauge state must be runnable end-to-end.
        clone_a.initialize().expect("clone_a must re-initialize");
        clone_b.initialize().expect("clone_b must re-initialize");
        let bc = (
            Temperature::from_value(10.0),
            HeatTransferCoefficient::from_value(25.0),
        );
        let _ = clone_a.step(
            0, 3600.0, bc.0, bc.1, 0.0, // solar_irradiance_wm2
            0.0, // solar_distribution_to_air
            0.0, // Q_internal_w
            0.0, // Q_infiltration_w
            0.0, // t_sky
            0.0, // h_rad_sky
            0.0, // ventilation_ach (Issue #3904: 0 = no night vent)
            // Issue #3918: solar lag parameters (0.0 = lag disabled)
            0.0, // h_tr_3
            0.0, // cm
            0.0, // h_tr_is
            0.0, // term_rest_1
            0.0, // convective_fraction
            0.0, // solar_beam_to_mass_fraction
        );
        let _ = clone_b.step(
            1, 3600.0, bc.0, bc.1, 0.0, // solar_irradiance_wm2
            0.0, // solar_distribution_to_air
            0.0, // Q_internal_w
            0.0, // Q_infiltration_w
            0.0, // t_sky
            0.0, // h_rad_sky
            0.0, // ventilation_ach (Issue #3904: 0 = no night vent)
            // Issue #3918: solar lag parameters (0.0 = lag disabled)
            0.0, // h_tr_3
            0.0, // cm
            0.0, // h_tr_is
            0.0, // term_rest_1
            0.0, // convective_fraction
            0.0, // solar_beam_to_mass_fraction
        );
        assert!(
            clone_a.is_initialized(),
            "clone_a must remain functional after re-init + step"
        );
        assert!(
            clone_b.is_initialized(),
            "clone_b must remain functional after re-init + step"
        );
    }

    /// `SurfaceGaugeSolver::clone` (transitively invoked by
    /// `GaugeZoneSolver::clone`) must re-initialize the per-surface
    /// `GaugeSolver` from `wall_spec` when available. After clone, the
    /// per-surface `GaugeSolver` must be `initialized = true` with the
    /// expected `r_total` / `C_mass` derived from the wall spec.
    #[test]
    fn issue_3729_clone_resets_per_surface_gauge_state() {
        use crate::physics::solver_trait::HeatConductionSolver;

        let original = build_initialized_zone();
        let pre_clone_surface_count = original.surface_count();
        let pre_clone_first_r_total = original.surfaces()[0].gauge.r_total_for_test();
        let pre_clone_first_c_mass = original.surfaces()[0].gauge.c_mass_for_test();

        let clone = original.clone();

        assert_eq!(clone.surface_count(), pre_clone_surface_count);
        for (i, clone_surface) in clone.surfaces().iter().enumerate() {
            assert!(
                clone_surface.gauge.is_valid(),
                "clone.surfaces[{i}].gauge must be in the initialized form (Issue #3729)"
            );
            // The wall-derived r_total / C_mass must round-trip across clone.
            let r_total = clone_surface.gauge.r_total_for_test();
            let c_mass = clone_surface.gauge.c_mass_for_test();
            assert!(
                (r_total - pre_clone_first_r_total).abs() < 1e-12,
                "clone.surfaces[{i}].gauge.r_total must match wall spec, got {r_total}"
            );
            assert!(
                (c_mass - pre_clone_first_c_mass).abs() < 1e-12,
                "clone.surfaces[{i}].gauge.C_mass must match wall spec, got {c_mass}"
            );
        }
    }

    /// `MultiZoneGaugeSolver::clone` must reset each per-zone state
    /// (via `GaugeZoneSolver::clone`) AND the aggregate `initialized`
    /// flag. Topology (zone count, IDs, surface counts, couplings) is
    /// preserved.
    #[test]
    fn issue_3729_multi_zone_clone_resets_aggregate_state() {
        let wall = conductive_stub_wall();
        let mut mz = MultiZoneGaugeSolver::new();
        mz.add_zone(0, 48.0, 2.7);
        mz.add_zone(1, 36.0, 2.7);
        mz.add_opaque_surface_to_zone(0, &wall, 48.0, SurfaceType::Wall, 180.0, 90.0)
            .unwrap();
        mz.add_opaque_surface_to_zone(1, &wall, 36.0, SurfaceType::Wall, 180.0, 90.0)
            .unwrap();
        mz.add_zone_coupling(0, 1, 10.0, 0.5).unwrap();
        mz.initialize().expect("multi-zone must initialize");
        mz.get_zone_mut(0).unwrap().set_T_air(28.0);
        mz.get_zone_mut(1).unwrap().set_T_air(22.0);

        let clone = mz.clone();

        // Topology preserved.
        assert_eq!(clone.num_zones(), 2);
        assert_eq!(clone.zone_ids(), &[0, 1]);
        assert_eq!(clone.get_zone(0).unwrap().surface_count(), 1);
        assert_eq!(clone.get_zone(1).unwrap().surface_count(), 1);

        // Per-zone state reset to defaults.
        assert!(
            (clone.get_zone(0).unwrap().T_air().to_value() - 20.0).abs() < 1e-9,
            "clone.zone[0].T_air must reset to 20.0, got {}",
            clone.get_zone(0).unwrap().T_air().to_value()
        );
        assert!(
            (clone.get_zone(1).unwrap().T_air().to_value() - 20.0).abs() < 1e-9,
            "clone.zone[1].T_air must reset to 20.0, got {}",
            clone.get_zone(1).unwrap().T_air().to_value()
        );

        // Aggregate `initialized` flag reset.
        assert!(
            !clone.is_initialized(),
            "clone.is_initialized must be false (Issue #3729 aggregate slot-reset contract)"
        );

        // Originals untouched.
        assert!(
            (mz.get_zone(0).unwrap().T_air().to_value() - 28.0).abs() < 1e-9,
            "original zone[0].T_air must remain at 28.0 (independence guarantee)"
        );
        assert!(mz.is_initialized());
    }
}
