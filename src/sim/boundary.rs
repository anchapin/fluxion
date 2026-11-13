//! Ground boundary condition models for building energy simulation.
//!
//! This module provides traits and implementations for modeling ground temperatures
//! as thermal boundary conditions. Ground coupling is a critical component of building
//! heat transfer, affecting annual heating loads significantly.
//!
//! # ASHRAE 140 Specification
//!
//! Per ASHRAE Standard 140, the ground temperature is specified as a constant 10°C
//! for baseline test cases. This simplification is appropriate for annual simulations
//! where ground temperature varies minimally at typical foundation depths.
//!
//! # Models Provided
//!
//! - [`ConstantGroundTemperature`](struct@ConstantGroundTemperature): Fixed temperature model
//! - [`DynamicGroundTemperature`](struct@DynamicGroundTemperature): Time-varying model using Kusuda formula
//!
//! # Example
//!
//! ```rust
//! use fluxion::sim::boundary::{GroundTemperature, ConstantGroundTemperature};
//!
//! // Create constant ground temperature (ASHRAE 140 default)
//! let ground = ConstantGroundTemperature::new(10.0);
//! let temp = ground.ground_temperature(1000); // Hour 1000 of year
//! assert_eq!(temp, 10.0);
//! ```

use std::boxed::Box;
use std::f64::consts::PI;

/// Trait for ground temperature models in building energy simulation.
///
/// Implementors provide ground temperature at specific timesteps, enabling
/// different modeling approaches from simple constant temperatures to
/// sophisticated time-varying models based on soil physics.
///
/// # Requirements
///
/// Implementors must be [`Send`] + [`Sync`] to enable thread-safe parallel
/// simulation of building populations.
///
/// # Example Implementation
///
/// ```rust
/// use fluxion::sim::boundary::GroundTemperature;
///
/// struct SimpleGround {
///     temp: f64,
/// }
///
/// impl GroundTemperature for SimpleGround {
///     fn clone_box(&self) -> Box<dyn GroundTemperature> {
///         Box::new(SimpleGround { temp: self.temp })
///     }
///
///     fn ground_temperature(&self, _hour_of_year: usize) -> f64 {
///         self.temp
///     }
/// }
/// ```
pub trait GroundTemperature: Send + Sync {
    /// Clone this trait object into a new Box.
    ///
    /// This is needed for implementing Clone on ThermalModel.
    fn clone_box(&self) -> Box<dyn GroundTemperature>;

    /// Get ground temperature at a given hour of the year.
    ///
    /// # Arguments
    ///
    /// * `hour_of_year` - Hour index (0-8759) in the annual cycle
    ///
    /// # Returns
    ///
    /// Ground temperature in degrees Celsius.
    ///
    /// # Notes
    ///
    /// - Hour values wrap around using modulo for year-to-year continuity
    /// - Implementations may ignore `hour_of_year` for constant models
    /// - Implementations that carry an internal state (e.g. the
    ///   ground-coupled floor boundary) march that state forward and keep it
    ///   across the year wrap, so the returned temperature is continuous
    ///   from one year into the next
    fn ground_temperature(&self, hour_of_year: usize) -> f64;
}

/// Constant ground temperature model.
///
/// This is the simplest ground model, suitable for annual simulations where
/// ground temperature variation at foundation depth is minimal. Per ASHRAE 140
/// specification, the default value is 10°C.
///
/// # Advantages
///
/// - Simple and fast (no calculation needed)
/// - Matches ASHRAE 140 baseline specification
/// - Appropriate for well-insulated slabs at typical depths (>1m)
///
/// # Limitations
///
/// - Ignores seasonal ground temperature variation
/// - Not suitable for shallow foundations or crawl spaces
/// - May underpredict summer heat gain/loss in some climates
///
/// # Example
///
/// ```rust
/// use fluxion::sim::boundary::{GroundTemperature, ConstantGroundTemperature};
///
/// // ASHRAE 140 default
/// let ground = ConstantGroundTemperature::new(10.0);
/// assert_eq!(ground.ground_temperature(0), 10.0);
/// assert_eq!(ground.ground_temperature(4380), 10.0);
/// ```
#[derive(Debug, Clone)]
pub struct ConstantGroundTemperature {
    /// Constant soil temperature in degrees Celsius
    temperature: f64,
}

impl ConstantGroundTemperature {
    /// Create a new constant ground temperature model.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Ground temperature in °C (typical range: 5-15°C for mid-latitude climates)
    ///
    /// # Example
    ///
    /// ```rust
    /// use fluxion::sim::boundary::ConstantGroundTemperature;
    ///
    /// // ASHRAE 140 specification
    /// let ground = ConstantGroundTemperature::new(10.0);
    /// ```
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    /// Get the constant temperature value.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Update the constant temperature value.
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
    }
}

impl GroundTemperature for ConstantGroundTemperature {
    fn clone_box(&self) -> Box<dyn GroundTemperature> {
        Box::new(self.clone())
    }
    fn ground_temperature(&self, _hour_of_year: usize) -> f64 {
        self.temperature
    }
}

/// Dynamic ground temperature model using the Kusuda formula.
///
/// This model calculates time-varying soil temperature based on annual climate
/// cycles, soil thermal properties, and depth below surface. The Kusuda formula
/// is widely used in building simulation for ground coupling calculations.
///
/// # Mathematical Model
///
/// The Kusuda formula for soil temperature at depth `z` and time `t`:
///
/// ```text
/// T(z,t) = T_mean - T_amp × exp(-d√(π/365α)) × cos(ωt - d√(π/365α))
/// ```
///
/// Where:
/// - `T_mean`: Mean annual soil temperature (°C)
/// - `T_amp`: Annual temperature amplitude (°C)
/// - `d`: Depth below surface (m)
/// - `α`: Soil thermal diffusivity (m²/day)
/// - `ω`: Angular frequency = 2π/365 (rad/day)
/// - `t`: Day of year (0-364)
///
/// # Physical Basis
///
/// - Temperature amplitude decays exponentially with depth
/// - Phase shift increases with depth (peak temp occurs later in year)
/// - Thermal diffusivity controls penetration depth and lag time
///
/// # Typical Parameter Values
///
/// | Parameter | Typical Range | Notes |
/// |-----------|--------------|-------|
/// | T_mean | 5-15°C | Varies by climate (Denver: ~10-12°C) |
/// | T_amp | 8-15°C | Half of annual air temp swing |
/// | Depth | 0.5-2.0m | Slab thickness + insulation |
/// | Diffusivity | 0.05-0.1 m²/day | Dry sand: 0.05, moist clay: 0.08 |
///
/// # When to Use
///
/// - Shallow foundations (<1m depth)
/// - Basements or crawl spaces
/// - High-precision modeling where ground coupling is significant
/// - Climate zones with large annual temperature swings
///
/// # Example
///
/// ```rust
/// use fluxion::sim::boundary::{GroundTemperature, DynamicGroundTemperature};
///
/// // Denver-like climate parameters
/// let ground = DynamicGroundTemperature::new(
///     11.0,  // Mean annual temperature (°C)
///     12.0,  // Annual amplitude (°C)
///     1.0,   // Depth (m)
///     0.07,  // Diffusivity (m²/day)
/// );
///
/// // Temperature varies by hour and depth
/// let temp_winter = ground.ground_temperature(0);      // ~Jan 1
/// let temp_summer = ground.ground_temperature(4380); // ~Jul 1
/// assert!(temp_summer > temp_winter);
/// ```
#[derive(Debug, Clone)]
pub struct DynamicGroundTemperature {
    /// Mean annual soil temperature (°C)
    t_mean: f64,
    /// Annual temperature amplitude (°C)
    t_amplitude: f64,
    /// Depth below ground surface (m)
    depth: f64,
    /// Soil thermal diffusivity (m²/day)
    diffusivity: f64,
}

impl DynamicGroundTemperature {
    /// Create a new dynamic ground temperature model using the Kusuda formula.
    ///
    /// # Arguments
    ///
    /// * `t_mean` - Mean annual soil temperature in °C (typical: 10-12°C)
    /// * `t_amplitude` - Annual temperature amplitude in °C (typical: 8-15°C)
    /// * `depth` - Depth below surface in meters (typical: 0.5-2.0m)
    /// * `diffusivity` - Soil thermal diffusivity in m²/day (typical: 0.05-0.1)
    ///
    /// # Physical Constraints
    ///
    /// - `depth` must be positive
    /// - `diffusivity` must be positive
    /// - `t_amplitude` must be non-negative
    ///
    /// # Panics
    ///
    /// Will panic if physical constraints are violated.
    ///
    /// # Example
    ///
    /// ```rust
    /// use fluxion::sim::boundary::DynamicGroundTemperature;
    ///
    /// // Denver climate parameters
    /// let ground = DynamicGroundTemperature::new(
    ///     11.0,  // Mean temperature
    ///     12.0,  // Amplitude
    ///     1.0,   // Depth
    ///     0.07,  // Diffusivity
    /// );
    /// ```
    pub fn new(t_mean: f64, t_amplitude: f64, depth: f64, diffusivity: f64) -> Self {
        assert!(depth > 0.0, "Depth must be positive");
        assert!(diffusivity > 0.0, "Diffusivity must be positive");
        assert!(t_amplitude >= 0.0, "Amplitude must be non-negative");

        Self {
            t_mean,
            t_amplitude,
            depth,
            diffusivity,
        }
    }

    /// Get the mean annual soil temperature.
    pub fn t_mean(&self) -> f64 {
        self.t_mean
    }

    /// Get the annual temperature amplitude.
    pub fn t_amplitude(&self) -> f64 {
        self.t_amplitude
    }

    /// Get the depth below surface.
    pub fn depth(&self) -> f64 {
        self.depth
    }

    /// Get the soil thermal diffusivity.
    pub fn diffusivity(&self) -> f64 {
        self.diffusivity
    }

    /// Calculate the temperature damping factor at current depth.
    ///
    /// This factor represents how much the annual temperature amplitude
    /// is attenuated at the specified depth.
    ///
    /// # Returns
    ///
    /// Damping factor (dimensionless, range 0-1).
    pub fn damping_factor(&self) -> f64 {
        let decay = self.depth * (PI / (365.0 * self.diffusivity)).sqrt();
        (-decay).exp()
    }

    /// Calculate the phase shift in days.
    ///
    /// This represents how many days the ground temperature lags
    /// behind the surface temperature cycle.
    ///
    /// # Returns
    ///
    /// Phase shift in days.
    pub fn phase_shift(&self) -> f64 {
        let decay = self.depth * (PI / (365.0 * self.diffusivity)).sqrt();
        decay * 365.0 / PI
    }
}

impl GroundTemperature for DynamicGroundTemperature {
    fn clone_box(&self) -> Box<dyn GroundTemperature> {
        Box::new(self.clone())
    }

    fn ground_temperature(&self, hour_of_year: usize) -> f64 {
        // Convert hour to day (0-364)
        let day = (hour_of_year / 24) as f64 % 365.0;

        // Angular frequency (rad/day)
        let omega = 2.0 * PI / 365.0;

        // Decay parameter (dimensionless)
        let decay = self.depth * (PI / (365.0 * self.diffusivity)).sqrt();

        // Kusuda formula
        let damping = (-decay).exp();
        let phase = decay;
        let annual_cycle = (omega * day - phase).cos();

        self.t_mean - self.t_amplitude * damping * annual_cycle
    }
}

/// Monthly ground temperature model — Issue #1435 (IDF import support).
///
/// Holds 12 monthly ground temperature values (°C) as used by EnergyPlus'
/// `Site:GroundTemperature:BuildingSurface` object. The ground temperature
/// at hour `h` of the year is the value for the corresponding month
/// (interpolated as a step function between the 1st of each month).
///
/// `monthly[0]` is January, `monthly[11]` is December. Leap years are
/// treated as 365-day years for the lookup.
#[derive(Debug, Clone)]
pub struct MonthlyGroundTemperature {
    monthly: [f64; 12],
}

impl MonthlyGroundTemperature {
    /// Build a new monthly ground temperature model. The slice must contain
    /// exactly 12 entries (Jan…Dec); values are stored verbatim.
    ///
    /// # Panics
    /// Panics if `monthly.len() != 12`.
    pub fn new(monthly: [f64; 12]) -> Self {
        Self { monthly }
    }

    /// Try to build from a slice of any length. Returns `None` if the
    /// length is not 12.
    pub fn from_slice(monthly: &[f64]) -> Option<Self> {
        if monthly.len() != 12 {
            return None;
        }
        let mut arr = [0.0_f64; 12];
        arr.copy_from_slice(monthly);
        Some(Self { monthly: arr })
    }

    /// Monthly ground temperature values, January → December.
    pub fn monthly(&self) -> &[f64; 12] {
        &self.monthly
    }

    /// First hour of each month in a non-leap year (Jan=0, Feb=744,
    /// Mar=1416, …, Dec=8016). Used to map `hour_of_year` to the
    /// appropriate monthly value.
    const MONTH_START_HOURS: [usize; 12] = [
        0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016,
    ];
}

impl GroundTemperature for MonthlyGroundTemperature {
    fn clone_box(&self) -> Box<dyn GroundTemperature> {
        Box::new(self.clone())
    }

    fn ground_temperature(&self, hour_of_year: usize) -> f64 {
        // Find the most recent month start ≤ hour_of_year.
        let h = hour_of_year.min(Self::MONTH_START_HOURS[11]);
        let mut month = 0_usize;
        for (idx, start) in Self::MONTH_START_HOURS.iter().enumerate() {
            if *start <= h {
                month = idx;
            } else {
                break;
            }
        }
        self.monthly[month]
    }
}

// === Issue #864: Per-surface solar and internal gain distribution ===

/// Per-surface solar gain distribution result (Issue #864).
///
/// Contains the portion of total opaque solar gains distributed to each
/// envelope surface mass node (wall, roof, floor) in Watts.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SurfaceSolarGains {
    /// Opaque solar gain to wall mass node [W]
    pub phi_m_wall: f64,
    /// Opaque solar gain to roof mass node [W]
    pub phi_m_roof: f64,
    /// Opaque solar gain to floor mass node [W]
    pub phi_m_floor: f64,
}

/// Distribute total opaque solar gains to per-surface mass nodes (Issue #864).
///
/// Gains are distributed proportional to irradiance-weighted area fraction.
/// If all irradiance values are zero, distributes by area fraction alone.
pub fn distribute_opaque_solar_gains(
    total_opaque_solar: f64,
    wall_area: f64,
    roof_area: f64,
    floor_area: f64,
    wall_irradiance: f64,
    roof_irradiance: f64,
    floor_irradiance: f64,
) -> SurfaceSolarGains {
    let wall_weight = wall_area * wall_irradiance;
    let roof_weight = roof_area * roof_irradiance;
    let floor_weight = floor_area * floor_irradiance;
    let total_weight = wall_weight + roof_weight + floor_weight;

    if total_weight > 1e-10 {
        SurfaceSolarGains {
            phi_m_wall: total_opaque_solar * wall_weight / total_weight,
            phi_m_roof: total_opaque_solar * roof_weight / total_weight,
            phi_m_floor: total_opaque_solar * floor_weight / total_weight,
        }
    } else if wall_area + roof_area + floor_area > 1e-10 {
        let total_area = wall_area + roof_area + floor_area;
        SurfaceSolarGains {
            phi_m_wall: total_opaque_solar * wall_area / total_area,
            phi_m_roof: total_opaque_solar * roof_area / total_area,
            phi_m_floor: total_opaque_solar * floor_area / total_area,
        }
    } else {
        SurfaceSolarGains::default()
    }
}

/// Distribute radiative internal gains across surfaces by area fraction (Issue #864).
///
/// Returns a tuple `(wall_share, roof_share, floor_share)` in Watts.
pub fn distribute_radiative_gains(
    radiative_gains: f64,
    wall_area: f64,
    roof_area: f64,
    floor_area: f64,
) -> (f64, f64, f64) {
    let total_area = wall_area + roof_area + floor_area;
    if total_area > 1e-10 {
        (
            radiative_gains * wall_area / total_area,
            radiative_gains * roof_area / total_area,
            radiative_gains * floor_area / total_area,
        )
    } else {
        (0.0, 0.0, 0.0)
    }
}

// === Issue #4775: ground-coupled Section 7 floor boundary (Refs #4270) ===

/// Hours in a (non-leap) year — the period of the annual ground-surface
/// temperature cycle driven into the ground domain.
const HOURS_PER_YEAR: f64 = 8760.0;

/// Lateral cells across the ground domain (the 2-D cross-section under the
/// building footprint).
const GROUND_LATERAL_CELLS: usize = 8;

/// Target cell size of the fine top layer of the ground domain [m].
const GROUND_TOP_CELL_M: f64 = 0.1;

/// First cell size of the graded part of the ground domain [m].
const GROUND_GRADED_FIRST_CELL_M: f64 = 0.2;

/// Growth ratio of the graded part of the ground domain.
const GROUND_GRADED_GROWTH: f64 = 1.2;

/// Largest cell size of the graded part of the ground domain [m].
const GROUND_GRADED_MAX_CELL_M: f64 = 1.0;

/// Time step of the ground-domain transient march [s] (one hour).
const GROUND_TIME_STEP_S: f64 = 3600.0;

/// Maximum conjugate-gradient iterations per ground-domain step.
const GROUND_MAX_ITERATIONS: usize = 20_000;

/// Target relative residual per ground-domain step.
const GROUND_TOLERANCE: f64 = 1.0e-9;

/// The transient state of the ground domain, advanced together with the hour
/// it represents.
#[derive(Debug, Clone)]
struct GroundCoupledFloorState {
    /// The ground-domain transient state.
    transient: crate::physics::ground_conduction::GroundTransientState,
    /// The absolute hour of the marched clock: the number of hourly steps
    /// taken since the initial state.
    ///
    /// The clock is monotonically increasing — it never rewinds and the soil
    /// state is never reset — so the ground state carries across the year
    /// boundary (Issue #4794). The hour of the year the state represents is
    /// `abs_hour % 8760`.
    abs_hour: u64,
}

/// Ground-coupled floor boundary (Issue #4775, Refs #4270).
///
/// This is the Section 7 floor boundary's ground-side counterpart of
/// [`ConstantGroundTemperature`]: instead of a constant temperature it drives
/// the floor boundary from the 2-D ground-conduction domain
/// (`crate::physics::ground_conduction`), the standalone ASHRAE 140 §8 ground
/// model that Issue #4270 landed and that Issue #4774 extended with
/// transient stepping.
///
/// # Model
///
/// The ground domain is the soil column below the building footprint:
///
/// * the top surface follows an annual ground-surface temperature (mean plus
///   cosine amplitude, both from the case specification) through a soil
///   surface film,
/// * the deep and lateral far-field boundaries hold the undisturbed ground
///   temperature,
/// * the soil's transient response damps and lags that surface forcing,
/// * the floor boundary reads the resulting soil temperature at the coupling
///   depth below the underside of the floor construction.
///
/// The coupling is one-way — the soil is driven by the ground-surface
/// temperature, not by the zone floor temperature — because the
/// [`GroundTemperature`] interface only exposes the hour of the year. A later
/// issue can thread the zone-side feedback through the same object.
///
/// # Resistance trace of the floor path (Issue #4789)
///
/// The zone model couples the floor to the ground with one conductance,
/// `h_tr_floor` [W/K], applied between the zone's floor surface node `T_s`
/// and the ground boundary temperature `T_g`:
///
/// ```text
/// Q_floor  = h_tr_floor × (T_g − T_s)            (src/sim/diagnostics.rs:26)
/// h_tr_floor = floor_u_value × floor_area         (src/sim/thermal_model_core/mod.rs:1541)
/// floor_u_value = construction.floor
///                 .u_value(Some(SurfaceType::Floor), None)
///                                              (src/sim/thermal_model_core/mod.rs:891-895)
/// ```
///
/// `Construction::u_value` is `1 / r_value_total`, and for a floor
/// `r_value_total` (`fluxion-core/src/construction.rs:394-424`) is
///
/// ```text
/// R_total = R_si + R_materials + R_exterior
/// ```
///
/// with
///
/// * `R_si = 1 / SurfaceType::interior_film_coeff(Floor) = 1/5.88 =
///   0.17007 m²K/W` — the interior (zone-side) film of the floor
///   (`fluxion-core/src/construction.rs:68,108`);
/// * `R_materials = Σ δ_i / k_i` — the floor construction layers
///   (`fluxion-core/src/construction.rs:408`);
/// * `R_exterior` — the *exterior* resistance of the floor. For a floor it is
///   **not** the wind film: Issue #588 replaced it with the lumped
///   ground-coupling resistance `R_g = 0.17 m²K/W`, commented
///   *"Approximate value: R_g = 0.17 m²K/W (includes soil resistance)"*
///   (`fluxion-core/src/construction.rs:410-424`).
///
/// ## Constant-temperature path (every shipped case)
///
/// `floor_ground_boundary` returns `ConstantGroundTemperature(ground_temperature_c)`
/// (`src/sim/boundary.rs:799-812`), so `T_g` is prescribed and the whole
/// series from the floor surface node to the ground is
///
/// ```text
/// T_s ──[ R_si + R_materials + R_g ]──> T_g (prescribed)
/// ```
///
/// `R_g` is the *only* representation of the ground-side (soil) resistance in
/// this path — no soil is modelled anywhere else — so **nothing is counted
/// twice** here.
///
/// ## Coupled path (opt-in, `CaseSpec::ground_coupled_floor`)
///
/// [`GroundCoupledFloorBoundary::new`] (`src/sim/boundary.rs:624-707`) builds
/// the 2-D ground domain of `crate::physics::ground_conduction`:
///
/// * the domain's top surface (`z = 0`) is the **underside of the floor
///   construction**: the "slab" layer of the domain is *soil*, not slab
///   concrete — `slab_thickness_m = coupling_depth_m` and
///   `slab_conductivity_w_mk = ground_conductivity_w_mk = soil_conductivity_w_mk`
///   (`src/sim/boundary.rs:679-681`);
/// * the top boundary is a Robin (film) condition towards the annual
///   ground-surface temperature — the soil surface film
///   `R_film_soil = 1 / spec.surface_film_coefficient_w_m2k`
///   (`src/sim/boundary.rs:660-667`);
/// * the deep and the lateral far-field boundaries hold the undisturbed ground
///   temperature (Dirichlet);
/// * the floor boundary reads the soil temperature at the **coupling depth**,
///   i.e. *below* that soil layer (`readout_depth_m = coupling_depth_m`,
///   `readout_x_m = L/2`; `src/sim/boundary.rs:701-702, 731-737`).
///
/// The ground-domain branch that produces `T_g` is therefore
///
/// ```text
/// T_gs(h) ──[ R_film_soil ]──> soil(z=0) ──[ R_soil(0 → z_coupling) ]──> T_g
///                                   │
///                                   └──[ R_soil(z_coupling → z_bottom)
///                                          + lateral far-field ]──> undisturbed ground
/// ```
///
/// while the zone-side series stays
///
/// ```text
/// T_s ──[ R_si + R_materials + R_g ]──> T_g
/// ```
///
/// ## Conclusion: the ground resistance is counted twice on the coupled path
///
/// `R_g = 0.17 m²K/W` is documented as an *effective ground-coupling
/// resistance that includes soil resistance* (Issue #588,
/// `fluxion-core/src/construction.rs:410-424`): it stands for the soil
/// resistance between the floor and the ground the zone sees. On the coupled
/// path the very same ground resistance is *also* resolved explicitly by the
/// ground domain — the soil layer between the floor underside (the domain's
/// top surface, `z = 0`) and the coupling depth is a soil resistance, and
/// `T_g` is read at the *bottom* of that layer. The ground resistance between
/// the floor and the coupling plane is therefore represented twice: once as
/// the lumped `R_g` inside `floor_u_value` → `h_tr_floor`, and once as the
/// explicit soil layer of the ground domain that the boundary value is read
/// through.
///
/// The soil-surface film `R_film_soil` and the soil *below* the coupling depth
/// are **not** part of the zone's series path: they sit beyond the readout
/// plane and only shape `T_g`.
///
/// Nuance (one-way coupling): because the coupling is one-way, the domain's
/// soil column does not carry the zone's heat, so this is a *representation*
/// double count — the same ground resistance appears in the zone conductance
/// *and* in the ground domain that produces the boundary temperature — rather
/// than two ground resistances multiplied into one denominator. Either way
/// the series accounting is wrong: the zone's floor conductance must carry
/// the ground-side resistance **once**, and on the coupled path the ground
/// domain is what carries it.
///
/// Minimal correction (Issue #4789): on the opt-in path the floor U-value
/// omits the lumped ground-coupling resistance `R_g`
/// (`Construction::u_value_without_ground_resistance`,
/// `fluxion-core/src/construction.rs`), because the ground domain carries the
/// ground-side resistance; every non-opt-in case keeps the Issue #588 value
/// bit-identically. The steady-state acceptance test lives in this module's
/// `tests` submodule
/// (`test_ground_coupled_floor_flux_matches_analytic_series`).
///
/// # State
///
/// The ground domain is marched forward in time as the simulation clock
/// advances; the march is idempotent per hour (asking twice for the same
/// hour returns the same value). The march is **forward only**: the soil
/// state is never rewound and never reset, so crossing the end of the year
/// continues the soil history instead of restarting the domain from a
/// uniform field (Issue #4794). The annual ground-surface forcing is
/// 8760-periodic and is sampled at `hour % 8760`, so every year drives the
/// soil with the same forcing while the soil itself keeps accumulating
/// history across the year boundary.
///
/// Cloning the boundary copies the marched state, so a cloned model starts
/// from the same ground state as its original (the clone does not share the
/// march with the original).
#[derive(Debug)]
pub struct GroundCoupledFloorBoundary {
    /// The ground-domain solver (configuration only — the solver itself is
    /// stateless; the march lives in `state`).
    solver: crate::physics::ground_conduction::GroundSolver,
    /// The marched ground-domain state.
    state: std::sync::Mutex<GroundCoupledFloorState>,
    /// Lateral extent of the ground domain [m] (the building footprint width).
    domain_length_m: f64,
    /// Lateral position the ground temperature is read at [m].
    readout_x_m: f64,
    /// Depth below the floor underside the ground temperature is read at [m].
    readout_depth_m: f64,
    /// Soil surface film coefficient towards the ground-surface temperature
    /// [W/(m²·K)].
    surface_film_w_m2k: f64,
    /// Annual mean ground-surface temperature [°C].
    surface_mean_c: f64,
    /// Annual amplitude of the ground-surface temperature [°C].
    surface_amplitude_c: f64,
    /// Phase of the annual ground-surface temperature cycle [h].
    surface_phase_hours: f64,
    /// Undisturbed deep-ground temperature [°C] — the fallback value when the
    /// ground solve fails.
    fallback_temperature_c: f64,
}

impl Clone for GroundCoupledFloorBoundary {
    fn clone(&self) -> Self {
        let state = match self.state.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };
        Self {
            solver: self.solver.clone(),
            state: std::sync::Mutex::new(state),
            domain_length_m: self.domain_length_m,
            readout_x_m: self.readout_x_m,
            readout_depth_m: self.readout_depth_m,
            surface_film_w_m2k: self.surface_film_w_m2k,
            surface_mean_c: self.surface_mean_c,
            surface_amplitude_c: self.surface_amplitude_c,
            surface_phase_hours: self.surface_phase_hours,
            fallback_temperature_c: self.fallback_temperature_c,
        }
    }
}

impl GroundCoupledFloorBoundary {
    /// Build a ground-coupled floor boundary for a footprint of
    /// `footprint_width_m` metres.
    ///
    /// # Errors
    ///
    /// Returns the ground-domain configuration error when the specification
    /// does not describe a solvable domain (non-positive conductivities,
    /// coupling depth outside the domain, ...).
    pub fn new(
        spec: &fluxion_core::ashrae_cases::GroundCoupledFloorSpec,
        footprint_width_m: f64,
    ) -> Result<Self, crate::physics::ground_conduction::GroundError> {
        use crate::physics::ground_conduction::{
            GridSegment, GroundDomainConfig, GroundSolver, SurfaceCondition, TopCondition,
            TopSegment,
        };

        let domain_length_m = footprint_width_m;
        let coupling_depth_m = spec.coupling_depth_m;
        let domain_depth_m = spec.domain_depth_m;

        // The coupling depth must be a vertical cell edge: the top layer is
        // uniform up to it, the graded layer runs from it to the domain
        // bottom.
        let top_cells = ((coupling_depth_m / GROUND_TOP_CELL_M).ceil() as usize).max(2);
        let lateral_segments = vec![GridSegment::Uniform {
            start_m: 0.0,
            end_m: domain_length_m,
            cells: GROUND_LATERAL_CELLS,
        }];
        let vertical_segments = vec![
            GridSegment::Uniform {
                start_m: 0.0,
                end_m: coupling_depth_m,
                cells: top_cells,
            },
            GridSegment::Geometric {
                start_m: coupling_depth_m,
                end_m: domain_depth_m,
                first_cell_m: GROUND_GRADED_FIRST_CELL_M,
                growth: GROUND_GRADED_GROWTH,
                max_cell_m: GROUND_GRADED_MAX_CELL_M,
            },
        ];
        let top_segments = vec![TopSegment {
            x_start_m: 0.0,
            x_end_m: domain_length_m,
            condition: TopCondition::Robin {
                film_coefficient_w_m2k: spec.surface_film_coefficient_w_m2k,
                ambient_temperature_c: spec.surface_temperature_mean_c,
            },
        }];

        let config = GroundDomainConfig {
            domain_length_m,
            domain_depth_m,
            // The domain carries the soil only: the floor construction stays
            // in the zone model, so the "slab" layer of the ground domain is
            // the soil layer between the floor underside and the coupling
            // depth (same material on both sides — the interface is
            // invisible to the temperature field).
            slab_x0_m: 0.0,
            slab_x1_m: domain_length_m,
            slab_thickness_m: coupling_depth_m,
            slab_conductivity_w_mk: spec.soil_conductivity_w_mk,
            ground_conductivity_w_mk: spec.soil_conductivity_w_mk,
            slab_volumetric_heat_capacity_j_m3k: spec.soil_volumetric_heat_capacity_j_m3k,
            ground_volumetric_heat_capacity_j_m3k: spec.soil_volumetric_heat_capacity_j_m3k,
            lateral_segments,
            vertical_segments,
            top_segments,
            deep_ground: SurfaceCondition::Dirichlet(spec.deep_ground_temperature_c),
            lateral_boundary: SurfaceCondition::Dirichlet(spec.lateral_far_field_temperature_c),
            max_iterations: GROUND_MAX_ITERATIONS,
            tolerance: GROUND_TOLERANCE,
        };

        let solver = GroundSolver::new(config)?;
        let transient = solver.initial_transient_state(spec.surface_temperature_mean_c)?;
        let state = GroundCoupledFloorState {
            transient,
            abs_hour: 0,
        };

        Ok(Self {
            solver,
            state: std::sync::Mutex::new(state),
            domain_length_m,
            readout_x_m: 0.5 * domain_length_m,
            readout_depth_m: coupling_depth_m,
            surface_film_w_m2k: spec.surface_film_coefficient_w_m2k,
            surface_mean_c: spec.surface_temperature_mean_c,
            surface_amplitude_c: spec.surface_temperature_amplitude_c,
            surface_phase_hours: spec.surface_temperature_phase_hours,
            fallback_temperature_c: spec.deep_ground_temperature_c,
        })
    }

    /// The ground-surface temperature of the annual cycle at `hour` [°C].
    pub fn surface_temperature_c(&self, hour: usize) -> f64 {
        let phase = 2.0 * PI * (hour as f64 - self.surface_phase_hours) / HOURS_PER_YEAR;
        self.surface_mean_c + self.surface_amplitude_c * phase.cos()
    }

    /// The top-surface boundary the ground domain sees at `hour`.
    fn top_segments(&self, hour: usize) -> Vec<crate::physics::ground_conduction::TopSegment> {
        use crate::physics::ground_conduction::{TopCondition, TopSegment};
        vec![TopSegment {
            x_start_m: 0.0,
            x_end_m: self.domain_length_m,
            condition: TopCondition::Robin {
                film_coefficient_w_m2k: self.surface_film_w_m2k,
                ambient_temperature_c: self.surface_temperature_c(hour),
            },
        }]
    }

    /// The soil temperature at the coupling plane [°C] for the marched state.
    fn coupled_temperature_c(&self, state: &GroundCoupledFloorState) -> f64 {
        state
            .transient
            .temperature_at(self.readout_x_m, self.readout_depth_m)
    }

    /// March the ground domain to `hour_of_year`, keeping the soil state
    /// across the year boundary (Issue #4794).
    ///
    /// The soil is marched **forward only**: the internal clock
    /// ([`GroundCoupledFloorState::abs_hour`]) never rewinds and the
    /// transient state is never reset, so a run that crosses the end of the
    /// year keeps the ground history the soil has accumulated instead of
    /// restarting from a uniform field.
    ///
    /// `hour_of_year` is an hour of the year (0-8759). The march stops at the
    /// *next occurrence* of that hour, so a request that goes "backwards" —
    /// the year wrap, or a genuinely out-of-order query — is served by
    /// marching forward to the following occurrence of the same hour. The
    /// old guard (a decreasing hour re-initialised the soil) is gone: it
    /// conflated the year wrap with an out-of-order call and destroyed the
    /// multi-year ground state.
    ///
    /// The annual ground-surface forcing is 8760-periodic, so the drive is
    /// sampled at `abs_hour % 8760` and the same forcing sequence repeats
    /// every year while the soil keeps its own history.
    fn advance_to(&self, state: &mut GroundCoupledFloorState, hour_of_year: usize) -> bool {
        let hours_per_year = HOURS_PER_YEAR as usize;
        let current_hour = (state.abs_hour % hours_per_year as u64) as usize;
        // Forward distance to the next occurrence of `hour_of_year`.
        let steps = (hour_of_year + hours_per_year - current_hour) % hours_per_year;
        for _ in 0..steps {
            // The step from `abs_hour` to `abs_hour + 1` is driven by the
            // ground-surface temperature of the hour it starts at.
            let drive_hour = (state.abs_hour % hours_per_year as u64) as usize;
            let top = self.top_segments(drive_hour);
            if self
                .solver
                .step_transient_with_top(&mut state.transient, GROUND_TIME_STEP_S, &top)
                .is_err()
            {
                return false;
            }
            state.abs_hour += 1;
        }
        true
    }
}

impl GroundTemperature for GroundCoupledFloorBoundary {
    fn clone_box(&self) -> Box<dyn GroundTemperature> {
        Box::new(self.clone())
    }

    /// Ground temperature at the coupling plane [°C].
    ///
    /// `hour_of_year` is an hour of the year; the annual cycle wraps with the
    /// modulo below, so a multi-year driver may hand over the absolute hour
    /// (8760 is the first hour of the second year). The soil state is
    /// marched forward across that wrap and never reset (Issue #4794).
    fn ground_temperature(&self, hour_of_year: usize) -> f64 {
        let hour = hour_of_year % (HOURS_PER_YEAR as usize);
        let mut state = match self.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if !self.advance_to(&mut state, hour) {
            // The ground solve failed: fall back to the undisturbed ground
            // temperature rather than corrupting the zone balance.
            return self.fallback_temperature_c;
        }
        self.coupled_temperature_c(&state)
    }
}

/// Build the Section 7 floor boundary of a case specification.
///
/// * A case that opts into the ground-coupled floor path
///   (`CaseSpec::ground_coupled_floor`) gets a [`GroundCoupledFloorBoundary`]
///   driven by the 2-D ground-conduction domain.
/// * Every other case keeps the Issue #746 behaviour: the constant ground
///   temperature of `CaseSpec::ground_temperature_c`, and no boundary at all
///   when the case carries no ground temperature (the model default then
///   applies).
///
/// The ground-coupled path falls back to the constant-temperature boundary
/// when the ground-domain configuration is invalid, so a malformed
/// specification degrades to the legacy behaviour instead of failing the run.
pub fn floor_ground_boundary(
    spec: &fluxion_core::ashrae_cases::CaseSpec,
) -> Option<Box<dyn GroundTemperature>> {
    if let Some(coupled) = spec.ground_coupled_floor.as_ref() {
        let footprint_width_m = spec.geometry.first().map(|g| g.width).unwrap_or(0.0);
        if let Ok(boundary) = GroundCoupledFloorBoundary::new(coupled, footprint_width_m) {
            return Some(Box::new(boundary));
        }
    }
    spec.ground_temperature_c
        .map(|t| Box::new(ConstantGroundTemperature::new(t)) as Box<dyn GroundTemperature>)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_ground_temperature() {
        let ground = ConstantGroundTemperature::new(10.0);

        // Temperature should be constant regardless of time
        assert_eq!(ground.ground_temperature(0), 10.0);
        assert_eq!(ground.ground_temperature(4380), 10.0); // Mid-year
        assert_eq!(ground.ground_temperature(8759), 10.0); // End of year
    }

    #[test]
    fn test_constant_temperature_getters() {
        let mut ground = ConstantGroundTemperature::new(15.0);
        assert_eq!(ground.temperature(), 15.0);

        ground.set_temperature(12.0);
        assert_eq!(ground.temperature(), 12.0);
        assert_eq!(ground.ground_temperature(100), 12.0);
    }

    #[test]
    fn test_ashrae_140_default() {
        // ASHRAE 140 specifies 10°C constant ground temperature
        let ground = ConstantGroundTemperature::new(10.0);
        assert_eq!(ground.ground_temperature(0), 10.0);
    }

    #[test]
    fn test_dynamic_ground_temperature_creation() {
        let ground = DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.07);

        assert_eq!(ground.t_mean(), 11.0);
        assert_eq!(ground.t_amplitude(), 12.0);
        assert_eq!(ground.depth(), 1.0);
        assert_eq!(ground.diffusivity(), 0.07);
    }

    #[test]
    #[should_panic(expected = "Depth must be positive")]
    fn test_dynamic_ground_temperature_zero_depth() {
        DynamicGroundTemperature::new(11.0, 12.0, 0.0, 0.07);
    }

    #[test]
    #[should_panic(expected = "Diffusivity must be positive")]
    fn test_dynamic_ground_temperature_zero_diffusivity() {
        DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "Amplitude must be non-negative")]
    fn test_dynamic_ground_temperature_negative_amplitude() {
        DynamicGroundTemperature::new(11.0, -1.0, 1.0, 0.07);
    }

    #[test]
    fn test_dynamic_ground_temperature_varies_with_time() {
        let ground = DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.07);

        let temp_winter = ground.ground_temperature(0); // ~Jan 1
        let temp_summer = ground.ground_temperature(4380); // ~Jul 1

        // Summer should be warmer than winter
        assert!(temp_summer > temp_winter);
    }

    #[test]
    fn test_dynamic_ground_temperature_mean() {
        let ground = DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.07);

        // Average over a full year should approximate the mean
        let mut sum = 0.0;
        for h in (0..8760).step_by(24) {
            sum += ground.ground_temperature(h);
        }
        let avg = sum / 365.0;

        // Should be close to mean (within 0.5°C)
        assert!((avg - ground.t_mean()).abs() < 0.5);
    }

    #[test]
    fn test_dynamic_ground_temperature_damping_factor() {
        // Shallow depth: more variation
        let shallow = DynamicGroundTemperature::new(11.0, 12.0, 0.5, 0.07);
        let damping_shallow = shallow.damping_factor();

        // Deep: less variation
        let deep = DynamicGroundTemperature::new(11.0, 12.0, 2.0, 0.07);
        let damping_deep = deep.damping_factor();

        // Shallow should have higher damping factor (less attenuation)
        assert!(damping_shallow > damping_deep);

        // Both should be between 0 and 1
        assert!(damping_shallow > 0.0 && damping_shallow < 1.0);
        assert!(damping_deep > 0.0 && damping_deep < 1.0);
    }

    #[test]
    fn test_dynamic_ground_temperature_phase_shift() {
        let ground = DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.07);

        let phase_shift = ground.phase_shift();

        // Phase shift should be positive
        assert!(phase_shift > 0.0);

        // At 1m depth, shift should be significant (several days)
        assert!(phase_shift > 10.0);
    }

    #[test]
    fn test_dynamic_ground_temperature_amplitude_reduces_with_depth() {
        let shallow = DynamicGroundTemperature::new(11.0, 12.0, 0.5, 0.07);
        let deep = DynamicGroundTemperature::new(11.0, 12.0, 2.0, 0.07);

        let temp_shallow_min = (0..365)
            .map(|d| shallow.ground_temperature(d * 24))
            .fold(f64::INFINITY, f64::min);
        let temp_shallow_max = (0..365)
            .map(|d| shallow.ground_temperature(d * 24))
            .fold(f64::NEG_INFINITY, f64::max);
        let amp_shallow = temp_shallow_max - temp_shallow_min;

        let temp_deep_min = (0..365)
            .map(|d| deep.ground_temperature(d * 24))
            .fold(f64::INFINITY, f64::min);
        let temp_deep_max = (0..365)
            .map(|d| deep.ground_temperature(d * 24))
            .fold(f64::NEG_INFINITY, f64::max);
        let amp_deep = temp_deep_max - temp_deep_min;

        // Shallow depth should have larger amplitude
        assert!(amp_shallow > amp_deep);
    }

    #[test]
    fn test_dynamic_ground_temperature_high_diffusivity() {
        // High diffusivity: temperature penetrates deeper
        let low_diff = DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.05);
        let high_diff = DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.10);

        let damping_low = low_diff.damping_factor();
        let damping_high = high_diff.damping_factor();

        // High diffusivity should have lower damping (more variation)
        assert!(damping_high > damping_low);
    }

    #[test]
    fn test_ground_temperature_trait_bounds() {
        // Verify that our implementations satisfy Send + Sync
        fn is_send_sync<T: Send + Sync>() {}

        is_send_sync::<ConstantGroundTemperature>();
        is_send_sync::<DynamicGroundTemperature>();

        // Can create boxed trait objects
        let _: Box<dyn GroundTemperature> = Box::new(ConstantGroundTemperature::new(10.0));
        let _: Box<dyn GroundTemperature> =
            Box::new(DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.07));
    }

    #[test]
    fn test_ground_temperature_continuity_across_year() {
        let ground = DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.07);

        // Temperature at end of year should be close to start of next year
        let temp_end = ground.ground_temperature(8759);
        let temp_start = ground.ground_temperature(0);

        // Within 0.1°C
        assert!((temp_end - temp_start).abs() < 0.1);
    }

    // === Issue #1435: MonthlyGroundTemperature ===

    #[test]
    fn test_monthly_ground_temperature_lookup() {
        let monthly = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let ground = MonthlyGroundTemperature::new(monthly);
        // January
        assert_eq!(ground.ground_temperature(0), 1.0);
        assert_eq!(ground.ground_temperature(743), 1.0);
        // February (starts at hour 744)
        assert_eq!(ground.ground_temperature(744), 2.0);
        // June (starts at hour 3624)
        assert_eq!(ground.ground_temperature(3624), 6.0);
        // December (starts at hour 8016)
        assert_eq!(ground.ground_temperature(8016), 12.0);
        assert_eq!(ground.ground_temperature(8759), 12.0);
    }

    #[test]
    fn test_monthly_ground_temperature_from_slice_wrong_length() {
        assert!(MonthlyGroundTemperature::from_slice(&[1.0; 11]).is_none());
        assert!(MonthlyGroundTemperature::from_slice(&[1.0; 13]).is_none());
        let ok = MonthlyGroundTemperature::from_slice(&[1.0; 12]).unwrap();
        assert_eq!(ok.monthly()[0], 1.0);
    }

    #[test]
    fn test_constant_ground_temperature_clone() {
        let ground1 = ConstantGroundTemperature::new(10.0);
        let ground2 = ground1.clone();

        assert_eq!(ground1.temperature(), ground2.temperature());
        assert_eq!(ground2.ground_temperature(100), 10.0);
    }

    #[test]
    fn test_dynamic_ground_temperature_clone() {
        let ground1 = DynamicGroundTemperature::new(11.0, 12.0, 1.0, 0.07);
        let ground2 = ground1.clone();

        assert_eq!(ground1.t_mean(), ground2.t_mean());
        assert_eq!(ground1.t_amplitude(), ground2.t_amplitude());
        assert_eq!(ground1.depth(), ground2.depth());
        assert_eq!(ground1.diffusivity(), ground2.diffusivity());

        // Same temperature at same time
        assert_eq!(
            ground1.ground_temperature(1000),
            ground2.ground_temperature(1000)
        );
    }

    // === Issue #864: Distribution function tests ===

    #[test]
    fn test_distribute_opaque_solar_equal_irradiance() {
        let result = distribute_opaque_solar_gains(300.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0);
        assert!((result.phi_m_wall - 100.0).abs() < 1e-10);
        assert!((result.phi_m_roof - 100.0).abs() < 1e-10);
        assert!((result.phi_m_floor - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_distribute_opaque_solar_weighted() {
        let result = distribute_opaque_solar_gains(600.0, 20.0, 20.0, 10.0, 50.0, 200.0, 0.0);
        assert!((result.phi_m_wall - 120.0).abs() < 1e-10);
        assert!((result.phi_m_roof - 480.0).abs() < 1e-10);
        assert!(result.phi_m_floor.abs() < 1e-10);
    }

    #[test]
    fn test_distribute_opaque_solar_zero_irradiance_fallback() {
        let result = distribute_opaque_solar_gains(300.0, 20.0, 10.0, 10.0, 0.0, 0.0, 0.0);
        assert!((result.phi_m_wall - 150.0).abs() < 1e-10);
        assert!((result.phi_m_roof - 75.0).abs() < 1e-10);
        assert!((result.phi_m_floor - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_distribute_opaque_solar_zero_total() {
        let result = distribute_opaque_solar_gains(0.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0);
        assert!(result.phi_m_wall.abs() < 1e-10);
        assert!(result.phi_m_roof.abs() < 1e-10);
        assert!(result.phi_m_floor.abs() < 1e-10);
    }

    // === Issue #4775: ground-coupled Section 7 floor boundary ===

    /// Soil and surface conditions of the ground-coupled floor acceptance
    /// tests below.
    ///
    /// The values are representative mid-latitude soil (k = 1 W/(m·K),
    /// rho*c = 2 MJ/(m³·K)) and a ground surface with a 10 K annual swing
    /// peaking in mid-July (hour 4400 of the year). The undisturbed deep and
    /// lateral far-field ground is 10 °C.
    fn ground_coupled_floor_spec() -> fluxion_core::ashrae_cases::GroundCoupledFloorSpec {
        fluxion_core::ashrae_cases::GroundCoupledFloorSpec {
            soil_conductivity_w_mk: 1.0,
            soil_volumetric_heat_capacity_j_m3k: 2.0e6,
            domain_depth_m: 10.0,
            coupling_depth_m: 3.0,
            deep_ground_temperature_c: 10.0,
            lateral_far_field_temperature_c: 10.0,
            surface_film_coefficient_w_m2k: 10.0,
            surface_temperature_mean_c: 10.0,
            surface_temperature_amplitude_c: 10.0,
            surface_temperature_phase_hours: 4400.0,
        }
    }

    /// The same ground-coupled floor specification with *uniform* ground
    /// temperatures: ground surface, deep ground and lateral far field all
    /// at 10 °C and no annual swing, so the soil column stays uniform and
    /// the boundary value read at the coupling depth is known analytically.
    fn uniform_ground_coupled_floor_spec() -> fluxion_core::ashrae_cases::GroundCoupledFloorSpec {
        fluxion_core::ashrae_cases::GroundCoupledFloorSpec {
            soil_conductivity_w_mk: 1.0,
            soil_volumetric_heat_capacity_j_m3k: 2.0e6,
            domain_depth_m: 10.0,
            coupling_depth_m: 3.0,
            deep_ground_temperature_c: 10.0,
            lateral_far_field_temperature_c: 10.0,
            surface_film_coefficient_w_m2k: 10.0,
            surface_temperature_mean_c: 10.0,
            surface_temperature_amplitude_c: 0.0,
            surface_temperature_phase_hours: 0.0,
        }
    }

    /// A minimal Section 7 case specification (8 m x 6 m footprint, one zone)
    /// with the ASHRAE 140 constant ground temperature of 9.4 °C.
    fn section_7_case_spec() -> fluxion_core::ashrae_cases::CaseSpec {
        use fluxion_core::ashrae_case_builder::CaseBuilder;
        CaseBuilder::new()
            .with_dimensions(8.0, 6.0, 2.7)
            .with_ground_temperature(9.4)
            .build()
            .expect("Section 7 case specification must build")
    }

    /// Acceptance test A (Issue #4775): a case that opts into the
    /// ground-coupled floor boundary gets a floor boundary temperature that
    /// is driven by the ground domain — it varies over the year and differs
    /// from the constant-temperature path.
    #[test]
    fn test_ground_coupled_floor_boundary_is_ground_driven() {
        use fluxion_core::ashrae_case_builder::CaseBuilder;

        let mut spec = section_7_case_spec();
        spec.ground_coupled_floor = Some(ground_coupled_floor_spec());

        let boundary = floor_ground_boundary(&spec).expect("floor boundary must exist");
        let constant = ConstantGroundTemperature::new(9.4);

        // Sample the year every 24 h. The hours are strictly increasing, so
        // the ground-domain march never has to wrap around; the year wrap
        // itself is covered by
        // `test_ground_coupled_floor_boundary_keeps_soil_state_across_year_wrap`.
        let mut series = Vec::new();
        let mut differs_from_constant = false;
        for hour in (0..8760).step_by(24) {
            let t = boundary.ground_temperature(hour);
            if (t - constant.ground_temperature(hour)).abs() > 1.0e-6 {
                differs_from_constant = true;
            }
            series.push(t);
        }

        let min = series.iter().copied().fold(f64::INFINITY, f64::min);
        let max = series.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max - min > 0.5,
            "ground-coupled floor boundary must vary over the year \
             (min {min:.3} °C, max {max:.3} °C)"
        );
        assert!(
            differs_from_constant,
            "ground-coupled floor boundary must differ from the constant-temperature path"
        );

        // The opt-in must be what selects the ground-driven boundary: the
        // same case without the opt-in keeps the constant path.
        let mut constant_spec = spec.clone();
        constant_spec.ground_coupled_floor = None;
        let constant_boundary =
            floor_ground_boundary(&constant_spec).expect("floor boundary must exist");
        assert_eq!(
            constant_boundary.ground_temperature(0),
            constant.ground_temperature(0)
        );
        assert_ne!(
            constant_boundary.ground_temperature(0),
            boundary.ground_temperature(0)
        );

        // Sanity: the builder carries the opt-in into the specification.
        let built = CaseBuilder::new()
            .with_dimensions(8.0, 6.0, 2.7)
            .with_ground_temperature(9.4)
            .with_ground_coupled_floor(ground_coupled_floor_spec())
            .build()
            .expect("Section 7 case specification must build");
        assert!(built.ground_coupled_floor.is_some());
    }

    /// Regression test (Issue #4794): the ground-coupled floor boundary keeps
    /// its soil state across the year wrap.
    ///
    /// The boundary is driven through the absolute hours 8750..8770, which
    /// crosses the 8760-hour year boundary, and the returned temperatures
    /// are compared against the equivalent **unwrapped** continuous march:
    /// the same solver, stepped from the initial state with the annual drive
    /// sampled at `abs_hour % 8760`.
    ///
    /// Before the fix the boundary re-initialised the soil at the wrap
    /// (`solver.initial_transient_state`), so the driven temperatures
    /// restarted the year-one sequence instead of continuing the march, and
    /// the multi-year ground state was destroyed.
    #[test]
    fn test_ground_coupled_floor_boundary_keeps_soil_state_across_year_wrap() {
        use crate::physics::ground_conduction::{TopCondition, TopSegment};

        /// First absolute hour driven by the test (the last ten hours of the
        /// first year).
        const FIRST_HOUR: usize = 8750;
        /// Last absolute hour driven by the test, exclusive (the first ten
        /// hours of the second year).
        const LAST_HOUR: usize = 8770;

        let spec = ground_coupled_floor_spec();
        let boundary = GroundCoupledFloorBoundary::new(&spec, 8.0)
            .expect("ground-coupled floor boundary must build");

        // Drive the boundary across the wrap. The trait method reduces the
        // argument to its hour of the year, so 8760 is the first hour of the
        // second year.
        let mut driven = Vec::new();
        for hour in FIRST_HOUR..LAST_HOUR {
            driven.push(boundary.ground_temperature(hour));
        }

        // Reference: the same continuous march without a year boundary — the
        // soil is stepped from the initial state and the annual drive is
        // sampled at `abs_hour % 8760`.
        let mut transient = boundary
            .solver
            .initial_transient_state(spec.surface_temperature_mean_c)
            .expect("initial transient state must build");
        let mut reference = Vec::new();
        for abs_hour in 0..LAST_HOUR {
            let drive_hour = abs_hour % (HOURS_PER_YEAR as usize);
            let top = vec![TopSegment {
                x_start_m: 0.0,
                x_end_m: boundary.domain_length_m,
                condition: TopCondition::Robin {
                    film_coefficient_w_m2k: boundary.surface_film_w_m2k,
                    ambient_temperature_c: boundary.surface_temperature_c(drive_hour),
                },
            }];
            boundary
                .solver
                .step_transient_with_top(&mut transient, GROUND_TIME_STEP_S, &top)
                .expect("ground-domain step must succeed");
            let represented = abs_hour + 1;
            if represented >= FIRST_HOUR && represented < LAST_HOUR {
                reference
                    .push(transient.temperature_at(boundary.readout_x_m, boundary.readout_depth_m));
            }
        }

        assert_eq!(
            driven.len(),
            reference.len(),
            "the driven and the reference series must cover the same hours"
        );
        for (offset, (got, want)) in driven.iter().zip(reference.iter()).enumerate() {
            let abs_hour = FIRST_HOUR + offset;
            assert!(
                (got - want).abs() < 1.0e-9,
                "ground temperature at absolute hour {abs_hour} must match the \
                 unwrapped continuous march (got {got:.9} °C, want {want:.9} °C)"
            );
        }

        // The soil state must not be reset at the wrap: the first hour of
        // the second year must not repeat the first hour of the first year
        // (the pre-fix boundary re-initialised the soil, so both were the
        // untouched initial field).
        let year_one_start = GroundCoupledFloorBoundary::new(&spec, 8.0)
            .expect("ground-coupled floor boundary must build")
            .ground_temperature(0);
        let year_two_start = driven[8760 - FIRST_HOUR];
        assert!(
            (year_two_start - year_one_start).abs() > 1.0e-3,
            "the soil state must not be reset at the year wrap: hour 0 of the \
             second year ({year_two_start:.6} °C) must differ from hour 0 of the \
             first year ({year_one_start:.6} °C)"
        );

        // No state discontinuity at the wrap beyond the normal step-to-step
        // change: the step across the boundary is of the same order as the
        // neighbouring hourly steps.
        let mut max_step = 0.0_f64;
        for pair in driven.windows(2) {
            max_step = max_step.max((pair[1] - pair[0]).abs());
        }
        let wrap_step = (driven[8760 - FIRST_HOUR] - driven[8759 - FIRST_HOUR]).abs();
        assert!(
            wrap_step <= 2.0 * max_step,
            "the step across the year wrap ({wrap_step:.9} K) must stay in the \
             range of the normal step-to-step change ({max_step:.9} K)"
        );
    }

    /// Acceptance test B (Issue #4775): a case that does NOT opt into the
    /// ground-coupled floor path produces exactly the same floor ground
    /// temperature as the constant-temperature boundary — bit for bit.
    #[test]
    fn test_floor_ground_boundary_without_opt_in_is_bit_identical() {
        let spec = section_7_case_spec();
        assert!(
            spec.ground_coupled_floor.is_none(),
            "the default case must not opt into the ground-coupled floor path"
        );

        let boundary = floor_ground_boundary(&spec).expect("floor boundary must exist");
        let constant = ConstantGroundTemperature::new(9.4);

        for hour in [0usize, 1, 24, 500, 1000, 4380, 8000, 8759] {
            assert_eq!(
                boundary.ground_temperature(hour),
                constant.ground_temperature(hour),
                "floor ground temperature must be bit-identical at hour {hour}"
            );
        }
    }

    /// Acceptance test (Issue #4789): the steady-state floor heat flux of the
    /// opt-in ground-coupled path equals the analytic series-resistance value
    /// of the traced path.
    ///
    /// Uniform driving temperatures (ground surface, deep ground and lateral
    /// far field all at 10 °C, no annual swing) leave the soil column
    /// uniform, so the boundary temperature the coupled path reads at the
    /// coupling depth is known analytically: `T_g = 10 °C`. The traced series
    /// path of the coupled path is then
    ///
    /// ```text
    /// T_s ──[ R_si + R_materials ]──> T_g
    /// ```
    ///
    /// — the interior film plus the floor construction, with the ground
    /// domain carrying the ground-side resistance — so the floor flux must be
    /// `A_floor × (T_g − T_s) / (R_si + R_materials)`. A double-counted
    /// ground resistance (the lumped `R_g = 0.17 m²K/W` of Issue #588 inside
    /// the floor U-value) shrinks the flux by the factor
    /// `(R_si + R_materials) / (R_si + R_materials + R_g)`, so this test
    /// discriminates between the two accountings.
    #[test]
    fn test_ground_coupled_floor_flux_matches_analytic_series() {
        use crate::physics::cta::VectorField;
        use crate::sim::thermal_model_core::ThermalModel;

        const T_GROUND_C: f64 = 10.0;
        const T_SURFACE_C: f64 = 20.0;

        let mut spec = section_7_case_spec();
        spec.ground_coupled_floor = Some(uniform_ground_coupled_floor_spec());

        let model = ThermalModel::<VectorField>::from_spec(&spec);

        // The uniform ground forcing leaves the soil column uniform, so the
        // value the coupled path reads at the coupling depth is the forcing
        // temperature itself.
        let t_ground = model.conduction.ground_temperature.ground_temperature(0);
        assert!(
            (t_ground - T_GROUND_C).abs() < 1.0e-9,
            "uniform ground forcing must leave the soil uniform, got {t_ground} °C"
        );

        let floor_area = spec.geometry[0].floor_area();
        let h_tr_floor = model.conduction.h_tr_floor[0];

        // Analytic series value of the traced path: the interior film and the
        // floor construction; the ground domain carries the ground-side
        // resistance.
        let r_si = 1.0
            / fluxion_core::construction::SurfaceType::interior_film_coeff(
                fluxion_core::construction::SurfaceType::Floor,
            );
        let r_materials: f64 = spec
            .construction
            .floor
            .layers
            .iter()
            .map(|l| l.r_value())
            .sum();
        let q_analytic = floor_area * (t_ground - T_SURFACE_C) / (r_si + r_materials);

        // The floor flux of the coupled path.
        let q_model = h_tr_floor * (t_ground - T_SURFACE_C);

        let tolerance = 1.0e-9 * q_analytic.abs();
        assert!(
            (q_model - q_analytic).abs() <= tolerance,
            "ground-coupled floor flux must equal the analytic series value \
             ({q_analytic} W); got {q_model} W (h_tr_floor {h_tr_floor} W/K)"
        );

        // The double-counted accounting would be lower by the factor
        // (R_si + R_materials) / (R_si + R_materials + R_g) — make sure the
        // test would actually catch it.
        let q_double_counted = floor_area * (t_ground - T_SURFACE_C)
            / (r_si + r_materials + fluxion_core::construction::GROUND_COUPLING_RESISTANCE_M2K_W);
        assert!(
            (q_model - q_double_counted).abs() > tolerance,
            "the test must discriminate against the double-counted accounting \
             ({q_double_counted} W)"
        );
    }

    /// The constant-temperature path keeps the Issue #588 ground-coupling
    /// resistance inside the floor U-value: the Issue #4789 correction is
    /// confined to the opt-in ground-coupled path, so every non-opt-in case
    /// stays bit-identical.
    #[test]
    fn test_constant_temperature_path_keeps_ground_coupling_resistance() {
        use crate::physics::cta::VectorField;
        use crate::sim::thermal_model_core::ThermalModel;

        let spec = section_7_case_spec();
        assert!(
            spec.ground_coupled_floor.is_none(),
            "the default case must not opt into the ground-coupled floor path"
        );

        let model = ThermalModel::<VectorField>::from_spec(&spec);

        let floor_area = spec.geometry[0].floor_area();
        let r_si = 1.0
            / fluxion_core::construction::SurfaceType::interior_film_coeff(
                fluxion_core::construction::SurfaceType::Floor,
            );
        let r_materials: f64 = spec
            .construction
            .floor
            .layers
            .iter()
            .map(|l| l.r_value())
            .sum();
        let expected = (1.0
            / (r_si + r_materials + fluxion_core::construction::GROUND_COUPLING_RESISTANCE_M2K_W))
            * floor_area;

        assert_eq!(
            model.conduction.h_tr_floor[0], expected,
            "the constant-temperature path must keep the Issue #588 ground-coupling \
             resistance inside the floor U-value"
        );
    }

    #[test]
    fn test_distribute_radiative_gains_by_area() {
        let (w, r, f) = distribute_radiative_gains(400.0, 30.0, 10.0, 10.0);
        assert!((w - 240.0).abs() < 1e-10);
        assert!((r - 80.0).abs() < 1e-10);
        assert!((f - 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_distribute_radiative_gains_zero() {
        let (w, r, f) = distribute_radiative_gains(0.0, 10.0, 10.0, 10.0);
        assert!(w.abs() < 1e-10);
        assert!(r.abs() < 1e-10);
        assert!(f.abs() < 1e-10);
    }

    #[test]
    fn test_distribute_radiative_gains_zero_area() {
        let (w, r, f) = distribute_radiative_gains(400.0, 0.0, 0.0, 0.0);
        assert!(w.abs() < 1e-10);
        assert!(r.abs() < 1e-10);
        assert!(f.abs() < 1e-10);
    }
}
