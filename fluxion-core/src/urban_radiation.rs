//! Urban radiation configuration for city-scale thermal modeling.
//!
//! Defines the configuration for inter-building longwave radiative exchange
//! using the Nusselt analog view factor method from `fluxion-city`.
//!
//! This module is intentionally dependency-free — it contains only plain data
//! types that can be serialized and passed to the actual `fluxion-city`
//! `UrbanRadiationSolver` in the main `fluxion` crate.

/// Configuration for an urban radiation simulation at city scale.
//
/// This struct holds the geometric and material parameters needed to construct
/// a `fluxion_city::sparse::UrbanRadiationSolver`. It is the pure-data
/// interface between the main `fluxion` engine and the `fluxion-city` crate,
/// enabling the solver to be constructed without embedding `fluxion-city`
/// types in `fluxion-core` (Issue #2369).
///
/// # Construction
///
/// The config is typically built programmatically from building geometry
/// (e.g. from an urban graph or CAD model) and then passed to
/// `PhysicsSurfaceFluxProvider::set_urban_radiation` which constructs
/// the actual solver in the `fluxion` crate where `fluxion-city` is available.
///
/// # Example
///
/// ```
/// use fluxion_core::urban_radiation::UrbanRadiationConfig;
///
/// let config = UrbanRadiationConfig::new()
///     .add_wall(30.0, 3.0, 0.0)   // area=30m², height=3m, x=0m
///     .add_wall(30.0, 3.0, 13.0)  // area=30m², height=3m, x=13m (3m gap)
///     .with_ground_area(200.0)
///     .with_emissivity(0.9);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct UrbanRadiationConfig {
    /// Per-wall parameters: (area_m2, height_m, x_position_m).
    walls: Vec<(f64, f64, f64)>,
    /// Ground plane area [m²].
    ground_area_m2: f64,
    /// Longwave emissivity of building surfaces [-].
    emissivity: f64,
}

impl UrbanRadiationConfig {
    /// Create a new empty config.
    pub fn new() -> Self {
        Self {
            walls: Vec::new(),
            ground_area_m2: 0.0,
            emissivity: 0.9,
        }
    }

    /// Add a wall surface to the urban configuration.
    ///
    /// # Arguments
    /// * `area_m2` - Wall surface area [m²]
    /// * `height_m` - Wall height [m]
    /// * `x_position_m` - Wall x position for inter-building spacing [m]
    pub fn add_wall(mut self, area_m2: f64, height_m: f64, x_position_m: f64) -> Self {
        self.walls.push((area_m2, height_m, x_position_m));
        self
    }

    /// Set the ground plane area [m²].
    pub fn with_ground_area(mut self, area_m2: f64) -> Self {
        self.ground_area_m2 = area_m2;
        self
    }

    /// Set the surface emissivity [-] (default: 0.9).
    pub fn with_emissivity(mut self, emissivity: f64) -> Self {
        self.emissivity = emissivity;
        self
    }

    /// Number of wall surfaces configured.
    pub fn num_walls(&self) -> usize {
        self.walls.len()
    }

    /// Reference to the wall parameters: (area_m2, height_m, x_position_m).
    pub fn walls(&self) -> &[(f64, f64, f64)] {
        &self.walls
    }

    /// Ground plane area [m²].
    pub fn ground_area_m2(&self) -> f64 {
        self.ground_area_m2
    }

    /// Surface emissivity [-].
    pub fn emissivity(&self) -> f64 {
        self.emissivity
    }
}

impl Default for UrbanRadiationConfig {
    fn default() -> Self {
        Self::new()
    }
}
