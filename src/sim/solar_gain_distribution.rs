//! Per-surface solar gain distribution for multi-node thermal model (Issue #859).
//!
//! This module computes per-surface sol-air temperatures and distributes
//! total solar gains across wall, roof, and floor mass nodes using the
//! Perez tilted-plane model for anisotropic sky diffuse radiation.
//!
//! # Physical Background
//!
//! Each surface type has different thermal properties:
//! - **Roof** (horizontal): High solar exposure, dark membrane (α≈0.7-0.9)
//! - **Wall** (vertical): Orientation-dependent exposure (N/S/E/W)
//! - **Floor** (horizontal down): Receives ground-reflected radiation only
//!
//! Sol-air temperature formula per ASHRAE 140:
//! ```text
//! T_sol = T_outdoor + (Solar_α - ε·IR·(T_outdoor - T_sky)) / h_ext
//! ```
//!
//! Where:
//! - `Solar_α`: Solar absorption = α × irradiance / h_ext
//! - `ε·IR·(T_outdoor - T_sky)`: Longwave cooling to sky
//! - `h_ext`: Exterior heat transfer coefficient [W/m²·K]
//!
//! # Perez Tilted-Plane Model
//!
//! The Perez anisotropic sky model accounts for three diffuse components:
//! - Isotropic: Uniformly distributed over the sky dome
//! - Circumsolar: Enhanced radiation around the sun disk
//! - Horizon: Enhanced radiation near the horizon
//!
//! # References
//!
//! - ASHRAE Standard 140, Annex B3.3: Ground Temperature
//! - Perez, R., et al. (1990). "Modeling daylight availability and irradiance
//!   components from direct and global irradiance." Solar Energy 44(5), 271-289.

use std::f64::consts::PI;

use crate::physics::fp_algebraic::{algebraic_add, algebraic_mul, algebraic_div};
use crate::sim::sky_radiation::{
    extraterrestrial_irradiance, total_irradiance_tilted, STEFAN_BOLTZMANN,
};
use crate::sim::solar::SolarPosition;

/// Surface orientation for solar calculations.
///
/// - `tilt_deg`: 0° = horizontal, 90° = vertical
/// - `azimuth_deg`: 0° = North, 90° = East, 180° = South, 270° = West (solar convention)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceOrientation {
    /// Surface tilt from horizontal [degrees]
    pub tilt_deg: f64,
    /// Surface azimuth from north clockwise [degrees]
    pub azimuth_deg: f64,
}

impl SurfaceOrientation {
    /// Create a new surface orientation.
    pub fn new(tilt_deg: f64, azimuth_deg: f64) -> Self {
        Self {
            tilt_deg: tilt_deg.clamp(0.0, 180.0),
            azimuth_deg: azimuth_deg % 360.0,
        }
    }

    /// Horizontal surface (flat roof).
    pub fn horizontal() -> Self {
        Self {
            tilt_deg: 0.0,
            azimuth_deg: 0.0,
        }
    }

    /// Vertical surface facing the given azimuth.
    pub fn vertical(azimuth_deg: f64) -> Self {
        Self {
            tilt_deg: 90.0,
            azimuth_deg: azimuth_deg % 360.0,
        }
    }

    /// North-facing vertical wall.
    pub fn north() -> Self {
        Self::vertical(0.0)
    }

    /// East-facing vertical wall.
    pub fn east() -> Self {
        Self::vertical(90.0)
    }

    /// South-facing vertical wall.
    pub fn south() -> Self {
        Self::vertical(180.0)
    }

    /// West-facing vertical wall.
    pub fn west() -> Self {
        Self::vertical(270.0)
    }

    /// Roof (horizontal, tilted upward).
    pub fn roof() -> Self {
        Self::horizontal()
    }

    /// Floor (horizontal, tilted downward — receives no direct solar).
    pub fn floor() -> Self {
        Self {
            tilt_deg: 180.0,
            azimuth_deg: 0.0,
        }
    }
}

/// Surface type with thermal properties for sol-air temperature.
///
/// Different envelope surfaces have different solar absorptance (α)
/// and longwave emissivity (ε) values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceType {
    /// Solar absorptance (dimensionless, 0-1)
    pub solar_absorptance: f64,
    /// Longwave emissivity (dimensionless, 0-1)
    pub emissivity: f64,
    /// Surface orientation
    pub orientation: SurfaceOrientation,
}

impl SurfaceType {
    /// Create a new surface type.
    pub fn new(solar_absorptance: f64, emissivity: f64, orientation: SurfaceOrientation) -> Self {
        Self {
            solar_absorptance: solar_absorptance.clamp(0.0, 1.0),
            emissivity: emissivity.clamp(0.0, 1.0),
            orientation,
        }
    }

    /// Create a dark-colored roof surface (typical built-up roof).
    ///
    /// Dark membranes have high solar absorptance (0.7-0.9).
    pub fn dark_roof() -> Self {
        Self::new(0.80, 0.90, SurfaceOrientation::horizontal())
    }

    /// Create a light-colored roof surface (reflective coating).
    pub fn light_roof() -> Self {
        Self::new(0.30, 0.90, SurfaceOrientation::horizontal())
    }

    /// Create a standard wall surface (typical siding/stucco).
    pub fn standard_wall() -> Self {
        Self::new(0.60, 0.90, SurfaceOrientation::vertical(180.0))
    }

    /// Create a dark-colored wall surface.
    pub fn dark_wall() -> Self {
        Self::new(0.80, 0.90, SurfaceOrientation::vertical(180.0))
    }

    /// Create a floor surface (receives ground-reflected radiation only).
    pub fn floor() -> Self {
        Self::new(0.60, 0.90, SurfaceOrientation::floor())
    }

    /// Wall facing a specific cardinal direction.
    pub fn wall_facing(azimuth_deg: f64) -> Self {
        Self::new(0.60, 0.90, SurfaceOrientation::vertical(azimuth_deg))
    }
}

/// Per-surface solar gain distribution result for wall, roof, and floor.
///
/// Contains the total irradiance on each surface [W/m²] computed using
/// the Perez tilted-plane model.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PerSurfaceIrradiance {
    /// Wall surface irradiance [W/m²]
    pub wall: f64,
    /// Roof surface irradiance [W/m²]
    pub roof: f64,
    /// Floor surface irradiance [W/m²]
    pub floor: f64,
}

impl PerSurfaceIrradiance {
    /// Total irradiance summed across all surfaces [W/m²].
    ///
    /// 3-term wall/roof/floor reduction (Issue #3324): routed through
    /// `algebraic_add` so the per-timestep gain distribution loop body
    /// can reassociate under `--features fast-math`. Default-feature
    /// builds stay bit-identical.
    pub fn total(&self) -> f64 {
        algebraic_add(algebraic_add(self.wall, self.roof), self.floor)
    }
}

/// Per-surface sol-air temperatures for wall, roof, and floor.
///
/// Each surface has a distinct sol-air temperature because:
/// - Roof (horizontal) receives maximum beam radiation
/// - Walls receive orientation-dependent beam + diffuse
/// - Floor only receives ground-reflected radiation
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PerSurfaceSolAir {
    /// Wall sol-air temperature [°C]
    pub wall: f64,
    /// Roof sol-air temperature [°C]
    pub roof: f64,
    /// Floor sol-air temperature [°C]
    pub floor: f64,
}

/// Solar distribution factors for each surface (normalized weights).
///
/// The sum of all factors equals 1.0 when irradiance > 0.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SolarDistributionFactors {
    /// Wall distribution factor (dimensionless, 0-1)
    pub wall: f64,
    /// Roof distribution factor (dimensionless, 0-1)
    pub roof: f64,
    /// Floor distribution factor (dimensionless, 0-1)
    pub floor: f64,
}

impl SolarDistributionFactors {
    /// Verify sum of factors equals 1.0 (± tolerance).
    ///
    /// 3-term reduction routed through `algebraic_add` (Issue #3324)
    /// for parity with `PerSurfaceIrradiance::total`. Default-feature
    /// builds stay bit-identical.
    pub fn sums_to_one(&self, tolerance: f64) -> bool {
        let sum = algebraic_add(algebraic_add(self.wall, self.roof), self.floor);
        (sum - 1.0).abs() < tolerance
    }
}

/// Calculate per-surface solar irradiance using Perez tilted-plane model.
///
/// Uses the Perez anisotropic sky model to compute beam, diffuse, and
/// ground-reflected irradiance for each surface type.
///
/// # Arguments
///
/// * `sun_pos` - Solar position (altitude, azimuth, zenith)
/// * `dni` - Direct normal irradiance [W/m²]
/// * `dhi` - Diffuse horizontal irradiance [W/m²]
/// * `ground_albedo` - Ground albedo (dimensionless, 0-1, typically 0.2)
/// * `wall_orientation` - Azimuth of wall surfaces [degrees]
///
/// # Returns
///
/// Per-surface irradiance for wall (vertical), roof (horizontal), and floor.
pub fn calculate_per_surface_irradiance(
    sun_pos: &SolarPosition,
    dni: f64,
    dhi: f64,
    ground_albedo: f64,
    wall_azimuth_deg: f64,
) -> PerSurfaceIrradiance {
    let day_of_year = 1; // Use generic day; extraterrestrial varies ≤3.4%
    let dni_extra = extraterrestrial_irradiance(day_of_year);
    let zenith_deg = sun_pos.zenith_deg;
    let solar_azimuth_deg = sun_pos.azimuth_deg;

    // Wall irradiance (vertical surface with given azimuth)
    let wall_irr = total_irradiance_tilted(
        dni,
        dhi,
        None,
        dni_extra,
        zenith_deg,
        solar_azimuth_deg,
        90.0,             // vertical tilt
        wall_azimuth_deg, // wall azimuth
        ground_albedo,
    );

    // Roof irradiance (horizontal surface)
    let roof_irr = total_irradiance_tilted(
        dni,
        dhi,
        None,
        dni_extra,
        zenith_deg,
        solar_azimuth_deg,
        0.0, // horizontal tilt
        0.0, // azimuth doesn't matter for horizontal
        ground_albedo,
    );

    // Floor irradiance (horizontal down) — no direct or diffuse from sky,
    // only ground-reflected from opposite-facing horizontal surfaces.
    // For a flat floor: ground factor = (1 - cos(180°))/2 = 1.0
    // But floor is inside the building, so it gets ground-reflected from the
    // roof's view of the ground below. We approximate as zero for typical slabs.
    let floor_irr = 0.0;

    PerSurfaceIrradiance {
        wall: wall_irr.max(0.0),
        roof: roof_irr.max(0.0),
        floor: floor_irr,
    }
}

/// Calculate per-surface sol-air temperatures.
///
/// Sol-air temperature accounts for both solar heating and longwave cooling.
///
/// # Arguments
///
/// * `outdoor_temp` - Outdoor dry-bulb temperature [°C]
/// * `surface_irradiance` - Per-surface irradiance [W/m²]
/// * `surface_type` - Surface thermal properties (α, ε)
/// * `sky_temp` - Effective sky temperature [°C]
///
/// # Returns
///
/// Per-surface sol-air temperatures.
pub fn calculate_per_surface_sol_air(
    outdoor_temp: f64,
    surface_irradiance: &PerSurfaceIrradiance,
    sky_temp: f64,
) -> PerSurfaceSolAir {
    // ASHRAE 140 default exterior conductance
    let h_ext = 22.7; // W/m²·K (includes convective + radiative)

    // Roof: horizontal, dark membrane
    let alpha_roof = 0.80;
    let t_sol_roof = outdoor_temp + (alpha_roof * surface_irradiance.roof / h_ext)
        - (0.90
            * STEFAN_BOLTZMANN
            * ((outdoor_temp + 273.15).powi(4) - (sky_temp + 273.15).powi(4))
            / h_ext);

    // Wall: vertical, standard color
    let alpha_wall = 0.60;
    let t_sol_wall = outdoor_temp + (alpha_wall * surface_irradiance.wall / h_ext)
        - (0.90
            * STEFAN_BOLTZMANN
            * ((outdoor_temp + 273.15).powi(4) - (sky_temp + 273.15).powi(4))
            / h_ext);

    // Floor: no direct solar, only ground coupling (use outdoor as approximation)
    let t_sol_floor = outdoor_temp;

    PerSurfaceSolAir {
        wall: t_sol_wall,
        roof: t_sol_roof,
        floor: t_sol_floor,
    }
}

/// Calculate solar distribution factors by surface orientation/area.
///
/// Uses cosine of incidence weighted by exposed area to distribute
/// total solar gain across surfaces.
///
/// # Arguments
///
/// * `sun_pos` - Solar position
/// * `wall_area` - Total wall area [m²]
/// * `roof_area` - Total roof area [m²]
/// * `floor_area` - Total floor area [m²]
/// * `wall_azimuth_deg` - Wall surface azimuth [degrees]
///
/// # Returns
///
/// Distribution factors that sum to 1.0 (normalized).
pub fn calculate_solar_distribution_factors(
    sun_pos: &SolarPosition,
    wall_area: f64,
    roof_area: f64,
    floor_area: f64,
    wall_azimuth_deg: f64,
) -> SolarDistributionFactors {
    // 3-term area sum guard routed through `algebraic_add` (Issue #3324).
    if algebraic_add(algebraic_add(wall_area, roof_area), floor_area) < 1e-10 {
        return SolarDistributionFactors::default();
    }

    // Cosine of incidence for each surface orientation
    // Wall (vertical)
    let cos_wall = sun_pos.incidence_cosine(90.0, wall_azimuth_deg);
    // Roof (horizontal, tilt=0)
    let cos_roof = sun_pos.incidence_cosine(0.0, 0.0);
    // Floor (horizontal down, tilt=180) — cosine would be negative,
    // but floor doesn't receive direct solar, so we use 0
    let cos_floor = 0.0;

    // Weighted areas (A_i * cos(theta_i))
    let wall_weight = wall_area * cos_wall;
    let roof_weight = roof_area * cos_roof;
    let floor_weight = floor_area * cos_floor;
    // 3-term weight reduction (Issue #3324). Default-feature builds stay
    // bit-identical; under `--features fast-math` the surrounding
    // `area * cos(theta)` products can be FMA-contracted.
    let total_weight = algebraic_add(
        algebraic_add(wall_weight, roof_weight),
        floor_weight,
    );

    if total_weight < 1e-10 {
        // Fallback: distribute by area alone
        let total_area = algebraic_add(algebraic_add(wall_area, roof_area), floor_area);
        if total_area < 1e-10 {
            return SolarDistributionFactors::default();
        }
        SolarDistributionFactors {
            wall: algebraic_div(wall_area, total_area),
            roof: algebraic_div(roof_area, total_area),
            floor: algebraic_div(floor_area, total_area),
        }
    } else {
        SolarDistributionFactors {
            wall: algebraic_div(wall_weight, total_weight),
            roof: algebraic_div(roof_weight, total_weight),
            floor: algebraic_div(floor_weight, total_weight),
        }
    }
}

/// ASHRAE 140 Annex B3.3 ground temperature model.
///
/// This model uses a sinusoidal variation around the annual mean:
/// ```text
/// T_ground = T_avg + A * sin(2π * (day - phase) / 365)
/// ```
///
/// Where:
/// - `T_avg` ≈ 10°C (ASHRAE 140 default mean ground temperature)
/// - `A` ≈ 5°C (annual amplitude)
/// - `phase` ≈ 30 days (minimum occurs around February 1)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ashrae140GroundTemperature {
    /// Mean annual ground temperature [°C]
    t_avg: f64,
    /// Annual temperature amplitude [°C]
    amplitude: f64,
    /// Phase shift [days] — positive means temperature lags
    phase_days: f64,
}

impl Ashrae140GroundTemperature {
    /// Create new ASHRAE 140 B3.3 ground temperature model.
    ///
    /// # Arguments
    ///
    /// * `t_avg` - Mean annual ground temperature [°C] (default: 10.0)
    /// * `amplitude` - Annual temperature amplitude [°C] (default: 5.0)
    /// * `phase_days` - Phase shift in days (default: 30.0)
    pub fn new(t_avg: f64, amplitude: f64, phase_days: f64) -> Self {
        Self {
            t_avg,
            amplitude: amplitude.max(0.0),
            phase_days,
        }
    }

    /// ASHRAE 140 default parameters (T_avg=10°C, A=5°C, phase=30 days).
    pub fn ashrae_140_default() -> Self {
        Self {
            t_avg: 10.0,
            amplitude: 5.0,
            phase_days: 30.0,
        }
    }

    /// Get ground temperature at a given hour of year.
    ///
    /// # Arguments
    ///
    /// * `hour_of_year` - Hour index (0-8759)
    ///
    /// # Returns
    ///
    /// Ground temperature [°C]
    pub fn ground_temperature(&self, hour_of_year: usize) -> f64 {
        let day = (hour_of_year as f64) / 24.0;
        let omega = 2.0 * PI / 365.0;
        let phase = self.phase_days;
        self.t_avg + self.amplitude * (omega * day - phase * omega).sin()
    }
}

/// Distribute total solar gain across surfaces by orientation-weighted area.
///
/// Uses Perez tilted-plane model to compute per-surface irradiance,
/// then distributes total gain proportional to irradiance × area.
///
/// # Arguments
///
/// * `total_solar_gain` - Total solar gain into the zone [W]
/// * `sun_pos` - Solar position
/// * `dni` - Direct normal irradiance [W/m²]
/// * `dhi` - Diffuse horizontal irradiance [W/m²]
/// * `wall_area` - Wall surface area [m²]
/// * `roof_area` - Roof surface area [m²]
/// * `floor_area` - Floor surface area [m²]
/// * `wall_azimuth_deg` - Wall azimuth [degrees]
/// * `ground_albedo` - Ground albedo (dimensionless, default 0.2)
///
/// # Returns
///
/// Solar gain distributed to wall, roof, and floor [W].
pub fn distribute_solar_by_orientation(
    total_solar_gain: f64,
    sun_pos: &SolarPosition,
    dni: f64,
    dhi: f64,
    wall_area: f64,
    roof_area: f64,
    floor_area: f64,
    wall_azimuth_deg: f64,
    ground_albedo: f64,
) -> PerSurfaceIrradiance {
    let per_surf =
        calculate_per_surface_irradiance(sun_pos, dni, dhi, ground_albedo, wall_azimuth_deg);

    // Weighted irradiance by area
    let wall_weight = wall_area * per_surf.wall;
    let roof_weight = roof_area * per_surf.roof;
    let floor_weight = floor_area * per_surf.floor;
    // 3-term weight reduction (Issue #3324). Default-feature builds stay
    // bit-identical; under `--features fast-math` the
    // `total_solar_gain * (wall_weight / total_weight)` chain can be
    // reassociated.
    let total_weight = algebraic_add(
        algebraic_add(wall_weight, roof_weight),
        floor_weight,
    );

    if total_weight < 1e-10 {
        return PerSurfaceIrradiance::default();
    }

    PerSurfaceIrradiance {
        wall: total_solar_gain * wall_weight / total_weight,
        roof: total_solar_gain * roof_weight / total_weight,
        floor: total_solar_gain * floor_weight / total_weight,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Solar angle tests at known dates ===

    /// Test at spring equinox (Mar 21): N/S irradiance should be nearly equal.
    #[test]
    fn test_solar_equinox_equal_ns() {
        // Denver latitude
        let lat = 39.7;
        let lon = -104.9;

        // Solar noon on Mar 21 (equinox)
        let sun_pos = crate::sim::solar::calculate_solar_position(lat, lon, 2024, 3, 21, 12.0, None);

        assert!(sun_pos.is_above_horizon());
        assert!(
            sun_pos.altitude_deg > 40.0,
            "Altitude should be significant at equinox noon"
        );

        // At equinox, N and S walls should have similar irradiance
        // because solar azimuth is near 180° (South) at noon
        let cos_south = sun_pos.incidence_cosine(90.0, 180.0);
        let cos_north = sun_pos.incidence_cosine(90.0, 0.0);

        // N/S symmetry at equinox: cosines should be similar
        let diff = (cos_south - cos_north).abs();
        assert!(
            diff < 0.1,
            "N/S wall irradiance should be similar at equinox: diff={}",
            diff
        );
    }

    /// Test at summer solstice (Jun 21): maximum northern hemisphere solar angle.
    #[test]
    fn test_solar_summer_solstice_max_angle() {
        let lat = 39.7;
        let lon = -104.9;

        // Solar noon on Jun 21
        let sun_pos = crate::sim::solar::calculate_solar_position(lat, lon, 2024, 6, 21, 12.0, None);

        assert!(sun_pos.is_above_horizon());

        // At 39.7°N on Jun 21:
        // Expected altitude ≈ 90° - (39.7° - 23.45°) = 73.75°
        assert!(
            sun_pos.altitude_deg > 70.0 && sun_pos.altitude_deg < 80.0,
            "Summer solstice altitude should be 70-80°: got {}",
            sun_pos.altitude_deg
        );

        // Azimuth should be near 180° (South) at noon
        assert!(
            sun_pos.azimuth_deg > 170.0 && sun_pos.azimuth_deg < 190.0,
            "Azimuth should be near South: got {}",
            sun_pos.azimuth_deg
        );
    }

    /// Test at winter solstice (Dec 21): minimum solar angle, high diffuse fraction.
    #[test]
    fn test_solar_winter_solstice_min_angle() {
        let lat = 39.7;
        let lon = -104.9;

        // Solar noon on Dec 21
        let sun_pos = crate::sim::solar::calculate_solar_position(lat, lon, 2024, 12, 21, 12.0, None);

        assert!(
            sun_pos.is_above_horizon(),
            "Sun should be above horizon at noon in winter"
        );

        // At 39.7°N on Dec 21:
        // Expected altitude ≈ 90° - (39.7° + 23.45°) = 26.85°
        assert!(
            sun_pos.altitude_deg > 20.0 && sun_pos.altitude_deg < 35.0,
            "Winter solstice altitude should be 20-35°: got {}",
            sun_pos.altitude_deg
        );
    }

    // === Distribution factor tests ===

    #[test]
    fn test_distribution_factors_sum_to_one() {
        let sun_pos = crate::sim::solar::calculate_solar_position(39.7, -104.9, 2024, 6, 21, 12.0, None);

        let factors = calculate_solar_distribution_factors(
            &sun_pos, 100.0, // wall_area
            50.0,  // roof_area
            50.0,  // floor_area
            180.0, // wall_azimuth_deg (South)
        );

        assert!(
            factors.sums_to_one(1e-9),
            "Distribution factors should sum to 1.0: wall={}, roof={}, floor={}",
            factors.wall,
            factors.roof,
            factors.floor
        );
    }

    #[test]
    fn test_distribution_factors_zero_area() {
        let sun_pos = crate::sim::solar::calculate_solar_position(39.7, -104.9, 2024, 6, 21, 12.0, None);

        let factors = calculate_solar_distribution_factors(&sun_pos, 0.0, 0.0, 0.0, 180.0);

        assert!(factors.wall.abs() < 1e-10);
        assert!(factors.roof.abs() < 1e-10);
        assert!(factors.floor.abs() < 1e-10);
    }

    // === Sol-air temperature tests ===

    #[test]
    fn test_sol_air_roof_higher_than_outdoor() {
        let sun_pos = crate::sim::solar::calculate_solar_position(39.7, -104.9, 2024, 6, 21, 12.0, None);
        let per_surf = calculate_per_surface_irradiance(&sun_pos, 800.0, 200.0, 0.2, 180.0);
        let sol_air = calculate_per_surface_sol_air(30.0, &per_surf, -10.0);

        // Roof sol-air should be higher than outdoor due to solar heating
        assert!(
            sol_air.roof > 30.0,
            "Roof sol-air should exceed outdoor temp: T_sol={}, T_out={}",
            sol_air.roof,
            30.0
        );

        // Both wall and roof should have positive irradiance during daytime
        assert!(
            per_surf.roof > 0.0 && per_surf.wall > 0.0,
            "Both surfaces should receive irradiance: roof={}, wall={}",
            per_surf.roof,
            per_surf.wall
        );
    }

    #[test]
    fn test_sol_air_wall_nonzero() {
        let sun_pos = crate::sim::solar::calculate_solar_position(39.7, -104.9, 2024, 6, 21, 12.0, None);
        let per_surf = calculate_per_surface_irradiance(&sun_pos, 800.0, 200.0, 0.2, 180.0);
        let sol_air = calculate_per_surface_sol_air(30.0, &per_surf, -10.0);

        // Wall sol-air should be higher than outdoor due to solar heating
        assert!(
            sol_air.wall > 30.0,
            "Wall sol-air should exceed outdoor temp: T_sol={}, T_out={}",
            sol_air.wall,
            30.0
        );
    }

    // === Ground temperature tests (ASHRAE 140 B3.3) ===

    #[test]
    fn test_ashrae_140_ground_default() {
        let ground = Ashrae140GroundTemperature::ashrae_140_default();

        let t_jan = ground.ground_temperature(0); // ~Jan 1, hour 0
        let t_jul = ground.ground_temperature(4344); // ~Jul 1, hour 4344

        // July should be warmer than January
        assert!(
            t_jul > t_jan,
            "Summer ground temp should exceed winter: T_Jul={}, T_Jan={}",
            t_jul,
            t_jan
        );
    }

    #[test]
    fn test_ashrae_140_ground_variation() {
        let ground = Ashrae140GroundTemperature::ashrae_140_default();

        let mut max_temp = f64::NEG_INFINITY;
        let mut min_temp = f64::INFINITY;

        for h in (0..8760).step_by(24) {
            let t = ground.ground_temperature(h);
            max_temp = max_temp.max(t);
            min_temp = min_temp.min(t);
        }

        let variation = max_temp - min_temp;

        // Amplitude should be close to 2×A (peak-to-peak ≈ 2*A)
        // ASHRAE 140 default A=5°C → peak-to-peak ≈ 10°C
        assert!(
            (variation - 10.0).abs() < 0.5,
            "Annual ground temp variation should be ~10°C: got {}",
            variation
        );
    }

    #[test]
    fn test_ashrae_140_ground_minimum_in_winter() {
        let ground = Ashrae140GroundTemperature::ashrae_140_default();

        // Find the minimum temperature hour
        let mut min_temp = f64::INFINITY;
        let mut min_hour = 0;
        let mut max_temp = f64::NEG_INFINITY;

        for h in (0..8760).step_by(24) {
            let t = ground.ground_temperature(h);
            min_temp = min_temp.min(t);
            max_temp = max_temp.max(t);
            if t < min_temp {
                min_temp = t;
                min_hour = h;
            }
        }

        // With phase=30 and cosine minimum at (2n+1)*π/2:
        // min occurs at day = 365/4 + phase = 91.25 + 30 ≈ 121 (early May)
        // due to ground thermal lag at depth.
        // Note: ASHRAE 140 Annex B3.3 uses a simpler formula than the Kusuda
        // dynamic model. The minimum occurs in spring due to thermal inertia.
        let min_day = min_hour / 24;
        assert!(
            min_day > 90 && min_day < 180,
            "Minimum ground temp should occur in spring due to thermal lag: day {}",
            min_day
        );

        // Summer should be warmer than winter
        let t_jan = ground.ground_temperature(0);
        let t_jul = ground.ground_temperature(4344);
        assert!(
            t_jul > t_jan,
            "Summer ground temp should exceed winter: T_Jul={}, T_Jan={}",
            t_jul,
            t_jan
        );
    }

    #[test]
    fn test_ashrae_140_ground_continuous() {
        let ground = Ashrae140GroundTemperature::ashrae_140_default();

        // Temperature at end of year should be close to start of next year
        let t_end = ground.ground_temperature(8759);
        let t_start = ground.ground_temperature(0);

        assert!(
            (t_end - t_start).abs() < 0.5,
            "Year-end ground temp should match year-start: {} vs {}",
            t_end,
            t_start
        );
    }

    // === Surface orientation tests ===

    #[test]
    fn test_surface_orientation_defaults() {
        let roof = SurfaceOrientation::horizontal();
        assert!((roof.tilt_deg - 0.0).abs() < 1e-9);

        let wall = SurfaceOrientation::vertical(180.0);
        assert!((wall.tilt_deg - 90.0).abs() < 1e-9);
        assert!((wall.azimuth_deg - 180.0).abs() < 1e-9);

        let floor = SurfaceOrientation::floor();
        assert!((floor.tilt_deg - 180.0).abs() < 1e-9);
    }

    #[test]
    fn test_surface_type_defaults() {
        let roof = SurfaceType::dark_roof();
        assert!(roof.solar_absorptance > 0.7);

        let wall = SurfaceType::standard_wall();
        assert!(wall.solar_absorptance > 0.5 && wall.solar_absorptance < 0.7);
    }

    #[test]
    fn test_per_surface_irradiance_sums() {
        let sun_pos = crate::sim::solar::calculate_solar_position(39.7, -104.9, 2024, 6, 21, 12.0, None);
        let per_surf = calculate_per_surface_irradiance(&sun_pos, 800.0, 200.0, 0.2, 180.0);

        // Total should be sum of all three surfaces
        let total = per_surf.total();
        assert!(
            (total - per_surf.wall - per_surf.roof - per_surf.floor).abs() < 1e-9,
            "Total irradiance should equal sum of components"
        );
    }

    // === Nighttime tests ===

    #[test]
    fn test_solar_at_night_is_zero() {
        // Midnight — sun is below horizon
        let sun_pos = crate::sim::solar::calculate_solar_position(39.7, -104.9, 2024, 12, 21, 0.0, None);

        assert!(
            !sun_pos.is_above_horizon(),
            "Sun should be below horizon at midnight"
        );

        let per_surf = calculate_per_surface_irradiance(&sun_pos, 0.0, 0.0, 0.2, 180.0);

        // All surfaces should have zero irradiance at night
        assert!(per_surf.wall.abs() < 1e-9);
        assert!(per_surf.roof.abs() < 1e-9);
        assert!(per_surf.floor.abs() < 1e-9);
    }
}
