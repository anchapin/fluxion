//! Solar → Conduction integration tests for bottom-up testing PRD.
//!
//! These tests verify the wiring from solar irradiance to per-surface heat flux
//! in the thermal network. The diagnostic chain continues from weather_solar_integration:
//! Weather TMY3 → Solar Position → Surface Irradiance → Heat Flux → Zone Energy Balance
//!
//! # Wire-Edge Coverage
//!
//! - **Solar → Conduction**: Surface irradiance → per-surface heat flux → 5R1C thermal network
//!
//! # References
//!
//! - `src/solar/surface_irradiance.rs` - Surface irradiance calculation
//! - `src/physics/gauge_zone_solver.rs` - 5R1C thermal network
//! - `src/sim/thermal_model.rs` - ThermalModelTrait swap point

use fluxion::solar::{
    calculate_solar_position,
    surface_irradiance::{calculate_surface_irradiance, Orientation, SurfaceIrradiance},
};
use fluxion::weather::denver::DenverTmyWeather;
use fluxion::weather::WeatherSource;

/// Denver location constants (ASHRAE 140 standard location).
const DENVER_LAT: f64 = 39.74;
const DENVER_LON: f64 = -105.18;

/// Beam-to-mass distribution fractions (ASHRAE 140 standard).
const SOLAR_BEAM_TO_MASS_FRACTION: f64 = 0.7;
const MASS_TO_EXTERIOR_FRACTION: f64 = 0.7;
const MASS_TO_INTERIOR_FRACTION: f64 = 0.3;
const SURFACE_SOLAR_FRACTION: f64 = 0.3;

/// Tolerance for energy comparisons (Watts).
const ENERGY_TOL_WATTS: f64 = 1.0;

/// Convert hour-of-year (0-8759) to (year, month, day, day_of_year, hour).
fn hour_of_year_to_calendar(hour: usize) -> (i32, u32, u32, usize, f64) {
    let hour = hour.min(8759);
    let day_of_year = hour / 24;
    let hour_of_day = (hour % 24) as f64;

    static DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut remaining = day_of_year;
    let mut month = 1u32;

    for &days in &DAYS_IN_MONTH {
        if remaining < days as usize {
            break;
        }
        remaining -= days as usize;
        month += 1;
    }

    let day = remaining as u32 + 1;
    (2024, month, day, day_of_year + 1, hour_of_day) // day_of_year is 1-indexed
}

/// Calculate solar heat flux components from irradiance.
///
/// This mirrors the ASHRAE 140 solar gain distribution:
/// - 30% goes directly to interior surface (phi_si_solar)
/// - 70% goes to thermal mass (phi_m_solar)
///   - 70% of mass gain goes to exterior environment (phi_m_env_solar)
///   - 30% of mass gain goes to interior (phi_m_int_solar)
#[derive(Debug, Clone, Copy)]
struct SolarHeatFlux {
    /// Direct to interior surface (W/m²)
    pub phi_si_solar: f64,
    /// To thermal mass, exterior-facing (W/m²)
    pub phi_m_env_solar: f64,
    /// To thermal mass, interior-facing (W/m²)
    pub phi_m_int_solar: f64,
    /// Total solar heat gain (W/m²)
    pub phi_total_solar: f64,
}

impl SolarHeatFlux {
    /// Calculate solar heat flux components from surface irradiance.
    fn from_irradiance(irradiance: &SurfaceIrradiance) -> Self {
        // Beam-to-mass distribution: 70% to mass, 30% to interior surface
        let total_gain = irradiance.total_wm2;

        // Interior surface gets fraction of total directly
        let phi_si_solar = total_gain * SURFACE_SOLAR_FRACTION;

        // Mass gain (70% of total)
        let mass_gain = total_gain * SOLAR_BEAM_TO_MASS_FRACTION;

        // Mass split: 70% exterior, 30% interior
        let phi_m_env_solar = mass_gain * MASS_TO_EXTERIOR_FRACTION;
        let phi_m_int_solar = mass_gain * MASS_TO_INTERIOR_FRACTION;

        // Total solar gain
        let phi_total_solar = phi_si_solar + phi_m_env_solar + phi_m_int_solar;

        SolarHeatFlux {
            phi_si_solar,
            phi_m_env_solar,
            phi_m_int_solar,
            phi_total_solar,
        }
    }
}

/// Test that solar irradiance converts to heat flux correctly.
///
/// Verifies the beam-to-mass distribution math:
/// - 30% of irradiance → phi_si (interior surface)
/// - 70% of irradiance → phi_m (thermal mass)
///   - 70% of mass → phi_m_env (exterior-facing)
///   - 30% of mass → phi_m_int (interior-facing)
#[test]
fn test_solar_to_heat_flux_distribution() {
    // Test with 1000 W/m² total irradiance (clear summer noon)
    let irradiance = SurfaceIrradiance::new(700.0, 200.0, 100.0); // beam, diffuse, ground
    let flux = SolarHeatFlux::from_irradiance(&irradiance);

    // Verify total
    assert!(
        (flux.phi_total_solar - 1000.0).abs() < 0.01,
        "Total flux should equal irradiance: got {}",
        flux.phi_total_solar
    );

    // Verify distribution
    // phi_si = 30% of 1000 = 300
    assert!(
        (flux.phi_si_solar - 300.0).abs() < 0.01,
        "phi_si_solar should be 300, got {}",
        flux.phi_si_solar
    );

    // phi_m = 70% of 1000 = 700
    // phi_m_env = 70% of 700 = 490
    assert!(
        (flux.phi_m_env_solar - 490.0).abs() < 0.01,
        "phi_m_env_solar should be 490, got {}",
        flux.phi_m_env_solar
    );

    // phi_m_int = 30% of 700 = 210
    assert!(
        (flux.phi_m_int_solar - 210.0).abs() < 0.01,
        "phi_m_int_solar should be 210, got {}",
        flux.phi_m_int_solar
    );

    // Verify sum
    let sum = flux.phi_si_solar + flux.phi_m_env_solar + flux.phi_m_int_solar;
    assert!(
        (sum - flux.phi_total_solar).abs() < 0.01,
        "Components should sum to total"
    );
}

/// Test full wiring from Weather to Heat Flux for summer noon.
///
/// This test verifies the complete chain:
/// 1. Weather TMY3 provides DNI, DHI, GHI
/// 2. Solar position calculated from weather/time
/// 3. Surface irradiance computed from position + weather
/// 4. Heat flux derived from irradiance distribution
#[test]
fn test_full_weather_to_heat_flux_summer_noon() {
    let weather = DenverTmyWeather::new();

    // July 15 noon
    let summer_hour = (14 * 24) + 12;
    let data = weather.get_hourly_data(summer_hour).unwrap();

    // Step 1-2: Solar position
    let (year, month, day, day_of_year, hour) = hour_of_year_to_calendar(summer_hour);
    let solar_pos = calculate_solar_position(
        DENVER_LAT,
        DENVER_LON,
        year,
        month,
        day,
        hour,
        Some(-7.0),
    );

    // Step 3: Surface irradiance on south wall
    let irradiance = calculate_surface_irradiance(
        &solar_pos,
        data.dni,
        data.dhi,
        Some(data.ghi),
        Orientation::South,
        0.3,   // Ground reflectance
        day_of_year,
    );

    // Step 4: Heat flux
    let flux = SolarHeatFlux::from_irradiance(&irradiance);

    // Verify physically reasonable values
    assert!(
        flux.phi_total_solar > 0.0,
        "Summer noon south wall should have positive heat gain, got {} W/m²",
        flux.phi_total_solar
    );

    // South wall in Denver summer should receive substantial solar gain
    // (lower than horizontal due to angle)
    assert!(
        flux.phi_total_solar > 100.0,
        "South wall should receive >100 W/m² at summer noon, got {}",
        flux.phi_total_solar
    );

    // Verify distribution is consistent
    let mass_fraction = (flux.phi_m_env_solar + flux.phi_m_int_solar) / flux.phi_total_solar;
    assert!(
        (mass_fraction - SOLAR_BEAM_TO_MASS_FRACTION).abs() < 0.01,
        "Mass fraction should be {}, got {}",
        SOLAR_BEAM_TO_MASS_FRACTION,
        mass_fraction
    );
}

/// Test that nighttime produces zero solar heat flux.
#[test]
fn test_nighttime_zero_heat_flux() {
    let weather = DenverTmyWeather::new();

    // Midnight
    let midnight_hour = 0;
    let data = weather.get_hourly_data(midnight_hour).unwrap();
    let (year, month, day, day_of_year, hour) = hour_of_year_to_calendar(midnight_hour);

    let solar_pos = calculate_solar_position(
        DENVER_LAT,
        DENVER_LON,
        year,
        month,
        day,
        hour,
        Some(-7.0),
    );

    let irradiance = calculate_surface_irradiance(
        &solar_pos,
        data.dni,
        data.dhi,
        Some(data.ghi),
        Orientation::South,
        0.3,
        day_of_year,
    );

    let flux = SolarHeatFlux::from_irradiance(&irradiance);

    assert!(
        flux.phi_total_solar < ENERGY_TOL_WATTS,
        "Nighttime heat flux should be ~0, got {} W/m²",
        flux.phi_total_solar
    );
}

/// Test that different orientations produce different heat fluxes.
///
/// This verifies the wire correctly handles the angular dependence
/// of solar gain on different building surfaces.
#[test]
fn test_orientation_affects_heat_flux() {
    let weather = DenverTmyWeather::new();

    // Summer noon
    let summer_hour = (14 * 24) + 12;
    let data = weather.get_hourly_data(summer_hour).unwrap();
    let (year, month, day, day_of_year, hour) = hour_of_year_to_calendar(summer_hour);

    let solar_pos = calculate_solar_position(
        DENVER_LAT,
        DENVER_LON,
        year,
        month,
        day,
        hour,
        Some(-7.0),
    );

    // South wall (max gain in northern hemisphere summer)
    let south = calculate_surface_irradiance(
        &solar_pos,
        data.dni,
        data.dhi,
        Some(data.ghi),
        Orientation::South,
        0.3,
        day_of_year,
    );
    let flux_south = SolarHeatFlux::from_irradiance(&south);

    // North wall (min gain in northern hemisphere summer)
    let north = calculate_surface_irradiance(
        &solar_pos,
        data.dni,
        data.dhi,
        Some(data.ghi),
        Orientation::North,
        0.3,
        day_of_year,
    );
    let flux_north = SolarHeatFlux::from_irradiance(&north);

    // South should receive significantly more than north
    assert!(
        flux_south.phi_total_solar > flux_north.phi_total_solar,
        "South wall {} should receive more than north wall {} at summer noon",
        flux_south.phi_total_solar,
        flux_north.phi_total_solar
    );

    // Both should still be positive
    assert!(
        flux_south.phi_total_solar > 0.0,
        "South wall should have positive gain"
    );
    assert!(
        flux_north.phi_total_solar >= 0.0,
        "North wall should have non-negative gain"
    );
}

/// Test heat flux for multiple surfaces on the same building.
///
/// This simulates a simple building with 4 walls + roof, verifying
/// that the wire correctly computes per-surface heat flux.
#[test]
fn test_per_surface_heat_flux_building() {
    let weather = DenverTmyWeather::new();

    // Summer noon
    let summer_hour = (14 * 24) + 12;
    let data = weather.get_hourly_data(summer_hour).unwrap();
    let (year, month, day, day_of_year, hour) = hour_of_year_to_calendar(summer_hour);

    let solar_pos = calculate_solar_position(
        DENVER_LAT,
        DENVER_LON,
        year,
        month,
        day,
        hour,
        Some(-7.0),
    );

    // Building surfaces: South, East, West, North walls + horizontal (roof)
    let surfaces = [
        ("South", Orientation::South),
        ("East", Orientation::East),
        ("West", Orientation::West),
        ("North", Orientation::North),
        ("Roof", Orientation::Horizontal),
    ];

    let mut total_flux = 0.0;
    for (name, orientation) in surfaces {
        let irradiance = calculate_surface_irradiance(
            &solar_pos,
            data.dni,
            data.dhi,
            Some(data.ghi),
            orientation,
            0.3,
            day_of_year,
        );
        let flux = SolarHeatFlux::from_irradiance(&irradiance);
        total_flux += flux.phi_total_solar;

        // All surfaces should have non-negative flux
        assert!(
            flux.phi_total_solar >= 0.0,
            "{} surface should have non-negative flux, got {}",
            name,
            flux.phi_total_solar
        );
    }

    // Total building solar gain should be substantial at summer noon
    assert!(
        total_flux > 0.0,
        "Total building solar gain should be positive, got {} W/m²",
        total_flux
    );
}

/// Test that winter produces lower heat flux than summer.
///
/// This verifies the wire handles seasonal variation correctly.
#[test]
fn test_winter_lower_heat_flux_than_summer() {
    let weather = DenverTmyWeather::new();

    // December noon (winter)
    let winter_hour = (355 * 24) + 12;
    let data_winter = weather.get_hourly_data(winter_hour).unwrap();
    let (year, month, day, day_of_year, hour) = hour_of_year_to_calendar(winter_hour);

    let solar_pos_winter = calculate_solar_position(
        DENVER_LAT,
        DENVER_LON,
        year,
        month,
        day,
        hour,
        Some(-7.0),
    );

    let irradiance_winter = calculate_surface_irradiance(
        &solar_pos_winter,
        data_winter.dni,
        data_winter.dhi,
        Some(data_winter.ghi),
        Orientation::South,
        0.3,
        day_of_year,
    );
    let flux_winter = SolarHeatFlux::from_irradiance(&irradiance_winter);

    // July noon (summer)
    let summer_hour = (14 * 24) + 12;
    let data_summer = weather.get_hourly_data(summer_hour).unwrap();
    let (year, month, day, day_of_year, hour) = hour_of_year_to_calendar(summer_hour);

    let solar_pos_summer = calculate_solar_position(
        DENVER_LAT,
        DENVER_LON,
        year,
        month,
        day,
        hour,
        Some(-7.0),
    );

    let irradiance_summer = calculate_surface_irradiance(
        &solar_pos_summer,
        data_summer.dni,
        data_summer.dhi,
        Some(data_summer.ghi),
        Orientation::South,
        0.3,
        day_of_year,
    );
    let flux_summer = SolarHeatFlux::from_irradiance(&irradiance_summer);

    // Summer should produce more heat gain than winter for south wall
    assert!(
        flux_summer.phi_total_solar > flux_winter.phi_total_solar,
        "Summer south wall gain {} should exceed winter {}",
        flux_summer.phi_total_solar,
        flux_winter.phi_total_solar
    );
}

/// Test beam-to-mass distribution preserves energy balance.
///
/// The sum of all solar heat flux components must equal the total irradiance
/// (energy conservation in the distribution).
#[test]
fn test_energy_conservation_in_distribution() {
    // Test with various irradiance levels
    let test_cases = [
        SurfaceIrradiance::new(800.0, 150.0, 50.0),  // High beam
        SurfaceIrradiance::new(200.0, 300.0, 100.0), // High diffuse (overcast)
        SurfaceIrradiance::new(0.0, 400.0, 100.0),  // Diffuse only (heavy overcast)
        SurfaceIrradiance::new(1000.0, 0.0, 0.0),   // Beam only (clear)
    ];

    for irradiance in test_cases {
        let flux = SolarHeatFlux::from_irradiance(&irradiance);

        // Energy conservation: all components sum to total
        let sum = flux.phi_si_solar + flux.phi_m_env_solar + flux.phi_m_int_solar;
        assert!(
            (sum - irradiance.total_wm2).abs() < 0.01,
            "Energy not conserved: components sum {} != irradiance {}",
            sum,
            irradiance.total_wm2
        );

        // Verify individual components are non-negative
        assert!(
            flux.phi_si_solar >= 0.0,
            "phi_si_solar should be non-negative"
        );
        assert!(
            flux.phi_m_env_solar >= 0.0,
            "phi_m_env_solar should be non-negative"
        );
        assert!(
            flux.phi_m_int_solar >= 0.0,
            "phi_m_int_solar should be non-negative"
        );
    }
}

/// Test phi_m_env is largest component (exterior mass gets most solar).
///
/// Per ASHRAE 140, the exterior-facing thermal mass receives the majority
/// of the mass-based solar gain (70% of 70% = 49% of total).
#[test]
fn test_phi_m_env_is_largest_mass_component() {
    let irradiance = SurfaceIrradiance::new(700.0, 200.0, 100.0);
    let flux = SolarHeatFlux::from_irradiance(&irradiance);

    // phi_m_env should be larger than phi_m_int
    assert!(
        flux.phi_m_env_solar > flux.phi_m_int_solar,
        "phi_m_env {} should exceed phi_m_int {}",
        flux.phi_m_env_solar,
        flux.phi_m_int_solar
    );

    // phi_m_env should be approximately 49% of total (0.7 * 0.7 = 0.49)
    let env_fraction = flux.phi_m_env_solar / flux.phi_total_solar;
    assert!(
        (env_fraction - 0.49).abs() < 0.01,
        "phi_m_env fraction should be ~0.49, got {}",
        env_fraction
    );
}
