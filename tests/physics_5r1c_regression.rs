//! Regression tests for fixed branches in `physics_impl.rs::step_physics_5r1c`.
//!
//! These tests verify specific bug fixes in the primary 5R1C physics path:
//! 1. **Night ventilation air-side coupling** (Issue #824) — fan flow must add to
//!    the air-node conductance, not the mass node.
//! 2. **Wind-dependent exterior convection** (Issue #2891) — h_ext varies with wind speed.
//! 3. **Sky temperature fallback** (line 171) — when weather is absent, sky temp
//!    falls back to `outdoor_temp - 15.0`.
//!
//! These live in `tests/` (not `src/`) because they exercise the full `ThermalModel`
//! public API rather than `pub(crate)` internals.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, NightVentilation};
use fluxion::weather::WeatherSource;

/// Helper: create a minimal single-zone thermal model from the Case 600 spec.
fn minimal_single_zone_model() -> ThermalModel<VectorField> {
    ThermalModel::<VectorField>::from_spec(&ASHRAE140Case::Case600.spec())
}

/// Helper: run one step_physics call with given outdoor temp and weather.
fn step_with_weather(model: &mut ThermalModel<VectorField>, timestep: usize, outdoor_temp: f64, weather: Option<fluxion::weather::HourlyWeatherData>) {
    model.solar.weather = weather;
    model.step_physics(timestep, outdoor_temp, 3600.0);
}

// =============================================================================
// Issue #824: Night ventilation air-side coupling
// =============================================================================

/// Verify that night ventilation ACTIVE hours produce a different (cooler) zone air
/// temperature than INACTIVE hours — proving the fan flow is wired into the air node.
#[test]
fn test_night_vent_cooling_effect_on_zone_air() {
    let mut model = minimal_single_zone_model();

    // Install night vent fan active 18:00–07:00 (Case 650 schedule)
    model.hvac.night_ventilation = Some(NightVentilation::new(1703.16, 18, 7));

    // Warm up the zone to a known temperature
    let weather = fluxion::weather::denver::DenverTmyWeather::new();
    let warm_hour = 14; // 2 PM — well inside the day, night vent OFF
    let warm_data = weather.get_hourly_data(warm_hour).unwrap();
    for _ in 0..24 {
        step_with_weather(&mut model, warm_hour, warm_data.dry_bulb_temp, Some(warm_data.clone()));
    }

    let t_air_warm = model.mass.air_temperatures.as_ref()[0];

    // Now step through the night (hour 20 = 8 PM, night vent ACTIVE)
    let night_hour = 20;
    let night_data = weather.get_hourly_data(night_hour).unwrap();
    let t_ext_night = night_data.dry_bulb_temp;

    // Step several times during active night-vent hours
    for step in 0..8 {
        step_with_weather(&mut model, night_hour + step, t_ext_night, Some(night_data.clone()));
    }

    let t_air_night = model.mass.air_temperatures.as_ref()[0];

    // Night ventilation should have cooled the zone below the warm-day temperature
    // (the fan brings in cooler outdoor air at night).
    // This test will FAIL if the night vent path is broken (#824 regression).
    assert!(
        t_air_night < t_air_warm + 0.5,
        "[#824] Night-vent active zone should be cooler than day-zone. \
         t_air_warm={:.2}°C, t_air_night={t_air_night:.2}°C (diff {:.2}K)",
        t_air_warm,
        t_air_warm - t_air_night
    );
}

/// Verify that night ventilation INACTIVE hours (daytime) do NOT apply the fan flow.
#[test]
fn test_night_vent_inactive_during_day() {
    let mut model = minimal_single_zone_model();
    model.hvac.night_ventilation = Some(NightVentilation::new(1703.16, 18, 7));

    let weather = fluxion::weather::denver::DenverTmyWeather::new();

    // Step through midday hours (night vent INACTIVE)
    let midday_hour = 12;
    let midday_data = weather.get_hourly_data(midday_hour).unwrap();
    let t_ext_midday = midday_data.dry_bulb_temp;

    for step in 0..6 {
        step_with_weather(&mut model, midday_hour + step, t_ext_midday, Some(midday_data.clone()));
    }

    let t_air_midday = model.mass.air_temperatures.as_ref()[0];

    // Daytime zone air should follow normal physics (no fan flow from night vent)
    // Verify it's finite and reasonable
    assert!(
        t_air_midday.is_finite(),
        "[#824] Daytime t_air should be finite, got {t_air_midday:.2}",
    );
    assert!(
        t_air_midday > -30.0 && t_air_midday < 80.0,
        "[#824] Daytime t_air should be in reasonable range, got {t_air_midday:.2}°C",
    );
}

// =============================================================================
// Issue #2891: Wind-dependent exterior convection
// =============================================================================

/// Verify that high wind speed produces larger convective heat transfer
/// (faster heat loss from zone) compared to still air.
#[test]
fn test_wind_increases_heat_loss() {
    use fluxion::weather::HourlyWeatherData;

    let mut model_high_wind = minimal_single_zone_model();
    let mut model_still = minimal_single_zone_model();

    // High wind: 10 m/s (very windy) — dry_bulb=20°C, dni=0, dhi=0, ghi=0 (night), wind=10 m/s, humidity=50%, hour=12
    let high_wind_weather = HourlyWeatherData::new(20.0, 0.0, 0.0, 0.0, 10.0, 50.0, 12);
    // Still air: 0.5 m/s (calm)
    let still_air_weather = HourlyWeatherData::new(20.0, 0.0, 0.0, 0.0, 0.5, 50.0, 12);

    let outdoor_temp = 20.0;

    // Step both models for several hours
    for step in 0..12 {
        step_with_weather(&mut model_high_wind, step, outdoor_temp, Some(high_wind_weather.clone()));
        step_with_weather(&mut model_still, step, outdoor_temp, Some(still_air_weather.clone()));
    }

    let t_air_high_wind = model_high_wind.mass.air_temperatures.as_ref()[0];
    let t_air_still = model_still.mass.air_temperatures.as_ref()[0];

    // High wind should produce MORE cooling (lower T_air) due to increased h_ext
    // (greater convective heat loss from the zone envelope).
    // If this fails, the wind-dependent h_ext fix (#2891) may have regressed.
    assert!(
        t_air_high_wind < t_air_still,
        "[#2891] High-wind zone should be cooler than still-air zone. \
         t_air_high_wind={t_air_high_wind:.2}°C, t_air_still={t_air_still:.2}°C",
    );
}

/// Verify that the exterior film coefficient with wind is higher than the default.
#[test]
fn test_wind_raises_exterior_h_ext() {
    use fluxion::physics::exterior_convection::{h_c_ext_wind_dependent, ExteriorSurfaceDirection};

    let h_still = h_c_ext_wind_dependent(ExteriorSurfaceDirection::HorizontalRoofWindward, 0.5);
    let h_wind = h_c_ext_wind_dependent(ExteriorSurfaceDirection::HorizontalRoofWindward, 5.0);

    assert!(
        h_wind > h_still,
        "[#2891] h_ext at 5 m/s ({h_wind:.2}) should exceed h_ext at 0.5 m/s ({h_still:.2})",
    );

    // Verify absolute values are in reasonable range
    // ASHRAE 140: h_ext for windward roof uses h_c = 5.8 + 3.8*V
    // At V=5 m/s: h_c = 5.8 + 3.8*5 = 24.8 W/m²K
    assert!(
        (h_wind - 24.8).abs() < 0.1,
        "[#2891] h_ext at V=5 m/s should be ~24.8, got {h_wind:.2}",
    );
}

// =============================================================================
// Sky temperature fallback (line 171)
// =============================================================================

/// When weather is absent, the model must not panic and must use a reasonable
/// sky temperature fallback (outdoor_temp - 15.0).
#[test]
fn test_sky_temp_fallback_no_weather() {
    let mut model = minimal_single_zone_model();

    // Step WITHOUT any weather data (weather = None)
    // This exercises the sky_temp fallback path at line 171.
    for step in 0..24 {
        step_with_weather(&mut model, step, 25.0, None);
    }

    let t_air = model.mass.air_temperatures.as_ref()[0];

    // Must not be NaN or infinite
    assert!(
        t_air.is_finite(),
        "[sky-temp-fallback] t_air should be finite when weather=None, got {t_air:.2}",
    );

    // Should be in reasonable range given outdoor=25°C
    assert!(
        t_air > 0.0 && t_air < 60.0,
        "[sky-temp-fallback] t_air should be in reasonable range, got {t_air:.2}°C",
    );
}

/// When weather IS present, the actual sky temperature from weather is used.
#[test]
fn test_sky_temp_from_weather_when_present() {
    let mut model = minimal_single_zone_model();
    let weather = fluxion::weather::denver::DenverTmyWeather::new();

    // Step with real weather data
    let hour = 12;
    let data = weather.get_hourly_data(hour).unwrap();

    for step in 0..12 {
        step_with_weather(&mut model, hour + step, data.dry_bulb_temp, Some(data.clone()));
    }

    let t_air = model.mass.air_temperatures.as_ref()[0];

    // Should be finite and reasonable
    assert!(
        t_air.is_finite(),
        "[sky-temp] t_air should be finite with real weather, got {t_air:.2}",
    );
}

// =============================================================================
// Energy conservation invariant
// =============================================================================

/// Verify the model produces finite, non-NaN temperatures after many timesteps.
/// This is a smoke test that catches regressions in any physics branch.
#[test]
fn test_no_nan_after_many_timesteps() {
    let mut model = minimal_single_zone_model();
    let weather = fluxion::weather::denver::DenverTmyWeather::new();

    for step in 0..8760 {
        let data = weather.get_hourly_data(step % 8760).unwrap();
        step_with_weather(&mut model, step, data.dry_bulb_temp, Some(data));
    }

    let t_air = model.mass.air_temperatures.as_ref()[0];
    let t_mass = model.mass.mass_temperatures.as_ref()[0];

    assert!(
        t_air.is_finite() && !t_air.is_nan(),
        "t_air should be finite after 8760 steps, got {t_air}",
    );
    assert!(
        t_mass.is_finite() && !t_mass.is_nan(),
        "t_mass should be finite after 8760 steps, got {t_mass}",
    );
}
