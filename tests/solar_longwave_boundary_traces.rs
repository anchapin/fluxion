//! Hourly Trace Tests for Solar and Long-wave Boundary Conditions
//!
//! PH-05: Audit solar and long-wave boundary conditions
//!
//! These tests verify the correctness of boundary condition math through
//! hourly traces over representative days.
//!
//! ## Definition of Done
//!
//! - Hourly trace tests exist for solar gains
//! - Hourly trace tests exist for sky temperature handling
//! - Hourly trace tests exist for peak solar timing
//!
//! ## Test Strategy
//!
//! Each trace test verifies a full day's behavior to catch edge cases
//! in the solar/long-wave calculations that may not appear in single
//! point-in-time tests.

use fluxion::sim::sky_radiation::{
    calculate_sky_emissivity_with_clouds, estimate_sky_emissivity, extraterrestrial_irradiance,
    relative_airmass, sol_air_temperature_simple, PerezSkyModel, SkyRadiationExchange,
    SolAirTemperature,
};
use fluxion::sim::solar::{
    calculate_day_of_year, calculate_hourly_solar, calculate_solar_position, WindowProperties,
};
use fluxion::validation::ashrae_140_cases::Orientation;

const DENVER_LAT: f64 = 39.7392;
const DENVER_LON: f64 = -104.9903;
const GROUND_REFLECTANCE: f64 = 0.2;
const TOLERANCE: f64 = 0.001;

#[derive(Debug, Clone, Copy)]
struct SolarTracePoint {
    hour: f64,
    solar_gain_w: f64,
    #[allow(dead_code)]
    total_irradiance_wm2: f64,
    sky_temperature_c: f64,
    #[allow(dead_code)]
    altitude_deg: f64,
    #[allow(dead_code)]
    azimuth_deg: f64,
}

struct SolarTrace {
    points: Vec<SolarTracePoint>,
}

impl SolarTrace {
    fn new() -> Self {
        Self { points: Vec::new() }
    }

    fn add_point(&mut self, point: SolarTracePoint) {
        self.points.push(point);
    }

    fn peak_solar_hour(&self) -> Option<usize> {
        let mut max_idx = 0;
        let mut max_gain = 0.0f64;
        for (i, p) in self.points.iter().enumerate() {
            if p.solar_gain_w > max_gain {
                max_gain = p.solar_gain_w;
                max_idx = i;
            }
        }
        if max_gain > 0.0 {
            Some(max_idx)
        } else {
            None
        }
    }

    fn hours_with_solar_gain(&self) -> usize {
        self.points.iter().filter(|p| p.solar_gain_w > 0.0).count()
    }
}

fn generate_solar_trace(
    year: i32,
    month: u32,
    day: u32,
    window: &WindowProperties,
    orientation: Orientation,
) -> SolarTrace {
    let mut trace = SolarTrace::new();
    let _day_of_year = calculate_day_of_year(year, month, day);

    for hour in 0..24 {
        let (sun_pos, irradiance, gain) = calculate_hourly_solar(
            DENVER_LAT,
            DENVER_LON,
            year,
            month,
            day,
            hour as f64,
            900.0,
            150.0,
            window,
            None,
            None,
            &[],
            orientation,
            Some(GROUND_REFLECTANCE),
        );

        let sky_temp = SkyRadiationExchange::sky_temperature_from_emissivity(25.0, 0.8);

        trace.add_point(SolarTracePoint {
            hour: hour as f64,
            solar_gain_w: gain.total_gain_w,
            total_irradiance_wm2: irradiance.total_wm2,
            sky_temperature_c: sky_temp,
            altitude_deg: sun_pos.altitude_deg,
            azimuth_deg: sun_pos.azimuth_deg,
        });
    }

    trace
}

#[cfg(test)]
mod solar_gain_traces {
    use super::*;

    #[test]
    fn test_solar_gain_trace_summer_solstice_south() {
        let window = WindowProperties::double_clear(12.0);
        let trace = generate_solar_trace(2024, 6, 21, &window, Orientation::South);

        let hours_with_gain = trace.hours_with_solar_gain();
        assert!(
            (10..=16).contains(&hours_with_gain),
            "Summer solstice should have 10-16 hours of solar gain, got {}",
            hours_with_gain
        );

        let peak_hour = trace.peak_solar_hour();
        assert!(peak_hour.is_some(), "Should have a peak solar hour");

        let peak_idx = peak_hour.unwrap();
        let peak = &trace.points[peak_idx];
        assert!(
            (9.0..=17.0).contains(&peak.hour),
            "Peak solar should occur between 09:00-17:00, got {:.0}:00",
            peak.hour
        );

        assert!(
            peak.solar_gain_w > 5000.0,
            "Peak solar gain should exceed 5000W, got {:.0}W",
            peak.solar_gain_w
        );
    }

    #[test]
    fn test_solar_gain_trace_winter_solstice_south() {
        let window = WindowProperties::double_clear(12.0);
        let trace = generate_solar_trace(2024, 12, 21, &window, Orientation::South);

        let hours_with_gain = trace.hours_with_solar_gain();
        assert!(
            (8..=12).contains(&hours_with_gain),
            "Winter solstice should have 8-12 hours of solar gain, got {}",
            hours_with_gain
        );

        let peak_hour = trace.peak_solar_hour();
        assert!(peak_hour.is_some(), "Should have a peak solar hour");

        let peak = &trace.points[peak_hour.unwrap()];
        assert!(
            peak.solar_gain_w > 1000.0,
            "Winter peak solar gain should exceed 1000W, got {:.0}W",
            peak.solar_gain_w
        );
    }

    #[test]
    fn test_solar_gain_trace_equinox() {
        let window = WindowProperties::double_clear(12.0);
        let trace = generate_solar_trace(2024, 3, 21, &window, Orientation::South);

        let hours_with_gain = trace.hours_with_solar_gain();
        assert!(
            (10..=14).contains(&hours_with_gain),
            "Equinox should have 10-14 hours of solar gain, got {}",
            hours_with_gain
        );
    }

    #[test]
    fn test_solar_gain_trace_north_vs_south() {
        let window = WindowProperties::double_clear(12.0);
        let trace_south = generate_solar_trace(2024, 6, 21, &window, Orientation::South);
        let trace_north = generate_solar_trace(2024, 6, 21, &window, Orientation::North);

        let south_peak = trace_south
            .peak_solar_hour()
            .map(|i| trace_south.points[i].solar_gain_w);
        let north_peak = trace_north
            .peak_solar_hour()
            .map(|i| trace_north.points[i].solar_gain_w);

        assert!(
            south_peak.is_some() && north_peak.is_some(),
            "Both orientations should have peak solar hours"
        );

        assert!(
            south_peak.unwrap() > north_peak.unwrap(),
            "South peak ({:.0}W) should exceed North peak ({:.0}W) in summer",
            south_peak.unwrap(),
            north_peak.unwrap()
        );
    }

    #[test]
    fn test_solar_gain_night_hours_zero() {
        let window = WindowProperties::double_clear(12.0);
        let trace = generate_solar_trace(2024, 6, 21, &window, Orientation::South);

        for hour in 0..5 {
            let gain = trace.points[hour].solar_gain_w;
            assert!(
                gain < TOLERANCE,
                "Solar gain at hour {} should be near zero, got {:.2}W",
                hour,
                gain
            );
        }
    }
}

#[cfg(test)]
mod sky_temperature_traces {
    use super::*;

    #[test]
    fn test_sky_temperature_daily_range() {
        let window = WindowProperties::double_clear(12.0);
        let trace = generate_solar_trace(2024, 6, 21, &window, Orientation::South);

        let mut min_sky_temp = f64::MAX;
        let mut max_sky_temp = f64::MIN;

        for point in &trace.points {
            min_sky_temp = min_sky_temp.min(point.sky_temperature_c);
            max_sky_temp = max_sky_temp.max(point.sky_temperature_c);
        }

        let sky_temp_reasonable = max_sky_temp > -50.0 && min_sky_temp < 50.0;
        assert!(
            sky_temp_reasonable,
            "Sky temperature should be physically reasonable: min={:.1}°C, max={:.1}°C",
            min_sky_temp, max_sky_temp
        );

        let first_point = &trace.points[0];
        let _midday_point = &trace.points[12];
        assert!(
            first_point.sky_temperature_c < 20.0,
            "Night sky temperature should be below ambient, got {:.1}°C",
            first_point.sky_temperature_c
        );
    }

    #[test]
    fn test_sky_temperature_from_ir_trace() {
        let mut trace_temps = Vec::new();

        for hour in 0..24 {
            let ir = if (6..=18).contains(&hour) {
                300.0 + (hour as f64 - 6.0) * 30.0
            } else {
                150.0
            };
            let sky_temp = SkyRadiationExchange::sky_temperature_from_ir(ir);
            trace_temps.push(sky_temp);
        }

        let max_temp = trace_temps.iter().fold(f64::MIN, |a, &b| a.max(b));
        let min_temp = trace_temps.iter().fold(f64::MAX, |a, &b| a.min(b));

        assert!(max_temp > min_temp, "Sky temperature should vary with IR");

        assert!(
            max_temp > 0.0,
            "Peak sky temperature should be positive, got {:.1}°C",
            max_temp
        );
    }

    #[test]
    fn test_sky_temperature_from_emissivity() {
        let outdoor_temp = 25.0;

        let emissivities = [0.6, 0.7, 0.8, 0.9];
        for emissivity in emissivities {
            let sky_temp =
                SkyRadiationExchange::sky_temperature_from_emissivity(outdoor_temp, emissivity);
            assert!(
                sky_temp < outdoor_temp,
                "Sky temp ({:.1}°C) should be below ambient ({:.1}°C) for emissivity {:.1}",
                sky_temp,
                outdoor_temp,
                emissivity
            );
            assert!(
                sky_temp > -50.0 && sky_temp < 50.0,
                "Sky temp should be physically reasonable, got {:.1}°C",
                sky_temp
            );
        }
    }

    #[test]
    fn test_sky_emissivity_with_clouds_trace() {
        let dry_bulb = 20.0;
        let clearness_values = [0.1, 0.3, 0.5, 0.7, 1.0];

        for kt in clearness_values {
            let emissivity = calculate_sky_emissivity_with_clouds(dry_bulb, kt);
            assert!(
                (0.6..=1.0).contains(&emissivity),
                "Sky emissivity should be in [0.6, 1.0], got {:.3} for kt={}",
                emissivity,
                kt
            );
        }

        let emissivity_clear = calculate_sky_emissivity_with_clouds(dry_bulb, 1.0);
        let emissivity_cloudy = calculate_sky_emissivity_with_clouds(dry_bulb, 0.1);

        assert!(
            emissivity_cloudy > emissivity_clear,
            "Cloudy sky ({:.3}) should have higher emissivity than clear ({:.3})",
            emissivity_cloudy,
            emissivity_clear
        );
    }

    #[test]
    fn test_estimate_sky_emissivity() {
        let humidity = 50.0;
        let cloud_covers = [0.0, 0.3, 0.6, 1.0];

        for cloud_cover in cloud_covers {
            let emissivity = estimate_sky_emissivity(humidity, cloud_cover);
            assert!(
                (0.6..=0.98).contains(&emissivity),
                "Sky emissivity should be in [0.6, 0.98], got {:.3}",
                emissivity
            );
        }

        let emissivity_clear = estimate_sky_emissivity(humidity, 0.0);
        let emissivity_cloudy = estimate_sky_emissivity(humidity, 1.0);

        assert!(
            emissivity_cloudy > emissivity_clear,
            "Cloudy should have higher emissivity than clear"
        );
    }

    #[test]
    fn test_sol_air_temperature_roof_trace() {
        let sol = SolAirTemperature::ashrae_140_default();
        let mut trace_temps = Vec::new();

        for hour in 0..24 {
            let solar = if (6..=18).contains(&hour) {
                600.0 * ((hour as f64 - 6.0) / 12.0) * (1.0 - (hour as f64 - 6.0) / 12.0)
            } else {
                0.0
            };
            let sky_temp = -10.0;
            let sol_air = sol.calculate(25.0, solar, sky_temp, None);
            trace_temps.push(sol_air);
        }

        let max_sol_air = trace_temps.iter().fold(f64::MIN, |a, &b| a.max(b));

        assert!(
            max_sol_air > 30.0,
            "Peak sol-air temp should exceed 30°C, got {:.1}°C",
            max_sol_air
        );
    }
}

#[cfg(test)]
mod peak_solar_timing {
    use super::*;

    #[test]
    fn test_peak_solar_timing_summer() {
        let window = WindowProperties::double_clear(12.0);
        let trace = generate_solar_trace(2024, 6, 21, &window, Orientation::South);

        let peak_hour = trace.peak_solar_hour().expect("Should have peak hour");

        assert!(
            (9..=17).contains(&peak_hour),
            "Summer peak should occur 09:00-17:00, got {:02}:00",
            peak_hour
        );
    }

    #[test]
    fn test_peak_solar_timing_winter() {
        let window = WindowProperties::double_clear(12.0);
        let trace = generate_solar_trace(2024, 12, 21, &window, Orientation::South);

        let peak_hour = trace.peak_solar_hour().expect("Should have peak hour");

        assert!(
            (10..=14).contains(&peak_hour),
            "Winter peak should occur 10:00-14:00, got {:02}:00",
            peak_hour
        );
    }

    #[test]
    fn test_peak_solar_timing_equinox() {
        let window = WindowProperties::double_clear(12.0);
        let trace = generate_solar_trace(2024, 3, 21, &window, Orientation::South);

        let peak_hour = trace.peak_solar_hour().expect("Should have peak hour");

        assert!(
            (11..=13).contains(&peak_hour),
            "Equinox peak should occur 11:00-13:00, got {:02}:00",
            peak_hour
        );
    }

    #[test]
    fn test_solar_position_noon_summer() {
        let sun_pos = calculate_solar_position(DENVER_LAT, DENVER_LON, 2024, 6, 21, 12.0);

        assert!(
            (70.0..=77.0).contains(&sun_pos.altitude_deg),
            "Summer noon altitude should be 70-77°, got {:.1}°",
            sun_pos.altitude_deg
        );

        assert!(
            (175.0..=185.0).contains(&sun_pos.azimuth_deg),
            "Summer noon azimuth should be ~180° (South), got {:.1}°",
            sun_pos.azimuth_deg
        );
    }

    #[test]
    fn test_solar_position_noon_winter() {
        let sun_pos = calculate_solar_position(DENVER_LAT, DENVER_LON, 2024, 12, 21, 12.0);

        assert!(
            (24.0..=30.0).contains(&sun_pos.altitude_deg),
            "Winter noon altitude should be 24-30°, got {:.1}°",
            sun_pos.altitude_deg
        );
    }

    #[test]
    fn test_solar_position_noon_equinox() {
        let sun_pos = calculate_solar_position(DENVER_LAT, DENVER_LON, 2024, 3, 21, 12.0);
        let expected_alt = 90.0 - DENVER_LAT;

        assert!(
            (expected_alt - 2.0..=expected_alt + 2.0).contains(&sun_pos.altitude_deg),
            "Equinox altitude should be ~{:.1}°, got {:.1}°",
            expected_alt,
            sun_pos.altitude_deg
        );
    }

    #[test]
    fn test_peak_solar_timing_east_west_comparison() {
        let window = WindowProperties::double_clear(12.0);

        let trace_east = generate_solar_trace(2024, 6, 21, &window, Orientation::East);
        let trace_west = generate_solar_trace(2024, 6, 21, &window, Orientation::West);

        let peak_east = trace_east.peak_solar_hour();
        let peak_west = trace_west.peak_solar_hour();

        assert!(
            peak_east.is_some() && peak_west.is_some(),
            "Both E and W should have peak hours"
        );

        let east_peak_hour = peak_east.unwrap();
        let west_peak_hour = peak_west.unwrap();

        assert!(
            east_peak_hour < west_peak_hour,
            "East peak ({:02}:00) should occur before West ({:02}:00)",
            east_peak_hour,
            west_peak_hour
        );

        assert!(
            east_peak_hour <= 10,
            "East-facing peak should occur by 10:00, got {:02}:00",
            east_peak_hour
        );

        assert!(
            west_peak_hour >= 14,
            "West-facing peak should occur at 14:00 or later, got {:02}:00",
            west_peak_hour
        );
    }
}

#[cfg(test)]
mod longwave_boundary_conditions {
    use super::*;

    #[test]
    fn test_sky_radiation_exchange_horizontal_roof() {
        let sky = SkyRadiationExchange::horizontal_roof();

        assert!(
            (sky.surface_emissivity - 0.9).abs() < TOLERANCE,
            "Default emissivity should be 0.9"
        );
        assert!(
            (sky.sky_view_factor - 1.0).abs() < TOLERANCE,
            "Horizontal roof sky view factor should be 1.0"
        );
    }

    #[test]
    fn test_sky_radiation_exchange_tilted_surface() {
        let sky_vertical = SkyRadiationExchange::tilted_surface(90.0, 0.9);
        let sky_45 = SkyRadiationExchange::tilted_surface(45.0, 0.9);

        assert!(
            sky_vertical.sky_view_factor < 1.0,
            "Vertical surface should have sky view factor < 1.0"
        );
        assert!(
            sky_45.sky_view_factor > sky_vertical.sky_view_factor,
            "45° tilt should have higher sky view factor than vertical"
        );

        let expected_45 = (1.0 + 45.0_f64.to_radians().cos()) / 2.0;
        assert!(
            (sky_45.sky_view_factor - expected_45).abs() < TOLERANCE,
            "45° sky view factor should be {:.3}, got {:.3}",
            expected_45,
            sky_45.sky_view_factor
        );
    }

    #[test]
    fn test_net_radiative_flux_cooling() {
        let sky = SkyRadiationExchange::horizontal_roof();

        let flux = sky.net_radiative_flux(30.0, -10.0);

        assert!(
            flux < 0.0,
            "Warm surface should lose heat to cold sky (negative flux), got {:.2} W/m²",
            flux
        );
    }

    #[test]
    fn test_net_radiative_flux_heating() {
        let sky = SkyRadiationExchange::horizontal_roof();

        let flux = sky.net_radiative_flux(-10.0, 10.0);

        assert!(
            flux > 0.0,
            "Cold surface should gain heat from warm sky (positive flux), got {:.2} W/m²",
            flux
        );
    }

    #[test]
    fn test_net_radiative_flux_equilibrium() {
        let sky = SkyRadiationExchange::horizontal_roof();

        let flux = sky.net_radiative_flux(20.0, 20.0);

        assert!(
            flux.abs() < 1e-6,
            "Equal temperatures should give near-zero flux, got {:.2} W/m²",
            flux
        );
    }

    #[test]
    fn test_radiative_coefficient_magnitude() {
        let sky = SkyRadiationExchange::horizontal_roof();

        let h_r = sky.radiative_coefficient(20.0, 0.0);

        assert!(
            (4.0..=8.0).contains(&h_r),
            "Radiative coefficient should be 4-8 W/m²K, got {:.2}",
            h_r
        );
    }

    #[test]
    fn test_sol_air_temperature_simple() {
        let t_sol = sol_air_temperature_simple(25.0, 500.0, 0.6, 22.7);

        let expected = 25.0 + (0.6 * 500.0 / 22.7);
        assert!(
            (t_sol - expected).abs() < TOLERANCE,
            "Sol-air temp should be {:.2}, got {:.2}",
            expected,
            t_sol
        );
    }

    #[test]
    fn test_sol_air_temperature_with_longwave() {
        let sol = SolAirTemperature::ashrae_140_default();

        let t_sol_day = sol.calculate(35.0, 500.0, -10.0, None);
        let t_sol_night = sol.calculate(25.0, 0.0, -20.0, None);

        assert!(
            t_sol_day > 35.0,
            "Day sol-air should exceed outdoor temp ({:.1}°C), got {:.1}°C",
            35.0,
            t_sol_day
        );

        assert!(
            t_sol_night > 25.0,
            "Night sol-air should exceed outdoor temp ({:.1}°C), got {:.1}°C (sky cooling effect)",
            25.0,
            t_sol_night
        );
    }

    #[test]
    fn test_exterior_conductance() {
        let h_calm = SolAirTemperature::calculate_exterior_conductance(0.0);
        let h_windy = SolAirTemperature::calculate_exterior_conductance(10.0);

        assert!(
            h_calm < h_windy,
            "Calm conductance ({:.1}) should be less than windy ({:.1})",
            h_calm,
            h_windy
        );

        assert!(
            (h_calm - 10.8).abs() < 0.1,
            "Calm conductance should be ~10.8, got {:.1}",
            h_calm
        );
    }
}

#[cfg(test)]
mod solar_constant_verification {
    use super::*;

    #[test]
    fn test_solar_constant_value() {
        const EXPECTED_SOLAR_CONSTANT: f64 = 1361.0;
        let dni_day182 = extraterrestrial_irradiance(182);
        let expected_at_aphelion = EXPECTED_SOLAR_CONSTANT * 0.967;

        let deviation = (dni_day182 - expected_at_aphelion).abs();
        assert!(
            deviation < 10.0,
            "Solar constant at aphelion (day 182) should be ~{:.0} W/m² (solar constant * 0.967), got {:.1} (deviation: {:.1})",
            expected_at_aphelion,
            dni_day182,
            deviation
        );
    }

    #[test]
    fn test_extraterrestrial_irradiance_variation() {
        let dni_jan = extraterrestrial_irradiance(1);
        let dni_jul = extraterrestrial_irradiance(182);

        assert!(
            dni_jan > dni_jul,
            "January (perihelion) DNI ({:.0}) should exceed July (aphelion) ({:.0})",
            dni_jan,
            dni_jul
        );
    }

    #[test]
    fn test_relative_airmass_at_zenith() {
        let am_noon = relative_airmass(0.0);

        assert!(
            (am_noon - 1.0).abs() < 0.1,
            "Airmass at zenith should be ~1, got {:.2}",
            am_noon
        );
    }

    #[test]
    fn test_relative_airmass_increases_with_zenith() {
        let am_30 = relative_airmass(30.0);
        let am_60 = relative_airmass(60.0);
        let am_80 = relative_airmass(80.0);

        assert!(
            am_60 > am_30,
            "Airmass at 60° ({:.2}) should exceed 30° ({:.2})",
            am_60,
            am_30
        );
        assert!(
            am_80 > am_60,
            "Airmass at 80° ({:.2}) should exceed 60° ({:.2})",
            am_80,
            am_60
        );
    }

    #[test]
    fn test_perez_diffuse_tilted_basic() {
        let diffuse = PerezSkyModel::calculate_diffuse_tilted(
            100.0, 800.0, 1366.0, 1.5, 30.0, 0.0, 0.0, 180.0,
        );

        assert!(
            diffuse > 0.0,
            "Diffuse irradiance should be positive, got {:.1}",
            diffuse
        );

        assert!(
            diffuse < 100.0,
            "Diffuse should be less than DHI for horizontal, got {:.1}",
            diffuse
        );
    }

    #[test]
    fn test_perez_diffuse_tilted_vertical() {
        let diffuse_horizontal = PerezSkyModel::calculate_diffuse_tilted(
            100.0, 800.0, 1366.0, 1.5, 30.0, 0.0, 0.0, 180.0,
        );

        let diffuse_vertical = PerezSkyModel::calculate_diffuse_tilted(
            100.0, 800.0, 1366.0, 1.5, 30.0, 90.0, 180.0, 180.0,
        );

        assert!(
            diffuse_vertical > 0.0,
            "Vertical diffuse should be positive"
        );
        assert!(
            diffuse_horizontal > diffuse_vertical,
            "Horizontal diffuse ({:.1}) should exceed vertical ({:.1})",
            diffuse_horizontal,
            diffuse_vertical
        );
    }

    #[test]
    fn test_perez_diffuse_zero_dhi() {
        let diffuse = PerezSkyModel::calculate_diffuse_tilted(
            0.0, 800.0, 1366.0, 1.5, 30.0, 45.0, 180.0, 180.0,
        );

        assert!(
            diffuse.abs() < TOLERANCE,
            "Zero DHI should give zero diffuse, got {:.4}",
            diffuse
        );
    }
}

#[cfg(test)]
mod window_solar_gain_traces {
    use super::*;

    #[test]
    fn test_window_gain_trace_south_summer() {
        let window = WindowProperties::double_clear(12.0);
        let trace = generate_solar_trace(2024, 6, 21, &window, Orientation::South);

        let total_daily_gain: f64 = trace.points.iter().map(|p| p.solar_gain_w).sum();

        assert!(
            total_daily_gain > 10000.0,
            "Daily solar gain should exceed 10kWh, got {:.0} Wh",
            total_daily_gain
        );

        let peak = &trace.points[trace.peak_solar_hour().unwrap()];
        assert!(
            peak.solar_gain_w > 5000.0,
            "Peak window gain should exceed 5kW, got {:.0}W",
            peak.solar_gain_w
        );
    }

    #[test]
    fn test_window_gain_different_orientations() {
        let window = WindowProperties::double_clear(12.0);

        let orientations = [
            (Orientation::South, 6, 21),
            (Orientation::East, 6, 21),
            (Orientation::West, 6, 21),
            (Orientation::North, 6, 21),
        ];

        let mut peak_gains = Vec::new();
        for (orient, month, day) in &orientations {
            let trace = generate_solar_trace(2024, *month, *day, &window, *orient);
            if let Some(peak_idx) = trace.peak_solar_hour() {
                peak_gains.push((orient, trace.points[peak_idx].solar_gain_w));
            }
        }

        let south_peak = peak_gains
            .iter()
            .find(|(o, _)| **o == Orientation::South)
            .map(|(_, g)| *g);

        assert!(south_peak.is_some(), "Should have south peak gain");

        for (_orient, gain) in &peak_gains {
            assert!(
                *gain > 0.0,
                "All orientations should have positive peak gain"
            );
        }
    }

    #[test]
    fn test_window_gain_angular_effect() {
        let window = WindowProperties::double_clear(12.0);

        let trace = generate_solar_trace(2024, 6, 21, &window, Orientation::South);

        let peak_idx = trace.peak_solar_hour().unwrap();
        let peak_point = &trace.points[peak_idx];

        let sun_pos =
            calculate_solar_position(DENVER_LAT, DENVER_LON, 2024, 6, 21, peak_point.hour);
        let cos_incidence = sun_pos.incidence_cosine(90.0, 180.0);

        assert!(
            cos_incidence > 0.5,
            "South-facing window should have high incidence at noon, got cos(θ)={:.3}",
            cos_incidence
        );
    }
}
