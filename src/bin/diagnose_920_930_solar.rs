//! Diagnostic tool to investigate Case 920/930 cooling underprediction
//!
//! This tool analyzes solar gain patterns for E/W vs south-facing windows
//! to identify why cases 920 and 930 severely underpredict cooling demand.

use fluxion::sim::solar::*;
use fluxion::validation::ashrae_140_cases::Orientation;

fn main() {
    let window = WindowProperties::double_clear(6.0); // 6 m² per window (like Case 920)

    let denver_lat = 39.7392;
    let denver_lon = -104.9903;

    // Test summer solstice (June 21) - peak cooling season
    println!("=== Solar Gain Analysis for E/W Windows ===");
    println!("Testing on June 21 (summer solstice) in Denver");
    println!("Window: 6 m² double clear glass (SHGC=0.789)");
    println!();

    let hours = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

    println!("Time | Sun Alt | Sun Az | East Gain | South Gain | West Gain");
    println!("-----|---------|--------|-----------|------------|-----------");

    let mut total_east = 0.0;
    let mut total_south = 0.0;
    let mut total_west = 0.0;

    for &hour in &hours {
        let sun_pos = calculate_solar_position(denver_lat, denver_lon, 2024, 6, 21, hour);

        if !sun_pos.is_above_horizon() {
            continue;
        }

        let day_of_year = calculate_day_of_year(2024, 6, 21);

        // Sample DNI/DHI for clear sky
        let dni = 900.0;
        let dhi = 150.0;

        // Calculate gains for each orientation
        let irr_e = calculate_surface_irradiance(&sun_pos, dni, dhi, None, Orientation::East, 0.2, day_of_year);
        let irr_s = calculate_surface_irradiance(&sun_pos, dni, dhi, None, Orientation::South, 0.2, day_of_year);
        let irr_w = calculate_surface_irradiance(&sun_pos, dni, dhi, None, Orientation::West, 0.2, day_of_year);

        let gain_e = calculate_window_solar_gain(&irr_e, &window, None, None, &[], &sun_pos, Orientation::East);
        let gain_s = calculate_window_solar_gain(&irr_s, &window, None, None, &[], &sun_pos, Orientation::South);
        let gain_w = calculate_window_solar_gain(&irr_w, &window, None, None, &[], &sun_pos, Orientation::West);

        total_east += gain_e.total_gain_w;
        total_south += gain_s.total_gain_w;
        total_west += gain_w.total_gain_w;

        println!("{:5.1} | {:7.1}° | {:6.1}° | {:9.0} W | {:10.0} W | {:9.0} W",
            hour, sun_pos.altitude_deg, sun_pos.azimuth_deg,
            gain_e.total_gain_w, gain_s.total_gain_w, gain_w.total_gain_w);
    }

    println!();
    println!("=== Total Gains (6m² each) ===");
    println!("East (6m²):  {:.0} Wh", total_east);
    println!("South (6m²): {:.0} Wh", total_south);
    println!("West (6m²):  {:.0} Wh", total_west);
    println!("E+W total:   {:.0} Wh", total_east + total_west);
    println!();

    // Compare to Case 900 (12m² south-facing)
    let window_south = WindowProperties::double_clear(12.0);
    println!("=== Case 900 (12m² South-facing) ===");
    let mut total_south_12 = 0.0;
    for &hour in &hours {
        let sun_pos = calculate_solar_position(denver_lat, denver_lon, 2024, 6, 21, hour);
        if !sun_pos.is_above_horizon() {
            continue;
        }
        let day_of_year = calculate_day_of_year(2024, 6, 21);
        let irr_s = calculate_surface_irradiance(&sun_pos, 900.0, 150.0, None, Orientation::South, 0.2, day_of_year);
        let gain_s = calculate_window_solar_gain(&irr_s, &window_south, None, None, &[], &sun_pos, Orientation::South);
        total_south_12 += gain_s.total_gain_w;
    }
    println!("South (12m²): {:.0} Wh", total_south_12);
    println!();

    println!("=== Ratio Analysis ===");
    println!("E+W (12m² total) / South (12m²): {:.2}", (total_east + total_west) / total_south_12);
    println!();

    // Now test with actual Case 920 spec (6m² East + 6m² West)
    println!("=== Case 920 Configuration ===");
    println!("Windows: 6m² East + 6m² West = 12m² total");
    println!("Expected cooling load should be similar to Case 900 (12m² South)");
    println!("But actual results show MUCH LOWER cooling:");
    println!("  Case 900: 6.18 MWh cooling");
    println!("  Case 920: 1.29 MWh cooling");
    println!("  Ratio: {:.1}x lower", 6.18 / 1.29);
}
