//! Diagnostic tool to investigate E/W shading calculation
//!
//! Checks if shading for E/W windows is calculated correctly

use fluxion::sim::solar::*;
use fluxion::sim::shading::{Overhang, ShadeFin, Side};
use fluxion::validation::ashrae_140_cases::Orientation;

fn main() {
    let window = WindowProperties::double_clear(6.0);

    let denver_lat = 39.7392;
    let denver_lon = -104.9903;

    println!("=== E/W Shading Diagnostic ===");
    println!("Testing June 21 (summer solstice) in Denver");
    println!();

    // Case 920: No shading
    println!("=== Case 920: E/W windows WITHOUT shading ===");
    let hours = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0];

    println!("Time | East Gain | West Gain | Total");
    println!("-----|-----------|-----------|-------");

    let mut total_920 = 0.0;
    for &hour in &hours {
        let sun_pos = calculate_solar_position(denver_lat, denver_lon, 2024, 6, 21, hour);
        if !sun_pos.is_above_horizon() {
            continue;
        }
        let day_of_year = calculate_day_of_year(2024, 6, 21);

        // East window (no shading)
        let irr_e = calculate_surface_irradiance(&sun_pos, 900.0, 150.0, None, Orientation::East, 0.2, day_of_year);
        let gain_e = calculate_window_solar_gain(&irr_e, &window, None, None, &[], &sun_pos, Orientation::East);

        // West window (no shading)
        let irr_w = calculate_surface_irradiance(&sun_pos, 900.0, 150.0, None, Orientation::West, 0.2, day_of_year);
        let gain_w = calculate_window_solar_gain(&irr_w, &window, None, None, &[], &sun_pos, Orientation::West);

        let total = gain_e.total_gain_w + gain_w.total_gain_w;
        total_920 += total;

        println!("{:5.1} | {:9.0} W | {:9.0} W | {:5.0} W",
            hour, gain_e.total_gain_w, gain_w.total_gain_w, total);
    }

    println!();
    println!("Total (6h sample): {:.0} Wh", total_920);
    println!();

    // Case 930: With overhang and shade fins
    println!("=== Case 930: E/W windows WITH overhang + fins ===");
    println!("Shading: 1m overhang + 1m shade fins");

    let overhang = Overhang {
        depth: 1.0,
        distance_above: 2.7,
        extension: 10.0,
    };
    let fins = vec![ShadeFin {
        depth: 1.0,
        distance_from_edge: 0.0,
        side: Side::Left,
    }];

    // Create window geometry for shading calculation
    use fluxion::validation::ashrae_140_cases::WindowArea;
    let geometry = WindowArea {
        area: 6.0,
        orientation: Orientation::East,
        height: 2.0,
        width: 3.0,
        sill_height: 1.0,
        left_offset: 0.0,
    };

    println!("Time | East Gain | West Gain | Total | Shaded");
    println!("-----|-----------|-----------|-------|--------");

    let mut total_930 = 0.0;
    for &hour in &hours {
        let sun_pos = calculate_solar_position(denver_lat, denver_lon, 2024, 6, 21, hour);
        if !sun_pos.is_above_horizon() {
            continue;
        }
        let day_of_year = calculate_day_of_year(2024, 6, 21);

        // East window (with shading)
        let irr_e = calculate_surface_irradiance(&sun_pos, 900.0, 150.0, None, Orientation::East, 0.2, day_of_year);
        let gain_e = calculate_window_solar_gain(&irr_e, &window, Some(&geometry), Some(&overhang), &fins, &sun_pos, Orientation::East);

        // West window (with shading)
        let irr_w = calculate_surface_irradiance(&sun_pos, 900.0, 150.0, None, Orientation::West, 0.2, day_of_year);
        let gain_w = calculate_window_solar_gain(&irr_w, &window, Some(&geometry), Some(&overhang), &fins, &sun_pos, Orientation::West);

        let total = gain_e.total_gain_w + gain_w.total_gain_w;
        total_930 += total;

        // Calculate shaded fraction for diagnostic
        let shaded_frac_e = if gain_e.beam_gain_w > 0.0 {
            let irr_e_no_shade = calculate_surface_irradiance(&sun_pos, 900.0, 150.0, None, Orientation::East, 0.2, day_of_year);
            let gain_e_no_shade = calculate_window_solar_gain(&irr_e_no_shade, &window, Some(&geometry), None, &[], &sun_pos, Orientation::East);
            1.0 - (gain_e.beam_gain_w / gain_e_no_shade.beam_gain_w.max(0.001))
        } else {
            0.0
        };

        println!("{:5.1} | {:9.0} W | {:9.0} W | {:5.0} W | {:.0}%",
            hour, gain_e.total_gain_w, gain_w.total_gain_w, total, shaded_frac_e * 100.0);
    }

    println!();
    println!("Total (6h sample): {:.0} Wh", total_930);
    println!();

    println!("=== Comparison ===");
    println!("Case 920 (no shading):    {:.0} Wh", total_920);
    println!("Case 930 (with shading):  {:.0} Wh", total_930);
    println!("Reduction: {:.1}%", (1.0 - total_930 / total_920) * 100.0);
    println!();

    println!("=== Validation Results ===");
    println!("Case 920 cooling: 1.29 MWh (Ref: 1.84-3.31)");
    println!("Case 930 cooling: 0.49 MWh (Ref: 1.04-2.24)");
    println!("Ratio (930/920): {:.1}", 0.49 / 1.29);
    println!();

    println!("=== Key Question ===");
    println!("Is the shading calculation reducing gains too much?");
    println!("Expected: 20-40% reduction from shading");
    println!("Actual cooling ratio: 62% reduction (0.49/1.29)");
}
