//! Annual solar gain analysis for Case 920 vs 900
//!
//! Compares total annual solar gains for E/W vs south-facing configurations

use fluxion::sim::solar::*;
use fluxion::validation::ashrae_140_cases::Orientation;

fn main() {
    let window_6 = WindowProperties::double_clear(6.0);
    let window_12 = WindowProperties::double_clear(12.0);

    let denver_lat = 39.7392;
    let denver_lon = -104.9903;

    println!("=== Annual Solar Gain Analysis ===");
    println!("Comparing Case 920 (6m² E + 6m² W) vs Case 900 (12m² S)");
    println!();

    // Sample summer and winter days
    let test_days = [
        (1, 21, "Winter Solstice"),
        (3, 21, "Spring Equinox"),
        (6, 21, "Summer Solstice"),
        (9, 21, "Fall Equinox"),
    ];

    for (month, day, label) in test_days {
        println!("=== {} (Month {}, Day {}) ===", label, month, day);

        let mut daily_east = 0.0;
        let mut daily_west = 0.0;
        let mut daily_south_12 = 0.0;
        let mut daily_south_6 = 0.0;

        // Sample every hour from 6 AM to 6 PM
        for hour in 6..=18 {
            let sun_pos =
                calculate_solar_position(denver_lat, denver_lon, 2024, month, day, hour as f64);

            if !sun_pos.is_above_horizon() {
                continue;
            }

            let day_of_year = calculate_day_of_year(2024, month, day);

            // Use clear sky values
            let dni = 900.0;
            let dhi = 150.0;

            let irr_e = calculate_surface_irradiance(
                &sun_pos,
                dni,
                dhi,
                None,
                Orientation::East,
                0.2,
                day_of_year,
            );
            let irr_w = calculate_surface_irradiance(
                &sun_pos,
                dni,
                dhi,
                None,
                Orientation::West,
                0.2,
                day_of_year,
            );
            let irr_s = calculate_surface_irradiance(
                &sun_pos,
                dni,
                dhi,
                None,
                Orientation::South,
                0.2,
                day_of_year,
            );

            let gain_e = calculate_window_solar_gain(
                &irr_e,
                &window_6,
                None,
                None,
                &[],
                &sun_pos,
                Orientation::East,
            );
            let gain_w = calculate_window_solar_gain(
                &irr_w,
                &window_6,
                None,
                None,
                &[],
                &sun_pos,
                Orientation::West,
            );
            let gain_s_12 = calculate_window_solar_gain(
                &irr_s,
                &window_12,
                None,
                None,
                &[],
                &sun_pos,
                Orientation::South,
            );
            let gain_s_6 = calculate_window_solar_gain(
                &irr_s,
                &window_6,
                None,
                None,
                &[],
                &sun_pos,
                Orientation::South,
            );

            daily_east += gain_e.total_gain_w;
            daily_west += gain_w.total_gain_w;
            daily_south_12 += gain_s_12.total_gain_w;
            daily_south_6 += gain_s_6.total_gain_w;
        }

        println!("East (6m²):    {:.0} Wh", daily_east);
        println!("West (6m²):    {:.0} Wh", daily_west);
        println!("E+W (12m²):    {:.0} Wh", daily_east + daily_west);
        println!("South (12m²):  {:.0} Wh", daily_south_12);
        println!("South (6m²):   {:.0} Wh", daily_south_6);
        println!(
            "Ratio (E+W)/S: {:.2}",
            (daily_east + daily_west) / daily_south_12
        );
        println!();
    }

    println!("=== Key Insight ===");
    println!("South-facing windows get MORE solar gain overall");
    println!("This explains why Case 900 has higher cooling load");
    println!();
    println!("BUT: Case 920 cooling is TOO LOW even accounting for this");
    println!("Case 900: 6.18 MWh (Ref: 2.13-3.67) - 68% above max");
    println!("Case 920: 1.29 MWh (Ref: 1.84-3.31) - 30% below min");
    println!();
    println!("Both cases are wrong, but in opposite directions!");
}
