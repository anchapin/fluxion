use fluxion::sim::solar::{WindowProperties, calculate_window_solar_gain, SurfaceIrradiance, SolarPosition};
use fluxion::validation::Orientation;

fn main() {
    // Test window: 12 m² area, SHGC = 0.77
    let window = WindowProperties::new(12.0, 0.77, 0.703);

    // Test irradiance: 500 W/m² beam, 200 W/m² diffuse, 50 W/m² ground
    let irradiance = SurfaceIrradiance {
        beam_wm2: 500.0,
        diffuse_wm2: 200.0,
        ground_reflected_wm2: 50.0,
        total_wm2: 750.0,
    };

    // Solar position: 30° altitude, 0° relative azimuth (normal incidence)
    let sun_pos = SolarPosition {
        azimuth_deg: 180.0,
        altitude_deg: 30.0,
        zenith_deg: 60.0,
        incidence_deg: 0.0,
    };

    // Calculate solar gain
    let gain = calculate_window_solar_gain(
        &irradiance,
        &window,
        None,
        None,
        &[],
        &sun_pos,
        Orientation::South,
    );

    println!("Window: area={:.1} m², shgc={:.3}", window.area, window.shgc);
    println!("Irradiance: beam={:.1}, diffuse={:.1}, ground={:.1} W/m²",
             irradiance.beam_wm2, irradiance.diffuse_wm2, irradiance.ground_reflected_wm2);
    println!("Solar gain: beam={:.1}, diffuse={:.1}, ground={:.1}, total={:.1} W",
             gain.beam_gain_w, gain.diffuse_gain_w, gain.ground_reflected_gain_w, gain.total_gain_w);

    // Expected diffuse_gain without 0.9 factor: 12.0 * 200 * 0.77 = 1848 W
    // Expected diffuse_gain with 0.9 factor: 12.0 * 200 * 0.77 * 0.9 = 1663.2 W
    let expected_without_0_9 = 12.0 * 200.0 * 0.77;
    let expected_with_0_9 = expected_without_0_9 * 0.9;

    println!("\nExpected diffuse_gain:");
    println!("  Without 0.9 factor: {:.1} W", expected_without_0_9);
    println!("  With 0.9 factor: {:.1} W", expected_with_0_9);
    println!("  Actual diffuse_gain: {:.1} W", gain.diffuse_gain_w);

    if (gain.diffuse_gain_w - expected_without_0_9).abs() < 1.0 {
        println!("\n✅ PASS: diffuse_shgc is NOT multiplied by 0.9 (fix is active)");
    } else if (gain.diffuse_gain_w - expected_with_0_9).abs() < 1.0 {
        println!("\n❌ FAIL: diffuse_shgc IS multiplied by 0.9 (fix is NOT active)");
    } else {
        println!("\n⚠️  UNEXPECTED: diffuse_gain doesn't match either expected value");
    }
}
