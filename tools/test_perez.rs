//! Test Perez sky model calculation

use fluxion::sim::sky_radiation::PerezSkyModel;

fn main() {
    println!("Testing Perez Sky Model");
    println!("======================\n");

    // Test case: Clear summer day, West surface at 2pm
    let dhi = 126.3; // W/m²
    let dni = 899.4; // W/m²
    let dni_extra = 1320.0; // W/m²
    let airmass = 1.1;
    let zenith_deg = 25.0; // Altitude = 65°
    let surface_tilt_deg = 90.0; // Vertical
    let surface_azimuth_deg = 270.0; // West
    let solar_azimuth_deg = 240.0; // WSW

    println!("Inputs:");
    println!("  DHI: {} W/m²", dhi);
    println!("  DNI: {} W/m²", dni);
    println!("  DNI_extra: {} W/m²", dni_extra);
    println!("  Airmass: {}", airmass);
    println!("  Zenith: {}°", zenith_deg);
    println!("  Surface tilt: {}°", surface_tilt_deg);
    println!("  Surface azimuth: {}°", surface_azimuth_deg);
    println!("  Solar azimuth: {}°\n", solar_azimuth_deg);

    let diffuse = PerezSkyModel::calculate_diffuse_tilted(
        dhi,
        dni,
        dni_extra,
        airmass,
        zenith_deg,
        surface_tilt_deg,
        surface_azimuth_deg,
        solar_azimuth_deg,
    );

    println!("Result:");
    println!("  Diffuse tilted: {} W/m²", diffuse);
    println!("  Expected: ~61 W/m²");
    println!("  Ratio: {:.2}", diffuse / 61.0);

    // Debug: Calculate intermediate values
    let kappa = 1.041;
    let delta = dhi * airmass / dni_extra;
    let zenith_rad = zenith_deg.to_radians();
    let z_cubed = zenith_rad.powi(3);
    let epsilon = ((dhi + dni) / dhi + kappa * z_cubed) / (1.0 + kappa * z_cubed);

    println!("\nIntermediate values:");
    println!("  Delta: {:.4}", delta);
    println!("  Epsilon: {:.2}", epsilon);

    // Determine ebin
    let bounds = [0.0, 1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2];
    let mut ebin = 7;
    for (i, &bound) in bounds.iter().enumerate() {
        if epsilon <= bound {
            ebin = i;
            break;
        }
    }
    println!("  Sky clearness bin: {}", ebin);

    // Get coefficients
    let f1c = match ebin {
        0 => [-0.008317, 0.587728, -0.062064],
        1 => [0.129967, 0.682595, -0.151375],
        2 => [0.329676, 0.486861, -0.221272],
        3 => [0.568205, 0.187452, -0.295250],
        4 => [0.873018, -0.393289, -0.369150],
        5 => [1.321297, -1.176777, -0.393994],
        6 => [0.999852, -1.634380, -0.291495],
        _ => [0.553776, 0.631414, -0.209172],
    };

    let f2c = match ebin {
        0 => [0.091000, 0.060000, 0.000000],
        1 => [0.055000, 0.060000, 0.000000],
        2 => [0.025000, 0.060000, 0.000000],
        3 => [-0.015000, 0.060000, 0.000000],
        4 => [-0.065000, 0.060000, 0.000000],
        5 => [-0.115000, 0.060000, 0.000000],
        6 => [-0.165000, 0.060000, 0.000000],
        _ => [-0.215000, 0.060000, 0.000000],
    };

    let f1 = (f1c[0] + f1c[1] * delta + f1c[2] * zenith_rad).max(0.0);
    let f2 = f2c[0] + f2c[1] * delta + f2c[2] * zenith_rad;

    println!("  F1: {:.4}", f1);
    println!("  F2: {:.4}", f2);

    // Calculate terms
    let surface_tilt = surface_tilt_deg.to_radians();
    let cos_incidence = (surface_tilt.sin()
        * (270.0_f64).to_radians().sin()
        * zenith_rad.cos()
        * (240.0_f64).to_radians().sin()
        + surface_tilt.sin()
            * (270.0_f64).to_radians().cos()
            * zenith_rad.cos()
            * (240.0_f64).to_radians().cos()
        + surface_tilt.cos() * zenith_rad.sin())
    .clamp(-1.0, 1.0);

    let a = cos_incidence.max(0.0);
    let b = zenith_rad.cos().max((85.0_f64).to_radians().cos());

    let term1 = 0.5 * (1.0 - f1) * (1.0 + surface_tilt.cos());
    let term2 = f1 * a / b;
    let term3 = f2 * surface_tilt.sin();

    println!("\nPerez terms:");
    println!("  Cos(incidence): {:.4}", cos_incidence);
    println!("  A: {:.4}", a);
    println!("  B: {:.4}", b);
    println!("  Term1 (isotropic): {:.4}", term1);
    println!("  Term2 (circumsolar): {:.4}", term2);
    println!("  Term3 (horizon): {:.4}", term3);
    println!("  Total factor: {:.4}", term1 + term2 + term3);
    println!(
        "  Diffuse tilted: {:.1f} W/m²",
        dhi * (term1 + term2 + term3)
    );
}
