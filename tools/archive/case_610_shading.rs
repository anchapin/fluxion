//! Case 610 Shading Diagnostic
//! Run: cargo run --release --bin case_610_shading

use fluxion::sim::shading::{calculate_shaded_fraction, LocalSolarPosition, Overhang};
use fluxion::validation::ashrae_140_cases::{Orientation, WindowArea};

fn main() {
    println!("\n=== Case 610 Shading Diagnostic Test ===");
    println!("Testing different overhang positions to find optimal shading");
    println!("Window: 12m² South-facing (6m wide × 2m high)");
    println!("Overhang depth: 1.0m");
    println!();

    // Case 610 window configuration
    let window = WindowArea::with_dimensions(12.0, Orientation::South, 2.0, 6.0, 0.2, 0.5);

    // Test different distance_above values
    let test_positions = [
        (0.0, "At window top (current impl)"),
        (0.5, "0.5m above window (recommended)"),
        (1.0, "1.0m above window"),
        (2.0, "2.0m above window"),
        (2.7, "2.7m above window (mounting_height spec)"),
    ];

    for (distance_above, label) in test_positions {
        println!("=== Position: {} ===", label);
        let overhang = Overhang {
            depth: 1.0,
            distance_above,
            extension: 10.0,
        };

        println!("{:<20} {:>8} {:>12}", "Condition", "Alt(°)", "Shaded Frac");
        println!("{}", "-".repeat(45));

        let test_cases: [(&str, f64, f64); 3] = [
            ("Summer noon", 73.5, 0.0),
            ("Equinox noon", 50.0, 0.0),
            ("Winter noon", 26.5, 0.0),
        ];

        for (label, alt_deg, az_deg) in test_cases {
            let solar = LocalSolarPosition {
                altitude: alt_deg.to_radians(),
                relative_azimuth: az_deg.to_radians(),
            };
            let shaded = calculate_shaded_fraction(&window, Some(&overhang), &[], &solar);
            println!("{:<20} {:>8.1} {:>12.2}", label, alt_deg, shaded);
        }
        println!();
    }

    println!("RECOMMENDATION:");
    println!("  distance_above = 0.5m gives:");
    println!("    - Summer: ~100% shading (blocks cooling load)");
    println!("    - Winter: ~0% shading (allows heating gain)");
    println!("  This matches ASHRAE 140 intent for Case 610");
}
