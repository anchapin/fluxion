//! Debug test to verify window areas are correctly set for multi-zone buildings.

use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::{ASHRAE140Case, Orientation};

#[test]
fn test_case_960_window_areas_from_spec() {
    let spec = ASHRAE140Case::Case960.spec();

    println!("\n=== Case 960 Spec Window Areas ===");
    for zone_idx in 0..spec.num_zones {
        let mut total_area = 0.0;
        for orientation in [
            Orientation::South,
            Orientation::West,
            Orientation::North,
            Orientation::East,
        ] {
            let area = spec.window_area_by_zone_and_orientation(zone_idx, orientation);
            total_area += area;
            println!("Zone {} {:?}: {:.2} m²", zone_idx, orientation, area);
        }
        println!("Zone {} Total: {:.2} m²", zone_idx, total_area);
    }
    println!("=== End ===\n");
}

#[test]
fn test_case_960_window_areas_from_model() {
    let spec = ASHRAE140Case::Case960.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("\n=== Case 960 Model Window Conductance ===");
    let h_tr_w = model.h_tr_w.as_ref();
    for zone_idx in 0..model.num_zones {
        println!("Zone {} h_tr_w: {:.2} W/K (U={:.2})", zone_idx, h_tr_w[zone_idx], spec.window_properties.u_value);
    }
    println!("=== End ===\n");
}
