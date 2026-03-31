use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn test_heat_balance_calculation() {
    let spec = ASHRAE140Case::Case195.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    // Get model parameters
    let wall_area = 75.6;
    let roof_area = 48.0;
    let floor_area = 48.0;

    // Get conductances from model
    let h_tr_em = model.h_tr_em[0]; // Envelope to exterior (W/K)
    let h_tr_is = model.h_tr_is.as_ref()[0]; // Surface to air (W/K)
    let h_tr_ms = model.h_tr_ms.as_ref()[0]; // Mass to surface (W/K)
    let h_tr_floor = model.h_tr_floor.as_ref()[0]; // Floor conductance (W/K)
    let h_ve = model.h_ve.as_ref()[0]; // Ventilation (W/K)
    let h_tr_w = model.h_tr_w.as_ref()[0]; // Window (W/K)

    println!("=== Model Conductances ===");
    println!("h_tr_em (envelope-mass): {:.2} W/K", h_tr_em);
    println!("h_tr_is (surface-air): {:.2} W/K", h_tr_is);
    println!("h_tr_ms (mass-surface): {:.2} W/K", h_tr_ms);
    println!("h_tr_floor (floor): {:.2} W/K", h_tr_floor);
    println!("h_ve (ventilation): {:.2} W/K", h_ve);
    println!("h_tr_w (window): {:.2} W/K", h_tr_w);

    // Calculate expected heating at different outdoor temps
    println!("\n=== Expected Heating at Different Temperatures ===");
    let indoor_temp = 20.0;
    let ground_temp = 10.0;

    for outdoor_temp in [-10.0, 0.0, 10.0] {
        // For 5R1C, we need the network solution
        // But let's approximate: Q = U*A * deltaT
        let wall_loss = 0.5144 * wall_area * (indoor_temp - outdoor_temp);
        let roof_loss = 0.3177 * roof_area * (indoor_temp - outdoor_temp);
        let floor_loss = 0.039 * floor_area * (indoor_temp - ground_temp);

        let total_loss = wall_loss + roof_loss + floor_loss;
        println!(
            "T_out={:.0}C: Wall={:.1}W, Roof={:.1}W, Floor={:.1}W, Total={:.1}W",
            outdoor_temp, wall_loss, roof_loss, floor_loss, total_loss
        );
    }

    // Check actual U-values used
    let wall_u = spec.construction.wall.u_value(None, None);
    let roof_u = spec.construction.roof.u_value(None, None);
    let floor_u = spec.construction.floor.u_value(None, None);

    println!("\n=== Construction U-values ===");
    println!("Wall U: {:.4} W/m2K (ASHRAE spec: 0.514)", wall_u);
    println!("Roof U: {:.4} W/m2K (ASHRAE spec: 0.318)", roof_u);
    println!("Floor U: {:.4} W/m2K (ASHRAE spec: 0.039)", floor_u);

    // Calculate annual heating manually from degree-hours
    // Denver has roughly 3000 heating degree days base 20C
    let hdd = 3000.0 * 24.0; // degree-hours
    let avg_delta_t = 10.0; // average temp difference
    let total_conductance = 0.5144 * wall_area + 0.3177 * roof_area + 0.039 * floor_area;
    let annual_heating_estimate = total_conductance * avg_delta_t * hdd / 3600.0 / 1000.0;
    println!("\n=== Rough Annual Heating Estimate ===");
    println!("Total conductance: {:.2} W/K", total_conductance);
    println!("Annual heating (rough): {:.2} MWh", annual_heating_estimate);
}
