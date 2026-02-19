use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::CaseSpec;
use fluxion::physics::cta::VectorField;

fn main() {
    let spec = CaseSpec::case_600_baseline();
    let mut model = ThermalModel::<VectorField>::from_spec(&spec);
    
    let outdoor_temp = 0.0; // Very cold, should trigger heating
    let timestep = 0;
    
    // Set temperatures below setpoint (20.0)
    model.temperatures = VectorField::from_scalar(15.0, model.num_zones);
    model.mass_temperatures = VectorField::from_scalar(15.0, model.num_zones);
    
    // Internal loads for Case 600: 200W
    // floor_area = 48m2
    // load_per_m2 = 200 / 48 = 4.166...
    // Already set by from_spec
    
    println!("Initial temp: {:?}", model.temperatures);
    println!("Heating setpoint: {}", model.heating_setpoint);
    
    let energy = model.step_physics(timestep, outdoor_temp);
    println!("Energy from step_physics: {}", energy);
    
    // Calculate expected heating
    // Heating needed to raise from 15 to 20.
    // Plus heat loss to outside (0C)
    // Plus internal gains (200W)
    
    let dt = 3600.0;
    let heating_joules = energy;
    let heating_kwh = heating_joules / 3.6e6;
    println!("Energy in kWh: {}", heating_kwh);
    
    let hvac_watts = heating_kwh * 1000.0; // This is what the validator does
    println!("HVAC Watts calculated by validator: {}", hvac_watts);
}
