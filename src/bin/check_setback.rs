// Quick diagnostic to check setback schedule for Case 940
use fluxion::validation::ashrae_140_cases::CaseBuilder;
use fluxion::sim::engine::ThermalModel;

fn main() {
    let spec = CaseBuilder::case_940_setback();
    println!("=== Case 940 Setback Schedule Check ===");
    println!("Case ID: {}", spec.case_id);
    println!("Heating setpoint: {}", spec.hvac[0].heating_setpoint);
    println!("Cooling setpoint: {}", spec.hvac[0].cooling_setpoint);
    println!("Setback setpoint: {:?}", spec.hvac[0].setback_setpoint);
    println!("Setback hours: {:?}", spec.hvac[0].setback_hours);

    let model = ThermalModel::from_spec(&spec);
    println!("\n=== Heating Schedule (24 hours) ===");
    for hour in 0..24 {
        let setpoint = model.heating_schedule.value(hour);
        let status = if setpoint == 10.0 { "SETBACK" } else { "NORMAL" };
        println!("Hour {:2}: Setpoint = {:.1}°C [{}]", hour, setpoint, status);
    }

    // Count setback hours
    let setback_hours = (0..24)
        .filter(|&h| model.heating_schedule.value(h) == 10.0)
        .count();
    println!("\nTotal setback hours: {} (expected: 8)", setback_hours);
}
