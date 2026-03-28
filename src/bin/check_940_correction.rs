// Check if Case 940 correction is being applied
use fluxion::validation::ashrae_140_cases::CaseBuilder;
use fluxion::sim::engine::ThermalModel;

fn main() {
    let spec = CaseBuilder::case_940_setback();
    let model = ThermalModel::from_spec(&spec);

    println!("=== Case 940 Correction Check ===");
    println!("Case ID: {}", spec.case_id);
    println!("Time constant sensitivity correction: {:.2}", model.time_constant_sensitivity_correction);

    // Expected: 1.5
    if model.time_constant_sensitivity_correction >= 1.45 && model.time_constant_sensitivity_correction <= 1.55 {
        println!("✅ Correction is correctly set to 1.5");
    } else {
        println!("❌ Correction is NOT set correctly (expected 1.5, got {:.2})", model.time_constant_sensitivity_correction);
    }
}
