use crate::ai::surrogate::SurrogateManager;
use crate::physics::cta::{ContinuousTensor, VectorField};
use crate::sim::engine::ThermalModel;
use crate::validation::benchmark;
use crate::validation::report::{BenchmarkReport, MetricType};

/// Validator for ASHRAE 140 standard cases.
pub struct ASHRAE140Validator;

impl Default for ASHRAE140Validator {
    fn default() -> Self {
        Self::new()
    }
}

impl ASHRAE140Validator {
    /// Creates a new ASHRAE 140 validator.
    pub fn new() -> Self {
        Self {}
    }

    /// Validates the analytical engine against the ASHRAE 140 cases.
    pub fn validate_analytical_engine(&mut self) -> BenchmarkReport {
        let mut report = BenchmarkReport::new();
        let benchmark_data = benchmark::get_all_benchmark_data();

        // Ensure we can create a surrogate manager (needed for solve_timesteps signature)
        let surrogates = match SurrogateManager::new() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to create SurrogateManager: {}", e);
                return report;
            }
        };

        // Case 600
        if let Some(data) = benchmark_data.get("600") {
            // Setup Case 600 Model
            let mut model = ThermalModel::<VectorField>::new(1);

            // Configure Model for Case 600 (Approximation)
            // Case 600: U=3.0, Setpoint=21.0 (Simplified)
            let params = vec![3.0, 21.0];
            model.apply_parameters(&params);

            // Run simulation (Annual = 8760 steps)
            let analytical_eui = model.solve_timesteps(8760, &surrogates, false);

            // Convert EUI (kWh/m²/yr) back to Total Energy (kWh)
            let total_area = model.zone_area.integrate();
            let analytical_energy_kwh = analytical_eui * total_area;
            let analytical_energy_mwh = analytical_energy_kwh / 1000.0;

            // Report as Annual Heating (Approximation for this simplified model)
            report.add_result_simple(
                "600",
                MetricType::AnnualHeating,
                analytical_energy_mwh,
                data.annual_heating_min,
                data.annual_heating_max,
            );
            
            // Add benchmark data
            report.add_benchmark_data("600", data.clone());
        }

        report
    }
}
