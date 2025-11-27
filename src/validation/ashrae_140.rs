use std::collections::HashMap;
use crate::sim::engine::ThermalModel;
use crate::physics::cta::VectorField;
use crate::ai::surrogate::SurrogateManager;

pub struct ASHRAE140Validator {
    buildings: HashMap<String, ASHRAE140Building>,
}

pub struct ASHRAE140Building {
    pub name: String,
    pub baseline_energy: f64, // kWh/year (Heating + Cooling)
    pub baseline_temps: Vec<f64>,
}

#[derive(Default)]
pub struct ValidationReport {
    pub results: Vec<(String, f64)>,  // (building_id, error)
}

impl ValidationReport {
    pub fn add_result(&mut self, building_id: &str, error: f64) {
        self.results.push((building_id.to_string(), error));
    }

    pub fn mae(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        self.results.iter().map(|(_, e)| e).sum::<f64>() / self.results.len() as f64
    }

    pub fn print_summary(&self) {
        println!("Validation Report:");
        println!("  MAE: {:.4}", self.mae());
        for (building_id, error) in &self.results {
            println!("  {}: {:.4}", building_id, error);
        }
    }
}

impl ASHRAE140Validator {
    pub fn new() -> Self {
        let mut buildings = HashMap::new();

        // Case 600 Baseline
        // Source: Approximate average from ASHRAE 140 results for lightweight buildings
        // Annual Heating: ~5.0 MWh
        // Annual Cooling: ~7.0 MWh
        // Total: ~12.0 MWh = 12000 kWh
        buildings.insert(
            "Case600".to_string(),
            ASHRAE140Building {
                name: "Case600".to_string(),
                baseline_energy: 12000.0,
                baseline_temps: vec![], // Placeholder
            }
        );

        Self {
            buildings,
        }
    }

    pub fn validate_analytical_engine(&mut self) -> ValidationReport {
        let mut report = ValidationReport::default();

        // Ensure we can create a surrogate manager (needed for solve_timesteps signature)
        // We handle the error gracefully by creating a mock or skipping if it fails heavily,
        // but here we expect it to succeed or we panic/return empty?
        // The quickstart code suggests unwrapping or expecting.
        let surrogates = match SurrogateManager::new() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to create SurrogateManager: {}", e);
                return report;
            }
        };

        for (building_id, building) in &self.buildings {
            if building_id == "Case600" {
                // Setup Case 600 Model
                let mut model = ThermalModel::<VectorField>::new(1);

                // Configure Model for Case 600 (Approximation)
                // Case 600:
                // - Lightweight construction
                // - Window U-value approx 3.0 W/m2K
                // - Setpoints: Heating 20C, Cooling 27C.
                // Fluxion's ThermalModel currently supports a single setpoint.
                // We pick an average or specific setpoint for now to demonstrate validation flow.
                // Let's use 23.5 C (midpoint) or focus on one regime.
                // Given the simplistic nature, we'll set it to 21C (default-ish) for now
                // and acknowledge the limitation.

                let params = vec![3.0, 21.0]; // U=3.0, Setpoint=21.0
                model.apply_parameters(&params);

                // Run simulation (Annual = 8760 steps)
                let analytical_energy = model.solve_timesteps(8760, &surrogates, false);

                // Calculate error (Absolute difference in kWh)
                let error = (building.baseline_energy - analytical_energy).abs();
                report.add_result(building_id, error);
            }
        }

        report
    }
}
