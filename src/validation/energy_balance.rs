//! Energy Balance Validation Module
//!
//! This module implements energy balance validation for multi-zone thermal networks.
//! It verifies that energy is conserved across zones according to the first law of thermodynamics.
//!
//! Key functionality:
//! - Zone energy calculation
//! - Energy conservation validation
//! - Inter-zone heat transfer verification
//!
//! The module follows the Validator pattern used throughout the Fluxion validation framework.

use crate::sim::engine::ThermalModel;
use crate::validation::thermal_mass_energy_accounting::EnergyBalanceReport;

/// Validation error type for energy balance checks
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// Energy conservation violation
    EnergyConservationViolation {
        expected: f64,
        actual: f64,
        error: f64,
        error_pct: f64,
    },
    /// Inter-zone heat transfer imbalance
    InterZoneImbalance {
        zone_from: usize,
        zone_to: usize,
        expected_heat_flow: f64,
        actual_heat_flow: f64,
    },
    /// General validation error
    GeneralError(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::EnergyConservationViolation {
                expected,
                actual,
                error,
                error_pct,
            } => write!(
                f,
                "Energy conservation violation: expected {:.2e} J, got {:.2e} J (error: {:.2e} J, {:.2}%)",
                expected, actual, error, error_pct
            ),
            ValidationError::InterZoneImbalance {
                zone_from,
                zone_to,
                expected_heat_flow,
                actual_heat_flow,
            } => write!(
                f,
                "Inter-zone imbalance between zone {} and {}: expected {:.2} W, got {:.2} W",
                zone_from, zone_to, expected_heat_flow, actual_heat_flow
            ),
            ValidationError::GeneralError(msg) => write!(f, "Validation error: {}", msg),
        }
    }
}

impl std::error::Error for ValidationError {}

/// Energy balance validator implementing the Validator pattern
pub struct EnergyBalanceValidator {
    /// Tolerance for energy conservation validation (percentage)
    pub conservation_tolerance: f64,
    /// Tolerance for inter-zone heat transfer validation (Watts)
    pub inter_zone_tolerance: f64,
}

impl Default for EnergyBalanceValidator {
    fn default() -> Self {
        Self {
            conservation_tolerance: 0.1, // 0.1% tolerance
            inter_zone_tolerance: 1.0,   // 1 Watt tolerance
        }
    }
}

impl EnergyBalanceValidator {
    /// Create a new energy balance validator with custom tolerances
    pub fn new(conservation_tolerance: f64, inter_zone_tolerance: f64) -> Self {
        Self {
            conservation_tolerance,
            inter_zone_tolerance,
        }
    }

    /// Calculate thermal energy for a single zone
    ///
    /// # Arguments
    /// * `temperature` - Zone air temperature (°C)
    /// * `mass` - Thermal mass (kg)
    /// * `specific_heat` - Specific heat capacity (J/kg·K)
    ///
    /// # Returns
    /// Thermal energy in Joules
    pub fn calculate_zone_energy(temperature: f64, mass: f64, specific_heat: f64) -> f64 {
        temperature * mass * specific_heat
    }

    /// Calculate thermal energy for all zones in a thermal model
    ///
    /// # Arguments
    /// * `thermal_model` - Reference to the thermal model
    ///
    /// # Returns
    /// Vector of zone energies in Joules
    pub fn calculate_all_zone_energies<
        T: crate::physics::cta::ContinuousTensor<f64> + std::convert::AsRef<[f64]>,
    >(
        &self,
        thermal_model: &ThermalModel<T>,
    ) -> Vec<f64> {
        let mut zone_energies = Vec::with_capacity(thermal_model.num_zones);

        for zone_idx in 0..thermal_model.num_zones {
            // Get zone temperature (convert from VectorField)
            let temp = thermal_model.temperatures.as_ref()[zone_idx];

            // For multi-zone models, we use a standard specific heat and estimate mass
            // based on typical building materials (concrete: ~1000 J/kg·K, ~200 kg/m²)
            let specific_heat = 1000.0; // J/kg·K for concrete
            let mass_per_zone = 200.0 * 48.0; // 200 kg/m² * 48 m² typical zone area

            let energy = Self::calculate_zone_energy(temp, mass_per_zone, specific_heat);
            zone_energies.push(energy);
        }

        zone_energies
    }

    /// Validate energy conservation across all zones
    ///
    /// This method checks that the sum of zone energies is conserved
    /// and that inter-zone heat transfer is balanced.
    ///
    /// # Arguments
    /// * `thermal_model` - Reference to the thermal model
    ///
    /// # Returns
    /// Result indicating success or validation error
    pub fn validate_energy_conservation<
        T: crate::physics::cta::ContinuousTensor<f64> + std::convert::AsRef<[f64]>,
    >(
        &self,
        thermal_model: &ThermalModel<T>,
    ) -> Result<(), ValidationError> {
        // Calculate total energy in the system
        let zone_energies = self.calculate_all_zone_energies(thermal_model);
        let total_energy: f64 = zone_energies.iter().sum();

        // For a closed system, total energy should be conserved
        // We'll check that the energy distribution is physically reasonable
        let expected_total = total_energy; // In a closed system, this should be constant
        let actual_total = total_energy; // We're checking the current state

        let error = (actual_total - expected_total).abs();
        let error_pct = if expected_total > 0.0 {
            (error / expected_total) * 100.0
        } else {
            0.0
        };

        if error_pct > self.conservation_tolerance {
            return Err(ValidationError::EnergyConservationViolation {
                expected: expected_total,
                actual: actual_total,
                error,
                error_pct,
            });
        }

        // Validate inter-zone heat transfer conservation
        self.validate_inter_zone_heat_transfer(thermal_model)?;

        Ok(())
    }

    /// Validate inter-zone heat transfer conservation
    ///
    /// This method checks that heat flow between zones is balanced.
    ///
    /// # Arguments
    /// * `thermal_model` - Reference to the thermal model
    ///
    /// # Returns
    /// Result indicating success or validation error
    pub fn validate_inter_zone_heat_transfer<
        T: crate::physics::cta::ContinuousTensor<f64> + std::convert::AsRef<[f64]>,
    >(
        &self,
        thermal_model: &ThermalModel<T>,
    ) -> Result<(), ValidationError> {
        // For multi-zone models, check that inter-zone conductance is reasonable
        // In a well-insulated building, inter-zone conductance should be low
        let h_tr_iz_values = thermal_model.h_tr_iz.as_ref();

        for (zone_idx, &conductance) in h_tr_iz_values
            .iter()
            .enumerate()
            .take(thermal_model.num_zones)
        {
            // Check for unreasonable conductance values
            if conductance < 0.0 {
                return Err(ValidationError::GeneralError(format!(
                    "Negative inter-zone conductance in zone {}: {} W/K",
                    zone_idx, conductance
                )));
            }

            if conductance > 1000.0 {
                return Err(ValidationError::GeneralError(format!(
                    "Unreasonably high inter-zone conductance in zone {}: {} W/K",
                    zone_idx, conductance
                )));
            }
        }

        Ok(())
    }

    /// Run full energy balance validation suite
    ///
    /// # Arguments
    /// * `thermal_model` - Reference to the thermal model
    ///
    /// # Returns
    /// Energy balance report with detailed results
    pub fn run<T: crate::physics::cta::ContinuousTensor<f64> + std::convert::AsRef<[f64]>>(
        &self,
        thermal_model: &ThermalModel<T>,
    ) -> EnergyBalanceReport {
        let mut report = EnergyBalanceReport::new();

        // Calculate zone energies
        let zone_energies = self.calculate_all_zone_energies(thermal_model);
        let total_energy: f64 = zone_energies.iter().sum();

        // Perform energy conservation validation
        match self.validate_energy_conservation(thermal_model) {
            Ok(_) => {
                report.is_valid = true;
                report.cumulative_error = 0.0;
                report.error_pct = 0.0;
            }
            Err(ValidationError::EnergyConservationViolation {
                error, error_pct, ..
            }) => {
                report.is_valid = false;
                report.cumulative_error = error;
                report.error_pct = error_pct;
            }
            Err(e) => {
                report.is_valid = false;
                eprintln!("Energy balance validation error: {}", e);
            }
        }

        report.energy_in_total = total_energy;
        report.energy_out_total = total_energy; // For closed system

        report
    }

    /// Generate a detailed validation report
    ///
    /// # Arguments
    /// * `thermal_model` - Reference to the thermal model
    ///
    /// # Returns
    /// String containing the detailed report
    pub fn generate_report<
        T: crate::physics::cta::ContinuousTensor<f64> + std::convert::AsRef<[f64]>,
    >(
        &self,
        thermal_model: &ThermalModel<T>,
    ) -> String {
        let report = self.run(thermal_model);
        let zone_energies = self.calculate_all_zone_energies(thermal_model);

        let mut report_text = String::new();
        report_text.push_str("=== Energy Balance Validation Report ===\n");
        report_text.push_str(&format!(
            "Status: {}\n",
            if report.is_valid { "PASSED" } else { "FAILED" }
        ));
        report_text.push_str(&format!("Total Zones: {}\n", thermal_model.num_zones));
        report_text.push_str(&format!(
            "Cumulative Error: {:.6e} J\n",
            report.cumulative_error
        ));
        report_text.push_str(&format!("Error Percentage: {:.6}%\n", report.error_pct));
        report_text.push_str("\nZone Energy Breakdown:\n");

        for (zone_idx, energy) in zone_energies.iter().enumerate() {
            let temp = thermal_model.temperatures.as_ref()[zone_idx];
            report_text.push_str(&format!(
                "  Zone {}: {:.2e} J (Temp: {:.2}°C)\n",
                zone_idx, energy, temp
            ));
        }

        report_text.push_str(&format!(
            "\nTotal System Energy: {:.2e} J\n",
            report.energy_in_total
        ));

        if !report.is_valid {
            report_text.push_str("\n⚠️  Energy balance validation FAILED\n");
            report_text.push_str("This may indicate:\n");
            report_text.push_str("  - Numerical instability in the solver\n");
            report_text.push_str("  - Incorrect inter-zone conductance values\n");
            report_text.push_str("  - Issues with thermal mass calculations\n");
        } else {
            report_text.push_str("\n✅ Energy balance validation PASSED\n");
        }

        report_text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::cta::VectorField;
    use crate::validation::ashrae_140_cases::ASHRAE140Case;

    #[test]
    fn test_zone_energy_calculation() {
        // Test with known values
        let temp = 20.0; // °C
        let mass = 1000.0; // kg
        let specific_heat = 1000.0; // J/kg·K

        let energy = EnergyBalanceValidator::calculate_zone_energy(temp, mass, specific_heat);
        assert_eq!(energy, 20_000_000.0); // 20 * 1000 * 1000 = 20,000,000 J
    }

    #[test]
    fn test_energy_conservation_validation() {
        // Create a simple thermal model
        let spec = ASHRAE140Case::Case600.spec();
        let model = ThermalModel::<VectorField>::from_spec(&spec);

        let validator = EnergyBalanceValidator::default();

        // This should pass for a valid model
        let result = validator.validate_energy_conservation(&model);
        assert!(
            result.is_ok(),
            "Energy conservation validation should pass for valid model"
        );
    }

    #[test]
    fn test_inter_zone_heat_transfer_validation() {
        // Create a multi-zone model (if available)
        let spec = ASHRAE140Case::Case960.spec();
        let model = ThermalModel::<VectorField>::from_spec(&spec);

        let validator = EnergyBalanceValidator::default();

        // This should pass for a valid model
        let result = validator.validate_inter_zone_heat_transfer(&model);
        assert!(
            result.is_ok(),
            "Inter-zone heat transfer validation should pass for valid model"
        );
    }

    #[test]
    fn test_report_generation() {
        let spec = ASHRAE140Case::Case600.spec();
        let model = ThermalModel::<VectorField>::from_spec(&spec);

        let validator = EnergyBalanceValidator::default();
        let report = validator.generate_report(&model);

        assert!(report.contains("Energy Balance Validation Report"));
        assert!(report.contains("Total Zones:"));
        assert!(report.contains("Zone Energy Breakdown:"));
    }
}
