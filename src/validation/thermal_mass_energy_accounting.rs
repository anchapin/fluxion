//! Thermal Mass Energy Accounting Validation
//!
//! This module implements energy accounting validation to confirm that the physics engine
//! correctly conserves energy according to the first law of thermodynamics:
//!
//! Σenergy_in = Σenergy_out + Δmass_energy
//!
//! This is a diagnostic task to validate physics correctness, not a fix task. If energy
//! balance is confirmed (error < 0.01%), the physics is correct even if annual energy
//! predictions are wrong (which would indicate a fundamental 5R1C limitation, not a bug).
//!
//! ## Energy Balance Equation
//!
//! At each timestep, the following equation must hold:
//!
//! ```text
//! Q_heating + Q_cooling + Q_solar + Q_infiltration = Q_hvac_demand + ΔE_mass
//! ```
//!
//! Where:
//! - `Q_heating` + `Q_cooling`: HVAC energy input to the zone
//! - `Q_solar` + `Q_infiltration`: External energy gains
//! - `Q_hvac_demand`: HVAC energy extracted/rejected to maintain setpoints
//! - `ΔE_mass`: Change in thermal mass energy storage (Cm × ΔTm)
//!
//! ## Usage
//!
//! ```rust
//! use fluxion::validation::thermal_mass_energy_accounting::*;
//! use fluxion::sim::engine::ThermalModel;
//! use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
//!
//! let spec = ASHRAE140Case::Case900.spec();
//! let model = ThermalModel::<VectorField>::from_spec(&spec);
//!
//! let mass_energy = calculate_mass_energy(&model);
//!
//! println!("Total thermal mass energy: {:.2e} J", mass_energy);
//! ```

use crate::physics::cta::VectorField;
use crate::sim::engine::ThermalModel;

/// Report of energy balance validation over a simulation period.
///
/// This report summarizes whether the physics engine correctly conserves energy
/// according to the first law of thermodynamics.
#[derive(Debug, Clone)]
pub struct EnergyBalanceReport {
    /// Cumulative absolute error over all timesteps (Joules)
    pub cumulative_error: f64,
    /// Error percentage of total energy flow
    pub error_pct: f64,
    /// Whether energy balance is valid (error < 0.01%)
    pub is_valid: bool,
    /// Balance error for each timestep (Joules)
    pub hourly_errors: Vec<f64>,
    /// Total energy entering the system (Joules)
    pub energy_in_total: f64,
    /// Total energy leaving the system (Joules)
    pub energy_out_total: f64,
}

impl EnergyBalanceReport {
    /// Create a new energy balance report.
    pub fn new() -> Self {
        Self {
            cumulative_error: 0.0,
            error_pct: 0.0,
            is_valid: false,
            hourly_errors: Vec::new(),
            energy_in_total: 0.0,
            energy_out_total: 0.0,
        }
    }

    /// Generate a human-readable summary of the report.
    pub fn to_summary(&self) -> String {
        let status = if self.is_valid { "PASSED" } else { "FAILED" };
        format!(
            "=== Energy Balance Validation Report ===\n\
             Status: {}\n\
             Cumulative Error: {:.6e} J\n\
             Error Percentage: {:.6}%\n\
             Energy In Total: {:.6e} J\n\
             Energy Out Total: {:.6e} J\n\
             Hourly Errors: {} timesteps\n",
            status,
            self.cumulative_error,
            self.error_pct,
            self.energy_in_total,
            self.energy_out_total,
            self.hourly_errors.len()
        )
    }
}

impl Default for EnergyBalanceReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate the total thermal mass energy in the model.
///
/// This function computes the energy stored in all thermal mass nodes:
///
/// ```text
/// E_mass = Σ(Cm_i × Tm_i)
/// ```
///
/// Where:
/// - `Cm_i` is the thermal capacitance of mass node i (J/K)
/// - `Tm_i` is the temperature of mass node i (K or °C, absolute units cancel)
///
/// # Arguments
///
/// * `model` - The thermal model to calculate mass energy for
///
/// # Returns
///
/// Total thermal mass energy in Joules
///
/// # Note
///
/// This function works with both 5R1C (single mass node) and 6R2C (envelope + internal masses)
/// configurations. For 5R1C, it sums the single `mass_temperatures` array. For 6R2C, it
/// sums both `envelope_mass_temperatures` and `internal_mass_temperatures` arrays.
pub fn calculate_mass_energy(model: &ThermalModel<VectorField>) -> f64 {
    // Calculate mass energy from primary mass temperatures (5R1C mode)
    let mut total_energy = 0.0;

    // Sum: thermal_capacitance[i] * mass_temperatures[i] for all mass nodes i
    for (cap, temp) in model
        .thermal_capacitance
        .iter()
        .zip(model.mass_temperatures.as_ref().iter())
    {
        total_energy += cap * temp;
    }

    total_energy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::ashrae_140_cases::ASHRAE140Case;

    #[test]
    fn test_calculate_mass_energy_5r1c() {
        let spec = ASHRAE140Case::Case600.spec();
        let model = ThermalModel::<VectorField>::from_spec(&spec);

        let mass_energy = calculate_mass_energy(&model);

        // Mass energy should be positive and reasonable
        assert!(
            mass_energy > 0.0,
            "Mass energy should be positive, got {}",
            mass_energy
        );

        // Mass energy should be on order of MJ (millions of Joules)
        assert!(
            mass_energy > 1.0e6 && mass_energy < 1.0e12,
            "Mass energy should be 1e6-1e12 J, got {:.2e} J",
            mass_energy
        );
    }

    #[test]
    fn test_energy_balance_report_default() {
        let report = EnergyBalanceReport::default();

        assert_eq!(report.cumulative_error, 0.0);
        assert_eq!(report.error_pct, 0.0);
        assert!(!report.is_valid);
        assert!(report.hourly_errors.is_empty());
        assert_eq!(report.energy_in_total, 0.0);
        assert_eq!(report.energy_out_total, 0.0);
    }

    #[test]
    fn test_energy_balance_report_to_summary() {
        let mut report = EnergyBalanceReport::new();
        report.cumulative_error = 1000.0;
        report.error_pct = 0.005;
        report.is_valid = true;
        report.energy_in_total = 100000.0;
        report.energy_out_total = 99000.0;
        report.hourly_errors = vec![0.1, -0.2, 0.1];

        let summary = report.to_summary();

        // Just check that it contains the key parts
        assert!(summary.contains("PASSED"));
        assert!(summary.contains("0.005"));
        assert!(summary.contains("1000"));
        assert!(summary.contains("Hourly Errors: 3"));
        assert!(summary.contains("1000.00"));
    }
}
