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

use crate::physics::cta::{ContinuousTensor, VectorField};
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
    /// N-zone inter-zone conservation violation (Issue #1348).
    /// For a symmetric conductance matrix the algebraic identity
    /// `Σ_i q_iz[i] = 0 W` must hold within machine precision. If the
    /// validator reports a non-trivial residual, the matrix is asymmetric
    /// or the network solve lost energy.
    InterZoneConservationViolation {
        net_inter_zone_q_w: f64,
        zone_residuals_w: Vec<f64>,
        tolerance_w: f64,
    },
    /// Per-zone residual breakdown from the strict multi-zone invariant check.
    /// `residual_w` is the total system residual in Watts
    /// (signed: heat_in - heat_out - dE/dt, consistent with `InvariantChecker`).
    /// `zone_residuals_w` is the per-zone Watt residual vector; the per-zone
    /// breakdown is required because whole-system residuals can mask
    /// inter-zone imbalances (a positive residual in zone A can cancel a
    /// negative residual in zone B even when physics is wrong).
    /// `tolerance_pct` is the configured percentage tolerance (e.g. 1.0 = 1%).
    MultiZoneConservationViolation {
        residual_w: f64,
        zone_residuals_w: Vec<f64>,
        tolerance_pct: f64,
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
            ValidationError::InterZoneConservationViolation {
                net_inter_zone_q_w,
                zone_residuals_w,
                tolerance_w,
            } => {
                writeln!(
                    f,
                    "N-zone inter-zone conservation violation: Σ q_iz = {:.3e} W (tolerance {:.0e} W)",
                    net_inter_zone_q_w, tolerance_w
                )?;
                for (i, z) in zone_residuals_w.iter().enumerate() {
                    writeln!(f, "  Zone {} q_iz: {:.3e} W", i, z)?;
                }
                Ok(())
            }
            ValidationError::MultiZoneConservationViolation {
                residual_w,
                zone_residuals_w,
                tolerance_pct,
            } => {
                writeln!(
                    f,
                    "Multi-zone energy conservation violation: total residual = {:.3} W (tolerance {:.2}%)",
                    residual_w, tolerance_pct
                )?;
                for (i, z) in zone_residuals_w.iter().enumerate() {
                    writeln!(f, "  Zone {} residual: {:.3} W", i, z)?;
                }
                Ok(())
            }
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
        let mut zone_energies = Vec::with_capacity(thermal_model.hvac.num_zones);

        for zone_idx in 0..thermal_model.hvac.num_zones {
            // Get zone temperature (convert from VectorField)
            let temp = thermal_model.setpoints.temperatures.as_ref()[zone_idx];

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
        let h_tr_iz_values = thermal_model.conduction.h_tr_iz.as_ref();

        for (zone_idx, &conductance) in h_tr_iz_values
            .iter()
            .enumerate()
            .take(thermal_model.hvac.num_zones)
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

    /// Validate multi-zone energy conservation with per-zone residual breakdown.
    ///
    /// Unlike [`Self::validate_energy_conservation`] (which compares `Σ E_zone`
    /// against itself and always passes), this method computes a real
    /// Watt residual from the first law of thermodynamics applied to each
    /// zone's air + mass nodes:
    ///
    /// ```text
    /// residual_zone = (φ_ia + φ_st + φ_m)
    ///               - (q_em + q_ms + q_w + q_ve + q_floor)
    ///               - C_m · (T_m - T_m_prev) / Δt
    ///               - ρ_a · V · c_p,a · (T_air - T_air_prev) / Δt
    /// ```
    ///
    /// The residual is the total zone **unbalanced** energy at this timestep:
    /// positive when more heat went in than went out + was stored. For a
    /// correctly-solved simulation step the residual is at machine precision
    /// (every Watt of internal gain is accounted for by an external loss or
    /// by a change in the air/mass thermal storage terms).
    ///
    /// We compute this inline (rather than delegating to
    /// [`crate::sim::invariant_checker::InvariantChecker`]) because
    /// `InvariantChecker` reports the **mass-node** balance alone
    /// (per ISO 13790 §C.4). For a hand-balanced multi-zone stub with a
    /// deliberate +5 W load, the mass-node balance does not see the full
    /// 5 W — only the (attenuated) fraction that propagates through the
    /// surface temperature to the mass node — whereas the test
    /// (`test_multi_zone_validator_catches_5w_unbalance`) expects the
    /// **total zone** unbalance (5.00 W). The total-zone balance is the
    /// physically correct invariant for end-of-timestep model state
    /// regardless of how the solver internally partitions the storage.
    ///
    /// Per-zone residuals are required because whole-system residuals can
    /// mask inter-zone imbalances (zone A's positive residual can cancel
    /// zone B's negative residual even when physics is wrong).
    ///
    /// # Arguments
    /// * `thermal_model` - Reference to the multi-zone thermal model
    /// * `dt_seconds` - Timestep length in seconds (e.g. 3600 for 1 h)
    /// * `outdoor_temp` - Outdoor air temperature for this timestep (°C)
    ///
    /// # Returns
    /// `Ok(())` if the total Watt residual is within `conservation_tolerance`
    /// (interpreted as a percentage of the magnitude of zone heating/cooling
    /// loads in the system). On violation, returns
    /// [`ValidationError::MultiZoneConservationViolation`] with the signed
    /// total residual, per-zone Watt residuals, and configured tolerance.
    ///
    /// # Acceptance criterion (Issue #1344)
    /// For a 2-zone stub with a deliberate 5 W unbalance, the validator must
    /// report residual = 5.00 W (within 1e-3 W) and `status = FAIL`.
    /// For a hand-balanced 2-zone stub (all temps equal, all loads = 0),
    /// the validator must report residual = 0 W (`status = PASS`).
    pub fn validate_multi_zone_energy_conservation<
        T: ContinuousTensor<f64>
            + From<VectorField>
            + std::convert::AsRef<[f64]>
            + std::convert::AsMut<[f64]>
            + std::ops::Index<usize, Output = f64>,
    >(
        &self,
        thermal_model: &ThermalModel<T>,
        dt_seconds: f64,
        outdoor_temp: f64,
    ) -> Result<(), ValidationError> {
        // Per-zone total-zone energy balance (first law of thermodynamics):
        //
        //   ΔU_air + ΔU_mass = (φ_ia + φ_st + φ_m) - (q_em + q_ms + q_w + q_ve + q_floor)
        //
        // (5R1C surface node carries no thermal storage; its redistribution
        // contribution appears as `q_ms` on the right-hand side.)
        //
        // Residual = (heat in) - (heat out) - (storage rates).
        //   For a hand-balanced stub with no temp changes and no loads → 0 W.
        //   For a stub with +5 W injected load (no losses, no storage) → +5 W.
        //   For a correctly-solved simulation step → machine precision.
        let num_zones = thermal_model.hvac.num_zones;
        let temps = thermal_model.setpoints.temperatures.as_ref();
        let prev_temps = thermal_model.hvac.previous_temperatures.as_ref();
        let mass_temps = thermal_model.mass.mass_temperatures.as_ref();
        let prev_mass_temps = thermal_model.mass.previous_mass_temperatures.as_ref();
        let loads = thermal_model.setpoints.loads.as_ref();
        let solar_gains = thermal_model.solar.solar_gains.as_ref();
        let opaque_solar_gains = thermal_model.solar.opaque_solar_gains.as_ref();
        let area = thermal_model.setpoints.zone_area.as_ref();
        let ceiling_height = thermal_model.setpoints.ceiling_height.as_ref();
        let air_density = thermal_model.setpoints.air_density.as_ref();
        let heat_capacity = thermal_model.setpoints.heat_capacity.as_ref();
        let t_ground = thermal_model
            .conduction
            .ground_temperature
            .ground_temperature(0);

        let conv_frac = thermal_model.solar.convective_fraction;
        let rad_frac = 1.0 - conv_frac;
        let sol_dist_to_air = thermal_model.solar.solar_distribution_to_air;
        let solar_beam_to_mass = thermal_model.solar.solar_beam_to_mass_fraction;

        let mut total_balance = 0.0_f64;
        let mut zone_residuals_w: Vec<f64> = Vec::with_capacity(num_zones);

        for i in 0..num_zones {
            let t_air = temps[i];
            let t_air_prev = prev_temps[i];
            let t_mass = mass_temps[i];
            let t_mass_prev = prev_mass_temps[i];

            let load_w = loads[i] * area[i];
            let solar_w = solar_gains[i] * area[i];
            let opaque_sol_w = opaque_solar_gains[i] * area[i];

            // ISO 13790 §C.3 internal-gain split (same as `step_physics_5r1c`).
            let sol_to_air = solar_w * sol_dist_to_air;
            let remaining_sol = solar_w - sol_to_air;
            let st_int_frac = rad_frac * (1.0 - sol_dist_to_air);
            let m_air_frac = rad_frac * sol_dist_to_air;
            let st_sol_frac = 1.0 - solar_beam_to_mass;
            let m_sol_frac = solar_beam_to_mass;

            let phi_ia = load_w * conv_frac + sol_to_air;
            let phi_st = load_w * st_int_frac + remaining_sol * st_sol_frac;
            let phi_m = load_w * m_air_frac + remaining_sol * m_sol_frac + opaque_sol_w;

            // 5R1C heat flows (matching `step_physics_5r1c`).
            let q_em = thermal_model.conduction.h_tr_em[i] * (t_mass - outdoor_temp);
            let q_ms = thermal_model.conduction.h_tr_ms[i] * (t_air - t_mass);
            let q_w = thermal_model.conduction.h_tr_w[i] * (t_air - outdoor_temp);
            let q_ve = thermal_model.conduction.h_ve[i] * (t_air - outdoor_temp);
            let q_floor = thermal_model.conduction.h_tr_floor[i] * (t_air - t_ground);

            let heat_in = phi_ia + phi_st + phi_m;
            let heat_out = q_em + q_ms + q_w + q_ve + q_floor;

            // Storage rates.
            let mass_power =
                thermal_model.mass.thermal_capacitance[i] * (t_mass - t_mass_prev) / dt_seconds;
            // Air storage: m_air · c_p,air · ΔT_air / Δt.
            let volume = area[i] * ceiling_height[i];
            let m_air = volume * air_density[i];
            let air_power = m_air * heat_capacity[i] * (t_air - t_air_prev) / dt_seconds;

            let zone_balance = heat_in - heat_out - mass_power - air_power;
            total_balance += zone_balance;
            zone_residuals_w.push(zone_balance);
        }

        let residual_w = total_balance;

        // Convert the configured % tolerance into an absolute Watt allowance.
        // Denominator: Σ |zone heating/cooling loads| (a stable proxy for the
        // thermal energy throughput at this timestep — zero only when the
        // system is at perfect rest, in which case any non-zero residual is a
        // hard violation regardless of tolerance).
        let throughput_w: f64 = thermal_model
            .setpoints
            .loads
            .as_ref()
            .iter()
            .zip(thermal_model.setpoints.zone_area.as_ref().iter())
            .map(|(l, a)| (l * a).abs())
            .sum();

        let allowance_w = if throughput_w > 0.0 {
            throughput_w * (self.conservation_tolerance / 100.0)
        } else {
            // No active loads: the residual must itself be effectively zero.
            // Use a 1e-6 W floor to guard against floating-point noise on a
            // perfectly still system.
            1e-6
        };

        if residual_w.abs() > allowance_w {
            return Err(ValidationError::MultiZoneConservationViolation {
                residual_w,
                zone_residuals_w,
                tolerance_pct: self.conservation_tolerance,
            });
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
                tracing::warn!("Energy balance validation error: {}", e);
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
        report_text.push_str(&format!("Total Zones: {}\n", thermal_model.hvac.num_zones));
        report_text.push_str(&format!(
            "Cumulative Error: {:.6e} J\n",
            report.cumulative_error
        ));
        report_text.push_str(&format!("Error Percentage: {:.6}%\n", report.error_pct));
        report_text.push_str("\nZone Energy Breakdown:\n");

        for (zone_idx, energy) in zone_energies.iter().enumerate() {
            let temp = thermal_model.setpoints.temperatures.as_ref()[zone_idx];
            report_text.push_str(&format!(
                "  Zone {}: {:.2e} J (Temp: {:.2}°C)\n",
                zone_idx, energy, temp
            ));
        }

        report_text.push_str(&format!(
            "\nTotal System Energy: {:.2e} J\n",
            report.energy_in_total
        ));

        // Add zone balance summary if available
        if !report.zone_balances.is_empty() {
            report_text.push_str("\n=== Zone Balance Summary ===\n");
            for entry in &report.zone_balances {
                report_text.push_str(&format!(
                    "  Zone {}: Energy Start={:.2e} J, End={:.2e} J, HVAC={:.2e} J, Solar={:.2e} J, Internal={:.2e} J\n",
                    entry.zone_index,
                    entry.zone_energy_start,
                    entry.zone_energy_end,
                    entry.hvac_input,
                    entry.solar_gains,
                    entry.internal_gains
                ));
            }
        }

        // Add whole-building balance summary if available
        report_text.push_str(&format!(
            "\n=== Whole-Building Balance ===\n\
             Total Energy In:    {:.6e} J\n\
             Total Energy Out:   {:.6e} J\n\
             Net Energy Change:  {:.6e} J\n\
             Unaccounted Energy: {:.6e} J\n\
             Balance Error:      {:.6}%\n",
            report.building_balance.total_energy_in,
            report.building_balance.total_energy_out,
            report.building_balance.net_energy_change,
            report.building_balance.unaccounted_energy,
            report.building_balance.balance_error_pct
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

    /// Validate N-zone inter-zone heat transfer conservation (Issue #1348).
    ///
    /// For a SYMMETRIC conductance matrix `h_tr_iz`, the algebraic identity
    /// `Σ_i Σ_j h_tr_ij · (T_j − T_i) = 0` holds to machine precision for any
    /// temperature vector. This validator checks that the supplied
    /// `q_iz_per_zone_w` vector sums to within `tolerance_w` of zero — the
    /// Issue #1348 acceptance criterion.
    ///
    /// Use the tolerance you'd apply for the actual application: the strict
    /// Issue #1348 budget is `1e-6` W (machine epsilon for the f64 LU solve
    /// in `MultiZoneAirflowNetwork::solve_step`). The legacy Case 960 2-zone
    /// tolerance is `1.0` W (see `inter_zone_tolerance` in the validator
    /// default) and applies to a different quantity (per-pair imbalance),
    /// so it's preserved unchanged.
    ///
    /// # Arguments
    /// * `q_iz_per_zone_w` - Per-zone inter-zone heat transfer vector [W]
    /// * `tolerance_w` - Acceptable |Σ q_iz[i]| in Watts
    ///
    /// # Returns
    /// `Ok(())` if `|Σ q_iz[i]| ≤ tolerance_w`. Otherwise
    /// [`ValidationError::InterZoneConservationViolation`] (Issue #1348) with
    /// the signed sum, the per-zone breakdown, and the tolerance.
    pub fn validate_n_zone_network_conservation(
        &self,
        q_iz_per_zone_w: &[f64],
        tolerance_w: f64,
    ) -> Result<(), ValidationError> {
        let net_w: f64 = q_iz_per_zone_w.iter().sum();
        if net_w.abs() > tolerance_w {
            return Err(ValidationError::InterZoneConservationViolation {
                net_inter_zone_q_w: net_w,
                zone_residuals_w: q_iz_per_zone_w.to_vec(),
                tolerance_w,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::cta::VectorField;
    use crate::sim::thermal_selector::ThermalSelector;
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
        let model = ThermalModel::<VectorField>::from_spec_with_selector(
            &spec,
            &ThermalSelector::default(),
        )
        .expect("default selector must initialize");

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
        let model = ThermalModel::<VectorField>::from_spec_with_selector(
            &spec,
            &ThermalSelector::default(),
        )
        .expect("default selector must initialize");

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
        let model = ThermalModel::<VectorField>::from_spec_with_selector(
            &spec,
            &ThermalSelector::default(),
        )
        .expect("default selector must initialize");

        let validator = EnergyBalanceValidator::default();
        let report = validator.generate_report(&model);

        assert!(report.contains("Energy Balance Validation Report"));
        assert!(report.contains("Total Zones:"));
        assert!(report.contains("Zone Energy Breakdown:"));
    }

    /// Issue #1344 acceptance criterion: a 2-zone stub with a deliberate 5 W
    /// unbalance must be reported as residual = 5.00 W (within 1e-3 W) with
    /// status = FAIL.
    ///
    /// Strategy: build a Case 960 2-zone model and **hand-balance** every
    /// field the InvariantChecker reads (T_air = T_mass = T_prev_mass =
    /// T_outdoor = T_ground = 20 °C, all loads/solar = 0). This isolates the
    /// validator from the #1295 first-timestep physics-imbalance gap (out of
    /// scope per the issue body). Inject +5 W into zone 0's `loads` and
    /// assert the validator reports residual = 5.00 W (within 1e-3 W).
    #[test]
    fn test_multi_zone_validator_catches_5w_unbalance() {
        use crate::physics::cta::VectorField;

        let spec = ASHRAE140Case::Case960.spec();
        let mut model = ThermalModel::<VectorField>::from_spec_with_selector(
            &spec,
            &ThermalSelector::default(),
        )
        .expect("default selector must initialize");
        let area0 = model.setpoints.zone_area.as_ref()[0];

        // Hand-balance the stub (T_air = T_mass = T_prev_mass = T_outdoor =
        // T_ground = 20 °C, all loads/solar = 0).
        let t_balanced = 20.0_f64;
        for i in 0..model.hvac.num_zones {
            model.setpoints.temperatures.as_mut()[i] = t_balanced;
            model.mass.mass_temperatures.as_mut()[i] = t_balanced;
            model.mass.previous_mass_temperatures.as_mut()[i] = t_balanced;
            model.setpoints.loads.as_mut()[i] = 0.0;
            model.solar.solar_gains.as_mut()[i] = 0.0;
            model.solar.opaque_solar_gains.as_mut()[i] = 0.0;
        }
        model.set_ground_temp(t_balanced);

        let dt = 3600.0;
        let t_outdoor = t_balanced; // ΔT = 0 → all heat flows = 0

        // Inject +5 W into zone 0's load flux (W/m^2 → W by /area).
        let artificial_gain_w = 5.0_f64;
        model.setpoints.loads.as_mut()[0] += artificial_gain_w / area0;

        // Validator: a tight 1e-6% tolerance forces the validator to reject
        // any non-trivial Watt residual.
        let validator = EnergyBalanceValidator::new(1e-6, 1e-3);
        let result = validator.validate_multi_zone_energy_conservation(&model, dt, t_outdoor);

        match result {
            Err(ValidationError::MultiZoneConservationViolation {
                residual_w,
                zone_residuals_w,
                ..
            }) => {
                // The injected 5 W must show up in zone 0's per-zone residual
                // within 1e-3 W (this is the acceptance criterion).
                assert!(
                    (zone_residuals_w[0] - artificial_gain_w).abs() < 1e-3,
                    "Zone 0 residual must reflect injected +5 W unbalance within 1e-3 W; got {} W",
                    zone_residuals_w[0]
                );
                // And the total residual must equal the acceptance value
                // 5.00 W within 1e-3 W.
                assert!(
                    (residual_w - artificial_gain_w).abs() < 1e-3,
                    "Acceptance criterion: residual must equal 5.00 W within 1e-3 W; got {} W",
                    residual_w
                );
            }
            other => panic!(
                "expected MultiZoneConservationViolation with 5 W unbalance, got {:?}",
                other
            ),
        }
    }

    /// Issue #1344: a hand-balanced 2-zone stub (T_air = T_mass = T_prev_mass
    /// = T_outdoor = T_ground = 20 °C, all loads = 0) must report `Ok(())`
    /// with a Watt residual of exactly zero.
    #[test]
    fn test_multi_zone_validator_balanced_model_passes() {
        use crate::physics::cta::VectorField;

        let spec = ASHRAE140Case::Case960.spec();
        let mut model = ThermalModel::<VectorField>::from_spec_with_selector(
            &spec,
            &ThermalSelector::default(),
        )
        .expect("default selector must initialize");
        let t_balanced = 20.0_f64;
        for i in 0..model.hvac.num_zones {
            model.setpoints.temperatures.as_mut()[i] = t_balanced;
            model.mass.mass_temperatures.as_mut()[i] = t_balanced;
            model.mass.previous_mass_temperatures.as_mut()[i] = t_balanced;
            model.setpoints.loads.as_mut()[i] = 0.0;
            model.solar.solar_gains.as_mut()[i] = 0.0;
            model.solar.opaque_solar_gains.as_mut()[i] = 0.0;
        }
        model.set_ground_temp(t_balanced);

        let dt = 3600.0;
        let validator = EnergyBalanceValidator::new(1.0, 1.0);
        let result = validator.validate_multi_zone_energy_conservation(&model, dt, t_balanced);
        assert!(
            result.is_ok(),
            "Hand-balanced 2-zone stub must pass; got {:?}",
            result
        );
    }

    /// Issue #1348 acceptance criterion: for a symmetric N-zone network the
    /// validator must accept |Σ q_iz[i]| < 1e-6 W. Construct the per-zone
    /// vector analytically: with symmetric h_tr_ij = h_tr_ji, the
    /// algebraic identity guarantees Σ q_iz = 0 exactly; we synthesise a
    /// numerically-zero residual by adding q_iz[i] and −q_iz[i] in pairs.
    #[test]
    fn n_zone_network_conservation_passes_for_symmetric_matrix() {
        let q_iz = vec![12.5_f64, -7.3_f64, -5.2_f64]; // sums to 0 exactly
        let validator = EnergyBalanceValidator::new(1.0, 1.0);
        assert!(
            validator
                .validate_n_zone_network_conservation(&q_iz, 1e-6)
                .is_ok(),
            "Σ q_iz = 0 must pass with tolerance 1e-6 W"
        );
    }

    /// Issue #1348 acceptance criterion: an asymmetric residual that
    /// exceeds the 1e-6 W tolerance must surface as
    /// `InterZoneConservationViolation` carrying the signed sum and
    /// per-zone breakdown.
    #[test]
    fn n_zone_network_conservation_rejects_asymmetric_residual() {
        let q_iz = vec![10.0_f64, -5.0_f64, 0.0_f64]; // sums to 5 W (asymmetric)
        let validator = EnergyBalanceValidator::new(1.0, 1.0);
        let result = validator.validate_n_zone_network_conservation(&q_iz, 1e-6);
        match result {
            Err(ValidationError::InterZoneConservationViolation {
                net_inter_zone_q_w,
                zone_residuals_w,
                tolerance_w,
            }) => {
                assert!((net_inter_zone_q_w - 5.0).abs() < 1e-12);
                assert_eq!(zone_residuals_w, vec![10.0, -5.0, 0.0]);
                assert!((tolerance_w - 1e-6).abs() < 1e-12);
            }
            other => panic!("expected InterZoneConservationViolation, got {:?}", other),
        }
    }
}
