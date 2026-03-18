//! Thermal Method Selector - Automatic solver selection based on building thermal mass.
//!
//! This module implements automatic method selection logic that chooses between
//! 5R1C, CTF, and FD solvers based on building thermal mass characteristics.
//!
//! # Overview
//!
//! The selection strategy is based on the thermal mass time constant (τ):
//! - τ < threshold (default 2 hours) → 5R1C (fast, low-mass)
//! - τ ≥ threshold → CTF (accurate, high-mass)
//! - CTF fails → FD (robust fallback for extreme constructions)
//!
//! # Example
//!
//! ```rust
//! use fluxion::physics::method_selector::{ThermalMethodSelector, ThermalMethod};
//!
//! let selector = ThermalMethodSelector::default();
//! let method = selector.select_method(&wall_assembly);
//!
//! match method {
//!     ThermalMethod::FiveR1C => println!("Using 5R1C for low-mass wall"),
//!     ThermalMethod::CTF => println!("Using CTF for high-mass wall"),
//!     ThermalMethod::FiniteDifference => println!("Using FD as fallback"),
//! }
//! ```

use crate::sim::assembly::BuildingAssembly;
use log::{info, warn};

/// Available thermal solution methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalMethod {
    /// 5R1C thermal network (fast, low-mass buildings)
    FiveR1C,
    /// Conduction Transfer Functions (accurate, high-mass buildings)
    CTF,
    /// Finite Difference (robust, fallback for extreme constructions)
    FiniteDifference,
}

impl ThermalMethod {
    /// Get human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            ThermalMethod::FiveR1C => "5R1C",
            ThermalMethod::CTF => "CTF",
            ThermalMethod::FiniteDifference => "FD",
        }
    }
}

/// Automatic method selection based on building thermal mass.
///
/// # Selection Strategy
///
/// The selector calculates the thermal mass time constant (τ) for each wall:
///
/// ```text
/// τ = (Σ ρ_i · c_p,i · L_i) / (h_interior + h_exterior)
/// ```
///
/// Where:
/// - ρ_i = density of layer i [kg/m³]
/// - c_p,i = specific heat of layer i [J/kg·K]
/// - L_i = thickness of layer i [m]
/// - h = convective heat transfer coefficient [W/m²·K]
///
/// Selection rules:
/// - τ < threshold → 5R1C (low mass, fast)
/// - τ ≥ threshold → CTF (high mass, accurate)
/// - CTF coefficients invalid → FD (fallback)
///
/// # Fields
///
/// * `threshold_hours` - Time constant threshold for method selection (default: 2.0 hours)
/// * `override_method` - Manual override (None = auto, Some = force method)
/// * `enable_fallback` - Enable CTF → FD fallback (default: true)
/// * `h_interior` - Interior convective coefficient [W/m²·K] (default: 8.0)
/// * `h_exterior` - Exterior convective coefficient [W/m²·K] (default: 25.0)
#[derive(Debug, Clone)]
pub struct ThermalMethodSelector {
    /// Selection threshold: τ > threshold → CTF/FD (default: 2.0 hours)
    pub threshold_hours: f64,
    /// Manual override (None = auto, Some = force method)
    pub override_method: Option<ThermalMethod>,
    /// Enable fallback (CTF → FD on failure)
    pub enable_fallback: bool,
    /// Interior convective coefficient [W/m²·K]
    pub h_interior: f64,
    /// Exterior convective coefficient [W/m²·K]
    pub h_exterior: f64,
}

impl ThermalMethodSelector {
    /// Create a new method selector with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create selector with custom threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold_hours` - Time constant threshold (typical: 1.5-3.0 hours)
    pub fn with_threshold(threshold_hours: f64) -> Self {
        Self {
            threshold_hours,
            ..Self::default()
        }
    }

    /// Create selector with manual override.
    pub fn with_override(override_method: ThermalMethod) -> Self {
        Self {
            override_method: Some(override_method),
            ..Self::default()
        }
    }

    /// Calculate thermal mass time constant for a wall assembly.
    ///
    /// # Formula
    ///
    /// ```text
    /// τ = (Σ ρ_i · c_p,i · L_i) / (h_interior + h_exterior)
    /// ```
    ///
    /// Where:
    /// - ρ_i · c_p,i · L_i = thermal capacity per unit area [J/m²·K]
    /// - h_total = h_interior + h_exterior [W/m²·K]
    ///
    /// # Arguments
    ///
    /// * `wall` - Wall assembly with material layers
    ///
    /// # Returns
    ///
    /// Time constant τ in hours
    ///
    /// # Example
    ///
    /// ```rust
    /// # use fluxion::physics::method_selector::ThermalMethodSelector;
    /// # use fluxion::sim::assembly::{AssemblyBuilder, ConcreteMaterial};
    /// let selector = ThermalMethodSelector::default();
    /// let wall = AssemblyBuilder::new("Wall".to_string())
    ///     .add_layer(Box::new(ConcreteMaterial::new(0.2)))
    ///     .build()
    ///     .unwrap();
    ///
    /// let tau = selector.calculate_time_constant(&wall);
    /// println!("Time constant: {:.2} hours", tau);
    /// ```
    pub fn calculate_time_constant(&self, wall: &BuildingAssembly) -> f64 {
        let mut thermal_mass = 0.0; // J/m²·K

        for layer in &wall.layers {
            // Mass per unit area [kg/m²]
            let mass_per_area = layer.density() * layer.thickness();
            // Heat capacity per unit area [J/m²·K]
            let heat_cap_per_area = mass_per_area * layer.specific_heat();
            thermal_mass += heat_cap_per_area;
        }

        let h_total = self.h_interior + self.h_exterior; // W/m²·K

        // Avoid division by zero
        if h_total <= 0.0 {
            warn!("Zero total heat transfer coefficient, using default h_total = 33 W/m²·K");
            return thermal_mass / 33.0 / 3600.0;
        }

        let tau_seconds = thermal_mass / h_total; // seconds
        tau_seconds / 3600.0 // Convert to hours
    }

    /// Select appropriate thermal method for a wall assembly.
    ///
    /// # Arguments
    ///
    /// * `wall` - Wall assembly to analyze
    ///
    /// # Returns
    ///
    /// Selected thermal method (5R1C, CTF, or FD)
    ///
    /// # Selection Logic
    ///
    /// 1. Check for manual override → use override method
    /// 2. Calculate time constant τ
    /// 3. τ < threshold → 5R1C (low mass, fast)
    /// 4. τ ≥ threshold → CTF (high mass, accurate)
    pub fn select_method(&self, wall: &BuildingAssembly) -> ThermalMethod {
        // Check for manual override
        if let Some(method) = self.override_method {
            return method;
        }

        // Calculate time constant
        let tau = self.calculate_time_constant(wall);

        // Select method based on thermal mass
        if tau < self.threshold_hours {
            ThermalMethod::FiveR1C // Low mass: use fast 5R1C
        } else {
            ThermalMethod::CTF // High mass: use accurate CTF
        }
    }

    /// Select method with CTF → FD fallback.
    ///
    /// # Arguments
    ///
    /// * `wall` - Wall assembly to analyze
    /// * `ctf_valid` - Whether CTF coefficients are valid
    ///
    /// # Returns
    ///
    /// Selected thermal method (may be FD if CTF invalid)
    pub fn select_with_fallback(&self, wall: &BuildingAssembly, ctf_valid: bool) -> ThermalMethod {
        let method = self.select_method(wall);

        if method == ThermalMethod::CTF && !ctf_valid {
            if self.enable_fallback {
                warn!(
                    "CTF coefficients invalid for wall '{}', falling back to FD",
                    wall.name
                );
                ThermalMethod::FiniteDifference
            } else {
                warn!(
                    "CTF coefficients invalid for wall '{}' and fallback disabled",
                    wall.name
                );
                ThermalMethod::CTF // Return CTF anyway, let caller handle error
            }
        } else {
            method
        }
    }

    /// Validate CTF coefficients.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - CTF coefficients to validate
    ///
    /// # Returns
    ///
    /// True if all coefficients are finite (not NaN or Inf)
    pub fn validate_ctf_coefficients(
        coeffs: &crate::physics::ctf_coefficients::CTFCoefficients,
    ) -> bool {
        coeffs.x.iter().all(|&x| x.is_finite())
            && coeffs.y.iter().all(|&y| y.is_finite())
            && coeffs.z.iter().all(|&z| z.is_finite())
            && coeffs.phi.iter().all(|&p| p.is_finite())
    }

    /// Log method selection for debugging.
    ///
    /// # Arguments
    ///
    /// * `wall` - Wall assembly
    /// * `method` - Selected method
    pub fn log_selection(&self, wall: &BuildingAssembly, method: ThermalMethod) {
        let tau = self.calculate_time_constant(wall);
        info!(
            "Wall '{}': τ = {:.2} h → method = {} (threshold = {:.1} h)",
            wall.name,
            tau,
            method.name(),
            self.threshold_hours
        );
    }

    /// Generate selection report for multiple walls.
    ///
    /// # Arguments
    ///
    /// * `walls` - Slice of wall assemblies
    ///
    /// # Returns
    ///
    /// Summary report string
    pub fn generate_report(&self, walls: &[BuildingAssembly]) -> String {
        let mut report = String::new();
        report.push_str("=== Method Selection Report ===\n");

        let mut counts = [0, 0, 0]; // [5R1C, CTF, FD]
        let mut tau_sum = 0.0;

        for wall in walls {
            let method = self.select_method(wall);
            let tau = self.calculate_time_constant(wall);
            tau_sum += tau;

            match method {
                ThermalMethod::FiveR1C => counts[0] += 1,
                ThermalMethod::CTF => counts[1] += 1,
                ThermalMethod::FiniteDifference => counts[2] += 1,
            }
        }

        let tau_avg = if walls.is_empty() {
            0.0
        } else {
            tau_sum / walls.len() as f64
        };

        report.push_str(&format!("Total walls: {}\n", walls.len()));
        report.push_str(&format!(
            "5R1C: {} walls ({:.1}%)\n",
            counts[0],
            counts[0] as f64 / walls.len().max(1) as f64 * 100.0
        ));
        report.push_str(&format!(
            "CTF:  {} walls ({:.1}%)\n",
            counts[1],
            counts[1] as f64 / walls.len().max(1) as f64 * 100.0
        ));
        report.push_str(&format!(
            "FD:   {} walls ({:.1}%)\n",
            counts[2],
            counts[2] as f64 / walls.len().max(1) as f64 * 100.0
        ));
        report.push_str(&format!("Average τ: {:.2} hours\n", tau_avg));

        report
    }
}

impl Default for ThermalMethodSelector {
    fn default() -> Self {
        Self {
            threshold_hours: 2.0, // ISO 13790 guidance
            override_method: None,
            enable_fallback: true,
            h_interior: 8.0,  // Typical interior film coefficient
            h_exterior: 25.0, // Typical exterior film coefficient
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::assembly::{AssemblyBuilder, ConcreteMaterial, InsulationMaterial};

    fn create_lightweight_wall() -> BuildingAssembly {
        AssemblyBuilder::new("Lightweight Wall".to_string())
            .add_layer(Box::new(InsulationMaterial::new(0.05))) // 50mm insulation
            .build()
            .unwrap()
    }

    fn create_heavyweight_wall() -> BuildingAssembly {
        AssemblyBuilder::new("Heavyweight Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2))) // 200mm concrete
            .build()
            .unwrap()
    }

    #[test]
    fn test_selector_creation() {
        let selector = ThermalMethodSelector::new();
        assert_eq!(selector.threshold_hours, 2.0);
        assert!(selector.override_method.is_none());
        assert!(selector.enable_fallback);
    }

    #[test]
    fn test_selector_with_threshold() {
        let selector = ThermalMethodSelector::with_threshold(3.0);
        assert_eq!(selector.threshold_hours, 3.0);
    }

    #[test]
    fn test_time_constant_lightweight() {
        let selector = ThermalMethodSelector::default();
        let wall = create_lightweight_wall();

        let tau = selector.calculate_time_constant(&wall);

        // Lightweight wall should have low time constant (< 2 hours)
        assert!(tau < 2.0, "Lightweight wall τ = {:.2}h, expected < 2h", tau);
    }

    #[test]
    fn test_time_constant_heavyweight() {
        let selector = ThermalMethodSelector::default();
        let wall = create_heavyweight_wall();

        let tau = selector.calculate_time_constant(&wall);

        // Heavyweight wall should have high time constant (> 2 hours)
        assert!(tau > 2.0, "Heavyweight wall τ = {:.2}h, expected > 2h", tau);
    }

    #[test]
    fn test_selection_auto_lightweight() {
        let selector = ThermalMethodSelector::default();
        let wall = create_lightweight_wall();

        let method = selector.select_method(&wall);

        assert_eq!(method, ThermalMethod::FiveR1C);
    }

    #[test]
    fn test_selection_auto_heavyweight() {
        let selector = ThermalMethodSelector::default();
        let wall = create_heavyweight_wall();

        let method = selector.select_method(&wall);

        assert_eq!(method, ThermalMethod::CTF);
    }

    #[test]
    fn test_selection_override() {
        let selector = ThermalMethodSelector::with_override(ThermalMethod::FiniteDifference);
        let wall = create_lightweight_wall();

        let method = selector.select_method(&wall);

        // Override should force FD regardless of thermal mass
        assert_eq!(method, ThermalMethod::FiniteDifference);
    }

    #[test]
    fn test_fallback_invalid_ctf() {
        let selector = ThermalMethodSelector::default();
        let wall = create_heavyweight_wall();

        // CTF invalid → should fall back to FD
        let method = selector.select_with_fallback(&wall, false);
        assert_eq!(method, ThermalMethod::FiniteDifference);

        // CTF valid → should use CTF
        let method = selector.select_with_fallback(&wall, true);
        assert_eq!(method, ThermalMethod::CTF);
    }

    #[test]
    fn test_fallback_disabled() {
        let selector = ThermalMethodSelector {
            enable_fallback: false,
            ..ThermalMethodSelector::default()
        };
        let wall = create_heavyweight_wall();

        // CTF invalid but fallback disabled → should still return CTF
        let method = selector.select_with_fallback(&wall, false);
        assert_eq!(method, ThermalMethod::CTF);
    }

    #[test]
    fn test_validate_ctf_coefficients() {
        use crate::physics::ctf_coefficients::CTFCoefficients;

        // Valid coefficients
        let valid_coeffs = CTFCoefficients {
            x: vec![1.0, 0.5, 0.25],
            y: vec![0.8, 0.4, 0.2],
            z: vec![1.2, 0.6, 0.3],
            phi: vec![0.1, 0.05, 0.025],
            timestep: 3600.0,
            num_coeffs: 3,
        };
        assert!(ThermalMethodSelector::validate_ctf_coefficients(
            &valid_coeffs
        ));

        // Invalid coefficients (NaN)
        let invalid_coeffs = CTFCoefficients {
            x: vec![1.0, f64::NAN, 0.25],
            y: vec![0.8, 0.4, 0.2],
            z: vec![1.2, 0.6, 0.3],
            phi: vec![0.1, 0.05, 0.025],
            timestep: 3600.0,
            num_coeffs: 3,
        };
        assert!(!ThermalMethodSelector::validate_ctf_coefficients(
            &invalid_coeffs
        ));
    }

    #[test]
    fn test_generate_report() {
        let selector = ThermalMethodSelector::default();
        let walls = vec![
            create_lightweight_wall(),
            create_heavyweight_wall(),
            create_heavyweight_wall(),
        ];

        let report = selector.generate_report(&walls);

        assert!(report.contains("Total walls: 3"));
        assert!(report.contains("5R1C: 1 walls"));
        assert!(report.contains("CTF:  2 walls"));
    }
}
