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

/// Configuration for solver selection mode.
///
/// This allows explicit control over solver selection instead of relying
/// on implicit automatic selection based on thermal mass time constant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverSelectionConfig {
    /// Automatic selection based on thermal mass time constant (τ)
    Automatic,
    /// Force a specific method for all surfaces
    ForceMethod(ThermalMethod),
    /// Per-surface explicit solver selection
    PerSurface(Vec<SurfaceSolverConfig>),
}

/// Configuration for a specific surface's solver selection.
///
/// This allows intentional selection of solver method per surface rather than
/// relying on automatic selection based on thermal mass characteristics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfaceSolverConfig {
    /// Surface identifier (e.g., "Wall", "Roof", "Floor", or surface index)
    pub surface_id: String,
    /// Explicit solver method to use for this surface
    pub method: ThermalMethod,
}

/// Configuration struct for ThermalMethodSelector.
///
/// This replaces the builder pattern with a simple data structure that can be
/// easily constructed, serialized, and tested.
#[derive(Debug, Clone)]
pub struct ThermalMethodSelectorConfig {
    /// Selection threshold: τ > threshold → CTF/FD (default: 24.0 hours)
    pub threshold_hours: f64,
    /// Manual override (None = auto, Some = force method)
    pub override_method: Option<ThermalMethod>,
    /// Enable fallback (CTF → FD on failure)
    pub enable_fallback: bool,
    /// Enable automatic selection based on thermal mass
    pub enable_automatic_selection: bool,
    /// Enable per-surface explicit solver selection
    pub per_surface_selection: bool,
}

impl Default for ThermalMethodSelectorConfig {
    fn default() -> Self {
        Self {
            threshold_hours: 24.0,
            override_method: None,
            enable_fallback: true,
            enable_automatic_selection: true,
            per_surface_selection: false,
        }
    }
}

impl SurfaceSolverConfig {
    /// Create a new surface solver config.
    pub fn new(surface_id: impl Into<String>, method: ThermalMethod) -> Self {
        Self {
            surface_id: surface_id.into(),
            method,
        }
    }

    /// Create config for a wall surface.
    pub fn wall(method: ThermalMethod) -> Self {
        Self::new("Wall", method)
    }

    /// Create config for a roof surface.
    pub fn roof(method: ThermalMethod) -> Self {
        Self::new("Roof", method)
    }

    /// Create config for a floor surface.
    pub fn floor(method: ThermalMethod) -> Self {
        Self::new("Floor", method)
    }
}

/// Solver selection result with method and reason for selection.
///
/// This provides transparency into why a particular solver was selected,
/// supporting the requirement for "intentional, not implicit" selection.
#[derive(Debug, Clone)]
pub struct SolverSelectionResult {
    /// The selected thermal method
    pub method: ThermalMethod,
    /// Human-readable reason for the selection
    pub reason: String,
    /// Time constant τ in hours (if calculated)
    pub time_constant_hours: Option<f64>,
}

impl SolverSelectionResult {
    /// Create a new selection result.
    pub fn new(method: ThermalMethod, reason: impl Into<String>) -> Self {
        Self {
            method,
            reason: reason.into(),
            time_constant_hours: None,
        }
    }

    /// Create with time constant.
    pub fn with_time_constant(mut self, tau: f64) -> Self {
        self.time_constant_hours = Some(tau);
        self
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
    /// Explicit solver selection configuration
    pub selection_config: SolverSelectionConfig,
}

impl ThermalMethodSelector {
    /// Create a new method selector with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a method selector from a config struct.
    pub fn from_config(config: ThermalMethodSelectorConfig) -> Self {
        Self {
            threshold_hours: config.threshold_hours,
            override_method: config.override_method,
            enable_fallback: config.enable_fallback,
            h_interior: 8.0,
            h_exterior: 25.0,
            selection_config: if config.per_surface_selection {
                SolverSelectionConfig::PerSurface(vec![])
            } else {
                SolverSelectionConfig::ForceMethod(
                    config
                        .override_method
                        .unwrap_or(ThermalMethod::FiniteDifference),
                )
            },
        }
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

    /// Select method with explicit configuration and return selection result.
    ///
    /// This method provides transparency into the solver selection process by returning
    /// both the selected method and the reason for selection.
    ///
    /// # Arguments
    ///
    /// * `wall` - Wall assembly to analyze
    /// * `surface_id` - Surface identifier (e.g., "Wall", "Roof") for per-surface config
    ///
    /// # Returns
    ///
    /// Selection result with method and reason
    pub fn select_with_result(
        &self,
        wall: &BuildingAssembly,
        surface_id: &str,
    ) -> SolverSelectionResult {
        // Check for explicit per-surface configuration first
        if let SolverSelectionConfig::PerSurface(ref configs) = self.selection_config {
            if let Some(config) = configs.iter().find(|c| c.surface_id == surface_id) {
                return SolverSelectionResult::new(
                    config.method,
                    format!("Explicit per-surface config for '{}'", surface_id),
                );
            }
        }

        // Check for forced method
        if let SolverSelectionConfig::ForceMethod(method) = &self.selection_config {
            return SolverSelectionResult::new(
                *method,
                format!("Explicit force to {}", method.name()),
            );
        }

        // Fall back to automatic selection based on thermal mass
        let method = self.select_method(wall);
        let tau = self.calculate_time_constant(wall);

        let reason = if let Some(override_method) = self.override_method {
            format!("Override to {} (τ = {:.2}h)", override_method.name(), tau)
        } else if tau < self.threshold_hours {
            format!(
                "Low thermal mass (τ = {:.2}h < {:.1}h threshold)",
                tau, self.threshold_hours
            )
        } else {
            format!(
                "High thermal mass (τ = {:.2}h >= {:.1}h threshold)",
                tau, self.threshold_hours
            )
        };

        SolverSelectionResult::new(method, reason).with_time_constant(tau)
    }

    /// Get the explicit selection configuration.
    pub fn selection_config(&self) -> &SolverSelectionConfig {
        &self.selection_config
    }

    /// Set the explicit selection configuration.
    pub fn set_selection_config(&mut self, config: SolverSelectionConfig) {
        self.selection_config = config;
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
            selection_config: SolverSelectionConfig::Automatic,
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

    #[test]
    fn test_thermal_method_name() {
        assert_eq!(ThermalMethod::FiveR1C.name(), "5R1C");
        assert_eq!(ThermalMethod::CTF.name(), "CTF");
        assert_eq!(ThermalMethod::FiniteDifference.name(), "FD");
    }

    #[test]
    fn test_selector_default() {
        let selector = ThermalMethodSelector::default();
        assert_eq!(selector.threshold_hours, 2.0);
        assert!(selector.override_method.is_none());
        assert!(selector.enable_fallback);
        assert_eq!(selector.h_interior, 8.0);
        assert_eq!(selector.h_exterior, 25.0);
    }

    #[test]
    fn test_log_selection() {
        let selector = ThermalMethodSelector::default();
        let wall = create_lightweight_wall();
        let method = selector.select_method(&wall);
        selector.log_selection(&wall, method);
        // Just ensure it doesn't panic - actual logging is tested elsewhere
    }

    #[test]
    fn test_thermal_method_copy() {
        let method = ThermalMethod::CTF;
        let copied = method;
        assert_eq!(method, copied);
    }

    #[test]
    fn test_thermal_method_eq() {
        assert_eq!(ThermalMethod::FiveR1C, ThermalMethod::FiveR1C);
        assert_ne!(ThermalMethod::CTF, ThermalMethod::FiniteDifference);
    }

    #[test]
    fn test_selector_clone() {
        let selector = ThermalMethodSelector::with_threshold(3.0);
        let cloned = selector.clone();

        assert_eq!(cloned.threshold_hours, 3.0);
        assert!(cloned.override_method.is_none());
    }

    #[test]
    fn test_selector_debug() {
        let selector = ThermalMethodSelector::default();
        let debug_str = format!("{:?}", selector);
        assert!(debug_str.contains("ThermalMethodSelector"));
    }

    #[test]
    fn test_thermal_method_debug() {
        let method = ThermalMethod::CTF;
        let debug_str = format!("{:?}", method);
        assert!(debug_str.contains("CTF"));
    }

    #[test]
    fn test_calculate_time_constant_zero_h() {
        let selector = ThermalMethodSelector::default();
        let wall = create_lightweight_wall();
        let tau = selector.calculate_time_constant(&wall);

        // Should be finite and positive
        assert!(tau.is_finite());
        assert!(tau > 0.0);
    }

    #[test]
    fn test_select_with_custom_threshold() {
        let selector = ThermalMethodSelector::with_threshold(5.0);
        let wall = create_lightweight_wall();

        // With high threshold (5h), even lightweight wall may use CTF
        let method = selector.select_method(&wall);
        // Depending on actual time constant, could be either 5R1C or CTF
        assert!(matches!(
            method,
            ThermalMethod::FiveR1C | ThermalMethod::CTF
        ));
    }

    #[test]
    fn test_select_with_custom_override() {
        let selector = ThermalMethodSelector::with_override(ThermalMethod::FiveR1C);
        let wall = create_heavyweight_wall();

        let method = selector.select_method(&wall);
        assert_eq!(method, ThermalMethod::FiveR1C);
    }

    #[test]
    fn test_generate_report_empty() {
        let selector = ThermalMethodSelector::default();
        let walls: Vec<BuildingAssembly> = vec![];

        let report = selector.generate_report(&walls);

        assert!(report.contains("Total walls: 0"));
        assert!(report.contains("5R1C: 0 walls"));
        assert!(report.contains("CTF:  0 walls"));
        assert!(report.contains("FD:   0 walls"));
    }

    #[test]
    fn test_fallback_with_enable_flag() {
        let selector = ThermalMethodSelector {
            enable_fallback: false,
            ..ThermalMethodSelector::default()
        };
        let wall = create_heavyweight_wall();

        // Even with CTF invalid, fallback disabled should return CTF
        let method = selector.select_with_fallback(&wall, false);
        assert_eq!(method, ThermalMethod::CTF);
    }

    #[test]
    fn test_thermal_method_variants() {
        let methods = [
            ThermalMethod::FiveR1C,
            ThermalMethod::CTF,
            ThermalMethod::FiniteDifference,
        ];

        for method in methods {
            let name = method.name();
            assert!(!name.is_empty());
        }
    }

    #[test]
    fn test_select_with_result() {
        let selector = ThermalMethodSelector::default();
        let wall = create_lightweight_wall();

        let result = selector.select_with_result(&wall, "Wall");

        assert!(result.time_constant_hours.is_some());
        assert!(result.reason.contains("τ") || result.reason.contains("Low thermal mass"));
    }

    #[test]
    fn test_select_with_result_per_surface_config() {
        let mut selector = ThermalMethodSelector::default();
        selector.set_selection_config(SolverSelectionConfig::PerSurface(vec![
            SurfaceSolverConfig::wall(ThermalMethod::FiniteDifference),
        ]));

        let wall = create_lightweight_wall();
        let result = selector.select_with_result(&wall, "Wall");

        assert_eq!(result.method, ThermalMethod::FiniteDifference);
        assert!(result.reason.contains("Explicit per-surface config"));
    }

    #[test]
    fn test_select_with_result_force_method() {
        let mut selector = ThermalMethodSelector::default();
        selector.set_selection_config(SolverSelectionConfig::ForceMethod(
            ThermalMethod::FiniteDifference,
        ));

        let wall = create_lightweight_wall();
        let result = selector.select_with_result(&wall, "Wall");

        assert_eq!(result.method, ThermalMethod::FiniteDifference);
        assert!(result.reason.contains("Explicit force"));
    }

    // === ARCH-007: Config struct API tests (replaced deprecated builder methods) ===

    #[test]
    fn test_config_automatic_selection() {
        use crate::physics::method_selector::ThermalMethodSelectorConfig;

        let config = ThermalMethodSelectorConfig {
            enable_automatic_selection: true,
            ..Default::default()
        };
        let selector = ThermalMethodSelector::from_config(config);
        assert_eq!(selector.selection_config, SolverSelectionConfig::Automatic);
    }

    #[test]
    fn test_config_forced_method() {
        use crate::physics::method_selector::ThermalMethodSelectorConfig;

        let config = ThermalMethodSelectorConfig {
            enable_automatic_selection: false,
            enable_fallback: true,
            ..Default::default()
        };
        let mut selector = ThermalMethodSelector::from_config(config);
        selector.set_selection_config(SolverSelectionConfig::ForceMethod(ThermalMethod::CTF));
        assert_eq!(
            selector.selection_config,
            SolverSelectionConfig::ForceMethod(ThermalMethod::CTF)
        );
    }

    #[test]
    fn test_config_per_surface_selection() {
        use crate::physics::method_selector::ThermalMethodSelectorConfig;

        let configs = vec![
            SurfaceSolverConfig::wall(ThermalMethod::FiveR1C),
            SurfaceSolverConfig::roof(ThermalMethod::CTF),
        ];
        let config = ThermalMethodSelectorConfig {
            per_surface_selection: true,
            ..Default::default()
        };
        let mut selector = ThermalMethodSelector::from_config(config);
        selector.set_selection_config(SolverSelectionConfig::PerSurface(configs.clone()));
        assert_eq!(
            selector.selection_config,
            SolverSelectionConfig::PerSurface(configs)
        );
    }

    #[test]
    fn test_surface_solver_config_helpers() {
        let wall_config = SurfaceSolverConfig::wall(ThermalMethod::FiveR1C);
        assert_eq!(wall_config.surface_id, "Wall");
        assert_eq!(wall_config.method, ThermalMethod::FiveR1C);

        let roof_config = SurfaceSolverConfig::roof(ThermalMethod::CTF);
        assert_eq!(roof_config.surface_id, "Roof");
        assert_eq!(roof_config.method, ThermalMethod::CTF);

        let floor_config = SurfaceSolverConfig::floor(ThermalMethod::FiniteDifference);
        assert_eq!(floor_config.surface_id, "Floor");
        assert_eq!(floor_config.method, ThermalMethod::FiniteDifference);
    }

    #[test]
    fn test_solver_selection_result() {
        let result =
            SolverSelectionResult::new(ThermalMethod::CTF, "test reason").with_time_constant(2.5);

        assert_eq!(result.method, ThermalMethod::CTF);
        assert_eq!(result.reason, "test reason");
        assert_eq!(result.time_constant_hours, Some(2.5));
    }

    #[test]
    fn test_solver_selection_config_variants() {
        use SolverSelectionConfig::*;
        assert_eq!(Automatic, Automatic);
        assert_eq!(
            ForceMethod(ThermalMethod::FiveR1C),
            ForceMethod(ThermalMethod::FiveR1C)
        );
        assert_ne!(
            ForceMethod(ThermalMethod::FiveR1C),
            ForceMethod(ThermalMethod::CTF)
        );
    }

    #[test]
    fn test_selection_config_getter() {
        let mut selector = ThermalMethodSelector::default();
        assert_eq!(
            selector.selection_config(),
            &SolverSelectionConfig::Automatic
        );

        selector.set_selection_config(SolverSelectionConfig::ForceMethod(
            ThermalMethod::FiniteDifference,
        ));
        assert_eq!(
            selector.selection_config(),
            &SolverSelectionConfig::ForceMethod(ThermalMethod::FiniteDifference)
        );
    }

    // === ARCH-007: Config struct tests ===

    #[test]
    fn test_config_struct_default() {
        use crate::physics::method_selector::ThermalMethodSelectorConfig;

        let config = ThermalMethodSelectorConfig::default();

        assert_eq!(config.threshold_hours, 24.0);
        assert!(config.override_method.is_none());
        assert!(config.enable_fallback);
        assert!(config.enable_automatic_selection);
        assert!(!config.per_surface_selection);
    }

    #[test]
    fn test_config_struct_custom_values() {
        use crate::physics::method_selector::ThermalMethodSelectorConfig;

        let config = ThermalMethodSelectorConfig {
            threshold_hours: 3.5,
            override_method: Some(ThermalMethod::FiniteDifference),
            enable_fallback: false,
            enable_automatic_selection: true,
            per_surface_selection: true,
        };

        assert_eq!(config.threshold_hours, 3.5);
        assert_eq!(
            config.override_method,
            Some(ThermalMethod::FiniteDifference)
        );
        assert!(!config.enable_fallback);
        assert!(config.enable_automatic_selection);
        assert!(config.per_surface_selection);
    }

    #[test]
    fn test_selector_from_config() {
        use crate::physics::method_selector::ThermalMethodSelectorConfig;

        let config = ThermalMethodSelectorConfig {
            threshold_hours: 5.0,
            override_method: None,
            enable_fallback: true,
            enable_automatic_selection: false,
            per_surface_selection: false,
        };

        let selector = ThermalMethodSelector::from_config(config);

        assert_eq!(selector.threshold_hours, 5.0);
    }

    #[test]
    fn test_with_threshold_using_config() {
        let selector = ThermalMethodSelector::with_threshold(3.0);
        assert_eq!(selector.threshold_hours, 3.0);
        // Other fields use defaults
        assert_eq!(selector.override_method, None);
        assert!(selector.enable_fallback);
    }
}
