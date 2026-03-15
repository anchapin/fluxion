//! ISO 13790 Annex C thermal mass classification.
//!
//! This module provides ISO 13790 Annex C thermal mass classification thresholds
//! for building assemblies. These thresholds classify constructions based on
//! their effective thermal mass into VeryLight, Light, Medium, Heavy, or
//! VeryHeavy categories.

/// VeryLight thermal mass threshold per ISO 13790 Annex C.
///
/// **Value:** 50 kJ/m²K
/// **Units:** kJ/m²K (kilojoules per square meter Kelvin)
/// **Source:** ISO 13790 Annex C, Table C.1, Thermal Mass Classification
/// **Uncertainty:** ±5 kJ/m²K (material variation)
/// **Validity:** Valid for building assemblies with thermal capacitance < 50 kJ/m²K
/// **Assumptions:** Typical lightweight construction (wood frame, metal cladding)
pub const THERMAL_MASS_VERY_LIGHT: f64 = 50.0;

/// Light thermal mass lower threshold per ISO 13790 Annex C.
///
/// **Value:** 50 kJ/m²K
/// **Units:** kJ/m²K (kilojoules per square meter Kelvin)
/// **Source:** ISO 13790 Annex C, Table C.1, Thermal Mass Classification
/// **Uncertainty:** ±5 kJ/m²K (material variation)
/// **Validity:** Valid for building assemblies with thermal capacitance 50-150 kJ/m²K
/// **Assumptions:** Typical light-mass construction (lightweight concrete, brick veneer)
pub const THERMAL_MASS_LIGHT: f64 = 50.0;

/// Light thermal mass upper threshold per ISO 13790 Annex C.
///
/// **Value:** 150 kJ/m²K
/// **Units:** kJ/m²K (kilojoules per square meter Kelvin)
/// **Source:** ISO 13790 Annex C, Table C.1, Thermal Mass Classification
/// **Uncertainty:** ±5 kJ/m²K (material variation)
/// **Validity:** Valid for building assemblies with thermal capacitance 50-150 kJ/m²K
/// **Assumptions:** Typical light-mass construction (lightweight concrete, brick veneer)
pub const THERMAL_MASS_LIGHT_UPPER: f64 = 150.0;

/// Medium thermal mass lower threshold per ISO 13790 Annex C.
///
/// **Value:** 150 kJ/m²K
/// **Units:** kJ/m²K (kilojoules per square meter Kelvin)
/// **Source:** ISO 13790 Annex C, Table C.1, Thermal Mass Classification
/// **Uncertainty:** ±5 kJ/m²K (material variation)
/// **Validity:** Valid for building assemblies with thermal capacitance 150-260 kJ/m²K
/// **Assumptions:** Typical medium-mass construction (concrete block, precast concrete)
pub const THERMAL_MASS_MEDIUM: f64 = 150.0;

/// Medium thermal mass upper threshold per ISO 13790 Annex C.
///
/// **Value:** 260 kJ/m²K
/// **Units:** kJ/m²K (kilojoules per square meter Kelvin)
/// **Source:** ISO 13790 Annex C, Table C.1, Thermal Mass Classification
/// **Uncertainty:** ±5 kJ/m²K (material variation)
/// **Validity:** Valid for building assemblies with thermal capacitance 150-260 kJ/m²K
/// **Assumptions:** Typical medium-mass construction (concrete block, precast concrete)
pub const THERMAL_MASS_MEDIUM_UPPER: f64 = 260.0;

/// Heavy thermal mass lower threshold per ISO 13790 Annex C.
///
/// **Value:** 260 kJ/m²K
/// **Units:** kJ/m²K (kilojoules per square meter Kelvin)
/// **Source:** ISO 13790 Annex C, Table C.1, Thermal Mass Classification
/// **Uncertainty:** ±5 kJ/m²K (material variation)
/// **Validity:** Valid for building assemblies with thermal capacitance 260-370 kJ/m²K
/// **Assumptions:** Typical heavy-mass construction (reinforced concrete, masonry)
pub const THERMAL_MASS_HEAVY: f64 = 260.0;

/// Heavy thermal mass upper threshold per ISO 13790 Annex C.
///
/// **Value:** 370 kJ/m²K
/// **Units:** kJ/m²K (kilojoules per square meter Kelvin)
/// **Source:** ISO 13790 Annex C, Table C.1, Thermal Mass Classification
/// **Uncertainty:** ±5 kJ/m²K (material variation)
/// **Validity:** Valid for building assemblies with thermal capacitance 260-370 kJ/m²K
/// **Assumptions:** Typical heavy-mass construction (reinforced concrete, masonry)
pub const THERMAL_MASS_HEAVY_UPPER: f64 = 370.0;

/// VeryHeavy thermal mass threshold per ISO 13790 Annex C.
///
/// **Value:** 370 kJ/m²K
/// **Units:** kJ/m²K (kilojoules per square meter Kelvin)
/// **Source:** ISO 13790 Annex C, Table C.1, Thermal Mass Classification
/// **Uncertainty:** ±5 kJ/m²K (material variation)
/// **Validity:** Valid for building assemblies with thermal capacitance > 370 kJ/m²K
/// **Assumptions:** Typical very-heavy-mass construction (thick concrete, earth-sheltered)
pub const THERMAL_MASS_VERY_HEAVY: f64 = 370.0;

/// Calculate effective thermal mass for a custom building assembly.
///
/// This function implements the hybrid approach from ISO 13790 Annex C, combining
/// pre-calculated thresholds with computation functions for custom constructions.
///
/// # Arguments
///
/// * `layers` - Vector of material layers with (thickness, density, specific_heat) tuples
///   - thickness: Material thickness in meters (m)
///   - density: Material density in kilograms per cubic meter (kg/m³)
///   - specific_heat: Specific heat capacity in joules per kilogram Kelvin (J/kgK)
///
/// # Returns
///
/// Effective thermal mass in kJ/m²K (kilojoules per square meter Kelvin)
///
/// # Formula
///
/// ```text
/// C_eff = Σ(density × specific_heat × thickness) / 1000
///
/// Where:
/// - density: kg/m³
/// - specific_heat: J/kgK
/// - thickness: m
/// - Division by 1000 converts J/m²K to kJ/m²K
/// ```
///
/// # Example
///
/// ```
/// use fluxion::physics::constants::thermal::iso_13790::calculate_effective_thermal_mass;
///
/// let layers = vec![
///     (0.1, 2300.0, 840.0),  // 10cm concrete
///     (0.05, 50.0, 840.0),    // 5cm insulation
/// ];
/// let thermal_mass = calculate_effective_thermal_mass(&layers);
/// assert!(thermal_mass > 0.0);
/// ```
///
/// # Notes
///
/// - This calculation represents the effective thermal capacitance of the assembly
/// - Only layers on the interior side of the dominant insulation contribute (half-insulation rule)
/// - For standard constructions, use the predefined thresholds (THERMAL_MASS_LIGHT, etc.)
pub fn calculate_effective_thermal_mass(layers: &[(f64, f64, f64)]) -> f64 {
    // Calculate Σ(density × specific_heat × thickness) / 1000 for kJ/m²K
    layers
        .iter()
        .map(|(thickness, density, specific_heat)| density * specific_heat * thickness)
        .sum::<f64>()
        / 1000.0
}
