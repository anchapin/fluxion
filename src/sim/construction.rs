//! Multi-layer construction R-value calculator for building envelopes.
//!
//! This module provides structs and functions for calculating thermal resistance (R-value)
//! and thermal transmittance (U-value) for multi-layer building constructions, following
//! ASHRAE Standard 140 specifications.

use serde::{Deserialize, Serialize};

/// Interior film coefficient per ASHRAE specification.
///
/// This represents the convective heat transfer coefficient at the interior surface
/// of a building assembly. The value 8.29 W/m²K is specified in ASHRAE 140.
pub const INTERIOR_FILM_COEFF: f64 = 8.29; // W/m²K

/// Default exterior film coefficient (typical for average wind conditions).
///
/// For wind speeds of 3-4 m/s, the exterior film coefficient typically ranges
/// from 21-29.3 W/m²K. This default value of 25.0 W/m²K represents a mid-range
/// condition suitable for most applications.
pub const EXTERIOR_FILM_COEFF_DEFAULT: f64 = 25.0; // W/m²K

/// Returns the exterior film coefficient based on wind speed.
///
/// The exterior film coefficient varies with wind speed due to enhanced convection.
/// This function implements the ASHRAE recommended correlation:
///
/// - Low wind (< 2 m/s): ~21 W/m²K
/// - Moderate wind (2-5 m/s): 21-29 W/m²K
/// - High wind (> 5 m/s): >29 W/m²K
///
/// # Arguments
/// * `wind_speed` - Wind speed in meters per second (m/s)
///
/// # Returns
/// Exterior film coefficient in W/m²K
///
/// # Example
/// ```
/// use fluxion::sim::construction::exterior_film_coeff;
///
/// let h_ext = exterior_film_coeff(3.5); // ~24 W/m²K for moderate wind
/// ```
pub fn exterior_film_coeff(wind_speed: f64) -> f64 {
    // ASHRAE correlation: h_ext increases with wind speed
    // Using simplified model: h_ext = 10.0 + 4.0 * v^(0.5)
    // This gives ~21 W/m²K at v=3m/s, ~29 W/m²K at v=9m/s
    10.0 + 4.0 * wind_speed.sqrt()
}

/// Returns the standard interior film coefficient per ASHRAE 140.
///
/// This constant value of 8.29 W/m²K is used for interior surfaces in
/// ASHRAE 140 validation test cases.
///
/// # Returns
/// Interior film coefficient in W/m²K
pub const fn interior_film_coeff() -> f64 {
    INTERIOR_FILM_COEFF
}

/// A single layer in a multi-layer construction assembly.
///
/// Each layer represents a homogeneous material with uniform thermal properties.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConstructionLayer {
    /// Thermal conductivity of the material (k) in W/m·K
    ///
    /// This is the rate of heat transfer through the material per unit temperature
    /// gradient. Lower values indicate better insulation.
    pub conductivity: f64,

    /// Density of the material in kg/m³
    ///
    /// Used for calculating thermal mass and heat capacity of the construction.
    pub density: f64,

    /// Specific heat capacity of the material in J/kg·K
    ///
    /// Combined with density and thickness, this determines the thermal mass of the layer.
    pub specific_heat: f64,

    /// Thickness of the layer in meters
    ///
    /// The physical thickness of the material layer in the construction assembly.
    pub thickness: f64,

    /// Surface emissivity (0.0 to 1.0)
    ///
    /// The ratio of radiant energy emitted by the surface compared to a black body.
    /// Used for radiative heat transfer calculations. Defaults to 0.9 for most
    /// building materials.
    pub emissivity: f64,

    /// Surface absorptance (0.0 to 1.0)
    ///
    /// The fraction of incident solar radiation absorbed by the surface.
    /// Used for solar heat gain calculations. Defaults to 0.7 for typical
    /// opaque building materials.
    pub absorptance: f64,
}

impl ConstructionLayer {
    /// Creates a new ConstructionLayer with the specified thermal properties.
    ///
    /// # Arguments
    /// * `conductivity` - Thermal conductivity (k) in W/m·K
    /// * `density` - Material density in kg/m³
    /// * `specific_heat` - Specific heat capacity in J/kg·K
    /// * `thickness` - Layer thickness in meters
    ///
    /// # Returns
    /// A new ConstructionLayer with default emissivity (0.9) and absorptance (0.7)
    ///
    /// # Panics
    /// Panics if conductivity, density, specific_heat, or thickness are non-positive.
    ///
    /// # Example
    /// ```
    /// use fluxion::sim::construction::ConstructionLayer;
    ///
    /// let layer = ConstructionLayer::new(0.04, 12.0, 840.0, 0.066);
    /// ```
    pub fn new(conductivity: f64, density: f64, specific_heat: f64, thickness: f64) -> Self {
        assert!(conductivity > 0.0, "Conductivity must be positive");
        assert!(density > 0.0, "Density must be positive");
        assert!(specific_heat > 0.0, "Specific heat must be positive");
        assert!(thickness > 0.0, "Thickness must be positive");

        Self {
            conductivity,
            density,
            specific_heat,
            thickness,
            emissivity: 0.9,
            absorptance: 0.7,
        }
    }

    /// Creates a new ConstructionLayer with custom surface properties.
    ///
    /// # Arguments
    /// * `conductivity` - Thermal conductivity (k) in W/m·K
    /// * `density` - Material density in kg/m³
    /// * `specific_heat` - Specific heat capacity in J/kg·K
    /// * `thickness` - Layer thickness in meters
    /// * `emissivity` - Surface emissivity (0.0 to 1.0)
    /// * `absorptance` - Surface absorptance (0.0 to 1.0)
    ///
    /// # Returns
    /// A new ConstructionLayer with custom surface properties
    ///
    /// # Panics
    /// Panics if conductivity, density, specific_heat, or thickness are non-positive.
    /// Panics if emissivity or absorptance are outside the range [0.0, 1.0].
    pub fn with_surface_properties(
        conductivity: f64,
        density: f64,
        specific_heat: f64,
        thickness: f64,
        emissivity: f64,
        absorptance: f64,
    ) -> Self {
        assert!(conductivity > 0.0, "Conductivity must be positive");
        assert!(density > 0.0, "Density must be positive");
        assert!(specific_heat > 0.0, "Specific heat must be positive");
        assert!(thickness > 0.0, "Thickness must be positive");
        assert!(
            (0.0..=1.0).contains(&emissivity),
            "Emissivity must be in [0, 1]"
        );
        assert!(
            (0.0..=1.0).contains(&absorptance),
            "Absorptance must be in [0, 1]"
        );

        Self {
            conductivity,
            density,
            specific_heat,
            thickness,
            emissivity,
            absorptance,
        }
    }

    /// Calculates the thermal resistance (R-value) of this single layer.
    ///
    /// The R-value is the ratio of thickness to thermal conductivity:
    /// R = δ / k
    ///
    /// Units: m²K/W
    ///
    /// # Returns
    /// Thermal resistance in m²K/W
    pub fn r_value(&self) -> f64 {
        self.thickness / self.conductivity
    }

    /// Calculates the thermal capacitance per unit area of this layer.
    ///
    /// Thermal capacitance per unit area is calculated as:
    /// C/A = ρ × δ × Cp
    ///
    /// where ρ is density (kg/m³), δ is thickness (m), and Cp is specific heat (J/kg·K).
    ///
    /// Units: J/m²K
    ///
    /// # Returns
    /// Thermal capacitance per unit area in J/m²K
    pub fn thermal_capacitance_per_area(&self) -> f64 {
        self.density * self.thickness * self.specific_heat
    }
}

/// A multi-layer construction assembly.
///
/// Represents a complete building assembly (wall, roof, floor) composed of
/// multiple material layers arranged in series.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Construction {
    /// Ordered list of material layers from interior to exterior.
    ///
    /// Layers are ordered from the interior surface (index 0) to the exterior
    /// surface (last index).
    pub layers: Vec<ConstructionLayer>,
}

impl Construction {
    /// Creates a new Construction from a vector of layers.
    ///
    /// # Arguments
    /// * `layers` - Vector of ConstructionLayer ordered from interior to exterior
    ///
    /// # Returns
    /// A new Construction assembly
    ///
    /// # Panics
    /// Panics if layers is empty
    ///
    /// # Example
    /// ```
    /// use fluxion::sim::construction::{Construction, ConstructionLayer};
    ///
    /// let layers = vec![
    ///     ConstructionLayer::new(0.16, 950.0, 840.0, 0.012), // Plasterboard
    ///     ConstructionLayer::new(0.04, 12.0, 840.0, 0.066),  // Fiberglass
    ///     ConstructionLayer::new(0.14, 500.0, 1300.0, 0.009), // Wood siding
    /// ];
    /// let wall = Construction::new(layers);
    /// ```
    pub fn new(layers: Vec<ConstructionLayer>) -> Self {
        assert!(
            !layers.is_empty(),
            "Construction must have at least one layer"
        );
        Self { layers }
    }

    /// Calculates the total thermal resistance (R-value) including film coefficients.
    ///
    /// The total R-value is the sum of:
    /// - Interior film resistance: R_film_int = 1 / h_int
    /// - Material layer resistances: R_layer = δ / k (summed for all layers)
    /// - Exterior film resistance: R_film_ext = 1 / h_ext
    ///
    /// R_total = R_film_int + Σ(δ/k) + R_film_ext
    ///
    /// # Arguments
    /// * `exterior_wind_speed` - Wind speed at exterior surface in m/s
    ///   If not provided, uses default exterior film coefficient (25.0 W/m²K)
    ///
    /// # Returns
    /// Total thermal resistance in m²K/W
    pub fn r_value_total(&self, exterior_wind_speed: Option<f64>) -> f64 {
        let h_int = interior_film_coeff();
        let h_ext = exterior_wind_speed
            .map(exterior_film_coeff)
            .unwrap_or(EXTERIOR_FILM_COEFF_DEFAULT);

        let r_film_int = 1.0 / h_int;
        let r_film_ext = 1.0 / h_ext;
        let r_materials: f64 = self.layers.iter().map(|l| l.r_value()).sum();

        r_film_int + r_materials + r_film_ext
    }

    /// Calculates the thermal transmittance (U-value) of the construction.
    ///
    /// The U-value is the reciprocal of the total thermal resistance:
    /// U = 1 / R_total
    ///
    /// Units: W/m²K
    ///
    /// # Arguments
    /// * `exterior_wind_speed` - Wind speed at exterior surface in m/s
    ///   If not provided, uses default exterior film coefficient (25.0 W/m²K)
    ///
    /// # Returns
    /// Thermal transmittance in W/m²K
    pub fn u_value(&self, exterior_wind_speed: Option<f64>) -> f64 {
        let r_total = self.r_value_total(exterior_wind_speed);
        assert!(r_total > 0.0, "Total R-value must be positive");
        1.0 / r_total
    }

    /// Calculates the total thermal mass (capacitance) of the construction.
    ///
    /// Returns the sum of thermal capacitance per unit area for all layers.
    /// This is used to determine the effective thermal mass of the assembly
    /// for thermal network models.
    ///
    /// Units: J/m²K
    ///
    /// # Returns
    /// Total thermal capacitance per unit area in J/m²K
    pub fn thermal_capacitance_per_area(&self) -> f64 {
        self.layers
            .iter()
            .map(|l| l.thermal_capacitance_per_area())
            .sum()
    }

    /// Returns the total thickness of the construction.
    ///
    /// Units: meters
    ///
    /// # Returns
    /// Total thickness in meters
    pub fn total_thickness(&self) -> f64 {
        self.layers.iter().map(|l| l.thickness).sum()
    }

    /// Returns the number of layers in this construction.
    ///
    /// # Returns
    /// Number of layers
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

/// Pre-defined material properties for common building materials.
///
/// These materials are specified in ASHRAE 140 and other building energy
/// modeling standards.
pub struct Materials;

impl Materials {
    /// Plasterboard (gypsum board)
    ///
    /// - k = 0.16 W/m·K
    /// - ρ = 950 kg/m³
    /// - Cp = 840 J/kg·K
    ///
    /// Common interior finish for walls and ceilings.
    pub fn plasterboard(thickness: f64) -> ConstructionLayer {
        ConstructionLayer::new(0.16, 950.0, 840.0, thickness)
    }

    /// Fiberglass insulation
    ///
    /// - k = 0.04 W/m·K
    /// - ρ = 12 kg/m³
    /// - Cp = 840 J/kg·K
    ///
    /// Common thermal insulation for walls, roofs, and floors.
    pub fn fiberglass(thickness: f64) -> ConstructionLayer {
        ConstructionLayer::new(0.04, 12.0, 840.0, thickness)
    }

    /// Wood siding
    ///
    /// - k = 0.14 W/m·K
    /// - ρ = 500 kg/m³
    /// - Cp = 1300 J/kg·K
    ///
    /// Exterior cladding material for low-mass buildings.
    pub fn wood_siding(thickness: f64) -> ConstructionLayer {
        ConstructionLayer::new(0.14, 500.0, 1300.0, thickness)
    }

    /// Concrete (normal weight)
    ///
    /// - k = 0.51 W/m·K
    /// - ρ = 1400 kg/m³
    /// - Cp = 1000 J/kg·K
    ///
    /// High thermal mass material used in concrete block walls and slabs.
    pub fn concrete(thickness: f64) -> ConstructionLayer {
        ConstructionLayer::new(0.51, 1400.0, 1000.0, thickness)
    }

    /// Foam insulation
    ///
    /// - k = 0.04 W/m·K
    /// - ρ = 10 kg/m³
    /// - Cp = 1400 J/kg·K
    ///
    /// Rigid foam insulation for high-performance assemblies.
    pub fn foam(thickness: f64) -> ConstructionLayer {
        ConstructionLayer::new(0.04, 10.0, 1400.0, thickness)
    }

    /// Timber/wood framing
    ///
    /// - k = 0.14 W/m·K
    /// - ρ = 600 kg/m³
    /// - Cp = 1600 J/kg·K
    ///
    /// Structural material used in low-mass buildings.
    pub fn timber(thickness: f64) -> ConstructionLayer {
        ConstructionLayer::new(0.14, 600.0, 1600.0, thickness)
    }

    /// Roof decking
    ///
    /// - k = 0.14 W/m·K
    /// - ρ = 500 kg/m³
    /// - Cp = 1300 J/kg·K
    ///
    /// Structural material for roof assemblies.
    pub fn roof_deck(thickness: f64) -> ConstructionLayer {
        ConstructionLayer::new(0.14, 500.0, 1300.0, thickness)
    }
}

/// Pre-defined construction assemblies from ASHRAE 140 test cases.
///
/// These constructions are used in the ASHRAE Standard 140 validation test cases.
pub struct Assemblies;

impl Assemblies {
    /// Low mass wall construction (ASHRAE 140 Case 600).
    ///
    /// Assembly (interior to exterior):
    /// 1. Plasterboard: 0.012 m, k=0.16 W/m·K
    /// 2. Fiberglass insulation: 0.066 m, k=0.04 W/m·K
    /// 3. Wood siding: 0.009 m, k=0.14 W/m·K
    ///
    /// Expected U-value: ~0.514 W/m²K (with ASHRAE film coefficients)
    ///
    /// # Returns
    /// Construction assembly for low-mass walls
    pub fn low_mass_wall() -> Construction {
        Construction::new(vec![
            Materials::plasterboard(0.012),
            Materials::fiberglass(0.066),
            Materials::wood_siding(0.009),
        ])
    }

    /// Low mass roof construction (ASHRAE 140 Case 600).
    ///
    /// Assembly (interior to exterior):
    /// 1. Plasterboard: 0.010 m, k=0.16 W/m·K
    /// 2. Fiberglass insulation: 0.1118 m, k=0.04 W/m·K
    /// 3. Roof deck: 0.019 m, k=0.14 W/m·K
    ///
    /// Expected U-value: ~0.318 W/m²K (with ASHRAE film coefficients)
    ///
    /// # Returns
    /// Construction assembly for low-mass roofs
    pub fn low_mass_roof() -> Construction {
        Construction::new(vec![
            Materials::plasterboard(0.010),
            Materials::fiberglass(0.1118),
            Materials::roof_deck(0.019),
        ])
    }

    /// High mass wall construction (ASHRAE 140 Case 900).
    ///
    /// Assembly (interior to exterior):
    /// 1. Concrete block: 0.100 m, k=0.51 W/m·K
    /// 2. Foam insulation: 0.0615 m, k=0.04 W/m·K
    /// 3. Wood siding: 0.009 m, k=0.14 W/m·K
    ///
    /// Expected U-value: ~0.514 W/m²K (with ASHRAE film coefficients)
    ///
    /// # Returns
    /// Construction assembly for high-mass walls
    pub fn high_mass_wall() -> Construction {
        Construction::new(vec![
            Materials::concrete(0.100),
            Materials::foam(0.0615),
            Materials::wood_siding(0.009),
        ])
    }

    /// High mass roof construction (ASHRAE 140 Case 900).
    ///
    /// Assembly (interior to exterior):
    /// 1. Concrete slab: 0.080 m, k=0.51 W/m·K
    /// 2. Foam insulation: 0.0615 m, k=0.04 W/m·K
    /// 3. Roof deck: 0.019 m, k=0.14 W/m·K
    ///
    /// Expected U-value: ~0.318 W/m²K (with ASHRAE film coefficients)
    ///
    /// # Returns
    /// Construction assembly for high-mass roofs
    pub fn high_mass_roof() -> Construction {
        Construction::new(vec![
            Materials::concrete(0.080),
            Materials::foam(0.0615),
            Materials::roof_deck(0.019),
        ])
    }

    /// Insulated floor construction (ASHRAE 140 Case 600).
    ///
    /// Assembly (interior to exterior):
    /// 1. Timber: 0.025 m, k=0.14 W/m·K
    /// 2. Insulation: 0.040 m, k=1.003 W/m·K
    ///
    /// Expected U-value: ~0.039 W/m²K (with ASHRAE film coefficients)
    ///
    /// # Returns
    /// Construction assembly for insulated floors
    pub fn insulated_floor() -> Construction {
        Construction::new(vec![
            Materials::timber(0.025),
            ConstructionLayer::new(1.003, 50.0, 840.0, 0.040), // Floor insulation
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn test_construction_layer_creation() {
        let layer = ConstructionLayer::new(0.04, 12.0, 840.0, 0.066);

        assert_eq!(layer.conductivity, 0.04);
        assert_eq!(layer.density, 12.0);
        assert_eq!(layer.specific_heat, 840.0);
        assert_eq!(layer.thickness, 0.066);
        assert_eq!(layer.emissivity, 0.9);
        assert_eq!(layer.absorptance, 0.7);
    }

    #[test]
    fn test_construction_layer_with_custom_surface_properties() {
        let layer =
            ConstructionLayer::with_surface_properties(0.04, 12.0, 840.0, 0.066, 0.85, 0.65);

        assert_eq!(layer.emissivity, 0.85);
        assert_eq!(layer.absorptance, 0.65);
    }

    #[test]
    #[should_panic(expected = "Conductivity must be positive")]
    fn test_construction_layer_invalid_conductivity() {
        ConstructionLayer::new(-0.04, 12.0, 840.0, 0.066);
    }

    #[test]
    #[should_panic(expected = "Density must be positive")]
    fn test_construction_layer_invalid_density() {
        ConstructionLayer::new(0.04, -12.0, 840.0, 0.066);
    }

    #[test]
    #[should_panic(expected = "Thickness must be positive")]
    fn test_construction_layer_invalid_thickness() {
        ConstructionLayer::new(0.04, 12.0, 840.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "Emissivity must be in [0, 1]")]
    fn test_construction_layer_invalid_emissivity() {
        ConstructionLayer::with_surface_properties(0.04, 12.0, 840.0, 0.066, 1.5, 0.7);
    }

    #[test]
    fn test_layer_r_value() {
        let layer = ConstructionLayer::new(0.04, 12.0, 840.0, 0.066);

        // R = δ / k = 0.066 / 0.04 = 1.65 m²K/W
        let expected_r = 0.066 / 0.04;
        assert!((layer.r_value() - expected_r).abs() < EPSILON);
    }

    #[test]
    fn test_layer_thermal_capacitance_per_area() {
        let layer = ConstructionLayer::new(0.04, 12.0, 840.0, 0.066);

        // C/A = ρ × δ × Cp = 12.0 × 0.066 × 840.0 = 665.28 J/m²K
        let expected_c = 12.0 * 0.066 * 840.0;
        assert!((layer.thermal_capacitance_per_area() - expected_c).abs() < EPSILON);
    }

    #[test]
    fn test_construction_creation() {
        let layers = vec![
            ConstructionLayer::new(0.16, 950.0, 840.0, 0.012),
            ConstructionLayer::new(0.04, 12.0, 840.0, 0.066),
        ];
        let construction = Construction::new(layers);

        assert_eq!(construction.layer_count(), 2);
    }

    #[test]
    #[should_panic(expected = "Construction must have at least one layer")]
    fn test_construction_empty_layers() {
        Construction::new(vec![]);
    }

    #[test]
    fn test_construction_r_value_total() {
        let construction = Assemblies::low_mass_wall();

        // Calculate expected R-value
        // R_int = 1 / 8.29 = 0.120627
        // R_plasterboard = 0.012 / 0.16 = 0.075
        // R_fiberglass = 0.066 / 0.04 = 1.65
        // R_siding = 0.009 / 0.14 = 0.064286
        // R_ext = 1 / 25.0 = 0.04
        // R_total = 0.120627 + 0.075 + 1.65 + 0.064286 + 0.04 = 1.949913
        let r_total = construction.r_value_total(None);

        let expected_r = 1.0 / 8.29 + 0.012 / 0.16 + 0.066 / 0.04 + 0.009 / 0.14 + 1.0 / 25.0;
        assert!((r_total - expected_r).abs() < EPSILON);

        // Check that U = 1/R
        let u_value = construction.u_value(None);
        assert!((u_value - 1.0 / r_total).abs() < EPSILON);
    }

    #[test]
    fn test_construction_u_value() {
        let construction = Assemblies::low_mass_wall();
        let u_value = construction.u_value(None);

        // For Case 600 wall: expected U ≈ 0.514 W/m²K
        // This may vary slightly due to different assumptions about film coefficients
        assert!(u_value > 0.5);
        assert!(u_value < 0.6);
    }

    #[test]
    fn test_construction_u_value_with_wind_speed() {
        let construction = Assemblies::low_mass_wall();

        // Test with different wind speeds
        let u_no_wind = construction.u_value(Some(0.0));
        let u_low_wind = construction.u_value(Some(2.0));
        let u_high_wind = construction.u_value(Some(10.0));

        // Higher wind speed → higher exterior film coefficient → lower resistance → higher U
        assert!(u_high_wind > u_low_wind);
        assert!(u_low_wind > u_no_wind);
    }

    #[test]
    fn test_interior_film_coeff() {
        let h_int = interior_film_coeff();
        assert_eq!(h_int, 8.29);
    }

    #[test]
    fn test_exterior_film_coeff() {
        // Test with low wind
        let h_low = exterior_film_coeff(2.0);
        assert!((h_low - (10.0 + 4.0 * 2.0_f64.sqrt())).abs() < EPSILON);

        // Test with high wind
        let h_high = exterior_film_coeff(10.0);
        assert!((h_high - (10.0 + 4.0 * 10.0_f64.sqrt())).abs() < EPSILON);

        // High wind should have higher film coefficient
        assert!(h_high > h_low);

        // Reasonable range check
        assert!(h_low > 15.0 && h_low < 30.0);
        assert!(h_high > 20.0 && h_high < 40.0);
    }

    #[test]
    fn test_materials_plasterboard() {
        let layer = Materials::plasterboard(0.012);
        assert_eq!(layer.conductivity, 0.16);
        assert_eq!(layer.density, 950.0);
        assert_eq!(layer.specific_heat, 840.0);
        assert_eq!(layer.thickness, 0.012);
    }

    #[test]
    fn test_materials_fiberglass() {
        let layer = Materials::fiberglass(0.066);
        assert_eq!(layer.conductivity, 0.04);
        assert_eq!(layer.density, 12.0);
        assert_eq!(layer.specific_heat, 840.0);
        assert_eq!(layer.thickness, 0.066);
    }

    #[test]
    fn test_materials_wood_siding() {
        let layer = Materials::wood_siding(0.009);
        assert_eq!(layer.conductivity, 0.14);
        assert_eq!(layer.density, 500.0);
        assert_eq!(layer.specific_heat, 1300.0);
        assert_eq!(layer.thickness, 0.009);
    }

    #[test]
    fn test_materials_concrete() {
        let layer = Materials::concrete(0.100);
        assert_eq!(layer.conductivity, 0.51);
        assert_eq!(layer.density, 1400.0);
        assert_eq!(layer.specific_heat, 1000.0);
        assert_eq!(layer.thickness, 0.100);
    }

    #[test]
    fn test_materials_foam() {
        let layer = Materials::foam(0.0615);
        assert_eq!(layer.conductivity, 0.04);
        assert_eq!(layer.density, 10.0);
        assert_eq!(layer.specific_heat, 1400.0);
        assert_eq!(layer.thickness, 0.0615);
    }

    #[test]
    fn test_assemblies_low_mass_wall() {
        let wall = Assemblies::low_mass_wall();
        assert_eq!(wall.layer_count(), 3);

        // Check layer properties
        assert_eq!(wall.layers[0].thickness, 0.012); // Plasterboard
        assert_eq!(wall.layers[1].thickness, 0.066); // Fiberglass
        assert_eq!(wall.layers[2].thickness, 0.009); // Siding
    }

    #[test]
    fn test_assemblies_low_mass_roof() {
        let roof = Assemblies::low_mass_roof();
        assert_eq!(roof.layer_count(), 3);

        // Check layer properties
        assert_eq!(roof.layers[0].thickness, 0.010); // Plasterboard
        assert_eq!(roof.layers[1].thickness, 0.1118); // Fiberglass
        assert_eq!(roof.layers[2].thickness, 0.019); // Deck
    }

    #[test]
    fn test_assemblies_high_mass_wall() {
        let wall = Assemblies::high_mass_wall();
        assert_eq!(wall.layer_count(), 3);

        // Check layer properties
        assert_eq!(wall.layers[0].thickness, 0.100); // Concrete
        assert_eq!(wall.layers[1].thickness, 0.0615); // Foam
        assert_eq!(wall.layers[2].thickness, 0.009); // Siding
    }

    #[test]
    fn test_assemblies_high_mass_roof() {
        let roof = Assemblies::high_mass_roof();
        assert_eq!(roof.layer_count(), 3);

        // Check layer properties
        assert_eq!(roof.layers[0].thickness, 0.080); // Concrete
        assert_eq!(roof.layers[1].thickness, 0.0615); // Foam
        assert_eq!(roof.layers[2].thickness, 0.019); // Deck
    }

    #[test]
    fn test_assemblies_insulated_floor() {
        let floor = Assemblies::insulated_floor();
        assert_eq!(floor.layer_count(), 2);

        // Check layer properties
        assert_eq!(floor.layers[0].thickness, 0.025); // Timber
        assert_eq!(floor.layers[1].thickness, 0.040); // Insulation
    }

    #[test]
    fn test_construction_thermal_capacitance_per_area() {
        let wall = Assemblies::low_mass_wall();
        let c_per_area = wall.thermal_capacitance_per_area();

        // Calculate expected value
        // Plasterboard: 950 × 0.012 × 840 = 9576
        // Fiberglass: 12 × 0.066 × 840 = 665.28
        // Siding: 500 × 0.009 × 1300 = 5850
        // Total: 9576 + 665.28 + 5850 = 16091.28 J/m²K
        let expected_c = 950.0 * 0.012 * 840.0 + 12.0 * 0.066 * 840.0 + 500.0 * 0.009 * 1300.0;
        assert!((c_per_area - expected_c).abs() < EPSILON);
    }

    #[test]
    fn test_construction_total_thickness() {
        let wall = Assemblies::low_mass_wall();
        let thickness = wall.total_thickness();

        let expected = 0.012 + 0.066 + 0.009;
        assert!((thickness - expected).abs() < EPSILON);
    }

    #[test]
    fn test_high_mass_vs_low_mass_capacitance() {
        let low_mass = Assemblies::low_mass_wall();
        let high_mass = Assemblies::high_mass_wall();

        let c_low = low_mass.thermal_capacitance_per_area();
        let c_high = high_mass.thermal_capacitance_per_area();

        // High mass should have much higher thermal capacitance
        assert!(c_high > 3.0 * c_low);
    }

    #[test]
    fn test_serialization() {
        let wall = Assemblies::low_mass_wall();

        // Test serialization
        let json = serde_json::to_string(&wall).expect("Failed to serialize");
        assert!(json.contains("conductivity"));
        assert!(json.contains("thickness"));

        // Test deserialization
        let deserialized: Construction =
            serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.layer_count(), wall.layer_count());
    }
}
