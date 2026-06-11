//! Minimal wall specification for isolated conduction testing.
//!
//! This module provides a lightweight `WallSpec` struct that contains only the
//! thermophysical properties needed for conduction calculations, without
//! depending on the full `BuildingAssembly` (which requires MaterialLayer trait
//! objects, YAML loading, and surface classification).
//!
//! # Purpose
//!
//! Enables data-driven conduction testing with just a CSV + WallSpec, supporting
//! the Phase 1 validation strategy of testing each module in isolation.
//!
//! # Example
//!
//! ```rust
//! use fluxion::physics::wall_spec::WallSpec;
//!
//! let spec = WallSpec::single_layer(
//!     "200mm Concrete",
//!     0.2,    // thickness [m]
//!     1.73,   // conductivity [W/m·K]
//!     2243.0, // density [kg/m³]
//!     837.0,  // specific heat [J/kg·K]
//! );
//!
//! assert!((spec.total_r_value() - 0.2 / 1.73).abs() < 1e-10);
//! assert!((spec.thermal_capacity() - 2243.0 * 837.0 * 0.2).abs() < 1.0);
//! ```

use crate::physics::ctf_coefficients::CTFMaterial;
use crate::physics::fd_discretization::MaterialLayer;
use crate::physics::wall_properties::{LayerProperties, WallProperties};
use crate::sim::assembly::BuildingAssembly;

/// Minimal specification for a single wall layer.
///
/// Contains only the four thermophysical properties required for
/// 1D transient heat conduction: thickness, conductivity, density, specific heat.
#[derive(Debug, Clone)]
pub struct LayerSpec {
    /// Layer name/identifier
    pub name: String,
    /// Layer thickness [m]
    pub thickness: f64,
    /// Thermal conductivity [W/(m·K)]
    pub conductivity: f64,
    /// Material density [kg/m³]
    pub density: f64,
    /// Specific heat capacity [J/(kg·K)]
    pub specific_heat: f64,
}

impl LayerSpec {
    /// Create a new layer specification.
    ///
    /// # Panics
    /// Panics if thickness, conductivity, density, or specific_heat are non-positive.
    pub fn new(
        name: impl Into<String>,
        thickness: f64,
        conductivity: f64,
        density: f64,
        specific_heat: f64,
    ) -> Self {
        assert!(thickness > 0.0, "Thickness must be positive");
        assert!(conductivity > 0.0, "Conductivity must be positive");
        assert!(density > 0.0, "Density must be positive");
        assert!(specific_heat > 0.0, "Specific heat must be positive");

        Self {
            name: name.into(),
            thickness,
            conductivity,
            density,
            specific_heat,
        }
    }

    /// Thermal resistance [m²·K/W]
    pub fn r_value(&self) -> f64 {
        self.thickness / self.conductivity
    }

    /// Thermal capacity per unit area [J/(m²·K)]
    pub fn thermal_capacity(&self) -> f64 {
        self.density * self.thickness * self.specific_heat
    }

    /// Thermal diffusivity [m²/s]
    pub fn diffusivity(&self) -> f64 {
        self.conductivity / (self.density * self.specific_heat)
    }
}

/// Wall specification composed of one or more material layers.
///
/// This is a data-only struct with no trait object dependencies, suitable
/// for use in integration tests and parameter sweeps.
#[derive(Debug, Clone)]
pub struct WallSpec {
    /// Wall name/identifier
    pub name: String,
    /// Material layers (exterior to interior)
    pub layers: Vec<LayerSpec>,
}

impl WallSpec {
    /// Create a single-layer wall specification.
    pub fn single_layer(
        name: &str,
        thickness: f64,
        conductivity: f64,
        density: f64,
        specific_heat: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            layers: vec![LayerSpec::new(
                name,
                thickness,
                conductivity,
                density,
                specific_heat,
            )],
        }
    }

    /// Create a multi-layer wall specification.
    pub fn multi_layer(name: &str, layers: Vec<LayerSpec>) -> Self {
        assert!(!layers.is_empty(), "Wall must have at least one layer");
        Self {
            name: name.to_string(),
            layers,
        }
    }

    /// Total thermal resistance [m²·K/W]
    pub fn total_r_value(&self) -> f64 {
        self.layers.iter().map(|l| l.r_value()).sum()
    }

    /// Total wall thickness [m]
    pub fn total_thickness(&self) -> f64 {
        self.layers.iter().map(|l| l.thickness).sum()
    }

    /// Total thermal capacity per unit area [J/(m²·K)]
    pub fn thermal_capacity(&self) -> f64 {
        self.layers.iter().map(|l| l.thermal_capacity()).sum()
    }

    /// Convert to CTF material list for CTF solver initialization.
    pub fn to_ctf_materials(&self) -> Vec<CTFMaterial> {
        self.layers
            .iter()
            .map(|l| {
                CTFMaterial::new(
                    l.name.as_str(),
                    l.thickness,
                    l.conductivity,
                    l.density,
                    l.specific_heat,
                )
            })
            .collect()
    }

    /// Convert to FD material layers for finite difference solver initialization.
    pub fn to_fd_material_layers(&self) -> Vec<MaterialLayer> {
        self.layers
            .iter()
            .map(|l| {
                MaterialLayer::new(
                    l.name.as_str(),
                    l.thickness,
                    l.conductivity,
                    l.density,
                    l.specific_heat,
                )
            })
            .collect()
    }

    /// Create a WallSpec from a BuildingAssembly.
    ///
    /// This conversion enables the HeatConductionSolver trait to accept WallSpec
    /// instead of BuildingAssembly, decoupling solver initialization from the
    /// assembly's internal representation.
    pub fn from_assembly(assembly: &BuildingAssembly) -> Self {
        let layers: Vec<LayerSpec> = assembly
            .layers
            .iter()
            .map(|layer| {
                let l = layer.as_ref();
                LayerSpec::new(
                    l.name(),
                    l.thickness(),
                    l.conductivity(),
                    l.density(),
                    l.specific_heat(),
                )
            })
            .collect();

        Self {
            name: assembly.name.clone(),
            layers,
        }
    }

    /// Convert to WallProperties for wrapper-based solver initialization.
    ///
    /// This enables WallSpec to be used with any solver that accepts BuildingAssembly
    /// via the WallProperties intermediate representation.
    pub fn to_wall_properties(&self) -> WallProperties {
        let layers: Vec<LayerProperties> = self
            .layers
            .iter()
            .map(|l| LayerProperties {
                name: l.name.clone(),
                thickness_m: l.thickness,
                conductivity_w_mk: l.conductivity,
                density_kg_m3: l.density,
                specific_heat_j_kgk: l.specific_heat,
                thermal_mass_kj_m2: l.thermal_capacity() / 1000.0,
            })
            .collect();

        let total_thermal_mass_kj_m2: f64 = layers.iter().map(|l| l.thermal_mass_kj_m2).sum();

        WallProperties {
            layers,
            total_thermal_mass_kj_m2,
            surface_resistance_inside: 1.0 / 8.29,
            surface_resistance_outside: 1.0 / 29.3,
        }
    }
}

/// 200mm concrete slab per issue #946 specification.
///
/// Properties taken from ASHRAE Handbook, Chapter 26:
/// - k = 1.73 W/(m·K) (normal weight concrete, 2243 kg/m³)
/// - ρ = 2243 kg/m³ (normal weight concrete)
/// - cₚ = 837 J/(kg·K) (concrete at ~24°C mean temperature)
/// - Thickness = 0.2 m
///
/// Thermal capacity: C = ρ × cₚ × L = 2243 × 837 × 0.2 = 375,448 J/(m²·K)
pub fn concrete_200mm_spec() -> WallSpec {
    WallSpec::single_layer(
        "200mm Concrete",
        0.2,    // thickness [m]
        1.73,   // conductivity [W/(m·K)]
        2243.0, // density [kg/m³]
        837.0,  // specific heat [J/(kg·K)]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_spec_creation() {
        let layer = LayerSpec::new("Test", 0.1, 1.0, 2000.0, 800.0);
        assert_eq!(layer.name, "Test");
        assert!((layer.thickness - 0.1).abs() < 1e-10);
        assert!((layer.conductivity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_layer_r_value() {
        let layer = LayerSpec::new("Concrete", 0.2, 1.73, 2243.0, 837.0);
        let expected_r = 0.2 / 1.73;
        assert!((layer.r_value() - expected_r).abs() < 1e-10);
    }

    #[test]
    fn test_layer_thermal_capacity() {
        let layer = LayerSpec::new("Concrete", 0.2, 1.73, 2243.0, 837.0);
        let expected = 2243.0 * 837.0 * 0.2; // 375,448 J/(m²·K)
        assert!((layer.thermal_capacity() - expected).abs() < 1.0);
    }

    #[test]
    fn test_layer_diffusivity() {
        let layer = LayerSpec::new("Concrete", 0.2, 1.73, 2243.0, 837.0);
        let expected = 1.73 / (2243.0 * 837.0); // ~9.22e-7 m²/s
        assert!((layer.diffusivity() - expected).abs() < 1e-12);
    }

    #[test]
    fn test_wall_spec_single_layer() {
        let spec = concrete_200mm_spec();
        assert_eq!(spec.layers.len(), 1);
        assert!((spec.total_thickness() - 0.2).abs() < 1e-10);
        assert!((spec.thermal_capacity() - 375478.2).abs() < 1.0);
    }

    #[test]
    fn test_wall_spec_total_r_value() {
        let spec = concrete_200mm_spec();
        let expected = 0.2 / 1.73;
        assert!((spec.total_r_value() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_wall_spec_multi_layer() {
        let spec = WallSpec::multi_layer(
            "Composite Wall",
            vec![
                LayerSpec::new("Brick", 0.1, 0.81, 1920.0, 790.0),
                LayerSpec::new("Insulation", 0.05, 0.04, 50.0, 840.0),
                LayerSpec::new("Gypsum", 0.013, 0.16, 800.0, 1090.0),
            ],
        );
        assert_eq!(spec.layers.len(), 3);
        assert!((spec.total_thickness() - 0.163).abs() < 1e-10);
        assert!(spec.total_r_value() > 1.0); // Insulation adds significant R
    }

    #[test]
    fn test_to_ctf_materials() {
        let spec = concrete_200mm_spec();
        let materials = spec.to_ctf_materials();
        assert_eq!(materials.len(), 1);
    }

    #[test]
    fn test_to_fd_material_layers() {
        let spec = concrete_200mm_spec();
        let layers = spec.to_fd_material_layers();
        assert_eq!(layers.len(), 1);
    }

    #[test]
    fn test_to_wall_properties() {
        let spec = concrete_200mm_spec();
        let props = spec.to_wall_properties();
        assert_eq!(props.layers.len(), 1);
        assert!((props.layers[0].thickness_m - 0.2).abs() < 1e-10);
        assert!((props.total_thermal_mass_kj_m2 - 375.448).abs() < 0.1);
    }

    #[test]
    #[should_panic(expected = "Thickness must be positive")]
    fn test_layer_spec_zero_thickness() {
        LayerSpec::new("Bad", 0.0, 1.0, 2000.0, 800.0);
    }

    #[test]
    #[should_panic(expected = "Conductivity must be positive")]
    fn test_layer_spec_negative_conductivity() {
        LayerSpec::new("Bad", 0.1, -1.0, 2000.0, 800.0);
    }
}
