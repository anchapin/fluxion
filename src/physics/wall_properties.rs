//! Wall Properties - Slim interface for thermal solver requirements.
//!
//! This module provides `WallProperties` and `LayerProperties` structs that
//! expose only what heat conduction solvers need from `BuildingAssembly`.
//!
//! # Purpose
//!
//! The solvers (CTF, FD) only need thermal properties to compute heat flux.
//! By converting `BuildingAssembly` to `WallProperties` at the adapter seam,
//! we hide `BuildingAssembly` internals from solver implementations.
//!
//! # Example
//!
//! ```rust
//! use fluxion::physics::wall_properties::WallProperties;
//! use fluxion::sim::assembly::{AssemblyBuilder, ConcreteMaterial};
//!
//! let assembly = AssemblyBuilder::new("Test Wall".to_string())
//!     .add_layer(Box::new(ConcreteMaterial::new(0.2)))
//!     .build()
//!     .unwrap();
//!
//! let props = WallProperties::from_assembly(&assembly);
//! assert_eq!(props.layers.len(), 1);
//! assert!(props.total_thermal_mass_kj_m2 > 0.0);
//! ```
//!
//! # Architecture
//!
//! ```text
//! SolverManager
//!     |
//!     v
//! BuildingAssembly  ----->  WallProperties (at seam)
//!     |                              |
//!     v                              v
//! CTFSolverWrapper              CTFSolver (internals)
//! FDSolverWrapper               FDSolver (internals)
//! ```
//!
//! The `HeatConductionSolver` trait still takes `BuildingAssembly` for
//! backwards compatibility. The conversion happens at the call site
//! in `SolverManager::get_or_create_solver()`.

use crate::sim::assembly::BuildingAssembly;

/// Layer properties needed by thermal solvers.
///
/// This is a flat, owned struct with no internal structure —
/// just the thermal properties a solver needs to compute heat transfer.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerProperties {
    /// Layer name
    pub name: String,
    /// Thickness [m]
    pub thickness_m: f64,
    /// Thermal conductivity [W/mK]
    pub conductivity_w_mk: f64,
    /// Density [kg/m³]
    pub density_kg_m3: f64,
    /// Specific heat capacity [J/kgK]
    pub specific_heat_j_kgk: f64,
    /// Thermal mass per unit area [kJ/m²K]
    pub thermal_mass_kj_m2: f64,
}

impl LayerProperties {
    /// Create layer properties from a material layer trait object.
    pub fn from_material_layer(layer: &dyn crate::sim::assembly::MaterialLayer) -> Self {
        let thickness = layer.thickness();
        let density = layer.density();
        let specific_heat = layer.specific_heat();
        let thermal_mass_kj_m2 = density * specific_heat * thickness / 1000.0;

        Self {
            name: layer.name().to_string(),
            thickness_m: thickness,
            conductivity_w_mk: layer.conductivity(),
            density_kg_m3: density,
            specific_heat_j_kgk: specific_heat,
            thermal_mass_kj_m2,
        }
    }
}

/// Wall properties needed by thermal solvers.
///
/// This is a flat, owned struct with surface resistances and layer data.
/// No `BuildingAssembly` internals leak past this interface.
#[derive(Debug, Clone, PartialEq)]
pub struct WallProperties {
    /// All layers (exterior to interior)
    pub layers: Vec<LayerProperties>,
    /// Total thermal mass per unit area [kJ/m²K]
    pub total_thermal_mass_kj_m2: f64,
    /// Interior surface resistance [m²K/W]
    pub surface_resistance_inside: f64,
    /// Exterior surface resistance [m²K/W]
    pub surface_resistance_outside: f64,
}

impl WallProperties {
    /// Surface resistance for interior per ASHRAE 140 Section 5.2 [m²K/W]
    ///
    /// h_int = 8.29 W/m²K → R = 1/8.29 ≈ 0.12063 m²K/W
    pub const R_INT: f64 = 1.0 / 8.29; // ASHRAE 140 Section 5.2: h_int = 8.29 W/m²K

    /// Surface resistance for exterior per ASHRAE 140 Section 5.2 [m²K/W]
    ///
    /// h_ext = 29.3 W/m²K at 6.7 m/s wind speed → R = 1/29.3 ≈ 0.03413 m²K/W
    pub const R_EXT: f64 = 1.0 / 29.3; // ASHRAE 140 Section 5.2: h_ext = 29.3 W/m²K

    /// Create wall properties from a building assembly.
    ///
    /// This conversion happens once at the solver seam. If `BuildingAssembly`
    /// internals change, only this method needs updating.
    pub fn from_assembly(assembly: &BuildingAssembly) -> Self {
        let layers: Vec<LayerProperties> = assembly
            .layers
            .iter()
            .map(|layer| LayerProperties::from_material_layer(layer.as_ref()))
            .collect();

        let total_thermal_mass_kj_m2: f64 = layers.iter().map(|l| l.thermal_mass_kj_m2).sum();

        Self {
            layers,
            total_thermal_mass_kj_m2,
            surface_resistance_inside: Self::R_INT,
            surface_resistance_outside: Self::R_EXT,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::assembly::{AssemblyBuilder, ConcreteMaterial, InsulationMaterial};

    #[test]
    fn test_layer_properties_from_concrete() {
        let concrete = ConcreteMaterial::new(0.2);
        let layer = LayerProperties::from_material_layer(&concrete);

        assert_eq!(layer.name, "Concrete");
        assert!((layer.thickness_m - 0.2).abs() < 1e-10);
        assert!((layer.conductivity_w_mk - 1.4).abs() < 1e-10);
        assert!((layer.density_kg_m3 - 2300.0).abs() < 1e-10);
        assert!((layer.specific_heat_j_kgk - 840.0).abs() < 1e-10);

        let expected_mass = 2300.0 * 840.0 * 0.2 / 1000.0;
        assert!((layer.thermal_mass_kj_m2 - expected_mass).abs() < 0.01);
    }

    #[test]
    fn test_layer_properties_thermal_mass_calculation() {
        let insulation = InsulationMaterial::new(0.1);
        let layer = LayerProperties::from_material_layer(&insulation);

        let expected_mass = 50.0 * 840.0 * 0.1 / 1000.0;
        assert!((layer.thermal_mass_kj_m2 - expected_mass).abs() < 0.01);
    }

    #[test]
    fn test_wall_properties_from_assembly_single_layer() {
        let assembly = AssemblyBuilder::new("Single Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .build()
            .unwrap();

        let props = WallProperties::from_assembly(&assembly);

        assert_eq!(props.layers.len(), 1);
        assert_eq!(props.layers[0].name, "Concrete");
        assert!((props.surface_resistance_inside - 1.0 / 8.29).abs() < 1e-10);
        assert!((props.surface_resistance_outside - 1.0 / 29.3).abs() < 1e-10);

        let expected_total: f64 = props.layers.iter().map(|l| l.thermal_mass_kj_m2).sum();
        assert!((props.total_thermal_mass_kj_m2 - expected_total).abs() < 0.01);
    }

    #[test]
    fn test_wall_properties_from_assembly_multi_layer() {
        let assembly = AssemblyBuilder::new("Composite Wall".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.2)))
            .add_layer(Box::new(InsulationMaterial::new(0.1)))
            .add_layer(Box::new(ConcreteMaterial::new(0.012)))
            .build()
            .unwrap();

        let props = WallProperties::from_assembly(&assembly);

        assert_eq!(props.layers.len(), 3);
        assert_eq!(props.layers[0].name, "Concrete");
        assert_eq!(props.layers[1].name, "Insulation");
        assert_eq!(props.layers[2].name, "Concrete");

        let expected_total: f64 = props.layers.iter().map(|l| l.thermal_mass_kj_m2).sum();
        assert!((props.total_thermal_mass_kj_m2 - expected_total).abs() < 0.01);
    }

    #[test]
    fn test_wall_properties_surface_resistances() {
        let assembly = AssemblyBuilder::new("Test".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1)))
            .build()
            .unwrap();

        let props = WallProperties::from_assembly(&assembly);

        assert!((props.surface_resistance_inside - 1.0 / 8.29).abs() < 1e-10);
        assert!((props.surface_resistance_outside - 1.0 / 29.3).abs() < 1e-10);
    }

    #[test]
    fn test_wall_properties_clone() {
        let assembly = AssemblyBuilder::new("Clone Test".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.15)))
            .build()
            .unwrap();

        let props1 = WallProperties::from_assembly(&assembly);
        let props2 = props1.clone();

        assert_eq!(props1.layers.len(), props2.layers.len());
        assert!((props1.total_thermal_mass_kj_m2 - props2.total_thermal_mass_kj_m2).abs() < 1e-10);
    }

    #[test]
    fn test_layer_properties_clone() {
        let concrete = ConcreteMaterial::new(0.2);
        let layer1 = LayerProperties::from_material_layer(&concrete);
        let layer2 = layer1.clone();

        assert_eq!(layer1.name, layer2.name);
        assert!((layer1.thickness_m - layer2.thickness_m).abs() < 1e-10);
    }

    #[test]
    fn test_layer_properties_debug() {
        let concrete = ConcreteMaterial::new(0.1);
        let layer = LayerProperties::from_material_layer(&concrete);
        let debug_str = format!("{:?}", layer);
        assert!(debug_str.contains("Concrete"));
        assert!(debug_str.contains("thickness_m"));
    }

    #[test]
    fn test_wall_properties_debug() {
        let assembly = AssemblyBuilder::new("Debug Test".to_string())
            .add_layer(Box::new(ConcreteMaterial::new(0.1)))
            .build()
            .unwrap();
        let props = WallProperties::from_assembly(&assembly);
        let debug_str = format!("{:?}", props);
        assert!(debug_str.contains("WallProperties"));
        assert!(debug_str.contains("layers"));
    }
}
