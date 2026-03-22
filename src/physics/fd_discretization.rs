//! Finite Difference discretization for multi-layer wall heat conduction.
//!
//! This module provides spatial discretization of building envelope constructions
//! into control volumes for implicit finite difference heat transfer simulation.
//!
//! # Overview
//!
//! The `WallDiscretization` struct converts a multi-layer construction into
//! a 1D grid of nodes with associated thermal properties (density, specific heat,
//! conductivity). Each layer is divided into uniform control volumes.
//!
//! # Example
//!
//! ```rust
//! use fluxion::physics::fd_discretization::{WallDiscretization, MaterialLayer};
//!
//! // Define Case 900 wall construction (4 layers)
//! let layers = vec![
//!     MaterialLayer::new("Gypsum", 0.013, 0.16, 800.0, 1090.0),
//!     MaterialLayer::new("Concrete", 0.150, 1.4, 2300.0, 880.0),
//!     MaterialLayer::new("Insulation", 0.050, 0.04, 50.0, 840.0),
//!     MaterialLayer::new("Brick", 0.100, 0.81, 1920.0, 790.0),
//! ];
//!
//! // Discretize with 10 nodes per layer
//! let discretization = WallDiscretization::from_layers(&layers, 10);
//!
//! assert_eq!(discretization.total_nodes, 40); // 4 layers × 10 nodes
//! ```

use std::fmt;

/// A material layer with thermal properties.
#[derive(Debug, Clone)]
pub struct MaterialLayer {
    /// Layer name for identification.
    pub name: String,
    /// Layer thickness [m].
    pub thickness: f64,
    /// Thermal conductivity [W/m·K].
    pub conductivity: f64,
    /// Density [kg/m³].
    pub density: f64,
    /// Specific heat capacity [J/kg·K].
    pub specific_heat: f64,
}

impl MaterialLayer {
    /// Create a new material layer.
    ///
    /// # Arguments
    ///
    /// * `name` - Descriptive name for the layer
    /// * `thickness` - Layer thickness in meters
    /// * `conductivity` - Thermal conductivity in W/m·K
    /// * `density` - Density in kg/m³
    /// * `specific_heat` - Specific heat in J/kg·K
    pub fn new(
        name: &str,
        thickness: f64,
        conductivity: f64,
        density: f64,
        specific_heat: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            thickness,
            conductivity,
            density,
            specific_heat,
        }
    }

    /// Calculate thermal diffusivity α = k/(ρ·c_p) [m²/s].
    #[inline]
    pub fn diffusivity(&self) -> f64 {
        self.conductivity / (self.density * self.specific_heat)
    }

    /// Calculate thermal resistance R = L/k [m²·K/W].
    #[inline]
    pub fn resistance(&self) -> f64 {
        self.thickness / self.conductivity
    }

    /// Calculate volumetric heat capacity ρ·c_p [J/m³·K].
    #[inline]
    pub fn volumetric_heat_capacity(&self) -> f64 {
        self.density * self.specific_heat
    }
}

/// Interface conductivity between two adjacent nodes.
#[derive(Debug, Clone)]
pub struct InterfaceConductivity {
    /// Position from interior surface [m].
    pub position: f64,
    /// Effective conductivity at interface [W/m·K].
    pub value: f64,
}

/// Finite difference wall discretization.
///
/// Represents a 1D spatial grid through a multi-layer wall construction.
/// Each node has associated thermal properties and control volume.
///
/// # Fields
///
/// * `layers` - Original material layers (for reference)
/// * `nodes_per_layer` - Number of nodes in each layer
/// * `total_nodes` - Total number of temperature nodes
/// * `total_thickness` - Total wall thickness [m]
/// * `node_positions` - Distance from interior surface for each node [m]
/// * `node_volumes` - Control volume thickness for each node [m]
/// * `density` - Density at each node [kg/m³]
/// * `specific_heat` - Specific heat at each node [J/kg·K]
/// * `conductivity` - Conductivity at each node [W/m·K]
/// * `interface_conductivity` - Effective conductivity at node interfaces [W/m·K]
/// * `diffusivity` - Thermal diffusivity at each node [m²/s]
#[derive(Debug, Clone)]
pub struct WallDiscretization {
    /// Original material layers.
    pub layers: Vec<MaterialLayer>,
    /// Number of nodes per layer.
    pub nodes_per_layer: usize,
    /// Total number of temperature nodes.
    pub total_nodes: usize,
    /// Total wall thickness [m].
    pub total_thickness: f64,
    /// Node positions from interior surface [m].
    pub node_positions: Vec<f64>,
    /// Control volume thickness for each node [m].
    pub node_volumes: Vec<f64>,
    /// Density at each node [kg/m³].
    pub density: Vec<f64>,
    /// Specific heat at each node [J/kg·K].
    pub specific_heat: Vec<f64>,
    /// Conductivity at each node [W/m·K].
    pub conductivity: Vec<f64>,
    /// Interface conductivities [W/m·K] (length = total_nodes + 1).
    pub interface_conductivity: Vec<InterfaceConductivity>,
    /// Thermal diffusivity at each node [m²/s].
    pub diffusivity: Vec<f64>,
}

impl WallDiscretization {
    /// Create discretization from material layers.
    ///
    /// # Arguments
    ///
    /// * `layers` - Vector of material layers (interior to exterior)
    /// * `nodes_per_layer` - Number of nodes in each layer
    ///
    /// # Returns
    ///
    /// A `WallDiscretization` with uniform node spacing within each layer.
    ///
    /// # Example
    ///
    /// ```rust
    /// let layers = vec![
    ///     MaterialLayer::new("Gypsum", 0.013, 0.16, 800.0, 1090.0),
    ///     MaterialLayer::new("Concrete", 0.150, 1.4, 2300.0, 880.0),
    /// ];
    /// let disc = WallDiscretization::from_layers(&layers, 10);
    /// ```
    pub fn from_layers(layers: &[MaterialLayer], nodes_per_layer: usize) -> Self {
        assert!(!layers.is_empty(), "At least one layer required");
        assert!(nodes_per_layer >= 1, "At least 1 node per layer required");

        let total_nodes = layers.len() * nodes_per_layer;
        let total_thickness: f64 = layers.iter().map(|l| l.thickness).sum();

        let mut node_positions = Vec::with_capacity(total_nodes);
        let mut node_volumes = Vec::with_capacity(total_nodes);
        let mut density = Vec::with_capacity(total_nodes);
        let mut specific_heat = Vec::with_capacity(total_nodes);
        let mut conductivity = Vec::with_capacity(total_nodes);
        let mut diffusivity = Vec::with_capacity(total_nodes);

        // Build node properties layer by layer
        let mut current_position = 0.0;

        for layer in layers {
            let dx = layer.thickness / nodes_per_layer as f64;

            for i in 0..nodes_per_layer {
                // Node at center of control volume
                let node_pos = current_position + (i as f64 + 0.5) * dx;

                node_positions.push(node_pos);
                node_volumes.push(dx);
                density.push(layer.density);
                specific_heat.push(layer.specific_heat);
                conductivity.push(layer.conductivity);
                diffusivity.push(layer.diffusivity());
            }

            current_position += layer.thickness;
        }

        // Calculate interface conductivities (harmonic mean at material interfaces)
        let interface_conductivity = Self::calculate_interface_conductivities(
            layers,
            nodes_per_layer,
            &node_positions,
            &conductivity,
        );

        Self {
            layers: layers.to_vec(),
            nodes_per_layer,
            total_nodes,
            total_thickness,
            node_positions,
            node_volumes,
            density,
            specific_heat,
            conductivity,
            interface_conductivity,
            diffusivity,
        }
    }

    /// Calculate interface conductivities using harmonic mean.
    ///
    /// At material interfaces, the effective conductivity is the harmonic mean
    /// of adjacent layer conductivities to ensure flux continuity.
    fn calculate_interface_conductivities(
        layers: &[MaterialLayer],
        nodes_per_layer: usize,
        node_positions: &[f64],
        conductivity: &[f64],
    ) -> Vec<InterfaceConductivity> {
        let mut interfaces = Vec::with_capacity(node_positions.len() + 1);

        // Interior surface interface (half-node from surface)
        let dx_interior = node_positions[0]; // Distance to first node center
        interfaces.push(InterfaceConductivity {
            position: 0.0,
            value: conductivity[0], // Same material
        });

        // Internal interfaces
        for i in 0..node_positions.len() - 1 {
            let interface_pos = (node_positions[i] + node_positions[i + 1]) / 2.0;
            let k1 = conductivity[i];
            let k2 = conductivity[i + 1];

            // Harmonic mean for different materials, arithmetic for same
            let k_eff = if k1 != k2 {
                2.0 * k1 * k2 / (k1 + k2)
            } else {
                k1
            };

            interfaces.push(InterfaceConductivity {
                position: interface_pos,
                value: k_eff,
            });
        }

        // Exterior surface interface
        interfaces.push(InterfaceConductivity {
            position: node_positions[node_positions.len() - 1]
                + (node_positions[1] - node_positions[0]),
            value: conductivity[conductivity.len() - 1],
        });

        interfaces
    }

    /// Get node index for a given position.
    #[inline]
    pub fn node_at_position(&self, position: f64) -> Option<usize> {
        if position < 0.0 || position > self.total_thickness {
            return None;
        }

        // Find which layer
        let mut cumulative_thickness = 0.0;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if position < cumulative_thickness + layer.thickness {
                let relative_pos = position - cumulative_thickness;
                let node_in_layer =
                    (relative_pos / (layer.thickness / self.nodes_per_layer as f64)) as usize;
                return Some(
                    layer_idx * self.nodes_per_layer + node_in_layer.min(self.nodes_per_layer - 1),
                );
            }
            cumulative_thickness += layer.thickness;
        }

        Some(self.total_nodes - 1)
    }

    /// Get layer index for a given node.
    #[inline]
    pub fn layer_for_node(&self, node_idx: usize) -> Option<usize> {
        if node_idx >= self.total_nodes {
            return None;
        }
        Some(node_idx / self.nodes_per_layer)
    }

    /// Calculate thermal mass (heat capacity) of the wall [J/K·m²].
    pub fn thermal_mass(&self) -> f64 {
        let mut total = 0.0;
        for i in 0..self.total_nodes {
            total += self.density[i] * self.specific_heat[i] * self.node_volumes[i];
        }
        total
    }

    /// Calculate overall U-value [W/m²·K].
    pub fn u_value(&self) -> f64 {
        let total_resistance: f64 = self.layers.iter().map(|l| l.resistance()).sum();
        1.0 / total_resistance
    }

    /// Calculate time constant τ = R·C [s].
    pub fn time_constant(&self) -> f64 {
        let r_total: f64 = self.layers.iter().map(|l| l.resistance()).sum();
        let c_total = self.thermal_mass();
        r_total * c_total
    }
}

impl fmt::Display for WallDiscretization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Wall Discretization:")?;
        writeln!(f, "  Layers: {}", self.layers.len())?;
        writeln!(f, "  Nodes per layer: {}", self.nodes_per_layer)?;
        writeln!(f, "  Total nodes: {}", self.total_nodes)?;
        writeln!(f, "  Total thickness: {:.3} m", self.total_thickness)?;
        writeln!(f, "  U-value: {:.3} W/m²·K", self.u_value())?;
        writeln!(f, "  Thermal mass: {:.0} J/K·m²", self.thermal_mass())?;
        writeln!(
            f,
            "  Time constant: {:.0} s ({:.1} hr)",
            self.time_constant(),
            self.time_constant() / 3600.0
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create Case 900 high-mass wall construction.
    fn case_900_wall() -> Vec<MaterialLayer> {
        vec![
            MaterialLayer::new("Gypsum", 0.013, 0.16, 800.0, 1090.0),
            MaterialLayer::new("Concrete", 0.150, 1.4, 2300.0, 880.0),
            MaterialLayer::new("Insulation", 0.050, 0.04, 50.0, 840.0),
            MaterialLayer::new("Brick", 0.100, 0.81, 1920.0, 790.0),
        ]
    }

    #[test]
    fn test_case_900_discretization() {
        let layers = case_900_wall();
        let disc = WallDiscretization::from_layers(&layers, 10);

        assert_eq!(disc.total_nodes, 40); // 4 layers × 10 nodes
        assert_eq!(disc.nodes_per_layer, 10);

        // Total thickness: 13 + 150 + 50 + 100 = 313 mm
        assert!((disc.total_thickness - 0.313).abs() < 0.001);

        // U-value ≈ 1 / (0.013/0.16 + 0.150/1.4 + 0.050/0.04 + 0.100/0.81)
        //         ≈ 1 / (0.081 + 0.107 + 1.25 + 0.123) ≈ 0.64 W/m²·K
        let u_expected = 0.64;
        assert!((disc.u_value() - u_expected).abs() < 0.05);

        // Thermal mass check (concrete dominates)
        // C ≈ 2300 × 880 × 0.150 ≈ 304,000 J/K·m² (concrete only)
        assert!(disc.thermal_mass() > 300_000.0);
    }

    #[test]
    fn test_node_positions() {
        let layers = case_900_wall();
        let disc = WallDiscretization::from_layers(&layers, 10);

        // First node should be at half dx from interior surface
        let dx_gypsum = 0.013 / 10.0;
        assert!((disc.node_positions[0] - dx_gypsum / 2.0).abs() < 1e-6);

        // Last node should be at total_thickness - dx_brick/2
        let dx_brick = 0.100 / 10.0;
        let expected_last = disc.total_thickness - dx_brick / 2.0;
        assert!((disc.node_positions[disc.total_nodes - 1] - expected_last).abs() < 1e-6);
    }

    #[test]
    fn test_layer_for_node() {
        let layers = case_900_wall();
        let disc = WallDiscretization::from_layers(&layers, 10);

        // Nodes 0-9: Gypsum (layer 0)
        for i in 0..10 {
            assert_eq!(disc.layer_for_node(i), Some(0));
        }

        // Nodes 10-19: Concrete (layer 1)
        for i in 10..20 {
            assert_eq!(disc.layer_for_node(i), Some(1));
        }

        // Nodes 20-29: Insulation (layer 2)
        for i in 20..30 {
            assert_eq!(disc.layer_for_node(i), Some(2));
        }

        // Nodes 30-39: Brick (layer 3)
        for i in 30..40 {
            assert_eq!(disc.layer_for_node(i), Some(3));
        }
    }

    #[test]
    fn test_interface_conductivities() {
        let layers = case_900_wall();
        let disc = WallDiscretization::from_layers(&layers, 10);

        // Should have total_nodes + 1 interfaces
        assert_eq!(disc.interface_conductivity.len(), disc.total_nodes + 1);

        // Interior surface: gypsum conductivity
        assert!((disc.interface_conductivity[0].value - 0.16).abs() < 0.01);

        // Exterior surface: brick conductivity
        let last_idx = disc.interface_conductivity.len() - 1;
        assert!((disc.interface_conductivity[last_idx].value - 0.81).abs() < 0.01);
    }

    #[test]
    fn test_thermal_mass_accuracy() {
        // Test with single homogeneous layer
        let layers = vec![MaterialLayer::new("Concrete", 0.200, 1.4, 2300.0, 880.0)];

        for n_nodes in [5, 10, 20, 40] {
            let disc = WallDiscretization::from_layers(&layers, n_nodes);

            // Analytical: C = ρ·c_p·L = 2300 × 880 × 0.200 = 404,800 J/K·m²
            let c_analytical = 2300.0 * 880.0 * 0.200;

            assert!((disc.thermal_mass() - c_analytical).abs() < 1.0);
        }
    }

    #[test]
    fn test_diffusivity() {
        let layers = case_900_wall();
        let disc = WallDiscretization::from_layers(&layers, 10);

        // Concrete diffusivity: α = k/(ρ·c_p) = 1.4/(2300×880) ≈ 6.9e-7 m²/s
        let alpha_concrete = 1.4 / (2300.0 * 880.0);

        // Check concrete nodes (10-19)
        for i in 10..20 {
            assert!((disc.diffusivity[i] - alpha_concrete).abs() < 1e-9);
        }
    }
}
