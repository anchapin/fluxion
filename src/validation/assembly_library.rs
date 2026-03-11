use crate::sim::construction::{Construction, ConstructionLayer};
use serde::Deserialize;
use serde_yaml;
use std::collections::HashMap;
use std::path::Path;

/// Assembly definition from YAML.
#[derive(Debug, Clone, Deserialize)]
pub struct AssemblyYAML {
    /// Optional description or type metadata (not used in construction)
    #[serde(default)]
    pub construction_type: Option<String>,
    /// Ordered list of material layers from interior to exterior.
    pub layers: Vec<LayerYAML>,
}

/// Layer specification from YAML.
#[derive(Debug, Clone, Deserialize)]
pub struct LayerYAML {
    /// Material name (for reference)
    pub material: String,
    /// Thickness of the layer in meters
    pub thickness: f64,
    /// Density of the material in kg/m³
    pub density: f64,
    /// Specific heat capacity in J/kg·K
    pub specific_heat: f64,
    /// Thermal conductivity in W/m·K
    pub conductivity: f64,
    /// Optional emissivity (0-1), defaults to 0.9
    #[serde(default)]
    pub emissivity: Option<f64>,
    /// Optional absorptance (0-1), defaults to 0.7
    #[serde(default)]
    pub absorptance: Option<f64>,
}

/// Library of reusable construction assemblies.
#[derive(Debug, Clone)]
/// Collection of named construction assemblies.
pub struct AssemblyLibrary {
    assemblies: HashMap<String, Construction>,
}

impl AssemblyLibrary {
    /// Load assembly definitions from a YAML file.
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let map: HashMap<String, AssemblyYAML> = serde_yaml::from_str(&content)?;
        let mut assemblies = HashMap::new();
        for (name, assembly) in map {
            let construction = create_construction_from_assembly(&assembly)?;
            assemblies.insert(name, construction);
        }
        Ok(AssemblyLibrary { assemblies })
    }

    /// Get a construction by name.
    pub fn get(&self, name: &str) -> Option<&Construction> {
        self.assemblies.get(name)
    }

    /// List all available assembly names.
    pub fn list(&self) -> Vec<&str> {
        self.assemblies.keys().map(|s| s.as_str()).collect()
    }
}

/// Create a Construction from an assembly definition.
fn create_construction_from_assembly(
    assembly: &AssemblyYAML,
) -> Result<Construction, Box<dyn std::error::Error>> {
    let mut layers = Vec::new();
    for layer in &assembly.layers {
        // Use provided emissivity/absorptance or defaults
        let emissivity = layer.emissivity.unwrap_or(0.9);
        let absorptance = layer.absorptance.unwrap_or(0.7);
        // Validate emissivity and absorptance ranges
        if !(0.0..=1.0).contains(&emissivity) {
            return Err(format!("Emissivity must be in [0,1], got {}", emissivity).into());
        }
        if !(0.0..=1.0).contains(&absorptance) {
            return Err(format!("Absorptance must be in [0,1], got {}", absorptance).into());
        }
        // Build layer with surface properties
        let clayer = ConstructionLayer::with_surface_properties(
            &layer.material,
            layer.conductivity,
            layer.density,
            layer.specific_heat,
            layer.thickness,
            emissivity,
            absorptance,
        );
        layers.push(clayer);
    }
    Ok(Construction::new(layers))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_assembly_library_load() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = Path::new(manifest_dir).join("config/assemblies.yaml");
        if !path.exists() {
            eprintln!("Skipping test: assemblies.yaml not found at {:?}", path);
            return;
        }
        let lib = AssemblyLibrary::from_file(&path).expect("Failed to load assemblies");
        assert!(!lib.list().is_empty());
        // Try to get a known assembly if exists
        let all_names = lib.list();
        if let Some(&first) = all_names.first() {
            let construction = lib
                .get(first)
                .expect(&format!("Assembly '{}' should be retrievable", first));
            // Validate construction has at least one layer
            assert!(!construction.layers.is_empty());
        }
    }

    #[test]
    fn test_construction_r_value() {
        // Verify that a simple construction yields expected R-value
        let assembly_yaml = AssemblyYAML {
            construction_type: Some("test".to_string()),
            layers: vec![LayerYAML {
                material: "Insulation".to_string(),
                thickness: 0.1,
                density: 50.0,
                specific_heat: 840.0,
                conductivity: 0.04,
                emissivity: None,
                absorptance: None,
            }],
        };
        let construction = create_construction_from_assembly(&assembly_yaml).unwrap();
        let r = construction.r_value_materials();
        assert!((r - 2.5).abs() < 0.01); // 0.1 / 0.04 = 2.5
    }

    #[test]
    fn test_missing_layer_fields() {
        // Test that creating with valid fields works
        let assembly_yaml = AssemblyYAML {
            construction_type: None,
            layers: vec![LayerYAML {
                material: "Test".to_string(),
                thickness: 0.05,
                density: 100.0,
                specific_heat: 1000.0,
                conductivity: 0.1,
                emissivity: None,
                absorptance: None,
            }],
        };
        let result = create_construction_from_assembly(&assembly_yaml);
        assert!(result.is_ok());
        let construction = result.unwrap();
        assert_eq!(construction.layers.len(), 1);
    }
}
