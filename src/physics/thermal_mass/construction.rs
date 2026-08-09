//! Construction-type physics module for thermal mass analysis.
//!
//! This module provides construction type definitions and thermal mass properties
//! for different building construction categories (lightweight, medium-weight, heavy-weight).

use crate::physics::constants::thermal::iso_13790::annex_c::{
    calculate_effective_thermal_mass, THERMAL_MASS_HEAVY, THERMAL_MASS_HEAVY_UPPER,
    THERMAL_MASS_LIGHT, THERMAL_MASS_LIGHT_UPPER, THERMAL_MASS_MEDIUM, THERMAL_MASS_MEDIUM_UPPER,
    THERMAL_MASS_VERY_HEAVY, THERMAL_MASS_VERY_LIGHT,
};
// Issue #2462 (Phase 2 of the crate split): `ConstructionLayer` now lives in
// `fluxion_core::construction` (the workspace leaf crate). Importing it via
// the leaf crate directly breaks the `physics ↔ sim` module cycle — see
// ARCHITECTURE.md §"Remaining cycles" and `docs/mutation_testing_crate_split.md`
// §"Phase 2". The `crate::sim::construction::ConstructionLayer` path stays
// valid as a re-export shim, but the leaf-crate import is preferred for new
// code (and required to keep the cycle guard in
// `scripts/check_physics_sim_cycle.py` reporting 0 edges).
use fluxion_core::construction::ConstructionLayer;
use serde::{Deserialize, Serialize};

/// Construction type enum defining standard building construction categories.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstructionType {
    /// Lightweight construction (wood frame, metal cladding)
    /// Typical effective capacitance: 50 kJ/m²K
    /// Typical time constant: 2 hours
    Lightweight,

    /// Medium-weight construction (lightweight concrete, brick veneer)
    /// Typical effective capacitance: 150 kJ/m²K
    /// Typical time constant: 6 hours
    MediumWeight,

    /// Heavy-weight construction (concrete, masonry)
    /// Typical effective capacitance: 300 kJ/m²K
    /// Typical time constant: 12 hours
    HeavyWeight,

    /// Custom construction with user-defined material layers
    /// Thermal properties calculated from actual material specifications
    Custom(Vec<MaterialLayer>),
}

impl Default for ConstructionType {
    fn default() -> Self {
        ConstructionType::MediumWeight
    }
}

/// Material layer definition for custom construction types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialLayer {
    /// Layer name
    pub name: String,
    /// Thermal conductivity of the material (k) in W/m·K
    pub conductivity: f64,
    /// Density of the material in kg/m³
    pub density: f64,
    /// Specific heat capacity of the material in J/kg·K
    pub specific_heat: f64,
    /// Layer thickness in meters
    pub thickness: f64,
    /// Surface emissivity (0.0 to 1.0)
    pub emissivity: f64,
    /// Surface absorptance (0.0 to 1.0)
    pub absorptance: f64,
}

impl MaterialLayer {
    /// Create a new material layer.
    ///
    /// # Arguments
    /// * `name` - Layer name
    /// * `conductivity` - Thermal conductivity (k) in W/m·K
    /// * `density` - Material density in kg/m³
    /// * `specific_heat` - Specific heat capacity in J/kg·K
    /// * `thickness` - Layer thickness in meters
    ///
    /// # Returns
    /// A new MaterialLayer instance
    pub fn new(
        name: impl Into<String>,
        conductivity: f64,
        density: f64,
        specific_heat: f64,
        thickness: f64,
    ) -> Self {
        Self {
            name: name.into(),
            conductivity,
            density,
            specific_heat,
            thickness,
            emissivity: 0.9,
            absorptance: 0.7,
        }
    }

    /// Create a new material layer with custom surface properties.
    ///
    /// # Arguments
    /// * `name` - Layer name
    /// * `conductivity` - Thermal conductivity (k) in W/m·K
    /// * `density` - Material density in kg/m³
    /// * `specific_heat` - Specific heat capacity in J/kg·K
    /// * `thickness` - Layer thickness in meters
    /// * `emissivity` - Surface emissivity (0.0 to 1.0)
    /// * `absorptance` - Surface absorptance (0.0 to 1.0)
    ///
    /// # Returns
    /// A new MaterialLayer instance with custom surface properties
    pub fn with_surface_properties(
        name: impl Into<String>,
        conductivity: f64,
        density: f64,
        specific_heat: f64,
        thickness: f64,
        emissivity: f64,
        absorptance: f64,
    ) -> Self {
        Self {
            name: name.into(),
            conductivity,
            density,
            specific_heat,
            thickness,
            emissivity,
            absorptance,
        }
    }
}

/// Thermal mass properties calculated from construction type.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThermalMassProperties {
    /// Effective thermal capacitance in kJ/m²K
    pub effective_capacitance: f64,
    /// Time constant in hours
    pub time_constant: f64,
    /// Damping factor (unitless, 0.0-1.0)
    pub damping_factor: f64,
}

impl Default for ThermalMassProperties {
    fn default() -> Self {
        Self {
            effective_capacitance: 150.0, // Medium-weight default
            time_constant: 6.0,
            damping_factor: 0.5,
        }
    }
}

impl ConstructionType {
    /// Get the typical thermal mass properties for this construction type.
    ///
    /// # Returns
    /// ThermalMassProperties with typical values for the construction type
    pub fn thermal_mass_properties(&self) -> ThermalMassProperties {
        match self {
            ConstructionType::Lightweight => ThermalMassProperties {
                effective_capacitance: 50.0,
                time_constant: 2.0,
                damping_factor: 0.3,
            },
            ConstructionType::MediumWeight => ThermalMassProperties {
                effective_capacitance: 150.0,
                time_constant: 6.0,
                damping_factor: 0.5,
            },
            ConstructionType::HeavyWeight => ThermalMassProperties {
                effective_capacitance: 300.0,
                time_constant: 12.0,
                damping_factor: 0.7,
            },
            ConstructionType::Custom(layers) => self.calculate_custom_properties(layers),
        }
    }

    /// Calculate thermal mass properties for custom construction.
    ///
    /// # Arguments
    /// * `layers` - Vector of material layers
    ///
    /// # Returns
    /// ThermalMassProperties calculated from the material layers
    pub fn calculate_custom_properties(&self, layers: &[MaterialLayer]) -> ThermalMassProperties {
        if layers.is_empty() {
            return ThermalMassProperties::default();
        }

        // Calculate effective thermal mass using ISO 13790 Annex C formula
        let layer_tuples: Vec<(f64, f64, f64)> = layers
            .iter()
            .map(|layer| (layer.thickness, layer.density, layer.specific_heat))
            .collect();

        let effective_capacitance = calculate_effective_thermal_mass(&layer_tuples);

        // For custom constructions, we need a reference heat loss coefficient
        // Use typical value for residential buildings: 10 W/m²K
        let typical_heat_loss_coefficient = 10.0; // W/m²K

        // Calculate time constant: τ = C_eff / H (convert kJ to J)
        let capacitance_joules = effective_capacitance * 1000.0;
        let time_constant_seconds = capacitance_joules / typical_heat_loss_coefficient;
        let time_constant_hours = time_constant_seconds / 3600.0;

        // Calculate damping factor for 1-hour timestep
        let timestep_hours = 1.0; // 1 hour
        let damping_factor = (-timestep_hours / time_constant_hours).exp();

        ThermalMassProperties {
            effective_capacitance,
            time_constant: time_constant_hours,
            damping_factor,
        }
    }

    /// Get the ISO 13790 Annex C classification for this construction type.
    ///
    /// # Returns
    /// Classification string according to ISO 13790 Annex C
    pub fn classification(&self) -> String {
        match self {
            ConstructionType::Lightweight => "Light".to_string(),
            ConstructionType::MediumWeight => "Medium".to_string(),
            ConstructionType::HeavyWeight => "Heavy".to_string(),
            ConstructionType::Custom(layers) => {
                let props = self.calculate_custom_properties(layers);
                self.classify_by_capacitance(props.effective_capacitance)
            }
        }
    }

    /// Classify construction by effective capacitance value.
    ///
    /// # Arguments
    /// * `capacitance` - Effective thermal capacitance in kJ/m²K
    ///
    /// # Returns
    /// Classification string according to ISO 13790 Annex C
    pub fn classify_by_capacitance(&self, capacitance: f64) -> String {
        if capacitance < THERMAL_MASS_VERY_LIGHT {
            "VeryLight".to_string()
        } else if capacitance < THERMAL_MASS_MEDIUM {
            "Light".to_string()
        } else if capacitance < THERMAL_MASS_HEAVY {
            "Medium".to_string()
        } else if capacitance < THERMAL_MASS_VERY_HEAVY {
            "Heavy".to_string()
        } else {
            "VeryHeavy".to_string()
        }
    }

    /// Check if this construction type is suitable for high-mass validation.
    ///
    /// # Returns
    /// true if construction meets high-mass criteria (Heavy or VeryHeavy)
    pub fn is_high_mass(&self) -> bool {
        match self {
            ConstructionType::HeavyWeight => true,
            ConstructionType::Custom(layers) => {
                let props = self.calculate_custom_properties(layers);
                props.effective_capacitance >= THERMAL_MASS_HEAVY
            }
            _ => false,
        }
    }

    /// Get typical material layers for this construction type.
    ///
    /// # Returns
    /// Vector of MaterialLayer representing typical construction
    pub fn typical_layers(&self) -> Vec<MaterialLayer> {
        match self {
            ConstructionType::Lightweight => {
                // Typical lightweight construction: wood frame with insulation
                vec![
                    MaterialLayer::new(
                        "Fiberglass Insulation",
                        0.04,  // W/mK
                        50.0,  // kg/m³
                        840.0, // J/kgK
                        0.1,   // 10cm
                    ),
                    MaterialLayer::new(
                        "Wood Frame",
                        0.12,   // W/mK
                        600.0,  // kg/m³
                        1200.0, // J/kgK
                        0.05,   // 5cm
                    ),
                    MaterialLayer::new(
                        "Plasterboard",
                        0.25,   // W/mK
                        800.0,  // kg/m³
                        1000.0, // J/kgK
                        0.015,  // 1.5cm
                    ),
                ]
            }
            ConstructionType::MediumWeight => {
                // Typical medium-weight construction: brick veneer
                vec![
                    MaterialLayer::new(
                        "Brick Veneer",
                        0.8,    // W/mK
                        1800.0, // kg/m³
                        840.0,  // J/kgK
                        0.1,    // 10cm
                    ),
                    MaterialLayer::new(
                        "Insulation",
                        0.04,  // W/mK
                        50.0,  // kg/m³
                        840.0, // J/kgK
                        0.08,  // 8cm
                    ),
                    MaterialLayer::new(
                        "Wood Frame",
                        0.12,   // W/mK
                        600.0,  // kg/m³
                        1200.0, // J/kgK
                        0.05,   // 5cm
                    ),
                ]
            }
            ConstructionType::HeavyWeight => {
                // Typical heavy-weight construction: concrete
                vec![
                    MaterialLayer::new(
                        "Concrete", 1.7,    // W/mK
                        2300.0, // kg/m³
                        840.0,  // J/kgK
                        0.2,    // 20cm concrete
                    ),
                    MaterialLayer::new(
                        "Insulation",
                        0.04,  // W/mK
                        50.0,  // kg/m³
                        840.0, // J/kgK
                        0.05,  // 5cm insulation (exterior)
                    ),
                ]
            }
            ConstructionType::Custom(layers) => layers.clone(),
        }
    }

    /// Convert from ConstructionLayer vector to ConstructionType.
    ///
    /// # Arguments
    /// * `layers` - Construction layers to convert
    ///
    /// # Returns
    /// ConstructionType enum variant
    pub fn from_construction_layers(layers: &[ConstructionLayer]) -> Self {
        // Calculate effective thermal mass from construction layers
        let layer_tuples: Vec<(f64, f64, f64)> = layers
            .iter()
            .map(|layer| (layer.thickness, layer.density, layer.specific_heat))
            .collect();

        let effective_capacitance = calculate_effective_thermal_mass(&layer_tuples);

        // Classify based on capacitance
        if effective_capacitance < THERMAL_MASS_LIGHT_UPPER {
            ConstructionType::Lightweight
        } else if effective_capacitance < THERMAL_MASS_MEDIUM_UPPER {
            ConstructionType::MediumWeight
        } else {
            ConstructionType::HeavyWeight
        }
    }
}
