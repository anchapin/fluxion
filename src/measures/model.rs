// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! `FluxionModel` — the in-memory building model that the Hybrid Measure
//! Approach mutates via JSON Patch (RFC 6902).
//!
//! # Why a new model type?
//!
//! The existing [`crate::api::schema::SimulationSchemaV1`] couples the model
//! shape to a schema-versioned envelope (versions, totals, embedded weather)
//! that's optimised for the Python API surface. For the declarative-Delta
//! use case (Issue #1811, Phase 1 of the Hybrid Measure Approach) we want:
//!
//! - **Stable, predictable JSON paths** so a Delta like
//!   `/zones/zone_1/volume` is unambiguous.
//! - **Deterministic serialisation** so a patch applied to a serialized→
//!   deserialized model yields byte-identical JSON (round-trip safety).
//! - **No trait-object fields** (everything is `Serialize + Deserialize`).
//! - **A `Default`-able, ASHRAE-140-shaped baseline** that unit tests can
//!   instantiate without an IDF/E+ import.
//!
//! # Module location
//!
//! `FluxionModel` lives under [`crate::measures`] (not `fluxion-core`) on
//! purpose: the cycle-breaking rule (see `AGENTS.md`) forbids `fluxion-core`
//! from importing anything from `sim/`, `physics/`, `ai/`, or `validation/`.
//! As more Delta-related types land here (M2, M4, M6), the entire measures
//! sub-system stays self-contained in the main crate.
//!
//! # JSON shape
//!
//! The default ASHRAE 140 Case 600 / Case 900 model serialises as:
//!
//! ```json
//! {
//!   "schema_version": "fluxion-measures-v1",
//!   "zones": {
//!     "zone_1": { "name": "Zone 1", "volume": 129.6, ... }
//!   },
//!   "constructions": {
//!     "wall": { "name": "Mass wall", "layers": [...] }
//!   },
//!   "assemblies": {
//!     "wall_1": { "name": "Wall 1", "layers": [...] }
//!   }
//! }
//! ```
//!
//! Pointer paths for common Delta operations:
//!
//! - `/zones/<zone_key>/<field>` — zone metadata
//! - `/constructions/<const_key>/layers/<idx>/<field>` — construction layers
//! - `/assemblies/<asm_key>/layers/<idx>/<field>` — assembly layers (R-value = `thickness / conductivity`)
//!
//! # Example
//!
//! ```
//! use fluxion::measures::model::FluxionModel;
//!
//! let model = FluxionModel::ashrae_140_case_600();
//! let json = serde_json::to_string_pretty(&model).unwrap();
//! let parsed: FluxionModel = serde_json::from_str(&json).unwrap();
//! assert_eq!(model, parsed);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Marker for the measures schema format.
///
/// Bumping this string is a breaking change for any persisted Delta / model
/// JSON. Add the new format alongside, then migrate.
pub const MEASURES_SCHEMA_VERSION: &str = "fluxion-measures-v1";

/// A single material layer in an assembly or construction.
///
/// `R_value = thickness / conductivity`. Increasing the R-value by 20%
/// is equivalent to either multiplying `thickness` by 1.2 (with the same
/// `conductivity`) or dividing `conductivity` by 1.2 (with the same
/// `thickness`). The acceptance-criteria unit test exercises the
/// conductivity-divide path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialLayer {
    /// Material name (e.g. "Insulation", "Concrete", "Gypsum").
    pub name: String,
    /// Thermal conductivity (W/m·K). Must be positive.
    pub conductivity: f64,
    /// Density (kg/m³). Must be positive.
    pub density: f64,
    /// Specific heat capacity (J/kg·K). Must be positive.
    pub specific_heat: f64,
    /// Thickness (m). Must be positive.
    pub thickness: f64,
    /// Surface emissivity (0.0–1.0). Defaults to 0.9.
    #[serde(default = "default_emissivity")]
    pub emissivity: f64,
    /// Solar absorptance (0.0–1.0). Defaults to 0.7.
    #[serde(default = "default_absorptance")]
    pub absorptance: f64,
}

fn default_emissivity() -> f64 {
    0.9
}
fn default_absorptance() -> f64 {
    0.7
}

impl MaterialLayer {
    /// R-value of this layer (m²·K/W).
    pub fn r_value(&self) -> f64 {
        self.thickness / self.conductivity
    }
}

/// A zone in the model.
///
/// Zones are keyed in [`FluxionModel::zones`] by their `key` field
/// (a stable identifier like `"zone_1"`), which determines their JSON
/// path. The `name` field is a human-readable label.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoneSpec {
    /// Human-readable name (e.g. "Zone 1", "Living Room").
    pub name: String,
    /// Zone floor area (m²). Positive.
    pub floor_area: f64,
    /// Zone volume (m³). Positive. The example path `/zones/zone_1/volume`
    /// in Issue #1811 targets this field.
    pub volume: f64,
    /// Floor-to-ceiling height (m).
    pub height: f64,
}

/// A construction (e.g. a wall type, roof type, floor type).
///
/// `layers` are listed exterior → interior. The JSON path to a specific
/// layer is `/constructions/<const_key>/layers/<index>/<field>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstructionSpec {
    /// Human-readable name (e.g. "Mass wall", "Wood frame roof").
    pub name: String,
    /// Material layers exterior → interior.
    pub layers: Vec<MaterialLayer>,
}

/// An assembly — a named bundle of layers that a surface references.
///
/// Conceptually similar to a construction, but assemblies are the unit
/// the Issue #1811 unit test mutates ("+20% insulation R-value"). An
/// assembly's R-value is the sum of its layer R-values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssemblySpec {
    /// Human-readable name (e.g. "Wall 1", "Roof 1").
    pub name: String,
    /// Material layers.
    pub layers: Vec<MaterialLayer>,
}

impl AssemblySpec {
    /// Total R-value of the assembly (m²·K/W).
    pub fn total_r_value(&self) -> f64 {
        self.layers.iter().map(MaterialLayer::r_value).sum()
    }
}

/// The in-memory building model that Deltas mutate.
///
/// Field types are deliberately concrete (`Vec<MaterialLayer>` instead of
/// `Vec<Box<dyn MaterialLayer>>`) so the model round-trips through serde
/// without trait-object shenanigans. `BTreeMap` is used for `zones`,
/// `constructions`, and `assemblies` so the serialised JSON has stable
/// key ordering (required for byte-identical round-trips).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FluxionModel {
    /// Schema version marker. Always `"fluxion-measures-v1"` for this
    /// revision; bumping the schema is a breaking change.
    #[serde(default = "default_schema_version")]
    pub schema_version: String,

    /// Zones keyed by stable identifier (e.g. `"zone_1"`).
    #[serde(default)]
    pub zones: BTreeMap<String, ZoneSpec>,

    /// Constructions keyed by stable identifier (e.g. `"wall"`, `"roof"`).
    #[serde(default)]
    pub constructions: BTreeMap<String, ConstructionSpec>,

    /// Assemblies keyed by stable identifier (e.g. `"wall_1"`).
    #[serde(default)]
    pub assemblies: BTreeMap<String, AssemblySpec>,
}

fn default_schema_version() -> String {
    MEASURES_SCHEMA_VERSION.to_string()
}

impl Default for FluxionModel {
    fn default() -> Self {
        // A minimal but valid empty model — useful for tests that don't
        // need ASHRAE 140 geometry. Real use cases should call
        // `FluxionModel::ashrae_140_case_600()` or `::ashrae_140_case_900()`.
        Self {
            schema_version: MEASURES_SCHEMA_VERSION.to_string(),
            zones: BTreeMap::new(),
            constructions: BTreeMap::new(),
            assemblies: BTreeMap::new(),
        }
    }
}

impl FluxionModel {
    /// Construct a FluxionModel from ASHRAE 140 Case 600 geometry/materials.
    ///
    /// Case 600 is the low-mass reference case: lightweight insulation,
    /// 48 m² floor, 129.6 m³ volume. Properties are pinned to the
    /// published ASHRAE 140 Table B1-1 values.
    pub fn ashrae_140_case_600() -> Self {
        let mut zones = BTreeMap::new();
        zones.insert(
            "zone_1".to_string(),
            ZoneSpec {
                name: "Zone 1".to_string(),
                floor_area: 48.0,
                volume: 129.6,
                height: 2.7,
            },
        );

        let mut constructions = BTreeMap::new();
        constructions.insert(
            "wall".to_string(),
            ConstructionSpec {
                name: "Mass wall".to_string(),
                // Mass wall — wood siding + insulation + gypsum
                // (Case 600 uses an exterior insulation finish system;
                // values per ASHRAE 140 Table B1-1).
                layers: vec![
                    MaterialLayer {
                        name: "Wood siding".to_string(),
                        conductivity: 0.14,
                        density: 500.0,
                        specific_heat: 1300.0,
                        thickness: 0.009,
                        emissivity: 0.9,
                        absorptance: 0.7,
                    },
                    MaterialLayer {
                        name: "Insulation".to_string(),
                        conductivity: 0.04,
                        density: 10.0,
                        specific_heat: 1400.0,
                        thickness: 0.066,
                        emissivity: 0.9,
                        absorptance: 0.5,
                    },
                    MaterialLayer {
                        name: "Gypsum".to_string(),
                        conductivity: 0.16,
                        density: 800.0,
                        specific_heat: 840.0,
                        thickness: 0.012,
                        emissivity: 0.9,
                        absorptance: 0.7,
                    },
                ],
            },
        );

        let mut assemblies = BTreeMap::new();
        assemblies.insert(
            "wall_1".to_string(),
            AssemblySpec {
                name: "Wall 1".to_string(),
                layers: vec![
                    MaterialLayer {
                        name: "Insulation".to_string(),
                        conductivity: 0.04,
                        density: 10.0,
                        specific_heat: 1400.0,
                        thickness: 0.066,
                        emissivity: 0.9,
                        absorptance: 0.5,
                    },
                    MaterialLayer {
                        name: "Gypsum".to_string(),
                        conductivity: 0.16,
                        density: 800.0,
                        specific_heat: 840.0,
                        thickness: 0.012,
                        emissivity: 0.9,
                        absorptance: 0.7,
                    },
                ],
            },
        );

        Self {
            schema_version: MEASURES_SCHEMA_VERSION.to_string(),
            zones,
            constructions,
            assemblies,
        }
    }

    /// Construct a FluxionModel from ASHRAE 140 Case 900 geometry/materials.
    ///
    /// Case 900 is the high-mass reference case: heavyweight concrete +
    /// foam-board insulation. Properties are pinned to the published
    /// ASHRAE 140 Table B1-3 values.
    pub fn ashrae_140_case_900() -> Self {
        let mut zones = BTreeMap::new();
        zones.insert(
            "zone_1".to_string(),
            ZoneSpec {
                name: "Zone 1".to_string(),
                floor_area: 48.0,
                volume: 129.6,
                height: 2.7,
            },
        );

        let mut assemblies = BTreeMap::new();
        assemblies.insert(
            "wall_1".to_string(),
            AssemblySpec {
                name: "Wall 1".to_string(),
                layers: vec![
                    MaterialLayer {
                        // HW concrete per ASHRAE 140 Table B1-3.
                        name: "Concrete".to_string(),
                        conductivity: 1.4,
                        density: 2240.0,
                        specific_heat: 900.0,
                        thickness: 0.200,
                        emissivity: 0.9,
                        absorptance: 0.7,
                    },
                    MaterialLayer {
                        // Foam-board insulation per ASHRAE 140 Table B1-3.
                        name: "Insulation".to_string(),
                        conductivity: 0.04,
                        density: 10.0,
                        specific_heat: 1400.0,
                        thickness: 0.0615,
                        emissivity: 0.9,
                        absorptance: 0.5,
                    },
                ],
            },
        );

        Self {
            schema_version: MEASURES_SCHEMA_VERSION.to_string(),
            zones,
            constructions: BTreeMap::new(),
            assemblies,
        }
    }

    /// Look up a layer's R-value by `(assembly_key, layer_index)`.
    ///
    /// Returns `None` if the assembly or layer doesn't exist.
    pub fn assembly_layer_r_value(&self, assembly: &str, layer_index: usize) -> Option<f64> {
        self.assemblies
            .get(assembly)
            .and_then(|asm| asm.layers.get(layer_index))
            .map(MaterialLayer::r_value)
    }

    /// Total R-value of an assembly (sum of layer R-values).
    pub fn assembly_total_r_value(&self, assembly: &str) -> Option<f64> {
        self.assemblies
            .get(assembly)
            .map(AssemblySpec::total_r_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_empty_but_valid() {
        let model = FluxionModel::default();
        assert_eq!(model.schema_version, MEASURES_SCHEMA_VERSION);
        assert!(model.zones.is_empty());
        assert!(model.constructions.is_empty());
        assert!(model.assemblies.is_empty());
    }

    #[test]
    fn case_600_has_expected_zone_volume() {
        let model = FluxionModel::ashrae_140_case_600();
        let zone = model.zones.get("zone_1").expect("zone_1 must exist");
        assert!((zone.volume - 129.6).abs() < 1e-9);
        assert!((zone.floor_area - 48.0).abs() < 1e-9);
    }

    #[test]
    fn case_900_assembly_has_two_layers() {
        let model = FluxionModel::ashrae_140_case_900();
        let asm = model.assemblies.get("wall_1").expect("wall_1 must exist");
        assert_eq!(asm.layers.len(), 2);
        assert_eq!(asm.layers[0].name, "Concrete");
        assert_eq!(asm.layers[1].name, "Insulation");
    }

    #[test]
    fn r_value_is_thickness_over_conductivity() {
        let layer = MaterialLayer {
            name: "test".to_string(),
            conductivity: 0.04,
            density: 10.0,
            specific_heat: 1400.0,
            thickness: 0.066,
            emissivity: 0.9,
            absorptance: 0.5,
        };
        // 0.066 / 0.04 = 1.65 m²K/W
        assert!((layer.r_value() - 1.65).abs() < 1e-9);
    }

    #[test]
    fn round_trip_preserves_data() {
        let model = FluxionModel::ashrae_140_case_600();
        let json = serde_json::to_string(&model).expect("serialize");
        let parsed: FluxionModel = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(model, parsed);
    }

    #[test]
    fn case_900_insulation_layer_r_value_is_positive() {
        let model = FluxionModel::ashrae_140_case_900();
        // Foam board: 0.0615 / 0.04 = 1.5375 m²K/W
        let r = model
            .assembly_layer_r_value("wall_1", 1)
            .expect("layer 1 must exist");
        assert!((r - 1.5375).abs() < 1e-9);
        assert!(r > 0.0);
    }

    #[test]
    fn missing_assembly_returns_none() {
        let model = FluxionModel::default();
        assert!(model.assembly_total_r_value("nonexistent").is_none());
    }
}
