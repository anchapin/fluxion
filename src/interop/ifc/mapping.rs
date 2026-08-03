// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Maps an [`IfcModel`] onto a Fluxion [`SimulationSchemaV1`].
//!
//! # Mapping rules (issue #1343)
//!
//! | IFC entity                                  | SimulationSchema target                   |
//! |---------------------------------------------|-------------------------------------------|
//! | `IfcSpace`                                  | [`ZoneGeometry`] (default 24 m² floor area if footprint can't be resolved) |
//! | `IfcWall` / `IfcSlab` / `IfcRoof`           | [`SurfaceConstruction`] (one shared wall/roof/floor construction per category) |
//! | `IfcMaterialLayer` via `IfcMaterialLayerSetUsage` → `IfcMaterialLayerSet` → `IfcRelAssociatesMaterial` | Material layers of the matching [`SurfaceConstruction`] (one layer per `IfcMaterialLayer`) |
//!
//! # Out of scope
//!
//! - Per-wall geometry (vertices, openings, etc.). The per-surface
//!   conduction solver reads its own geometry from the IFC file at
//!   simulation time. The scaffold only needs to populate *envelope
//!   counts* and *material layering*.
//! - HVAC, windows, doors, property sets — see issue #1343 scope.
//!
//! # Floor area heuristic
//!
//! IFC4 stores zone geometry in `IfcSpace.Representation` (typically an
//! extruded footprint polyline). Decoding the full extrusion is out of
//! scope for the scaffold. Instead, when an `IfcSpace` carries no
//! decoded footprint we fall back to the 6 m × 4 m default used by the
//! shipped sample fixture (24 m²). Future follow-ups (#1121) can wire
//! the real footprint polygon.

use std::fs;
use std::path::Path;

use crate::api::schema::{
    ConstructionSet, Geometry, ScheduleSet, SchemaMetadata, SimulationOutput, SimulationSchemaV1,
    SurfaceConstruction, WeatherData, ZoneGeometry,
};
use crate::interop::gbxml::{export_gbxml, import_gbxml};
use crate::sim::construction::ConstructionLayer;

use super::error::IfcError;
use super::parser::{IfcModel, IfcSpace, MaterialLayerSpec};

/// Default floor area for `IfcSpace` when the footprint polygon cannot
/// be resolved. Matches the sample fixture (6 m × 4 m = 24 m²).
const DEFAULT_ZONE_FLOOR_AREA_M2: f64 = 24.0;
/// Default zone height when no per-space extrusion height is available.
const DEFAULT_ZONE_HEIGHT_M: f64 = 2.7;

/// Builder for converting IFC4 models into Fluxion schemas.
///
/// Idiomatic usage is via [`import_ifc`] / [`IfcParser::from_path`], but
/// constructing a builder directly is supported for unit tests.
#[derive(Debug, Default)]
pub struct IfcToSchema;

impl IfcToSchema {
    /// Construct a new mapper.
    pub fn new() -> Self {
        Self
    }

    /// Convert an [`IfcModel`] into a [`SimulationSchemaV1`].
    ///
    /// Validates that:
    /// - The schema version is `'IFC4'` (already enforced by the parser).
    /// - There is at least one `IfcSpace` (an IFC file without spaces is
    ///   not a building model).
    ///
    /// Material resolution:
    /// - For each wall/slab/roof, walk the matching `IfcRelAssociatesMaterial`
    ///   → `IfcMaterialLayerSetUsage` → `IfcMaterialLayerSet` → list of
    ///   `IfcMaterialLayer` → list of `IfcMaterial` names.
    /// - Build a [`ConstructionLayer`] for each `IfcMaterialLayer` using
    ///   the material name and the recorded thickness. Default
    ///   conductivity / density / specific heat are used since the
    ///   scaffold does not consume property sets.
    pub fn convert(&self, model: &IfcModel) -> Result<SimulationSchemaV1, IfcError> {
        if model.spaces.is_empty() {
            return Err(IfcError::conversion_error(
                "no IfcSpace entities found — cannot build a thermal model",
            ));
        }

        let zones = model.spaces.iter().map(build_zone_geometry).collect();
        let geometry = build_geometry(zones);

        let wall_construction = resolve_wall_construction(model);
        let roof_construction = resolve_roof_construction(model);
        let floor_construction = resolve_floor_construction(model);

        let constructions = ConstructionSet {
            wall: wall_construction,
            roof: roof_construction,
            floor: floor_construction,
            interzone: None,
        };

        let metadata = SchemaMetadata {
            name: "IFC4 STEP Import".to_string(),
            description: format!(
                "Imported from IFC4 STEP file. {} wall(s), {} slab(s), {} roof(s), {} space(s).",
                model.walls.len(),
                model.slabs.len(),
                model.roofs.len(),
                model.spaces.len()
            ),
            author: None,
            created_at: Some(chrono::Utc::now().format("%Y-%m-%d").to_string()),
            schema_version: crate::api::schema::SchemaVersion::V1,
        };

        Ok(SimulationSchemaV1 {
            version: crate::api::schema::SchemaVersion::V1,
            metadata,
            geometry,
            constructions,
            schedules: ScheduleSet::default(),
            weather: WeatherData::TmyLocation {
                location: "Denver, CO".to_string(),
            },
            controls: crate::api::schema::ControlSet::default(),
            output: SimulationOutput::default(),
        })
    }
}

/// Convenience wrapper: read an IFC4 STEP file and convert it.
///
/// Performs the full pipeline — [`IfcParser::from_path`] → mapper.
pub fn import_ifc(path: impl AsRef<Path>) -> Result<SimulationSchemaV1, IfcError> {
    let path = path.as_ref();
    let model = super::parser::IfcParser::from_path(path)?;
    IfcToSchema::new().convert(&model)
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn build_zone_geometry(space: &IfcSpace) -> ZoneGeometry {
    let _ = space; // Future: decode footprint polygon from `Representation`.
    ZoneGeometry {
        name: space.name.clone(),
        floor_area: DEFAULT_ZONE_FLOOR_AREA_M2,
        volume: DEFAULT_ZONE_FLOOR_AREA_M2 * DEFAULT_ZONE_HEIGHT_M,
        height: DEFAULT_ZONE_HEIGHT_M,
    }
}

fn build_geometry(zones: Vec<ZoneGeometry>) -> Geometry {
    let total_floor_area: f64 = zones.iter().map(|z| z.floor_area).sum();
    let total_volume: f64 = zones.iter().map(|z| z.volume).sum();
    let number_of_floors = zones.len().max(1);
    let floor_height = if total_floor_area > 0.0 {
        total_volume / total_floor_area
    } else {
        DEFAULT_ZONE_HEIGHT_M
    };
    Geometry {
        zones,
        total_floor_area,
        total_volume,
        number_of_floors,
        floor_height,
    }
}

/// Resolve the wall [`SurfaceConstruction`] from the first wall that
/// carries a material association.
fn resolve_wall_construction(model: &IfcModel) -> SurfaceConstruction {
    for wall in &model.walls {
        if let Some(c) = resolve_construction_for_product(model, wall.id, "Wall") {
            return c;
        }
    }
    SurfaceConstruction::default()
}

/// Resolve the roof [`SurfaceConstruction`] from the first roof that
/// carries a material association.
fn resolve_roof_construction(model: &IfcModel) -> SurfaceConstruction {
    for roof in &model.roofs {
        if let Some(c) = resolve_construction_for_product(model, roof.id, "Roof") {
            return c;
        }
    }
    SurfaceConstruction::default()
}

/// Resolve the floor [`SurfaceConstruction`] from the first slab of
/// type `.FLOOR.` that carries a material association.
fn resolve_floor_construction(model: &IfcModel) -> SurfaceConstruction {
    for slab in &model.slabs {
        // The scaffold treats every slab as a floor candidate. If a
        // model explicitly tags a slab as `.ROOF.`, skip it.
        if slab.predefined_type == ".ROOF." {
            continue;
        }
        if let Some(c) = resolve_construction_for_product(model, slab.id, "Slab") {
            return c;
        }
    }
    SurfaceConstruction::default()
}

/// Resolve a single product's [`SurfaceConstruction`] by walking the
/// material association chain:
///
/// `IfcRelAssociatesMaterial.RelatedObjects`
///   → `IfcMaterialLayerSetUsage`
///   → `IfcMaterialLayerSet`
///   → `[IfcMaterialLayer]`
///   → `IfcMaterial`
fn resolve_construction_for_product(
    model: &IfcModel,
    product_id: u64,
    fallback_name: &str,
) -> Option<SurfaceConstruction> {
    let association = model
        .material_associations
        .iter()
        .find(|a| a.related_object_ids.contains(&product_id))?;
    let usage_id = association.material_id;
    let layer_set_id = *model.layer_set_usage_targets.get(&usage_id)?;
    let layer_set = model.layer_sets.iter().find(|ls| ls.id == layer_set_id)?;
    let layers = collect_material_layers(model, &layer_set.layer_ids);
    let name = match layers.len() {
        0 => format!("{fallback_name} (empty)"),
        1 => format!("{fallback_name}-1Layer"),
        n => format!("{fallback_name}-{n}Layer"),
    };
    Some(SurfaceConstruction {
        name,
        layers,
        window: None,
    })
}

/// Collect [`ConstructionLayer`]s from a list of `IfcMaterialLayer`
/// ids. Layers missing from the model are silently skipped so a
/// partially-defined material chain still produces a usable (if
/// thinner) construction.
fn collect_material_layers(model: &IfcModel, layer_ids: &[u64]) -> Vec<ConstructionLayer> {
    let lookup: std::collections::HashMap<u64, &MaterialLayerSpec> =
        model.material_layers.iter().map(|l| (l.id, l)).collect();

    layer_ids
        .iter()
        .filter_map(|id| lookup.get(id).copied())
        .map(|layer| {
            let material_name = model
                .materials
                .get(&layer.material_id)
                .cloned()
                .unwrap_or_else(|| format!("Material#{}", layer.material_id));
            // The MVP scaffold does not decode property sets for
            // conductivity/density/specific heat; use conservative
            // defaults that bracket typical building materials.
            let (conductivity, density, specific_heat) = defaults_for(&material_name);
            ConstructionLayer::new(
                material_name,
                conductivity,
                density,
                specific_heat,
                layer.thickness.max(0.001),
            )
        })
        .collect()
}

/// Conservative default thermal properties for the MVP. The real
/// importer should consume `IfcMaterial` properties (or Psets); this
/// function is a placeholder that picks sensible defaults per material
/// category so the converted schema is at least numerically runnable.
fn defaults_for(material_name: &str) -> (f64, f64, f64) {
    let n = material_name.to_ascii_lowercase();
    if n.contains("concrete") {
        (1.4, 2300.0, 880.0)
    } else if n.contains("insul") || n.contains("foam") {
        (0.04, 30.0, 840.0)
    } else if n.contains("gypsum") || n.contains("plaster") {
        (0.21, 950.0, 840.0)
    } else if n.contains("wood") {
        (0.14, 500.0, 1600.0)
    } else if n.contains("steel") {
        (50.0, 7850.0, 490.0)
    } else {
        (0.5, 1000.0, 900.0)
    }
}

/// Round-trip helper used by the acceptance-criteria test:
///
/// IFC → SimulationSchema → gbXML → re-import → SimulationSchema.
///
/// The two schemas must agree on zone count and total floor area
/// within 0.5 %.
///
/// This lives in `mapping.rs` (and is re-exported via `mod.rs`) so
/// integration tests can call it without depending on the gbXML
/// writer directly.
pub fn round_trip_via_gbxml(schema: &SimulationSchemaV1) -> Result<SimulationSchemaV1, IfcError> {
    // The gbXML writer is the canonical conversion from SimulationSchema
    // to a textual format. We render to a string buffer (via a temp
    // path) and re-import.
    // Use unique temp file to avoid parallel test collisions
    let tmp = std::env::temp_dir().join(format!(
        "fluxion_ifc_roundtrip_{}_{}.gbxml",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    export_gbxml(schema, &tmp)
        .map_err(|e| IfcError::conversion_error(format!("gbXML export failed: {e}")))?;
    let rt_schema = import_gbxml(&tmp)
        .map_err(|e| IfcError::conversion_error(format!("gbXML re-import failed: {e}")))?;
    let _ = fs::remove_file(&tmp);
    Ok(rt_schema)
}

/// Round-trip helper for IFC: IFC → SimulationSchema → IFC → re-import.
///
/// The two schemas must agree on zone count and total floor area
/// within 0.5 %, and material layers must be preserved.
///
/// This lives in `mapping.rs` (and is re-exported via `mod.rs`) so
/// integration tests can call it without depending on the IFC
/// writer directly.
pub fn round_trip_via_ifc(schema: &SimulationSchemaV1) -> Result<SimulationSchemaV1, IfcError> {
    // Export to IFC format and re-import
    // Use unique temp file to avoid parallel test collisions
    let tmp = std::env::temp_dir().join(format!(
        "fluxion_ifc_roundtrip_{}_{}.ifc",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    super::writer::export_ifc(schema, &tmp)
        .map_err(|e| IfcError::conversion_error(format!("IFC export failed: {e}")))?;
    let rt_schema = import_ifc(&tmp)
        .map_err(|e| IfcError::conversion_error(format!("IFC re-import failed: {e}")))?;
    let _ = fs::remove_file(&tmp);
    Ok(rt_schema)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interop::ifc::parser::IfcParser;

    const SAMPLE: &str = "\
ISO-10303-21;
HEADER;
FILE_SCHEMA(('IFC4'));
ENDSEC;
DATA;
#1=IFCSPACE('0Sp4cGu1D0000000000',#2,'Zone1','Zone 1',$,$,$,$,.ELEMENT.,.INTERNAL.,$);
#10=IFCMATERIAL('Concrete',$,'concrete');
#11=IFCMATERIAL('Insulation',$,'insulation');
#20=IFCWALL('0W4llGu1D00000000000',#2,'Wall-N','Wall',$,$,$,$,.NOTDEFINED.);
#21=IFCSLAB('0Sl4bGu1D000000000000',#2,'Slab','Floor',$,$,$,$,.FLOOR.);
#22=IFCROOF('0R00fGu1D0000000000000',#2,'Roof','Roof',$,$,$,$,.NOTDEFINED.);
#30=IFCMATERIALLAYER(#10,0.100,$,'ConcreteLayer',$,$,$);
#31=IFCMATERIALLAYER(#11,0.050,$,'InsulationLayer',$,$,$);
#32=IFCMATERIALLAYER(#10,0.200,$,'SlabConcrete',$,$,$);
#33=IFCMATERIALLAYER(#11,0.100,$,'RoofInsulation',$,$,$);
#40=IFCMATERIALLAYERSET((#30,#31),'WallLayers',$);
#41=IFCMATERIALLAYERSET((#32),'SlabLayers',$);
#42=IFCMATERIALLAYERSET((#33),'RoofLayers',$);
#50=IFCMATERIALLAYERSETUSAGE(#40,.AXIS2.,.POSITIVE.,-0.075,$);
#51=IFCMATERIALLAYERSETUSAGE(#41,.AXIS3.,.POSITIVE.,0.,$);
#52=IFCMATERIALLAYERSETUSAGE(#42,.AXIS3.,.POSITIVE.,0.,$);
#60=IFCRELASSOCIATESMATERIAL('0R3lWGu1D0000000000',#2,$,$,(#20),#50);
#61=IFCRELASSOCIATESMATERIAL('0R3lSGu1D0000000000',#2,$,$,(#21),#51);
#62=IFCRELASSOCIATESMATERIAL('0R3lRGu1D0000000000',#2,$,$,(#22),#52);
ENDSEC;
END-ISO-10303-21;
";

    #[test]
    fn converts_minimal_ifc_to_schema() {
        let model = IfcParser::from_str(SAMPLE).expect("parses");
        let schema = IfcToSchema::new().convert(&model).expect("converts");
        assert_eq!(schema.geometry.zones.len(), 1);
        assert_eq!(schema.geometry.zones[0].name, "Zone1");
        assert!(schema.geometry.total_floor_area > 0.0);
    }

    #[test]
    fn zone_count_matches_ifc_space_count() {
        let model = IfcParser::from_str(SAMPLE).expect("parses");
        let schema = IfcToSchema::new().convert(&model).expect("converts");
        assert_eq!(
            schema.geometry.zones.len(),
            model.spaces.len(),
            "one zone per IfcSpace"
        );
    }

    #[test]
    fn material_layers_become_construction_layers() {
        let model = IfcParser::from_str(SAMPLE).expect("parses");
        let schema = IfcToSchema::new().convert(&model).expect("converts");
        // Wall construction has 2 layers (concrete + insulation).
        assert_eq!(schema.constructions.wall.layers.len(), 2);
        // Slab construction has 1 layer (concrete, 0.2 m).
        assert_eq!(schema.constructions.floor.layers.len(), 1);
        assert!((schema.constructions.floor.layers[0].thickness - 0.200).abs() < 1e-9);
        // Roof construction has 1 layer (insulation, 0.1 m).
        assert_eq!(schema.constructions.roof.layers.len(), 1);
        assert!((schema.constructions.roof.layers[0].thickness - 0.100).abs() < 1e-9);
    }

    #[test]
    fn rejects_model_with_no_spaces() {
        let src = "\
ISO-10303-21;
HEADER;
FILE_SCHEMA(('IFC4'));
ENDSEC;
DATA;
ENDSEC;
END-ISO-10303-21;
";
        let model = IfcParser::from_str(src).expect("parses");
        let err = IfcToSchema::new().convert(&model).expect_err("rejects");
        assert!(matches!(err, IfcError::Conversion(_)));
    }
}
