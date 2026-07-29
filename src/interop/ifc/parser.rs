// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! Typed parser for the four IFC4 entities required by issue #1343.
//!
//! Consumes the [`RawEntity`] stream produced by
//! [`super::step_lexer::tokenize`] and decodes
//! `IFCWALL`, `IFCSLAB`, `IFCROOF`, and `IFCSPACE` into typed structs.
//! All other entities are stored as raw [`GenericEntity`] records keyed
//! by id so the mapper can resolve cross-references (e.g.
//! `IfcRelAssociatesMaterial.RelatedObjects` → walls).
//!
//! # Field decoding
//!
//! Per IFC4 ADD2 (see
//! <https://standards.buildingsmart.org/IFC/RELEASE/IFC4/ADD2_TC1/HTML/>):
//!
//! | Entity | Relevant fields (after `GlobalId`, `OwnerHistory`) |
//! |--------|----------------------------------------------------|
//! | [`IfcWall`]/[`IfcSlab`]/[`IfcRoof`] | `Name`, `ObjectPlacement`, `Representation`, `Tag` |
//! | [`IfcSpace`] | `Name`, `ObjectPlacement`, `Representation`, `LongName`, `InteriorOrExteriorSpace` |
//!
//! For the MVP scaffold we only need the `Name` (for `ZoneGeometry.name`)
//! and the id (for cross-reference resolution). Geometry is left to the
//! per-surface solver and not consumed here.
//!
//! # Out of scope (issue #1343)
//!
//! - IfcWindow / IfcDoor — follow-up.
//! - Property sets (Pset_*) and material lists — minimal handling only.
//! - Full EXPRESS parsing — we only decode the subset of fields the
//!   mapper needs.

use std::collections::HashMap;

use super::error::IfcError;
use super::step_lexer::{tokenize_with_schema, RawEntity};

/// A typed `IFCWALL` entity.
///
/// `name` is the third positional field of the IFC entity (`Name`),
/// post-`GlobalId` and `OwnerHistory`. It corresponds to the user-facing
/// wall label (e.g. `"Wall-North"`).
#[derive(Debug, Clone, PartialEq)]
pub struct IfcWall {
    pub id: u64,
    pub global_id: String,
    pub name: String,
    pub line: usize,
}

impl IfcWall {
    pub fn new(id: u64, global_id: String, name: String, line: usize) -> Self {
        Self {
            id,
            global_id,
            name,
            line,
        }
    }
}

/// A typed `IFCSLAB` entity (floor/roof/landing/etc).
#[derive(Debug, Clone, PartialEq)]
pub struct IfcSlab {
    pub id: u64,
    pub global_id: String,
    pub name: String,
    /// `PredefinedType` (`.FLOOR.`, `.ROOF.`, `.LANDING.`, …) — captured
    /// verbatim with the leading/trailing dots so the mapper can keep the
    /// slab vs. floor distinction without inventing enum values.
    pub predefined_type: String,
    pub line: usize,
}

/// A typed `IFCBUILDING` entity (building).
#[derive(Debug, Clone, PartialEq)]
pub struct IfcBuilding {
    pub id: u64,
    pub global_id: String,
    pub name: String,
    pub line: usize,
}

/// A typed `IFCBUILDINGSTOREY` entity (floor level).
#[derive(Debug, Clone, PartialEq)]
pub struct IfcBuildingStorey {
    pub id: u64,
    pub global_id: String,
    pub name: String,
    pub line: usize,
}

/// A typed `IFCROOF` entity.
#[derive(Debug, Clone, PartialEq)]
pub struct IfcRoof {
    pub id: u64,
    pub global_id: String,
    pub name: String,
    pub predefined_type: String,
    pub line: usize,
}

/// A typed `IFCSPACE` entity (thermal zone).
#[derive(Debug, Clone, PartialEq)]
pub struct IfcSpace {
    pub id: u64,
    pub global_id: String,
    pub name: String,
    pub line: usize,
}

/// A single material-layer record (`IFCMATERIALLAYER`).
///
/// `thickness` is in metres; `material_id` is the STEP id of the
/// matching `IFCMATERIAL`. The raw `category` field is preserved so
/// callers can group layers by material category (`.CONCRETE.` etc).
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialLayerSpec {
    pub id: u64,
    pub material_id: u64,
    pub thickness: f64,
    pub category: String,
    pub line: usize,
}

/// A material association (`IFCRELASSOCIATESMATERIAL`).
///
/// Maps each `related_object_id` (wall / slab / roof / space) to its
/// material assignment. The matching [`MaterialLayerSpec`]s are looked
/// up via the `layer_set_usage_id` chain at mapping time.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialAssociation {
    pub id: u64,
    pub related_object_ids: Vec<u64>,
    pub material_id: u64,
    pub line: usize,
}

/// A generic entity record used for every non-MVP entity.
///
/// The `args` field preserves the raw arg body produced by the lexer so
/// downstream tools can re-decode it. `name` is the upper-case IFC
/// class name (e.g. `IFCRELAGGREGATES`).
#[derive(Debug, Clone, PartialEq)]
pub struct GenericEntity {
    pub id: u64,
    pub name: String,
    pub args: String,
    pub line: usize,
}

/// Decoded IFC4 model.
///
/// Holds the typed records for the four MVP entities and a generic
/// entity map keyed by id for resolving cross-references. The mapper in
/// [`super::mapping`] walks these structures to build a
/// [`crate::api::schema::SimulationSchemaV1`].
#[derive(Debug, Clone, Default)]
pub struct IfcModel {
    /// Schema identifier from `FILE_SCHEMA(('IFC4'))` — the parser
    /// currently accepts only `'IFC4'`. Set to `None` if the file did
    /// not declare a schema (we still attempt to decode entities).
    pub schema: Option<String>,

    /// All buildings discovered in the file.
    pub buildings: Vec<IfcBuilding>,
    /// All building storeys discovered in the file.
    pub storeys: Vec<IfcBuildingStorey>,
    /// All walls discovered in the file.
    pub walls: Vec<IfcWall>,
    /// All slabs discovered in the file.
    pub slabs: Vec<IfcSlab>,
    /// All roofs discovered in the file.
    pub roofs: Vec<IfcRoof>,
    /// All spaces (thermal zones) discovered in the file.
    pub spaces: Vec<IfcSpace>,

    /// All material associations (`IFCRELASSOCIATESMATERIAL`).
    pub material_associations: Vec<MaterialAssociation>,

    /// Material layer set usages (`IFCMATERIALLAYERSETUSAGE`).
    ///
    /// `layer_set_id` points to the parent `IFCMATERIALLAYERSET`.
    /// `layers` is the chain of material layer ids captured by name.
    /// The mapper later resolves each layer to its
    /// [`MaterialLayerSpec`].
    pub layer_set_usages: Vec<LayerSetUsage>,
    /// Material layer sets (`IFCMATERIALLAYERSET`) — list of layer ids.
    pub layer_sets: Vec<LayerSet>,
    /// Material layers (`IFCMATERIALLAYER`).
    pub material_layers: Vec<MaterialLayerSpec>,
    /// Material associations to layer-set-usages (transitive target).
    pub layer_set_usage_targets: HashMap<u64, u64>,

    /// Materials (`IFCMATERIAL`) keyed by id.
    pub materials: HashMap<u64, String>,

    /// Every other entity, keyed by id.
    pub entities: HashMap<u64, GenericEntity>,
}

/// `IFCMATERIALLAYERSETUSAGE` decoded minimally — just enough to walk
/// the chain: `usage → layer_set → [layer, layer, ...] → material`.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSetUsage {
    pub id: u64,
    pub layer_set_id: u64,
}

/// `IFCMATERIALLAYERSET` decoded minimally — list of layer ids.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSet {
    pub id: u64,
    pub layer_ids: Vec<u64>,
}

/// Entry point for parsing IFC4 STEP physical files.
///
/// Mirrors the `IdfParser` API surface (issue #1341) so the rest of the
/// crate can call `IfcParser::from_str(...)` / `from_path(...)` uniformly.
pub struct IfcParser;

impl IfcParser {
    /// Parse an in-memory IFC4 STEP document.
    ///
    /// Performs the full two-stage pipeline: lex → decode. Errors from
    /// either stage bubble up as [`IfcError::Parse`] with the offending
    /// line number.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(source: &str) -> Result<IfcModel, IfcError> {
        let (schema, raw) = tokenize_with_schema(source)?;
        if let Some(ref s) = schema {
            if s != "IFC4" {
                return Err(IfcError::UnsupportedSchema(s.clone()));
            }
        }
        let mut model = IfcModel::default();
        model.schema = schema;
        for entity in raw {
            Self::classify(entity, &mut model)?;
        }
        Ok(model)
    }

    /// Parse an IFC4 STEP document from a filesystem path.
    pub fn from_path(path: &std::path::Path) -> Result<IfcModel, IfcError> {
        let content = std::fs::read_to_string(path)?;
        Self::from_str(&content)
    }

    /// Dispatch one raw entity into the appropriate typed bucket.
    fn classify(entity: RawEntity, model: &mut IfcModel) -> Result<(), IfcError> {
        match entity.name.as_str() {
            "IFCWALL" | "IFCWALLSTANDARDCASE" => {
                let (global_id, name) = parse_root_like(&entity)?;
                model
                    .walls
                    .push(IfcWall::new(entity.id, global_id, name, entity.line));
            }
            "IFCSLAB" => {
                let (global_id, name) = parse_root_like(&entity)?;
                // IFC4 ADD2 §IfcSlab: `PredefinedType` is the 9th
                // 1-indexed attribute (0-indexed 8), after the inherited
                // chain `IfcRoot(GlobalId, OwnerHistory, Name, Description)
                // → IfcObject(ObjectType) → IfcProduct(ObjectPlacement,
                // Representation) → IfcElement(Tag) → IfcSlab(PredefinedType)`.
                let predefined_type =
                    extract_enum(&entity.args, 8).unwrap_or_else(|| ".NOTDEFINED.".to_string());
                model.slabs.push(IfcSlab {
                    id: entity.id,
                    global_id,
                    name,
                    predefined_type,
                    line: entity.line,
                });
            }
            "IFCBUILDING" => {
                let (global_id, name) = parse_root_like(&entity)?;
                model.buildings.push(IfcBuilding {
                    id: entity.id,
                    global_id,
                    name,
                    line: entity.line,
                });
            }
            "IFCBUILDINGSTOREY" => {
                let (global_id, name) = parse_root_like(&entity)?;
                model.storeys.push(IfcBuildingStorey {
                    id: entity.id,
                    global_id,
                    name,
                    line: entity.line,
                });
            }
            "IFCROOF" => {
                let (global_id, name) = parse_root_like(&entity)?;
                // Same chain as IfcSlab — see comment above.
                let predefined_type =
                    extract_enum(&entity.args, 8).unwrap_or_else(|| ".NOTDEFINED.".to_string());
                model.roofs.push(IfcRoof {
                    id: entity.id,
                    global_id,
                    name,
                    predefined_type,
                    line: entity.line,
                });
            }
            "IFCSPACE" => {
                let (global_id, name) = parse_root_like(&entity)?;
                model.spaces.push(IfcSpace {
                    id: entity.id,
                    global_id,
                    name,
                    line: entity.line,
                });
            }
            "IFCMATERIAL" => {
                // IFCMATERIAL(Name, Description, Category).
                let name = extract_nth_quoted(&entity.args, 0).unwrap_or_default();
                model.materials.insert(entity.id, name);
                model.entities.insert(
                    entity.id,
                    GenericEntity {
                        id: entity.id,
                        name: entity.name,
                        args: entity.args,
                        line: entity.line,
                    },
                );
            }
            "IFCMATERIALLAYER" => {
                // IFCMATERIALLAYER(Material, LayerThickness, isVentilated,
                //                   Name, Description, Category, Priority).
                let material_id = extract_nth_ref(&entity.args, 0).ok_or_else(|| {
                    IfcError::parse_error(entity.line, "IFCMATERIALLAYER missing material ref")
                })?;
                let thickness = extract_nth_real(&entity.args, 1).ok_or_else(|| {
                    IfcError::parse_error(entity.line, "IFCMATERIALLAYER missing LayerThickness")
                })?;
                let category = extract_nth_quoted(&entity.args, 5).unwrap_or_default();
                model.material_layers.push(MaterialLayerSpec {
                    id: entity.id,
                    material_id,
                    thickness,
                    category,
                    line: entity.line,
                });
            }
            "IFCMATERIALLAYERSET" => {
                // IFCMATERIALLAYERSET(Layers, LayerSetName, Description).
                let layer_ids = extract_nth_ref_list(&entity.args, 0).unwrap_or_default();
                model.layer_sets.push(LayerSet {
                    id: entity.id,
                    layer_ids,
                });
            }
            "IFCMATERIALLAYERSETUSAGE" => {
                // IFCMATERIALLAYERSETUSAGE(ForLayerSet, LayerSetDirection,
                //                           DirectionSense, OffsetFromReferenceLine,
                //                           ReferenceExtent).
                let layer_set_id = extract_nth_ref(&entity.args, 0).ok_or_else(|| {
                    IfcError::parse_error(
                        entity.line,
                        "IFCMATERIALLAYERSETUSAGE missing layer set ref",
                    )
                })?;
                model.layer_set_usages.push(LayerSetUsage {
                    id: entity.id,
                    layer_set_id,
                });
                model
                    .layer_set_usage_targets
                    .insert(entity.id, layer_set_id);
            }
            "IFCRELASSOCIATESMATERIAL" => {
                // IFCRELASSOCIATESMATERIAL(GlobalId, OwnerHistory, Name,
                //                          Description, RelatedObjects, RelatingMaterial).
                let related_object_ids = extract_nth_ref_list(&entity.args, 4).unwrap_or_default();
                let material_id = extract_nth_ref(&entity.args, 5).ok_or_else(|| {
                    IfcError::parse_error(
                        entity.line,
                        "IFCRELASSOCIATESMATERIAL missing RelatingMaterial",
                    )
                })?;
                model.material_associations.push(MaterialAssociation {
                    id: entity.id,
                    related_object_ids,
                    material_id,
                    line: entity.line,
                });
            }
            _ => {
                // Capture every other entity for cross-reference lookup
                // (e.g. IfcRelContainedInSpatialStructure, IfcRelAggregates).
                model.entities.insert(
                    entity.id,
                    GenericEntity {
                        id: entity.id,
                        name: entity.name,
                        args: entity.args,
                        line: entity.line,
                    },
                );
            }
        }
        Ok(())
    }
}

/// Decode the first two fields of an `IfcRoot`-derived entity
/// (`GlobalId`, `OwnerHistory`) and return `(GlobalId, Name)`.
///
/// For IFC4 `IfcWall` / `IfcSlab` / `IfcRoof` / `IfcSpace` the field
/// layout is:
///
/// ```text
/// GlobalId          : IfcGloballyUniqueId  (single-quoted string)
/// OwnerHistory      : IfcOwnerHistory      (#ref)
/// Name              : IfcLabel             (single-quoted string | $)
/// ...remaining IFC fields...
/// ```
///
/// We only need the first and third; the remaining fields (placement,
/// representation, predefined type, etc.) are decoded separately.
fn parse_root_like(entity: &RawEntity) -> Result<(String, String), IfcError> {
    let global_id = extract_nth_quoted(&entity.args, 0).ok_or_else(|| {
        IfcError::parse_error(entity.line, format!("{} missing GlobalId", entity.name))
    })?;
    let name = extract_nth_quoted(&entity.args, 2).unwrap_or_default();
    Ok((global_id, name))
}

/// Extract the n-th argument (0-indexed) from a comma-separated arg
/// body, respecting parentheses, single quotes, and enums.
fn extract_nth_arg(args: &str, n: usize) -> Option<String> {
    let bytes = args.as_bytes();
    let mut i = 0;
    let mut depth: usize = 0;
    let mut in_string = false;
    let mut arg_idx: usize = 0;
    let mut start: usize = 0;

    while i < bytes.len() {
        let c = bytes[i];
        if in_string {
            if c == b'\'' {
                if bytes.get(i + 1) == Some(&b'\'') {
                    i += 2;
                    continue;
                }
                in_string = false;
            }
            i += 1;
            continue;
        }
        match c {
            b'\'' => {
                in_string = true;
                i += 1;
            }
            b'(' => {
                depth += 1;
                i += 1;
            }
            b')' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            b',' if depth == 0 => {
                if arg_idx == n {
                    return Some(args[start..i].trim().to_string());
                }
                arg_idx += 1;
                i += 1;
                while i < bytes.len() && (bytes[i] as char).is_whitespace() {
                    i += 1;
                }
                start = i;
            }
            _ => i += 1,
        }
    }
    if arg_idx == n && start <= i {
        return Some(args[start..i].trim().to_string());
    }
    None
}

fn extract_nth_quoted(args: &str, n: usize) -> Option<String> {
    let arg = extract_nth_arg(args, n)?;
    let s = arg.trim();
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'\'') || bytes.last() != Some(&b'\'') {
        return None;
    }
    let inner = &s[1..s.len() - 1];
    Some(inner.replace("''", "'"))
}

fn extract_nth_enum(args: &str, n: usize) -> Option<String> {
    let arg = extract_nth_arg(args, n)?;
    let s = arg.trim();
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'.') || bytes.last() != Some(&b'.') {
        return None;
    }
    Some(s.to_string())
}

fn extract_enum(args: &str, n: usize) -> Option<String> {
    extract_nth_enum(args, n)
}

fn extract_nth_ref(args: &str, n: usize) -> Option<u64> {
    let arg = extract_nth_arg(args, n)?;
    let s = arg.trim();
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'#') {
        return None;
    }
    s[1..].parse().ok()
}

fn extract_nth_ref_list(args: &str, n: usize) -> Option<Vec<u64>> {
    let arg = extract_nth_arg(args, n)?;
    let s = arg.trim();
    if !s.starts_with('(') || !s.ends_with(')') {
        return None;
    }
    let inner = &s[1..s.len() - 1];
    let mut out = Vec::new();
    for piece in inner.split(',') {
        let piece = piece.trim();
        if let Some(rest) = piece.strip_prefix('#') {
            if let Ok(id) = rest.parse::<u64>() {
                out.push(id);
            }
        }
    }
    Some(out)
}

fn extract_nth_real(args: &str, n: usize) -> Option<f64> {
    let arg = extract_nth_arg(args, n)?;
    let s = arg.trim();
    if s == "$" || s == "*" {
        return None;
    }
    s.parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_ifc4_document() {
        let src = "\
ISO-10303-21;
HEADER;
FILE_SCHEMA(('IFC4'));
ENDSEC;
DATA;
#1=IFCWALL('0W4llGu1D00000000000',#2,'Wall-1','Description',$,$,$,$,.NOTDEFINED.);
ENDSEC;
END-ISO-10303-21;
";
        let model = IfcParser::from_str(src).expect("parses");
        assert_eq!(model.schema.as_deref(), Some("IFC4"));
        assert_eq!(model.walls.len(), 1);
        assert_eq!(model.walls[0].name, "Wall-1");
        assert_eq!(model.walls[0].global_id, "0W4llGu1D00000000000");
    }

    #[test]
    fn classifies_wall_standard_case_as_wall() {
        let src = "\
ISO-10303-21;
HEADER;
FILE_SCHEMA(('IFC4'));
ENDSEC;
DATA;
#1=IFCWALLSTANDARDCASE('0W4llGu1D00000000000',#2,'Wall-Std',$,$,$,$,$,.NOTDEFINED.);
ENDSEC;
END-ISO-10303-21;
";
        let model = IfcParser::from_str(src).expect("parses");
        assert_eq!(model.walls.len(), 1);
        assert_eq!(model.walls[0].name, "Wall-Std");
    }

    #[test]
    fn parses_slab_with_predefined_type() {
        // IFC4 ADD2 IfcSlab inherits through IfcRoot → IfcObjectDefinition
        // → IfcObject → IfcProduct → IfcElement → IfcBuildingElement →
        // IfcSlab, giving 9 total 0-indexed fields:
        //   0=GlobalId, 1=OwnerHistory, 2=Name, 3=Description,
        //   4=ObjectType, 5=ObjectPlacement, 6=Representation,
        //   7=Tag, 8=PredefinedType.
        let src = "\
ISO-10303-21;
HEADER;
FILE_SCHEMA(('IFC4'));
ENDSEC;
DATA;
#1=IFCSLAB('0Sl4bGu1D000000000000',#2,'Slab','Slab desc','FloorType',$,$,$,.FLOOR.);
ENDSEC;
END-ISO-10303-21;
";
        let model = IfcParser::from_str(src).expect("parses");
        assert_eq!(model.slabs.len(), 1);
        assert_eq!(model.slabs[0].name, "Slab");
        assert_eq!(model.slabs[0].predefined_type, ".FLOOR.");
    }

    #[test]
    fn parses_space_and_material_chain() {
        let src = "\
ISO-10303-21;
HEADER;
FILE_SCHEMA(('IFC4'));
ENDSEC;
DATA;
#1=IFCSPACE('0Sp4cGu1D0000000000',#2,'Zone1',$,$,$,$,.ELEMENT.,.INTERNAL.,$);
#10=IFCMATERIAL('Concrete',$,'concrete');
#11=IFCMATERIALLAYER(#10,0.200,$,'SlabConcrete',$,$,$);
#12=IFCMATERIALLAYERSET((#11),'SlabLayers',$);
#13=IFCMATERIALLAYERSETUSAGE(#12,.AXIS3.,.POSITIVE.,0.,$);
#14=IFCRELASSOCIATESMATERIAL('0R3lGu1D0000000000',#2,$,$,(#1),#13);
ENDSEC;
END-ISO-10303-21;
";
        let model = IfcParser::from_str(src).expect("parses");
        assert_eq!(model.spaces.len(), 1);
        assert_eq!(model.spaces[0].name, "Zone1");
        assert_eq!(model.material_layers.len(), 1);
        assert_eq!(model.material_layers[0].thickness, 0.200);
        assert_eq!(model.material_layers[0].material_id, 10);
        assert_eq!(model.layer_sets.len(), 1);
        assert_eq!(model.layer_sets[0].layer_ids, vec![11]);
        assert_eq!(model.layer_set_usages.len(), 1);
        assert_eq!(model.material_associations.len(), 1);
        assert_eq!(model.material_associations[0].related_object_ids, vec![1]);
    }

    #[test]
    fn rejects_non_ifc4_schema() {
        let src = "\
ISO-10303-21;
HEADER;
FILE_SCHEMA(('IFC2X3'));
ENDSEC;
DATA;
ENDSEC;
END-ISO-10303-21;
";
        let err = IfcParser::from_str(src).expect_err("rejects IFC2X3");
        assert!(matches!(err, IfcError::UnsupportedSchema(_)));
    }

    #[test]
    fn extract_nth_quoted_handles_doubled_quote() {
        let args = "'It''s ok',$,.NOTDEFINED.";
        let v = extract_nth_quoted(args, 0).expect("first arg is quoted");
        assert_eq!(v, "It's ok");
        assert_eq!(extract_nth_quoted(args, 1), None);
        assert_eq!(extract_nth_enum(args, 2).as_deref(), Some(".NOTDEFINED."));
    }
}
