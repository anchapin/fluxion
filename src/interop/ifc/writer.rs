// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! IFC4 STEP export - converts Fluxion schemas to IFC4 STEP physical files.
//!
//! Exports zone geometry as [`IfcBuilding`](crate::interop::ifc::parser::IfcBuilding),
//! [`IfcBuildingStorey`](crate::interop::ifc::parser::IfcBuildingStorey), and
//! [`IfcSpace`](crate::interop::ifc::parser::IfcSpace) entities. Building surfaces are
//! exported as [`IfcBuildingElementProxy`](crate::interop::ifc::parser::IfcBuildingElementProxy).
//! Material properties are exported as [`IfcMaterialLayer`](crate::interop::ifc::parser::MaterialLayerSpec).
//!
//! # Example
//!
//! ```ignore
//! use fluxion::interop::ifc::export_ifc;
//! use crate::api::schema::SimulationSchemaV1;
//!
//! let schema = SimulationSchemaV1::default();
//! export_ifc(&schema, "output.ifc")?;
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::api::schema::SimulationSchemaV1;
use crate::sim::construction::ConstructionLayer;
use crate::interop::ifc::error::IfcError;

/// Export a SimulationSchemaV1 to an IFC4 STEP physical file.
pub fn export_ifc(schema: &SimulationSchemaV1, path: impl AsRef<Path>) -> Result<(), IfcError> {
    let file = File::create(path.as_ref())
        .map_err(|e| IfcError::conversion_error(format!("failed to create IFC file: {}", e)))?;
    let mut writer = BufWriter::new(file);
    write_ifc_file(schema, &mut writer)?;
    Ok(())
}

/// Write an IFC4 STEP file from a SimulationSchemaV1.
pub fn write_ifc_file<W: Write>(schema: &SimulationSchemaV1, output: W) -> Result<(), IfcError> {
    let mut writer = IfcWriter::new(output);
    writer.write_schema(schema)
}

/// IFC4 STEP file writer.
pub struct IfcWriter<W: Write> {
    output: W,
    next_id: u64,
}

impl<W: Write> IfcWriter<W> {
    /// Create a new IfcWriter.
    pub fn new(output: W) -> Self {
        IfcWriter {
            output,
            next_id: 1,
        }
    }

    /// Write a SimulationSchemaV1 as an IFC4 STEP file.
    pub fn write_schema(&mut self, schema: &SimulationSchemaV1) -> Result<(), IfcError> {
        self.write_header(schema)?;
        self.write_data(schema)?;
        Ok(())
    }

    fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn write_header(&mut self, _schema: &SimulationSchemaV1) -> Result<(), IfcError> {
        writeln!(self.output, "ISO-10303-21;").map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "HEADER;").map_err(|e| IfcError::conversion_error(e.to_string()))?;

        // FILE_DESCRIPTION
        writeln!(self.output, "FILE_DESCRIPTION(('ViewDefinition [CoordinationView]','Fluxion IFC4 export'),'2;1');")
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;

        // FILE_NAME
        let now = chrono::Utc::now();
        let date_str = now.format("%Y-%m-%d").to_string();
        writeln!(self.output, "FILE_NAME('export.ifc','{}',('Fluxion'),('Fluxion'),'fluxion','fluxion','IFC4 STEP export');", date_str)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;

        // FILE_SCHEMA
        writeln!(self.output, "FILE_SCHEMA(('IFC4'));").map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "ENDSEC;").map_err(|e| IfcError::conversion_error(e.to_string()))?;

        Ok(())
    }

    fn write_data(&mut self, schema: &SimulationSchemaV1) -> Result<(), IfcError> {
        writeln!(self.output, "DATA;").map_err(|e| IfcError::conversion_error(e.to_string()))?;

        // Write owner history (minimal)
        let person_id = self.next_id();
        let org_id = self.next_id();
        let person_org_id = self.next_id();
        let app_id = self.next_id();
        let owner_hist_id = self.next_id();

        writeln!(self.output, "#{}=IFCPERSON('unknown','unknown',$,$,$,$,$,$);", person_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#{}=IFCORGANIZATION($,'Fluxion',$,$,$);", org_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#{}=IFCPERSONANDORGANIZATION(#{},#{},$);", person_org_id, person_id, org_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#{}=IFCAPPLICATION(#{},'0.0','fluxion','fluxion');", app_id, org_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#{}=IFCOWNERHISTORY(#{},#{},.READWRITE.,.ADDED.,1735689600,$,$,1735689600);", owner_hist_id, person_org_id, app_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;

        // Write unit assignment
        let pt_id = self.next_id();
        let dir_z_id = self.next_id();
        let dir_x_id = self.next_id();
        let dim_id = self.next_id();
        let length_unit_id = self.next_id();
        let area_unit_id = self.next_id();
        let vol_unit_id = self.next_id();
        let unit_assign_id = self.next_id();
        let context_id = self.next_id();
        let project_id = self.next_id();

        writeln!(self.output, "#{}=IFCCARTESIANPOINT((0.,0.,0.));", pt_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#{}=IFCDIRECTION((0.,0.,1.));", dir_z_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#{}=IFCDIRECTION((1.,0.,0.));", dir_x_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#{}=IFCDIMENSIONALEXPONENTS(1,0,0,0,0,0,0);", dim_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#{}=IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);", length_unit_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#{}=IFCSIUNIT(*,.AREAUNIT.,$,.SQUARE_METRE.);", area_unit_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#{}=IFCSIUNIT(*,.VOLUMEUNIT.,$,.CUBIC_METRE.);", vol_unit_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#{}=IFCUNITASSIGNMENT((#{},#{},#{}));", unit_assign_id, length_unit_id, area_unit_id, vol_unit_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#{}=IFCGEOMETRICREPRESENTATIONCONTEXT('3D','Model',3,1.0E-05,#{},#{});", context_id, pt_id, dir_z_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#{}=IFCPROJECT('0Exp0rtPr0jGu1D00000',#{},'{}',$,$,$,$,(#{}),#{});",
            project_id, owner_hist_id, escape_ifc_string(&schema.metadata.name), context_id, unit_assign_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;

        // Build structure: building -> storey -> spaces
        let building_id = self.next_id();
        let storey_id = self.next_id();
        let site_id = self.next_id();
        let local_placement_base = self.next_id();
        let local_placement_building = self.next_id();
        let local_placement_storey = self.next_id();

        writeln!(self.output, "#{}=IFCLOCALPLACEMENT($,#{});", local_placement_base, pt_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#{}=IFCSITE('0Exp0rtS1teGu1D000000',#{},'Site',$,$,#{},$,$,.ELEMENT.,$,$,$,$,$);",
            site_id, owner_hist_id, local_placement_base)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#1000=IFCRELAGGREGATES('0Exp0rtRAggSite000',#{},$,$,#{},(#{}));",
            owner_hist_id, project_id, site_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;

        writeln!(self.output, "#{}=IFCLOCALPLACEMENT(#{},#{});", local_placement_building, local_placement_base, pt_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#{}=IFCBUILDING('0Exp0rtBu1Gu1D0000000',#{},'{}',$,$,#{},$,$,.ELEMENT.,$,$,$);",
            building_id, owner_hist_id, escape_ifc_string(&schema.metadata.name), local_placement_building)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#1001=IFCRELAGGREGATES('0Exp0rtRAggBld0000',#{},$,$,#{},(#{}));",
            owner_hist_id, site_id, building_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;

        writeln!(self.output, "#{}=IFCLOCALPLACEMENT(#{},#{});", local_placement_storey, local_placement_building, pt_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#{}=IFCBUILDINGSTOREY('0Exp0rtStGu1D0000000',#{},'Storey',$,$,#{},$,$,.ELEMENT.,0.);",
            storey_id, owner_hist_id, local_placement_storey)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "#1002=IFCRELAGGREGATES('0Exp0rtRAggSt000000',#{},$,$,#{},(#{}));",
            owner_hist_id, building_id, storey_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;

        // Collect all unique materials from constructions
        let mut material_map: std::collections::HashMap<String, (u64, &ConstructionLayer)> = std::collections::HashMap::new();
        let mut mat_counter = 1;

        for construction in [&schema.constructions.wall, &schema.constructions.roof, &schema.constructions.floor].iter() {
            for layer in &construction.layers {
                if !material_map.contains_key(&layer.name) {
                    let mat_id = 2000 + mat_counter;
                    material_map.insert(layer.name.clone(), (mat_id, layer));
                    mat_counter += 1;
                }
            }
        }

        // Write IFCMATERIAL entities
        for (name, (mat_id, _layer)) in &material_map {
            writeln!(self.output, "#{}=IFCMATERIAL('{}',$,'{}');", mat_id, escape_ifc_string(name), escape_ifc_string(name))
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        }

        // Write IFCMATERIALLAYER entities for each construction layer
        let mut layer_counter = 3000;
        let mut wall_layer_ids: Vec<u64> = Vec::new();
        let mut roof_layer_ids: Vec<u64> = Vec::new();
        let mut floor_layer_ids: Vec<u64> = Vec::new();

        // Wall layers
        for layer in &schema.constructions.wall.layers {
            let mat_id = material_map.get(&layer.name).map(|(id, _)| *id).unwrap_or(2001);
            writeln!(self.output, "#{}=IFCMATERIALLAYER(#{},{},$,'{}',$,$,$);",
                layer_counter, mat_id, layer.thickness, escape_ifc_string(&layer.name))
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
            wall_layer_ids.push(layer_counter);
            layer_counter += 1;
        }

        // Roof layers
        for layer in &schema.constructions.roof.layers {
            let mat_id = material_map.get(&layer.name).map(|(id, _)| *id).unwrap_or(2001);
            writeln!(self.output, "#{}=IFCMATERIALLAYER(#{},{},$,'{}',$,$,$);",
                layer_counter, mat_id, layer.thickness, escape_ifc_string(&layer.name))
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
            roof_layer_ids.push(layer_counter);
            layer_counter += 1;
        }

        // Floor layers
        for layer in &schema.constructions.floor.layers {
            let mat_id = material_map.get(&layer.name).map(|(id, _)| *id).unwrap_or(2001);
            writeln!(self.output, "#{}=IFCMATERIALLAYER(#{},{},$,'{}',$,$,$);",
                layer_counter, mat_id, layer.thickness, escape_ifc_string(&layer.name))
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
            floor_layer_ids.push(layer_counter);
            layer_counter += 1;
        }

        // Write IFCMATERIALLAYERSET entities
        let wall_layer_set_id = layer_counter;
        if !wall_layer_ids.is_empty() {
            let layer_refs: Vec<String> = wall_layer_ids.iter().map(|id| format!("#{}", id)).collect();
            writeln!(self.output, "#{}=IFCMATERIALLAYERSET(({}),'WallLayers',$);",
                wall_layer_set_id, layer_refs.join(","))
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
            layer_counter += 1;
        }

        let roof_layer_set_id = layer_counter;
        if !roof_layer_ids.is_empty() {
            let layer_refs: Vec<String> = roof_layer_ids.iter().map(|id| format!("#{}", id)).collect();
            writeln!(self.output, "#{}=IFCMATERIALLAYERSET(({}),'RoofLayers',$);",
                roof_layer_set_id, layer_refs.join(","))
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
            layer_counter += 1;
        }

        let floor_layer_set_id = layer_counter;
        if !floor_layer_ids.is_empty() {
            let layer_refs: Vec<String> = floor_layer_ids.iter().map(|id| format!("#{}", id)).collect();
            writeln!(self.output, "#{}=IFCMATERIALLAYERSET(({}),'FloorLayers',$);",
                floor_layer_set_id, layer_refs.join(","))
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
            layer_counter += 1;
        }

        // Write IFCMATERIALLAYERSETUSAGE entities
        let mut wall_usage_id = 0u64;
        let mut roof_usage_id = 0u64;
        let mut floor_usage_id = 0u64;

        if !wall_layer_ids.is_empty() {
            wall_usage_id = layer_counter;
            writeln!(self.output, "#{}=IFCMATERIALLAYERSETUSAGE(#{},.AXIS2.,.POSITIVE.,0.,$);",
                wall_usage_id, wall_layer_set_id)
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
            layer_counter += 1;
        }

        if !roof_layer_ids.is_empty() {
            roof_usage_id = layer_counter;
            writeln!(self.output, "#{}=IFCMATERIALLAYERSETUSAGE(#{},.AXIS3.,.POSITIVE.,0.,$);",
                roof_usage_id, roof_layer_set_id)
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
            layer_counter += 1;
        }

        if !floor_layer_ids.is_empty() {
            floor_usage_id = layer_counter;
            writeln!(self.output, "#{}=IFCMATERIALLAYERSETUSAGE(#{},.AXIS3.,.POSITIVE.,0.,$);",
                floor_usage_id, floor_layer_set_id)
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
            layer_counter += 1;
        }

        // Write spaces (zones)
        let mut space_ids: Vec<u64> = Vec::new();
        for (idx, zone) in schema.geometry.zones.iter().enumerate() {
            let space_local_placement = self.next_id();
            let space_id = self.next_id();

            writeln!(self.output, "#{}=IFCLOCALPLACEMENT(#{},#{});", space_local_placement, local_placement_storey, pt_id)
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
            writeln!(self.output, "#{}=IFCSPACE('0Exp0rtSp{{}}Gu1D000000',#{},'{}','{}',$,#{},$,$,.ELEMENT.,.INTERNAL.,$);",
                space_id, owner_hist_id, escape_ifc_string(&zone.name), escape_ifc_string(&zone.name), space_local_placement)
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
            space_ids.push(space_id);

            writeln!(self.output, "#{}000=IFCRELAGGREGATES('0Exp0rtRAggZ{}000000',#{},$,$,#{},(#{}));",
                1000 + idx, idx, owner_hist_id, storey_id, space_id)
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        }

        // Write building elements (walls as IfcBuildingElementProxy)
        let mut wall_ids: Vec<u64> = Vec::new();
        let mut slab_ids: Vec<u64> = Vec::new();
        let mut roof_ids: Vec<u64> = Vec::new();

        let placement_for_elements = self.next_id();
        writeln!(self.output, "#{}=IFCLOCALPLACEMENT(#{},#{});", placement_for_elements, local_placement_storey, pt_id)
            .map_err(|e| IfcError::conversion_error(e.to_string()))?;

        // Write wall elements
        for (idx, _) in schema.geometry.zones.iter().enumerate() {
            // Export walls based on zone's floor area (approximate 4 walls per zone)
            let num_walls = 4;
            for w in 0..num_walls {
                let wall_id = layer_counter;
                let wall_name = format!("Wall-{}-{}", idx + 1, (b'A' + w as u8) as char);
                writeln!(self.output, "#{}=IFCBUILDINGELEMENTPROXY('0Exp0rtW{{}}Gu1D00000000',#{},'{}','{}',$,#{},$,$,.NOTDEFINED.);",
                    wall_id, wall_id, escape_ifc_string(&wall_name), escape_ifc_string(&wall_name), placement_for_elements)
                    .map_err(|e| IfcError::conversion_error(e.to_string()))?;
                wall_ids.push(wall_id);
                layer_counter += 1;
            }
        }

        // Write roof elements
        for (idx, _) in schema.geometry.zones.iter().enumerate() {
            let roof_id = layer_counter;
            writeln!(self.output, "#{}=IFCBUILDINGELEMENTPROXY('0Exp0rtRf{{}}Gu1D00000000',#{},'Roof-{}','Roof-{}',$,#{},$,$,.NOTDEFINED.);",
                roof_id, roof_id, idx + 1, idx + 1, placement_for_elements)
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
            roof_ids.push(roof_id);
            layer_counter += 1;
        }

        // Write floor/slab elements
        for (idx, _) in schema.geometry.zones.iter().enumerate() {
            let slab_id = layer_counter;
            writeln!(self.output, "#{}=IFCBUILDINGELEMENTPROXY('0Exp0rtSl{{}}Gu1D00000000',#{},'Slab-{}','Slab-{}',$,#{},$,$,.FLOOR.);",
                slab_id, slab_id, idx + 1, idx + 1, placement_for_elements)
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
            slab_ids.push(slab_id);
            layer_counter += 1;
        }

        // Write material associations for walls
        if !wall_ids.is_empty() && wall_usage_id != 0 {
            let wall_refs: Vec<String> = wall_ids.iter().map(|id| format!("#{}", id)).collect();
            writeln!(self.output, "#4000=IFCRELASSOCIATESMATERIAL('0Exp0rtWMat000000000',#{},$,$,({}),#{});",
                owner_hist_id, wall_refs.join(","), wall_usage_id)
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        }

        // Write material associations for roofs
        if !roof_ids.is_empty() && roof_usage_id != 0 {
            let roof_refs: Vec<String> = roof_ids.iter().map(|id| format!("#{}", id)).collect();
            writeln!(self.output, "#4001=IFCRELASSOCIATESMATERIAL('0Exp0rtRMat000000000',#{},$,$,({}),#{});",
                owner_hist_id, roof_refs.join(","), roof_usage_id)
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        }

        // Write material associations for slabs
        if !slab_ids.is_empty() && floor_usage_id != 0 {
            let slab_refs: Vec<String> = slab_ids.iter().map(|id| format!("#{}", id)).collect();
            writeln!(self.output, "#4002=IFCRELASSOCIATESMATERIAL('0Exp0rtSMat000000000',#{},$,$,({}),#{});",
                owner_hist_id, slab_refs.join(","), floor_usage_id)
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        }

        // Write spatial containment for all elements
        let all_element_ids: Vec<String> = wall_ids.iter().chain(slab_ids.iter()).chain(roof_ids.iter())
            .map(|id| format!("#{}", id)).collect();
        if !all_element_ids.is_empty() && !space_ids.is_empty() {
            // Contain all elements in the first space for simplicity
            let space_ref = space_ids[0];
            writeln!(self.output, "#5000=IFCRELCONTAINEDINSPATIALSTRUCTURE('0Exp0rtRCont00000000',#{},$,$,({}),#{});",
                owner_hist_id, all_element_ids.join(","), space_ref)
                .map_err(|e| IfcError::conversion_error(e.to_string()))?;
        }

        writeln!(self.output, "ENDSEC;").map_err(|e| IfcError::conversion_error(e.to_string()))?;
        writeln!(self.output, "END-ISO-10303-21;").map_err(|e| IfcError::conversion_error(e.to_string()))?;

        Ok(())
    }
}

/// Escape a string for IFC format (handle quotes and special chars).
fn escape_ifc_string(s: &str) -> String {
    s.replace('\'', "''")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::schema::{
        ConstructionSet, ControlSet, Geometry, SchemaMetadata, SchemaVersion, SimulationOutput,
        SimulationSchemaV1, WeatherData, ZoneGeometry,
    };
    use crate::interop::ifc::parser::IfcParser;
    use crate::interop::ifc::IfcGeometryParser;

    fn create_test_schema() -> SimulationSchemaV1 {
        SimulationSchemaV1 {
            version: SchemaVersion::V1,
            metadata: SchemaMetadata {
                name: "Test Building".to_string(),
                description: "Test building for IFC export".to_string(),
                author: Some("Test".to_string()),
                created_at: Some("2026-01-01".to_string()),
                schema_version: SchemaVersion::V1,
            },
            geometry: Geometry {
                zones: vec![
                    ZoneGeometry {
                        name: "Zone 1".to_string(),
                        floor_area: 48.0,
                        volume: 129.6,
                        height: 2.7,
                    },
                    ZoneGeometry {
                        name: "Zone 2".to_string(),
                        floor_area: 36.0,
                        volume: 97.2,
                        height: 2.7,
                    },
                ],
                total_floor_area: 84.0,
                total_volume: 226.8,
                number_of_floors: 1,
                floor_height: 2.7,
            },
            constructions: ConstructionSet::default(),
            schedules: crate::api::schema::ScheduleSet::default(),
            weather: WeatherData::TmyLocation {
                location: "Denver, CO".to_string(),
            },
            controls: ControlSet::default(),
            output: SimulationOutput::default(),
        }
    }

    #[test]
    fn test_export_ifc_basic() {
        let schema = create_test_schema();
        let mut output = Vec::new();
        let mut writer = IfcWriter::new(&mut output);
        writer.write_schema(&schema).expect("should export");

        let content = String::from_utf8(output).expect("valid UTF-8");
        assert!(content.contains("ISO-10303-21"));
        assert!(content.contains("IFC4"));
        assert!(content.contains("IFCBUILDING"));
        assert!(content.contains("IFCSPACE"));
        assert!(content.contains("Test Building"));
    }

    #[test]
    fn test_export_contains_zones() {
        let schema = create_test_schema();
        let mut output = Vec::new();
        let mut writer = IfcWriter::new(&mut output);
        writer.write_schema(&schema).expect("should export");

        let content = String::from_utf8(output).expect("valid UTF-8");
        assert!(content.contains("Zone 1"));
        assert!(content.contains("Zone 2"));
    }

    #[test]
    fn test_export_roundtrip() {
        let schema = create_test_schema();
        let mut output = Vec::new();
        let mut writer = IfcWriter::new(&mut output);
        writer.write_schema(&schema).expect("should export");

        let content = String::from_utf8(output).expect("valid UTF-8");

        // Parse the exported IFC
        let model = IfcParser::from_str(&content).expect("should parse");
        let geometry = IfcGeometryParser::parse_model(&model);

        // Verify zone count matches
        assert_eq!(geometry.zones.len(), schema.geometry.zones.len());

        // Verify building exists
        assert!(!geometry.buildings.is_empty());
    }

    #[test]
    fn test_export_contains_materials() {
        let schema = create_test_schema();
        let mut output = Vec::new();
        let mut writer = IfcWriter::new(&mut output);
        writer.write_schema(&schema).expect("should export");

        let content = String::from_utf8(output).expect("valid UTF-8");
        assert!(content.contains("IFCMATERIAL"));
        assert!(content.contains("IFCMATERIALLAYER"));
        assert!(content.contains("IFCMATERIALLAYERSET"));
    }

    #[test]
    fn test_export_contains_building_elements() {
        let schema = create_test_schema();
        let mut output = Vec::new();
        let mut writer = IfcWriter::new(&mut output);
        writer.write_schema(&schema).expect("should export");

        let content = String::from_utf8(output).expect("valid UTF-8");
        assert!(content.contains("IFCBUILDINGELEMENTPROXY"));
    }

    #[test]
    fn test_escape_ifc_string() {
        assert_eq!(escape_ifc_string("Test"), "Test");
        assert_eq!(escape_ifc_string("It's a test"), "It''s a test");
        assert_eq!(escape_ifc_string("Quote'test"), "Quote''test");
    }

    #[test]
    fn test_export_with_custom_construction() {
        let mut schema = create_test_schema();
        schema.constructions.wall.layers = vec![
            ConstructionLayer::new("Concrete", 1.4, 2300.0, 880.0, 0.1),
            ConstructionLayer::new("Insulation", 0.04, 30.0, 840.0, 0.05),
        ];

        let mut output = Vec::new();
        let mut writer = IfcWriter::new(&mut output);
        writer.write_schema(&schema).expect("should export");

        let content = String::from_utf8(output).expect("valid UTF-8");
        assert!(content.contains("Concrete"));
        assert!(content.contains("Insulation"));
    }
}
