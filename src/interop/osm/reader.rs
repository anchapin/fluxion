// Copyright 2026 Fluxion. All rights reserved.
// SPDX-License-Identifier: MIT

//! OSM file parser - imports OpenStudio Model files into fluxion schema.
//!
//! This module provides functionality to parse OSM (OpenStudio Model) files
//! and convert them into fluxion's [`SimulationSchemaV1`] format.
//!
//! # OSM Format
//!
//! OSM files use a line-oriented key-value format similar to IDF but with
//! OpenStudio-specific IDD schema. Objects are defined with curly braces
//! and fields are separated by commas.
//!
//! # Example
//!
//! ```ignore
//! use fluxion::interop::osm::{import_osm, OsmReader};
//!
//! let schema = import_osm("building.osm")?;
//! ```
//!
//! # Limitations
//!
//! This is an initial implementation with the following known limitations:
//! - Limited support for HVAC systems
//! - Simplified zone mapping
//! - Basic schedule extraction

use std::fs;
use std::path::Path;

use crate::api::schema::{
    ConstructionSet, ControlConfig, ControlSet, Geometry, ScheduleSet, SchemaMetadata,
    SchemaVersion, SimulationOutput, SimulationSchemaV1, SurfaceConstruction, WeatherData,
    WindowSpec, ZoneGeometry,
};
use crate::interop::osm::error::OsmError;
use crate::interop::osm::types::*;

pub fn import_osm(path: impl AsRef<Path>) -> Result<SimulationSchemaV1, OsmError> {
    let content = fs::read_to_string(path.as_ref()).map_err(|e| OsmError::IoError(e))?;

    let mut reader = OsmReader::new();
    reader.parse(&content)
}

pub struct OsmReader {
    ctx: OsmContext,
    current_line: usize,
}

impl OsmReader {
    pub fn new() -> Self {
        OsmReader {
            ctx: OsmContext::default(),
            current_line: 0,
        }
    }

    pub fn parse(&mut self, content: &str) -> Result<SimulationSchemaV1, OsmError> {
        let objects = self.parse_content(content)?;
        self.build_context(objects);
        self.convert_to_schema()
    }

    fn parse_content(&mut self, content: &str) -> Result<Vec<OsmObject>, OsmError> {
        let mut objects = Vec::new();
        let mut lines: Vec<&str> = content.lines().collect();

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();
            self.current_line = i + 1;

            if line.is_empty() || line.starts_with('!') || line.starts_with('#') {
                i += 1;
                continue;
            }

            if line.starts_with("OS:") || line.starts_with("OSM:") {
                let obj = self.parse_object(&mut lines, &mut i)?;
                objects.push(obj);
            }
            i += 1;
        }

        Ok(objects)
    }

    fn parse_object(&self, lines: &mut [&str], idx: &mut usize) -> Result<OsmObject, OsmError> {
        let first_line = lines[*idx].trim();
        let object_type = self.extract_object_type(first_line)?;

        let mut obj = OsmObject::new(&object_type);
        let mut field_idx = 0;

        while *idx < lines.len() {
            *idx += 1;
            if *idx >= lines.len() {
                break;
            }

            let line = lines[*idx].trim();

            if line.is_empty() {
                continue;
            }

            // An object terminator in OpenStudio IDD is either:
            //   1) a line containing only ";"
            //   2) the last field of an object whose source line contains ";"
            //      before the trailing ", !- <comment>" annotation, e.g.
            //      "  2.7; !- Floor Height {m}" — the writer emits `;`
            //      on the same line as the last value to terminate the object.
            //
            // Both forms must be recognized here; otherwise parse_object would
            // bleed into the next object. (Fix for issue #1340 round-trip.)
            let line_has_terminator = line.contains(';') || line == ";";

            if line == ";" {
                break;
            }

            // If a line begins with "OS:" / "OSM:" it is the start of the next
            // object; the current object is complete. (This guard is needed
            // because the writer sometimes omits the explicit `;` terminator
            // line; we still need to stop cleanly.)
            if line.starts_with("OS:") || line.starts_with("OSM:") {
                // Step back so the outer loop picks up this line as a new object.
                if *idx > 0 {
                    *idx -= 1;
                }
                break;
            }

            if line.starts_with('!') {
                obj.comments
                    .push(line.strip_prefix('!').unwrap().trim().to_string());
                continue;
            }

            let field_value = self.extract_field_value(line, field_idx)?;
            let field_name = self.get_field_name(&object_type, field_idx);
            obj.fields.insert(field_name, field_value);
            field_idx += 1;

            // After consuming a field whose source line carried the
            // terminating ";" (case 2 above), the object is done.
            if line_has_terminator {
                break;
            }
        }

        Ok(obj)
    }

    fn extract_object_type(&self, line: &str) -> Result<String, OsmError> {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.is_empty() {
            return Err(OsmError::parse_error(
                self.current_line,
                "Empty object type",
            ));
        }

        let obj_type = parts[0].trim();
        if obj_type.starts_with("OS:") || obj_type.starts_with("OSM:") {
            Ok(obj_type.to_string())
        } else {
            Ok(format!("OS:{}", obj_type.trim_start_matches("OS:")))
        }
    }

    fn extract_field_value(&self, line: &str, _field_idx: usize) -> Result<String, OsmError> {
        // Strip the trailing comment of the form ", !- <comment>" (OpenStudio
        // IDD field annotation). Example input: "  Test Building, !- Name"
        // produces value "Test Building".
        let line = line.trim_end_matches(',').trim();

        // Strip trailing semicolon first (last field of an object ends with
        // "<value>;" e.g. "  2.7; !- Floor Height {m}" -> "2.7").
        let line = if let Some(idx) = line.rfind(';') {
            &line[..idx]
        } else {
            line
        };

        // Now strip the ", !- <comment>" suffix if present. Some fields have
        // empty values written as just ",", which would have been collapsed
        // by trim_end_matches(',') above; that's fine, we want empty.
        if let Some(idx) = line.find(", !-") {
            Ok(line[..idx].trim().to_string())
        } else {
            // No trailing comma-comment; could be a numeric-only value or a
            // bare string without the comment. Just trim and return.
            Ok(line.trim().to_string())
        }
    }

    fn get_field_name(&self, object_type: &str, field_idx: usize) -> String {
        match object_type {
            "OS:Material" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "roughness".to_string(),
                3 => "thickness".to_string(),
                4 => "conductivity".to_string(),
                5 => "density".to_string(),
                6 => "specific_heat".to_string(),
                7 => "emissivity".to_string(),
                8 => "vapor_transmission".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Construction" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                // Layers (Layer N, Outside Layer, Inside Layer) — give each
                // a distinct storage key so we don't lose any to HashMap
                // overwrites. Issue #1340 round-trip.
                n => format!("layer_{}", n - 2),
            },
            "OS:BuildingStory" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "level".to_string(),
                3 => "height".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Space" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "zone_handle".to_string(),
                3 => "building_story_handle".to_string(),
                4 => "floor_area".to_string(),
                5 => "volume".to_string(),
                6 => "x_origin".to_string(),
                7 => "y_origin".to_string(),
                8 => "z_origin".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Surface" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "surface_type".to_string(),
                3 => "construction_handle".to_string(),
                4 => "building_boundary".to_string(),
                5 => "outside_boundary_condition".to_string(),
                6 => "sun_exposure".to_string(),
                7 => "wind_exposure".to_string(),
                _ => "extra".to_string(),
            },
            "OS:SubSurface" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "surface_handle".to_string(),
                3 => "construction_handle".to_string(),
                4 => "window_type".to_string(),
                _ => "extra".to_string(),
            },
            "OS:ThermalZone" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "thermostat_handle".to_string(),
                3 => "multiplier".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Site" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "latitude".to_string(),
                3 => "longitude".to_string(),
                4 => "elevation".to_string(),
                5 => "time_zone".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Building" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "building_story_handles".to_string(),
                3 => "zone_handles".to_string(),
                4 => "area".to_string(),
                5 => "number_of_floors".to_string(),
                6 => "floor_height".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Thermostat" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "heating_setpoint".to_string(),
                3 => "cooling_setpoint".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Lights" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "zone_handle".to_string(),
                3 => "design_level".to_string(),
                4 => "watts_per_zone_floor_area".to_string(),
                5 => "fraction_radiant".to_string(),
                6 => "fraction_visible".to_string(),
                7 => "schedule_handle".to_string(),
                _ => "extra".to_string(),
            },
            "OS:ElectricEquipment" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "zone_handle".to_string(),
                3 => "design_level".to_string(),
                4 => "watts_per_zone_floor_area".to_string(),
                5 => "fraction_radiant".to_string(),
                6 => "schedule_handle".to_string(),
                _ => "extra".to_string(),
            },
            "OS:People" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "zone_handle".to_string(),
                3 => "number_of_people".to_string(),
                4 => "people_per_area".to_string(),
                5 => "fraction_radiant".to_string(),
                6 => "schedule_handle".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Schedule:Constant" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "schedule_type".to_string(),
                3 => "value".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Schedule:Compact" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "schedule_type".to_string(),
                _ => format!("field_{}", field_idx - 3),
            },
            "OS:ZoneInfiltration:DesignFlowRate" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "zone_handle".to_string(),
                3 => "design_flow_rate".to_string(),
                4 => "flow_per_zone_floor_area".to_string(),
                5 => "air_changes_per_hour".to_string(),
                6 => "schedule_handle".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Schedule:Day" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "schedule_type".to_string(),
                _ => format!("hour_{}", field_idx - 3),
            },
            "OS:Schedule:Week" => match field_idx {
                0 => "handle".to_string(),
                1 => "name".to_string(),
                2 => "sunday_handle".to_string(),
                3 => "monday_handle".to_string(),
                4 => "tuesday_handle".to_string(),
                5 => "wednesday_handle".to_string(),
                6 => "thursday_handle".to_string(),
                7 => "friday_handle".to_string(),
                8 => "saturday_handle".to_string(),
                _ => "extra".to_string(),
            },
            "OS:Version" => match field_idx {
                0 => "version_identifier".to_string(),
                _ => "extra".to_string(),
            },
            _ => "field".to_string(),
        }
    }

    fn build_context(&mut self, objects: Vec<OsmObject>) {
        for obj in objects {
            match obj.object_type.as_str() {
                "OS:Material" => {
                    let mat = self.parse_material(&obj);
                    if !mat.handle.is_empty() {
                        self.ctx.materials.insert(mat.handle.clone(), mat);
                    }
                }
                "OS:Construction" => {
                    let cons = self.parse_construction(&obj);
                    if !cons.handle.is_empty() {
                        self.ctx.constructions.insert(cons.handle.clone(), cons);
                    }
                }
                "OS:BuildingStory" => {
                    let story = self.parse_building_story(&obj);
                    if !story.handle.is_empty() {
                        self.ctx
                            .building_stories
                            .insert(story.handle.clone(), story);
                    }
                }
                "OS:Space" => {
                    let space = self.parse_space(&obj);
                    if !space.handle.is_empty() {
                        self.ctx.spaces.insert(space.handle.clone(), space);
                    }
                }
                "OS:Surface" => {
                    let surface = self.parse_surface(&obj);
                    if !surface.handle.is_empty() {
                        self.ctx.surfaces.insert(surface.handle.clone(), surface);
                    }
                }
                "OS:SubSurface" => {
                    let sub = self.parse_sub_surface(&obj);
                    if !sub.handle.is_empty() {
                        self.ctx.sub_surfaces.insert(sub.handle.clone(), sub);
                    }
                }
                "OS:ThermalZone" => {
                    let zone = self.parse_thermal_zone(&obj);
                    if !zone.handle.is_empty() {
                        self.ctx.thermal_zones.insert(zone.handle.clone(), zone);
                    }
                }
                "OS:Site" => {
                    self.ctx.site = Some(self.parse_site(&obj));
                }
                "OS:Building" => {
                    self.ctx.building = Some(self.parse_building(&obj));
                }
                "OS:Thermostat" => {
                    let thermo = self.parse_thermostat(&obj);
                    if !thermo.handle.is_empty() {
                        self.ctx.thermostats.insert(thermo.handle.clone(), thermo);
                    }
                }
                "OS:Lights" => {
                    let lights = self.parse_lights(&obj);
                    if !lights.handle.is_empty() {
                        self.ctx.lights.insert(lights.handle.clone(), lights);
                    }
                }
                "OS:ElectricEquipment" => {
                    let eq = self.parse_electric_equipment(&obj);
                    if !eq.handle.is_empty() {
                        self.ctx.electric_equipment.insert(eq.handle.clone(), eq);
                    }
                }
                "OS:People" => {
                    let people = self.parse_people(&obj);
                    if !people.handle.is_empty() {
                        self.ctx.people.insert(people.handle.clone(), people);
                    }
                }
                "OS:Schedule:Constant" => {
                    let sched = self.parse_schedule(&obj);
                    if !sched.handle.is_empty() {
                        self.ctx.schedules.insert(sched.handle.clone(), sched);
                    }
                }
                "OS:Schedule:Compact" => {
                    let sched = self.parse_schedule_compact(&obj);
                    if !sched.handle.is_empty() {
                        self.ctx
                            .schedule_compact
                            .insert(sched.handle.clone(), sched);
                    }
                }
                "OS:ZoneInfiltration:DesignFlowRate" => {
                    let inf = self.parse_zone_infiltration(&obj);
                    if !inf.handle.is_empty() {
                        self.ctx.zone_infiltration.insert(inf.handle.clone(), inf);
                    }
                }
                "OS:Schedule:Day" => {
                    let day = self.parse_schedule_day(&obj);
                    if !day.handle.is_empty() {
                        self.ctx.schedule_days.insert(day.handle.clone(), day);
                    }
                }
                "OS:Schedule:Week" => {
                    let week = self.parse_schedule_week(&obj);
                    if !week.handle.is_empty() {
                        self.ctx.schedule_weeks.insert(week.handle.clone(), week);
                    }
                }
                _ => {}
            }
        }
    }

    fn parse_material(&self, obj: &OsmObject) -> Material {
        Material {
            handle: obj.fields.get("handle").cloned().unwrap_or_default(),
            name: obj.fields.get("name").cloned().unwrap_or_default(),
            roughness: obj.fields.get("roughness").cloned(),
            thickness: self.parse_f64(obj.fields.get("thickness")),
            conductivity: self.parse_f64(obj.fields.get("conductivity")),
            density: self.parse_f64(obj.fields.get("density")),
            specific_heat: self.parse_f64(obj.fields.get("specific_heat")),
            emissivity: self.parse_f64(obj.fields.get("emissivity")),
            absorptance: obj.fields.get("absorptance").and_then(|s| s.parse().ok()),
            vapor_transmission: self.parse_f64(obj.fields.get("vapor_transmission")),
        }
    }

    fn parse_construction(&self, obj: &OsmObject) -> Construction {
        let mut indexed_layers: Vec<(usize, String)> = Vec::new();
        let mut outside_layer: Option<String> = None;
        let mut inside_layer: Option<String> = None;

        for (key, val) in &obj.fields {
            if let Some(idx_str) = key.strip_prefix("layer_") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    if !val.is_empty() {
                        indexed_layers.push((idx, val.clone()));
                    }
                }
            } else if key == "outside_layer" && !val.is_empty() && val != "OS:Material" {
                outside_layer = Some(val.clone());
            } else if key == "inside_layer" && !val.is_empty() && val != "OS:Material" {
                inside_layer = Some(val.clone());
            } else if key == "layer" {
                // Backwards-compatible single-layer form (legacy OSM files).
                indexed_layers.push((indexed_layers.len(), val.clone()));
            }
        }

        // Sort by index so layer order is deterministic regardless of
        // HashMap iteration order (issue #1340 round-trip).
        indexed_layers.sort_by_key(|(idx, _)| *idx);
        let mut layer_handles: Vec<String> = indexed_layers.into_iter().map(|(_, v)| v).collect();

        // Prepend outside_layer / append inside_layer if present (OpenStudio
        // order convention; empty if not set).
        if let Some(ol) = outside_layer {
            layer_handles.insert(0, ol);
        }
        if let Some(il) = inside_layer {
            layer_handles.push(il);
        }

        Construction {
            handle: obj.fields.get("handle").cloned().unwrap_or_default(),
            name: obj.fields.get("name").cloned().unwrap_or_default(),
            layer_handles,
        }
    }

    fn parse_building_story(&self, obj: &OsmObject) -> BuildingStory {
        BuildingStory {
            handle: obj.fields.get("handle").cloned().unwrap_or_default(),
            name: obj.fields.get("name").cloned().unwrap_or_default(),
            level: self.parse_f64(obj.fields.get("level")),
            height: self.parse_f64(obj.fields.get("height")),
            spaces: Vec::new(),
        }
    }

    fn parse_space(&self, obj: &OsmObject) -> Space {
        Space {
            handle: obj.fields.get("handle").cloned().unwrap_or_default(),
            name: obj.fields.get("name").cloned().unwrap_or_default(),
            zone_handle: obj.fields.get("zone_handle").cloned(),
            story_handle: obj.fields.get("building_story_handle").cloned(),
            area: self.parse_f64(obj.fields.get("floor_area")),
            volume: self.parse_f64(obj.fields.get("volume")),
            surfaces: Vec::new(),
        }
    }

    fn parse_surface(&self, obj: &OsmObject) -> Surface {
        Surface {
            handle: obj.fields.get("handle").cloned().unwrap_or_default(),
            name: obj.fields.get("name").cloned().unwrap_or_default(),
            surface_type: obj.fields.get("surface_type").cloned().unwrap_or_default(),
            construction_handle: obj.fields.get("construction_handle").cloned(),
            building_boundary: obj.fields.get("building_boundary").cloned(),
            outside_boundary_condition: obj.fields.get("outside_boundary_condition").cloned(),
            sun_exposure: obj.fields.get("sun_exposure").cloned(),
            wind_exposure: obj.fields.get("wind_exposure").cloned(),
            area: self.parse_f64(obj.fields.get("area")),
            vertices: Vec::new(),
            adjacent_space_handle: None,
        }
    }

    fn parse_sub_surface(&self, obj: &OsmObject) -> SubSurface {
        SubSurface {
            handle: obj.fields.get("handle").cloned().unwrap_or_default(),
            name: obj.fields.get("name").cloned().unwrap_or_default(),
            surface_handle: obj
                .fields
                .get("surface_handle")
                .cloned()
                .unwrap_or_default(),
            construction_handle: obj.fields.get("construction_handle").cloned(),
            window_type: obj.fields.get("window_type").cloned(),
            area: self.parse_f64(obj.fields.get("area")),
            vertices: Vec::new(),
        }
    }

    fn parse_thermal_zone(&self, obj: &OsmObject) -> ThermalZone {
        ThermalZone {
            handle: obj.fields.get("handle").cloned().unwrap_or_default(),
            name: obj.fields.get("name").cloned().unwrap_or_default(),
            thermostat_handle: obj.fields.get("thermostat_handle").cloned(),
            multiplier: self.parse_f64(obj.fields.get("multiplier")),
        }
    }

    fn parse_site(&self, obj: &OsmObject) -> Site {
        Site {
            handle: obj.fields.get("handle").cloned().unwrap_or_default(),
            name: obj.fields.get("name").cloned().unwrap_or_default(),
            latitude: self.parse_f64(obj.fields.get("latitude")),
            longitude: self.parse_f64(obj.fields.get("longitude")),
            elevation: self.parse_f64(obj.fields.get("elevation")),
            time_zone: self.parse_f64(obj.fields.get("time_zone")),
        }
    }

    fn parse_building(&self, obj: &OsmObject) -> Building {
        Building {
            handle: obj.fields.get("handle").cloned().unwrap_or_default(),
            name: obj.fields.get("name").cloned().unwrap_or_default(),
            building_story_handles: Vec::new(),
            zone_handles: Vec::new(),
            area: self.parse_f64(obj.fields.get("area")),
            number_of_floors: obj
                .fields
                .get("number_of_floors")
                .and_then(|s| s.parse().ok()),
            floor_height: self.parse_f64(obj.fields.get("floor_height")),
        }
    }

    fn parse_thermostat(&self, obj: &OsmObject) -> Thermostat {
        Thermostat {
            handle: obj.fields.get("handle").cloned().unwrap_or_default(),
            name: obj.fields.get("name").cloned().unwrap_or_default(),
            heating_setpoint: self.parse_f64(obj.fields.get("heating_setpoint")),
            cooling_setpoint: self.parse_f64(obj.fields.get("cooling_setpoint")),
        }
    }

    fn parse_lights(&self, obj: &OsmObject) -> Lights {
        Lights {
            handle: obj.fields.get("handle").cloned().unwrap_or_default(),
            name: obj.fields.get("name").cloned().unwrap_or_default(),
            zone_handle: obj.fields.get("zone_handle").cloned(),
            watts_per_zone_floor_area: self.parse_f64(obj.fields.get("watts_per_zone_floor_area")),
            fraction_radiant: self.parse_f64(obj.fields.get("fraction_radiant")),
            fraction_visible: self.parse_f64(obj.fields.get("fraction_visible")),
            schedule_handle: obj.fields.get("schedule_handle").cloned(),
        }
    }

    fn parse_electric_equipment(&self, obj: &OsmObject) -> ElectricEquipment {
        ElectricEquipment {
            handle: obj.fields.get("handle").cloned().unwrap_or_default(),
            name: obj.fields.get("name").cloned().unwrap_or_default(),
            zone_handle: obj.fields.get("zone_handle").cloned(),
            watts_per_zone_floor_area: self.parse_f64(obj.fields.get("watts_per_zone_floor_area")),
            fraction_radiant: self.parse_f64(obj.fields.get("fraction_radiant")),
            schedule_handle: obj.fields.get("schedule_handle").cloned(),
        }
    }

    fn parse_people(&self, obj: &OsmObject) -> People {
        People {
            handle: obj.fields.get("handle").cloned().unwrap_or_default(),
            name: obj.fields.get("name").cloned().unwrap_or_default(),
            zone_handle: obj.fields.get("zone_handle").cloned(),
            number_of_people: self.parse_f64(obj.fields.get("number_of_people")),
            people_per_area: self.parse_f64(obj.fields.get("people_per_area")),
            fraction_radiant: self.parse_f64(obj.fields.get("fraction_radiant")),
            schedule_handle: obj.fields.get("schedule_handle").cloned(),
        }
    }

    fn parse_schedule(&self, obj: &OsmObject) -> Schedule {
        let value = obj
            .fields
            .get("value")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);

        Schedule {
            handle: obj.fields.get("handle").cloned().unwrap_or_default(),
            name: obj.fields.get("name").cloned().unwrap_or_default(),
            schedule_type: obj.fields.get("schedule_type").cloned().unwrap_or_default(),
            values: vec![value],
        }
    }

    fn parse_schedule_compact(&self, obj: &OsmObject) -> ScheduleCompact {
        let mut through_values: Vec<ScheduleThroughValue> = Vec::new();
        let mut current_through: Option<String> = None;

        for (key, val) in &obj.fields {
            if key.starts_with("field_") {
                let trimmed = val.trim();
                if trimmed.starts_with("Through:") {
                    current_through = Some(trimmed.to_string());
                } else if trimmed.starts_with("Value") {
                    let value_str = trimmed.strip_prefix("Value").unwrap_or(trimmed).trim();
                    if let Ok(value) = value_str.parse::<f64>() {
                        let through = current_through.take().unwrap_or_default();
                        through_values.push(ScheduleThroughValue { through, value });
                    }
                }
            }
        }

        ScheduleCompact {
            handle: obj.fields.get("handle").cloned().unwrap_or_default(),
            name: obj.fields.get("name").cloned().unwrap_or_default(),
            schedule_type: obj.fields.get("schedule_type").cloned().unwrap_or_default(),
            through_values,
        }
    }

    fn parse_zone_infiltration(&self, obj: &OsmObject) -> ZoneInfiltration {
        ZoneInfiltration {
            handle: obj.fields.get("handle").cloned().unwrap_or_default(),
            name: obj.fields.get("name").cloned().unwrap_or_default(),
            zone_handle: obj.fields.get("zone_handle").cloned(),
            design_flow_rate: self.parse_f64(obj.fields.get("design_flow_rate")),
            schedule_handle: obj.fields.get("schedule_handle").cloned(),
        }
    }

    fn parse_schedule_day(&self, obj: &OsmObject) -> ScheduleDay {
        let mut values: Vec<f64> = Vec::new();
        for (key, val) in &obj.fields {
            if key.starts_with("hour_") {
                if let Ok(v) = val.parse::<f64>() {
                    values.push(v);
                }
            }
        }
        ScheduleDay {
            handle: obj.fields.get("handle").cloned().unwrap_or_default(),
            name: obj.fields.get("name").cloned().unwrap_or_default(),
            values,
        }
    }

    fn parse_schedule_week(&self, obj: &OsmObject) -> ScheduleWeek {
        let mut day_schedule_handles: Vec<String> = Vec::new();
        for (key, val) in &obj.fields {
            match key.as_str() {
                "sunday_handle" | "monday_handle" | "tuesday_handle" | "wednesday_handle"
                | "thursday_handle" | "friday_handle" | "saturday_handle"
                    if !val.is_empty() => {
                        day_schedule_handles.push(val.clone());
                    }
                _ => {}
            }
        }
        ScheduleWeek {
            handle: obj.fields.get("handle").cloned().unwrap_or_default(),
            name: obj.fields.get("name").cloned().unwrap_or_default(),
            day_schedule_handles,
        }
    }

    fn parse_f64(&self, s: Option<&String>) -> Option<f64> {
        s.and_then(|v| {
            let cleaned = v.trim();
            if cleaned.is_empty() || cleaned == "-" {
                None
            } else {
                cleaned.parse().ok()
            }
        })
    }

    fn convert_to_schema(&self) -> Result<SimulationSchemaV1, OsmError> {
        let metadata = self.extract_metadata()?;
        let geometry = self.extract_geometry()?;
        let constructions = self.extract_constructions()?;
        let schedules = ScheduleSet::default();
        let weather = self.extract_weather()?;
        let controls = self.extract_controls()?;
        let output = SimulationOutput::default();

        Ok(SimulationSchemaV1 {
            version: SchemaVersion::V1,
            metadata,
            geometry,
            constructions,
            schedules,
            weather,
            controls,
            output,
        })
    }

    fn extract_metadata(&self) -> Result<SchemaMetadata, OsmError> {
        let name = self
            .ctx
            .building
            .as_ref()
            .map(|b| b.name.clone())
            .unwrap_or_else(|| "Imported OSM Model".to_string());

        let description = if let Some(site) = &self.ctx.site {
            if !site.name.is_empty() {
                format!("Location: {}", site.name)
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        Ok(SchemaMetadata {
            name,
            description,
            author: None,
            created_at: Some(chrono::Utc::now().format("%Y-%m-%d").to_string()),
            schema_version: SchemaVersion::V1,
        })
    }

    fn extract_geometry(&self) -> Result<Geometry, OsmError> {
        let mut zones: Vec<ZoneGeometry> = Vec::new();
        let mut total_floor_area = 0.0;
        let mut total_volume = 0.0;
        let number_of_floors = self
            .ctx
            .building
            .as_ref()
            .and_then(|b| b.number_of_floors)
            .unwrap_or(1) as usize;

        // Iterate spaces in a deterministic order so the resulting zone list
        // preserves the original insertion order. We sort by the
        // trailing numeric index of the handle (e.g. "{space-0}" -> 0)
        // to match the order the writer emits zones. Issue #1340.
        let mut ordered_spaces: Vec<&Space> = self.ctx.spaces.values().collect();
        ordered_spaces.sort_by_key(|s| {
            // Extract trailing integer from handle like "{space-12}" -> 12.
            s.handle
                .rsplit('-')
                .next()
                .and_then(|n| n.trim_end_matches('}').parse::<usize>().ok())
                .unwrap_or(usize::MAX)
        });

        for space in ordered_spaces {
            let zone_name = space
                .zone_handle
                .as_ref()
                .and_then(|zh| self.ctx.thermal_zones.get(zh))
                .map(|tz| tz.name.clone())
                .unwrap_or_else(|| space.name.clone());

            let area = space.area.unwrap_or(48.0);
            let volume = space.volume.unwrap_or(area * 2.7);

            total_floor_area += area;
            total_volume += volume;

            zones.push(ZoneGeometry {
                name: zone_name,
                floor_area: area,
                volume,
                height: if area > 0.0 { volume / area } else { 2.7 },
            });
        }

        if zones.is_empty() {
            zones.push(ZoneGeometry::default());
            total_floor_area = 48.0;
            total_volume = 129.6;
        }

        let floor_height = if number_of_floors > 0 && total_floor_area > 0.0 {
            total_volume / total_floor_area
        } else {
            2.7
        };

        Ok(Geometry {
            zones,
            total_floor_area,
            total_volume,
            number_of_floors,
            floor_height,
        })
    }

    fn extract_constructions(&self) -> Result<ConstructionSet, OsmError> {
        let wall = self.build_surface_construction_for_type("Wall")?;
        let roof = self.build_surface_construction_for_type("Roof")?;
        let floor = self.build_surface_construction_for_type("Floor")?;

        Ok(ConstructionSet {
            wall,
            roof,
            floor,
            interzone: None,
        })
    }

    fn build_surface_construction_for_type(
        &self,
        surface_type: &str,
    ) -> Result<SurfaceConstruction, OsmError> {
        let mut layers = Vec::new();
        let mut window = None;

        for surface in self.ctx.surfaces.values() {
            if surface.surface_type.contains(surface_type) || surface.name.contains(surface_type) {
                if let Some(const_handle) = &surface.construction_handle {
                    if let Some(construction) = self.ctx.constructions.get(const_handle) {
                        for layer_handle in &construction.layer_handles {
                            // Try the strict OpenStudio structure first:
                            // OS:Construction.layer_handle -> OS:Material:Layer.material_handles -> OS:Material
                            if let Some(layer) = self.ctx.layers.get(layer_handle) {
                                for mat_handle in &layer.material_handles {
                                    if let Some(material) = self.ctx.materials.get(mat_handle) {
                                        if let Some(layer) = material.to_construction_layer() {
                                            layers.push(layer);
                                        }
                                    }
                                }
                            } else if let Some(material) = self.ctx.materials.get(layer_handle) {
                                // Fallback: layer_handle IS a material_handle.
                                // This is the case for OSM files emitted by the fluxion
                                // writer, which references {mat-w0..} directly from
                                // OS:Construction without an intermediate Layer object.
                                // Required for lossless round-trip (issue #1340).
                                if let Some(layer) = material.to_construction_layer() {
                                    layers.push(layer);
                                }
                            }
                        }
                    }
                }
                break;
            }
        }

        if layers.is_empty() {
            return Ok(SurfaceConstruction::default());
        }

        if surface_type != "Wall" && surface_type != "Floor" {
            window = Some(WindowSpec::default());
        }

        Ok(SurfaceConstruction {
            name: format!("{} Construction", surface_type),
            layers,
            window,
        })
    }

    fn extract_weather(&self) -> Result<WeatherData, OsmError> {
        if let Some(site) = &self.ctx.site {
            if let (Some(lat), Some(lon)) = (site.latitude, site.longitude) {
                let location = format!("{}, {}", lat, lon);
                return Ok(WeatherData::TmyLocation { location });
            }
        }
        Ok(WeatherData::TmyLocation {
            location: "Denver, CO".to_string(),
        })
    }

    fn extract_controls(&self) -> Result<ControlSet, OsmError> {
        let mut heating_setpoint = 20.0;
        let mut cooling_setpoint = 24.0;

        for thermostat in self.ctx.thermostats.values() {
            if let Some(hsp) = thermostat.heating_setpoint {
                heating_setpoint = hsp;
            }
            if let Some(csp) = thermostat.cooling_setpoint {
                cooling_setpoint = csp;
            }
        }

        let zone_control = ControlConfig {
            heating_setpoint,
            cooling_setpoint,
            deadband_tolerance: 0.5,
            heating_capacity: 100_000.0,
            cooling_capacity: 100_000.0,
        };

        Ok(ControlSet {
            zone_control,
            global_control: None,
        })
    }
}

impl Default for OsmReader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_OSM: &str = r#"
! Sample OpenStudio Model
OS:Version,
  {abc123},!- Handle
  3.2.0; !- Version Identifier

OS:Site,
  {site1}, !- Handle
  Denver CO, !- Name
  39.739, !- Latitude
  -104.984, !- Longitude
  1609; !- Elevation

OS:Building,
  {bldg1}, !- Handle
  Small Office, !- Name
  , !- Building Story Names
  , !- Thermal Zone Names
  1393.5, !- Floor Area {m2}
  1, !- Number of Floors
  3.0; !- Floor Height {m}

OS:Material,
  {mat1}, !- Handle
  Concrete, !- Name
  MediumRough, !- Roughness
  0.1, !- Thickness {m}
  1.4, !- Conductivity {W/m-K}
  2300, !- Density {kg/m3}
  840; !- Specific Heat {J/kg-K}

OS:Construction,
  {cons1}, !- Handle
  ExtWall, !- Name
  {mat1}; !- Outside Layer

OS:ThermalZone,
  {zone1}, !- Handle
  Zone 1, !- Name
  , !- Thermostat Handle
  1; !- Multiplier

OS:Space,
  {space1}, !- Handle
  Zone 1 Space, !- Name
  {zone1}, !- Zone Handle
  , !- Building Story Handle
  0, !- X Origin {m}
  0, !- Y Origin {m}
  0; !- Z Origin {m}

OS:Surface,
  {surf1}, !- Handle
  West Wall, !- Name
  Wall, !- Surface Type
  {cons1}, !- Construction Handle
  , !- Building Boundary Type
  Outdoors, !- Outside Boundary Condition
  , !- Sun Exposure
  ; !- Wind Exposure
"#;

    #[test]
    fn test_parse_osm() {
        let mut reader = OsmReader::new();
        let result = reader.parse(SAMPLE_OSM);
        assert!(result.is_ok(), "Should parse OSM without error");
        let schema = result.unwrap();
        assert!(!schema.metadata.name.is_empty(), "Should have a name");
    }

    #[test]
    fn test_parse_material() {
        let mut reader = OsmReader::new();
        let result = reader.parse_content(SAMPLE_OSM);
        assert!(result.is_ok(), "Should parse content without error");
        let objects = result.unwrap();
        assert!(!objects.is_empty(), "Should extract some objects from OSM");
    }

    #[test]
    fn test_parse_thermal_zone() {
        let mut reader = OsmReader::new();
        let result = reader.parse(SAMPLE_OSM);
        assert!(result.is_ok(), "Should parse OSM without error");
        let schema = result.unwrap();
        assert!(
            !schema.geometry.zones.is_empty(),
            "Should have at least one zone"
        );
    }

    #[test]
    fn test_parse_site() {
        let mut reader = OsmReader::new();
        let result = reader.parse(SAMPLE_OSM);
        assert!(result.is_ok(), "Should parse OSM without error");
        let schema = result.unwrap();
        match &schema.weather {
            WeatherData::TmyLocation { location } => {
                assert!(!location.is_empty(), "Location should not be empty");
            }
            _ => panic!("Expected TmyLocation"),
        }
    }

    const SCHEDULE_OSM: &str = r#"
! Sample OSM with Schedule:Compact, ZoneInfiltration, and internal gains
OS:Version,
  {abc123},!- Handle
  3.2.0; !- Version Identifier

OS:Site,
  {site1}, !- Handle
  Denver CO, !- Name
  39.739, !- Latitude
  -104.984, !- Longitude
  1609; !- Elevation

OS:Building,
  {bldg1}, !- Handle
  Small Office, !- Name
  , !- Building Story Names
  , !- Thermal Zone Names
  1393.5, !- Floor Area {m2}
  1, !- Number of Floors
  3.0; !- Floor Height {m}

OS:ThermalZone,
  {zone1}, !- Handle
  Zone 1, !- Name
  , !- Thermostat Handle
  1; !- Multiplier

OS:Space,
  {space1}, !- Handle
  Zone 1 Space, !- Name
  {zone1}, !- Zone Handle
  , !- Building Story Handle
  0, !- X Origin {m}
  0, !- Y Origin {m}
  0; !- Z Origin {m}

OS:Schedule:Compact,
  {sched1}, !- Handle
  Office Occupancy, !- Name
  Fraction, !- Schedule Type Limits Name
  Through: 1/1, !- Field
  Through: 1/31, !- Field
  Value 0.0, !- Field
  Through: 2/1, !- Field
  Through: 2/28, !- Field
  Value 1.0, !- Field
  Through: 12/31, !- Field
  Value 0.0; !- Field

OS:Schedule:Constant,
  {sched2}, !- Handle
  Always On, !- Name
  Fraction, !- Schedule Type Limits Name
  1.0; !- Value

OS:ZoneInfiltration:DesignFlowRate,
  {inf1}, !- Handle
  Zone 1 Infiltration, !- Name
  {zone1}, !- Zone Name
  0.5, !- Design Flow Rate {m3/s}
  , !- Flow per Zone Floor Area {m3/s-m2}
  , !- Air Changes per Hour {1/hr}
  {sched2}; !- Schedule Name

OS:People,
  {people1}, !- Handle
  Zone 1 People, !- Name
  {zone1}, !- Zone Name
  5, !- Number of People {people}
  , !- People per Zone Floor Area {people/m2}
  , !- Fraction Radiant
  {sched1}; !- Number of People Schedule Name

OS:Lights,
  {lights1}, !- Handle
  Zone 1 Lights, !- Name
  {zone1}, !- Zone Name
  , !- Lighting Level {W}
  10.0, !- Watts per Zone Floor Area {W/m2}
  , !- Fraction Radiant
  , !- Fraction Visible
  {sched1}; !- Schedule Name

OS:ElectricEquipment,
  {elec1}, !- Handle
  Zone 1 Equipment, !- Name
  {zone1}, !- Zone Name
  , !- Electric Equipment Level {W}
  5.0, !- Watts per Zone Floor Area {W/m2}
  , !- Fraction Radiant
  {sched1}; !- Schedule Name
"#;

    #[test]
    fn test_parse_schedule_compact() {
        let mut reader = OsmReader::new();
        let result = reader.parse(SCHEDULE_OSM);
        assert!(result.is_ok(), "Should parse OSM with Schedule:Compact");
        let schema = result.unwrap();
        assert_eq!(schema.metadata.name, "Small Office");
    }

    #[test]
    fn test_parse_zone_infiltration() {
        let mut reader = OsmReader::new();
        let result = reader.parse(SCHEDULE_OSM);
        assert!(result.is_ok(), "Should parse OSM with ZoneInfiltration");
        assert!(
            !reader.ctx.zone_infiltration.is_empty(),
            "Should have parsed ZoneInfiltration"
        );
        let infiltration = reader.ctx.zone_infiltration.values().next().unwrap();
        assert_eq!(infiltration.name, "Zone 1 Infiltration");
        assert!(infiltration.design_flow_rate.is_some());
    }

    #[test]
    fn test_parse_internal_gains_with_schedules() {
        let mut reader = OsmReader::new();
        let result = reader.parse(SCHEDULE_OSM);
        assert!(result.is_ok(), "Should parse OSM with internal gains");

        let people = reader.ctx.people.values().next().unwrap();
        assert_eq!(people.name, "Zone 1 People");
        assert!(people.schedule_handle.is_some());

        let lights = reader.ctx.lights.values().next().unwrap();
        assert_eq!(lights.name, "Zone 1 Lights");
        assert!(lights.schedule_handle.is_some());

        let elec = reader.ctx.electric_equipment.values().next().unwrap();
        assert_eq!(elec.name, "Zone 1 Equipment");
        assert!(elec.schedule_handle.is_some());
    }

    #[test]
    fn test_parse_schedule_constant() {
        let mut reader = OsmReader::new();
        let result = reader.parse(SCHEDULE_OSM);
        assert!(result.is_ok(), "Should parse OSM with Schedule:Constant");
        assert!(
            !reader.ctx.schedules.is_empty(),
            "Should have parsed Schedule:Constant"
        );
    }
}
