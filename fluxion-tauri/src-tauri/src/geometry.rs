use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vertex {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min: Vertex,
    pub max: Vertex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Surface {
    pub id: String,
    pub name: String,
    pub surface_type: String,
    pub level_id: String,
    pub space_id: String,
    pub thermal_zone_id: Option<String>,
    pub area: f64,
    pub vertices: Vec<Vertex>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Space {
    pub id: String,
    pub name: String,
    pub level_id: String,
    pub thermal_zone_id: Option<String>,
    pub floor_area: f64,
    pub volume: f64,
    pub surfaces: Vec<Surface>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingLevel {
    pub id: String,
    pub name: String,
    pub elevation: f64,
    pub height: f64,
    pub spaces: Vec<Space>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalZone {
    pub id: String,
    pub name: String,
    pub level_id: String,
    pub space_ids: Vec<String>,
    pub floor_area: f64,
    pub volume: f64,
    pub lighting_load: f64,
    pub equipment_load: f64,
    pub occupancy: f64,
    pub ventilation_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingGeometry {
    pub id: String,
    pub name: String,
    pub levels: Vec<BuildingLevel>,
    pub zones: Vec<ThermalZone>,
    pub total_floor_area: f64,
    pub bounding_box: BoundingBox,
}

pub fn create_sample_geometry() -> BuildingGeometry {
    let level1_id = uuid::Uuid::new_v4().to_string();
    let level2_id = uuid::Uuid::new_v4().to_string();
    let zone1_id = uuid::Uuid::new_v4().to_string();
    let zone2_id = uuid::Uuid::new_v4().to_string();
    let space1_id = uuid::Uuid::new_v4().to_string();
    let space2_id = uuid::Uuid::new_v4().to_string();
    let space3_id = uuid::Uuid::new_v4().to_string();
    let space4_id = uuid::Uuid::new_v4().to_string();

    BuildingGeometry {
        id: uuid::Uuid::new_v4().to_string(),
        name: "Sample Office Building".to_string(),
        levels: vec![
            BuildingLevel {
                id: level1_id.clone(),
                name: "Level 1".to_string(),
                elevation: 0.0,
                height: 3.0,
                spaces: vec![
                    Space {
                        id: space1_id.clone(),
                        name: "Office 101".to_string(),
                        level_id: level1_id.clone(),
                        thermal_zone_id: Some(zone1_id.clone()),
                        floor_area: 50.0,
                        volume: 150.0,
                        surfaces: create_surfaces(
                            &space1_id, &level1_id, &zone1_id, 0.0, 0.0, 10.0, 5.0, 0.0,
                        ),
                    },
                    Space {
                        id: space2_id.clone(),
                        name: "Office 102".to_string(),
                        level_id: level1_id.clone(),
                        thermal_zone_id: Some(zone1_id.clone()),
                        floor_area: 50.0,
                        volume: 150.0,
                        surfaces: create_surfaces(
                            &space2_id, &level1_id, &zone1_id, 10.0, 0.0, 10.0, 5.0, 0.0,
                        ),
                    },
                ],
            },
            BuildingLevel {
                id: level2_id.clone(),
                name: "Level 2".to_string(),
                elevation: 3.0,
                height: 3.0,
                spaces: vec![
                    Space {
                        id: space3_id.clone(),
                        name: "Office 201".to_string(),
                        level_id: level2_id.clone(),
                        thermal_zone_id: Some(zone2_id.clone()),
                        floor_area: 50.0,
                        volume: 150.0,
                        surfaces: create_surfaces(
                            &space3_id, &level2_id, &zone2_id, 0.0, 0.0, 10.0, 5.0, 3.0,
                        ),
                    },
                    Space {
                        id: space4_id.clone(),
                        name: "Office 202".to_string(),
                        level_id: level2_id.clone(),
                        thermal_zone_id: Some(zone2_id.clone()),
                        floor_area: 50.0,
                        volume: 150.0,
                        surfaces: create_surfaces(
                            &space4_id, &level2_id, &zone2_id, 10.0, 0.0, 10.0, 5.0, 3.0,
                        ),
                    },
                ],
            },
        ],
        zones: vec![
            ThermalZone {
                id: zone1_id.clone(),
                name: "Zone 1 - Level 1".to_string(),
                level_id: level1_id.clone(),
                space_ids: vec![space1_id.clone(), space2_id.clone()],
                floor_area: 100.0,
                volume: 300.0,
                lighting_load: 1200.0,
                equipment_load: 1500.0,
                occupancy: 8.0,
                ventilation_rate: 0.5,
            },
            ThermalZone {
                id: zone2_id.clone(),
                name: "Zone 2 - Level 2".to_string(),
                level_id: level2_id.clone(),
                space_ids: vec![space3_id.clone(), space4_id.clone()],
                floor_area: 100.0,
                volume: 300.0,
                lighting_load: 1200.0,
                equipment_load: 1500.0,
                occupancy: 8.0,
                ventilation_rate: 0.5,
            },
        ],
        total_floor_area: 200.0,
        bounding_box: BoundingBox {
            min: Vertex {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            max: Vertex {
                x: 20.0,
                y: 5.0,
                z: 6.0,
            },
        },
    }
}

fn create_surfaces(
    space_id: &str,
    level_id: &str,
    zone_id: &str,
    offset_x: f64,
    offset_y: f64,
    width: f64,
    depth: f64,
    level_elevation: f64,
) -> Vec<Surface> {
    let height = 3.0;

    vec![
        Surface {
            id: uuid::Uuid::new_v4().to_string(),
            name: format!("{} Floor", space_id),
            surface_type: "Floor".to_string(),
            level_id: level_id.to_string(),
            space_id: space_id.to_string(),
            thermal_zone_id: Some(zone_id.to_string()),
            area: width * depth,
            vertices: vec![
                Vertex {
                    x: offset_x,
                    y: offset_y,
                    z: level_elevation,
                },
                Vertex {
                    x: offset_x + width,
                    y: offset_y,
                    z: level_elevation,
                },
                Vertex {
                    x: offset_x + width,
                    y: offset_y + depth,
                    z: level_elevation,
                },
                Vertex {
                    x: offset_x,
                    y: offset_y + depth,
                    z: level_elevation,
                },
            ],
        },
        Surface {
            id: uuid::Uuid::new_v4().to_string(),
            name: format!("{} Ceiling", space_id),
            surface_type: "Ceiling".to_string(),
            level_id: level_id.to_string(),
            space_id: space_id.to_string(),
            thermal_zone_id: Some(zone_id.to_string()),
            area: width * depth,
            vertices: vec![
                Vertex {
                    x: offset_x,
                    y: offset_y,
                    z: level_elevation + height,
                },
                Vertex {
                    x: offset_x,
                    y: offset_y + depth,
                    z: level_elevation + height,
                },
                Vertex {
                    x: offset_x + width,
                    y: offset_y + depth,
                    z: level_elevation + height,
                },
                Vertex {
                    x: offset_x + width,
                    y: offset_y,
                    z: level_elevation + height,
                },
            ],
        },
        Surface {
            id: uuid::Uuid::new_v4().to_string(),
            name: format!("{} South Wall", space_id),
            surface_type: "Wall".to_string(),
            level_id: level_id.to_string(),
            space_id: space_id.to_string(),
            thermal_zone_id: Some(zone_id.to_string()),
            area: width * height,
            vertices: vec![
                Vertex {
                    x: offset_x,
                    y: offset_y,
                    z: level_elevation,
                },
                Vertex {
                    x: offset_x + width,
                    y: offset_y,
                    z: level_elevation,
                },
                Vertex {
                    x: offset_x + width,
                    y: offset_y,
                    z: level_elevation + height,
                },
                Vertex {
                    x: offset_x,
                    y: offset_y,
                    z: level_elevation + height,
                },
            ],
        },
        Surface {
            id: uuid::Uuid::new_v4().to_string(),
            name: format!("{} North Wall", space_id),
            surface_type: "Wall".to_string(),
            level_id: level_id.to_string(),
            space_id: space_id.to_string(),
            thermal_zone_id: Some(zone_id.to_string()),
            area: width * height,
            vertices: vec![
                Vertex {
                    x: offset_x,
                    y: offset_y + depth,
                    z: level_elevation,
                },
                Vertex {
                    x: offset_x,
                    y: offset_y + depth,
                    z: level_elevation + height,
                },
                Vertex {
                    x: offset_x + width,
                    y: offset_y + depth,
                    z: level_elevation + height,
                },
                Vertex {
                    x: offset_x + width,
                    y: offset_y + depth,
                    z: level_elevation,
                },
            ],
        },
        Surface {
            id: uuid::Uuid::new_v4().to_string(),
            name: format!("{} East Wall", space_id),
            surface_type: "Wall".to_string(),
            level_id: level_id.to_string(),
            space_id: space_id.to_string(),
            thermal_zone_id: Some(zone_id.to_string()),
            area: depth * height,
            vertices: vec![
                Vertex {
                    x: offset_x + width,
                    y: offset_y,
                    z: level_elevation,
                },
                Vertex {
                    x: offset_x + width,
                    y: offset_y + depth,
                    z: level_elevation,
                },
                Vertex {
                    x: offset_x + width,
                    y: offset_y + depth,
                    z: level_elevation + height,
                },
                Vertex {
                    x: offset_x + width,
                    y: offset_y,
                    z: level_elevation + height,
                },
            ],
        },
        Surface {
            id: uuid::Uuid::new_v4().to_string(),
            name: format!("{} West Wall", space_id),
            surface_type: "Wall".to_string(),
            level_id: level_id.to_string(),
            space_id: space_id.to_string(),
            thermal_zone_id: Some(zone_id.to_string()),
            area: depth * height,
            vertices: vec![
                Vertex {
                    x: offset_x,
                    y: offset_y,
                    z: level_elevation,
                },
                Vertex {
                    x: offset_x,
                    y: offset_y,
                    z: level_elevation + height,
                },
                Vertex {
                    x: offset_x,
                    y: offset_y + depth,
                    z: level_elevation + height,
                },
                Vertex {
                    x: offset_x,
                    y: offset_y + depth,
                    z: level_elevation,
                },
            ],
        },
    ]
}

pub fn parse_gbxml_content(_content: &str) -> Result<BuildingGeometry, String> {
    Ok(create_sample_geometry())
}

pub fn parse_ifc_content(_content: &str) -> Result<BuildingGeometry, String> {
    Ok(create_sample_geometry())
}
