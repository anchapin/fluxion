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
    pub vertices: Vec<Vertex>,
    pub normal: Vertex,
    pub area: f64,
    pub surface_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Space {
    pub id: String,
    pub name: String,
    pub surfaces: Vec<Surface>,
    pub bounding_box: BoundingBox,
    pub zone_id: Option<String>,
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
    pub setpoint_heating: Option<f64>,
    pub setpoint_cooling: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingGeometry {
    pub id: String,
    pub name: String,
    pub levels: Vec<BuildingLevel>,
    pub zones: Vec<ThermalZone>,
    pub bounding_box: BoundingBox,
}

impl BuildingGeometry {
    pub fn sample() -> Self {
        let ground_level = BuildingLevel {
            id: "level-1".to_string(),
            name: "Ground Floor".to_string(),
            elevation: 0.0,
            height: 3.0,
            spaces: vec![
                Space {
                    id: "space-1".to_string(),
                    name: "Office A".to_string(),
                    surfaces: vec![
                        Surface {
                            id: "surface-1".to_string(),
                            vertices: vec![
                                Vertex { x: 0.0, y: 0.0, z: 0.0 },
                                Vertex { x: 4.0, y: 0.0, z: 0.0 },
                                Vertex { x: 4.0, y: 0.0, z: 3.0 },
                                Vertex { x: 0.0, y: 0.0, z: 3.0 },
                            ],
                            normal: Vertex { x: 0.0, y: -1.0, z: 0.0 },
                            area: 12.0,
                            surface_type: "Wall".to_string(),
                        },
                    ],
                    bounding_box: BoundingBox {
                        min: Vertex { x: 0.0, y: 0.0, z: 0.0 },
                        max: Vertex { x: 4.0, y: 4.0, z: 3.0 },
                    },
                    zone_id: Some("zone-1".to_string()),
                },
            ],
        };

        BuildingGeometry {
            id: "building-1".to_string(),
            name: "Sample Building".to_string(),
            levels: vec![ground_level],
            zones: vec![ThermalZone {
                id: "zone-1".to_string(),
                name: "Office Zone".to_string(),
                level_id: "level-1".to_string(),
                space_ids: vec!["space-1".to_string()],
                setpoint_heating: Some(20.0),
                setpoint_cooling: Some(24.0),
            }],
            bounding_box: BoundingBox {
                min: Vertex { x: 0.0, y: 0.0, z: 0.0 },
                max: Vertex { x: 10.0, y: 10.0, z: 3.0 },
            },
        }
    }
}
