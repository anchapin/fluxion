use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildingGeometry {
    pub id: String,
    pub name: String,
    pub levels: Vec<BuildingLevel>,
    pub zones: Vec<ThermalZone>,
    pub total_floor_area: f64,
    pub bounding_box: BoundingBox,
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
pub struct Space {
    pub id: String,
    pub name: String,
    pub level_id: String,
    pub thermal_zone_id: Option<String>,
    pub floor_area: f64,
    pub volume: f64,
    pub surfaces: Vec<Surface>,
    pub openings: Vec<Opening>,
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
pub struct Surface {
    pub id: String,
    pub name: String,
    pub surface_type: SurfaceType,
    pub level_id: String,
    pub space_id: String,
    pub thermal_zone_id: Option<String>,
    pub area: f64,
    pub vertices: Vec<Vertex>,
    pub normal: Vertex,
    pub construction_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SurfaceType {
    Wall,
    Floor,
    Roof,
    Ceiling,
    Internal,
    Ground,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Opening {
    pub id: String,
    pub name: String,
    pub opening_type: OpeningType,
    pub surface_id: String,
    pub area: f64,
    pub vertices: Vec<Vertex>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OpeningType {
    Window,
    Door,
    Skylight,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vertex {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vertex {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn distance(&self, other: &Vertex) -> f64 {
        ((self.x - other.x).powi(2)
            + (self.y - other.y).powi(2)
            + (self.z - other.z).powi(2))
        .sqrt()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min: Vertex,
    pub max: Vertex,
}

impl BoundingBox {
    pub fn from_vertices(vertices: &[Vertex]) -> Self {
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut min_z = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        let mut max_z = f64::MIN;

        for v in vertices {
            min_x = min_x.min(v.x);
            min_y = min_y.min(v.y);
            min_z = min_z.min(v.z);
            max_x = max_x.max(v.x);
            max_y = max_y.max(v.y);
            max_z = max_z.max(v.z);
        }

        Self {
            min: Vertex::new(min_x, min_y, min_z),
            max: Vertex::new(max_x, max_y, max_z),
        }
    }

    pub fn center(&self) -> Vertex {
        Vertex::new(
            (self.min.x + self.max.x) / 2.0,
            (self.min.y + self.max.y) / 2.0,
            (self.min.z + self.max.z) / 2.0,
        )
    }

    pub fn size(&self) -> Vertex {
        Vertex::new(
            self.max.x - self.min.x,
            self.max.y - self.min.y,
            self.max.z - self.min.z,
        )
    }
}

pub fn create_sample_geometry() -> BuildingGeometry {
    let level1_id = Uuid::new_v4().to_string();
    let level2_id = Uuid::new_v4().to_string();

    let zone1_id = Uuid::new_v4().to_string();
    let zone2_id = Uuid::new_v4().to_string();

    let space1_id = Uuid::new_v4().to_string();
    let space2_id = Uuid::new_v4().to_string();
    let space3_id = Uuid::new_v4().to_string();
    let space4_id = Uuid::new_v4().to_string();

    BuildingGeometry {
        id: Uuid::new_v4().to_string(),
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
                        surfaces: create_zone_surfaces(&space1_id, &level1_id, &zone1_id, 0.0, 0.0, 10.0, 5.0),
                        openings: vec![],
                    },
                    Space {
                        id: space2_id.clone(),
                        name: "Office 102".to_string(),
                        level_id: level1_id.clone(),
                        thermal_zone_id: Some(zone1_id.clone()),
                        floor_area: 50.0,
                        volume: 150.0,
                        surfaces: create_zone_surfaces(&space2_id, &level1_id, &zone1_id, 10.0, 0.0, 10.0, 5.0),
                        openings: vec![],
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
                        surfaces: create_zone_surfaces(&space3_id, &level2_id, &zone2_id, 0.0, 0.0, 10.0, 5.0),
                        openings: vec![],
                    },
                    Space {
                        id: space4_id.clone(),
                        name: "Office 202".to_string(),
                        level_id: level2_id.clone(),
                        thermal_zone_id: Some(zone2_id.clone()),
                        floor_area: 50.0,
                        volume: 150.0,
                        surfaces: create_zone_surfaces(&space4_id, &level2_id, &zone2_id, 10.0, 0.0, 10.0, 5.0),
                        openings: vec![],
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
            min: Vertex::new(0.0, 0.0, 0.0),
            max: Vertex::new(20.0, 5.0, 6.0),
        },
    }
}

fn create_zone_surfaces(
    space_id: &str,
    level_id: &str,
    zone_id: &str,
    offset_x: f64,
    offset_y: f64,
    width: f64,
    depth: f64,
) -> Vec<Surface> {
    let height = 3.0;
    let level_elevation = if level_id.contains("2") { 3.0 } else { 0.0 };

    vec![
        // Floor
        Surface {
            id: Uuid::new_v4().to_string(),
            name: format!("{} Floor", space_id),
            surface_type: SurfaceType::Floor,
            level_id: level_id.to_string(),
            space_id: space_id.to_string(),
            thermal_zone_id: Some(zone_id.to_string()),
            area: width * depth,
            vertices: vec![
                Vertex::new(offset_x, offset_y, level_elevation),
                Vertex::new(offset_x + width, offset_y, level_elevation),
                Vertex::new(offset_x + width, offset_y + depth, level_elevation),
                Vertex::new(offset_x, offset_y + depth, level_elevation),
            ],
            normal: Vertex::new(0.0, 0.0, -1.0),
            construction_id: None,
        },
        // Ceiling
        Surface {
            id: Uuid::new_v4().to_string(),
            name: format!("{} Ceiling", space_id),
            surface_type: SurfaceType::Ceiling,
            level_id: level_id.to_string(),
            space_id: space_id.to_string(),
            thermal_zone_id: Some(zone_id.to_string()),
            area: width * depth,
            vertices: vec![
                Vertex::new(offset_x, offset_y, level_elevation + height),
                Vertex::new(offset_x, offset_y + depth, level_elevation + height),
                Vertex::new(offset_x + width, offset_y + depth, level_elevation + height),
                Vertex::new(offset_x + width, offset_y, level_elevation + height),
            ],
            normal: Vertex::new(0.0, 0.0, 1.0),
            construction_id: None,
        },
        // South Wall
        Surface {
            id: Uuid::new_v4().to_string(),
            name: format!("{} South Wall", space_id),
            surface_type: SurfaceType::Wall,
            level_id: level_id.to_string(),
            space_id: space_id.to_string(),
            thermal_zone_id: Some(zone_id.to_string()),
            area: width * height,
            vertices: vec![
                Vertex::new(offset_x, offset_y, level_elevation),
                Vertex::new(offset_x + width, offset_y, level_elevation),
                Vertex::new(offset_x + width, offset_y, level_elevation + height),
                Vertex::new(offset_x, offset_y, level_elevation + height),
            ],
            normal: Vertex::new(0.0, -1.0, 0.0),
            construction_id: None,
        },
        // North Wall
        Surface {
            id: Uuid::new_v4().to_string(),
            name: format!("{} North Wall", space_id),
            surface_type: SurfaceType::Wall,
            level_id: level_id.to_string(),
            space_id: space_id.to_string(),
            thermal_zone_id: Some(zone_id.to_string()),
            area: width * height,
            vertices: vec![
                Vertex::new(offset_x, offset_y + depth, level_elevation),
                Vertex::new(offset_x, offset_y + depth, level_elevation + height),
                Vertex::new(offset_x + width, offset_y + depth, level_elevation + height),
                Vertex::new(offset_x + width, offset_y + depth, level_elevation),
            ],
            normal: Vertex::new(0.0, 1.0, 0.0),
            construction_id: None,
        },
        // East Wall
        Surface {
            id: Uuid::new_v4().to_string(),
            name: format!("{} East Wall", space_id),
            surface_type: SurfaceType::Wall,
            level_id: level_id.to_string(),
            space_id: space_id.to_string(),
            thermal_zone_id: Some(zone_id.to_string()),
            area: depth * height,
            vertices: vec![
                Vertex::new(offset_x + width, offset_y, level_elevation),
                Vertex::new(offset_x + width, offset_y + depth, level_elevation),
                Vertex::new(offset_x + width, offset_y + depth, level_elevation + height),
                Vertex::new(offset_x + width, offset_y, level_elevation + height),
            ],
            normal: Vertex::new(1.0, 0.0, 0.0),
            construction_id: None,
        },
        // West Wall
        Surface {
            id: Uuid::new_v4().to_string(),
            name: format!("{} West Wall", space_id),
            surface_type: SurfaceType::Wall,
            level_id: level_id.to_string(),
            space_id: space_id.to_string(),
            thermal_zone_id: Some(zone_id.to_string()),
            area: depth * height,
            vertices: vec![
                Vertex::new(offset_x, offset_y, level_elevation),
                Vertex::new(offset_x, offset_y, level_elevation + height),
                Vertex::new(offset_x, offset_y + depth, level_elevation + height),
                Vertex::new(offset_x, offset_y + depth, level_elevation),
            ],
            normal: Vertex::new(-1.0, 0.0, 0.0),
            construction_id: None,
        },
    ]
}

pub fn parse_gbxml_content(content: &str) -> Result<BuildingGeometry, String> {
    Ok(create_sample_geometry())
}

pub fn parse_ifc_content(content: &str) -> Result<BuildingGeometry, String> {
    Ok(create_sample_geometry())
}
