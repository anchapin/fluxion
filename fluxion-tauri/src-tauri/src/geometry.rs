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

fn vertex(x: f64, y: f64, z: f64) -> Vertex {
    Vertex { x, y, z }
}

/// Rectangular surface from four corners listed counter-clockwise when viewed
/// from outside (i.e. from the direction the normal points towards).
fn rect_surface(id: &str, surface_type: &str, corners: [Vertex; 4], normal: Vertex) -> Surface {
    // Adjacent edges from corner 0 span the full parallelogram, whose area
    // equals the polygon area for the rectangles produced by `box_space`.
    let area = {
        let (a, b, d) = (&corners[0], &corners[1], &corners[3]);
        let e1 = (b.x - a.x, b.y - a.y, b.z - a.z);
        let e2 = (d.x - a.x, d.y - a.y, d.z - a.z);
        let cross = (
            e1.1 * e2.2 - e1.2 * e2.1,
            e1.2 * e2.0 - e1.0 * e2.2,
            e1.0 * e2.1 - e1.1 * e2.0,
        );
        (cross.0 * cross.0 + cross.1 * cross.1 + cross.2 * cross.2).sqrt()
    };
    Surface {
        id: id.to_string(),
        vertices: corners.to_vec(),
        normal,
        area,
        surface_type: surface_type.to_string(),
    }
}

/// A closed rectangular box space (floor + 4 walls + roof) spanning
/// `min` to `max`, assigned to `zone_id`.
fn box_space(
    id: &str,
    name: &str,
    zone_id: &str,
    min: (f64, f64, f64),
    max: (f64, f64, f64),
) -> Space {
    let (x0, y0, z0) = min;
    let (x1, y1, z1) = max;
    let surfaces = vec![
        // Floor (outward normal -z)
        rect_surface(
            &format!("{id}-floor"),
            "Floor",
            [
                vertex(x0, y0, z0),
                vertex(x0, y1, z0),
                vertex(x1, y1, z0),
                vertex(x1, y0, z0),
            ],
            vertex(0.0, 0.0, -1.0),
        ),
        // Roof (outward normal +z)
        rect_surface(
            &format!("{id}-roof"),
            "Roof",
            [
                vertex(x0, y0, z1),
                vertex(x1, y0, z1),
                vertex(x1, y1, z1),
                vertex(x0, y1, z1),
            ],
            vertex(0.0, 0.0, 1.0),
        ),
        // South wall, y = y0 (outward normal -y)
        rect_surface(
            &format!("{id}-wall-south"),
            "Wall",
            [
                vertex(x0, y0, z0),
                vertex(x1, y0, z0),
                vertex(x1, y0, z1),
                vertex(x0, y0, z1),
            ],
            vertex(0.0, -1.0, 0.0),
        ),
        // North wall, y = y1 (outward normal +y)
        rect_surface(
            &format!("{id}-wall-north"),
            "Wall",
            [
                vertex(x1, y1, z0),
                vertex(x0, y1, z0),
                vertex(x0, y1, z1),
                vertex(x1, y1, z1),
            ],
            vertex(0.0, 1.0, 0.0),
        ),
        // West wall, x = x0 (outward normal -x)
        rect_surface(
            &format!("{id}-wall-west"),
            "Wall",
            [
                vertex(x0, y1, z0),
                vertex(x0, y0, z0),
                vertex(x0, y0, z1),
                vertex(x0, y1, z1),
            ],
            vertex(-1.0, 0.0, 0.0),
        ),
        // East wall, x = x1 (outward normal +x)
        rect_surface(
            &format!("{id}-wall-east"),
            "Wall",
            [
                vertex(x1, y0, z0),
                vertex(x1, y1, z0),
                vertex(x1, y1, z1),
                vertex(x1, y0, z1),
            ],
            vertex(1.0, 0.0, 0.0),
        ),
    ];

    Space {
        id: id.to_string(),
        name: name.to_string(),
        surfaces,
        bounding_box: BoundingBox {
            min: vertex(x0, y0, z0),
            max: vertex(x1, y1, z1),
        },
        zone_id: Some(zone_id.to_string()),
    }
}

impl BuildingGeometry {
    /// Two-level sample building with three closed spaces and three thermal
    /// zones. Consumed by the Tauri `load_geometry` command and mirrored by
    /// the web-mode fixture in `fluxion-tauri/frontend/src/lib/sampleGeometry.ts`.
    pub fn sample() -> Self {
        let ground = BuildingLevel {
            id: "level-1".to_string(),
            name: "Ground Floor".to_string(),
            elevation: 0.0,
            height: 3.0,
            spaces: vec![
                box_space(
                    "space-1",
                    "Office A",
                    "zone-1",
                    (0.0, 0.0, 0.0),
                    (8.0, 6.0, 3.0),
                ),
                box_space(
                    "space-2",
                    "Office B",
                    "zone-2",
                    (8.0, 0.0, 0.0),
                    (14.0, 6.0, 3.0),
                ),
            ],
        };
        let first = BuildingLevel {
            id: "level-2".to_string(),
            name: "First Floor".to_string(),
            elevation: 3.0,
            height: 3.0,
            spaces: vec![box_space(
                "space-3",
                "Meeting Suite",
                "zone-3",
                (0.0, 0.0, 3.0),
                (8.0, 6.0, 6.0),
            )],
        };

        BuildingGeometry {
            id: "building-1".to_string(),
            name: "Fluxion Sample Building".to_string(),
            levels: vec![ground, first],
            zones: vec![
                ThermalZone {
                    id: "zone-1".to_string(),
                    name: "Office Zone A".to_string(),
                    level_id: "level-1".to_string(),
                    space_ids: vec!["space-1".to_string()],
                    setpoint_heating: Some(20.0),
                    setpoint_cooling: Some(24.0),
                },
                ThermalZone {
                    id: "zone-2".to_string(),
                    name: "Office Zone B".to_string(),
                    level_id: "level-1".to_string(),
                    space_ids: vec!["space-2".to_string()],
                    setpoint_heating: Some(20.0),
                    setpoint_cooling: Some(24.0),
                },
                ThermalZone {
                    id: "zone-3".to_string(),
                    name: "Meeting Zone".to_string(),
                    level_id: "level-2".to_string(),
                    space_ids: vec!["space-3".to_string()],
                    setpoint_heating: Some(19.0),
                    setpoint_cooling: Some(25.0),
                },
            ],
            bounding_box: BoundingBox {
                min: vertex(0.0, 0.0, 0.0),
                max: vertex(14.0, 6.0, 6.0),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Newell's method normal for a planar polygon.
    fn newell_normal(vertices: &[Vertex]) -> (f64, f64, f64) {
        let mut nx = 0.0;
        let mut ny = 0.0;
        let mut nz = 0.0;
        for (i, v) in vertices.iter().enumerate() {
            let w = &vertices[(i + 1) % vertices.len()];
            nx += (v.y - w.y) * (v.z + w.z);
            ny += (v.z - w.z) * (v.x + w.x);
            nz += (v.x - w.x) * (v.y + w.y);
        }
        (nx, ny, nz)
    }

    fn sample_spaces() -> Vec<Space> {
        let sample = BuildingGeometry::sample();
        sample
            .levels
            .iter()
            .flat_map(|l| l.spaces.iter().cloned())
            .collect()
    }

    #[test]
    fn sample_spaces_are_closed_boxes() {
        let spaces = sample_spaces();
        assert_eq!(spaces.len(), 3, "expected three sample spaces");
        for space in &spaces {
            assert_eq!(space.surfaces.len(), 6, "space {} not closed", space.id);
            let floors = space
                .surfaces
                .iter()
                .filter(|s| s.surface_type == "Floor")
                .count();
            let roofs = space
                .surfaces
                .iter()
                .filter(|s| s.surface_type == "Roof")
                .count();
            let walls = space
                .surfaces
                .iter()
                .filter(|s| s.surface_type == "Wall")
                .count();
            assert_eq!((floors, roofs, walls), (1, 1, 4), "space {}", space.id);
            assert!(space.zone_id.is_some(), "space {} unzoned", space.id);
        }
    }

    #[test]
    fn declared_normals_match_polygon_geometry() {
        for space in sample_spaces() {
            for surface in &space.surfaces {
                assert_eq!(surface.vertices.len(), 4, "surface {}", surface.id);
                let (nx, ny, nz) = newell_normal(&surface.vertices);
                let len = (nx * nx + ny * ny + nz * nz).sqrt();
                assert!(len > 1e-9, "surface {} degenerate", surface.id);
                // Declared unit normal must match the polygon's Newell normal.
                let dot =
                    (nx * surface.normal.x + ny * surface.normal.y + nz * surface.normal.z) / len;
                assert!(
                    (dot - 1.0).abs() < 1e-9,
                    "surface {} normal mismatch (dot={dot})",
                    surface.id
                );
                // Winding must be counter-clockwise seen from outside.
                assert!(dot > 0.0, "surface {} wound inside-out", surface.id);
            }
        }
    }

    #[test]
    fn declared_areas_match_polygon_areas() {
        for space in sample_spaces() {
            for surface in &space.surfaces {
                let (nx, ny, nz) = newell_normal(&surface.vertices);
                let poly_area = 0.5 * (nx * nx + ny * ny + nz * nz).sqrt();
                assert!(
                    (poly_area - surface.area).abs() < 1e-9,
                    "surface {} area {} != polygon area {poly_area}",
                    surface.id,
                    surface.area
                );
            }
        }
    }

    #[test]
    fn bounding_boxes_enclose_all_vertices() {
        let sample = BuildingGeometry::sample();
        for level in &sample.levels {
            for space in &level.spaces {
                for surface in &space.surfaces {
                    for v in &surface.vertices {
                        assert!(v.x >= space.bounding_box.min.x - 1e-9);
                        assert!(v.y >= space.bounding_box.min.y - 1e-9);
                        assert!(v.z >= space.bounding_box.min.z - 1e-9);
                        assert!(v.x <= space.bounding_box.max.x + 1e-9);
                        assert!(v.y <= space.bounding_box.max.y + 1e-9);
                        assert!(v.z <= space.bounding_box.max.z + 1e-9);
                    }
                }
            }
        }
        // Building bbox covers the full footprint across both levels.
        assert_eq!(sample.bounding_box.max.x, 14.0);
        assert_eq!(sample.bounding_box.max.y, 6.0);
        assert_eq!(sample.bounding_box.max.z, 6.0);
    }

    /// Pins the JSON contract consumed by the R3F frontend: field names and
    /// the vertex/surface/zone object shapes must stay stable.
    #[test]
    fn serde_json_contract_matches_frontend_expectations() {
        let json = serde_json::to_value(BuildingGeometry::sample()).unwrap();
        assert_eq!(json["id"], "building-1");
        assert_eq!(json["name"], "Fluxion Sample Building");

        let levels = json["levels"].as_array().unwrap();
        assert_eq!(levels.len(), 2);
        let space = &levels[0]["spaces"][0];
        assert_eq!(space["id"], "space-1");
        let surface = &space["surfaces"][0];
        for key in ["id", "vertices", "normal", "area", "surface_type"] {
            assert!(surface.get(key).is_some(), "surface missing {key}");
        }
        let v = &surface["vertices"][0];
        assert_eq!(
            v.as_object().unwrap().keys().collect::<Vec<_>>(),
            vec!["x", "y", "z"]
        );

        let zones = json["zones"].as_array().unwrap();
        assert_eq!(zones.len(), 3);
        for key in ["id", "name", "level_id", "space_ids"] {
            assert!(zones[0].get(key).is_some(), "zone missing {key}");
        }
        assert_eq!(zones[0]["space_ids"][0], "space-1");
    }
}
