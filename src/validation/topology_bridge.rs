//! ASHRAE 140 `CaseSpec` → topology-graph bridge (Issue #3963).
//!
//! Converts a registry case specification into the thermal-network topology
//! defined in [`crate::topology`]. The bridge lives in `validation/`
//! (which may import from `sim`) so the cycle rule
//! `scripts/check_ashrae_cases_cycle.py` keeps holding: `sim` never imports
//! `validation`.
//!
//! Physics conventions (documented, spec-level — this is an introspection
//! export, not a solver path):
//!
//! * Exterior film coefficient `18.3 W/m²K` — the canonical fluxion value
//!   (`fluxion-core/src/construction.rs`).
//! * Interior film coefficient `8.29 W/m²K` (uniform across surface tilts);
//!   the ASHRAE 140 BESTEST canonical `h_in` for vertical walls. Per-tilt
//!   refinement is intentionally not applied at the topology level.
//! * Air node capacitance uses ρ = `1.2 kg/m³`, c_p = `1006 J/kgK`.
//! * `wall_layer` nodes carry the full layer capacitance `A·d·ρ·cp`; the two
//!   conduction edges flanking a layer each carry half the layer resistance
//!   (centered-capacitance RC discretization), so the series sum of a
//!   one-layer wall reproduces `A·k/d`.
//! * Windows are subtracted from the gross area of their host wall; glazing
//! * `internal_mass` is the zone radiant-mass star node (5R1C `T_m`
//!   analogue). Its runtime capacitance is computed inside the thermal
//!   solvers and is deliberately not fabricated here (`null`).

use std::collections::{BTreeMap, BTreeSet};

use crate::topology::{
    ToTopologyGraph, TopologyContext, TopologyEdge, TopologyEdgeKind, TopologyGraph, TopologyNode,
    TopologyNodeKind,
};

use crate::validation::ashrae_140_cases::{CaseSpec, CommonWall};
use fluxion_core::ashrae_cases::{GeometrySpec, Orientation, WindowArea};
use fluxion_core::construction::Construction;

/// Canonical exterior film coefficient (W/m²K).
pub const EXTERIOR_FILM_COEFF_W_M2K: f64 = 18.3;
/// Uniform interior film coefficient (W/m²K) — ASHRAE 140 BESTEST `h_in`.
pub const INTERIOR_FILM_COEFF_W_M2K: f64 = 8.29;
/// Reference air density (kg/m³) for air-node capacitance reporting.
pub const AIR_DENSITY_KG_M3: f64 = 1.2;
/// Reference air specific heat (J/kg·K).
pub const AIR_SPECIFIC_HEAT_J_KG_K: f64 = 1006.0;

/// Compass azimuth (degrees clockwise from North) for a wall orientation.
fn azimuth_deg(orientation: &Orientation) -> Option<f64> {
    match orientation {
        Orientation::North => Some(0.0),
        Orientation::East => Some(90.0),
        Orientation::South => Some(180.0),
        Orientation::West => Some(270.0),
        _ => None,
    }
}

/// Stable surface key for a wall orientation (used in node ids).
fn wall_key(orientation: &Orientation) -> &'static str {
    match orientation {
        Orientation::North => "wall-north",
        Orientation::East => "wall-east",
        Orientation::South => "wall-south",
        Orientation::West => "wall-west",
        Orientation::Up | Orientation::Horizontal => "roof",
        _ => "floor",
    }
}

/// Returns solar distribution factors for beam (direct) radiation based on
/// window orientation. For each `(surface_suffix, fraction)` pair, the full
/// surface key is `zone-{zi}:{suffix}:int`.
///
/// Fractions follow projected irradiated area for ASHRAE 140 Case 600 geometry:
/// - SOUTH beam: floor 50%, north 25%, east 5%, west 10%, south 10%
/// - EAST/WEST beam: floor 45%, north+south+west 15% each, host wall 10%
/// - NORTH beam: floor 100% (beam never hits north directly)
fn solar_beam_factors(
    orientation: &Orientation,
    hosts: &BTreeSet<String>,
    zi: usize,
) -> Vec<(String, f64)> {
    let floor_key = format!("zone-{}:floor:int", zi);
    let north_key = format!("zone-{}:wall-north:int", zi);
    let south_key = format!("zone-{}:wall-south:int", zi);
    let east_key = format!("zone-{}:wall-east:int", zi);
    let west_key = format!("zone-{}:wall-west:int", zi);

    let has_floor = hosts.contains(&floor_key);
    let has_north = hosts.contains(&north_key);
    let has_south = hosts.contains(&south_key);
    let has_east = hosts.contains(&east_key);
    let has_west = hosts.contains(&west_key);

    match orientation {
        Orientation::South => {
            let mut factors = Vec::new();
            if has_floor {
                factors.push((floor_key, 0.50)); // 50%
            }
            if has_north {
                factors.push((north_key, 0.25)); // 25%
            }
            if has_south {
                factors.push((south_key, 0.10)); // 10%
            }
            if has_east {
                factors.push((east_key, 0.05)); // 5%
            }
            if has_west {
                factors.push((west_key, 0.10)); // 10%
            }
            factors
        }
        Orientation::East | Orientation::West => {
            let mut factors = Vec::new();
            let host_key = if has_south {
                south_key.clone()
            } else {
                north_key.clone()
            };
            if has_floor {
                factors.push((floor_key, 0.45)); // 45%
            }
            if has_north {
                factors.push((north_key, 0.15));
            }
            if has_south {
                factors.push((south_key, 0.15));
            }
            if has_west {
                factors.push((west_key, 0.15));
            }
            // Host wall (north for east-facing, south for west-facing) gets 10%
            if hosts.contains(&host_key) {
                factors.push((host_key, 0.10));
            }
            factors
        }
        Orientation::North => {
            // Beam never hits north directly, 100% to floor
            if has_floor {
                vec![(floor_key, 1.0)]
            } else {
                vec![]
            }
        }
        _ => {
            // Default: all to floor if available
            if has_floor {
                vec![(floor_key, 1.0)]
            } else {
                vec![]
            }
        }
    }
}

/// Returns solar distribution factors for diffuse radiation. Diffuse radiation
/// is isotropic from the sky dome, so the distribution is the same for all
/// window orientations:
/// - floor 40%, north 15%, east 15%, south 15%, west 15%
fn solar_diffuse_factors(hosts: &BTreeSet<String>, zi: usize) -> Vec<(String, f64)> {
    let floor_key = format!("zone-{}:floor:int", zi);
    let north_key = format!("zone-{}:wall-north:int", zi);
    let south_key = format!("zone-{}:wall-south:int", zi);
    let east_key = format!("zone-{}:wall-east:int", zi);
    let west_key = format!("zone-{}:wall-west:int", zi);

    let has_floor = hosts.contains(&floor_key);
    let has_north = hosts.contains(&north_key);
    let has_south = hosts.contains(&south_key);
    let has_east = hosts.contains(&east_key);
    let has_west = hosts.contains(&west_key);

    let mut factors = Vec::new();
    if has_floor {
        factors.push((floor_key, 0.40)); // 40%
    }
    if has_north {
        factors.push((north_key, 0.15)); // 15%
    }
    if has_south {
        factors.push((south_key, 0.15)); // 15%
    }
    if has_east {
        factors.push((east_key, 0.15)); // 15%
    }
    if has_west {
        factors.push((west_key, 0.15)); // 15%
    }
    factors
}

fn attr<K: Into<String>>(key: K, value: f64) -> BTreeMap<String, f64> {
    let mut m = BTreeMap::new();
    m.insert(key.into(), value);
    m
}

/// Adds "common_wall" -> 1.0 to an existing attributes BTreeMap
fn with_common_wall(attrs: BTreeMap<String, f64>) -> BTreeMap<String, f64> {
    let mut result = attrs;
    result.insert("common_wall".to_string(), 1.0);
    result
}

/// Zone display name from geometry.
fn zone_name(zi: usize, g: &GeometrySpec) -> String {
    g.name.clone().unwrap_or_else(|| format!("Zone {zi}"))
}

/// Gross opaque area of one orientation's wall before window subtraction.
fn gross_wall_area(g: &GeometrySpec, orientation: &Orientation) -> f64 {
    match orientation {
        // North/South walls span the width (X-axis) of the zone
        Orientation::North | Orientation::South => g.width * g.height,
        // East/West walls span the depth (Y-axis) of the zone
        Orientation::East | Orientation::West => g.depth * g.height,
        Orientation::Up | Orientation::Down | Orientation::Horizontal => g.width * g.depth,
    }
}

/// Builds and returns the surface node chain for one construction assembly.
// A builder for one assembly needs the zone, surface, geometry, construction,
// and boundary context; bundling them into a struct would obscure the call
// sites, so the arity is accepted here.
#[allow(clippy::too_many_arguments)]
fn push_surface_chain(
    graph: &mut TopologyGraph,
    zi: usize,
    zname: &str,
    surface_key: &str,
    area: f64,
    azimuth: Option<f64>,
    tilt: f64,
    construction: &Construction,
    ground_coupled: Option<f64>,
) -> String {
    let prefix = format!("zone-{zi}:{surface_key}");
    let zone_ref = Some(format!("zone-{zi}:air"));

    let mut ext_attrs = BTreeMap::new();
    if let Some(t_ground) = ground_coupled {
        ext_attrs.insert("ground_coupled".to_string(), 1.0);
        ext_attrs.insert("ground_temperature_c".to_string(), t_ground);
    }
    graph.push_node(TopologyNode {
        id: format!("{prefix}:ext"),
        kind: TopologyNodeKind::ExteriorSurface,
        zone_id: zone_ref.clone(),
        name: format!("{zname} {surface_key} exterior film"),
        capacitance_j_per_k: None,
        area_m2: Some(area),
        volume_m3: None,
        azimuth_deg: azimuth,
        tilt_deg: Some(tilt),
        elevation_m: None,
        attributes: ext_attrs,
    });

    // Material layer nodes: interior (index 0) → exterior (last).
    for (j, layer) in construction.layers.iter().enumerate() {
        let mut attrs = BTreeMap::new();
        attrs.insert("thickness_m".to_string(), layer.thickness);
        attrs.insert("conductivity_w_per_mk".to_string(), layer.conductivity);
        attrs.insert("density_kg_per_m3".to_string(), layer.density);
        attrs.insert("specific_heat_j_per_kgk".to_string(), layer.specific_heat);
        graph.push_node(TopologyNode {
            id: format!("{prefix}:layer-{j}"),
            kind: TopologyNodeKind::WallLayer,
            zone_id: zone_ref.clone(),
            name: format!("{zname} {surface_key} layer {j}"),
            capacitance_j_per_k: Some(area * layer.thickness * layer.density * layer.specific_heat),
            area_m2: Some(area),
            volume_m3: None,
            azimuth_deg: azimuth,
            tilt_deg: Some(tilt),
            elevation_m: None,
            attributes: attrs,
        });
    }

    graph.push_node(TopologyNode {
        id: format!("{prefix}:int"),
        kind: TopologyNodeKind::InteriorSurface,
        zone_id: zone_ref.clone(),
        name: format!("{zname} {surface_key} interior film"),
        capacitance_j_per_k: None,
        area_m2: Some(area),
        volume_m3: None,
        azimuth_deg: azimuth,
        tilt_deg: Some(tilt),
        elevation_m: None,
        attributes: BTreeMap::new(),
    });

    // --- Coupling edges -------------------------------------------------
    let outdoor = "ambient";
    if ground_coupled.is_none() {
        graph.push_edge(TopologyEdge {
            source_id: outdoor.to_string(),
            target_id: format!("{prefix}:ext"),
            coupling_type: TopologyEdgeKind::ConvectionExterior,
            conductance_w_per_k: Some(EXTERIOR_FILM_COEFF_W_M2K * area),
            fraction: None,
            bidirectional: false,
            attributes: BTreeMap::new(),
        });
    }

    // Centered-capacitance conduction chain: each layer node carries its full
    // capacitance and half its resistance on each side.
    let layer_r: Vec<f64> = construction
        .layers
        .iter()
        .map(|l| l.thickness / (l.conductivity * area))
        .collect();
    let mut prev = format!("{prefix}:ext");
    for (j, r_j) in layer_r.iter().enumerate() {
        let next = format!("{prefix}:layer-{j}");
        graph.push_edge(TopologyEdge {
            source_id: prev.clone(),
            target_id: next.clone(),
            coupling_type: TopologyEdgeKind::Conduction,
            conductance_w_per_k: Some(1.0 / (r_j / 2.0)),
            fraction: None,
            bidirectional: true,
            attributes: BTreeMap::new(),
        });
        prev = next;
    }
    graph.push_edge(TopologyEdge {
        source_id: prev,
        target_id: format!("{prefix}:int"),
        coupling_type: TopologyEdgeKind::Conduction,
        conductance_w_per_k: Some(1.0 / (layer_r.last().copied().unwrap_or(0.0) / 2.0)),
        fraction: None,
        bidirectional: true,
        attributes: BTreeMap::new(),
    });

    graph.push_edge(TopologyEdge {
        source_id: format!("{prefix}:int"),
        target_id: format!("zone-{zi}:air"),
        coupling_type: TopologyEdgeKind::ConvectionInterior,
        conductance_w_per_k: Some(INTERIOR_FILM_COEFF_W_M2K * area),
        fraction: None,
        bidirectional: false,
        attributes: BTreeMap::new(),
    });

    // Longwave star edge: interior film ↔ zone radiant-mass node.
    graph.push_edge(TopologyEdge {
        source_id: format!("{prefix}:int"),
        target_id: format!("zone-{zi}:mass"),
        coupling_type: TopologyEdgeKind::LongwaveRadiation,
        conductance_w_per_k: None,
        fraction: None,
        bidirectional: true,
        attributes: BTreeMap::new(),
    });
    format!("{prefix}:int")
}

fn push_common_wall(graph: &mut TopologyGraph, cw: &CommonWall) {
    // Inter-zone common wall: modeled as a proper wall chain with explicit
    // thermal mass nodes, using centered-capacitance for each layer.
    //
    // Chain structure:
    //   zone-a:air → zone-a:common-wall:int
    //   zone-a:common-wall:int ↔ common-wall:layer-0 (split at midplane)
    //   ...
    //   common-wall:layer-N → zone-b:common-wall:int
    //   zone-b:common-wall:int → zone-b:air
    //
    // Both sides use INTERIOR film coefficient since both zones are conditioned
    // interior spaces (not exterior ambient).
    //
    // Zone indices are guarded by graph validation.
    let area = cw.area;
    let Construction { layers } = &cw.construction;

    let zone_a_int_id = format!("zone-{}:common-wall:int", cw.zone_a);
    let zone_b_int_id = format!("zone-{}:common-wall:int", cw.zone_b);
    let zone_a_id = format!("zone-{}:air", cw.zone_a);
    let zone_b_id = format!("zone-{}:air", cw.zone_b);

    // Add zone-a interior film node (convection from zone air to film)
    graph.push_node(TopologyNode {
        id: zone_a_int_id.clone(),
        kind: TopologyNodeKind::InteriorSurface,
        zone_id: Some(zone_a_id.clone()),
        name: format!("Zone {} common-wall interior film", cw.zone_a),
        capacitance_j_per_k: None,
        area_m2: Some(area),
        volume_m3: None,
        azimuth_deg: None,
        tilt_deg: None,
        elevation_m: None,
        attributes: attr("common_wall", 1.0),
    });

    // Add zone-b interior film node (convection from film to zone air)
    graph.push_node(TopologyNode {
        id: zone_b_int_id.clone(),
        kind: TopologyNodeKind::InteriorSurface,
        zone_id: Some(zone_b_id.clone()),
        name: format!("Zone {} common-wall interior film", cw.zone_b),
        capacitance_j_per_k: None,
        area_m2: Some(area),
        volume_m3: None,
        azimuth_deg: None,
        tilt_deg: None,
        elevation_m: None,
        attributes: attr("common_wall", 1.0),
    });

    // Calculate layer resistances for centered-capacitance model
    let layer_r: Vec<f64> = layers
        .iter()
        .map(|l| l.thickness / (l.conductivity * area))
        .collect();

    // Add layer nodes and conduction edges (centered-capacitance model)
    let num_layers = layers.len();
    for (i, layer) in layers.iter().enumerate() {
        let layer_id = format!("common-wall:layer-{}", i);
        let r_layer = layer.thickness / (layer.conductivity * area);
        let capacitance = area * layer.thickness * layer.density * layer.specific_heat;

        // Layer node
        graph.push_node(TopologyNode {
            id: layer_id.clone(),
            kind: TopologyNodeKind::WallLayer,
            zone_id: None, // Layer nodes are not zone-specific
            name: format!("Common-wall layer {}", i),
            capacitance_j_per_k: Some(capacitance),
            area_m2: Some(area),
            volume_m3: None,
            azimuth_deg: None,
            tilt_deg: None,
            elevation_m: None,
            attributes: {
                let mut attrs = attr("common_wall_layer", 1.0);
                attrs.insert("layer_index".to_string(), (i as f64).into());
                attrs.insert("thickness_m".to_string(), layer.thickness);
                attrs.insert("conductivity_w_per_mk".to_string(), layer.conductivity);
                attrs.insert("density_kg_per_m3".to_string(), layer.density);
                attrs.insert("specific_heat_j_per_kgk".to_string(), layer.specific_heat);
                attrs
            },
        });

        // Conduction from zone-a side to first layer (or between layers)
        if i == 0 {
            // First layer: connect from zone-a int film
            let r_half = r_layer / 2.0;
            let conductance_a = 1.0 / r_half;
            graph.push_edge(TopologyEdge {
                source_id: zone_a_int_id.clone(),
                target_id: layer_id.clone(),
                coupling_type: TopologyEdgeKind::Conduction,
                conductance_w_per_k: Some(conductance_a),
                fraction: None,
                bidirectional: true,
                attributes: with_common_wall(attr("common_wall_conduction", 1.0)),
            });
        } else {
            // Subsequent layers: connect from previous layer node
            let prev_id = format!("common-wall:layer-{}", i - 1);
            let r_half = layer_r[i - 1] / 2.0;
            let conductance_a = 1.0 / r_half;
            graph.push_edge(TopologyEdge {
                source_id: prev_id,
                target_id: layer_id.clone(),
                coupling_type: TopologyEdgeKind::Conduction,
                conductance_w_per_k: Some(conductance_a),
                fraction: None,
                bidirectional: true,
                attributes: with_common_wall(attr("common_wall_conduction", 1.0)),
            });
        }

        // Conduction from this layer to zone-b side (or next layer)
        if i == num_layers - 1 {
            // Last layer: connect to zone-b int film
            let r_half = r_layer / 2.0;
            let conductance_b = 1.0 / r_half;
            graph.push_edge(TopologyEdge {
                source_id: layer_id,
                target_id: zone_b_int_id.clone(),
                coupling_type: TopologyEdgeKind::Conduction,
                conductance_w_per_k: Some(conductance_b),
                fraction: None,
                bidirectional: true,
                attributes: with_common_wall(attr("common_wall_conduction", 1.0)),
            });
        }
    }

    // If no layers, directly connect zone-a int to zone-b int
    if layers.is_empty() {
        let r_total = 1.0 / (INTERIOR_FILM_COEFF_W_M2K * area);
        graph.push_edge(TopologyEdge {
            source_id: zone_a_int_id.clone(),
            target_id: zone_b_int_id.clone(),
            coupling_type: TopologyEdgeKind::Conduction,
            conductance_w_per_k: Some(1.0 / r_total),
            fraction: None,
            bidirectional: true,
            attributes: with_common_wall(attr("common_wall_conduction", 1.0)),
        });
    }

    // Add convection edges: zone air → interior film (zone-a side)
    graph.push_edge(TopologyEdge {
        source_id: format!("zone-{}:air", cw.zone_a),
        target_id: zone_a_int_id.clone(),
        coupling_type: TopologyEdgeKind::ConvectionInterior,
        conductance_w_per_k: Some(INTERIOR_FILM_COEFF_W_M2K * area),
        fraction: None,
        bidirectional: false,
        attributes: with_common_wall(attr("common_wall_convection", 1.0)),
    });

    // Add convection edges: interior film → zone air (zone-b side)
    graph.push_edge(TopologyEdge {
        source_id: zone_b_int_id.clone(),
        target_id: format!("zone-{}:air", cw.zone_b),
        coupling_type: TopologyEdgeKind::ConvectionInterior,
        conductance_w_per_k: Some(INTERIOR_FILM_COEFF_W_M2K * area),
        fraction: None,
        bidirectional: false,
        attributes: with_common_wall(attr("common_wall_convection", 1.0)),
    });

    // Add longwave radiation from interior films to zone mass nodes
    graph.push_edge(TopologyEdge {
        source_id: zone_a_int_id.clone(),
        target_id: format!("zone-{}:mass", cw.zone_a),
        coupling_type: TopologyEdgeKind::LongwaveRadiation,
        conductance_w_per_k: Some(INTERIOR_FILM_COEFF_W_M2K * area),
        fraction: None,
        bidirectional: false,
        attributes: with_common_wall(attr("common_wall_lw", 1.0)),
    });

    graph.push_edge(TopologyEdge {
        source_id: zone_b_int_id,
        target_id: format!("zone-{}:mass", cw.zone_b),
        coupling_type: TopologyEdgeKind::LongwaveRadiation,
        conductance_w_per_k: Some(INTERIOR_FILM_COEFF_W_M2K * area),
        fraction: None,
        bidirectional: false,
        attributes: with_common_wall(attr("common_wall_lw", 1.0)),
    });
}

/// Stable window key for a wall orientation (used in node ids).
fn window_key(orientation: &Orientation) -> &'static str {
    match orientation {
        Orientation::North => "north",
        Orientation::East => "east",
        Orientation::South => "south",
        Orientation::West => "west",
        Orientation::Up | Orientation::Horizontal => "skylight",
        Orientation::Down => "floor-window",
    }
}

/// Tilt for a window from its orientation (windows sit in walls; skylights
/// lie flat).
fn window_tilt_deg(orientation: &Orientation) -> f64 {
    match orientation {
        Orientation::North | Orientation::East | Orientation::South | Orientation::West => 90.0,
        Orientation::Up | Orientation::Horizontal => 0.0,
        Orientation::Down => 180.0,
    }
}

fn push_window(
    graph: &mut TopologyGraph,
    zi: usize,
    zname: &str,
    w: &WindowArea,
    spec: &CaseSpec,
    hosts: &std::collections::BTreeSet<String>,
    index: usize,
) {
    if w.area <= 0.0 {
        return;
    }
    let outdoor = "ambient";
    // Per-(zone, orientation) index keeps node ids unique when a zone has
    // several windows on the same wall.
    let win_id = format!("zone-{zi}:window-{}:{index}", window_key(&w.orientation));
    let zone_air = format!("zone-{zi}:air");
    let zone_ref = Some(zone_air.clone());
    let shgc = spec.window_properties.shgc;
    // Glazing conduction (glass + frame in series, per WindowSpec doc).
    let u_eff = spec.window_properties.u_value + spec.window_properties.frame_u_value;

    // Window node (Issue #3972): the glazing assembly as a first-class graph
    // node carrying the audit-relevant properties (area, orientation, U, SHGC).
    let mut attrs = BTreeMap::new();
    attrs.insert("u_value_w_per_m2k".to_string(), u_eff);
    attrs.insert("shgc".to_string(), shgc);
    attrs.insert("glazing".to_string(), 1.0);
    if spec.shading.is_some() {
        attrs.insert("shading_present".to_string(), 1.0);
    }
    graph.push_node(TopologyNode {
        id: win_id.clone(),
        kind: TopologyNodeKind::Window,
        zone_id: zone_ref,
        name: format!("{zname} window {} #{index}", window_key(&w.orientation)),
        capacitance_j_per_k: None,
        area_m2: Some(w.area),
        volume_m3: None,
        azimuth_deg: azimuth_deg(&w.orientation),
        tilt_deg: Some(window_tilt_deg(&w.orientation)),
        elevation_m: None,
        attributes: attrs,
    });

    // Glazing conduction: outdoor -> glazing assembly carries the full
    // U_eff x A (unchanged value from the pre-#3972 direct ambient->air edge).
    graph.push_edge(TopologyEdge {
        source_id: outdoor.to_string(),
        target_id: win_id.clone(),
        coupling_type: TopologyEdgeKind::Conduction,
        conductance_w_per_k: Some(u_eff * w.area),
        fraction: None,
        bidirectional: true,
        attributes: BTreeMap::new(),
    });
    // Glazing -> zone air: pure topological link (the conductance lives on
    // the outdoor leg; the glazing node models no capacitance).
    graph.push_edge(TopologyEdge {
        source_id: win_id.clone(),
        target_id: zone_air.clone(),
        coupling_type: TopologyEdgeKind::Conduction,
        conductance_w_per_k: None,
        fraction: None,
        bidirectional: true,
        attributes: BTreeMap::new(),
    });

    // Solar admission through the glazing: window -> receiving surface shows where
    // the admitted solar lands. Solar distribution is multi-target: beam and
    // diffuse radiation are distributed across multiple interior surfaces based
    // on orientation-dependent factors (Issue #4048).
    //
    // For beam radiation, use distribution factors based on orientation.
    // For diffuse radiation, the distribution is the same for all window orientations
    // (diffuse is isotropic from the sky dome): floor 40%, north 15%,
    // east 15%, south 15%, west 15%.
    let direct_factors = solar_beam_factors(&w.orientation, hosts, zi);
    let diffuse_factors = solar_diffuse_factors(hosts, zi);

    let mut solar_attrs = BTreeMap::new();
    solar_attrs.insert("window_area_m2".to_string(), w.area);
    if spec.shading.is_some() {
        solar_attrs.insert("shading_present".to_string(), 1.0);
    }

    // ambient -> window: solar admission with full SHGC fraction.
    // This edge represents solar radiation entering through the window.
    for kind in [
        TopologyEdgeKind::ShortwaveSolarDirect,
        TopologyEdgeKind::ShortwaveSolarDiffuse,
    ] {
        graph.push_edge(TopologyEdge {
            source_id: outdoor.to_string(),
            target_id: win_id.clone(),
            coupling_type: kind,
            conductance_w_per_k: None,
            fraction: Some(shgc),
            bidirectional: false,
            attributes: solar_attrs.clone(),
        });
    }

    // ShortwaveSolarDirect: distribute beam radiation across interior surfaces.
    // Only create edges for surfaces that are in hosts. Missing surfaces
    // (e.g., fully-glazed wall) simply don't receive solar.
    for (surface, fraction) in &direct_factors {
        if hosts.contains(surface) {
            graph.push_edge(TopologyEdge {
                source_id: win_id.clone(),
                target_id: surface.clone(),
                coupling_type: TopologyEdgeKind::ShortwaveSolarDirect,
                conductance_w_per_k: None,
                fraction: Some(shgc * fraction),
                bidirectional: false,
                attributes: solar_attrs.clone(),
            });
        }
    }

    // ShortwaveSolarDiffuse: distribute diffuse radiation across interior surfaces.
    for (surface, fraction) in &diffuse_factors {
        if hosts.contains(surface) {
            graph.push_edge(TopologyEdge {
                source_id: win_id.clone(),
                target_id: surface.clone(),
                coupling_type: TopologyEdgeKind::ShortwaveSolarDiffuse,
                conductance_w_per_k: None,
                fraction: Some(shgc * fraction),
                bidirectional: false,
                attributes: solar_attrs.clone(),
            });
        }
    }
}

impl ToTopologyGraph for CaseSpec {
    fn to_topology_graph(&self, ctx: &TopologyContext) -> TopologyGraph {
        let mut graph = TopologyGraph::new(ctx);

        graph.push_node(TopologyNode {
            id: "ambient".to_string(),
            kind: TopologyNodeKind::OutdoorAmbient,
            zone_id: None,
            name: "Outdoor ambient".to_string(),
            capacitance_j_per_k: None,
            area_m2: None,
            volume_m3: None,
            azimuth_deg: None,
            tilt_deg: None,
            elevation_m: None,
            attributes: BTreeMap::new(),
        });

        let wall_orientations = [
            Orientation::North,
            Orientation::East,
            Orientation::South,
            Orientation::West,
        ];

        for zi in 0..self.num_zones {
            let g = &self.geometry[zi];
            let zname = zone_name(zi, g);
            let volume = g.width * g.depth * g.height;
            let floor_area = g.width * g.depth;
            let zone_ref = Some(format!("zone-{zi}:air"));

            // Zone air node.
            graph.push_node(TopologyNode {
                id: format!("zone-{zi}:air"),
                kind: TopologyNodeKind::ZoneAir,
                zone_id: Some(format!("zone-{zi}:air")),
                name: format!("{zname} air"),
                capacitance_j_per_k: Some(AIR_DENSITY_KG_M3 * AIR_SPECIFIC_HEAT_J_KG_K * volume),
                area_m2: None,
                volume_m3: Some(volume),
                azimuth_deg: None,
                tilt_deg: None,
                elevation_m: None,
                attributes: BTreeMap::new(),
            });

            // Radiant-mass star node (5R1C T_m analogue).
            graph.push_node(TopologyNode {
                id: format!("zone-{zi}:mass"),
                kind: TopologyNodeKind::InternalMass,
                zone_id: zone_ref.clone(),
                name: format!("{zname} radiant mass"),
                capacitance_j_per_k: None,
                area_m2: Some(floor_area),
                volume_m3: None,
                azimuth_deg: None,
                tilt_deg: None,
                elevation_m: None,
                attributes: BTreeMap::new(),
            });

            // HVAC terminal node.
            let hvac = self.hvac.get(zi);
            let mut hvac_attrs = BTreeMap::new();
            if let Some(h) = hvac {
                hvac_attrs.insert("heating_setpoint_c".to_string(), h.heating_setpoint);
                hvac_attrs.insert("cooling_setpoint_c".to_string(), h.cooling_setpoint);
                hvac_attrs.insert(
                    "hvac_start_hour".to_string(),
                    f64::from(h.operating_hours.0),
                );
                hvac_attrs.insert("hvac_end_hour".to_string(), f64::from(h.operating_hours.1));
                hvac_attrs.insert("efficiency".to_string(), h.efficiency);
            }
            graph.push_node(TopologyNode {
                id: format!("zone-{zi}:hvac"),
                kind: TopologyNodeKind::HvacTerminal,
                zone_id: zone_ref.clone(),
                name: format!("{zname} HVAC terminal"),
                capacitance_j_per_k: None,
                area_m2: None,
                volume_m3: None,
                azimuth_deg: None,
                tilt_deg: None,
                elevation_m: None,
                attributes: hvac_attrs,
            });
            if hvac.is_some() {
                graph.push_edge(TopologyEdge {
                    source_id: format!("zone-{zi}:hvac"),
                    target_id: format!("zone-{zi}:air"),
                    coupling_type: TopologyEdgeKind::HvacSensible,
                    conductance_w_per_k: None,
                    fraction: None,
                    bidirectional: true,
                    attributes: BTreeMap::new(),
                });
            }

            // Internal gains: convective + radiative split paths.
            if let Some(il) = self.internal_loads.get(zi).and_then(|o| o.as_ref()) {
                let mut gain_attrs = attr("total_load_w", il.total_load);
                gain_attrs.insert("radiative_fraction".to_string(), il.radiative_fraction);
                gain_attrs.insert("convective_fraction".to_string(), il.convective_fraction);
                graph.push_node(TopologyNode {
                    id: format!("zone-{zi}:gain"),
                    kind: TopologyNodeKind::InternalGain,
                    zone_id: zone_ref.clone(),
                    name: format!("{zname} internal gain"),
                    capacitance_j_per_k: None,
                    area_m2: None,
                    volume_m3: None,
                    azimuth_deg: None,
                    tilt_deg: None,
                    elevation_m: None,
                    attributes: gain_attrs,
                });
                graph.push_edge(TopologyEdge {
                    source_id: format!("zone-{zi}:gain"),
                    target_id: format!("zone-{zi}:air"),
                    coupling_type: TopologyEdgeKind::InternalGainSplit,
                    conductance_w_per_k: None,
                    fraction: Some(il.convective_fraction),
                    bidirectional: false,
                    attributes: attr("path", 0.0),
                });
                graph.push_edge(TopologyEdge {
                    source_id: format!("zone-{zi}:gain"),
                    target_id: format!("zone-{zi}:mass"),
                    coupling_type: TopologyEdgeKind::InternalGainSplit,
                    conductance_w_per_k: None,
                    fraction: Some(il.radiative_fraction),
                    bidirectional: false,
                    attributes: attr("path", 1.0),
                });
            }

            // Window area per wall orientation (subtracted from gross wall).
            let mut window_area = BTreeMap::new();
            if let Some(zone_windows) = self.windows.get(zi) {
                for w in zone_windows {
                    *window_area.entry(wall_key(&w.orientation)).or_insert(0.0) += w.area;
                }
            }

            // Opaque surfaces: four walls + roof + floor.
            let mut hosts = std::collections::BTreeSet::new();
            for orientation in &wall_orientations {
                let key = wall_key(orientation);
                let gross = gross_wall_area(g, orientation);
                let net = (gross - window_area.get(key).copied().unwrap_or(0.0)).max(0.0);
                if net <= 0.0 {
                    continue;
                }
                hosts.insert(push_surface_chain(
                    &mut graph,
                    zi,
                    &zname,
                    key,
                    net,
                    azimuth_deg(orientation),
                    90.0,
                    &self.construction.wall,
                    None,
                ));
            }
            hosts.insert(push_surface_chain(
                &mut graph,
                zi,
                &zname,
                "roof",
                floor_area,
                None,
                0.0,
                &self.construction.roof,
                None,
            ));
            hosts.insert(push_surface_chain(
                &mut graph,
                zi,
                &zname,
                "floor",
                floor_area,
                None,
                180.0,
                &self.construction.floor,
                Some(self.ground_temperature_c.unwrap_or(10.0)),
            ));

            // Windows (Issue #3972): glazing assemblies as first-class nodes.
            // Per-orientation counters keep window node ids unique when a
            // zone has several windows on the same wall.
            if let Some(zone_windows) = self.windows.get(zi) {
                let mut counters: BTreeMap<&str, usize> = BTreeMap::new();
                for w in zone_windows {
                    let key = window_key(&w.orientation);
                    let index = counters.entry(key).or_insert(0);
                    let i = *index;
                    *index += 1;
                    push_window(&mut graph, zi, &zname, w, self, &hosts, i);
                }
            }

            // Infiltration.
            if self.infiltration_ach > 0.0 {
                let g_inf =
                    self.infiltration_ach * volume * AIR_DENSITY_KG_M3 * AIR_SPECIFIC_HEAT_J_KG_K
                        / 3600.0;
                graph.push_edge(TopologyEdge {
                    source_id: format!("zone-{zi}:air"),
                    target_id: "ambient".to_string(),
                    coupling_type: TopologyEdgeKind::AirExchange,
                    conductance_w_per_k: Some(g_inf),
                    fraction: None,
                    bidirectional: true,
                    attributes: attr("air_changes_per_hour", self.infiltration_ach),
                });
            }
        }

        // Night ventilation (whole-model, first zone schedule per spec).
        if let Some(nv) = &self.night_ventilation {
            for zi in 0..self.num_zones {
                let volume = self
                    .geometry
                    .get(zi)
                    .map(|g| g.width * g.depth * g.height)
                    .unwrap_or(0.0);
                if volume <= 0.0 {
                    continue;
                }
                // Night ventilation is driven by a whole-model fan (m³/h).
                let g_nv = nv.fan_capacity * AIR_DENSITY_KG_M3 * AIR_SPECIFIC_HEAT_J_KG_K / 3600.0;
                let mut nv_attrs = attr("fan_capacity_m3_per_h", nv.fan_capacity);
                nv_attrs.insert("start_hour".to_string(), f64::from(nv.operating_hours.0));
                nv_attrs.insert("end_hour".to_string(), f64::from(nv.operating_hours.1));
                nv_attrs.insert("adds_heat".to_string(), f64::from(nv.adds_heat));
                graph.push_edge(TopologyEdge {
                    source_id: format!("zone-{zi}:air"),
                    target_id: "ambient".to_string(),
                    coupling_type: TopologyEdgeKind::AirExchange,
                    conductance_w_per_k: Some(g_nv),
                    fraction: None,
                    bidirectional: true,
                    attributes: nv_attrs,
                });
            }
        }

        for cw in &self.common_walls {
            push_common_wall(&mut graph, cw);
        }

        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::TOPOLOGY_SCHEMA_VERSION;
    use crate::validation::ashrae_140_cases::ASHRAE140Case;

    fn graph_for(case: ASHRAE140Case) -> TopologyGraph {
        let spec = case.spec();
        let mut g = spec.to_topology_graph(&TopologyContext::new(
            format!("ASHRAE 140 Case {}", spec.case_id),
            format!("ashrae-140-registry:{}", spec.case_id),
        ));
        g.finalize().expect("finalize");
        g.validate().expect("validate");
        g
    }

    fn count_nodes(g: &TopologyGraph, kind: TopologyNodeKind) -> usize {
        g.nodes.iter().filter(|n| n.kind == kind).count()
    }

    fn count_edges(g: &TopologyGraph, kind: TopologyEdgeKind) -> usize {
        g.edges.iter().filter(|e| e.coupling_type == kind).count()
    }

    #[test]
    fn case_600_has_all_node_kinds() {
        let g = graph_for(ASHRAE140Case::Case600);
        assert_eq!(count_nodes(&g, TopologyNodeKind::OutdoorAmbient), 1);
        assert_eq!(count_nodes(&g, TopologyNodeKind::ZoneAir), 1);
        assert_eq!(count_nodes(&g, TopologyNodeKind::InternalMass), 1);
        assert_eq!(count_nodes(&g, TopologyNodeKind::InternalGain), 1);
        assert_eq!(count_nodes(&g, TopologyNodeKind::HvacTerminal), 1);
        assert!(count_nodes(&g, TopologyNodeKind::WallLayer) >= 6);
        assert!(count_nodes(&g, TopologyNodeKind::ExteriorSurface) >= 5);
        assert!(count_nodes(&g, TopologyNodeKind::InteriorSurface) >= 5);
    }

    #[test]
    fn case_600_layer_chain_matches_construction() {
        let spec = ASHRAE140Case::Case600.spec();
        let g = graph_for(ASHRAE140Case::Case600);
        // Each chain with L layers contributes L+1 conduction edges
        // (ext→l0, l_j→l_{j+1}, l_last→int); each window contributes two
        // glazing conduction edges (ambient→window, window→air).
        let windows: usize = spec
            .windows
            .iter()
            .map(|z| z.iter().filter(|w| w.area > 0.0).count())
            .sum();
        let expected: usize = (spec.construction.wall.layers.len() + 1) * 4
            + (spec.construction.roof.layers.len() + 1)
            + (spec.construction.floor.layers.len() + 1)
            + 2 * windows;
        assert_eq!(count_edges(&g, TopologyEdgeKind::Conduction), expected);
    }

    #[test]
    fn case_600_solar_and_gain_split_paths() {
        let g = graph_for(ASHRAE140Case::Case600);
        assert!(count_edges(&g, TopologyEdgeKind::ShortwaveSolarDirect) >= 1);
        assert!(count_edges(&g, TopologyEdgeKind::ShortwaveSolarDiffuse) >= 1);
        let splits: Vec<f64> = g
            .edges
            .iter()
            .filter(|e| e.coupling_type == TopologyEdgeKind::InternalGainSplit)
            .filter_map(|e| e.fraction)
            .collect();
        assert_eq!(splits.len(), 2);
        assert!((splits.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
    /// Regression test: window-transmitted solar must be distributed across
    /// multiple interior surfaces (floor, N/E/S/W walls), not 100% to the host
    /// wall (Issue #4048).
    ///
    /// The topology model is:
    /// - ambient -> Window (solar admission, fraction = SHGC)
    /// - Window -> receiving_surface (solar distribution, fractions sum to SHGC)
    #[test]
    fn case_600_window_solar_distributed() {
        let g = graph_for(ASHRAE140Case::Case600);

        // Collect all window solar edges
        let direct_edges: Vec<&TopologyEdge> = g
            .edges
            .iter()
            .filter(|e| e.coupling_type == TopologyEdgeKind::ShortwaveSolarDirect)
            .collect();

        let diffuse_edges: Vec<&TopologyEdge> = g
            .edges
            .iter()
            .filter(|e| e.coupling_type == TopologyEdgeKind::ShortwaveSolarDiffuse)
            .collect();

        assert!(
            !direct_edges.is_empty(),
            "Should have ShortwaveSolarDirect edges"
        );
        assert!(
            !diffuse_edges.is_empty(),
            "Should have ShortwaveSolarDiffuse edges"
        );

        // Separate admission edges (ambient -> Window) from distribution edges
        // (Window -> receiving_surface). Window nodes have id containing "window-".
        let direct_admission: Vec<&&TopologyEdge> = direct_edges
            .iter()
            .filter(|e| e.source_id == "ambient")
            .collect();
        let direct_distribution: Vec<&&TopologyEdge> = direct_edges
            .iter()
            .filter(|e| e.source_id.contains("window-"))
            .collect();

        let diffuse_admission: Vec<&&TopologyEdge> = diffuse_edges
            .iter()
            .filter(|e| e.source_id == "ambient")
            .collect();
        let diffuse_distribution: Vec<&&TopologyEdge> = diffuse_edges
            .iter()
            .filter(|e| e.source_id.contains("window-"))
            .collect();

        // Check admission edges: ambient -> Window should have fraction = SHGC
        let expected_shgc = 0.77;
        for edge in direct_admission.iter() {
            let frac = edge.fraction.unwrap_or(0.0);
            assert!(
                (frac - expected_shgc).abs() < 0.01,
                "Direct admission edge fraction should be SHGC (~0.77), got {:.3}",
                frac
            );
        }
        for edge in diffuse_admission.iter() {
            let frac = edge.fraction.unwrap_or(0.0);
            assert!(
                (frac - expected_shgc).abs() < 0.01,
                "Diffuse admission edge fraction should be SHGC (~0.77), got {:.3}",
                frac
            );
        }

        // Check distribution edges: Window -> receiving_surface fractions should
        // sum to SHGC
        let direct_dist_sum: f64 = direct_distribution.iter().filter_map(|e| e.fraction).sum();
        let diffuse_dist_sum: f64 = diffuse_distribution.iter().filter_map(|e| e.fraction).sum();

        assert!(
            (direct_dist_sum - expected_shgc).abs() < 0.01,
            "Direct distribution fractions should sum to SHGC (~0.77), got {:.3}",
            direct_dist_sum
        );
        assert!(
            (diffuse_dist_sum - expected_shgc).abs() < 0.01,
            "Diffuse distribution fractions should sum to SHGC (~0.77), got {:.3}",
            diffuse_dist_sum
        );

        // KEY ASSERTION: Solar should be distributed across multiple surfaces,
        // NOT 100% to one surface (the original bug, Issue #4048)
        let direct_targets: std::collections::HashSet<&String> =
            direct_distribution.iter().map(|e| &e.target_id).collect();
        let diffuse_targets: std::collections::HashSet<&String> =
            diffuse_distribution.iter().map(|e| &e.target_id).collect();

        assert!(
            direct_targets.len() >= 2,
            "Direct solar should be distributed across >= 2 surfaces, got {}: {:?}",
            direct_targets.len(),
            direct_targets
        );
        assert!(
            diffuse_targets.len() >= 2,
            "Diffuse solar should be distributed across >= 2 surfaces, got {}: {:?}",
            diffuse_targets.len(),
            diffuse_targets
        );
    }
    #[test]
    fn case_600_infiltration_edge_present() {
        let spec = ASHRAE140Case::Case600.spec();
        let g = graph_for(ASHRAE140Case::Case600);
        let inf: Vec<&TopologyEdge> = g
            .edges
            .iter()
            .filter(|e| e.coupling_type == TopologyEdgeKind::AirExchange)
            .collect();
        assert!(!inf.is_empty());
        if spec.infiltration_ach > 0.0 {
            assert!(inf
                .iter()
                .any(|e| e.attributes.get("air_changes_per_hour") == Some(&spec.infiltration_ach)));
        }
    }

    #[test]
    fn case_960_is_multi_zone_with_common_wall() {
        let g = graph_for(ASHRAE140Case::Case960);
        assert_eq!(count_nodes(&g, TopologyNodeKind::ZoneAir), 2);
        let common: Vec<&TopologyEdge> = g
            .edges
            .iter()
            .filter(|e| e.attributes.contains_key("common_wall"))
            .collect();
        assert!(!common.is_empty());
        // Edges should connect zone nodes or zone nodes to wall chain nodes
        assert!(common.iter().all(|e| {
            let src_zone = e.source_id.starts_with("zone-");
            let tgt_zone = e.target_id.starts_with("zone-");
            let src_chain = e.source_id.contains("common-wall");
            let tgt_chain = e.target_id.contains("common-wall");
            // Valid patterns: zone->zone, zone->common-wall chain, common-wall chain->zone
            (src_zone && tgt_zone) || (src_zone && tgt_chain) || (src_chain && tgt_zone)
        }));
    }

    #[test]
    fn export_is_deterministic_across_builds() {
        let a = graph_for(ASHRAE140Case::Case900);
        let b = graph_for(ASHRAE140Case::Case900);
        assert_eq!(a.to_json_string().unwrap(), b.to_json_string().unwrap());
    }

    #[test]
    fn metadata_carries_schema_and_tool_versions() {
        let g = graph_for(ASHRAE140Case::Case600);
        assert_eq!(g.metadata.schema_version, TOPOLOGY_SCHEMA_VERSION);
        assert_eq!(g.metadata.tool_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(g.metadata.node_count, g.nodes.len());
        assert_eq!(g.metadata.edge_count, g.edges.len());
        assert!(g.metadata.model_source.starts_with("ashrae-140-registry:"));
    }

    #[test]
    fn high_mass_case_has_more_layer_capacitance_than_low_mass() {
        let low = graph_for(ASHRAE140Case::Case600);
        let high = graph_for(ASHRAE140Case::Case900);
        let cap = |g: &TopologyGraph| {
            g.nodes
                .iter()
                .filter(|n| n.kind == TopologyNodeKind::WallLayer)
                .filter_map(|n| n.capacitance_j_per_k)
                .sum::<f64>()
        };
        assert!(cap(&high) > cap(&low));
    }

    /// Regression test: North/South wall area must use `width` (X-axis, spans N-S),
    /// and East/West must use `depth` (Y-axis, spans E-W) — even when width ≠ depth.
    #[test]
    fn gross_wall_area_uses_correct_geometric_dimension() {
        // Zone with asymmetric footprint: 8 m wide (X) × 6 m deep (Y) × 2.7 m tall
        let g = GeometrySpec {
            name: Some("AsymmetricZone".into()),
            ..GeometrySpec::new(8.0, 6.0, 2.7)
        };

        // N/S walls span X-axis → 8 * 2.7
        assert!((gross_wall_area(&g, &Orientation::North) - 8.0 * 2.7).abs() < 1e-9);
        assert!((gross_wall_area(&g, &Orientation::South) - 8.0 * 2.7).abs() < 1e-9);
        // E/W walls span Y-axis → 6 * 2.7
        assert!((gross_wall_area(&g, &Orientation::East) - 6.0 * 2.7).abs() < 1e-9);
        assert!((gross_wall_area(&g, &Orientation::West) - 6.0 * 2.7).abs() < 1e-9);
    }

    /// Issue #3972: windows must appear as first-class nodes in the topology
    /// export (kind=window), with glazing-conduction and solar edges routed
    /// through them.
    #[test]
    fn case_600_window_nodes_present_with_glazing_and_solar_paths() {
        let spec = ASHRAE140Case::Case600.spec();
        let g = graph_for(ASHRAE140Case::Case600);

        let windows: Vec<&TopologyNode> = g
            .nodes
            .iter()
            .filter(|n| n.kind == TopologyNodeKind::Window)
            .collect();
        assert_eq!(windows.len(), 1, "Case 600 has one south window");

        let w = windows[0];
        assert_eq!(w.id, "zone-0:window-south:0");
        assert_eq!(w.zone_id.as_deref(), Some("zone-0:air"));
        assert!((w.area_m2.unwrap() - 12.0).abs() < 1e-9);
        assert_eq!(w.azimuth_deg, Some(180.0));
        assert_eq!(w.tilt_deg, Some(90.0));
        let u_eff = spec.window_properties.u_value + spec.window_properties.frame_u_value;
        assert!((w.attributes["u_value_w_per_m2k"] - u_eff).abs() < 1e-12);
        assert!((w.attributes["shgc"] - spec.window_properties.shgc).abs() < 1e-12);

        // Glazing conduction: ambient -> window carries U_eff x A, then a
        // topological link window -> zone air.
        let cond: Vec<&TopologyEdge> = g
            .edges
            .iter()
            .filter(|e| {
                e.coupling_type == TopologyEdgeKind::Conduction
                    && (e.source_id == w.id || e.target_id == w.id)
            })
            .collect();
        assert_eq!(cond.len(), 2);
        let in_leg = cond
            .iter()
            .find(|e| e.source_id == "ambient")
            .expect("ambient->window conduction");
        assert!((in_leg.conductance_w_per_k.unwrap() - u_eff * 12.0).abs() < 1e-9);
        let out_leg = cond
            .iter()
            .find(|e| e.target_id == "zone-0:air")
            .expect("window->air conduction");
        assert!(out_leg.conductance_w_per_k.is_none());

        // Solar admission passes through the window node: ambient -> window
        // (SHGC fraction), window -> receiving surfaces (multi-target distribution,
        // Issue #4048).
        for kind in [
            TopologyEdgeKind::ShortwaveSolarDirect,
            TopologyEdgeKind::ShortwaveSolarDiffuse,
        ] {
            let solar: Vec<&TopologyEdge> = g
                .edges
                .iter()
                .filter(|e| e.coupling_type == kind && (e.source_id == w.id || e.target_id == w.id))
                .collect();

            // Should have 6 edges per kind: 1 ambient->window (admission) +
            // 5 window->receiving_surface (distribution for floor, N, E, S, W)
            assert!(
                solar.len() >= 2,
                "Should have at least 2 {kind:?} legs per window (admission + distribution)"
            );

            // Verify ambient -> window admission edge has fraction = SHGC
            let admit = solar
                .iter()
                .find(|e| e.source_id == "ambient")
                .expect("ambient->window solar");
            assert!((admit.fraction.unwrap() - spec.window_properties.shgc).abs() < 1e-12);

            // Verify distribution edges go from window to multiple receiving surfaces
            let distribution: Vec<&&TopologyEdge> =
                solar.iter().filter(|e| e.source_id == w.id).collect();
            assert!(
                distribution.len() >= 2,
                "Should have >= 2 distribution edges (multiple interior surfaces)"
            );

            // Distribution fractions should sum to SHGC
            let dist_sum: f64 = distribution.iter().filter_map(|e| e.fraction).sum();
            assert!(
                (dist_sum - spec.window_properties.shgc).abs() < 1e-6,
                "Distribution fractions should sum to SHGC, got {:.4}",
                dist_sum
            );
        }
    }
}

/// Explicit production entry point for the CLI's `--case` export path.
/// Exists so the module dependency is textual (the orphan gate's detector
/// cannot see trait impls); see scripts/check_orphan_modules.py.
pub fn case_topology_graph(
    spec: &CaseSpec,
    ctx: &crate::topology::TopologyContext,
) -> crate::topology::TopologyGraph {
    use crate::topology::ToTopologyGraph;
    spec.to_topology_graph(ctx)
}
