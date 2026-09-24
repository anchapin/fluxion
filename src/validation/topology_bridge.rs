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

use std::collections::BTreeMap;

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
        Orientation::Down => "floor",
    }
}

/// Total R-value (K/W) of an assembly including both film coefficients.
fn total_resistance(area: f64, construction: &Construction) -> f64 {
    let r = construction
        .layers
        .iter()
        .map(|l| l.thickness / (l.conductivity * area))
        .sum::<f64>();
    1.0 / (INTERIOR_FILM_COEFF_W_M2K * area) + r + 1.0 / (EXTERIOR_FILM_COEFF_W_M2K * area)
}

fn attr<K: Into<String>>(key: K, value: f64) -> BTreeMap<String, f64> {
    let mut m = BTreeMap::new();
    m.insert(key.into(), value);
    m
}

/// Zone display name from geometry.
fn zone_name(zi: usize, g: &GeometrySpec) -> String {
    g.name.clone().unwrap_or_else(|| format!("Zone {zi}"))
}

/// Gross opaque area of one orientation's wall before window subtraction.
fn gross_wall_area(g: &GeometrySpec, orientation: &Orientation) -> f64 {
    match orientation {
        Orientation::North | Orientation::South => g.depth * g.height,
        Orientation::East | Orientation::West => g.width * g.height,
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
    // Inter-zone common wall: represented as a single air-to-air conduction
    // coupling with the assembly U-value (films included). Zone indices are
    // guarded by graph validation.
    let area = cw.area;
    let conductance = area / total_resistance(area, &cw.construction);
    let mut attributes = attr("common_wall", 1.0);
    attributes.insert("area_m2".to_string(), area);
    graph.push_edge(TopologyEdge {
        source_id: format!("zone-{}:air", cw.zone_a),
        target_id: format!("zone-{}:air", cw.zone_b),
        coupling_type: TopologyEdgeKind::Conduction,
        conductance_w_per_k: Some(conductance),
        fraction: None,
        bidirectional: true,
        attributes,
    });
}

fn push_window(
    graph: &mut TopologyGraph,
    zi: usize,
    w: &WindowArea,
    spec: &CaseSpec,
    hosts: &std::collections::BTreeSet<String>,
) {
    if w.area <= 0.0 {
        return;
    }
    let outdoor = "ambient";
    // Solar + glazing conduction: glazing inner surface is convectively close
    // to zone air; solar gains land on the host orientation's interior film
    // when that orientation has an opaque chain (a fully-glazed wall has
    // none — its solar gain couples straight to the zone air node).
    let host_surface = format!("zone-{}:{}:int", zi, wall_key(&w.orientation));
    let solar_target = if hosts.contains(&host_surface) {
        host_surface
    } else {
        format!("zone-{zi}:air")
    };
    let shgc = spec.window_properties.shgc;
    let mut solar_attrs = attr("window_area_m2", w.area);
    if spec.shading.is_some() {
        solar_attrs.insert("shading_present".to_string(), 1.0);
    }
    for kind in [
        TopologyEdgeKind::ShortwaveSolarDirect,
        TopologyEdgeKind::ShortwaveSolarDiffuse,
    ] {
        graph.push_edge(TopologyEdge {
            source_id: outdoor.to_string(),
            target_id: solar_target.clone(),
            coupling_type: kind,
            conductance_w_per_k: None,
            fraction: Some(shgc),
            bidirectional: false,
            attributes: solar_attrs.clone(),
        });
    }
    // Glazing conduction (glass + frame in series, per WindowSpec doc).
    let u_eff = spec.window_properties.u_value + spec.window_properties.frame_u_value;
    let mut cond_attrs = attr("glazing", 1.0);
    cond_attrs.insert("u_value_w_per_m2k".to_string(), u_eff);
    cond_attrs.insert("window_area_m2".to_string(), w.area);
    graph.push_edge(TopologyEdge {
        source_id: outdoor.to_string(),
        target_id: format!("zone-{zi}:air"),
        coupling_type: TopologyEdgeKind::Conduction,
        conductance_w_per_k: Some(u_eff * w.area),
        fraction: None,
        bidirectional: true,
        attributes: cond_attrs,
    });
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

            // Windows (solar + glazing conduction).
            if let Some(zone_windows) = self.windows.get(zi) {
                for w in zone_windows {
                    push_window(&mut graph, zi, w, self, &hosts);
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
        // (ext→l0, l_j→l_{j+1}, l_last→int); windows add one glazing
        // conduction edge each.
        let expected: usize = (spec.construction.wall.layers.len() + 1) * 4
            + (spec.construction.roof.layers.len() + 1)
            + (spec.construction.floor.layers.len() + 1)
            + spec
                .windows
                .iter()
                .map(|z| z.iter().filter(|w| w.area > 0.0).count())
                .sum::<usize>();
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
        assert!(common
            .iter()
            .all(|e| e.source_id.starts_with("zone-") && e.target_id.starts_with("zone-")));
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
