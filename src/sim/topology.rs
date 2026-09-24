//! In-memory simulation topology graph (Issue #3963).
//!
//! This module defines the engine-agnostic topology representation used by
//! `fluxion topology export`: a directed (optionally bidirectional) graph whose
//! nodes are thermal-network elements (`zone_air`, `exterior_surface`,
//! `wall_layer`, `interior_surface`, `internal_mass`, `internal_gain`,
//! `hvac_terminal`, `outdoor_ambient`) and whose edges are heat-transfer
//! couplings (`conduction`, `convection_exterior`, `convection_interior`,
//! `longwave_radiation`, `shortwave_solar_direct`, `shortwave_solar_diffuse`,
//! `air_exchange`, `internal_gain_split`, `hvac_sensible`).
//!
//! The module lives in `sim/` and must stay dependency-light: it must not
//! import from `crate::validation` (cycle rule enforced by
//! `scripts/check_ashrae_cases_cycle.py`). Model-specific bridges — e.g. the
//! ASHRAE 140 `CaseSpec` bridge — implement [`ToTopologyGraph`] next to their
//! model definitions.
//!
//! Determinism contract: identical inputs produce byte-identical JSON. Struct
//! field order is fixed by declaration, node/edge arrays are sorted by stable
//! ids in [`TopologyGraph::finalize`], and numeric attributes serialize through
//! `serde_json`'s shortest round-trip formatting. The export timestamp is
//! deliberately optional (`None` by default) so repeated exports are stable.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Schema version of the exported topology document (`metadata.schema_version`).
pub const TOPOLOGY_SCHEMA_VERSION: &str = "1.0.0";

/// Thermal-network node kinds (Issue #3963 §"Node kinds").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TopologyNodeKind {
    /// Outdoor ambient boundary (dry-bulb + solar source).
    OutdoorAmbient,
    /// Exterior-facing film node of an opaque surface.
    ExteriorSurface,
    /// A discrete material layer inside a construction assembly.
    WallLayer,
    /// Interior-facing film node of a surface.
    InteriorSurface,
    /// Zone radiant-mass star node (5R1C `T_m` analogue).
    InternalMass,
    /// Zone air capacitance node.
    ZoneAir,
    /// Internal heat gain source (split convective/radiative).
    InternalGain,
    /// HVAC sensible injection/extraction terminal.
    HvacTerminal,
}

impl TopologyNodeKind {
    /// Stable snake_case id used for schema enum membership checks.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OutdoorAmbient => "outdoor_ambient",
            Self::ExteriorSurface => "exterior_surface",
            Self::WallLayer => "wall_layer",
            Self::InteriorSurface => "interior_surface",
            Self::InternalMass => "internal_mass",
            Self::ZoneAir => "zone_air",
            Self::InternalGain => "internal_gain",
            Self::HvacTerminal => "hvac_terminal",
        }
    }
}

/// Heat-transfer coupling kinds between topology nodes (Issue #3963
/// §"Edge kinds", plus `hvac_sensible` for HVAC terminal delivery).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TopologyEdgeKind {
    /// Conduction through an opaque assembly layer.
    Conduction,
    /// Exterior film convection (surface ↔ outdoor air).
    ConvectionExterior,
    /// Interior film convection (surface ↔ zone air).
    ConvectionInterior,
    /// Longwave radiation exchange (surface ↔ radiant mass).
    LongwaveRadiation,
    /// Beam solar admission through a window.
    ShortwaveSolarDirect,
    /// Diffuse solar admission through a window.
    ShortwaveSolarDiffuse,
    /// Air exchange (infiltration / night ventilation / inter-zone).
    AirExchange,
    /// Internal gain split path (convective or radiative branch).
    InternalGainSplit,
    /// HVAC sensible coupling (terminal ↔ zone air).
    HvacSensible,
}

impl TopologyEdgeKind {
    /// Stable snake_case id used for schema enum membership checks.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Conduction => "conduction",
            Self::ConvectionExterior => "convection_exterior",
            Self::ConvectionInterior => "convection_interior",
            Self::LongwaveRadiation => "longwave_radiation",
            Self::ShortwaveSolarDirect => "shortwave_solar_direct",
            Self::ShortwaveSolarDiffuse => "shortwave_solar_diffuse",
            Self::AirExchange => "air_exchange",
            Self::InternalGainSplit => "internal_gain_split",
            Self::HvacSensible => "hvac_sensible",
        }
    }
}

/// A topology graph node. Numeric attributes are `None` when not applicable to
/// the node kind (serialized as absent, not `null`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyNode {
    /// Stable unique node id (e.g. `zone-0:air`).
    pub id: String,
    /// Node classification.
    pub kind: TopologyNodeKind,
    /// Owning zone index/name, when the node belongs to a zone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zone_id: Option<String>,
    /// Human-readable name.
    pub name: String,
    /// Thermal capacitance (J/K).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capacitance_j_per_k: Option<f64>,
    /// Reference surface area (m²).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub area_m2: Option<f64>,
    /// Reference volume (m³).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume_m3: Option<f64>,
    /// Surface azimuth (degrees clockwise from North; absent for flat/ground surfaces).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub azimuth_deg: Option<f64>,
    /// Surface tilt (degrees from horizontal; 90 = vertical wall).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tilt_deg: Option<f64>,
    /// Node elevation (m above grade).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elevation_m: Option<f64>,
    /// Extensible numeric attributes (HVAC setpoints, ACH, shading flags...).
    /// `BTreeMap` keeps key order deterministic.
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub attributes: BTreeMap<String, f64>,
}

/// A heat-transfer coupling between two nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyEdge {
    /// Source node id.
    pub source_id: String,
    /// Target node id.
    pub target_id: String,
    /// Coupling classification.
    pub coupling_type: TopologyEdgeKind,
    /// Coupling conductance (W/K); `None` for non-conductive couplings
    /// (e.g. solar admission, unresisted radiation star edges).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conductance_w_per_k: Option<f64>,
    /// Coupling fraction (e.g. SHGC, gain split fraction); `None` when N/A.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fraction: Option<f64>,
    /// Whether heat flow is modeled in both directions.
    #[serde(default)]
    pub bidirectional: bool,
    /// Extensible numeric attributes (setpoints, schedule hours, ACH...).
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub attributes: BTreeMap<String, f64>,
}

/// Document metadata block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyMetadata {
    /// Topology schema version ("1.0.0").
    pub schema_version: String,
    /// Exporting engine version (`CARGO_PKG_VERSION`).
    pub tool_version: String,
    /// Name of the exported model.
    pub model_name: String,
    /// Provenance of the model ("ashrae-140-registry:600", file path...).
    pub model_source: String,
    /// Export timestamp. Optional by design: omitted unless explicitly set,
    /// so identical inputs yield byte-identical output (Issue #3963 §"Deterministic
    /// serialization").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    /// Number of nodes after finalization.
    pub node_count: usize,
    /// Number of edges after finalization.
    pub edge_count: usize,
}

/// Build context passed to [`ToTopologyGraph`] implementations.
#[derive(Debug, Clone)]
pub struct TopologyContext {
    /// Name of the exported model.
    pub model_name: String,
    /// Provenance of the model.
    pub model_source: String,
    /// Optional explicit timestamp (omitted by default for determinism).
    pub timestamp: Option<String>,
}

impl TopologyContext {
    /// Creates a context with no timestamp.
    pub fn new(model_name: impl Into<String>, model_source: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            model_source: model_source.into(),
            timestamp: None,
        }
    }
}

/// The complete exported topology document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyGraph {
    /// Document metadata.
    pub metadata: TopologyMetadata,
    /// Sorted-by-id node array.
    pub nodes: Vec<TopologyNode>,
    /// Sorted-by-(source, target, kind) edge array.
    pub edges: Vec<TopologyEdge>,
}

/// Conversion from a simulation model into a [`TopologyGraph`].
///
/// Implementations live next to their model types (e.g. the ASHRAE 140
/// `CaseSpec` bridge in `src/validation/topology_bridge.rs`) so `sim` never
/// imports from `validation`.
pub trait ToTopologyGraph {
    /// Builds the topology graph for this model.
    fn to_topology_graph(&self, ctx: &TopologyContext) -> TopologyGraph;
}

impl TopologyGraph {
    /// Creates an empty graph with metadata for the given context.
    pub fn new(ctx: &TopologyContext) -> Self {
        Self {
            metadata: TopologyMetadata {
                schema_version: TOPOLOGY_SCHEMA_VERSION.to_string(),
                tool_version: env!("CARGO_PKG_VERSION").to_string(),
                model_name: ctx.model_name.clone(),
                model_source: ctx.model_source.clone(),
                timestamp: ctx.timestamp.clone(),
                node_count: 0,
                edge_count: 0,
            },
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Appends a node.
    pub fn push_node(&mut self, node: TopologyNode) {
        self.nodes.push(node);
    }

    /// Appends an edge.
    pub fn push_edge(&mut self, edge: TopologyEdge) {
        self.edges.push(edge);
    }

    /// Finalizes the document: sorts node/edge arrays by stable keys and
    /// refreshes metadata counts. Returns an error on duplicate node ids.
    pub fn finalize(&mut self) -> Result<(), String> {
        let mut seen = std::collections::BTreeSet::new();
        for n in &self.nodes {
            if !seen.insert(n.id.clone()) {
                return Err(format!("duplicate topology node id: {}", n.id));
            }
        }
        self.nodes.sort_by(|a, b| a.id.cmp(&b.id));
        self.edges.sort_by(|a, b| {
            (&a.source_id, &a.target_id, a.coupling_type.as_str()).cmp(&(
                &b.source_id,
                &b.target_id,
                b.coupling_type.as_str(),
            ))
        });
        self.metadata.node_count = self.nodes.len();
        self.metadata.edge_count = self.edges.len();
        Ok(())
    }

    /// Structural validation: unique ids, edge endpoints resolve, zone
    /// references resolve, fractions in [0, 1], conductances/capacitances
    /// finite and positive where present.
    pub fn validate(&self) -> Result<(), String> {
        if self.metadata.schema_version != TOPOLOGY_SCHEMA_VERSION {
            return Err(format!(
                "schema_version mismatch: {} != {}",
                self.metadata.schema_version, TOPOLOGY_SCHEMA_VERSION
            ));
        }
        if self.metadata.node_count != self.nodes.len()
            || self.metadata.edge_count != self.edges.len()
        {
            return Err("metadata node/edge counts do not match arrays".to_string());
        }
        let ids: std::collections::BTreeSet<&str> =
            self.nodes.iter().map(|n| n.id.as_str()).collect();
        let zones: std::collections::BTreeSet<&str> = self
            .nodes
            .iter()
            .filter(|n| n.kind == TopologyNodeKind::ZoneAir)
            .map(|n| n.id.as_str())
            .collect();
        for n in &self.nodes {
            if !ids.contains(n.id.as_str()) {
                return Err(format!("node id not in id set: {}", n.id));
            }
            if let Some(z) = &n.zone_id {
                if !zones.contains(z.as_str()) {
                    return Err(format!("node {} references unknown zone_id {}", n.id, z));
                }
            }
            if let Some(c) = n.capacitance_j_per_k {
                if !c.is_finite() || c < 0.0 {
                    return Err(format!("node {} has invalid capacitance {}", n.id, c));
                }
            }
            for (k, v) in &n.attributes {
                if !v.is_finite() {
                    return Err(format!("node {} has non-finite attribute {}", n.id, k));
                }
            }
        }
        for e in &self.edges {
            if !ids.contains(e.source_id.as_str()) {
                return Err(format!("edge source {} does not resolve", e.source_id));
            }
            if !ids.contains(e.target_id.as_str()) {
                return Err(format!("edge target {} does not resolve", e.target_id));
            }
            if let Some(f) = e.fraction {
                if !f.is_finite() || !(0.0..=1.0).contains(&f) {
                    return Err(format!(
                        "edge {}->{} has fraction out of [0,1]: {}",
                        e.source_id, e.target_id, f
                    ));
                }
            }
            if let Some(g) = e.conductance_w_per_k {
                if !g.is_finite() || g <= 0.0 {
                    return Err(format!(
                        "edge {}->{} has invalid conductance {}",
                        e.source_id, e.target_id, g
                    ));
                }
            }
            for (k, v) in &e.attributes {
                if !v.is_finite() {
                    return Err(format!(
                        "edge {}->{} has non-finite attribute {}",
                        e.source_id, e.target_id, k
                    ));
                }
            }
        }
        Ok(())
    }

    /// Deterministic pretty-printed JSON serialization.
    pub fn to_json_string(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, kind: TopologyNodeKind) -> TopologyNode {
        TopologyNode {
            id: id.to_string(),
            kind,
            zone_id: None,
            name: id.to_string(),
            capacitance_j_per_k: None,
            area_m2: None,
            volume_m3: None,
            azimuth_deg: None,
            tilt_deg: None,
            elevation_m: None,
            attributes: BTreeMap::new(),
        }
    }

    fn edge(src: &str, dst: &str, kind: TopologyEdgeKind) -> TopologyEdge {
        TopologyEdge {
            source_id: src.to_string(),
            target_id: dst.to_string(),
            coupling_type: kind,
            conductance_w_per_k: Some(1.0),
            fraction: None,
            bidirectional: false,
            attributes: BTreeMap::new(),
        }
    }

    #[test]
    fn finalize_sorts_and_counts() {
        let mut g = TopologyGraph::new(&TopologyContext::new("t", "test"));
        g.push_node(node("b", TopologyNodeKind::ZoneAir));
        g.push_node(node("a", TopologyNodeKind::OutdoorAmbient));
        g.push_edge(edge("b", "a", TopologyEdgeKind::Conduction));
        g.push_edge(edge("a", "b", TopologyEdgeKind::ConvectionExterior));
        g.finalize().unwrap();
        assert_eq!(g.nodes[0].id, "a");
        assert_eq!(g.nodes[1].id, "b");
        assert_eq!(
            g.edges[0].coupling_type,
            TopologyEdgeKind::ConvectionExterior
        );
        assert_eq!(g.metadata.node_count, 2);
        assert_eq!(g.metadata.edge_count, 2);
        g.validate().unwrap();
    }

    #[test]
    fn finalize_rejects_duplicate_ids() {
        let mut g = TopologyGraph::new(&TopologyContext::new("t", "test"));
        g.push_node(node("x", TopologyNodeKind::ZoneAir));
        g.push_node(node("x", TopologyNodeKind::ZoneAir));
        assert!(g.finalize().is_err());
    }

    #[test]
    fn validate_rejects_dangling_edge() {
        let mut g = TopologyGraph::new(&TopologyContext::new("t", "test"));
        g.push_node(node("a", TopologyNodeKind::ZoneAir));
        g.push_edge(edge("a", "ghost", TopologyEdgeKind::Conduction));
        g.finalize().unwrap();
        assert!(g.validate().is_err());
    }

    #[test]
    fn validate_rejects_fraction_out_of_range() {
        let mut g = TopologyGraph::new(&TopologyContext::new("t", "test"));
        g.push_node(node("a", TopologyNodeKind::ZoneAir));
        g.push_node(node("b", TopologyNodeKind::InternalGain));
        let mut e = edge("b", "a", TopologyEdgeKind::InternalGainSplit);
        e.conductance_w_per_k = None;
        e.fraction = Some(1.5);
        g.push_edge(e);
        g.finalize().unwrap();
        assert!(g.validate().is_err());
    }

    #[test]
    fn validate_rejects_unknown_zone_reference() {
        let mut g = TopologyGraph::new(&TopologyContext::new("t", "test"));
        let mut n = node("gain", TopologyNodeKind::InternalGain);
        n.zone_id = Some("zone-9:air".to_string());
        g.push_node(node("z", TopologyNodeKind::ZoneAir));
        g.push_node(n);
        g.finalize().unwrap();
        assert!(g.validate().is_err());
    }

    #[test]
    fn serialization_is_deterministic_and_timestamp_optional() {
        let build = || {
            let mut g = TopologyGraph::new(&TopologyContext::new("t", "test"));
            g.push_node(node("ambient", TopologyNodeKind::OutdoorAmbient));
            g.push_node(node("zone-0:air", TopologyNodeKind::ZoneAir));
            let mut n = node("zone-0:gain", TopologyNodeKind::InternalGain);
            n.zone_id = Some("zone-0:air".to_string());
            n.attributes.insert("total_load_w".to_string(), 200.0);
            g.push_node(n);
            g.push_edge(edge("ambient", "zone-0:air", TopologyEdgeKind::Conduction));
            g.finalize().unwrap();
            g
        };
        let a = build().to_json_string().unwrap();
        let b = build().to_json_string().unwrap();
        assert_eq!(a, b, "identical inputs must produce byte-identical JSON");
        assert!(!a.contains("timestamp"), "timestamp omitted by default");
        assert!(a.contains("\"schema_version\": \"1.0.0\""));
        assert!(a.contains("\"tool_version\""));
    }

    #[test]
    fn node_and_edge_kinds_use_snake_case() {
        assert_eq!(TopologyNodeKind::WallLayer.as_str(), "wall_layer");
        assert_eq!(
            TopologyEdgeKind::ShortwaveSolarDirect.as_str(),
            "shortwave_solar_direct"
        );
        let s = serde_json::to_string(&TopologyEdgeKind::AirExchange).unwrap();
        assert_eq!(s, "\"air_exchange\"");
        let s = serde_json::to_string(&TopologyNodeKind::HvacTerminal).unwrap();
        assert_eq!(s, "\"hvac_terminal\"");
    }
}
