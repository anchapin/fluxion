//! Simulation topology validation linter (Issue #3964).
//!
//! Pure `&TopologyGraph` analysis: nine rules — seven errors (E-codes) and
//! two warnings (W-codes) — that flag structurally suspicious topologies
//! before simulation. The linter is deliberately offline: it never touches
//! the ASHRAE 140 registry or the model builder, so it can lint any
//! previously exported topology document (`fluxion topology lint --input`).
//!
//! Rule codes are stable, machine-readable identifiers: never renumber or
//! rename them, only extend the taxonomy (the JSON output is consumed by
//! tooling built on top of #3966's export pipeline).
//!
//! - `E001_ORPHAN_NODE` — node with zero incident edges
//! - `E002_DANGLING_BOUNDARY` — exterior surface with neither an ambient
//!   convection-film edge nor a ground-coupling attribute; interior surface
//!   without a convection-film edge to its zone air
//! - `E003_ORPHAN_MASS` — radiant mass with neither convective coupling to
//!   its zone air nor radiative coupling to an enclosure surface
//! - `E004_RECIPROCAL_MISMATCH` — inter-zone coupling without a matching
//!   reciprocal, or with mismatched conductance/area between antiparallel legs
//! - `E005_INVALID_SPLIT_FRACTION` — internal-gain split fractions that do
//!   not sum to 1.0 (± [`FRACTION_SUM_TOLERANCE`])
//! - `E006_UNRESOLVED_RADIANT_TARGET` — radiative gain path with no
//!   receiving interior-surface node in the zone
//! - `E007_NON_POSITIVE_CAPACITY` — ZoneAir/InternalMass node with a
//!   non-positive (or non-finite) thermal capacity
//! - `W001_EXTREME_FILM_COEFFICIENT` — convection film coefficient
//!   h_c = G/A outside (`FILM_COEFF_MIN_W_M2K`, `FILM_COEFF_MAX_W_M2K`)
//! - `W002_HIGH_CONDUCTANCE_RATIO` — adjacent-edge conductance ratio at a
//!   node exceeding [`CONDUCTANCE_RATIO_MAX`]

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::topology::{
    TopologyEdge, TopologyEdgeKind, TopologyGraph, TopologyNode, TopologyNodeKind,
};

/// Tolerance for the E005 gain-split fraction sum (1.0 ± 1e-4).
pub const FRACTION_SUM_TOLERANCE: f64 = 1e-4;
/// Lower plausible bound for a convection film coefficient (W/m²K).
pub const FILM_COEFF_MIN_W_M2K: f64 = 0.1;
/// Upper plausible bound for a convection film coefficient (W/m²K).
pub const FILM_COEFF_MAX_W_M2K: f64 = 100.0;
/// Conductance-ratio threshold at a single node (max/min incident edges).
pub const CONDUCTANCE_RATIO_MAX: f64 = 1e6;
/// Relative tolerance for E004 antiparallel conductance parity.
const CONDUCTANCE_PARITY_TOLERANCE: f64 = 1e-6;

/// Finding severity. Warnings only block the exit code under `--strict`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum LintSeverity {
    /// Structural defect: the topology cannot be trusted.
    Error,
    /// Suspicious but simulable.
    Warning,
}

/// Reference to the edge a finding belongs to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LintEdgeRef {
    /// Source node id of the offending edge.
    pub source: String,
    /// Target node id of the offending edge.
    pub target: String,
}

/// A single linter finding with a stable machine-readable code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LintFinding {
    /// Stable rule code (e.g. `E001_ORPHAN_NODE`).
    pub code: String,
    /// ERROR or WARNING.
    pub severity: LintSeverity,
    /// Human-readable description.
    pub message: String,
    /// Offending node id, when the finding is node-scoped.
    #[serde(rename = "node", skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    /// Offending edge reference, when the finding is edge-scoped.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge: Option<LintEdgeRef>,
}

impl LintFinding {
    fn new(
        code: &str,
        severity: LintSeverity,
        message: String,
        node_id: Option<String>,
        edge: Option<LintEdgeRef>,
    ) -> Self {
        Self {
            code: code.to_string(),
            severity,
            message,
            node_id,
            edge,
        }
    }

    /// Sort key: code first, then node/edge reference, for deterministic
    /// output regardless of graph insertion order.
    fn sort_key(&self) -> (String, String, String) {
        (
            self.code.clone(),
            self.node_id.clone().unwrap_or_default(),
            self.edge
                .as_ref()
                .map(|e| format!("{}->{}", e.source, e.target))
                .unwrap_or_default(),
        )
    }
}

/// Result of linting a topology graph.
#[derive(Debug, Clone, Default, Serialize)]
pub struct LintReport {
    /// All findings, sorted by (code, node, edge).
    pub findings: Vec<LintFinding>,
}

impl LintReport {
    /// Number of ERROR-severity findings.
    pub fn error_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == LintSeverity::Error)
            .count()
    }

    /// Number of WARNING-severity findings.
    pub fn warning_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == LintSeverity::Warning)
            .count()
    }

    /// True when the topology produced no findings at all.
    pub fn is_clean(&self) -> bool {
        self.findings.is_empty()
    }

    /// True when the report should fail the run: any error, or any warning
    /// once `strict` elevates warnings to errors.
    pub fn has_blocking(&self, strict: bool) -> bool {
        self.error_count() > 0 || (strict && self.warning_count() > 0)
    }
}

/// Incident edge with the direction flag (`true` = node is the edge source).
type IncidentEdge<'a> = (&'a TopologyEdge, bool);

fn incident_map(graph: &TopologyGraph) -> HashMap<&str, Vec<IncidentEdge<'_>>> {
    let mut map: HashMap<&str, Vec<IncidentEdge<'_>>> = HashMap::new();
    for edge in &graph.edges {
        map.entry(edge.source_id.as_str())
            .or_default()
            .push((edge, true));
        map.entry(edge.target_id.as_str())
            .or_default()
            .push((edge, false));
    }
    map
}

fn node_map(graph: &TopologyGraph) -> HashMap<&str, &TopologyNode> {
    graph.nodes.iter().map(|n| (n.id.as_str(), n)).collect()
}

/// The node on the other side of `edge` from the incident endpoint.
fn other_end<'a>(
    nodes: &HashMap<&str, &'a TopologyNode>,
    edge: &TopologyEdge,
    is_source: bool,
) -> Option<&'a TopologyNode> {
    let other = if is_source {
        &edge.target_id
    } else {
        &edge.source_id
    };
    nodes.get(other.as_str()).copied()
}

fn edge_ref(edge: &TopologyEdge) -> LintEdgeRef {
    LintEdgeRef {
        source: edge.source_id.clone(),
        target: edge.target_id.clone(),
    }
}

/// Stable human label for a node kind (the kind enum itself is not `Display`).
fn kind_label(kind: &TopologyNodeKind) -> &'static str {
    match kind {
        TopologyNodeKind::OutdoorAmbient => "outdoor_ambient",
        TopologyNodeKind::ExteriorSurface => "exterior_surface",
        TopologyNodeKind::WallLayer => "wall_layer",
        TopologyNodeKind::InteriorSurface => "interior_surface",
        TopologyNodeKind::InternalMass => "internal_mass",
        TopologyNodeKind::ZoneAir => "zone_air",
        TopologyNodeKind::InternalGain => "internal_gain",
        TopologyNodeKind::HvacTerminal => "hvac_terminal",
    }
}

/// Lint a topology graph with the full #3964 rule set.
pub fn lint_topology(graph: &TopologyGraph) -> LintReport {
    let nodes = node_map(graph);
    let incident = incident_map(graph);
    let mut findings = Vec::new();

    e001_orphan_nodes(graph, &incident, &mut findings);
    e002_dangling_boundary(graph, &nodes, &incident, &mut findings);
    e003_orphan_mass(graph, &nodes, &incident, &mut findings);
    e004_reciprocal_mismatch(graph, &nodes, &mut findings);
    e005_invalid_split_fraction(graph, &incident, &mut findings);
    e006_unresolved_radiant_target(graph, &nodes, &mut findings);
    e007_non_positive_capacity(graph, &mut findings);
    w001_extreme_film_coefficient(graph, &nodes, &mut findings);
    w002_high_conductance_ratio(graph, &incident, &mut findings);

    findings.sort_by_key(LintFinding::sort_key);
    LintReport { findings }
}

/// E001: every node must participate in at least one edge.
fn e001_orphan_nodes(
    graph: &TopologyGraph,
    incident: &HashMap<&str, Vec<IncidentEdge<'_>>>,
    findings: &mut Vec<LintFinding>,
) {
    for node in &graph.nodes {
        let degree = incident.get(node.id.as_str()).map_or(0, Vec::len);
        if degree == 0 {
            findings.push(LintFinding::new(
                "E001_ORPHAN_NODE",
                LintSeverity::Error,
                format!(
                    "node `{}` ({}) has no incident edges; it is invisible to the solver",
                    node.id,
                    kind_label(&node.kind)
                ),
                Some(node.id.clone()),
                None,
            ));
        }
    }
}

/// E002: boundary surfaces must be coupled to their boundary.
fn e002_dangling_boundary(
    graph: &TopologyGraph,
    nodes: &HashMap<&str, &TopologyNode>,
    incident: &HashMap<&str, Vec<IncidentEdge<'_>>>,
    findings: &mut Vec<LintFinding>,
) {
    for node in &graph.nodes {
        let edges = incident
            .get(node.id.as_str())
            .map_or(&[][..], |v| v.as_slice());
        match node.kind {
            TopologyNodeKind::ExteriorSurface => {
                let has_ambient_film = edges
                    .iter()
                    .any(|(e, _)| e.coupling_type == TopologyEdgeKind::ConvectionExterior);
                let ground_coupled = node
                    .attributes
                    .get("ground_coupled")
                    .is_some_and(|v| *v > 0.5);
                if !has_ambient_film && !ground_coupled {
                    findings.push(LintFinding::new(
                        "E002_DANGLING_BOUNDARY",
                        LintSeverity::Error,
                        format!(
                            "exterior surface `{}` has neither an ambient convection-film edge \
                             nor a ground-coupling attribute",
                            node.id
                        ),
                        Some(node.id.clone()),
                        None,
                    ));
                }
            }
            TopologyNodeKind::InteriorSurface => {
                let has_air_film = edges.iter().any(|(e, is_src)| {
                    e.coupling_type == TopologyEdgeKind::ConvectionInterior
                        && other_end(nodes, e, *is_src)
                            .is_some_and(|n| n.kind == TopologyNodeKind::ZoneAir)
                });
                if !has_air_film {
                    findings.push(LintFinding::new(
                        "E002_DANGLING_BOUNDARY",
                        LintSeverity::Error,
                        format!(
                            "interior surface `{}` has no convection-film edge to a zone-air node",
                            node.id
                        ),
                        Some(node.id.clone()),
                        None,
                    ));
                }
            }
            _ => {}
        }
    }
}

/// E003: radiant mass must exchange heat with the zone (convective path to
/// zone air or radiative path to an enclosure surface).
fn e003_orphan_mass(
    graph: &TopologyGraph,
    nodes: &HashMap<&str, &TopologyNode>,
    incident: &HashMap<&str, Vec<IncidentEdge<'_>>>,
    findings: &mut Vec<LintFinding>,
) {
    for node in &graph.nodes {
        if node.kind != TopologyNodeKind::InternalMass {
            continue;
        }
        let edges = incident
            .get(node.id.as_str())
            .map_or(&[][..], |v| v.as_slice());
        let has_convective = edges.iter().any(|(e, is_src)| {
            other_end(nodes, e, *is_src).is_some_and(|n| n.kind == TopologyNodeKind::ZoneAir)
        });
        let has_radiative = edges.iter().any(|(e, is_src)| {
            e.coupling_type == TopologyEdgeKind::LongwaveRadiation
                && other_end(nodes, e, *is_src).is_some_and(|n| {
                    n.kind == TopologyNodeKind::InteriorSurface
                        || n.kind == TopologyNodeKind::ExteriorSurface
                })
        });
        if !has_convective && !has_radiative {
            findings.push(LintFinding::new(
                "E003_ORPHAN_MASS",
                LintSeverity::Error,
                format!(
                    "radiant mass `{}` has neither convective coupling to a zone-air node nor \
                     radiative coupling to an enclosure surface",
                    node.id
                ),
                Some(node.id.clone()),
                None,
            ));
        }
    }
}

fn conductances_close(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => {
            (x - y).abs() <= CONDUCTANCE_PARITY_TOLERANCE * x.abs().max(y.abs()).max(1.0)
        }
        _ => false,
    }
}

/// E004: inter-zone couplings must be reciprocal (h_tr,ij == h_tr,ji).
fn e004_reciprocal_mismatch(
    graph: &TopologyGraph,
    nodes: &HashMap<&str, &TopologyNode>,
    findings: &mut Vec<LintFinding>,
) {
    // Unordered node-pair -> directed inter-zone air-coupling edges.
    let mut pairs: BTreeMap<(&str, &str), Vec<&TopologyEdge>> = BTreeMap::new();
    for edge in &graph.edges {
        let (Some(src), Some(dst)) = (
            nodes.get(edge.source_id.as_str()),
            nodes.get(edge.target_id.as_str()),
        ) else {
            continue;
        };
        if src.kind == TopologyNodeKind::ZoneAir
            && dst.kind == TopologyNodeKind::ZoneAir
            && edge.source_id != edge.target_id
        {
            let key = if edge.source_id <= edge.target_id {
                (edge.source_id.as_str(), edge.target_id.as_str())
            } else {
                (edge.target_id.as_str(), edge.source_id.as_str())
            };
            pairs.entry(key).or_default().push(edge);
        }
    }

    for ((a, b), edges) in pairs {
        // A single bidirectional edge is the canonical reciprocal form.
        let has_bidirectional = edges.iter().any(|e| e.bidirectional);
        if has_bidirectional {
            continue;
        }
        // Otherwise every directed leg needs an antiparallel partner with a
        // matching conductance (and area, when annotated on both legs).
        let reciprocal = edges.iter().all(|e| {
            edges.iter().any(|other| {
                other.source_id == e.target_id
                    && other.target_id == e.source_id
                    && conductances_close(e.conductance_w_per_k, other.conductance_w_per_k)
                    && conductances_close(
                        e.attributes.get("area_m2").copied(),
                        other.attributes.get("area_m2").copied(),
                    )
            })
        });
        if !reciprocal {
            let first = edges[0];
            findings.push(LintFinding::new(
                "E004_RECIPROCAL_MISMATCH",
                LintSeverity::Error,
                format!(
                    "inter-zone coupling between `{a}` and `{b}` lacks a matching reciprocal leg \
                     with equal conductance (h_tr,ij != h_tr,ji)"
                ),
                None,
                Some(edge_ref(first)),
            ));
        }
    }
}

/// E005: gain split fractions must partition unity.
fn e005_invalid_split_fraction(
    graph: &TopologyGraph,
    incident: &HashMap<&str, Vec<IncidentEdge<'_>>>,
    findings: &mut Vec<LintFinding>,
) {
    for node in &graph.nodes {
        if node.kind != TopologyNodeKind::InternalGain {
            continue;
        }
        let edges = incident
            .get(node.id.as_str())
            .map_or(&[][..], |v| v.as_slice());
        let fractions: Vec<f64> = edges
            .iter()
            .filter(|(e, _)| e.coupling_type == TopologyEdgeKind::InternalGainSplit)
            .filter_map(|(e, _)| e.fraction)
            .collect();
        if fractions.is_empty() {
            // No split edges at all is E001/E006 territory, not a bad sum.
            continue;
        }
        let sum: f64 = fractions.iter().sum();
        if (sum - 1.0).abs() > FRACTION_SUM_TOLERANCE {
            findings.push(LintFinding::new(
                "E005_INVALID_SPLIT_FRACTION",
                LintSeverity::Error,
                format!(
                    "internal gain `{}` splits sum to {sum:.6}, expected 1.000000 \
                     (±{FRACTION_SUM_TOLERANCE}); {} split edge(s) considered",
                    node.id,
                    fractions.len()
                ),
                Some(node.id.clone()),
                None,
            ));
        }
    }
}

/// E006: radiative gain paths need a receiving interior surface in the zone.
fn e006_unresolved_radiant_target(
    graph: &TopologyGraph,
    nodes: &HashMap<&str, &TopologyNode>,
    findings: &mut Vec<LintFinding>,
) {
    for edge in &graph.edges {
        if edge.coupling_type != TopologyEdgeKind::InternalGainSplit {
            continue;
        }
        let radiative = edge.fraction.is_some_and(|f| f > 0.0)
            && nodes.get(edge.target_id.as_str()).map(|n| n.kind)
                != Some(TopologyNodeKind::ZoneAir);
        if !radiative {
            continue;
        }
        let zone = nodes
            .get(edge.source_id.as_str())
            .and_then(|n| n.zone_id.as_deref())
            .or_else(|| {
                nodes
                    .get(edge.target_id.as_str())
                    .and_then(|n| n.zone_id.as_deref())
            });
        let has_receiver = graph.nodes.iter().any(|n| {
            n.zone_id.as_deref() == zone
                && (n.kind == TopologyNodeKind::InteriorSurface
                    || n.kind == TopologyNodeKind::InternalMass)
        });
        if !has_receiver {
            findings.push(LintFinding::new(
                "E006_UNRESOLVED_RADIANT_TARGET",
                LintSeverity::Error,
                format!(
                    "radiative gain split `{} -> {}` has no receiving interior-surface node \
                     in zone {:?}",
                    edge.source_id, edge.target_id, zone
                ),
                Some(edge.source_id.clone()),
                Some(edge_ref(edge)),
            ));
        }
    }
}

/// E007: dynamic nodes need positive thermal capacity when capacity is modeled.
fn e007_non_positive_capacity(graph: &TopologyGraph, findings: &mut Vec<LintFinding>) {
    for node in &graph.nodes {
        if !matches!(
            node.kind,
            TopologyNodeKind::ZoneAir | TopologyNodeKind::InternalMass
        ) {
            continue;
        }
        if let Some(c) = node.capacitance_j_per_k {
            if !c.is_finite() || c <= 0.0 {
                findings.push(LintFinding::new(
                    "E007_NON_POSITIVE_CAPACITY",
                    LintSeverity::Error,
                    format!(
                        "{} node `{}` has non-positive thermal capacity ({c} J/K)",
                        kind_label(&node.kind),
                        node.id
                    ),
                    Some(node.id.clone()),
                    None,
                ));
            }
        }
    }
}

/// W001: convection film coefficients must stay in a physically plausible band.
fn w001_extreme_film_coefficient(
    graph: &TopologyGraph,
    nodes: &HashMap<&str, &TopologyNode>,
    findings: &mut Vec<LintFinding>,
) {
    for edge in &graph.edges {
        if !matches!(
            edge.coupling_type,
            TopologyEdgeKind::ConvectionExterior | TopologyEdgeKind::ConvectionInterior
        ) {
            continue;
        }
        let Some(g) = edge.conductance_w_per_k else {
            continue;
        };
        if !g.is_finite() || g <= 0.0 {
            continue;
        }
        // The film endpoint: the surface node carrying a reference area.
        let area = nodes
            .get(edge.source_id.as_str())
            .into_iter()
            .chain(nodes.get(edge.target_id.as_str()))
            .find(|n| {
                matches!(
                    n.kind,
                    TopologyNodeKind::ExteriorSurface | TopologyNodeKind::InteriorSurface
                )
            })
            .and_then(|n| n.area_m2)
            .filter(|a| a.is_finite() && *a > 0.0);
        let Some(area) = area else {
            // No reference area modeled on the film endpoint: cannot evaluate.
            continue;
        };
        let h = g / area;
        if !(h > FILM_COEFF_MIN_W_M2K && h < FILM_COEFF_MAX_W_M2K) {
            findings.push(LintFinding::new(
                "W001_EXTREME_FILM_COEFFICIENT",
                LintSeverity::Warning,
                format!(
                    "convection edge `{} -> {}` implies film coefficient {h:.3} W/m²K, \
                     outside plausible band ({FILM_COEFF_MIN_W_M2K}, {FILM_COEFF_MAX_W_M2K})",
                    edge.source_id, edge.target_id
                ),
                None,
                Some(edge_ref(edge)),
            ));
        }
    }
}

/// W002: wildly different conductances meeting at one node usually mean a
/// unit or area bug (one edge silently dominating the timestep).
fn w002_high_conductance_ratio(
    graph: &TopologyGraph,
    incident: &HashMap<&str, Vec<IncidentEdge<'_>>>,
    findings: &mut Vec<LintFinding>,
) {
    for node in &graph.nodes {
        let edges = incident
            .get(node.id.as_str())
            .map_or(&[][..], |v| v.as_slice());
        let conductances: Vec<f64> = edges
            .iter()
            .filter_map(|(e, _)| e.conductance_w_per_k)
            .filter(|g| g.is_finite() && *g > 0.0)
            .collect();
        let (Some(&max), Some(&min)) = (
            conductances.iter().max_by(|a, b| a.total_cmp(b)),
            conductances.iter().min_by(|a, b| a.total_cmp(b)),
        ) else {
            continue;
        };
        if max / min > CONDUCTANCE_RATIO_MAX {
            findings.push(LintFinding::new(
                "W002_HIGH_CONDUCTANCE_RATIO",
                LintSeverity::Warning,
                format!(
                    "node `{}` couples edges spanning conductance {min:.3e}..{max:.3e} W/K \
                     (ratio {ratio:.3e} > {CONDUCTANCE_RATIO_MAX:.0e}); check areas/units",
                    node.id,
                    ratio = max / min
                ),
                Some(node.id.clone()),
                None,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{TopologyContext, TopologyGraph};

    fn node(id: &str, kind: TopologyNodeKind) -> TopologyNode {
        TopologyNode {
            id: id.to_string(),
            kind,
            zone_id: Some("zone-0".to_string()),
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

    fn edge(src: &str, dst: &str, kind: TopologyEdgeKind, g: Option<f64>) -> TopologyEdge {
        TopologyEdge {
            source_id: src.to_string(),
            target_id: dst.to_string(),
            coupling_type: kind,
            conductance_w_per_k: g,
            fraction: None,
            bidirectional: false,
            attributes: BTreeMap::new(),
        }
    }

    /// Minimal clean single-zone topology: ambient -> ext -> int -> air with
    /// a radiant mass hung off the interior surface.
    fn clean_graph() -> TopologyGraph {
        let mut g = TopologyGraph::new(&TopologyContext::new("lint-fixture", "test"));
        let mut ambient = node("ambient", TopologyNodeKind::OutdoorAmbient);
        ambient.zone_id = None;
        let mut air = node("zone-0:air", TopologyNodeKind::ZoneAir);
        air.capacitance_j_per_k = Some(1207.2);
        let mut ext = node("zone-0:wall-north:ext", TopologyNodeKind::ExteriorSurface);
        ext.area_m2 = Some(10.0);
        let mut int = node("zone-0:wall-north:int", TopologyNodeKind::InteriorSurface);
        int.area_m2 = Some(10.0);
        let mass = node("zone-0:mass", TopologyNodeKind::InternalMass);
        for n in [ambient, ext.clone(), int.clone(), air, mass] {
            g.push_node(n);
        }
        g.push_edge(edge(
            "ambient",
            &ext.id,
            TopologyEdgeKind::ConvectionExterior,
            Some(183.0),
        ));
        g.push_edge(edge(
            "ambient",
            "zone-0:air",
            TopologyEdgeKind::AirExchange,
            Some(30.0),
        ));
        g.push_edge(edge(
            &ext.id,
            &int.id,
            TopologyEdgeKind::Conduction,
            Some(100.0),
        ));
        g.push_edge(edge(
            &int.id,
            "zone-0:air",
            TopologyEdgeKind::ConvectionInterior,
            Some(82.9),
        ));
        g.push_edge(edge(
            &int.id,
            "zone-0:mass",
            TopologyEdgeKind::LongwaveRadiation,
            None,
        ));
        g
    }

    fn codes(report: &LintReport) -> Vec<String> {
        report.findings.iter().map(|f| f.code.clone()).collect()
    }

    #[test]
    fn clean_fixture_lints_clean() {
        let report = lint_topology(&clean_graph());
        assert!(
            report.is_clean(),
            "expected clean report, got {:?}",
            report.findings
        );
    }

    #[test]
    fn e001_fires_on_edgeless_node() {
        let mut g = clean_graph();
        g.push_node(node("zone-0:ghost", TopologyNodeKind::InternalGain));
        let report = lint_topology(&g);
        assert_eq!(codes(&report), vec!["E001_ORPHAN_NODE"]);
        assert_eq!(report.findings[0].node_id.as_deref(), Some("zone-0:ghost"));
    }

    #[test]
    fn e002_fires_when_exterior_film_removed() {
        let mut g = clean_graph();
        g.edges
            .retain(|e| e.coupling_type != TopologyEdgeKind::ConvectionExterior);
        let report = lint_topology(&g);
        assert_eq!(codes(&report), vec!["E002_DANGLING_BOUNDARY"]);
        assert_eq!(
            report.findings[0].node_id.as_deref(),
            Some("zone-0:wall-north:ext")
        );
    }

    #[test]
    fn e002_accepts_ground_coupled_exterior_surface() {
        let mut g = clean_graph();
        g.edges
            .retain(|e| e.coupling_type != TopologyEdgeKind::ConvectionExterior);
        let ext = g
            .nodes
            .iter_mut()
            .find(|n| n.kind == TopologyNodeKind::ExteriorSurface)
            .unwrap();
        ext.attributes.insert("ground_coupled".to_string(), 1.0);
        let report = lint_topology(&g);
        assert!(report.is_clean(), "{:?}", report.findings);
    }

    #[test]
    fn e003_fires_on_uncoupled_mass() {
        let mut g = clean_graph();
        // Redirect the mass's only edge to the ambient node: still coupled to
        // *something* (no E001) but neither to zone air nor an enclosure surface.
        let idx = g
            .edges
            .iter()
            .position(|e| e.coupling_type == TopologyEdgeKind::LongwaveRadiation)
            .unwrap();
        g.edges[idx].source_id = "ambient".to_string();
        let report = lint_topology(&g);
        assert_eq!(codes(&report), vec!["E003_ORPHAN_MASS"]);
    }

    #[test]
    fn e004_fires_on_one_way_interzone_edge() {
        let mut g = clean_graph();
        let mut z1 = node("zone-1:air", TopologyNodeKind::ZoneAir);
        z1.capacitance_j_per_k = Some(900.0);
        g.push_node(z1);
        let mut coupled = edge(
            "zone-0:air",
            "zone-1:air",
            TopologyEdgeKind::Conduction,
            Some(50.0),
        );
        coupled.bidirectional = false;
        g.push_edge(coupled);
        let report = lint_topology(&g);
        assert_eq!(codes(&report), vec!["E004_RECIPROCAL_MISMATCH"]);
    }

    #[test]
    fn e004_accepts_bidirectional_interzone_edge() {
        let mut g = clean_graph();
        let mut z1 = node("zone-1:air", TopologyNodeKind::ZoneAir);
        z1.capacitance_j_per_k = Some(900.0);
        g.push_node(z1);
        let mut coupled = edge(
            "zone-0:air",
            "zone-1:air",
            TopologyEdgeKind::Conduction,
            Some(50.0),
        );
        coupled.bidirectional = true;
        g.push_edge(coupled);
        let report = lint_topology(&g);
        assert!(report.is_clean(), "{:?}", report.findings);
    }

    #[test]
    fn e004_fires_on_conductance_asymmetric_antiparallel_legs() {
        let mut g = clean_graph();
        let mut z1 = node("zone-1:air", TopologyNodeKind::ZoneAir);
        z1.capacitance_j_per_k = Some(900.0);
        g.push_node(z1);
        g.push_edge(edge(
            "zone-0:air",
            "zone-1:air",
            TopologyEdgeKind::Conduction,
            Some(50.0),
        ));
        g.push_edge(edge(
            "zone-1:air",
            "zone-0:air",
            TopologyEdgeKind::Conduction,
            Some(40.0),
        ));
        let report = lint_topology(&g);
        assert_eq!(codes(&report), vec!["E004_RECIPROCAL_MISMATCH"]);
    }

    #[test]
    fn e005_fires_on_bad_split_sum() {
        let mut g = clean_graph();
        g.push_node(node("zone-0:gain", TopologyNodeKind::InternalGain));
        let mut to_air = edge(
            "zone-0:gain",
            "zone-0:air",
            TopologyEdgeKind::InternalGainSplit,
            None,
        );
        to_air.fraction = Some(0.4);
        let mut to_mass = edge(
            "zone-0:gain",
            "zone-0:mass",
            TopologyEdgeKind::InternalGainSplit,
            None,
        );
        to_mass.fraction = Some(0.5);
        g.push_edge(to_air);
        g.push_edge(to_mass);
        let report = lint_topology(&g);
        assert_eq!(codes(&report), vec!["E005_INVALID_SPLIT_FRACTION"]);
    }

    #[test]
    fn e005_accepts_unity_split_sum() {
        let mut g = clean_graph();
        g.push_node(node("zone-0:gain", TopologyNodeKind::InternalGain));
        let mut to_air = edge(
            "zone-0:gain",
            "zone-0:air",
            TopologyEdgeKind::InternalGainSplit,
            None,
        );
        to_air.fraction = Some(0.6);
        let mut to_mass = edge(
            "zone-0:gain",
            "zone-0:mass",
            TopologyEdgeKind::InternalGainSplit,
            None,
        );
        to_mass.fraction = Some(0.4);
        g.push_edge(to_air);
        g.push_edge(to_mass);
        let report = lint_topology(&g);
        assert!(report.is_clean(), "{:?}", report.findings);
    }

    #[test]
    fn e006_fires_when_zone_has_no_interior_receiver() {
        let mut g = clean_graph();
        // Remove the interior surface AND the mass so the zone has no receiver.
        g.nodes.retain(|n| {
            n.kind != TopologyNodeKind::InteriorSurface && n.kind != TopologyNodeKind::InternalMass
        });
        g.edges.clear();
        g.push_node(node("zone-0:gain", TopologyNodeKind::InternalGain));
        let mut rad = edge(
            "zone-0:gain",
            "zone-0:mass",
            TopologyEdgeKind::InternalGainSplit,
            None,
        );
        rad.fraction = Some(0.4);
        g.push_edge(rad);
        let report = lint_topology(&g);
        assert!(codes(&report).contains(&"E006_UNRESOLVED_RADIANT_TARGET".to_string()));
    }

    #[test]
    fn e007_fires_on_non_positive_capacity() {
        let mut g = clean_graph();
        g.nodes
            .iter_mut()
            .find(|n| n.kind == TopologyNodeKind::ZoneAir)
            .unwrap()
            .capacitance_j_per_k = Some(-5.0);
        let report = lint_topology(&g);
        assert_eq!(codes(&report), vec!["E007_NON_POSITIVE_CAPACITY"]);
    }

    #[test]
    fn w001_fires_on_out_of_band_film_coefficient() {
        let mut g = clean_graph();
        let idx = g
            .edges
            .iter()
            .position(|e| e.coupling_type == TopologyEdgeKind::ConvectionExterior)
            .unwrap();
        // 10 m² surface with a 5000 W/K film => h = 500 W/m²K, out of band.
        g.edges[idx].conductance_w_per_k = Some(5000.0);
        let report = lint_topology(&g);
        assert_eq!(codes(&report), vec!["W001_EXTREME_FILM_COEFFICIENT"]);
        assert!(report.has_blocking(true));
        assert!(!report.has_blocking(false));
    }

    #[test]
    fn w002_fires_on_extreme_conductance_ratio() {
        let mut g = clean_graph();
        let idx = g
            .edges
            .iter()
            .position(|e| e.coupling_type == TopologyEdgeKind::Conduction)
            .unwrap();
        g.edges[idx].conductance_w_per_k = Some(1.0e8);
        let report = lint_topology(&g);
        assert!(codes(&report).contains(&"W002_HIGH_CONDUCTANCE_RATIO".to_string()));
    }

    #[test]
    fn findings_are_sorted_deterministically() {
        let mut g = clean_graph();
        g.push_node(node("zone-0:ghost-a", TopologyNodeKind::InternalGain));
        g.push_node(node("zone-0:ghost-b", TopologyNodeKind::InternalGain));
        let report = lint_topology(&g);
        let ids: Vec<&str> = report
            .findings
            .iter()
            .filter_map(|f| f.node_id.as_deref())
            .collect();
        assert_eq!(ids, vec!["zone-0:ghost-a", "zone-0:ghost-b"]);
    }
}
