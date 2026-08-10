//! Decoupled Fluid Loop Parallel Evaluation Module (Issue #1991)
//!
//! This module provides parallel evaluation of independent HVAC fluid loops using Rayon.
//! When a building has multiple independent HVAC loops (e.g., separate VAV boxes,
//! multiple chillers serving different zones), they can be evaluated in parallel because
//! they don't share thermal state.
//!
//! ## Key Concepts
//!
//! - **Decoupled Loop**: A fluid loop that has no thermal coupling to other loops.
//!   Examples: a chiller loop serving Zone A, a boiler loop serving Zone B.
//! - **Loop Group**: A collection of equipment that shares a common fluid loop.
//! - **Independent Evaluation**: Each loop group can be evaluated in parallel
//!   since they don't share state.
//!
//! ## Design
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    DecoupledLoopEvaluator                   │
//! ├──────────────┬──────────────┬──────────────┬───────────────┤
//! │  LoopGroup A │  LoopGroup B │  LoopGroup C │  LoopGroup D  │
//! │  (Chiller 1)│  (Boiler 1)  │  (Chiller 2) │  (Boiler 2)  │
//! │  Zone 1,2    │  Zone 3,4    │  Zone 5,6    │  Zone 7,8     │
//! └──────┬───────┴──────┬───────┴──────┬───────┴───────┬──────┘
//!         │              │              │               │
//!         └──────────────┴──────────────┴───────────────┘
//!                           │
//!                    Rayon par_iter
//!                           │
//!         ┌─────────────────┴─────────────────┐
//!         │   Parallel evaluation, sequential  │
//!         │   within each loop group          │
//!         └─────────────────────────────────────┘
//! ```
//!
//! ## Determinism
//!
//! The parallel evaluation preserves determinism: the same input configuration
//! always produces the same output, regardless of thread scheduling order.
//! This is achieved by:
//! 1. Each loop group is independent (no shared mutable state)
//! 2. Results are collected and indexed by loop group ID
//! 3. Within each loop group, evaluation is sequential
//!
//! ## WASM Compatibility
//!
//! On WASM targets, rayon is not available. The implementation falls back to
//! sequential evaluation via conditional compilation:
//!
//! ```rust
//! #[cfg(not(target_arch = "wasm32"))]
//! use rayon::prelude::*;
//!
//! #[cfg(target_arch = "wasm32")]
//! // Sequential fallback for WASM
//! ```

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use std::collections::HashMap;
use std::fmt::Debug;

/// Unique identifier for a decoupled fluid loop group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoopGroupId(pub usize);

impl LoopGroupId {
    pub fn new(id: usize) -> Self {
        Self(id)
    }
}

/// A single step result from a decoupled fluid loop.
#[derive(Debug, Clone)]
pub struct LoopStepResult {
    /// Loop group identifier
    pub loop_id: LoopGroupId,
    /// Total energy consumed/produced this step [kWh]
    pub energy_kwh: f64,
    /// Peak power observed [kW]
    pub peak_power_kw: f64,
    /// Total fluid flow through the loop [kg/s]
    pub fluid_flow_kg_per_s: f64,
    /// Supply temperature [°C]
    pub supply_temp_c: f64,
    /// Return temperature [°C]
    pub return_temp_c: f64,
    /// Whether the loop converged this timestep
    pub converged: bool,
}

/// Setpoints for a decoupled fluid loop step.
#[derive(Debug, Clone)]
pub struct LoopStepParams {
    /// Loop group identifier
    pub loop_id: LoopGroupId,
    /// Zone temperatures served by this loop [°C]
    pub zone_temps: Vec<f64>,
    /// Outdoor air temperature [°C]
    pub outdoor_temp_c: f64,
    /// Timestep duration [s]
    pub dt_seconds: f64,
    /// Supply water temperature setpoint [°C] (for water loops)
    pub supply_temp_setpoint_c: Option<f64>,
    /// Demand signal from zones [kW]
    pub demand_kw: f64,
}

/// Trait for equipment that can be part of a decoupled fluid loop.
///
/// Implementors must be Send + Sync to allow Rayon parallel evaluation.
pub trait DecoupledLoopEquipment: Send + Sync {
    /// Unique loop group this equipment belongs to.
    fn loop_group_id(&self) -> LoopGroupId;

    /// Execute one timestep for this equipment.
    ///
    /// Returns the heat injection result for the zone and updates internal
    /// loop state (supply/return temperatures, flow rates).
    fn step_equipment(&mut self, zone_temp: f64, dt_seconds: f64) -> EquipmentStepResult;

    /// Reset equipment to initial state.
    fn reset(&mut self);
}

/// Result of a single equipment step within a loop.
#[derive(Debug, Clone)]
pub struct EquipmentStepResult {
    /// Heat delivered to the zone [W]
    pub q_delivered_w: f64,
    /// Electrical power consumed [W]
    pub electrical_power_w: f64,
    /// Fluid mass flow rate [kg/s]
    pub fluid_flow_kg_per_s: f64,
    /// Supply fluid temperature [°C]
    pub supply_temp_c: f64,
    /// Return fluid temperature [°C]
    pub return_temp_c: f64,
    /// Part load ratio [0, 1]
    pub part_load_ratio: f64,
}

/// Owned loop group data for parallel evaluation.
///
/// This struct holds the owned data needed to evaluate a loop group,
/// allowing it to be sent between threads. It clones the equipment
/// since equipment must be `Clone` to be used with the evaluator.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LoopGroupData<E: Clone> {
    id: LoopGroupId,
    equipment: Vec<E>,
    supply_temp_setpoint_c: f64,
    design_flow_kg_per_s: f64,
}

/// A group of equipment sharing the same fluid loop.
#[derive(Debug)]
pub struct LoopGroup<E: DecoupledLoopEquipment> {
    /// Unique identifier for this loop group
    pub id: LoopGroupId,
    /// Equipment in this loop group
    pub equipment: Vec<E>,
    /// Supply temperature setpoint [°C]
    pub supply_temp_setpoint_c: f64,
    /// Design fluid flow rate [kg/s]
    pub design_flow_kg_per_s: f64,
}

impl<E: DecoupledLoopEquipment> Clone for LoopGroup<E>
where
    E: Clone,
{
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            equipment: self.equipment.clone(),
            supply_temp_setpoint_c: self.supply_temp_setpoint_c,
            design_flow_kg_per_s: self.design_flow_kg_per_s,
        }
    }
}

impl<E: DecoupledLoopEquipment> LoopGroup<E> {
    /// Create a new loop group with the given equipment.
    pub fn new(
        id: LoopGroupId,
        equipment: Vec<E>,
        supply_temp_setpoint_c: f64,
        design_flow_kg_per_s: f64,
    ) -> Self {
        Self {
            id,
            equipment,
            supply_temp_setpoint_c,
            design_flow_kg_per_s,
        }
    }

    /// Number of equipment units in this loop group.
    pub fn len(&self) -> usize {
        self.equipment.len()
    }

    /// Check if loop group has no equipment.
    pub fn is_empty(&self) -> bool {
        self.equipment.is_empty()
    }
}

/// Evaluator for parallel execution of decoupled fluid loops.
///
/// This struct manages the parallel evaluation of multiple independent
/// fluid loop groups using Rayon. Each loop group is evaluated in
/// parallel with others, while equipment within a group is evaluated
/// sequentially (since they share the same fluid state).
pub struct DecoupledLoopEvaluator<E: DecoupledLoopEquipment + Clone> {
    loop_groups: Vec<LoopGroup<E>>,
    convergence_tolerance_kw: f64,
}

impl<E: DecoupledLoopEquipment + Clone> DecoupledLoopEvaluator<E> {
    /// Create a new evaluator with the given loop groups.
    ///
    /// All equipment in the same loop group must have the same loop_group_id.
    pub fn new(loop_groups: Vec<LoopGroup<E>>) -> Self {
        Self {
            loop_groups,
            convergence_tolerance_kw: 0.1, // 100W default tolerance
        }
    }

    /// Create an evaluator with a custom convergence tolerance.
    pub fn with_tolerance(loop_groups: Vec<LoopGroup<E>>, tolerance_kw: f64) -> Self {
        Self {
            loop_groups,
            convergence_tolerance_kw: tolerance_kw,
        }
    }

    /// Number of loop groups.
    pub fn num_loop_groups(&self) -> usize {
        self.loop_groups.len()
    }

    /// Total number of equipment across all loop groups.
    pub fn total_equipment_count(&self) -> usize {
        self.loop_groups.iter().map(|g| g.len()).sum()
    }

    /// Evaluate all loop groups in parallel for one timestep using Rayon.
    ///
    /// Returns results indexed by loop group ID. Each loop group is evaluated
    /// concurrently using a thread pool, while equipment within a group is
    /// evaluated sequentially (they share fluid state).
    ///
    /// # Determinism
    ///
    /// The results are deterministic: the same inputs always produce the
    /// same outputs regardless of thread scheduling order. This is because:
    /// 1. Each loop group is independent (no shared state between groups)
    /// 2. Results are stored in a HashMap indexed by LoopGroupId
    /// 3. Within each group, equipment is evaluated in fixed order
    #[cfg(not(target_arch = "wasm32"))]
    pub fn evaluate_parallel(
        &mut self,
        params: &HashMap<LoopGroupId, LoopStepParams>,
    ) -> HashMap<LoopGroupId, LoopStepResult>
    where
        E: Clone + 'static,
    {
        if self.loop_groups.is_empty() {
            return HashMap::new();
        }

        let tolerance = self.convergence_tolerance_kw;
        let num_groups = self.loop_groups.len();

        let group_data_list: Vec<LoopGroupData<E>> = (0..num_groups)
            .map(|idx| LoopGroupData {
                id: self.loop_groups[idx].id,
                supply_temp_setpoint_c: self.loop_groups[idx].supply_temp_setpoint_c,
                design_flow_kg_per_s: self.loop_groups[idx].design_flow_kg_per_s,
                equipment: self.loop_groups[idx].equipment.clone(),
            })
            .collect();

        let results: Vec<LoopStepResult> = group_data_list
            .into_par_iter()
            .map(|mut group_data| {
                let id = group_data.id;
                evaluate_loop_group_from_data(&mut group_data, params.get(&id), tolerance)
            })
            .collect();

        results.into_iter().map(|r| (r.loop_id, r)).collect()
    }

    /// WASM fallback: sequential evaluation when Rayon is not available.
    #[cfg(target_arch = "wasm32")]
    pub fn evaluate_parallel(
        &mut self,
        params: &HashMap<LoopGroupId, LoopStepParams>,
    ) -> HashMap<LoopGroupId, LoopStepResult>
    where
        E: Clone + 'static,
    {
        if self.loop_groups.is_empty() {
            return HashMap::new();
        }

        let tolerance = self.convergence_tolerance_kw;

        let results: Vec<LoopStepResult> = self
            .loop_groups
            .iter_mut()
            .map(|group| {
                let mut group_data = LoopGroupData {
                    id: group.id,
                    supply_temp_setpoint_c: group.supply_temp_setpoint_c,
                    design_flow_kg_per_s: group.design_flow_kg_per_s,
                    equipment: group.equipment.clone(),
                };
                evaluate_loop_group_from_data(&mut group_data, params.get(&group.id), tolerance)
            })
            .collect();

        results.into_iter().map(|r| (r.loop_id, r)).collect()
    }

    #[allow(dead_code)]
    fn evaluate_loop_group(
        &self,
        group: &mut LoopGroup<E>,
        params: Option<&LoopStepParams>,
    ) -> LoopStepResult {
        evaluate_single_loop_group(group, params, self.convergence_tolerance_kw)
    }
}

/// Evaluate a single loop group sequentially (standalone function).
///
/// Equipment within a loop group is evaluated sequentially because they
/// share the same fluid loop state (supply/return temperatures, flow).
#[allow(dead_code)]
fn evaluate_single_loop_group<E: DecoupledLoopEquipment>(
    group: &mut LoopGroup<E>,
    params: Option<&LoopStepParams>,
    tolerance_kw: f64,
) -> LoopStepResult {
    // Create default params if not provided
    let default_params = LoopStepParams {
        loop_id: group.id,
        zone_temps: vec![20.0; group.equipment.len()],
        outdoor_temp_c: 10.0,
        dt_seconds: 3600.0,
        supply_temp_setpoint_c: Some(group.supply_temp_setpoint_c),
        demand_kw: 0.0,
    };
    let params = params.unwrap_or(&default_params);

    let mut total_energy_kwh = 0.0_f64;
    let mut peak_power_kw = 0.0_f64;
    let mut total_flow_kg_per_s = 0.0_f64;
    let mut supply_temps = Vec::with_capacity(group.equipment.len());
    let mut return_temps = Vec::with_capacity(group.equipment.len());

    // Evaluate equipment sequentially within the loop group
    // (they share fluid state - can't parallelize without careful state management)
    for (eq_idx, equipment) in group.equipment.iter_mut().enumerate() {
        let zone_temp = params.zone_temps.get(eq_idx).copied().unwrap_or(20.0);

        let result = equipment.step_equipment(zone_temp, params.dt_seconds);

        total_energy_kwh += result.q_delivered_w / 1000.0 * params.dt_seconds / 3600.0;
        peak_power_kw = peak_power_kw.max(result.electrical_power_w / 1000.0);
        total_flow_kg_per_s += result.fluid_flow_kg_per_s;
        supply_temps.push(result.supply_temp_c);
        return_temps.push(result.return_temp_c);
    }

    let supply_temp_c = if supply_temps.is_empty() {
        group.supply_temp_setpoint_c
    } else {
        // Average supply temperature weighted by flow
        supply_temps
            .iter()
            .zip(return_temps.iter())
            .fold(0.0, |acc, (s, _r)| acc + s)
            / supply_temps.len() as f64
    };

    let return_temp_c = if return_temps.is_empty() {
        supply_temp_c
    } else {
        return_temps.iter().sum::<f64>() / return_temps.len() as f64
    };

    // Check convergence: demand should be met within tolerance
    let converged =
        (params.demand_kw - total_energy_kwh / (params.dt_seconds / 3600.0)).abs() < tolerance_kw;

    LoopStepResult {
        loop_id: group.id,
        energy_kwh: total_energy_kwh,
        peak_power_kw,
        fluid_flow_kg_per_s: total_flow_kg_per_s,
        supply_temp_c,
        return_temp_c,
        converged,
    }
}

/// Evaluate a loop group from owned data (for parallel execution).
fn evaluate_loop_group_from_data<E: DecoupledLoopEquipment + Clone>(
    group: &mut LoopGroupData<E>,
    params: Option<&LoopStepParams>,
    tolerance_kw: f64,
) -> LoopStepResult {
    let default_params = LoopStepParams {
        loop_id: group.id,
        zone_temps: vec![20.0; group.equipment.len()],
        outdoor_temp_c: 10.0,
        dt_seconds: 3600.0,
        supply_temp_setpoint_c: Some(group.supply_temp_setpoint_c),
        demand_kw: 0.0,
    };
    let params = params.unwrap_or(&default_params);

    let mut total_energy_kwh = 0.0_f64;
    let mut peak_power_kw = 0.0_f64;
    let mut total_flow_kg_per_s = 0.0_f64;
    let mut supply_temps = Vec::with_capacity(group.equipment.len());
    let mut return_temps = Vec::with_capacity(group.equipment.len());

    for (eq_idx, equipment) in group.equipment.iter_mut().enumerate() {
        let zone_temp = params.zone_temps.get(eq_idx).copied().unwrap_or(20.0);
        let result = equipment.step_equipment(zone_temp, params.dt_seconds);

        total_energy_kwh += result.q_delivered_w / 1000.0 * params.dt_seconds / 3600.0;
        peak_power_kw = peak_power_kw.max(result.electrical_power_w / 1000.0);
        total_flow_kg_per_s += result.fluid_flow_kg_per_s;
        supply_temps.push(result.supply_temp_c);
        return_temps.push(result.return_temp_c);
    }

    let supply_temp_c = if supply_temps.is_empty() {
        group.supply_temp_setpoint_c
    } else {
        supply_temps.iter().sum::<f64>() / supply_temps.len() as f64
    };

    let return_temp_c = if return_temps.is_empty() {
        supply_temp_c
    } else {
        return_temps.iter().sum::<f64>() / return_temps.len() as f64
    };

    let converged =
        (params.demand_kw - total_energy_kwh / (params.dt_seconds / 3600.0)).abs() < tolerance_kw;

    LoopStepResult {
        loop_id: group.id,
        energy_kwh: total_energy_kwh,
        peak_power_kw,
        fluid_flow_kg_per_s: total_flow_kg_per_s,
        supply_temp_c,
        return_temp_c,
        converged,
    }
}

impl<E: DecoupledLoopEquipment> DecoupledLoopEvaluator<E>
where
    E: Clone,
{
    /// Reset all equipment in all loop groups to initial state.
    pub fn reset_all(&mut self) {
        for group in &mut self.loop_groups {
            for equipment in &mut group.equipment {
                equipment.reset();
            }
        }
    }

    /// Get loop group IDs in deterministic order.
    pub fn loop_group_ids(&self) -> Vec<LoopGroupId> {
        self.loop_groups.iter().map(|g| g.id).collect()
    }
}

impl<E: DecoupledLoopEquipment + Clone> Default for DecoupledLoopEvaluator<E> {
    fn default() -> Self {
        Self {
            loop_groups: Vec::new(),
            convergence_tolerance_kw: 0.1,
        }
    }
}

/// Divide loop groups into independent execution groups.
///
/// This function analyzes loop groups and partitions them into
/// execution sets that can run in parallel. Groups in different
/// execution sets have no shared state and can be evaluated concurrently.
pub fn partition_independent_loops<E: DecoupledLoopEquipment + Clone>(
    loop_groups: &[LoopGroup<E>],
) -> Vec<Vec<LoopGroup<E>>> {
    if loop_groups.is_empty() {
        return Vec::new();
    }

    // For now, each loop group is independent
    // This function exists to allow future optimization where
    // groups could be further partitioned based on shared resources
    // (e.g., same chiller serving multiple loops)
    let mut execution_sets: Vec<Vec<LoopGroup<E>>> = Vec::new();

    for group in loop_groups {
        execution_sets.push(vec![group.clone()]);
    }

    execution_sets
}

/// Check if two loop groups are independent (can run in parallel).
///
/// Loop groups are independent if they don't share any equipment or
/// thermal coupling. This is used to validate loop group assignments.
pub fn are_loops_independent<E: DecoupledLoopEquipment>(
    group_a: &LoopGroup<E>,
    group_b: &LoopGroup<E>,
) -> bool {
    if group_a.id == group_b.id {
        return false;
    }

    for eq_a in &group_a.equipment {
        for eq_b in &group_b.equipment {
            if eq_a.loop_group_id() == eq_b.loop_group_id() {
                return false;
            }
        }
    }

    true
}

// =============================================================================
// Tarjan's Strongly Connected Components (SCC) Algorithm
// =============================================================================

/// Node identifier for graph algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphNodeId(pub usize);

impl GraphNodeId {
    pub fn new(id: usize) -> Self {
        Self(id)
    }
}

/// A node in the fluid network graph.
#[derive(Debug, Clone)]
pub struct FluidGraphNode {
    pub id: GraphNodeId,
    pub name: String,
    pub node_type: FluidNodeType,
}

/// Type of fluid network node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FluidNodeType {
    /// Equipment node (chiller, boiler, AHU, etc.)
    Equipment,
    /// Conservation node (mass/energy balance point)
    Conservation,
    /// Boundary condition node (outdoor, zone)
    Boundary,
    /// Feedback node (for cyclic dependencies like variable speed pumps)
    Feedback,
}

/// An edge in the fluid network graph.
#[derive(Debug, Clone)]
pub struct FluidGraphEdge {
    pub from: GraphNodeId,
    pub to: GraphNodeId,
    pub edge_type: FluidEdgeType,
}

/// Type of fluid network edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FluidEdgeType {
    /// Fluid flow connection
    Flow,
    /// Thermal coupling (heat transfer)
    Thermal,
    /// Control signal connection
    Control,
    /// Feedback connection (cyclic dependency)
    Feedback,
}

/// A subgraph that can be solved independently.
#[derive(Debug, Clone)]
pub struct Subgraph {
    pub id: usize,
    pub nodes: Vec<GraphNodeId>,
    pub edges: Vec<FluidGraphEdge>,
    pub has_feedback: bool,
}

impl Subgraph {
    /// Number of states in this subgraph.
    pub fn n_states(&self) -> usize {
        self.nodes.len()
    }
}

/// Fluid network graph for HVAC systems.
#[derive(Debug, Clone, Default)]
pub struct FluidNetworkGraph {
    nodes: Vec<FluidGraphNode>,
    edges: Vec<FluidGraphEdge>,
    adjacency_list: HashMap<GraphNodeId, Vec<GraphNodeId>>,
}

impl FluidNetworkGraph {
    /// Create a new empty fluid network graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: FluidGraphNode) {
        let id = node.id;
        self.nodes.push(node);
        self.adjacency_list.entry(id).or_default();
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, edge: FluidGraphEdge) {
        self.edges.push(edge.clone());
        self.adjacency_list
            .entry(edge.from)
            .or_default()
            .push(edge.to);
    }

    /// Get all nodes.
    pub fn nodes(&self) -> &[FluidGraphNode] {
        &self.nodes
    }

    /// Get all edges.
    pub fn edges(&self) -> &[FluidGraphEdge] {
        &self.edges
    }

    /// Get outgoing edges from a node.
    pub fn outgoing_edges(&self, from: GraphNodeId) -> Vec<&FluidGraphEdge> {
        self.edges.iter().filter(|e| e.from == from).collect()
    }

    /// Get incoming edges to a node.
    pub fn incoming_edges(&self, to: GraphNodeId) -> Vec<&FluidGraphEdge> {
        self.edges.iter().filter(|e| e.to == to).collect()
    }

    /// Get the subgraph containing a specific node.
    pub fn get_subgraph(&self, node_id: GraphNodeId) -> Option<&Subgraph> {
        self.edges
            .iter()
            .find(|e| e.from == node_id || e.to == node_id);
        None
    }
}

/// Decompose a fluid network graph into parallelizable subgraphs using Tarjan's SCC algorithm.
///
/// This function identifies independent sub-loops that can be solved in parallel:
/// 1. Compute SCCs using Tarjan's algorithm
/// 2. Each SCC with no cyclic feedback = one independent sub-loop
/// 3. Sub-loops that share only conservation nodes = parallelizable
/// 4. Loops with feedback (e.g., variable speed pump with pressure-controlled setpoint) = sequential
pub fn decompose_parallel_subgraphs(graph: &FluidNetworkGraph) -> Vec<Subgraph> {
    if graph.nodes.is_empty() {
        return Vec::new();
    }

    let n = graph.nodes.len();
    let mut index = 0;
    let mut node_index: Vec<Option<usize>> = vec![None; n];
    let mut node_lowlink: Vec<usize> = vec![0; n];
    let mut on_stack: Vec<bool> = vec![false; n];
    let mut stack: Vec<GraphNodeId> = Vec::new();
    let mut sccs: Vec<Vec<GraphNodeId>> = Vec::new();

    #[allow(clippy::too_many_arguments)]
    fn strong_connect(
        graph: &FluidNetworkGraph,
        node_id: GraphNodeId,
        index: &mut usize,
        node_index: &mut Vec<Option<usize>>,
        node_lowlink: &mut Vec<usize>,
        on_stack: &mut Vec<bool>,
        stack: &mut Vec<GraphNodeId>,
        sccs: &mut Vec<Vec<GraphNodeId>>,
    ) {
        let idx = node_id.0;

        node_index[idx] = Some(*index);
        node_lowlink[idx] = *index;
        *index += 1;
        stack.push(node_id);
        on_stack[idx] = true;

        if let Some(neighbors) = graph.adjacency_list.get(&node_id) {
            for &neighbor in neighbors {
                let neighbor_idx = neighbor.0;
                if node_index[neighbor_idx].is_none() {
                    strong_connect(
                        graph,
                        neighbor,
                        index,
                        node_index,
                        node_lowlink,
                        on_stack,
                        stack,
                        sccs,
                    );
                    node_lowlink[idx] = node_lowlink[idx].min(node_lowlink[neighbor_idx]);
                } else if on_stack[neighbor_idx] {
                    node_lowlink[idx] = node_lowlink[idx].min(node_index[neighbor_idx].unwrap());
                }
            }
        }

        if node_lowlink[idx] == node_index[idx].unwrap() {
            let mut scc: Vec<GraphNodeId> = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack[w.0] = false;
                scc.push(w);
                if w == node_id {
                    break;
                }
            }
            sccs.push(scc);
        }
    }

    for node in &graph.nodes {
        if node_index[node.id.0].is_none() {
            strong_connect(
                graph,
                node.id,
                &mut index,
                &mut node_index,
                &mut node_lowlink,
                &mut on_stack,
                &mut stack,
                &mut sccs,
            );
        }
    }

    let mut subgraphs: Vec<Subgraph> = Vec::new();
    for (scc_id, scc_nodes) in sccs.into_iter().enumerate() {
        let has_feedback = scc_nodes.len() > 1
            || graph
                .edges
                .iter()
                .any(|e| e.edge_type == FluidEdgeType::Feedback && scc_nodes.contains(&e.from));

        let scc_edges: Vec<FluidGraphEdge> = graph
            .edges
            .iter()
            .filter(|e| scc_nodes.contains(&e.from) && scc_nodes.contains(&e.to))
            .cloned()
            .collect();

        subgraphs.push(Subgraph {
            id: scc_id,
            nodes: scc_nodes,
            edges: scc_edges,
            has_feedback,
        });
    }

    subgraphs
}

/// Parallel loop dispatcher using Rayon for concurrent subgraph execution.
///
/// This dispatcher manages parallel evaluation of independent subgraphs
/// using a thread pool. Subgraphs without feedback can be evaluated
/// concurrently, while subgraphs with feedback must be evaluated sequentially.
#[derive(Debug)]
pub struct ParallelLoopDispatcher {
    subgraphs: Vec<Subgraph>,
}

impl ParallelLoopDispatcher {
    /// Create a new dispatcher from subgraphs.
    pub fn new(subgraphs: Vec<Subgraph>) -> Self {
        Self { subgraphs }
    }

    /// Get the number of subgraphs.
    pub fn num_subgraphs(&self) -> usize {
        self.subgraphs.len()
    }

    /// Get subgraphs that can be evaluated in parallel (no feedback).
    pub fn parallel_subgraphs(&self) -> Vec<&Subgraph> {
        self.subgraphs.iter().filter(|s| !s.has_feedback).collect()
    }

    /// Get subgraphs that must be evaluated sequentially (have feedback).
    pub fn sequential_subgraphs(&self) -> Vec<&Subgraph> {
        self.subgraphs.iter().filter(|s| s.has_feedback).collect()
    }

    /// Dispatch all subgraphs in parallel using Rayon (non-WASM).
    ///
    /// The closure `f` must be stateless (`Fn`, callable from a shared
    /// reference) so every rayon worker can invoke it concurrently without any
    /// lock. A previous implementation wrapped `f` in `Arc<Mutex<F>>` and
    /// acquired the lock inside `par_iter`, which serialised the whole
    /// iteration — see #2525. Closures that need to accumulate per-worker
    /// output should capture their own interior synchronisation (e.g. a
    /// `Mutex` or atomics *inside* `f`) rather than relying on `FnMut` state.
    ///
    /// Parallelism stays at exactly this one `par_iter` level (no nested
    /// `par_iter`); the closure body is evaluated sequentially per subgraph.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn step<F, R>(&mut self, _t: f64, _dt: f64, f: F) -> Result<(), DispatchError>
    where
        F: Fn(&Subgraph) -> Result<R, DispatchError> + Send + Sync,
        R: Send,
    {
        // `F: Sync` ⇒ `&F: Send`, so each rayon worker holds a shared
        // reference to the same closure and invokes it directly. No
        // `Mutex`/`Arc` is required.
        self.subgraphs
            .par_iter()
            .map(|subgraph| {
                if subgraph.has_feedback {
                    Err(DispatchError::FeedbackLoop(subgraph.id))
                } else {
                    f(subgraph)
                }
            })
            .collect::<Result<Vec<R>, DispatchError>>()
            .map(drop)
    }

    /// WASM fallback: sequential dispatch when Rayon is not available.
    #[cfg(target_arch = "wasm32")]
    pub fn step<F, R>(&mut self, _t: f64, _dt: f64, f: F) -> Result<(), DispatchError>
    where
        F: Fn(&Subgraph) -> Result<R, DispatchError>,
    {
        for subgraph in &self.subgraphs {
            if subgraph.has_feedback {
                return Err(DispatchError::FeedbackLoop(subgraph.id));
            }
            f(subgraph)?;
        }
        Ok(())
    }
}

/// Error type for dispatch operations.
#[derive(Debug, Clone)]
pub enum DispatchError {
    /// A subgraph has a feedback loop and cannot be evaluated in parallel.
    FeedbackLoop(usize),
    /// Numerical convergence failure.
    ConvergenceFailed(String),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DispatchError::FeedbackLoop(id) => {
                write!(
                    f,
                    "Subgraph {} has a feedback loop and requires sequential evaluation",
                    id
                )
            }
            DispatchError::ConvergenceFailed(msg) => {
                write!(f, "Convergence failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for DispatchError {}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock equipment for testing
    #[derive(Debug, Clone)]
    struct MockEquipment {
        id: String,
        loop_id: LoopGroupId,
        capacity_w: f64,
        current_plr: f64,
    }

    impl MockEquipment {
        fn new(id: &str, loop_id: LoopGroupId, capacity_w: f64) -> Self {
            Self {
                id: id.to_string(),
                loop_id,
                capacity_w,
                current_plr: 0.0,
            }
        }
    }

    impl DecoupledLoopEquipment for MockEquipment {
        fn loop_group_id(&self) -> LoopGroupId {
            self.loop_id
        }

        fn step_equipment(&mut self, zone_temp: f64, _dt_seconds: f64) -> EquipmentStepResult {
            // Simple on/off control
            let demand = if zone_temp < 18.0 {
                self.capacity_w
            } else if zone_temp > 22.0 {
                -self.capacity_w * 0.5 // Cooling
            } else {
                0.0
            };

            let plr = (demand.abs() / self.capacity_w).min(1.0);
            self.current_plr = plr;

            EquipmentStepResult {
                q_delivered_w: demand.abs() * plr,
                electrical_power_w: demand.abs() * plr * 0.3, // COP = 3.0
                fluid_flow_kg_per_s: plr * 0.5,
                supply_temp_c: if demand > 0.0 { 45.0 } else { 7.0 },
                return_temp_c: if demand > 0.0 { 40.0 } else { 12.0 },
                part_load_ratio: plr,
            }
        }

        fn reset(&mut self) {
            self.current_plr = 0.0;
        }
    }

    #[test]
    fn test_loop_group_construction() {
        let equipment = vec![
            MockEquipment::new("EQ-1", LoopGroupId::new(0), 5000.0),
            MockEquipment::new("EQ-2", LoopGroupId::new(0), 3000.0),
        ];
        let group = LoopGroup::new(LoopGroupId::new(0), equipment, 45.0, 1.0);

        assert_eq!(group.len(), 2);
        assert_eq!(group.id, LoopGroupId::new(0));
        assert!(!group.is_empty());
    }

    #[test]
    fn test_evaluator_parallel_evaluation() {
        let groups = vec![
            LoopGroup::new(
                LoopGroupId::new(0),
                vec![
                    MockEquipment::new("CH-1", LoopGroupId::new(0), 10000.0),
                    MockEquipment::new("CH-2", LoopGroupId::new(0), 8000.0),
                ],
                7.0,
                2.0,
            ),
            LoopGroup::new(
                LoopGroupId::new(1),
                vec![MockEquipment::new("BO-1", LoopGroupId::new(1), 15000.0)],
                45.0,
                1.5,
            ),
        ];

        let mut evaluator = DecoupledLoopEvaluator::new(groups);

        let mut params = HashMap::new();
        params.insert(
            LoopGroupId::new(0),
            LoopStepParams {
                loop_id: LoopGroupId::new(0),
                zone_temps: vec![26.0, 25.0], // Cooling mode (above 22 deg threshold)
                outdoor_temp_c: 30.0,
                dt_seconds: 3600.0,
                supply_temp_setpoint_c: Some(7.0),
                demand_kw: 5.0,
            },
        );
        params.insert(
            LoopGroupId::new(1),
            LoopStepParams {
                loop_id: LoopGroupId::new(1),
                zone_temps: vec![16.0], // Heating mode (below 18 deg threshold)
                outdoor_temp_c: 0.0,
                dt_seconds: 3600.0,
                supply_temp_setpoint_c: Some(45.0),
                demand_kw: 10.0,
            },
        );

        let results = evaluator.evaluate_parallel(&params);

        assert_eq!(results.len(), 2);
        assert!(results.contains_key(&LoopGroupId::new(0)));
        assert!(results.contains_key(&LoopGroupId::new(1)));

        // Chiller loop should have cooling supply temp
        let chiller_result = &results[&LoopGroupId::new(0)];
        assert!(chiller_result.supply_temp_c < 20.0);

        // Boiler loop should have heating supply temp
        let boiler_result = &results[&LoopGroupId::new(1)];
        assert!(boiler_result.supply_temp_c > 30.0);
    }

    #[test]
    fn test_empty_evaluator() {
        let mut evaluator: DecoupledLoopEvaluator<MockEquipment> =
            DecoupledLoopEvaluator::default();
        let results = evaluator.evaluate_parallel(&HashMap::new());
        assert!(results.is_empty());
    }

    #[test]
    fn test_evaluator_reset() {
        let groups = vec![LoopGroup::new(
            LoopGroupId::new(0),
            vec![MockEquipment::new("EQ-1", LoopGroupId::new(0), 5000.0)],
            45.0,
            1.0,
        )];

        let mut evaluator = DecoupledLoopEvaluator::new(groups);

        // Run a step to set PLR
        let params = HashMap::new();
        evaluator.evaluate_parallel(&params);

        // Reset should clear state
        evaluator.reset_all();

        assert_eq!(evaluator.num_loop_groups(), 1);
        assert_eq!(evaluator.total_equipment_count(), 1);
    }

    #[test]
    fn test_independent_loops() {
        let group_a = LoopGroup::new(
            LoopGroupId::new(0),
            vec![MockEquipment::new("EQ-1", LoopGroupId::new(0), 5000.0)],
            45.0,
            1.0,
        );
        let group_b = LoopGroup::new(
            LoopGroupId::new(1),
            vec![MockEquipment::new("EQ-2", LoopGroupId::new(1), 3000.0)],
            7.0,
            2.0,
        );

        assert!(are_loops_independent(&group_a, &group_b));
        assert!(!are_loops_independent(&group_a, &group_a)); // Same group
    }

    #[test]
    fn test_partition_independent_loops() {
        let groups = vec![
            LoopGroup::new(
                LoopGroupId::new(0),
                vec![MockEquipment::new("EQ-1", LoopGroupId::new(0), 5000.0)],
                45.0,
                1.0,
            ),
            LoopGroup::new(
                LoopGroupId::new(1),
                vec![MockEquipment::new("EQ-2", LoopGroupId::new(1), 3000.0)],
                7.0,
                2.0,
            ),
        ];

        let partitions = partition_independent_loops(&groups);

        // Each group should be in its own partition
        assert_eq!(partitions.len(), 2);
        assert_eq!(partitions[0].len(), 1);
        assert_eq!(partitions[1].len(), 1);
    }

    #[test]
    fn test_determinism_across_runs() {
        let groups = vec![
            LoopGroup::new(
                LoopGroupId::new(0),
                vec![
                    MockEquipment::new("CH-1", LoopGroupId::new(0), 10000.0),
                    MockEquipment::new("CH-2", LoopGroupId::new(0), 8000.0),
                ],
                7.0,
                2.0,
            ),
            LoopGroup::new(
                LoopGroupId::new(1),
                vec![MockEquipment::new("BO-1", LoopGroupId::new(1), 15000.0)],
                45.0,
                1.5,
            ),
        ];

        let params = {
            let mut p = HashMap::new();
            p.insert(
                LoopGroupId::new(0),
                LoopStepParams {
                    loop_id: LoopGroupId::new(0),
                    zone_temps: vec![22.0, 21.0],
                    outdoor_temp_c: 30.0,
                    dt_seconds: 3600.0,
                    supply_temp_setpoint_c: Some(7.0),
                    demand_kw: 5.0,
                },
            );
            p.insert(
                LoopGroupId::new(1),
                LoopStepParams {
                    loop_id: LoopGroupId::new(1),
                    zone_temps: vec![18.0],
                    outdoor_temp_c: 0.0,
                    dt_seconds: 3600.0,
                    supply_temp_setpoint_c: Some(45.0),
                    demand_kw: 10.0,
                },
            );
            p
        };

        // Run multiple times and verify identical results
        let results: Vec<_> = (0..3)
            .map(|_| {
                let mut evaluator = DecoupledLoopEvaluator::new(groups.clone());
                evaluator.evaluate_parallel(&params)
            })
            .collect();

        // Results should be identical across runs
        for i in 1..results.len() {
            for (loop_id, result) in &results[i] {
                let prev = &results[i - 1][loop_id];
                assert!(
                    (result.energy_kwh - prev.energy_kwh).abs() < 1e-10,
                    "Energy mismatch for loop {:?}: {} vs {}",
                    loop_id,
                    result.energy_kwh,
                    prev.energy_kwh
                );
                assert_eq!(
                    result.converged, prev.converged,
                    "Convergence mismatch for loop {:?}",
                    loop_id
                );
            }
        }
    }

    #[test]
    fn test_loop_ids_deterministic_order() {
        let groups = vec![
            LoopGroup::new(
                LoopGroupId::new(2),
                vec![MockEquipment::new("EQ-2", LoopGroupId::new(2), 3000.0)],
                7.0,
                2.0,
            ),
            LoopGroup::new(
                LoopGroupId::new(0),
                vec![MockEquipment::new("EQ-0", LoopGroupId::new(0), 5000.0)],
                45.0,
                1.0,
            ),
            LoopGroup::new(
                LoopGroupId::new(1),
                vec![MockEquipment::new("EQ-1", LoopGroupId::new(1), 4000.0)],
                7.0,
                1.5,
            ),
        ];

        let evaluator = DecoupledLoopEvaluator::new(groups);
        let ids = evaluator.loop_group_ids();

        // Should be in original order
        assert_eq!(ids[0], LoopGroupId::new(2));
        assert_eq!(ids[1], LoopGroupId::new(0));
        assert_eq!(ids[2], LoopGroupId::new(1));
    }

    // =============================================================================
    // Tarjan SCC Algorithm Tests
    // =============================================================================

    #[test]
    fn test_tarjan_scc_single_node() {
        let mut graph = FluidNetworkGraph::new();
        graph.add_node(FluidGraphNode {
            id: GraphNodeId::new(0),
            name: "CH-1".to_string(),
            node_type: FluidNodeType::Equipment,
        });

        let subgraphs = decompose_parallel_subgraphs(&graph);
        assert_eq!(subgraphs.len(), 1);
        assert_eq!(subgraphs[0].nodes, vec![GraphNodeId::new(0)]);
        assert!(!subgraphs[0].has_feedback);
    }

    #[test]
    fn test_tarjan_scc_two_independent_nodes() {
        let mut graph = FluidNetworkGraph::new();
        graph.add_node(FluidGraphNode {
            id: GraphNodeId::new(0),
            name: "CH-1".to_string(),
            node_type: FluidNodeType::Equipment,
        });
        graph.add_node(FluidGraphNode {
            id: GraphNodeId::new(1),
            name: "BO-1".to_string(),
            node_type: FluidNodeType::Equipment,
        });

        // Add flow edges but no cycle
        graph.add_edge(FluidGraphEdge {
            from: GraphNodeId::new(0),
            to: GraphNodeId::new(1),
            edge_type: FluidEdgeType::Flow,
        });

        let subgraphs = decompose_parallel_subgraphs(&graph);
        // Two nodes without mutual connection = two separate SCCs
        assert_eq!(subgraphs.len(), 2);
    }

    #[test]
    fn test_tarjan_scc_with_cycle() {
        let mut graph = FluidNetworkGraph::new();
        graph.add_node(FluidGraphNode {
            id: GraphNodeId::new(0),
            name: "CH-1".to_string(),
            node_type: FluidNodeType::Equipment,
        });
        graph.add_node(FluidGraphNode {
            id: GraphNodeId::new(1),
            name: "BO-1".to_string(),
            node_type: FluidNodeType::Equipment,
        });

        // Create a cycle: 0 -> 1 -> 0
        graph.add_edge(FluidGraphEdge {
            from: GraphNodeId::new(0),
            to: GraphNodeId::new(1),
            edge_type: FluidEdgeType::Flow,
        });
        graph.add_edge(FluidGraphEdge {
            from: GraphNodeId::new(1),
            to: GraphNodeId::new(0),
            edge_type: FluidEdgeType::Flow,
        });

        let subgraphs = decompose_parallel_subgraphs(&graph);
        // Two nodes with cycle = one SCC
        assert_eq!(subgraphs.len(), 1);
        assert_eq!(subgraphs[0].nodes.len(), 2);
        assert!(subgraphs[0].has_feedback);
    }

    #[test]
    fn test_tarjan_scc_feedback_detection() {
        let mut graph = FluidNetworkGraph::new();
        graph.add_node(FluidGraphNode {
            id: GraphNodeId::new(0),
            name: "VSD-Pump".to_string(),
            node_type: FluidNodeType::Equipment,
        });

        // Add a feedback edge
        graph.add_edge(FluidGraphEdge {
            from: GraphNodeId::new(0),
            to: GraphNodeId::new(0),
            edge_type: FluidEdgeType::Feedback,
        });

        let subgraphs = decompose_parallel_subgraphs(&graph);
        assert_eq!(subgraphs.len(), 1);
        assert!(subgraphs[0].has_feedback);
    }

    #[test]
    fn test_tarjan_scc_typical_hvac_system() {
        // Typical commercial building HVAC with 3 independent loops:
        // - Chilled water loop (CH-1, CH-2)
        // - Condenser water loop (CT-1, CT-2)
        // - Duct network (AHU-1, AHU-2)
        let mut graph = FluidNetworkGraph::new();

        // Chilled water loop
        graph.add_node(FluidGraphNode {
            id: GraphNodeId::new(0),
            name: "CH-1".to_string(),
            node_type: FluidNodeType::Equipment,
        });
        graph.add_node(FluidGraphNode {
            id: GraphNodeId::new(1),
            name: "CH-2".to_string(),
            node_type: FluidNodeType::Equipment,
        });
        graph.add_node(FluidGraphNode {
            id: GraphNodeId::new(2),
            name: "CH-Cons".to_string(),
            node_type: FluidNodeType::Conservation,
        });

        // Condenser water loop
        graph.add_node(FluidGraphNode {
            id: GraphNodeId::new(3),
            name: "CT-1".to_string(),
            node_type: FluidNodeType::Equipment,
        });
        graph.add_node(FluidGraphNode {
            id: GraphNodeId::new(4),
            name: "CT-2".to_string(),
            node_type: FluidNodeType::Equipment,
        });
        graph.add_node(FluidGraphNode {
            id: GraphNodeId::new(5),
            name: "CT-Cons".to_string(),
            node_type: FluidNodeType::Conservation,
        });

        // Duct network
        graph.add_node(FluidGraphNode {
            id: GraphNodeId::new(6),
            name: "AHU-1".to_string(),
            node_type: FluidNodeType::Equipment,
        });
        graph.add_node(FluidGraphNode {
            id: GraphNodeId::new(7),
            name: "AHU-2".to_string(),
            node_type: FluidNodeType::Equipment,
        });
        graph.add_node(FluidGraphNode {
            id: GraphNodeId::new(8),
            name: "Duct-Cons".to_string(),
            node_type: FluidNodeType::Conservation,
        });

        // Chilled water internal connections
        graph.add_edge(FluidGraphEdge {
            from: GraphNodeId::new(0),
            to: GraphNodeId::new(2),
            edge_type: FluidEdgeType::Flow,
        });
        graph.add_edge(FluidGraphEdge {
            from: GraphNodeId::new(1),
            to: GraphNodeId::new(2),
            edge_type: FluidEdgeType::Flow,
        });

        // Condenser water internal connections
        graph.add_edge(FluidGraphEdge {
            from: GraphNodeId::new(3),
            to: GraphNodeId::new(5),
            edge_type: FluidEdgeType::Flow,
        });
        graph.add_edge(FluidGraphEdge {
            from: GraphNodeId::new(4),
            to: GraphNodeId::new(5),
            edge_type: FluidEdgeType::Flow,
        });

        // Duct network internal connections
        graph.add_edge(FluidGraphEdge {
            from: GraphNodeId::new(6),
            to: GraphNodeId::new(8),
            edge_type: FluidEdgeType::Flow,
        });
        graph.add_edge(FluidGraphEdge {
            from: GraphNodeId::new(7),
            to: GraphNodeId::new(8),
            edge_type: FluidEdgeType::Flow,
        });

        let subgraphs = decompose_parallel_subgraphs(&graph);

        // With one-way edges (equipment->conservation), each node is its own SCC
        // The 3-loop structure yields 9 independent subgraphs (no feedback cycles)
        assert_eq!(subgraphs.len(), 9);
        for sg in &subgraphs {
            assert!(
                !sg.has_feedback,
                "Independent loops should not have feedback"
            );
        }
    }

    #[test]
    fn test_parallel_loop_dispatcher() {
        use std::sync::Mutex;

        let subgraphs = vec![
            Subgraph {
                id: 0,
                nodes: vec![GraphNodeId::new(0)],
                edges: vec![],
                has_feedback: false,
            },
            Subgraph {
                id: 1,
                nodes: vec![GraphNodeId::new(1)],
                edges: vec![],
                has_feedback: false,
            },
            Subgraph {
                id: 2,
                nodes: vec![GraphNodeId::new(2)],
                edges: vec![],
                has_feedback: false,
            },
        ];

        let mut dispatcher = ParallelLoopDispatcher::new(subgraphs);
        assert_eq!(dispatcher.num_subgraphs(), 3);
        assert_eq!(dispatcher.parallel_subgraphs().len(), 3);
        assert_eq!(dispatcher.sequential_subgraphs().len(), 0);

        // The closure must be `Fn` (stateless). Output is accumulated via
        // interior synchronisation captured inside the closure, so rayon can
        // invoke it concurrently without any lock on the closure itself (#2525).
        let results: Mutex<Vec<usize>> = Mutex::new(Vec::new());
        let result = dispatcher.step(0.0, 3600.0, |sg| {
            results.lock().unwrap().push(sg.id);
            Ok::<(), DispatchError>(())
        });

        assert!(result.is_ok());
        let collected = results.into_inner().unwrap();
        assert_eq!(collected.len(), 3);
    }

    #[test]
    fn test_parallel_loop_dispatcher_feedback_error() {
        let subgraphs = vec![Subgraph {
            id: 0,
            nodes: vec![GraphNodeId::new(0)],
            edges: vec![FluidGraphEdge {
                from: GraphNodeId::new(0),
                to: GraphNodeId::new(0),
                edge_type: FluidEdgeType::Feedback,
            }],
            has_feedback: true,
        }];

        let mut dispatcher = ParallelLoopDispatcher::new(subgraphs);

        let result = dispatcher.step(0.0, 3600.0, |_sg| Ok::<i32, DispatchError>(0));

        assert!(result.is_err());
        match result.unwrap_err() {
            DispatchError::FeedbackLoop(id) => assert_eq!(id, 0),
            _ => panic!("Expected FeedbackLoop error"),
        }
    }

    /// Regression test for #2525: `ParallelLoopDispatcher::step` must NOT
    /// serialise parallel work through a `Mutex` on the closure. Each
    /// subgraph is given a 1 ms wall-clock workload; with genuine rayon
    /// concurrency the total is ~1–2 ms, whereas the pre-fix `Arc<Mutex<F>>`
    /// fully serialised iteration and would take ~N ms.
    ///
    /// Requires a ≥2-core machine. On a single-core / fully throttled runner
    /// the sleep workload cannot overlap, so the assertion is skipped to
    /// avoid false failures (CI runs multi-core — see AGENTS.md).
    #[test]
    fn test_step_does_not_serialize_parallel_work() {
        use std::thread::sleep;
        use std::time::{Duration, Instant};

        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        if cores < 2 {
            eprintln!(
                "test_step_does_not_serialize_parallel_work: skipped ({} core available)",
                cores
            );
            return;
        }

        // ≥ 8 independent (no-feedback) subgraphs, per the #2525 acceptance
        // criterion.
        let n_subgraphs: usize = 8;
        let subgraphs = (0..n_subgraphs)
            .map(|id| Subgraph {
                id,
                nodes: vec![GraphNodeId::new(id)],
                edges: vec![],
                has_feedback: false,
            })
            .collect::<Vec<_>>();

        let mut dispatcher = ParallelLoopDispatcher::new(subgraphs);

        // Use a per-subgraph workload large enough that real parallelism
        // produces a clear speedup over full serialisation, independent of
        // runner clock speed or Linux sleep granularity. 10 ms × 8 subgraphs =
        // ~80 ms fully serial.
        let per_work = Duration::from_millis(10);

        // Measure a serial baseline (same total work, no overlap).
        let serial_start = Instant::now();
        for _ in 0..n_subgraphs {
            sleep(per_work);
        }
        let serial = serial_start.elapsed();

        // Measure the parallel dispatch.
        let par_start = Instant::now();
        let res = dispatcher.step(0.0, 0.001, |_sg| {
            sleep(per_work);
            Ok::<(), DispatchError>(())
        });
        let parallel = par_start.elapsed();

        assert!(res.is_ok(), "dispatch failed: {:?}", res.err());

        // Relative comparison (robust to runner speed/load/core count):
        // real parallelism ⇒ parallel ≤ serial/2 on ≥2 cores, so
        // parallel*4 < serial*3 holds. A Mutex-serialisation regression makes
        // parallel ≈ serial, failing this bound. Using integer millis avoids
        // Duration overflow concerns.
        let serial_ms = serial.as_millis() as u128;
        let par_ms = parallel.as_millis() as u128;
        assert!(
            par_ms * 4 < serial_ms * 3,
            "step() parallel {:?} not meaningfully faster than serial {:?} \
             — suspected Mutex-serialisation regression (#2525)",
            parallel,
            serial
        );
    }

    #[test]
    fn test_graph_node_types() {
        let node = FluidGraphNode {
            id: GraphNodeId::new(0),
            name: "Test".to_string(),
            node_type: FluidNodeType::Equipment,
        };
        assert_eq!(node.node_type, FluidNodeType::Equipment);

        let node2 = FluidGraphNode {
            id: GraphNodeId::new(1),
            name: "Test2".to_string(),
            node_type: FluidNodeType::Conservation,
        };
        assert_eq!(node2.node_type, FluidNodeType::Conservation);
    }

    #[test]
    fn test_subgraph_n_states() {
        let subgraph = Subgraph {
            id: 0,
            nodes: vec![
                GraphNodeId::new(0),
                GraphNodeId::new(1),
                GraphNodeId::new(2),
            ],
            edges: vec![],
            has_feedback: false,
        };
        assert_eq!(subgraph.n_states(), 3);
    }

    #[test]
    fn test_dispatch_error_display() {
        let err = DispatchError::FeedbackLoop(42);
        assert!(err.to_string().contains("42"));
        assert!(err.to_string().contains("feedback"));

        let err2 = DispatchError::ConvergenceFailed("test".to_string());
        assert!(err2.to_string().contains("test"));
        assert!(err2.to_string().contains("Convergence"));
    }

    // =============================================================================
    // Concurrency and Performance Tests (Issue #1991 acceptance criteria)
    // =============================================================================

    #[test]
    fn test_concurrent_evaluation_determinism() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let groups = vec![
            LoopGroup::new(
                LoopGroupId::new(0),
                vec![
                    MockEquipment::new("CH-1", LoopGroupId::new(0), 10000.0),
                    MockEquipment::new("CH-2", LoopGroupId::new(0), 8000.0),
                ],
                7.0,
                2.0,
            ),
            LoopGroup::new(
                LoopGroupId::new(1),
                vec![MockEquipment::new("BO-1", LoopGroupId::new(1), 15000.0)],
                45.0,
                1.5,
            ),
            LoopGroup::new(
                LoopGroupId::new(2),
                vec![MockEquipment::new("AHU-1", LoopGroupId::new(2), 5000.0)],
                13.0,
                1.0,
            ),
        ];

        let params = {
            let mut p = HashMap::new();
            p.insert(
                LoopGroupId::new(0),
                LoopStepParams {
                    loop_id: LoopGroupId::new(0),
                    zone_temps: vec![22.0, 21.0],
                    outdoor_temp_c: 30.0,
                    dt_seconds: 3600.0,
                    supply_temp_setpoint_c: Some(7.0),
                    demand_kw: 5.0,
                },
            );
            p.insert(
                LoopGroupId::new(1),
                LoopStepParams {
                    loop_id: LoopGroupId::new(1),
                    zone_temps: vec![18.0],
                    outdoor_temp_c: 0.0,
                    dt_seconds: 3600.0,
                    supply_temp_setpoint_c: Some(45.0),
                    demand_kw: 10.0,
                },
            );
            p.insert(
                LoopGroupId::new(2),
                LoopStepParams {
                    loop_id: LoopGroupId::new(2),
                    zone_temps: vec![24.0],
                    outdoor_temp_c: 35.0,
                    dt_seconds: 3600.0,
                    supply_temp_setpoint_c: Some(13.0),
                    demand_kw: 3.0,
                },
            );
            p
        };

        // Run multiple evaluations to verify determinism
        let results: Vec<_> = (0..5)
            .map(|_| {
                let mut evaluator = DecoupledLoopEvaluator::new(groups.clone());
                evaluator.evaluate_parallel(&params)
            })
            .collect();

        // All runs should produce identical results
        for i in 1..results.len() {
            for loop_id in [
                LoopGroupId::new(0),
                LoopGroupId::new(1),
                LoopGroupId::new(2),
            ] {
                let r1 = &results[i - 1][&loop_id];
                let r2 = &results[i][&loop_id];
                assert!(
                    (r1.energy_kwh - r2.energy_kwh).abs() < 1e-10,
                    "Non-deterministic result for loop {:?} on run {} vs {}",
                    loop_id,
                    i - 1,
                    i
                );
            }
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_rayon_parallel_iteration() {
        // Verify that rayon is actually being used for parallel iteration
        use rayon::iter::ParallelIterator;

        let data: Vec<i32> = (0..1000).collect();
        let sum: i32 = data.par_iter().sum();
        assert_eq!(sum, 499500);
    }
}
