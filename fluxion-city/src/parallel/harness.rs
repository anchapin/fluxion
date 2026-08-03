//! Parallel Harness for Urban Building Simulation (Issue #2034)
//!
//! Thread-safe parallel execution of urban radiation/thermal simulations
//! with configurable worker threads and memory-efficient building management.
//!
//! # Determinism Guarantee (Issue #2033)
//!
//! The parallel dispatcher produces deterministic results regardless of thread pool
//! size by:
//! 1. Pre-sorting nodes by ID before parallel iteration
//! 2. Using deterministic reduction order (BTreeMap)
//!
//! This means `RAYON_NUM_THREADS=1` and `RAYON_NUM_THREADS=8` produce identical results.

use std::collections::BTreeMap;
use std::time::Duration;
use thiserror::Error;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

const STEFAN_BOLTZMANN: f64 = 5.67e-8;

/// Errors that can occur during building step computation.
#[derive(Debug, Error)]
pub enum StepError {
    #[error("Building {0} has missing or invalid thermal data: {1}")]
    MissingData(String, String),

    #[error("Computation error for building {0}: {1}")]
    ComputationError(String, String),

    #[error("Empty graph: no buildings to step")]
    EmptyGraph,

    #[error("Invalid time step: {0}")]
    InvalidTimeStep(String),
}

/// Result of stepping a single building.
///
/// Contains per-building outputs: heat flow, temperature change, and radiation values.
#[derive(Debug, Clone)]
pub struct BuildingResult {
    /// Building identifier (UUID as string for display).
    pub building_id: String,
    /// Net heat flow into the building (W).
    pub heat_flow_w: f64,
    /// Temperature change during this step (K).
    pub temperature_change_k: f64,
    /// Solar radiation absorbed by the building (W).
    pub absorbed_solar_w: f64,
    /// Longwave radiation emitted by the building (W).
    pub emitted_longwave_w: f64,
    /// Surface temperature after the step (K).
    pub surface_temperature_k: f64,
}

impl BuildingResult {
    pub fn new(
        building_id: String,
        heat_flow_w: f64,
        temperature_change_k: f64,
        absorbed_solar_w: f64,
        emitted_longwave_w: f64,
        surface_temperature_k: f64,
    ) -> Self {
        Self {
            building_id,
            heat_flow_w,
            temperature_change_k,
            absorbed_solar_w,
            emitted_longwave_w,
            surface_temperature_k,
        }
    }
}

/// Per-building thermal data for urban simulation.
///
/// This is stored separately from the graph's BuildingNode to allow
/// mutable access during parallel stepping without borrowing conflicts.
#[derive(Debug, Clone)]
pub struct BuildingThermalData {
    /// Building identifier matching BuildingNode.id.
    pub id: uuid::Uuid,
    /// Floor area (m²).
    pub floor_area: f64,
    /// Wall U-value (W/m²·K).
    pub u_wall: f64,
    /// Roof U-value (W/m²·K).
    pub u_roof: f64,
    /// Floor U-value (W/m²·K).
    pub u_floor: f64,
    /// Current surface temperature (K).
    pub temperature: f64,
    /// Outdoor/air temperature (K).
    pub outdoor_temperature: f64,
    /// Solar radiation absorbed (W).
    pub absorbed_solar: f64,
    /// Longwave radiation emitted (W).
    pub emitted_longwave: f64,
}

impl BuildingThermalData {
    pub fn from_bounding_box(id: uuid::Uuid, bb: &crate::BoundingBox3D) -> Self {
        // Estimate floor area from bounding box
        let dx = bb.max_x - bb.min_x;
        let dy = bb.max_y - bb.min_y;
        let _dz = bb.max_z - bb.min_z;
        let floor_area = dx * dy;

        Self {
            id,
            floor_area,
            u_wall: 0.5,
            u_roof: 0.3,
            u_floor: 2.0,
            temperature: 293.15,
            outdoor_temperature: 293.15,
            absorbed_solar: 0.0,
            emitted_longwave: 0.0,
        }
    }

    /// Compute the step for this building given radiation conditions.
    pub fn step(
        &mut self,
        dt: &Duration,
        radiation: &UrbanRadiationSystem,
        outdoor_temp: f64,
    ) -> BuildingResult {
        let dt_hours = dt.as_secs_f64() / 3600.0;
        let surface_area = self.floor_area * 4.0;
        let wall_area = surface_area * 0.8;
        let roof_area = surface_area * 0.2;

        let q_solar = radiation.solar_irradiance * self.floor_area * radiation.absorptivity;
        let q_sky = radiation.sky_temperature.powi(4) - self.temperature.powi(4);
        let q_longwave = radiation.emissivity * STEFAN_BOLTZMANN * wall_area * q_sky;
        let q_conduction_wall = self.u_wall * wall_area * (outdoor_temp - self.temperature);
        let q_conduction_roof = self.u_roof * roof_area * (outdoor_temp - self.temperature);
        let q_conduction_floor =
            self.u_floor * (self.floor_area * 0.1) * (self.outdoor_temperature - self.temperature);

        let net_gain =
            q_solar + q_longwave + q_conduction_wall + q_conduction_roof + q_conduction_floor;
        let heat_capacity = self.floor_area * 500.0 * 1005.0;

        let temperature_change = (net_gain / heat_capacity) * dt_hours;
        self.temperature += temperature_change;
        self.temperature = self.temperature.clamp(200.0, 400.0);
        self.absorbed_solar = q_solar;
        self.emitted_longwave = q_longwave;

        BuildingResult::new(
            self.id.to_string(),
            net_gain,
            temperature_change,
            q_solar,
            q_longwave,
            self.temperature,
        )
    }
}

#[derive(Debug, Clone)]
pub struct BuildingGroup {
    pub id: u32,
    pub area: f64,
    pub u_wall: f64,
    pub u_roof: f64,
    pub u_floor: f64,
    pub temperature: f64,
    pub outdoor_temperature: f64,
    pub absorbed_solar: f64,
    pub emitted_longwave: f64,
}

impl BuildingGroup {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            area: 100.0,
            u_wall: 0.5,
            u_roof: 0.3,
            u_floor: 2.0,
            temperature: 293.15,
            outdoor_temperature: 293.15,
            absorbed_solar: 0.0,
            emitted_longwave: 0.0,
        }
    }

    pub fn with_area(mut self, area: f64) -> Self {
        self.area = area;
        self
    }

    pub fn with_u_values(mut self, u_wall: f64, u_roof: f64, u_floor: f64) -> Self {
        self.u_wall = u_wall;
        self.u_roof = u_roof;
        self.u_floor = u_floor;
        self
    }

    pub fn step(&mut self, dt: &Duration, radiation: &UrbanRadiationSystem, outdoor_temp: f64) {
        let dt_hours = dt.as_secs_f64() / 3600.0;
        let surface_area = self.area * 4.0;
        let wall_area = surface_area * 0.8;
        let roof_area = surface_area * 0.2;

        let q_solar = radiation.solar_irradiance * self.area * radiation.absorptivity;
        let q_sky = radiation.sky_temperature.powi(4) - self.temperature.powi(4);
        let q_longwave = radiation.emissivity * STEFAN_BOLTZMANN * wall_area * q_sky;
        let q_conduction_wall = self.u_wall * wall_area * (outdoor_temp - self.temperature);
        let q_conduction_roof = self.u_roof * roof_area * (outdoor_temp - self.temperature);
        let q_conduction_floor =
            self.u_floor * (self.area * 0.1) * (self.outdoor_temperature - self.temperature);

        let net_gain =
            q_solar + q_longwave + q_conduction_wall + q_conduction_roof + q_conduction_floor;
        let heat_capacity = self.area * 500.0 * 1005.0;

        self.temperature += (net_gain / heat_capacity) * dt_hours;
        self.temperature = self.temperature.clamp(200.0, 400.0);
        self.absorbed_solar = q_solar;
        self.emitted_longwave = q_longwave;
    }
}

#[derive(Debug, Clone)]
pub struct UrbanRadiationSystem {
    pub solar_irradiance: f64,
    pub sky_temperature: f64,
    pub emissivity: f64,
    pub absorptivity: f64,
    pub latitude: f64,
    pub longitude: f64,
}

impl UrbanRadiationSystem {
    pub fn new(
        solar_irradiance: f64,
        sky_temperature: f64,
        emissivity: f64,
        absorptivity: f64,
        latitude: f64,
        longitude: f64,
    ) -> Self {
        Self {
            solar_irradiance,
            sky_temperature,
            emissivity,
            absorptivity,
            latitude,
            longitude,
        }
    }
}

#[cfg(feature = "parallel")]
#[derive(Debug, Clone)]
pub struct UrbanStepDispatcher {
    buildings: Vec<BuildingGroup>,
    num_threads: usize,
}

#[cfg(feature = "parallel")]
impl UrbanStepDispatcher {
    pub fn with_buildings(buildings: Vec<BuildingGroup>) -> Self {
        Self {
            buildings,
            num_threads: rayon::current_num_threads(),
        }
    }

    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    pub fn num_buildings(&self) -> usize {
        self.buildings.len()
    }

    pub fn step_all(&mut self, dt: Duration, radiation: &UrbanRadiationSystem, outdoor_temp: f64) {
        let num_threads = self.num_threads;
        let chunk_size = (self.buildings.len() / num_threads).max(1);

        self.buildings.par_chunks_mut(chunk_size).for_each(|chunk| {
            for building in chunk.iter_mut() {
                building.step(&dt, radiation, outdoor_temp);
            }
        });
    }

    pub fn step_sequential(
        &mut self,
        dt: &Duration,
        radiation: &UrbanRadiationSystem,
        outdoor_temp: f64,
    ) {
        for building in self.buildings.iter_mut() {
            building.step(dt, radiation, outdoor_temp);
        }
    }

    pub fn get_buildings(&self) -> &[BuildingGroup] {
        &self.buildings
    }

    pub fn set_num_threads(&mut self, num_threads: usize) {
        self.num_threads = num_threads;
    }
}

#[cfg(not(feature = "parallel"))]
#[derive(Debug, Clone)]
pub struct UrbanStepDispatcher {
    buildings: Vec<BuildingGroup>,
}

#[cfg(not(feature = "parallel"))]
impl UrbanStepDispatcher {
    pub fn with_buildings(buildings: Vec<BuildingGroup>) -> Self {
        Self { buildings }
    }

    pub fn with_threads(mut self, _num_threads: usize) -> Self {
        self
    }

    pub fn num_buildings(&self) -> usize {
        self.buildings.len()
    }

    pub fn step_all(&mut self, dt: Duration, radiation: &UrbanRadiationSystem, outdoor_temp: f64) {
        for building in self.buildings.iter_mut() {
            building.step(&dt, radiation, outdoor_temp);
        }
    }

    pub fn step_sequential(
        &mut self,
        dt: &Duration,
        radiation: &UrbanRadiationSystem,
        outdoor_temp: f64,
    ) {
        self.step_all(dt, radiation, outdoor_temp);
    }

    pub fn get_buildings(&self) -> &[BuildingGroup] {
        &self.buildings
    }

    pub fn set_num_threads(&mut self, _num_threads: usize) {}
}

// =============================================================================
// UrbanGraphStepDispatcher: Rayon-based parallel dispatcher for UrbanGraph
// Issue #2032
// =============================================================================

#[cfg(feature = "parallel")]
use crate::urban_graph::{BuildingNode, SpatialEdge, UrbanGraph};

/// Parallel dispatcher for stepping buildings in an UrbanGraph using Rayon.
///
/// This dispatcher operates on the graph structure directly, stepping each
/// building node in parallel using `rayon::par_iter()`. Results are collected
/// into a `Vec<BuildingResult>` for aggregation.
///
/// # Type Parameters
/// - `N`: Node data (must be `BuildingNode` or compatible)
/// - `E`: Edge data (unused in stepping)
///
/// # Example
/// ```ignore
/// let dispatcher = UrbanGraphStepDispatcher::new(&graph, thermal_data);
/// let results = dispatcher.step_buildings(dt, &radiation, outdoor_temp)?;
/// ```
#[cfg(feature = "parallel")]
#[derive(Debug)]
pub struct UrbanGraphStepDispatcher<'a> {
    graph: &'a UrbanGraph<BuildingNode, SpatialEdge>,
    thermal_data: Vec<BuildingThermalData>,
    num_threads: usize,
}

#[cfg(feature = "parallel")]
impl<'a> UrbanGraphStepDispatcher<'a> {
    /// Create a new dispatcher from an UrbanGraph with default thermal data.
    ///
    /// Thermal data is initialized from bounding boxes in the graph nodes.
    pub fn new(graph: &'a UrbanGraph<BuildingNode, SpatialEdge>) -> Self {
        let thermal_data: Vec<BuildingThermalData> = graph
            .node_indices()
            .map(|idx| {
                let node = graph.node_weight(idx);
                BuildingThermalData::from_bounding_box(node.id, &node.bounding_box)
            })
            .collect();

        Self {
            graph,
            thermal_data,
            num_threads: rayon::current_num_threads(),
        }
    }

    /// Create a dispatcher with pre-populated thermal data.
    ///
    /// The thermal data must have one entry per graph node, in matching order.
    pub fn with_thermal_data(
        graph: &'a UrbanGraph<BuildingNode, SpatialEdge>,
        thermal_data: Vec<BuildingThermalData>,
    ) -> Self {
        Self {
            graph,
            thermal_data,
            num_threads: rayon::current_num_threads(),
        }
    }

    /// Set the number of threads for parallel execution.
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Returns the number of buildings in the graph.
    pub fn num_buildings(&self) -> usize {
        self.graph.node_count()
    }

    /// Step all buildings in parallel using Rayon.
    ///
    /// Iterates over all buildings in the graph in parallel, computes radiation
    /// flux, internal gains, and temperature changes, and returns results.
    ///
    /// # Determinism (Issue #2033)
    /// Results are returned in a `BTreeMap<Uuid, BuildingResult>` sorted by building ID,
    /// ensuring deterministic output regardless of thread pool size or scheduling order.
    ///
    /// # Arguments
    /// * `dt` - Time step duration
    /// * `radiation` - Solar and atmospheric radiation conditions
    /// * `outdoor_temp` - Ambient outdoor temperature (K)
    ///
    /// # Returns
    /// * `Ok(BTreeMap<Uuid, BuildingResult>)` - Per-building results sorted by building ID
    /// * `Err(StepError)` - If the graph is empty or computation fails
    pub fn step_buildings(
        &self,
        dt: Duration,
        radiation: &UrbanRadiationSystem,
        outdoor_temp: f64,
    ) -> Result<BTreeMap<uuid::Uuid, BuildingResult>, StepError> {
        if self.graph.node_count() == 0 {
            return Err(StepError::EmptyGraph);
        }

        if dt.as_secs() == 0 {
            return Err(StepError::InvalidTimeStep(
                "Time step must be positive".into(),
            ));
        }

        // Pre-sort nodes by ID for deterministic iteration order (Issue #2033)
        let mut node_indices: Vec<_> = self.graph.node_indices().collect();
        node_indices.sort_by_key(|&idx| self.graph.node_weight(idx).id);

        // Use rayon par_iter over sorted indices for true parallelism
        // Results collected into BTreeMap for deterministic reduction order
        let results: BTreeMap<uuid::Uuid, BuildingResult> = node_indices
            .par_iter()
            .map(|&idx| {
                // Get thermal data for this node
                let node = self.graph.node_weight(idx);

                // Find thermal data by matching UUID
                let thermal = self
                    .thermal_data
                    .iter()
                    .find(|t| t.id == node.id)
                    .cloned()
                    .unwrap_or_else(|| {
                        BuildingThermalData::from_bounding_box(node.id, &node.bounding_box)
                    });

                // Compute step - use a local mutable copy
                let mut local_thermal = thermal;
                let result = local_thermal.step(&dt, radiation, outdoor_temp);
                (node.id, result)
            })
            .collect();

        Ok(results)
    }

    /// Step all buildings sequentially (for comparison/debugging).
    ///
    /// Returns results in sorted order by building ID for deterministic comparison
    /// with parallel results.
    pub fn step_buildings_sequential(
        &self,
        dt: &Duration,
        radiation: &UrbanRadiationSystem,
        outdoor_temp: f64,
    ) -> Result<BTreeMap<uuid::Uuid, BuildingResult>, StepError> {
        if self.graph.node_count() == 0 {
            return Err(StepError::EmptyGraph);
        }

        // Pre-sort nodes by ID for deterministic iteration order (Issue #2033)
        let mut node_indices: Vec<_> = self.graph.node_indices().collect();
        node_indices.sort_by_key(|&idx| self.graph.node_weight(idx).id);

        let mut results = BTreeMap::new();

        for idx in node_indices {
            let node = self.graph.node_weight(idx);
            let thermal = self
                .thermal_data
                .iter()
                .find(|t| t.id == node.id)
                .cloned()
                .unwrap_or_else(|| {
                    BuildingThermalData::from_bounding_box(node.id, &node.bounding_box)
                });

            let mut local_thermal = thermal;
            let result = local_thermal.step(dt, radiation, outdoor_temp);
            results.insert(node.id, result);
        }

        Ok(results)
    }

    /// Get a reference to the thermal data for inspection.
    pub fn get_thermal_data(&self) -> &[BuildingThermalData] {
        &self.thermal_data
    }

    /// Set the number of threads for the thread pool.
    pub fn set_num_threads(&mut self, num_threads: usize) {
        self.num_threads = num_threads;
    }
}

// Sequential fallback for non-parallel feature
#[cfg(not(feature = "parallel"))]
#[derive(Debug)]
pub struct UrbanGraphStepDispatcher<'a> {
    _phantom: std::marker::PhantomData<&'a ()>,
}

#[cfg(not(feature = "parallel"))]
impl<'a> UrbanGraphStepDispatcher<'a> {
    pub fn new(_graph: &'a UrbanGraph<BuildingNode, SpatialEdge>) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn with_thermal_data(
        _graph: &'a UrbanGraph<BuildingNode, SpatialEdge>,
        _thermal_data: Vec<BuildingThermalData>,
    ) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn with_threads(self, _num_threads: usize) -> Self {
        self
    }

    pub fn num_buildings(&self) -> usize {
        0
    }

    pub fn step_buildings(
        &self,
        _dt: Duration,
        _radiation: &UrbanRadiationSystem,
        _outdoor_temp: f64,
    ) -> Result<BTreeMap<uuid::Uuid, BuildingResult>, StepError> {
        Err(StepError::ComputationError(
            "parallel feature not enabled".into(),
            "Rebuild with --features parallel".into(),
        ))
    }

    pub fn step_buildings_sequential(
        &self,
        _dt: &Duration,
        _radiation: &UrbanRadiationSystem,
        _outdoor_temp: f64,
    ) -> Result<BTreeMap<uuid::Uuid, BuildingResult>, StepError> {
        Err(StepError::ComputationError(
            "parallel feature not enabled".into(),
            "Rebuild with --features parallel".into(),
        ))
    }

    pub fn get_thermal_data(&self) -> &[BuildingThermalData] {
        &[]
    }

    pub fn set_num_threads(&mut self, _num_threads: usize) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_building_group_creation() {
        let building = BuildingGroup::new(1);
        assert_eq!(building.id, 1);
        assert_eq!(building.area, 100.0);
    }

    #[test]
    fn test_building_group_with_area() {
        let building = BuildingGroup::new(1).with_area(200.0);
        assert_eq!(building.area, 200.0);
    }

    #[test]
    fn test_building_group_with_u_values() {
        let building = BuildingGroup::new(1).with_u_values(0.3, 0.2, 1.5);
        assert_eq!(building.u_wall, 0.3);
        assert_eq!(building.u_roof, 0.2);
        assert_eq!(building.u_floor, 1.5);
    }

    #[test]
    fn test_building_step() {
        let mut building = BuildingGroup::new(1);
        let radiation = UrbanRadiationSystem::new(500.0, 270.0, 0.9, 0.7, 45.0, 0.0);
        let dt = Duration::from_secs(3600);
        let initial_temp = building.temperature;

        building.step(&dt, &radiation, 300.0);
        assert_ne!(building.temperature, initial_temp);
    }

    #[test]
    fn test_urban_radiation_system_creation() {
        let radiation = UrbanRadiationSystem::new(800.0, 120.0, 0.2, 0.85, 0.1, 2.0);
        assert_eq!(radiation.solar_irradiance, 800.0);
        assert_eq!(radiation.sky_temperature, 120.0);
    }

    #[test]
    fn test_urban_step_dispatcher_creation() {
        let buildings: Vec<BuildingGroup> = (0..5).map(|i| BuildingGroup::new(i as u32)).collect();
        let dispatcher = UrbanStepDispatcher::with_buildings(buildings);
        assert_eq!(dispatcher.num_buildings(), 5);
    }

    #[test]
    fn test_urban_step_dispatcher_step() {
        let buildings: Vec<BuildingGroup> = (0..10).map(|i| BuildingGroup::new(i as u32)).collect();
        let mut dispatcher = UrbanStepDispatcher::with_buildings(buildings);
        let radiation = UrbanRadiationSystem::new(500.0, 270.0, 0.9, 0.7, 45.0, 0.0);
        let dt = Duration::from_secs(3600);

        dispatcher.step_all(dt, &radiation, 300.0);
        assert_eq!(dispatcher.num_buildings(), 10);
    }

    #[test]
    fn test_sequential_vs_parallel_consistency() {
        let buildings: Vec<BuildingGroup> = (0..20)
            .map(|i| BuildingGroup::new(i as u32).with_area(100.0 + i as f64))
            .collect();
        let mut dispatcher_seq = UrbanStepDispatcher::with_buildings(buildings.clone());
        let mut dispatcher_par = UrbanStepDispatcher::with_buildings(buildings.clone());

        let radiation = UrbanRadiationSystem::new(500.0, 270.0, 0.9, 0.7, 45.0, 0.0);
        let dt = Duration::from_secs(3600);

        dispatcher_seq.step_sequential(&dt, &radiation, 300.0);
        dispatcher_par.step_all(dt, &radiation, 300.0);

        let seq_buildings = dispatcher_seq.get_buildings();
        let par_buildings = dispatcher_par.get_buildings();

        for (seq, par) in seq_buildings.iter().zip(par_buildings.iter()) {
            approx::assert_abs_diff_eq!(seq.temperature, par.temperature, epsilon = 1e-10);
        }
    }

    // =====================================================================
    // UrbanGraphStepDispatcher tests (Issue #2032)
    // =====================================================================

    #[cfg(feature = "parallel")]
    #[test]
    fn test_urban_graph_step_dispatcher_creation() {
        use crate::urban_graph::{BoundingBox3D, BuildingNode, UrbanGraph};
        use uuid::Uuid;

        let mut graph: UrbanGraph<BuildingNode, SpatialEdge> = UrbanGraph::new();
        let bb = BoundingBox3D::new(0.0, 0.0, 0.0, 10.0, 10.0, 30.0);
        graph.add_building(BuildingNode::new(Uuid::new_v4(), bb));

        let dispatcher = UrbanGraphStepDispatcher::new(&graph);
        assert_eq!(dispatcher.num_buildings(), 1);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_urban_graph_step_dispatcher_empty_graph() {
        use crate::urban_graph::{BuildingNode, UrbanGraph};

        let graph: UrbanGraph<BuildingNode, SpatialEdge> = UrbanGraph::new();
        let dispatcher = UrbanGraphStepDispatcher::new(&graph);

        let radiation = UrbanRadiationSystem::new(500.0, 270.0, 0.9, 0.7, 45.0, 0.0);
        let dt = Duration::from_secs(3600);

        let result = dispatcher.step_buildings(dt, &radiation, 300.0);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StepError::EmptyGraph));
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_urban_graph_step_dispatcher_parallel_vs_sequential() {
        use crate::urban_graph::{BoundingBox3D, BuildingNode, UrbanGraph};
        use uuid::Uuid;

        let mut graph: UrbanGraph<BuildingNode, SpatialEdge> = UrbanGraph::new();
        for i in 0..5 {
            let bb = BoundingBox3D::new(
                0.0 + i as f64 * 15.0,
                0.0,
                0.0,
                10.0 + i as f64 * 15.0,
                10.0,
                30.0,
            );
            graph.add_building(BuildingNode::new(Uuid::new_v4(), bb));
        }

        let dispatcher_par = UrbanGraphStepDispatcher::new(&graph);
        let dispatcher_seq = UrbanGraphStepDispatcher::new(&graph);

        let radiation = UrbanRadiationSystem::new(500.0, 270.0, 0.9, 0.7, 45.0, 0.0);
        let dt = Duration::from_secs(3600);

        let results_par = dispatcher_par
            .step_buildings(dt, &radiation, 300.0)
            .expect("parallel step should succeed");
        let results_seq = dispatcher_seq
            .step_buildings_sequential(&dt, &radiation, 300.0)
            .expect("sequential step should succeed");

        assert_eq!(results_par.len(), 5);
        assert_eq!(results_seq.len(), 5);

        // Results should be identical (both return BTreeMap sorted by ID)
        for (id, par_result) in results_par.iter() {
            let seq_result = results_seq
                .get(id)
                .expect("building ID should exist in sequential results");
            approx::assert_abs_diff_eq!(
                par_result.temperature_change_k,
                seq_result.temperature_change_k,
                epsilon = 1e-10
            );
            approx::assert_abs_diff_eq!(
                par_result.surface_temperature_k,
                seq_result.surface_temperature_k,
                epsilon = 1e-10
            );
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_urban_graph_step_dispatcher_results_valid() {
        use crate::urban_graph::{BoundingBox3D, BuildingNode, UrbanGraph};
        use uuid::Uuid;

        let mut graph: UrbanGraph<BuildingNode, SpatialEdge> = UrbanGraph::new();
        let bb = BoundingBox3D::new(0.0, 0.0, 0.0, 10.0, 10.0, 30.0);
        graph.add_building(BuildingNode::new(Uuid::new_v4(), bb));

        let dispatcher = UrbanGraphStepDispatcher::new(&graph);
        let radiation = UrbanRadiationSystem::new(500.0, 270.0, 0.9, 0.7, 45.0, 0.0);
        let dt = Duration::from_secs(3600);

        let results = dispatcher
            .step_buildings(dt, &radiation, 300.0)
            .expect("step should succeed");

        assert_eq!(results.len(), 1);
        let result = results.values().next().expect("should have one result");
        assert!(result.heat_flow_w.is_finite());
        assert!(result.temperature_change_k.is_finite());
        assert!(result.surface_temperature_k > 0.0);
        assert!(result.absorbed_solar_w >= 0.0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_urban_graph_step_dispatcher_invalid_timestep() {
        use crate::urban_graph::{BoundingBox3D, BuildingNode, UrbanGraph};
        use uuid::Uuid;

        let mut graph: UrbanGraph<BuildingNode, SpatialEdge> = UrbanGraph::new();
        let bb = BoundingBox3D::new(0.0, 0.0, 0.0, 10.0, 10.0, 30.0);
        graph.add_building(BuildingNode::new(Uuid::new_v4(), bb));

        let dispatcher = UrbanGraphStepDispatcher::new(&graph);
        let radiation = UrbanRadiationSystem::new(500.0, 270.0, 0.9, 0.7, 45.0, 0.0);
        let dt = Duration::from_secs(0); // Invalid zero timestep

        let result = dispatcher.step_buildings(dt, &radiation, 300.0);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), StepError::InvalidTimeStep(_)));
    }

    // =====================================================================
    // Issue #2033: Determinism Tests
    // =====================================================================

    #[cfg(feature = "parallel")]
    #[test]
    fn test_deterministic_results_with_thread_count_variation() {
        // Verify that parallel execution produces identical results regardless of thread count
        // by comparing single-threaded vs multi-threaded execution.
        use crate::urban_graph::{BoundingBox3D, BuildingNode, UrbanGraph};
        use rayon::ThreadPool;
        use std::sync::Arc;

        // Create a graph with multiple buildings
        let mut graph: UrbanGraph<BuildingNode, SpatialEdge> = UrbanGraph::new();
        let building_ids: Vec<uuid::Uuid> = (0..20)
            .map(|i| {
                let bb = BoundingBox3D::new(
                    0.0 + i as f64 * 15.0,
                    0.0,
                    0.0,
                    10.0 + i as f64 * 15.0,
                    10.0,
                    30.0,
                );
                let id = uuid::Uuid::new_v4();
                graph.add_building(BuildingNode::new(id, bb));
                id
            })
            .collect();

        let radiation = UrbanRadiationSystem::new(500.0, 270.0, 0.9, 0.7, 45.0, 0.0);
        let dt = Duration::from_secs(3600);

        // Run with thread pool of 1
        let pool1 = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("failed to build thread pool with 1 thread");
        let dispatcher1 = UrbanGraphStepDispatcher::new(&graph);
        let results1: std::collections::BTreeMap<uuid::Uuid, BuildingResult> =
            pool1.install(|| {
                dispatcher1
                    .step_buildings(dt, &radiation, 300.0)
                    .expect("step should succeed")
            });

        // Run with thread pool of 8
        let pool8 = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("failed to build thread pool with 8 threads");
        let dispatcher8 = UrbanGraphStepDispatcher::new(&graph);
        let results8: std::collections::BTreeMap<uuid::Uuid, BuildingResult> =
            pool8.install(|| {
                dispatcher8
                    .step_buildings(dt, &radiation, 300.0)
                    .expect("step should succeed")
            });

        // Verify same keys
        assert_eq!(results1.len(), results8.len());
        assert_eq!(
            results1.keys().copied().collect::<Vec<_>>(),
            results8.keys().copied().collect::<Vec<_>>()
        );

        // Verify identical results for each building
        for id in &building_ids {
            let r1 = results1.get(id).expect("building should exist");
            let r8 = results8
                .get(id)
                .expect("building should exist in 8-thread results");
            approx::assert_abs_diff_eq!(r1.heat_flow_w, r8.heat_flow_w, epsilon = 1e-10);
            approx::assert_abs_diff_eq!(
                r1.temperature_change_k,
                r8.temperature_change_k,
                epsilon = 1e-10
            );
            approx::assert_abs_diff_eq!(
                r1.surface_temperature_k,
                r8.surface_temperature_k,
                epsilon = 1e-10
            );
            approx::assert_abs_diff_eq!(r1.absorbed_solar_w, r8.absorbed_solar_w, epsilon = 1e-10);
            approx::assert_abs_diff_eq!(
                r1.emitted_longwave_w,
                r8.emitted_longwave_w,
                epsilon = 1e-10
            );
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_deterministic_results_multiple_runs() {
        // Verify that running the same computation multiple times produces identical results
        use crate::urban_graph::{BoundingBox3D, BuildingNode, UrbanGraph};

        let mut graph: UrbanGraph<BuildingNode, SpatialEdge> = UrbanGraph::new();
        for i in 0..10 {
            let bb = BoundingBox3D::new(
                i as f64 * 10.0,
                0.0,
                0.0,
                10.0 + i as f64 * 10.0,
                10.0,
                30.0,
            );
            graph.add_building(BuildingNode::new(uuid::Uuid::new_v4(), bb));
        }

        let radiation = UrbanRadiationSystem::new(500.0, 270.0, 0.9, 0.7, 45.0, 0.0);
        let dt = Duration::from_secs(3600);

        // Run multiple times
        let results1 = UrbanGraphStepDispatcher::new(&graph)
            .step_buildings(dt, &radiation, 300.0)
            .expect("step should succeed");

        let results2 = UrbanGraphStepDispatcher::new(&graph)
            .step_buildings(dt, &radiation, 300.0)
            .expect("step should succeed");

        // Results should be identical
        assert_eq!(results1.len(), results2.len());
        for (id, r1) in results1.iter() {
            let r2 = results2
                .get(id)
                .expect("building should exist in second run");
            approx::assert_abs_diff_eq!(r1.heat_flow_w, r2.heat_flow_w, epsilon = 1e-10);
            approx::assert_abs_diff_eq!(
                r1.temperature_change_k,
                r2.temperature_change_k,
                epsilon = 1e-10
            );
            approx::assert_abs_diff_eq!(
                r1.surface_temperature_k,
                r2.surface_temperature_k,
                epsilon = 1e-10
            );
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_deterministic_results_sorted_order() {
        // Verify that results are returned in sorted order by building ID
        use crate::urban_graph::{BoundingBox3D, BuildingNode, UrbanGraph};
        use std::collections::BTreeMap;

        let mut graph: UrbanGraph<BuildingNode, SpatialEdge> = UrbanGraph::new();
        // Add buildings in random order (by using decreasing IDs)
        let ids: Vec<uuid::Uuid> = (0..5).map(|_| uuid::Uuid::new_v4()).collect();
        for id in ids.iter().rev() {
            let bb = BoundingBox3D::new(0.0, 0.0, 0.0, 10.0, 10.0, 30.0);
            graph.add_building(BuildingNode::new(*id, bb));
        }

        let radiation = UrbanRadiationSystem::new(500.0, 270.0, 0.9, 0.7, 45.0, 0.0);
        let dt = Duration::from_secs(3600);

        let results = UrbanGraphStepDispatcher::new(&graph)
            .step_buildings(dt, &radiation, 300.0)
            .expect("step should succeed");

        // Verify results are in sorted key order
        let keys: Vec<_> = results.keys().copied().collect();
        let mut sorted_keys = keys.clone();
        sorted_keys.sort();

        assert_eq!(keys, sorted_keys, "results should be sorted by building ID");

        // Verify it's a BTreeMap behavior (sorted)
        let mut prev_id = uuid::Uuid::nil();
        for id in results.keys() {
            assert!(*id > prev_id, "IDs should be strictly increasing");
            prev_id = *id;
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_deterministic_parallel_vs_sequential_100_iterations() {
        // Stress test: run parallel vs sequential 100 times to catch any non-determinism
        use crate::urban_graph::{BoundingBox3D, BuildingNode, UrbanGraph};

        let mut graph: UrbanGraph<BuildingNode, SpatialEdge> = UrbanGraph::new();
        for i in 0..10 {
            let bb = BoundingBox3D::new(
                i as f64 * 12.0,
                0.0,
                0.0,
                10.0 + i as f64 * 12.0,
                10.0,
                30.0,
            );
            graph.add_building(BuildingNode::new(uuid::Uuid::new_v4(), bb));
        }

        let radiation = UrbanRadiationSystem::new(500.0, 270.0, 0.9, 0.7, 45.0, 0.0);
        let dt = Duration::from_secs(3600);

        for iteration in 0..100 {
            let dispatcher_par = UrbanGraphStepDispatcher::new(&graph);
            let dispatcher_seq = UrbanGraphStepDispatcher::new(&graph);

            let results_par = dispatcher_par
                .step_buildings(dt, &radiation, 300.0)
                .expect("parallel step should succeed");
            let results_seq = dispatcher_seq
                .step_buildings_sequential(&dt, &radiation, 300.0)
                .expect("sequential step should succeed");

            assert_eq!(results_par.len(), results_seq.len());

            for (id, par_result) in results_par.iter() {
                let seq_result = results_seq.get(id).expect("building should exist");
                let diff =
                    (par_result.temperature_change_k - seq_result.temperature_change_k).abs();
                assert!(
                    diff < 1e-10,
                    "iteration {}: temperature_change_k mismatch for building {:?}: diff={}",
                    iteration,
                    id,
                    diff
                );
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_deterministic_btreemap_collection() {
        // Verify that collecting into BTreeMap produces deterministic key ordering
        use crate::urban_graph::{BoundingBox3D, BuildingNode, UrbanGraph};
        use std::collections::BTreeMap;

        let mut graph: UrbanGraph<BuildingNode, SpatialEdge> = UrbanGraph::new();
        // Create explicit UUIDs that will be shuffled
        let ids: Vec<uuid::Uuid> = (0..20).map(|_| uuid::Uuid::new_v4()).collect();

        // Add in reverse order to test sorting
        for id in ids.iter().rev() {
            let bb = BoundingBox3D::new(0.0, 0.0, 0.0, 10.0, 10.0, 30.0);
            graph.add_building(BuildingNode::new(*id, bb));
        }

        let radiation = UrbanRadiationSystem::new(500.0, 270.0, 0.9, 0.7, 45.0, 0.0);
        let dt = Duration::from_secs(3600);

        let results = UrbanGraphStepDispatcher::new(&graph)
            .step_buildings(dt, &radiation, 300.0)
            .expect("step should succeed");

        // Verify BTreeMap is sorted
        let keys: Vec<_> = results.keys().copied().collect();
        let mut sorted_keys = keys.clone();
        sorted_keys.sort();

        assert_eq!(
            keys, sorted_keys,
            "BTreeMap should maintain sorted key order"
        );

        // Verify all 20 buildings present
        assert_eq!(results.len(), 20);
    }
}
