//! Python bindings for 9R4C Multi-Node Thermal Solver
//!
//! This module provides PyO3 bindings for the 9R4C multi-node thermal model
//! used for heavy-mass buildings (Case 900+ series).

use crate::physics::multi_node_solver::{MultiNodeSolver, SurfaceExteriorTemperatures};
use fluxion_core::multi_node::{MassAirCouplingMode, MultiNodeThermalMass, ThermalMassNode};
use pyo3::prelude::*;

#[pyclass(name = "ThermalMassNode")]
#[derive(Clone, Debug)]
pub struct PyThermalMassNode {
    pub temperature: f64,
    pub capacitance: f64,
    pub h_tr_ms: f64,
    pub h_tr_em: f64,
    pub h_tr_me: f64,
    pub heat_flux_cumulative: f64,
}

impl From<&ThermalMassNode> for PyThermalMassNode {
    fn from(node: &ThermalMassNode) -> Self {
        Self {
            temperature: node.temperature,
            capacitance: node.capacitance,
            h_tr_ms: node.h_tr_ms,
            h_tr_em: node.h_tr_em,
            h_tr_me: node.h_tr_me,
            heat_flux_cumulative: node.heat_flux_cumulative,
        }
    }
}

#[pymethods]
impl PyThermalMassNode {
    #[new]
    pub fn new(
        temperature: f64,
        capacitance: f64,
        h_tr_ms: f64,
        h_tr_em: f64,
        h_tr_me: f64,
    ) -> Self {
        Self {
            temperature,
            capacitance,
            h_tr_ms,
            h_tr_em,
            h_tr_me,
            heat_flux_cumulative: 0.0,
        }
    }

    #[getter]
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    #[setter]
    pub fn set_temperature(&mut self, value: f64) {
        self.temperature = value;
    }

    #[getter]
    pub fn capacitance(&self) -> f64 {
        self.capacitance
    }

    #[setter]
    pub fn set_capacitance(&mut self, value: f64) {
        self.capacitance = value;
    }

    #[getter]
    pub fn h_tr_ms(&self) -> f64 {
        self.h_tr_ms
    }

    #[setter]
    pub fn set_h_tr_ms(&mut self, value: f64) {
        self.h_tr_ms = value;
    }

    #[getter]
    pub fn h_tr_em(&self) -> f64 {
        self.h_tr_em
    }

    #[setter]
    pub fn set_h_tr_em(&mut self, value: f64) {
        self.h_tr_em = value;
    }

    #[getter]
    pub fn h_tr_me(&self) -> f64 {
        self.h_tr_me
    }

    #[setter]
    pub fn set_h_tr_me(&mut self, value: f64) {
        self.h_tr_me = value;
    }

    #[getter]
    pub fn heat_flux_cumulative(&self) -> f64 {
        self.heat_flux_cumulative
    }
}

#[pyclass(name = "MultiNodeThermalMass")]
#[derive(Clone, Debug)]
pub struct PyMultiNodeThermalMass {
    pub wall: PyThermalMassNode,
    pub roof: PyThermalMassNode,
    pub floor: PyThermalMassNode,
    pub internal: PyThermalMassNode,
}

impl From<&MultiNodeThermalMass> for PyMultiNodeThermalMass {
    fn from(mass: &MultiNodeThermalMass) -> Self {
        Self {
            wall: PyThermalMassNode::from(&mass.wall),
            roof: PyThermalMassNode::from(&mass.roof),
            floor: PyThermalMassNode::from(&mass.floor),
            internal: PyThermalMassNode::from(&mass.internal),
        }
    }
}

#[pymethods]
impl PyMultiNodeThermalMass {
    #[new]
    pub fn new(
        wall: PyThermalMassNode,
        roof: PyThermalMassNode,
        floor: PyThermalMassNode,
        internal: PyThermalMassNode,
    ) -> Self {
        Self {
            wall,
            roof,
            floor,
            internal,
        }
    }

    #[getter]
    pub fn wall(&self) -> PyThermalMassNode {
        self.wall.clone()
    }

    #[setter]
    pub fn set_wall(&mut self, value: PyThermalMassNode) {
        self.wall = value;
    }

    #[getter]
    pub fn roof(&self) -> PyThermalMassNode {
        self.roof.clone()
    }

    #[setter]
    pub fn set_roof(&mut self, value: PyThermalMassNode) {
        self.roof = value;
    }

    #[getter]
    pub fn floor(&self) -> PyThermalMassNode {
        self.floor.clone()
    }

    #[setter]
    pub fn set_floor(&mut self, value: PyThermalMassNode) {
        self.floor = value;
    }

    #[getter]
    pub fn internal(&self) -> PyThermalMassNode {
        self.internal.clone()
    }

    #[setter]
    pub fn set_internal(&mut self, value: PyThermalMassNode) {
        self.internal = value;
    }
}

#[pyclass(name = "MassAirCouplingMode", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyMassAirCouplingMode {
    AdditiveSum,
    ParallelResistance,
}

impl From<MassAirCouplingMode> for PyMassAirCouplingMode {
    fn from(mode: MassAirCouplingMode) -> Self {
        match mode {
            MassAirCouplingMode::AdditiveSum => PyMassAirCouplingMode::AdditiveSum,
            MassAirCouplingMode::ParallelResistance => PyMassAirCouplingMode::ParallelResistance,
        }
    }
}

impl From<PyMassAirCouplingMode> for MassAirCouplingMode {
    fn from(mode: PyMassAirCouplingMode) -> Self {
        match mode {
            PyMassAirCouplingMode::AdditiveSum => MassAirCouplingMode::AdditiveSum,
            PyMassAirCouplingMode::ParallelResistance => MassAirCouplingMode::ParallelResistance,
        }
    }
}

#[pymethods]
impl PyMassAirCouplingMode {
    fn __repr__(&self) -> &'static str {
        match self {
            PyMassAirCouplingMode::AdditiveSum => "MassAirCouplingMode.AdditiveSum",
            PyMassAirCouplingMode::ParallelResistance => "MassAirCouplingMode.ParallelResistance",
        }
    }
}

#[pyclass(name = "SurfaceExteriorTemperatures")]
#[derive(Clone, Debug)]
pub struct PySurfaceExteriorTemperatures {
    pub t_ext_wall: f64,
    pub t_ext_roof: f64,
    pub t_ext_floor: f64,
}

impl From<&SurfaceExteriorTemperatures> for PySurfaceExteriorTemperatures {
    fn from(temps: &SurfaceExteriorTemperatures) -> Self {
        Self {
            t_ext_wall: temps.t_ext_wall,
            t_ext_roof: temps.t_ext_roof,
            t_ext_floor: temps.t_ext_floor,
        }
    }
}

impl From<&PySurfaceExteriorTemperatures> for SurfaceExteriorTemperatures {
    fn from(temps: &PySurfaceExteriorTemperatures) -> Self {
        SurfaceExteriorTemperatures {
            t_ext_wall: temps.t_ext_wall,
            t_ext_roof: temps.t_ext_roof,
            t_ext_floor: temps.t_ext_floor,
        }
    }
}

#[pymethods]
impl PySurfaceExteriorTemperatures {
    #[new]
    pub fn new(t_ext_wall: f64, t_ext_roof: f64, t_ext_floor: f64) -> Self {
        Self {
            t_ext_wall,
            t_ext_roof,
            t_ext_floor,
        }
    }

    #[getter]
    pub fn t_ext_wall(&self) -> f64 {
        self.t_ext_wall
    }

    #[setter]
    pub fn set_t_ext_wall(&mut self, value: f64) {
        self.t_ext_wall = value;
    }

    #[getter]
    pub fn t_ext_roof(&self) -> f64 {
        self.t_ext_roof
    }

    #[setter]
    pub fn set_t_ext_roof(&mut self, value: f64) {
        self.t_ext_roof = value;
    }

    #[getter]
    pub fn t_ext_floor(&self) -> f64 {
        self.t_ext_floor
    }

    #[setter]
    pub fn set_t_ext_floor(&mut self, value: f64) {
        self.t_ext_floor = value;
    }
}

#[pyclass(name = "MultiNodeSolver")]
#[derive(Clone, Debug)]
pub struct PyMultiNodeSolver {
    pub mass: PyMultiNodeThermalMass,
    pub h_tr_is: f64,
    pub zone_temperature: f64,
    pub surface_temperature: f64,
    pub exterior_temperature: f64,
    pub exterior_temperatures: PySurfaceExteriorTemperatures,
    pub timestep_seconds: f64,
    pub coupling_mode: PyMassAirCouplingMode,
    pub r_total: f64,
    pub r_se: f64,
    pub initialized: bool,
    pub last_dt: f64,
}

impl From<&MultiNodeSolver> for PyMultiNodeSolver {
    fn from(solver: &MultiNodeSolver) -> Self {
        Self {
            mass: PyMultiNodeThermalMass::from(&solver.mass),
            h_tr_is: solver.h_tr_is,
            zone_temperature: solver.zone_temperature,
            surface_temperature: solver.surface_temperature,
            exterior_temperature: solver.exterior_temperature,
            exterior_temperatures: PySurfaceExteriorTemperatures::from(
                &solver.exterior_temperatures,
            ),
            timestep_seconds: solver.timestep_seconds,
            coupling_mode: PyMassAirCouplingMode::from(solver.coupling_mode),
            r_total: solver.r_total,
            r_se: solver.r_se,
            initialized: solver.initialized,
            last_dt: solver.last_dt,
        }
    }
}

#[pymethods]
impl PyMultiNodeSolver {
    #[new]
    pub fn new(
        h_tr_is: f64,
        wall: PyThermalMassNode,
        roof: PyThermalMassNode,
        floor: PyThermalMassNode,
        internal: PyThermalMassNode,
    ) -> Self {
        let wall_node = ThermalMassNode::new(
            wall.temperature,
            wall.capacitance,
            wall.h_tr_ms,
            wall.h_tr_em,
        )
        .with_h_tr_me(wall.h_tr_me);
        let roof_node = ThermalMassNode::new(
            roof.temperature,
            roof.capacitance,
            roof.h_tr_ms,
            roof.h_tr_em,
        )
        .with_h_tr_me(roof.h_tr_me);
        let floor_node = ThermalMassNode::new(
            floor.temperature,
            floor.capacitance,
            floor.h_tr_ms,
            floor.h_tr_em,
        )
        .with_h_tr_me(floor.h_tr_me);
        let internal_node = ThermalMassNode::new(
            internal.temperature,
            internal.capacitance,
            internal.h_tr_ms,
            internal.h_tr_em,
        )
        .with_h_tr_me(internal.h_tr_me);

        let solver = MultiNodeSolver::new(h_tr_is, wall_node, roof_node, floor_node, internal_node);

        Self::from(&solver)
    }

    #[getter]
    pub fn mass(&self) -> PyMultiNodeThermalMass {
        self.mass.clone()
    }

    #[setter]
    pub fn set_mass(&mut self, value: PyMultiNodeThermalMass) {
        self.mass = value;
    }

    #[getter]
    pub fn h_tr_is(&self) -> f64 {
        self.h_tr_is
    }

    #[setter]
    pub fn set_h_tr_is(&mut self, value: f64) {
        self.h_tr_is = value;
    }

    #[getter]
    pub fn zone_temperature(&self) -> f64 {
        self.zone_temperature
    }

    #[setter]
    pub fn set_zone_temperature(&mut self, value: f64) {
        self.zone_temperature = value;
    }

    #[getter]
    pub fn surface_temperature(&self) -> f64 {
        self.surface_temperature
    }

    #[setter]
    pub fn set_surface_temperature(&mut self, value: f64) {
        self.surface_temperature = value;
    }

    #[getter]
    pub fn exterior_temperature(&self) -> f64 {
        self.exterior_temperature
    }

    #[setter]
    pub fn set_exterior_temperature(&mut self, value: f64) {
        self.exterior_temperature = value;
    }

    #[getter]
    pub fn exterior_temperatures(&self) -> PySurfaceExteriorTemperatures {
        self.exterior_temperatures.clone()
    }

    #[setter]
    pub fn set_exterior_temperatures(&mut self, value: PySurfaceExteriorTemperatures) {
        self.exterior_temperatures = value;
    }

    #[getter]
    pub fn timestep_seconds(&self) -> f64 {
        self.timestep_seconds
    }

    #[setter]
    pub fn set_timestep_seconds(&mut self, value: f64) {
        self.timestep_seconds = value;
    }

    #[getter]
    pub fn coupling_mode(&self) -> PyMassAirCouplingMode {
        self.coupling_mode
    }

    #[setter]
    pub fn set_coupling_mode(&mut self, value: PyMassAirCouplingMode) {
        self.coupling_mode = value;
    }

    #[getter]
    pub fn r_total(&self) -> f64 {
        self.r_total
    }

    #[setter]
    pub fn set_r_total(&mut self, value: f64) {
        self.r_total = value;
    }

    #[getter]
    pub fn r_se(&self) -> f64 {
        self.r_se
    }

    #[setter]
    pub fn set_r_se(&mut self, value: f64) {
        self.r_se = value;
    }

    #[getter]
    pub fn initialized(&self) -> bool {
        self.initialized
    }

    #[setter]
    pub fn set_initialized(&mut self, value: bool) {
        self.initialized = value;
    }

    #[getter]
    pub fn last_dt(&self) -> f64 {
        self.last_dt
    }

    #[setter]
    pub fn set_last_dt(&mut self, value: f64) {
        self.last_dt = value;
    }

    pub fn wall_temperature(&self) -> f64 {
        self.mass.wall.temperature
    }

    pub fn roof_temperature(&self) -> f64 {
        self.mass.roof.temperature
    }

    pub fn floor_temperature(&self) -> f64 {
        self.mass.floor.temperature
    }

    pub fn internal_temperature(&self) -> f64 {
        self.mass.internal.temperature
    }

    pub fn set_wall_conductances(&mut self, h_tr_em: f64, h_tr_ms: f64) {
        self.mass.wall.h_tr_em = h_tr_em;
        self.mass.wall.h_tr_ms = h_tr_ms;
    }

    pub fn set_roof_conductances(&mut self, h_tr_em: f64, h_tr_ms: f64) {
        self.mass.roof.h_tr_em = h_tr_em;
        self.mass.roof.h_tr_ms = h_tr_ms;
    }

    pub fn set_floor_conductances(&mut self, h_tr_em: f64, h_tr_ms: f64) {
        self.mass.floor.h_tr_em = h_tr_em;
        self.mass.floor.h_tr_ms = h_tr_ms;
    }

    pub fn set_internal_conductance(&mut self, h_tr_me: f64) {
        self.mass.internal.h_tr_me = h_tr_me;
    }

    pub fn set_wall_capacitance(&mut self, cm: f64) {
        self.mass.wall.capacitance = cm;
    }

    pub fn set_roof_capacitance(&mut self, cm: f64) {
        self.mass.roof.capacitance = cm;
    }

    pub fn set_floor_capacitance(&mut self, cm: f64) {
        self.mass.floor.capacitance = cm;
    }

    pub fn set_internal_capacitance(&mut self, cm: f64) {
        self.mass.internal.capacitance = cm;
    }

    pub fn initialize_temperatures(&mut self, t_initial: f64) {
        self.mass.wall.temperature = t_initial;
        self.mass.roof.temperature = t_initial;
        self.mass.floor.temperature = t_initial;
        self.mass.internal.temperature = t_initial;
        self.zone_temperature = t_initial;
        self.surface_temperature = t_initial;
    }

    pub fn step(&mut self, dt: f64) {
        let mut solver = self.to_solver();
        solver.step(dt);
        self.sync_from_solver(&solver);
    }

    pub fn effective_time_constant(&self) -> f64 {
        let c_total = self.mass.wall.capacitance
            + self.mass.roof.capacitance
            + self.mass.floor.capacitance
            + self.mass.internal.capacitance;
        let h_tr_ms_total =
            self.mass.wall.h_tr_ms + self.mass.roof.h_tr_ms + self.mass.floor.h_tr_ms;
        let h_eff = self.h_tr_is + h_tr_ms_total / 3.0;
        c_total / h_eff
    }
}

impl PyMultiNodeSolver {
    fn to_solver(&self) -> MultiNodeSolver {
        let wall_node = ThermalMassNode::new(
            self.mass.wall.temperature,
            self.mass.wall.capacitance,
            self.mass.wall.h_tr_ms,
            self.mass.wall.h_tr_em,
        )
        .with_h_tr_me(self.mass.wall.h_tr_me);
        let roof_node = ThermalMassNode::new(
            self.mass.roof.temperature,
            self.mass.roof.capacitance,
            self.mass.roof.h_tr_ms,
            self.mass.roof.h_tr_em,
        )
        .with_h_tr_me(self.mass.roof.h_tr_me);
        let floor_node = ThermalMassNode::new(
            self.mass.floor.temperature,
            self.mass.floor.capacitance,
            self.mass.floor.h_tr_ms,
            self.mass.floor.h_tr_em,
        )
        .with_h_tr_me(self.mass.floor.h_tr_me);
        let internal_node = ThermalMassNode::new(
            self.mass.internal.temperature,
            self.mass.internal.capacitance,
            self.mass.internal.h_tr_ms,
            self.mass.internal.h_tr_em,
        )
        .with_h_tr_me(self.mass.internal.h_tr_me);

        let mut solver = MultiNodeSolver::new(
            self.h_tr_is,
            wall_node,
            roof_node,
            floor_node,
            internal_node,
        );
        solver.zone_temperature = self.zone_temperature;
        solver.surface_temperature = self.surface_temperature;
        solver.exterior_temperature = self.exterior_temperature;
        solver.exterior_temperatures =
            SurfaceExteriorTemperatures::from(&self.exterior_temperatures);
        solver.timestep_seconds = self.timestep_seconds;
        solver.coupling_mode = MassAirCouplingMode::from(self.coupling_mode);
        solver.r_total = self.r_total;
        solver.r_se = self.r_se;
        solver.initialized = self.initialized;
        solver.last_dt = self.last_dt;
        solver
    }

    fn sync_from_solver(&mut self, solver: &MultiNodeSolver) {
        self.mass = PyMultiNodeThermalMass::from(&solver.mass);
        self.h_tr_is = solver.h_tr_is;
        self.zone_temperature = solver.zone_temperature;
        self.surface_temperature = solver.surface_temperature;
        self.exterior_temperature = solver.exterior_temperature;
        self.exterior_temperatures =
            PySurfaceExteriorTemperatures::from(&solver.exterior_temperatures);
        self.timestep_seconds = solver.timestep_seconds;
        self.coupling_mode = PyMassAirCouplingMode::from(solver.coupling_mode);
        self.r_total = solver.r_total;
        self.r_se = solver.r_se;
        self.initialized = solver.initialized;
        self.last_dt = solver.last_dt;
    }
}

#[cfg(all(test, feature = "python-bindings"))]
mod tests {
    //! Rust-side inline tests for the PyO3 wrappers in this module (Issue #2532).
    //!
    //! Coverage focuses on the pure-Rust conversion / helper layer:
    //! - `From` round-trips for `ThermalMassNode`, `MultiNodeThermalMass`,
    //!   `SurfaceExteriorTemperatures`, and `MassAirCouplingMode`,
    //! - `PyMultiNodeSolver` field setters (conductances, capacitances,
    //!   temperatures) and aggregate helpers (`wall_temperature`,
    //!   `effective_time_constant`),
    //! - `PyMultiNodeSolver::new` end-to-end (constructs an inner solver and
    //!   snapshots it back into the Python wrapper),
    //! - `PyMultiNodeSolver::initialize_temperatures` mass-setter,
    //! - `step()` smoke test (executes one solver step without panic).

    use super::*;

    // ========================================================================
    // PyThermalMassNode round-trip + constructor
    // ========================================================================

    #[test]
    fn thermal_mass_node_constructor_defaults_cumulative_flux_to_zero() {
        let n = PyThermalMassNode::new(20.0, 1e6, 100.0, 50.0, 30.0);
        assert_eq!(n.temperature, 20.0);
        assert_eq!(n.capacitance, 1e6);
        assert_eq!(n.h_tr_ms, 100.0);
        assert_eq!(n.h_tr_em, 50.0);
        assert_eq!(n.h_tr_me, 30.0);
        assert_eq!(n.heat_flux_cumulative, 0.0);
    }

    #[test]
    fn thermal_mass_node_from_inner_copies_all_fields() {
        let inner = ThermalMassNode::new(22.0, 5e5, 80.0, 40.0).with_h_tr_me(25.0);
        let py = PyThermalMassNode::from(&inner);
        assert_eq!(py.temperature, 22.0);
        assert_eq!(py.capacitance, 5e5);
        assert_eq!(py.h_tr_ms, 80.0);
        assert_eq!(py.h_tr_em, 40.0);
        assert_eq!(py.h_tr_me, 25.0);
    }

    // ========================================================================
    // PyMultiNodeThermalMass round-trip
    // ========================================================================

    #[test]
    fn multi_node_thermal_mass_from_inner_copies_all_four_nodes() {
        let inner = MultiNodeThermalMass {
            wall: ThermalMassNode::new(1.0, 10.0, 100.0, 200.0).with_h_tr_me(50.0),
            roof: ThermalMassNode::new(2.0, 20.0, 110.0, 210.0).with_h_tr_me(51.0),
            floor: ThermalMassNode::new(3.0, 30.0, 120.0, 220.0).with_h_tr_me(52.0),
            internal: ThermalMassNode::new(4.0, 40.0, 130.0, 230.0).with_h_tr_me(53.0),
        };
        let py = PyMultiNodeThermalMass::from(&inner);
        assert_eq!(py.wall.temperature, 1.0);
        assert_eq!(py.roof.temperature, 2.0);
        assert_eq!(py.floor.temperature, 3.0);
        assert_eq!(py.internal.temperature, 4.0);
        assert_eq!(py.wall.h_tr_me, 50.0);
        assert_eq!(py.internal.capacitance, 40.0);
    }

    // ========================================================================
    // PySurfaceExteriorTemperatures round-trip
    // ========================================================================

    #[test]
    fn surface_exterior_temperatures_round_trip_preserves_fields() {
        let src = SurfaceExteriorTemperatures {
            t_ext_wall: 5.0,
            t_ext_roof: 10.0,
            t_ext_floor: -2.0,
        };
        let py = PySurfaceExteriorTemperatures::from(&src);
        assert_eq!(py.t_ext_wall, 5.0);
        assert_eq!(py.t_ext_roof, 10.0);
        assert_eq!(py.t_ext_floor, -2.0);

        let back: SurfaceExteriorTemperatures = SurfaceExteriorTemperatures::from(&py);
        assert_eq!(back.t_ext_wall, src.t_ext_wall);
        assert_eq!(back.t_ext_roof, src.t_ext_roof);
        assert_eq!(back.t_ext_floor, src.t_ext_floor);
    }

    // ========================================================================
    // PyMassAirCouplingMode round-trip
    // ========================================================================

    #[test]
    fn mass_air_coupling_mode_round_trip_preserves_both_variants() {
        for v in [
            MassAirCouplingMode::AdditiveSum,
            MassAirCouplingMode::ParallelResistance,
        ] {
            let py: PyMassAirCouplingMode = v.into();
            let back: MassAirCouplingMode = py.into();
            assert_eq!(back, v);
        }
    }

    // ========================================================================
    // PyMultiNodeSolver construction
    // ========================================================================

    fn sample_solver() -> PyMultiNodeSolver {
        // Realistic-ish 9R4C node values.
        let wall = PyThermalMassNode::new(20.0, 1.0e5, 100.0, 50.0, 30.0);
        let roof = PyThermalMassNode::new(20.0, 0.8e5, 90.0, 45.0, 25.0);
        let floor = PyThermalMassNode::new(20.0, 1.2e5, 110.0, 55.0, 35.0);
        let internal = PyThermalMassNode::new(20.0, 0.5e5, 80.0, 40.0, 20.0);
        PyMultiNodeSolver::new(8.0, wall, roof, floor, internal)
    }

    #[test]
    fn solver_constructor_initializes_with_provided_node_temperatures() {
        let s = sample_solver();
        assert!((s.wall_temperature() - 20.0).abs() < 1e-12);
        assert!((s.roof_temperature() - 20.0).abs() < 1e-12);
        assert!((s.floor_temperature() - 20.0).abs() < 1e-12);
        assert!((s.internal_temperature() - 20.0).abs() < 1e-12);
        // MultiNodeSolver::new defaults zone_temperature to 20.0 °C.
        assert!((s.zone_temperature() - 20.0).abs() < 1e-12);
    }

    #[test]
    fn solver_initialize_temperatures_sets_all_nodes_and_zone() {
        let mut s = sample_solver();
        s.initialize_temperatures(18.5);
        assert!((s.wall_temperature() - 18.5).abs() < 1e-12);
        assert!((s.roof_temperature() - 18.5).abs() < 1e-12);
        assert!((s.floor_temperature() - 18.5).abs() < 1e-12);
        assert!((s.internal_temperature() - 18.5).abs() < 1e-12);
        assert!((s.zone_temperature() - 18.5).abs() < 1e-12);
        assert!((s.surface_temperature() - 18.5).abs() < 1e-12);
    }

    #[test]
    fn solver_setters_update_individual_node_fields() {
        let mut s = sample_solver();

        s.set_wall_conductances(11.0, 22.0);
        assert!((s.mass.wall.h_tr_em - 11.0).abs() < 1e-12);
        assert!((s.mass.wall.h_tr_ms - 22.0).abs() < 1e-12);

        s.set_roof_conductances(13.0, 17.0);
        assert!((s.mass.roof.h_tr_em - 13.0).abs() < 1e-12);
        assert!((s.mass.roof.h_tr_ms - 17.0).abs() < 1e-12);

        s.set_floor_conductances(19.0, 23.0);
        assert!((s.mass.floor.h_tr_em - 19.0).abs() < 1e-12);
        assert!((s.mass.floor.h_tr_ms - 23.0).abs() < 1e-12);

        s.set_internal_conductance(42.0);
        assert!((s.mass.internal.h_tr_me - 42.0).abs() < 1e-12);

        s.set_wall_capacitance(1e6);
        assert!((s.mass.wall.capacitance - 1e6).abs() < 1e-6);
    }

    #[test]
    fn solver_effective_time_constant_matches_formula() {
        // τ = C_total / (h_tr_is + (h_tr_ms_wall + h_tr_ms_roof + h_tr_ms_floor) / 3)
        let s = sample_solver();
        let c_total = s.mass.wall.capacitance
            + s.mass.roof.capacitance
            + s.mass.floor.capacitance
            + s.mass.internal.capacitance;
        let h_tr_ms_total = s.mass.wall.h_tr_ms + s.mass.roof.h_tr_ms + s.mass.floor.h_tr_ms;
        let expected = c_total / (s.h_tr_is + h_tr_ms_total / 3.0);
        assert!((s.effective_time_constant() - expected).abs() < 1e-9);
    }

    #[test]
    fn solver_exterior_temperatures_round_trip_through_setter() {
        let mut s = sample_solver();
        let py = PySurfaceExteriorTemperatures::new(-3.0, 8.0, 12.0);
        s.set_exterior_temperatures(py);
        let got = s.exterior_temperatures();
        assert!((got.t_ext_wall - (-3.0)).abs() < 1e-12);
        assert!((got.t_ext_roof - 8.0).abs() < 1e-12);
        assert!((got.t_ext_floor - 12.0).abs() < 1e-12);
    }

    #[test]
    fn solver_coupling_mode_setter_round_trips() {
        let mut s = sample_solver();
        s.set_coupling_mode(PyMassAirCouplingMode::ParallelResistance);
        assert_eq!(s.coupling_mode(), PyMassAirCouplingMode::ParallelResistance);
        s.set_coupling_mode(PyMassAirCouplingMode::AdditiveSum);
        assert_eq!(s.coupling_mode(), PyMassAirCouplingMode::AdditiveSum);
    }

    #[test]
    fn solver_step_does_not_panic_for_short_dt() {
        // One short step. We're not asserting on the resulting temperatures
        // (the underlying MultiNodeSolver already has its own unit tests); the
        // point here is to exercise the `to_solver → step → sync_from_solver`
        // plumbing so a regression in either direction surfaces immediately.
        let mut s = sample_solver();
        s.initialize_temperatures(20.0);
        s.set_exterior_temperature(20.0);
        s.set_timestep_seconds(3600.0);
        s.step(3600.0);
        // After one step with equal interior/exterior, the zone temperature
        // should still be finite (no NaN / inf).
        assert!(s.zone_temperature().is_finite());
        assert!(s.wall_temperature().is_finite());
    }
}
