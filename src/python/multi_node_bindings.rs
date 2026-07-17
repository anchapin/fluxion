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
