"""
Python smoke tests for 9R4C Multi-Node Thermal Solver bindings (Issue #1795)
"""

import fluxion
import pytest

ThermalMassNode = getattr(fluxion, "ThermalMassNode", None)
MultiNodeThermalMass = getattr(fluxion, "MultiNodeThermalMass", None)
MassAirCouplingMode = getattr(fluxion, "MassAirCouplingMode", None)
SurfaceExteriorTemperatures = getattr(fluxion, "SurfaceExteriorTemperatures", None)
MultiNodeSolver = getattr(fluxion, "MultiNodeSolver", None)

if ThermalMassNode is None:
    pytest.skip("fluxion 9R4C bindings not available", allow_module_level=True)


def test_thermal_mass_node_creation():
    """Test ThermalMassNode creation and property access"""
    node = ThermalMassNode(
        temperature=20.0,
        capacitance=5e6,
        h_tr_ms=50.0,
        h_tr_em=20.0,
        h_tr_me=10.0,
    )
    assert node.temperature == 20.0
    assert node.capacitance == 5e6
    assert node.h_tr_ms == 50.0
    assert node.h_tr_em == 20.0
    assert node.h_tr_me == 10.0


def test_thermal_mass_node_setters():
    """Test ThermalMassNode property setters"""
    node = ThermalMassNode(20.0, 5e6, 50.0, 20.0, 10.0)
    node.temperature = 25.0
    node.capacitance = 6e6
    node.h_tr_ms = 55.0
    node.h_tr_em = 22.0
    node.h_tr_me = 12.0
    assert node.temperature == 25.0
    assert node.capacitance == 6e6
    assert node.h_tr_ms == 55.0
    assert node.h_tr_em == 22.0
    assert node.h_tr_me == 12.0


def test_mass_air_coupling_mode():
    """Test MassAirCouplingMode enum values"""
    assert MassAirCouplingMode.AdditiveSum is not None
    assert MassAirCouplingMode.ParallelResistance is not None
    assert MassAirCouplingMode.AdditiveSum != MassAirCouplingMode.ParallelResistance


def test_mass_air_coupling_mode_repr():
    """Test MassAirCouplingMode string representation"""
    assert repr(MassAirCouplingMode.AdditiveSum) == "MassAirCouplingMode.AdditiveSum"
    assert repr(MassAirCouplingMode.ParallelResistance) == "MassAirCouplingMode.ParallelResistance"


def test_surface_exterior_temps_creation():
    """Test SurfaceExteriorTemperatures creation"""
    temps = SurfaceExteriorTemperatures(
        t_ext_wall=30.0,
        t_ext_roof=35.0,
        t_ext_floor=15.0,
    )
    assert temps.t_ext_wall == 30.0
    assert temps.t_ext_roof == 35.0
    assert temps.t_ext_floor == 15.0


def test_surface_exterior_temps_setters():
    """Test SurfaceExteriorTemperatures property setters"""
    temps = SurfaceExteriorTemperatures(30.0, 35.0, 15.0)
    temps.t_ext_wall = 31.0
    temps.t_ext_roof = 36.0
    temps.t_ext_floor = 16.0
    assert temps.t_ext_wall == 31.0
    assert temps.t_ext_roof == 36.0
    assert temps.t_ext_floor == 16.0


def test_multi_node_thermal_mass_creation():
    """Test MultiNodeThermalMass creation from ThermalMassNodes"""
    wall = ThermalMassNode(20.0, 5e6, 50.0, 20.0, 0.0)
    roof = ThermalMassNode(20.0, 3e6, 30.0, 15.0, 0.0)
    floor = ThermalMassNode(20.0, 2e6, 20.0, 10.0, 0.0)
    internal = ThermalMassNode(20.0, 1e6, 0.0, 0.0, 100.0)

    mass = MultiNodeThermalMass(
        wall=wall,
        roof=roof,
        floor=floor,
        internal=internal,
    )
    assert mass.wall.temperature == 20.0
    assert mass.roof.temperature == 20.0
    assert mass.floor.temperature == 20.0
    assert mass.internal.temperature == 20.0


def test_multi_node_solver_creation():
    """Test MultiNodeSolver creation with all parameters"""
    wall = ThermalMassNode(20.0, 5e6, 50.0, 20.0, 0.0)
    roof = ThermalMassNode(20.0, 3e6, 30.0, 15.0, 0.0)
    floor = ThermalMassNode(20.0, 2e6, 20.0, 10.0, 0.0)
    internal = ThermalMassNode(20.0, 1e6, 0.0, 0.0, 100.0)

    solver = MultiNodeSolver(
        h_tr_is=10.0,
        wall=wall,
        roof=roof,
        floor=floor,
        internal=internal,
    )
    assert solver.h_tr_is == 10.0
    assert solver.zone_temperature == 20.0
    assert solver.exterior_temperature == 10.0
    assert solver.surface_temperature == 20.0
    assert solver.timestep_seconds == 3600.0
    assert solver.initialized == False
    assert solver.r_total == 0.0
    assert solver.r_se == 0.0


def test_multi_node_solver_config_roundtrip():
    """Test that all 9R4C config parameters can be set and retrieved"""
    wall = ThermalMassNode(20.0, 5e6, 50.0, 20.0, 0.0)
    roof = ThermalMassNode(20.0, 3e6, 30.0, 15.0, 0.0)
    floor = ThermalMassNode(20.0, 2e6, 20.0, 10.0, 0.0)
    internal = ThermalMassNode(20.0, 1e6, 0.0, 0.0, 100.0)

    solver = MultiNodeSolver(
        h_tr_is=10.0,
        wall=wall,
        roof=roof,
        floor=floor,
        internal=internal,
    )

    solver.h_tr_is = 12.0
    solver.zone_temperature = 22.0
    solver.surface_temperature = 18.0
    solver.exterior_temperature = 5.0
    solver.timestep_seconds = 1800.0
    solver.coupling_mode = MassAirCouplingMode.ParallelResistance
    solver.r_total = 0.5
    solver.r_se = 0.04
    solver.initialized = True
    solver.last_dt = 1800.0

    assert solver.h_tr_is == 12.0
    assert solver.zone_temperature == 22.0
    assert solver.surface_temperature == 18.0
    assert solver.exterior_temperature == 5.0
    assert solver.timestep_seconds == 1800.0
    assert solver.coupling_mode == MassAirCouplingMode.ParallelResistance
    assert solver.r_total == 0.5
    assert solver.r_se == 0.04
    assert solver.initialized == True
    assert solver.last_dt == 1800.0


def test_multi_node_solver_exterior_temps():
    """Test setting per-surface exterior temperatures"""
    wall = ThermalMassNode(20.0, 5e6, 50.0, 20.0, 0.0)
    roof = ThermalMassNode(20.0, 3e6, 30.0, 15.0, 0.0)
    floor = ThermalMassNode(20.0, 2e6, 20.0, 10.0, 0.0)
    internal = ThermalMassNode(20.0, 1e6, 0.0, 0.0, 100.0)

    solver = MultiNodeSolver(
        h_tr_is=10.0,
        wall=wall,
        roof=roof,
        floor=floor,
        internal=internal,
    )

    ext_temps = SurfaceExteriorTemperatures(
        t_ext_wall=30.0,
        t_ext_roof=35.0,
        t_ext_floor=15.0,
    )
    solver.exterior_temperatures = ext_temps
    assert solver.exterior_temperatures.t_ext_wall == 30.0
    assert solver.exterior_temperatures.t_ext_roof == 35.0
    assert solver.exterior_temperatures.t_ext_floor == 15.0


def test_multi_node_solver_node_conductance_setters():
    """Test setting conductances on individual nodes"""
    wall = ThermalMassNode(20.0, 5e6, 50.0, 20.0, 0.0)
    roof = ThermalMassNode(20.0, 3e6, 30.0, 15.0, 0.0)
    floor = ThermalMassNode(20.0, 2e6, 20.0, 10.0, 0.0)
    internal = ThermalMassNode(20.0, 1e6, 0.0, 0.0, 100.0)

    solver = MultiNodeSolver(
        h_tr_is=10.0,
        wall=wall,
        roof=roof,
        floor=floor,
        internal=internal,
    )

    solver.set_wall_conductances(h_tr_em=25.0, h_tr_ms=55.0)
    solver.set_roof_conductances(h_tr_em=20.0, h_tr_ms=40.0)
    solver.set_floor_conductances(h_tr_em=15.0, h_tr_ms=30.0)
    solver.set_internal_conductance(h_tr_me=80.0)

    assert solver.mass.wall.h_tr_em == 25.0
    assert solver.mass.wall.h_tr_ms == 55.0
    assert solver.mass.roof.h_tr_em == 20.0
    assert solver.mass.roof.h_tr_ms == 40.0
    assert solver.mass.floor.h_tr_em == 15.0
    assert solver.mass.floor.h_tr_ms == 30.0
    assert solver.mass.internal.h_tr_me == 80.0


def test_multi_node_solver_node_capacitance_setters():
    """Test setting capacitance on individual nodes"""
    wall = ThermalMassNode(20.0, 5e6, 50.0, 20.0, 0.0)
    roof = ThermalMassNode(20.0, 3e6, 30.0, 15.0, 0.0)
    floor = ThermalMassNode(20.0, 2e6, 20.0, 10.0, 0.0)
    internal = ThermalMassNode(20.0, 1e6, 0.0, 0.0, 100.0)

    solver = MultiNodeSolver(
        h_tr_is=10.0,
        wall=wall,
        roof=roof,
        floor=floor,
        internal=internal,
    )

    solver.set_wall_capacitance(1e7)
    solver.set_roof_capacitance(2e7)
    solver.set_floor_capacitance(3e7)
    solver.set_internal_capacitance(4e6)

    assert solver.mass.wall.capacitance == 1e7
    assert solver.mass.roof.capacitance == 2e7
    assert solver.mass.floor.capacitance == 3e7
    assert solver.mass.internal.capacitance == 4e6


def test_multi_node_solver_initialize_temperatures():
    """Test initializing all temperatures at once"""
    wall = ThermalMassNode(20.0, 5e6, 50.0, 20.0, 0.0)
    roof = ThermalMassNode(20.0, 3e6, 30.0, 15.0, 0.0)
    floor = ThermalMassNode(20.0, 2e6, 20.0, 10.0, 0.0)
    internal = ThermalMassNode(20.0, 1e6, 0.0, 0.0, 100.0)

    solver = MultiNodeSolver(
        h_tr_is=10.0,
        wall=wall,
        roof=roof,
        floor=floor,
        internal=internal,
    )

    solver.initialize_temperatures(15.0)
    assert solver.mass.wall.temperature == 15.0
    assert solver.mass.roof.temperature == 15.0
    assert solver.mass.floor.temperature == 15.0
    assert solver.mass.internal.temperature == 15.0
    assert solver.zone_temperature == 15.0
    assert solver.surface_temperature == 15.0


def test_multi_node_solver_temperature_accessors():
    """Test temperature accessors for individual nodes"""
    wall = ThermalMassNode(25.0, 5e6, 50.0, 20.0, 0.0)
    roof = ThermalMassNode(22.0, 3e6, 30.0, 15.0, 0.0)
    floor = ThermalMassNode(18.0, 2e6, 20.0, 10.0, 0.0)
    internal = ThermalMassNode(20.0, 1e6, 0.0, 0.0, 100.0)

    solver = MultiNodeSolver(
        h_tr_is=10.0,
        wall=wall,
        roof=roof,
        floor=floor,
        internal=internal,
    )

    assert solver.wall_temperature() == 25.0
    assert solver.roof_temperature() == 22.0
    assert solver.floor_temperature() == 18.0
    assert solver.internal_temperature() == 20.0


def test_multi_node_solver_step():
    """Test that step() advances the solver"""
    wall = ThermalMassNode(20.0, 5e6, 50.0, 20.0, 0.0)
    roof = ThermalMassNode(20.0, 3e6, 30.0, 15.0, 0.0)
    floor = ThermalMassNode(20.0, 2e6, 20.0, 10.0, 0.0)
    internal = ThermalMassNode(20.0, 1e6, 0.0, 0.0, 100.0)

    solver = MultiNodeSolver(
        h_tr_is=10.0,
        wall=wall,
        roof=roof,
        floor=floor,
        internal=internal,
    )
    solver.zone_temperature = 22.0
    solver.exterior_temperature = 5.0
    solver.surface_temperature = 18.0

    t_wall_before = solver.wall_temperature()
    solver.step(dt=3600.0)
    assert solver.wall_temperature() < t_wall_before


def test_multi_node_solver_effective_time_constant():
    """Test effective time constant calculation"""
    wall = ThermalMassNode(20.0, 5e6, 50.0, 20.0, 0.0)
    roof = ThermalMassNode(20.0, 3e6, 30.0, 15.0, 0.0)
    floor = ThermalMassNode(20.0, 2e6, 20.0, 10.0, 0.0)
    internal = ThermalMassNode(20.0, 1e6, 0.0, 0.0, 100.0)

    solver = MultiNodeSolver(
        h_tr_is=10.0,
        wall=wall,
        roof=roof,
        floor=floor,
        internal=internal,
    )

    tau = solver.effective_time_constant()
    assert tau > 0.0
    assert tau < 1e8
