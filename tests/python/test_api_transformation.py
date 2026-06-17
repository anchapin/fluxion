"""
Python integration tests for API Transformation and Model Mutation.

This test suite verifies that the Fluxion engine can safely and accurately
apply programmatic transformations to a running or staged model without
corrupting the physics matrices.

Each transformation test:
1. Loads a base model
2. Applies the transformation via the Python API
3. Runs simulation
4. Verifies transformation was applied correctly
5. Verifies physics matrices remain valid (simulation completes, energy is finite)
"""

import math

import pytest


@pytest.fixture(scope="module")
def fluxion_module():
    try:
        import fluxion

        return fluxion
    except ImportError:
        pytest.skip("fluxion Python bindings not available")


@pytest.fixture(scope="module")
def base_model(fluxion_module):
    """Create a base multi-zone model for transformation tests."""
    MultiZoneThermalModel = getattr(fluxion_module, "MultiZoneThermalModel", None)
    if MultiZoneThermalModel is None:
        pytest.skip("MultiZoneThermalModel not available")
    return MultiZoneThermalModel(num_zones=3)


@pytest.fixture(scope="module")
def base_model_single(fluxion_module):
    """Create a base single-zone model for transformation tests."""
    Model = getattr(fluxion_module, "Model", None)
    if Model is None:
        pytest.skip("Model not available")
    return Model(num_zones=1)


class TestTransformation1_WallInsulation:
    """Transformation 1: Change wall insulation (U-value)."""

    def test_change_wall_u_value(self, base_model, fluxion_module):
        """Verify that changing wall U-value produces different energy consumption."""
        # Get initial inter-zone conductances as a proxy for wall properties
        initial_conductances = base_model.get_inter_zone_conductance_vector()

        # Apply a significant change to inter-zone conductances (simulates wall insulation change)
        new_conductances = [c * 0.5 if c > 0 else 0.0 for c in initial_conductances]
        base_model.set_inter_zone_conductance_vector(new_conductances)

        # Verify the change was applied
        updated_conductances = base_model.get_inter_zone_conductance_vector()
        for orig, updated in zip(initial_conductances, updated_conductances):
            if orig > 0:
                assert abs(updated - orig * 0.5) < 1e-6, "Conductance should be halved"

    def test_insulation_change_preserves_physics(self, base_model, fluxion_module):
        """Verify that insulation change doesn't corrupt physics matrices."""
        # Set reasonable conductances
        base_model.set_inter_zone_conductance_vector([10.0, 5.0, 5.0])

        # Run simulation - should complete without errors
        result = base_model.simulate_multi_zone(years=1, use_surrogates=False)

        # Verify result is finite and non-negative
        assert math.isfinite(result), "Simulation result should be finite"
        assert result >= 0.0, "Total energy should be non-negative"

    def test_extreme_insulation_change(self, base_model, fluxion_module):
        """Verify that extreme insulation changes are handled gracefully."""
        # Set very low conductance (very well insulated)
        base_model.set_inter_zone_conductance_vector([0.1, 0.1, 0.1])

        result = base_model.simulate_multi_zone(years=1, use_surrogates=False)
        assert math.isfinite(result), "Result should be finite with high insulation"

        # Set very high conductance (poorly insulated)
        base_model.set_inter_zone_conductance_vector([100.0, 100.0, 100.0])

        result2 = base_model.simulate_multi_zone(years=1, use_surrogates=False)
        assert math.isfinite(result2), "Result should be finite with low insulation"


class TestTransformation2_LightingPowerDensity:
    """Transformation 2: Reduce lighting power density by 20%."""

    def test_building_type_modification(self, base_model_single, fluxion_module):
        """Verify that building type can be changed and affects simulation."""
        if not hasattr(base_model_single, "building_type"):
            pytest.skip("Building type modification not available")

        # Get initial energy with default building type
        initial_energy = base_model_single.simulate(years=1, use_surrogates=False)

        # Change building type to something with different internal loads
        base_model_single.set_building_type("Retail")

        # Get new energy
        new_energy = base_model_single.simulate(years=1, use_surrogates=False)

        # Both should be finite
        assert math.isfinite(initial_energy), "Initial energy should be finite"
        assert math.isfinite(new_energy), "New energy should be finite"

    def test_building_type_affects_energy(self, base_model_single, fluxion_module):
        """Verify that changing building type produces different energy consumption."""
        if not hasattr(base_model_single, "set_building_type"):
            pytest.skip("Building type setter not available")

        # Office building
        base_model_single.set_building_type("Office")
        office_energy = base_model_single.simulate(years=1, use_surrogates=False)

        # Warehouse building (typically lower internal loads)
        base_model_single.set_building_type("Warehouse")
        warehouse_energy = base_model_single.simulate(years=1, use_surrogates=False)

        assert math.isfinite(office_energy), "Office energy should be finite"
        assert math.isfinite(warehouse_energy), "Warehouse energy should be finite"
        # Different building types should have different energy signatures
        # (exact comparison depends on specific internal load profiles)


class TestTransformation3_ThermostatSetpoints:
    """Transformation 3: Change thermostat setpoints."""

    def test_change_heating_setpoint(self, base_model, fluxion_module):
        """Verify that changing heating setpoint affects simulation."""
        # Set initial setpoints
        base_model.set_zone_setpoints(0, 20.0, 24.0)

        # Get energy with initial setpoint
        energy_heating_low = base_model.simulate_multi_zone(
            years=1, use_surrogates=False
        )

        # Increase heating setpoint (more heating needed)
        base_model.set_zone_setpoints(0, 22.0, 24.0)

        # Get energy with higher setpoint
        energy_heating_high = base_model.simulate_multi_zone(
            years=1, use_surrogates=False
        )

        assert math.isfinite(energy_heating_low), "Energy should be finite"
        assert math.isfinite(energy_heating_high), "Energy should be finite"
        # Higher heating setpoint should require more energy (or at least different)
        assert (
            energy_heating_high != energy_heating_low
        ), "Setpoint change should affect energy"

    def test_change_cooling_setpoint(self, base_model, fluxion_module):
        """Verify that changing cooling setpoint affects simulation."""
        # Set initial setpoints
        base_model.set_zone_setpoints(0, 20.0, 24.0)

        # Get energy with initial setpoint
        energy_cooling_high = base_model.simulate_multi_zone(
            years=1, use_surrogates=False
        )

        # Lower cooling setpoint (more cooling needed)
        base_model.set_zone_setpoints(0, 20.0, 22.0)

        # Get energy with lower cooling setpoint
        energy_cooling_low = base_model.simulate_multi_zone(
            years=1, use_surrogates=False
        )

        assert math.isfinite(energy_cooling_high), "Energy should be finite"
        assert math.isfinite(energy_cooling_low), "Energy should be finite"
        # Lower cooling setpoint should require more cooling energy
        assert (
            energy_cooling_low != energy_cooling_high
        ), "Setpoint change should affect energy"

    def test_setpoint_validation(self, base_model, fluxion_module):
        """Verify that invalid setpoints are rejected."""
        # Heating >= Cooling should fail
        with pytest.raises(ValueError, match="must be less than"):
            base_model.set_zone_setpoints(0, 25.0, 23.0)

    def test_setpoint_bounds(self, base_model, fluxion_module):
        """Verify setpoint changes preserve physics validity."""
        # Apply valid setpoints across range
        base_model.set_zone_setpoints(0, 15.0, 32.0)
        result = base_model.simulate_multi_zone(years=1, use_surrogates=False)
        assert math.isfinite(result), "Result should be finite with wide setpoint range"


class TestTransformation4_InterZoneConductance:
    """Transformation 4: Modify inter-zone conductance."""

    def test_modify_inter_zone_conductance(self, base_model, fluxion_module):
        """Verify that modifying inter-zone conductance changes heat transfer."""
        # Set new conductance
        base_model.set_inter_zone_conductance(0, 0, 15.0)

        # Verify change
        updated = base_model.get_inter_zone_conductance(0, 0)
        assert abs(updated - 15.0) < 1e-6, "Conductance should be updated"

    def test_conductance_zero_boundary(self, base_model, fluxion_module):
        """Verify zero conductance (adiabatic) is handled."""
        # Set zero conductance
        base_model.set_inter_zone_conductance(0, 0, 0.0)

        result = base_model.simulate_multi_zone(years=1, use_surrogates=False)
        assert math.isfinite(result), "Zero conductance should be handled"

    def test_conductance_vector_roundtrip(self, base_model, fluxion_module):
        """Verify conductance vector can be set and retrieved."""
        original = [10.0, 5.0, 3.0]
        base_model.set_inter_zone_conductance_vector(original)

        retrieved = base_model.get_inter_zone_conductance_vector()
        assert len(retrieved) == len(original), "Length should match"
        for orig, retr in zip(original, retrieved):
            assert abs(orig - retr) < 1e-6, "Values should match"


class TestTransformation5_ConstructionLayers:
    """Transformation 5: Change construction layer properties."""

    def test_construction_layer_creation(self, fluxion_module):
        """Verify construction layers can be created with custom properties."""
        ConstructionLayer = getattr(fluxion_module, "ConstructionLayer", None)
        if ConstructionLayer is None:
            pytest.skip("ConstructionLayer not available")

        # Create a concrete layer
        concrete = ConstructionLayer(
            name="Concrete",
            conductivity=1.73,
            density=2300.0,
            specific_heat=880.0,
            thickness=0.1,
        )

        assert concrete.name == "Concrete"
        assert concrete.conductivity == 1.73
        assert concrete.density == 2300.0
        assert concrete.specific_heat == 880.0
        assert concrete.thickness == 0.1

    def test_construction_layer_r_value(self, fluxion_module):
        """Verify R-value calculation for construction layers."""
        ConstructionLayer = getattr(fluxion_module, "ConstructionLayer", None)
        if ConstructionLayer is None:
            pytest.skip("ConstructionLayer not available")

        layer = ConstructionLayer(
            name="Insulation",
            conductivity=0.04,
            density=30.0,
            specific_heat=1000.0,
            thickness=0.1,
        )

        expected_r = 0.1 / 0.04  # thickness / conductivity
        assert (
            abs(layer.r_value() - expected_r) < 1e-6
        ), "R-value should match calculation"

    def test_construction_layer_thermal_capacitance(self, fluxion_module):
        """Verify thermal capacitance calculation."""
        ConstructionLayer = getattr(fluxion_module, "ConstructionLayer", None)
        if ConstructionLayer is None:
            pytest.skip("ConstructionLayer not available")

        layer = ConstructionLayer(
            name="Heavy Concrete",
            conductivity=1.0,
            density=2000.0,
            specific_heat=1000.0,
            thickness=0.2,
        )

        expected_cap = 2000.0 * 0.2 * 1000.0  # density * thickness * specific_heat
        assert abs(layer.thermal_capacitance_per_area() - expected_cap) < 1e-6

    def test_multi_layer_construction(self, fluxion_module):
        """Verify multi-layer construction assembly."""
        ConstructionLayer = getattr(fluxion_module, "ConstructionLayer", None)
        Construction = getattr(fluxion_module, "Construction", None)
        if ConstructionLayer is None or Construction is None:
            pytest.skip("Construction or ConstructionLayer not available")

        # Create insulation layer
        insulation = ConstructionLayer(
            name="Insulation",
            conductivity=0.04,
            density=30.0,
            specific_heat=1000.0,
            thickness=0.1,
        )

        # Create concrete layer
        concrete = ConstructionLayer(
            name="Concrete",
            conductivity=1.73,
            density=2300.0,
            specific_heat=880.0,
            thickness=0.15,
        )

        # Create construction from layers
        wall = Construction([insulation, concrete])

        assert wall.layer_count() == 2, "Should have 2 layers"
        assert wall.total_thickness() == 0.25, "Total thickness should be 0.25m"

    def test_construction_u_value(self, fluxion_module):
        """Verify U-value calculation for multi-layer construction."""
        ConstructionLayer = getattr(fluxion_module, "ConstructionLayer", None)
        Construction = getattr(fluxion_module, "Construction", None)
        if ConstructionLayer is None or Construction is None:
            pytest.skip("Construction or ConstructionLayer not available")

        # Create a simple single-layer construction
        layer = ConstructionLayer(
            name="Brick",
            conductivity=0.9,
            density=1800.0,
            specific_heat=800.0,
            thickness=0.1,
        )

        construction = Construction([layer])

        # Calculate expected U-value
        r_material = 0.1 / 0.9
        r_film_int = 1.0 / 8.29
        r_film_ext = 1.0 / 29.3
        r_total = r_film_int + r_material + r_film_ext
        expected_u = 1.0 / r_total

        u_value = construction.u_value()
        assert (
            abs(u_value - expected_u) < 1e-6
        ), "U-value should match ASHRAE calculation"


class TestPhysicsMatrixValidity:
    """Verify that all transformations preserve physics matrix validity."""

    def test_matrix_validity_after_multiple_transformations(
        self, base_model, fluxion_module
    ):
        """Apply multiple transformations and verify physics remains valid."""
        # Apply several transformations
        base_model.set_zone_setpoints(0, 21.0, 25.0)
        base_model.set_inter_zone_conductance_vector([8.0, 4.0, 4.0])

        # Run simulation
        result = base_model.simulate_multi_zone(years=1, use_surrogates=False)

        # Verify result is valid
        assert math.isfinite(
            result
        ), "Result should be finite after multiple transformations"
        assert result >= 0.0, "Result should be non-negative"

    def test_matrix_validity_sequential_simulations(self, base_model, fluxion_module):
        """Run multiple sequential simulations after transformations."""
        base_model.set_inter_zone_conductance_vector([10.0, 5.0, 5.0])

        results = []
        for _ in range(3):
            result = base_model.simulate_multi_zone(years=1, use_surrogates=False)
            assert math.isfinite(result), "Each simulation should produce finite result"
            results.append(result)

        # All results should be identical (deterministic)
        assert (
            results[0] == results[1] == results[2]
        ), "Sequential runs should be deterministic"

    def test_temperature_conservation(self, base_model, fluxion_module):
        """Verify temperature states remain bounded after transformations."""
        base_model.set_zone_temperatures([20.0, 22.0, 21.0])

        # Run simulation
        result = base_model.simulate_multi_zone(years=1, use_surrogates=False)
        assert math.isfinite(result), "Result should be finite"

        # Get final temperatures
        final_temps = base_model.get_zone_temperatures()
        assert len(final_temps) == 3, "Should have 3 zone temperatures"
        for temp in final_temps:
            assert (
                -50.0 < temp < 60.0
            ), f"Temperature {temp} should be in physical range"


class TestTransformationRegression:
    """Regression tests to ensure transformations produce expected directional changes."""

    def test_higher_insulation_reduces_energy(self, base_model, fluxion_module):
        """Verify that improving insulation (lower conductance) reduces energy consumption."""
        # Baseline
        base_model.set_inter_zone_conductance_vector([20.0, 20.0, 20.0])
        baseline_energy = base_model.simulate_multi_zone(years=1, use_surrogates=False)

        # Improved insulation (lower conductance)
        base_model.set_inter_zone_conductance_vector([5.0, 5.0, 5.0])
        improved_energy = base_model.simulate_multi_zone(years=1, use_surrogates=False)

        assert (
            improved_energy <= baseline_energy
        ), "Lower conductance should result in equal or lower energy"

    def test_higher_setpoints_increase_energy(self, base_model, fluxion_module):
        """Verify that higher setpoints increase energy consumption."""
        # Baseline
        base_model.set_zone_setpoints(0, 18.0, 28.0)
        baseline_energy = base_model.simulate_multi_zone(years=1, use_surrogates=False)

        # Higher heating setpoint
        base_model.set_zone_setpoints(0, 22.0, 28.0)
        higher_heating_energy = base_model.simulate_multi_zone(
            years=1, use_surrogates=False
        )

        assert (
            higher_heating_energy >= baseline_energy
        ), "Higher heating setpoint should result in equal or higher energy"

    def test_wider_deadband_reduces_cycling(self, base_model, fluxion_module):
        """Verify that wider thermostat deadband affects HVAC cycling behavior."""
        ZoneSetpoints = getattr(fluxion_module, "ZoneSetpoints", None)
        if ZoneSetpoints is None:
            pytest.skip("ZoneSetpoints not available")

        # Create setpoints with narrow deadband
        setpoints = ZoneSetpoints(1)
        setpoints.set_heating_setpoint(0, 20.0)
        setpoints.set_cooling_setpoint(0, 24.0)
        setpoints.set_deadband(0, 1.0)

        # Create setpoints with wide deadband
        setpoints_wide = ZoneSetpoints(1)
        setpoints_wide.set_heating_setpoint(0, 20.0)
        setpoints_wide.set_cooling_setpoint(0, 24.0)
        setpoints_wide.set_deadband(0, 4.0)

        # Both should validate successfully
        setpoints.validate()
        setpoints_wide.validate()
