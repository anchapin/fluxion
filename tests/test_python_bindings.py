import pytest
import numpy as np
from typing import List


@pytest.fixture(scope="module")
def fluxion_module():
    try:
        import fluxion
        return fluxion
    except ImportError:
        pytest.skip("fluxion Python bindings not available")


@pytest.fixture(scope="module")
def model(fluxion_module):
    return fluxion_module.Model(num_zones=1)


@pytest.fixture(scope="module")
def batch_oracle(fluxion_module):
    return fluxion_module.BatchOracle()


class TestImportModule:
    def test_module_imports(self, fluxion_module):
        assert fluxion_module is not None
        assert hasattr(fluxion_module, 'Model')
        assert hasattr(fluxion_module, 'BatchOracle')
        assert hasattr(fluxion_module, 'VectorField')
        assert hasattr(fluxion_module, 'Construction')
        assert hasattr(fluxion_module, 'ConstructionLayer')
        assert hasattr(fluxion_module, 'WallSurface')
        assert hasattr(fluxion_module, 'MassClass')
        assert hasattr(fluxion_module, 'SurfaceType')


class TestVectorField:
    def test_create_from_list(self, fluxion_module):
        vf = fluxion_module.VectorField([1.0, 2.0, 3.0, 4.0, 5.0])
        assert vf.len() == 5
        
    def test_create_from_scalar(self, fluxion_module):
        vf = fluxion_module.VectorField.from_scalar(2.5, 10)
        assert vf.len() == 10
        data = vf.to_list()
        assert all(x == 2.5 for x in data)
        
    def test_to_list(self, fluxion_module):
        vf = fluxion_module.VectorField([1.0, 2.0, 3.0])
        data = vf.to_list()
        assert isinstance(data, list)
        assert len(data) == 3
        assert data == [1.0, 2.0, 3.0]
        
    def test_integrate(self, fluxion_module):
        vf = fluxion_module.VectorField([1.0, 2.0, 3.0, 4.0])
        result = vf.integrate()
        assert result == 10.0
        
    def test_gradient(self, fluxion_module):
        vf = fluxion_module.VectorField([1.0, 3.0, 6.0, 10.0])
        grad = vf.gradient()
        grad_data = grad.to_list()
        assert len(grad_data) == 4
        assert isinstance(grad_data, list)


class TestConstructionLayer:
    def test_create_layer(self, fluxion_module):
        layer = fluxion_module.ConstructionLayer(
            name="Concrete",
            conductivity=1.73,
            density=2300.0,
            specific_heat=880.0,
            thickness=0.1,
            emissivity=0.9,
            absorptance=0.7
        )
        assert layer.name == "Concrete"
        assert layer.conductivity == 1.73
        assert layer.density == 2300.0
        assert layer.specific_heat == 880.0
        assert layer.thickness == 0.1
        assert layer.emissivity == 0.9
        assert layer.absorptance == 0.7
        
    def test_layer_defaults(self, fluxion_module):
        layer = fluxion_module.ConstructionLayer(
            name="Default",
            conductivity=1.0,
            density=2000.0,
            specific_heat=1000.0,
            thickness=0.1
        )
        assert layer.emissivity == 0.9
        assert layer.absorptance == 0.7
        
    def test_r_value(self, fluxion_module):
        layer = fluxion_module.ConstructionLayer(
            name="Test",
            conductivity=2.0,
            density=2000.0,
            specific_heat=1000.0,
            thickness=0.2
        )
        r_value = layer.r_value()
        assert abs(r_value - 0.1) < 1e-10
        
    def test_thermal_capacitance_per_area(self, fluxion_module):
        layer = fluxion_module.ConstructionLayer(
            name="Test",
            conductivity=2.0,
            density=2000.0,
            specific_heat=1000.0,
            thickness=0.1
        )
        capacitance = layer.thermal_capacitance_per_area()
        expected = 2000.0 * 0.1 * 1000.0
        assert abs(capacitance - expected) < 1e-6


class TestConstruction:
    def test_create_construction(self, fluxion_module):
        layer1 = fluxion_module.ConstructionLayer(
            name="Layer1",
            conductivity=1.0,
            density=2000.0,
            specific_heat=1000.0,
            thickness=0.1
        )
        layer2 = fluxion_module.ConstructionLayer(
            name="Layer2",
            conductivity=0.5,
            density=1500.0,
            specific_heat=900.0,
            thickness=0.05
        )
        construction = fluxion_module.Construction([layer1, layer2])
        assert construction.layer_count() == 2
        
    def test_r_value_total(self, fluxion_module):
        layer1 = fluxion_module.ConstructionLayer(
            name="Layer1",
            conductivity=1.0,
            density=2000.0,
            specific_heat=1000.0,
            thickness=0.1
        )
        layer2 = fluxion_module.ConstructionLayer(
            name="Layer2",
            conductivity=0.5,
            density=1500.0,
            specific_heat=900.0,
            thickness=0.05
        )
        construction = fluxion_module.Construction([layer1, layer2])
        r_total = construction.r_value_total()
        
        r_layer1 = 0.1 / 1.0
        r_layer2 = 0.05 / 0.5
        r_layers = r_layer1 + r_layer2
        assert r_total > r_layers
        
    def test_u_value(self, fluxion_module):
        layer1 = fluxion_module.ConstructionLayer(
            name="Layer1",
            conductivity=1.0,
            density=2000.0,
            specific_heat=1000.0,
            thickness=0.1
        )
        construction = fluxion_module.Construction([layer1])
        u_value = construction.u_value()
        
        r_value = 0.1 / 1.0
        expected_no_film = 1.0 / r_value
        assert u_value < expected_no_film
        
    def test_total_thickness(self, fluxion_module):
        layer1 = fluxion_module.ConstructionLayer(
            name="Layer1",
            conductivity=1.0,
            density=2000.0,
            specific_heat=1000.0,
            thickness=0.1
        )
        layer2 = fluxion_module.ConstructionLayer(
            name="Layer2",
            conductivity=0.5,
            density=1500.0,
            specific_heat=900.0,
            thickness=0.05
        )
        construction = fluxion_module.Construction([layer1, layer2])
        thickness = construction.total_thickness()
        assert abs(thickness - 0.15) < 1e-10
        
    def test_mass_class(self, fluxion_module):
        layer = fluxion_module.ConstructionLayer(
            name="Heavy",
            conductivity=1.73,
            density=2300.0,
            specific_heat=880.0,
            thickness=0.2
        )
        construction = fluxion_module.Construction([layer])
        mass_class = construction.mass_class()
        assert hasattr(fluxion_module, 'MassClass')
        
    def test_thermal_capacitance_per_area(self, fluxion_module):
        layer = fluxion_module.ConstructionLayer(
            name="Test",
            conductivity=1.0,
            density=2000.0,
            specific_heat=1000.0,
            thickness=0.1
        )
        construction = fluxion_module.Construction([layer])
        capacitance = construction.thermal_capacitance_per_area()
        expected = 2000.0 * 1000.0 * 0.1
        assert abs(capacitance - expected) < 1e-6


class TestWallSurface:
    def test_create_wall_surface(self, fluxion_module):
        wall = fluxion_module.WallSurface(
            area=50.0,
            u_value=2.5,
            orientation="south"
        )
        assert wall.area == 50.0
        assert wall.u_value == 2.5
        assert hasattr(wall, 'orientation')
        
    def test_invalid_orientation(self, fluxion_module):
        with pytest.raises(Exception):
            fluxion_module.WallSurface(
                area=50.0,
                u_value=2.5,
                orientation="invalid"
            )
            
    def test_all_orientations(self, fluxion_module):
        orientations = ["south", "north", "east", "west"]
        for orientation in orientations:
            wall = fluxion_module.WallSurface(
                area=50.0,
                u_value=2.5,
                orientation=orientation
            )
            assert wall.area == 50.0


class TestModelCreation:
    def test_create_model_single_zone(self, model):
        assert model.num_zones() == 1
        
    def test_create_model_multi_zone(self, fluxion_module):
        model = fluxion_module.Model(num_zones=5)
        assert model.num_zones() == 5
        
    def test_get_temperatures(self, model):
        temps = model.get_temperatures()
        assert isinstance(temps, list)
        assert len(temps) == model.num_zones()
        assert all(isinstance(t, float) for t in temps)
        
    def test_set_temperatures(self, fluxion_module):
        temps = [20.0, 21.0, 22.0]
        model_multi = fluxion_module.Model(num_zones=3)
        model_multi.set_temperatures(temps)
        retrieved = model_multi.get_temperatures()
        assert retrieved == temps
        
    def test_set_temperatures_invalid_length(self, model):
        with pytest.raises(Exception):
            model.set_temperatures([20.0, 21.0])
            
    def test_ground_temperature(self, fluxion_module):
        model = fluxion_module.Model(num_zones=1)
        model.set_ground_temp(15.0)
        temp_at_0 = model.ground_temperature_at(0)
        assert temp_at_0 == 15.0


class TestSimulation:
    def test_simulate_one_year_analytical(self, model):
        result = model.simulate(years=1, use_surrogates=False)
        assert isinstance(result, float)
        assert result >= 0.0
        
    def test_simulate_one_year_surrogates(self, model):
        result = model.simulate(years=1, use_surrogates=True)
        assert isinstance(result, float)
        assert result >= 0.0
        
    def test_simulate_multiple_years(self, model):
        result_1yr = model.simulate(years=1, use_surrogates=False)
        result_2yr = model.simulate(years=2, use_surrogates=False)
        assert isinstance(result_1yr, float)
        assert isinstance(result_2yr, float)
        assert result_2yr >= 0.0
        
    def test_simulation_consistency(self, model):
        result1 = model.simulate(years=1, use_surrogates=False)
        result2 = model.simulate(years=1, use_surrogates=False)
        assert result1 == result2


class TestBatchOracle:
    def test_create_oracle(self, batch_oracle):
        assert batch_oracle is not None
        
    def test_evaluate_single_candidate(self, batch_oracle):
        population = [[1.5, 20.0, 27.0]]
        results = batch_oracle.evaluate_population(population, use_surrogates=False)
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], float)
        assert results[0] >= 0.0
        
    def test_evaluate_multiple_candidates(self, batch_oracle):
        population = [
            [1.5, 20.0, 27.0],
            [2.0, 21.0, 28.0],
            [1.0, 22.0, 29.0]
        ]
        results = batch_oracle.evaluate_population(population, use_surrogates=False)
        
        assert len(results) == 3
        assert all(isinstance(r, float) for r in results)
        assert all(r >= 0.0 for r in results)
        
    def test_evaluate_with_surrogates(self, batch_oracle):
        population = [[1.5, 20.0, 27.0], [2.0, 21.0, 28.0]]
        results = batch_oracle.evaluate_population(population, use_surrogates=True)
        
        assert len(results) == 2
        assert all(isinstance(r, float) for r in results)
        assert all(r >= 0.0 for r in results)
        
    def test_large_population(self, batch_oracle):
        population = [
            [1.0 + i * 0.1, 20.0 + i * 0.5, 27.0 + i * 0.5]
            for i in range(10)
        ]
        results = batch_oracle.evaluate_population(population, use_surrogates=False)
        
        assert len(results) == 10
        assert all(isinstance(r, float) for r in results)
        
    def test_empty_population(self, batch_oracle):
        population = []
        results = batch_oracle.evaluate_population(population, use_surrogates=False)
        
        assert isinstance(results, list)
        assert len(results) == 0


class TestParameterValidation:
    def test_invalid_u_value_negative(self, batch_oracle):
        population = [[-1.0, 20.0, 27.0]]
        results = batch_oracle.evaluate_population(population, use_surrogates=False)
        assert results[0] != results[0]
        
    def test_invalid_u_value_too_high(self, batch_oracle):
        population = [[10.0, 20.0, 27.0]]
        results = batch_oracle.evaluate_population(population, use_surrogates=False)
        assert results[0] != results[0]
        
    def test_invalid_heating_setpoint(self, batch_oracle):
        population = [[1.5, 500.0, 27.0]]
        results = batch_oracle.evaluate_population(population, use_surrogates=False)
        assert results[0] != results[0]
        
    def test_invalid_cooling_setpoint(self, batch_oracle):
        population = [[1.5, 20.0, 5.0]]
        results = batch_oracle.evaluate_population(population, use_surrogates=False)
        assert results[0] != results[0]
        
    def test_heating_equals_cooling(self, batch_oracle):
        population = [[1.5, 25.0, 25.0]]
        results = batch_oracle.evaluate_population(population, use_surrogates=False)
        assert results[0] != results[0]
        
    def test_heating_greater_than_cooling(self, batch_oracle):
        population = [[1.5, 27.0, 20.0]]
        results = batch_oracle.evaluate_population(population, use_surrogates=False)
        assert results[0] != results[0]
        
    def test_valid_boundary_values(self, batch_oracle):
        population = [
            [0.1, 15.0, 22.0],
            [5.0, 25.0, 32.0],
            [1.5, 20.0, 27.0]
        ]
        results = batch_oracle.evaluate_population(population, use_surrogates=False)
        
        assert all(r >= 0.0 for r in results)
        assert all(isinstance(r, float) for r in results)
        
    def test_mixed_valid_invalid(self, batch_oracle):
        population = [
            [1.5, 20.0, 27.0],
            [-1.0, 20.0, 27.0],
            [1.5, 500.0, 27.0],
            [1.5, 20.0, 27.0]
        ]
        results = batch_oracle.evaluate_population(population, use_surrogates=False)
        
        assert len(results) == 4
        assert results[0] >= 0.0
        assert results[1] != results[1]
        assert results[2] != results[2]
        assert results[3] >= 0.0


class TestMassClass:
    def test_mass_class_enum_values(self, fluxion_module):
        assert hasattr(fluxion_module, 'MassClass')
        mass_class = fluxion_module.MassClass
        
        assert hasattr(mass_class, 'VeryLight')
        assert hasattr(mass_class, 'Light')
        assert hasattr(mass_class, 'Medium')
        assert hasattr(mass_class, 'Heavy')
        assert hasattr(mass_class, 'VeryHeavy')


class TestSurfaceType:
    def test_surface_type_enum_values(self, fluxion_module):
        assert hasattr(fluxion_module, 'SurfaceType')
        surface_type = fluxion_module.SurfaceType
        
        assert hasattr(surface_type, 'Wall')
        assert hasattr(surface_type, 'Ceiling')
        assert hasattr(surface_type, 'Floor')
