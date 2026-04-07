"""
Python tests for HVAC bindings
"""

import pytest
import fluxion

# Use direct imports since hvac submodule is not working
ZoneSetpoints = fluxion.ZoneSetpoints
ZoneControl = fluxion.ZoneControl
create_zone_setpoints = fluxion.create_zone_setpoints


def test_setpoint_creation():
    """Test ZoneSetpoints creation and basic properties"""
    setpoints = ZoneSetpoints(3)
    assert setpoints.num_zones() == 3

    # Test default values
    assert setpoints.get_heating_setpoint(0) == 20.0
    assert setpoints.get_cooling_setpoint(0) == 24.0
    assert setpoints.get_deadband(0) == 2.0


def test_setpoint_getters_setters():
    """Test all getter and setter methods"""
    setpoints = ZoneSetpoints(2)

    # Set and verify heating setpoint
    setpoints.set_heating_setpoint(0, 22.0)
    assert setpoints.get_heating_setpoint(0) == 22.0

    # Set and verify cooling setpoint
    setpoints.set_cooling_setpoint(1, 26.0)
    assert setpoints.get_cooling_setpoint(1) == 26.0

    # Set and verify deadband
    setpoints.set_deadband(0, 3.0)
    assert setpoints.get_deadband(0) == 3.0


def test_setpoint_validation():
    """Test validation with various configurations"""
    # Valid configuration
    setpoints = ZoneSetpoints(1)
    setpoints.set_heating_setpoint(0, 21.0)
    setpoints.set_cooling_setpoint(0, 25.0)
    setpoints.set_deadband(0, 2.0)
    setpoints.validate()  # Should not raise

    # Invalid temperature range
    with pytest.raises(ValueError, match="out of valid range"):
        setpoints.set_heating_setpoint(0, 5.0)

    with pytest.raises(ValueError, match="out of valid range"):
        setpoints.set_cooling_setpoint(0, 45.0)

    # Invalid deadband range
    with pytest.raises(ValueError, match="out of valid range"):
        setpoints.set_deadband(0, 0.0)

    with pytest.raises(ValueError, match="out of valid range"):
        setpoints.set_deadband(0, 6.0)

    # Invalid zone ID
    with pytest.raises(ValueError, match="out of range"):
        setpoints.set_heating_setpoint(5, 22.0)

    # Heating setpoint >= cooling setpoint
    setpoints2 = ZoneSetpoints(1)
    setpoints2.set_heating_setpoint(0, 25.0)
    setpoints2.set_cooling_setpoint(0, 23.0)
    with pytest.raises(ValueError, match="must be below"):
        setpoints2.validate()


def test_zone_control_creation():
    """Test ZoneControl creation from thermal model and setpoints"""
    from fluxion.multi_zone import MultiZoneThermalModel

    # Create thermal model
    thermal_model = MultiZoneThermalModel(3)

    # Create setpoints
    setpoints = fluxion.hvac.ZoneSetpoints(3)
    setpoints.set_heating_setpoint(0, 22.0)
    setpoints.set_cooling_setpoint(0, 26.0)
    setpoints.set_deadband(0, 2.0)

    # Create zone control
    zone_control = fluxion.hvac.ZoneControl(thermal_model, setpoints)

    # Verify it was created successfully
    assert zone_control is not None


def test_hvac_control_update():
    """Test HVAC control logic with various temperatures"""
    from fluxion.multi_zone import MultiZoneThermalModel

    # Create thermal model
    thermal_model = MultiZoneThermalModel(1)

    # Create setpoints
    setpoints = fluxion.hvac.ZoneSetpoints(1)
    setpoints.set_heating_setpoint(0, 22.0)
    setpoints.set_cooling_setpoint(0, 26.0)
    setpoints.set_deadband(0, 2.0)

    # Create zone control
    zone_control = fluxion.hvac.ZoneControl(thermal_model, setpoints)

    # Test heating (below heating threshold: 22 - 1 = 21°C)
    energy = zone_control.update_controls([20.0])
    assert energy[0] > 0.0  # Should have heating energy
    assert zone_control.get_zone_status(0) == "heating"

    # Test cooling (above cooling threshold: 26 + 1 = 27°C)
    energy = zone_control.update_controls([28.0])
    assert energy[0] > 0.0  # Should have cooling energy
    assert zone_control.get_zone_status(0) == "cooling"

    # Test deadband (21°C to 25°C)
    energy = zone_control.update_controls([23.0])
    assert energy[0] == 0.0  # Should have no energy
    assert zone_control.get_zone_status(0) == "off"


def test_energy_calculation():
    """Test energy calculation accuracy"""
    from fluxion.multi_zone import MultiZoneThermalModel

    # Create thermal model
    thermal_model = MultiZoneThermalModel(1)

    # Create setpoints
    setpoints = fluxion.hvac.ZoneSetpoints(1)
    setpoints.set_heating_setpoint(0, 22.0)
    setpoints.set_cooling_setpoint(0, 26.0)
    setpoints.set_deadband(0, 2.0)

    # Create zone control
    zone_control = fluxion.hvac.ZoneControl(thermal_model, setpoints)

    # Test heating energy: 2°C difference * 1000W/°C = 2000W
    energy = zone_control.get_energy_input(0, 20.0)
    assert abs(energy - 2000.0) < 0.01

    # Test cooling energy: 2°C difference * 1000W/°C = 2000W
    energy = zone_control.get_energy_input(0, 28.0)
    assert abs(energy - 2000.0) < 0.01

    # Test no energy in deadband
    energy = zone_control.get_energy_input(0, 23.0)
    assert energy == 0.0


def test_create_zone_setpoints_from_config():
    """Test creating ZoneSetpoints from configuration dictionary"""
    config = {
        "num_zones": 2,
        "zones": {
            "zone_0": {"heating": 21.0, "cooling": 25.0, "deadband": 2.0},
            "zone_1": {"heating": 20.0, "cooling": 24.0, "deadband": 1.5},
        },
    }

    setpoints = fluxion.hvac.create_zone_setpoints(config)

    # Verify configuration was applied
    assert setpoints.num_zones() == 2
    assert setpoints.get_heating_setpoint(0) == 21.0
    assert setpoints.get_cooling_setpoint(0) == 25.0
    assert setpoints.get_deadband(0) == 2.0
    assert setpoints.get_heating_setpoint(1) == 20.0
    assert setpoints.get_cooling_setpoint(1) == 24.0
    assert setpoints.get_deadband(1) == 1.5


def test_independent_zone_control():
    """Test that zones operate independently"""
    from fluxion.multi_zone import MultiZoneThermalModel

    # Create thermal model with 3 zones
    thermal_model = MultiZoneThermalModel(3)

    # Create setpoints with different values for each zone
    setpoints = fluxion.hvac.ZoneSetpoints(3)
    setpoints.set_heating_setpoint(0, 22.0)  # Zone 0: heating
    setpoints.set_cooling_setpoint(0, 26.0)
    setpoints.set_deadband(0, 2.0)

    setpoints.set_heating_setpoint(1, 20.0)  # Zone 1: deadband
    setpoints.set_cooling_setpoint(1, 24.0)
    setpoints.set_deadband(1, 2.0)

    setpoints.set_heating_setpoint(2, 18.0)  # Zone 2: cooling
    setpoints.set_cooling_setpoint(2, 22.0)
    setpoints.set_deadband(2, 2.0)

    # Create zone control
    zone_control = fluxion.hvac.ZoneControl(thermal_model, setpoints)

    # Test with temperatures that trigger different states
    # Zone 0: 19°C (below heating setpoint) -> heating
    # Zone 1: 22°C (within deadband) -> off
    # Zone 2: 25°C (above cooling setpoint) -> cooling
    energy = zone_control.update_controls([19.0, 22.0, 25.0])

    assert zone_control.get_zone_status(0) == "heating"
    assert zone_control.get_zone_status(1) == "off"
    assert zone_control.get_zone_status(2) == "cooling"

    assert energy[0] > 0.0  # Heating energy
    assert energy[1] == 0.0  # No energy in deadband
    assert energy[2] > 0.0  # Cooling energy
