"""
Python tests for HVAC bindings
"""

import fluxion
import pytest

ZoneSetpoints = getattr(fluxion, "ZoneSetpoints", None)
ZoneControl = getattr(fluxion, "ZoneControl", None)
create_zone_setpoints = getattr(fluxion, "create_zone_setpoints", None)

if ZoneSetpoints is None or ZoneControl is None or create_zone_setpoints is None:
    pytest.skip("fluxion HVAC bindings not available", allow_module_level=True)


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
    MultiZoneThermalModel = fluxion.MultiZoneThermalModel

    # Create thermal model
    thermal_model = MultiZoneThermalModel(3)

    # Create setpoints
    setpoints = fluxion.ZoneSetpoints(3)
    setpoints.set_heating_setpoint(0, 22.0)
    setpoints.set_cooling_setpoint(0, 26.0)
    setpoints.set_deadband(0, 2.0)

    # Create zone control
    zone_control = fluxion.ZoneControl(thermal_model, setpoints)

    # Verify it was created successfully
    assert zone_control is not None


def test_hvac_control_update():
    """Test HVAC control logic with various temperatures"""
    MultiZoneThermalModel = fluxion.MultiZoneThermalModel

    # Create thermal model
    thermal_model = MultiZoneThermalModel(1)

    # Create setpoints
    setpoints = fluxion.ZoneSetpoints(1)
    setpoints.set_heating_setpoint(0, 22.0)
    setpoints.set_cooling_setpoint(0, 26.0)
    setpoints.set_deadband(0, 2.0)

    # Create zone control
    zone_control = fluxion.ZoneControl(thermal_model, setpoints)

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
    MultiZoneThermalModel = fluxion.MultiZoneThermalModel

    # Create thermal model
    thermal_model = MultiZoneThermalModel(1)

    # Create setpoints
    setpoints = fluxion.ZoneSetpoints(1)
    setpoints.set_heating_setpoint(0, 22.0)
    setpoints.set_cooling_setpoint(0, 26.0)
    setpoints.set_deadband(0, 2.0)

    # Create zone control
    zone_control = fluxion.ZoneControl(thermal_model, setpoints)

    # Must call update_controls first to compute zone status
    energy = zone_control.update_controls([20.0])

    # Test heating energy: thermodynamic calculation with zone_volume=129.6 m³,
    # ACH=0.5, supply_heating_temp=40°C, heating_efficiency=0.9
    # airflow = 129.6 * 0.5 / 3600 = 0.018 m³/s
    # mass_flow = 0.018 * 1.2 = 0.0216 kg/s
    # delta_t = 40 - 20 = 20°C
    # Q = 0.0216 * 1005 * 20 = 434.16 W (thermal)
    # Electrical = 434.16 / 0.9 = 482.4 W
    assert abs(energy[0] - 482.4) < 1.0

    # Test cooling energy: update with higher temperature
    # Zone at 28°C, cooling setpoint 26°C, supply_cooling_temp=13°C, COP=3.0
    # delta_t = 28 - 13 = 15°C
    # Q = 0.0216 * 1005 * 15 = 325.62 W (thermal)
    # Electrical = 325.62 / 3.0 = 108.5 W
    energy = zone_control.update_controls([28.0])
    assert abs(energy[0] - 108.5) < 1.0

    # Test no energy in deadband
    energy = zone_control.update_controls([23.0])
    assert energy[0] == 0.0


def test_create_zone_setpoints_from_config():
    """Test creating ZoneSetpoints from configuration dictionary"""
    config = {
        "num_zones": 2,
        "zones": {
            "zone_0": {"heating": 21.0, "cooling": 25.0, "deadband": 2.0},
            "zone_1": {"heating": 20.0, "cooling": 24.0, "deadband": 1.5},
        },
    }

    setpoints = fluxion.create_zone_setpoints(config)

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
    MultiZoneThermalModel = fluxion.MultiZoneThermalModel

    # Create thermal model with 3 zones
    thermal_model = MultiZoneThermalModel(3)

    # Create setpoints with different values for each zone
    setpoints = fluxion.ZoneSetpoints(3)
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
    zone_control = fluxion.ZoneControl(thermal_model, setpoints)

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


# =============================================================================
# Deep HVAC configuration tests (Issue #1797)
# =============================================================================

# Bindings may be absent in builds without the python-bindings feature; skip
# the whole module if the core HVAC surface is missing.
_VAVUnit = getattr(fluxion, "VavTerminalUnit", None)
_HVACSystemType = getattr(fluxion, "HVACSystemType", None)

if _VAVUnit is None or _HVACSystemType is None:
    pytest.skip(
        "fluxion deep HVAC bindings not available", allow_module_level=True
    )


def test_hvac_system_type_enum():
    """HVACSystemType enum is exposed and comparable."""
    assert fluxion.HVACSystemType.VAV != fluxion.HVACSystemType.CAV
    assert fluxion.HVACSystemType.VAV == fluxion.HVACSystemType.VAV
    assert fluxion.HVACSystemType.Ideal != fluxion.HVACSystemType.Simple


def test_hvac_mode_enum():
    """HVACMode and HeatPumpMode enums are exposed."""
    assert fluxion.HVACMode.Heating != fluxion.HVACMode.Cooling
    assert fluxion.HVACMode.Off == fluxion.HVACMode.Off
    assert fluxion.HeatPumpMode.Heating != fluxion.HeatPumpMode.Cooling


def test_vav_operating_mode_enum():
    """VavOperatingMode enum is exposed."""
    assert fluxion.VavOperatingMode.Cooling != fluxion.VavOperatingMode.Heating
    assert fluxion.VavOperatingMode.Deadband == fluxion.VavOperatingMode.Deadband


def test_chiller_configuration():
    """Chiller equipment configuration round-trips through Python."""
    chiller = fluxion.Chiller("CH-1", 35000.0, 4.5, 35.0)
    assert chiller.id == "CH-1"
    assert chiller.cooling_capacity == 35000.0
    assert chiller.cooling_cop == 4.5
    assert chiller.design_temp == 35.0
    # Capacity scales with part-load ratio.
    cap_full = chiller.calculate_capacity(1.0, 35.0)
    cap_half = chiller.calculate_capacity(0.5, 35.0)
    assert abs(cap_half - cap_full * 0.5) < 1.0
    assert chiller.current_plr == 0.0


def test_boiler_configuration():
    """Boiler equipment configuration round-trips through Python."""
    boiler = fluxion.Boiler("BLR-1", 50000.0, 0.90, -5.0)
    assert boiler.id == "BLR-1"
    assert boiler.heating_capacity == 50000.0
    assert boiler.efficiency == 0.90
    assert boiler.design_temp == -5.0
    cap = boiler.calculate_capacity(1.0, 0.0)
    assert cap > 0.0


def test_heat_pump_configuration_and_cop():
    """HeatPump configuration, COP, and mode selection from Python."""
    hp = fluxion.HeatPump("HP-1", 12000.0, 10000.0, 3.5, 3.0)
    assert hp.id == "HP-1"
    assert hp.heating_capacity == 12000.0
    assert hp.cooling_capacity == 10000.0
    assert hp.heating_cop == 3.5
    assert hp.cooling_cop == 3.0
    assert hp.mode == "off"

    # COP degrades away from design temperature.
    cop_design = hp.heating_cop_at_temperature(-5.0)
    cop_cold = hp.heating_cop_at_temperature(-15.0)
    assert cop_cold < cop_design

    # Mode selection from setpoints.
    hp.set_mode(18.0, 20.0, 27.0)
    assert hp.mode == "heating"
    hp.set_mode(28.0, 20.0, 27.0)
    assert hp.mode == "cooling"
    hp.set_mode(22.0, 20.0, 27.0)
    assert hp.mode == "off"


def test_cav_system_configuration():
    """CAVSystem configuration and setters round-trip."""
    cav = fluxion.CAVSystem("CAV-1", 1.0)
    assert cav.id == "CAV-1"
    assert cav.design_airflow == 1.0
    cav.set_fan_efficiency(0.8)
    assert cav.fan_efficiency == 0.8
    cav.set_cooling_capacity(15000.0)
    assert cav.cooling_capacity == 15000.0
    assert cav.fan_power_consumption() > 0.0


def test_simple_vav_terminal_configuration():
    """High-level VAVTerminal configuration round-trips."""
    vav = fluxion.VAVTerminal("VAV-1", 0, 0.5)
    assert vav.id == "VAV-1"
    assert vav.zone_id == 0
    assert vav.max_airflow == 0.5
    # Defaults: min = 30% of max, reheat = 5000 W.
    assert abs(vav.min_airflow - 0.15) < 1e-9
    assert vav.reheat_capacity == 5000.0

    vav.set_min_airflow(0.2)
    assert vav.min_airflow == 0.2
    vav.set_reheat_capacity(6000.0)
    assert vav.reheat_capacity == 6000.0


def test_vav_terminal_unit_round_trip():
    """Round-trip test building a detailed VAV system from Python.

    Verifies the acceptance criterion: a VAV system built from Python can be
    read back and the configuration matches what was set.
    """
    # Build a VAV terminal with cooling + reheat coils from Python.
    terminal = fluxion.VavTerminalUnit(
        id="VAV-RT-1",
        zone_id=2,
        max_airflow=0.5,
        cooling_capacity=8000.0,
        reheat_capacity=5000.0,
    )

    # Round-trip: read back every configured value.
    assert terminal.id == "VAV-RT-1"
    assert terminal.zone_id == 2
    assert terminal.max_airflow == 0.5
    assert terminal.rated_cooling_capacity == 8000.0
    assert terminal.rated_reheat_capacity == 5000.0
    assert terminal.has_reheat is True

    # Default turndown is 30%.
    assert abs(terminal.min_airflow_ratio - 0.30) < 1e-9
    assert abs(terminal.min_airflow - 0.5 * 0.30) < 1e-9

    # Mutate the turndown and confirm it persists.
    terminal.set_min_airflow_ratio(0.40)
    assert abs(terminal.min_airflow_ratio - 0.40) < 1e-9
    assert abs(terminal.min_airflow - 0.5 * 0.40) < 1e-9


def test_vav_terminal_unit_cooling_only():
    """A cooling-only terminal (no reheat) reports has_reheat == False."""
    terminal = fluxion.VavTerminalUnit(
        id="VAV-CO",
        zone_id=0,
        max_airflow=0.3,
        cooling_capacity=5000.0,
        reheat_capacity=0.0,
    )
    assert terminal.has_reheat is False
    assert terminal.rated_reheat_capacity == 0.0


def test_vav_terminal_control_modes():
    """VavTerminalControl factories resolve to the correct operating modes."""
    cooling = fluxion.VavTerminalControl.cooling(0.8)
    assert cooling.cooling_active is True
    assert cooling.mode == fluxion.VavOperatingMode.Cooling

    heating = fluxion.VavTerminalControl.heating(35.0)
    assert heating.cooling_active is False
    assert heating.mode == fluxion.VavOperatingMode.Heating

    deadband = fluxion.VavTerminalControl.deadband()
    assert deadband.mode == fluxion.VavOperatingMode.Deadband

    # Generic constructor.
    custom = fluxion.VavTerminalControl(0.5, cooling_active=False, reheat_setpoint=40.0)
    assert custom.mode == fluxion.VavOperatingMode.Heating
    assert custom.damper_position == 0.5


def test_compute_vav_terminal_performance_cooling():
    """compute_vav_terminal_performance produces a valid cooling result."""
    terminal = fluxion.VavTerminalUnit(
        id="VAV-PERF",
        zone_id=0,
        max_airflow=0.5,
        cooling_capacity=10000.0,
        reheat_capacity=0.0,
    )
    control = fluxion.VavTerminalControl.cooling(1.0)
    perf = fluxion.compute_vav_terminal_performance(
        terminal,
        entering_dry_bulb_c=26.0,
        entering_humidity_ratio=0.010,
        air_density_kg_per_m3=1.2,
        control=control,
    )
    assert perf.mode == fluxion.VavOperatingMode.Cooling
    assert perf.cooling_total_capacity_w > 0.0
    assert perf.cooling_sensible_capacity_w > 0.0
    assert perf.volumetric_flow_m3_per_s > 0.0
    assert perf.fan_motor_power_w > 0.0
    # Supply air should be cooler than entering air in cooling mode.
    assert perf.supply_dry_bulb_c < 26.0


def test_compute_vav_terminal_performance_deadband():
    """In deadband the terminal delivers minimum flow with no coil capacity."""
    terminal = fluxion.VavTerminalUnit(
        id="VAV-DB",
        zone_id=0,
        max_airflow=0.5,
        cooling_capacity=10000.0,
        reheat_capacity=5000.0,
    )
    control = fluxion.VavTerminalControl.deadband()
    perf = fluxion.compute_vav_terminal_performance(
        terminal,
        entering_dry_bulb_c=22.0,
        entering_humidity_ratio=0.009,
        air_density_kg_per_m3=1.2,
        control=control,
    )
    assert perf.mode == fluxion.VavOperatingMode.Deadband
    assert perf.cooling_total_capacity_w == 0.0
    assert perf.reheat_capacity_w == 0.0
    # Minimum airflow is still delivered (30% of 0.5).
    assert perf.volumetric_flow_m3_per_s > 0.0


def test_vav_terminal_unit_invalid_inputs():
    """Constructor rejects non-positive / non-finite inputs."""
    with pytest.raises(ValueError, match="max_airflow"):
        fluxion.VavTerminalUnit("X", 0, -0.5, 1000.0, 0.0)
    with pytest.raises(ValueError, match="cooling_capacity"):
        fluxion.VavTerminalUnit("X", 0, 0.5, 0.0, 0.0)
