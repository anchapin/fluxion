"""
Tests for ASHRAE 140 case generator.

Run with: pytest tools/data_gen/test_ashrae_140_generator.py
"""

import json
import os
import tempfile

from tools.data_gen.ashrae_140_generator import (
    ASHRAE140CaseGenerator,
    CaseSpec,
    ConstructionType,
    GlassType,
    Orientation,
    ShadingType,
    save_cases_to_json,
)


def test_generator_initialization():
    """Test generator initialization with seed."""
    gen = ASHRAE140CaseGenerator(seed=42)
    assert gen.seed == 42


def test_generator_initialization_no_seed():
    """Test generator initialization without seed."""
    gen = ASHRAE140CaseGenerator()
    assert gen.seed is None


def test_generate_all_cases():
    """Test generating all 18 standard ASHRAE 140 cases."""
    gen = ASHRAE140CaseGenerator()
    cases = gen.generate_all_cases()

    assert len(cases) == 18

    case_ids = [c.case_id for c in cases]
    expected_ids = [
        "600",
        "610",
        "620",
        "630",
        "640",
        "650",
        "600FF",
        "650FF",
        "900",
        "910",
        "920",
        "930",
        "940",
        "950",
        "900FF",
        "950FF",
        "960",
        "195",
    ]

    for expected_id in expected_ids:
        assert expected_id in case_ids, f"Case {expected_id} not found"


def test_case_600_baseline():
    """Test Case 600 baseline specification."""
    gen = ASHRAE140CaseGenerator()
    case = gen.case_600_baseline()

    assert case.case_id == "600"
    assert "Low mass baseline" in case.description
    assert case.construction_type == ConstructionType.LOW_MASS
    assert len(case.windows) == 1
    assert len(case.windows[0]) == 1
    assert case.windows[0][0].orientation == Orientation.SOUTH
    assert case.windows[0][0].area == 12.0
    assert case.infiltration_ach == 0.5


def test_case_900_baseline():
    """Test Case 900 baseline specification."""
    gen = ASHRAE140CaseGenerator()
    case = gen.case_900_baseline()

    assert case.case_id == "900"
    assert "High mass baseline" in case.description
    assert case.construction_type == ConstructionType.HIGH_MASS
    assert len(case.windows) == 1
    assert len(case.windows[0]) == 1
    assert case.windows[0][0].orientation == Orientation.SOUTH
    assert case.windows[0][0].area == 12.0


def test_case_610_shading():
    """Test Case 610 with south shading."""
    gen = ASHRAE140CaseGenerator()
    case = gen.case_610_south_shading()

    assert case.case_id == "610"
    assert case.shading is not None
    assert case.shading.shading_type == ShadingType.OVERHANG
    assert case.shading.overhang_depth == 1.0


def test_case_630_shading():
    """Test Case 630 with east/west shading."""
    gen = ASHRAE140CaseGenerator()
    case = gen.case_630_ew_shading()

    assert case.case_id == "630"
    assert case.shading is not None
    assert case.shading.shading_type == ShadingType.OVERHANG_AND_FINS
    assert case.shading.overhang_depth == 1.0
    assert case.shading.fin_width == 1.0


def test_case_640_setback():
    """Test Case 640 with thermostat setback."""
    gen = ASHRAE140CaseGenerator()
    case = gen.case_640_setback()

    assert case.case_id == "640"
    assert len(case.hvac) == 1
    assert case.hvac[0].setback_setpoint is not None
    assert case.hvac[0].setback_setpoint == 10.0
    assert case.hvac[0].setback_hours == (23, 7)


def test_case_650_night_ventilation():
    """Test Case 650 with night ventilation."""
    gen = ASHRAE140CaseGenerator()
    case = gen.case_650_night_vent()

    assert case.case_id == "650"
    assert case.night_ventilation is not None
    assert case.night_ventilation.fan_capacity == 1703.16
    assert case.night_ventilation.operating_hours == (18, 7)


def test_case_600ff_free_floating():
    """Test Case 600FF free-floating."""
    gen = ASHRAE140CaseGenerator()
    case = gen.case_600ff()

    assert case.case_id == "600FF"
    assert case.hvac[0].is_free_floating()


def test_case_960_sunspace():
    """Test Case 960 sunspace (2-zone)."""
    gen = ASHRAE140CaseGenerator()
    case = gen.case_960_sunspace()

    assert case.case_id == "960"
    assert case.num_zones == 2
    assert len(case.geometry) == 2
    assert len(case.windows) == 2
    assert case.geometry[0].depth == 6.0
    assert case.geometry[1].depth == 3.0


def test_case_195_solid_conduction():
    """Test Case 195 solid conduction."""
    gen = ASHRAE140CaseGenerator()
    case = gen.case_195_solid_conduction()

    assert case.case_id == "195"
    assert len(case.windows[0]) == 0
    assert case.infiltration_ach == 0.0
    assert case.internal_loads[0] is None
    assert case.hvac[0].is_free_floating()


def test_generate_variations():
    """Test generating parameter variations."""
    gen = ASHRAE140CaseGenerator(seed=42)
    base_case = gen.case_600_baseline()

    variations = gen.generate_variations(
        base_case=base_case,
        u_value_variation=0.3,
        setpoint_variation=2.0,
        num_variations=5,
    )

    assert len(variations) == 5

    # Check that variations have different case IDs
    variation_ids = [v.case_id for v in variations]
    assert len(variation_ids) == len(set(variation_ids))

    # Check that at least one variation has different U-values
    u_values = [v.construction.wall_u_value for v in variations]
    assert len(set(u_values)) > 1

    # Check that at least one variation has different setpoints
    heating_setpoints = [v.hvac[0].heating_setpoint for v in variations]
    assert len(set(heating_setpoints)) > 1


def test_generate_all_variations():
    """Test generating variations for all cases."""
    gen = ASHRAE140CaseGenerator(seed=42)

    variations = gen.generate_all_variations(
        u_value_variation=0.2,
        setpoint_variation=1.0,
        num_variations_per_case=2,
    )

    # 18 base cases * 2 variations each = 36 total
    assert len(variations) == 36


def test_variation_reproducibility():
    """Test that variations are reproducible with same seed."""
    gen1 = ASHRAE140CaseGenerator(seed=42)
    gen2 = ASHRAE140CaseGenerator(seed=42)

    base_case = gen1.case_600_baseline()

    variations1 = gen1.generate_variations(
        base_case=base_case,
        u_value_variation=0.5,
        setpoint_variation=5.0,
        num_variations=10,
    )

    variations2 = gen2.generate_variations(
        base_case=base_case,
        u_value_variation=0.5,
        setpoint_variation=5.0,
        num_variations=10,
    )

    assert len(variations1) == len(variations2)

    for i, (v1, v2) in enumerate(zip(variations1, variations2)):
        assert v1.construction.wall_u_value == v2.construction.wall_u_value
        assert v1.hvac[0].heating_setpoint == v2.hvac[0].heating_setpoint


def test_to_dict_serialization():
    """Test CaseSpec to_dict method."""
    gen = ASHRAE140CaseGenerator()
    case = gen.case_600_baseline()

    case_dict = case.to_dict()

    assert "case_id" in case_dict
    assert "description" in case_dict
    assert "construction_type" in case_dict
    assert "geometry" in case_dict
    assert "windows" in case_dict
    assert "hvac" in case_dict

    assert case_dict["case_id"] == "600"
    assert case_dict["construction_type"] == "LowMass"


def test_save_and_load_json():
    """Test saving and loading cases from JSON."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        temp_path = f.name

    try:
        gen = ASHRAE140CaseGenerator()
        cases = gen.generate_all_cases()

        save_cases_to_json(cases, temp_path)

        with open(temp_path, "r") as f:
            loaded_data = json.load(f)

        assert len(loaded_data) == len(cases)
        assert loaded_data[0]["case_id"] == cases[0].case_id

    finally:
        os.unlink(temp_path)


def test_window_spec_double_clear_glass():
    """Test WindowSpec.double_clear_glass factory method."""
    from tools.data_gen.ashrae_140_generator import WindowSpec

    window_spec = WindowSpec.double_clear_glass()

    assert window_spec.u_value == 3.0
    assert window_spec.shgc == 0.789
    assert window_spec.normal_transmittance == 0.86156
    assert window_spec.glass_type == GlassType.DOUBLE_CLEAR


def test_shading_device_factories():
    """Test ShadingDevice factory methods."""
    from tools.data_gen.ashrae_140_generator import ShadingDevice

    none_shading = ShadingDevice.none()
    assert none_shading.shading_type == ShadingType.NONE

    overhang = ShadingDevice.overhang(1.0, 2.7)
    assert overhang.shading_type == ShadingType.OVERHANG
    assert overhang.overhang_depth == 1.0
    assert overhang.mounting_height == 2.7

    fins = ShadingDevice.fins(1.0)
    assert fins.shading_type == ShadingType.FINS
    assert fins.fin_width == 1.0

    both = ShadingDevice.overhang_and_fins(1.0, 1.0, 2.7)
    assert both.shading_type == ShadingType.OVERHANG_AND_FINS
    assert both.overhang_depth == 1.0
    assert both.fin_width == 1.0


def test_hvac_schedule_factories():
    """Test HvacSchedule factory methods."""
    from tools.data_gen.ashrae_140_generator import HvacSchedule

    constant = HvacSchedule.constant(20.0, 27.0)
    assert constant.heating_setpoint == 20.0
    assert constant.cooling_setpoint == 27.0
    assert constant.operating_hours == (0, 24)
    assert not constant.is_free_floating()

    setback = HvacSchedule.with_setback(20.0, 27.0, 10.0, 23, 7)
    assert setback.setback_setpoint == 10.0
    assert setback.setback_hours == (23, 7)

    operating = HvacSchedule.with_operating_hours(20.0, 27.0, 7, 18)
    assert operating.operating_hours == (7, 18)

    free = HvacSchedule.free_floating()
    assert free.is_free_floating()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
