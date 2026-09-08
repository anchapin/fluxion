from __future__ import annotations

import sys

import numpy as np
import pytest

SCRIPT_NAME = "generate_training_data"


@pytest.fixture
def generator(load_script):
    return load_script(SCRIPT_NAME)


def test_latin_hypercube_is_reproducible_and_within_bounds(generator):
    bounds = [("x", 0.0, 1.0), ("y", 10.0, 20.0)]
    first = generator.latin_hypercube_sample(bounds, 8, seed=7)
    assert first == generator.latin_hypercube_sample(bounds, 8, seed=7)
    assert all(0.0 <= row["x"] <= 1.0 and 10.0 <= row["y"] <= 20.0 for row in first)


def test_weather_sequence_is_deterministic(generator):
    profile = generator.WeatherProfile("temperate", transient_probability=0.0)
    first = generator.generate_weather_sequence(profile, 24, seed=4)
    second = generator.generate_weather_sequence(profile, 24, seed=4)
    assert first.shape == (24, 4)
    assert np.array_equal(first, second)
    assert np.all(first[:, 1:] >= 0)


def test_building_thermal_mass_and_invalid_profile_fallback(generator):
    building = generator.BuildingConfig(zone_area_m2=100.0, ceiling_height_m=3.0)
    assert building.thermal_mass_mj_k() > 0
    weather = generator.generate_weather_sequence(generator.WeatherProfile("unknown"), 2, seed=1)
    assert weather.shape == (2, 4)


def test_parse_args_accepts_small_scenario_run(generator, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["generate_training_data.py", "--n-scenarios", "2", "--output-dir", str(tmp_path)])
    args = generator.parse_args()
    assert args.n_scenarios == 2
    assert args.output_dir == tmp_path
