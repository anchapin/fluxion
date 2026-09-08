from __future__ import annotations

import pytest

SCRIPT_NAME = "generate_reference_data"


@pytest.fixture
def generator(load_script):
    return load_script(SCRIPT_NAME)


def test_hourly_temperature_is_deterministic_and_rounded(generator):
    assert generator.generate_hourly_temperature(1, 21.0, 5.0) == 16.1
    assert generator.generate_hourly_temperature(8760, 21.0, 5.0) == 16.4
    assert generator.generate_hourly_temperature(1, 21.0, 5.0) == generator.generate_hourly_temperature(1, 21.0, 5.0)


def test_hourly_energy_uses_occupancy_and_deterministic_hash(generator):
    occupied = generator.generate_hourly_energy(9, 1000.0, 0.0, 1.0)
    unoccupied = generator.generate_hourly_energy(1, 1000.0, 0.0, 1.0)
    assert occupied == 999.0
    assert unoccupied == 198.2
    assert generator.hash(1) == 2654435761


def test_case_data_handles_supported_and_unknown_cases(generator):
    assert generator.generate_case_data(799) == []
    data = generator.generate_case_data(801)
    assert len(data) == 8760
    assert data[0] == {
        "case": 801,
        "hour": 1,
        "zone1_temp": 18.1,
        "zone1_heating": 49.68000000000001,
        "zone1_cooling": 65.52000000000001,
        "zone2_temp": 17.9,
        "zone2_heating": 49.52,
        "zone2_cooling": 65.36,
        "total_energy": 230.08000000000004,
    }
    assert data[-1]["hour"] == 8760


def test_main_emits_golden_csv_rows(generator, monkeypatch, capsys):
    monkeypatch.setattr(generator.sys, "argv", ["generate_reference_data.py", "800", "800"])
    generator.main()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "case,hour,zone1_temp,zone1_heating,zone1_cooling,zone2_temp,zone2_heating,zone2_cooling,total_energy"
    assert lines[1] == "800,1,16.1,82.8,83.2,16.1,82.4,82.8,331.2"
    assert lines[-1].startswith("800,8760,")
    assert len(lines) == 8761


@pytest.mark.parametrize("argv", [["generate_reference_data.py"], ["generate_reference_data.py", "x", "810"], ["generate_reference_data.py", "799", "810"], ["generate_reference_data.py", "800", "811"]])
def test_main_rejects_invalid_arguments(generator, monkeypatch, argv, capsys):
    monkeypatch.setattr(generator.sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        generator.main()
    assert exc.value.code == 2
    assert "error" in capsys.readouterr().err.lower()
