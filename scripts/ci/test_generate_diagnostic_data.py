from __future__ import annotations

import csv

import pytest

SCRIPT_NAME = "generate_diagnostic_data"


@pytest.fixture
def generator(load_script):
    return load_script(SCRIPT_NAME)


def test_temperature_and_invalid_case(generator):
    assert generator.generate_hourly_temperature(1, 23.0, 0.5) == 18.1
    import io
    output = io.StringIO()
    generator.generate_case_data(194, output)
    assert output.getvalue() == ""


def test_generate_case_writes_peak_and_rows(generator):
    import io
    output = io.StringIO()
    generator.generate_case_data(195, output)
    rows = list(csv.reader(output.getvalue().splitlines()))
    assert len(rows) == 8760
    assert rows[0][:2] == ["195", "1"]
    assert len(rows[0]) == 7
    assert all(row[6] == rows[0][6] for row in rows)


def test_main_writes_requested_range(generator, monkeypatch, tmp_path, capsys):
    output = tmp_path / "diagnostic.csv"
    monkeypatch.setattr(generator.sys, "argv", ["generate_diagnostic_data.py", "195", "195", str(output)])
    generator.main()
    rows = output.read_text().splitlines()
    assert rows[0].startswith("case,hour,")
    assert len(rows) == 8761
    assert "Generating case 195" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["generate_diagnostic_data.py"], ["generate_diagnostic_data.py", "x", "200", "out.csv"], ["generate_diagnostic_data.py", "194", "200", "out.csv"], ["generate_diagnostic_data.py", "195", "471", "out.csv"]])
def test_main_rejects_invalid_arguments(generator, monkeypatch, argv):
    monkeypatch.setattr(generator.sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        generator.main()
    assert exc.value.code == 1
