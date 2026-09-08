from __future__ import annotations

import json

import pytest

SCRIPT_NAME = "check_tdqs_regression"


@pytest.fixture
def checker(load_script):
    return load_script(SCRIPT_NAME)


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_extract_tdqs_supports_mapping_and_scalar_entries(checker):
    overall, values = checker.extract_tdqs({"overall": "0.8", "per_type": {"solver_selection": {"tdqs": 0.7}, "adaptive_timestep": 0.6, "surrogate_routing": 0.5, "constraint_warning": 0.4, "hvac_horizon": 0.3}})
    assert overall == 0.8
    assert values["solver_selection"] == 0.7
    assert values["adaptive_timestep"] == 0.6
    assert values["hvac_horizon"] == 0.3


def test_check_regression_clean_and_regressed(checker, tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    base = {"overall": 0.8, "per_type": {name: {"tdqs": 0.8} for name in checker.DECISION_TYPES}}
    write_json(baseline, base)
    write_json(current, {"overall": 0.76, "per_type": base["per_type"]})
    assert checker.check_regression(str(current), str(baseline), 0.05) == 0
    write_json(current, {"overall": 0.7, "per_type": base["per_type"]})
    assert checker.check_regression(str(current), str(baseline), 0.05) == 1
    assert checker.check_regression(str(current), str(baseline), 0.05, warn_only=True) == 0


def test_check_regression_writes_github_outputs(checker, monkeypatch, tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    per_type = {name: 0.5 for name in checker.DECISION_TYPES}
    write_json(baseline, {"overall": 0.5, "per_type": per_type})
    write_json(current, {"overall": 0.6, "per_type": per_type})
    output = tmp_path / "output"
    summary = tmp_path / "summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert checker.check_regression(str(current), str(baseline)) == 0
    assert "tdqs_overall=0.600000" in output.read_text()
    assert "TDQS Regression Check" in summary.read_text()
