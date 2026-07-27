"""Tests for osimflow.data_gen.generate_monte_carlo_deltas (Issue #1813).

Run with: ``pytest osimflow/data_gen/test_generate_deltas.py``
"""

import json
from pathlib import Path

import pytest

from osimflow.data_gen.generate_monte_carlo_deltas import (
    DEFAULT_SAMPLES,
    DEFAULT_SEED,
    Distribution,
    DeltaSpec,
    ParameterConfig,
    generate_delta_file,
    materialize_patches,
    default_parameter_set,
    main,
)


def _spec(samples=5, seed=42) -> DeltaSpec:
    return DeltaSpec(
        parameters=[
            ParameterConfig("infiltration_ach", Distribution.uniform(0.3, 1.5)),
            ParameterConfig(
                "window_properties.u_value", Distribution.normal(3.0, 0.3)
            ),
        ],
        samples=samples,
        seed=seed,
        warm_up_years=0,
    )


def test_uniform_respects_bounds():
    rng = __import__("random").Random(1)
    d = Distribution.uniform(2.0, 4.0)
    for _ in range(100):
        v = d.sample(rng)
        assert 2.0 <= v <= 4.0


def test_normal_rejects_nonpositive_std():
    with pytest.raises(ValueError):
        Distribution.normal(1.0, 0.0)


def test_triangular_rejects_bad_mode():
    with pytest.raises(ValueError):
        Distribution.triangular(0.0, 5.0, 1.0)  # mode > max


def test_fixed_is_constant():
    rng = __import__("random").Random(1)
    d = Distribution.fixed(1.25)
    assert all(d.sample(rng) == 1.25 for _ in range(10))


def test_delta_file_has_expected_shape():
    delta = generate_delta_file(_spec())
    assert delta["samples"] == 5
    assert delta["seed"] == 42
    assert set(delta["parameters"]) == {"infiltration_ach", "window_properties.u_value"}
    assert delta["parameters"]["infiltration_ach"]["distribution"] == "uniform"


def test_generate_rejects_zero_samples():
    with pytest.raises(ValueError):
        generate_delta_file(DeltaSpec(parameters=default_parameter_set(), samples=0))


def test_generate_rejects_no_parameters():
    with pytest.raises(ValueError):
        generate_delta_file(DeltaSpec(parameters=[], samples=10))


def test_materialize_writes_one_file_per_draw(tmp_path: Path):
    spec = _spec(samples=10, seed=7)
    draws = materialize_patches(spec, out_dir=tmp_path)
    assert len(draws) == 10
    files = sorted(tmp_path.glob("delta_*.json"))
    assert len(files) == 10
    first = json.loads(files[0].read_text())
    assert first["index"] == 0
    assert "infiltration_ach" in first["values"]
    assert "window_properties.u_value" in first["values"]


def test_materialize_is_deterministic_for_seed():
    spec = _spec(samples=20, seed=99)
    a = materialize_patches(spec)
    b = materialize_patches(spec)
    assert a == b


def test_different_seeds_yield_different_draws():
    a = materialize_patches(_spec(samples=20, seed=1))
    b = materialize_patches(_spec(samples=20, seed=2))
    assert a != b


def test_default_parameter_set_covers_issue_params():
    paths = {p.path for p in default_parameter_set()}
    # Issue #1813 names infiltration, window U-value / SHGC explicitly.
    assert "infiltration_ach" in paths
    assert "window_properties.u_value" in paths


def test_cli_main_writes_delta_file(tmp_path: Path):
    out = tmp_path / "delta.yaml"
    rc = main(["-n", "12", "--seed", "5", "-o", str(out)])
    assert rc == 0
    text = out.read_text()
    assert "samples: 12" in text
    assert "seed: 5" in text


def test_cli_main_materializes(tmp_path: Path):
    out = tmp_path / "delta.json"
    mat = tmp_path / "patches"
    rc = main(
        ["-n", "8", "--seed", "3", "-o", str(out), "--materialize", str(mat)]
    )
    assert rc == 0
    delta = json.loads(out.read_text())
    assert delta["samples"] == 8
    patches = sorted(mat.glob("delta_*.json"))
    assert len(patches) == 8


def test_defaults_match_issue():
    assert DEFAULT_SAMPLES == 1000
    assert DEFAULT_SEED == 0x5EED_1813
