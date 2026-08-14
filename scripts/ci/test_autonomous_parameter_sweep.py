"""
Tests for ``scripts/autonomous_parameter_sweep.py`` — Issue #1847.

Targets the parameter-space enumeration, sweep-config persistence, and
the high-value ``--dry-run`` path that is the most heavily-used CI
smoke check.  ``subprocess.run`` is mocked so no real ``cargo test``
invocation occurs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import autonomous_parameter_sweep as aps
import pytest

# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def sweep_config() -> aps.SweepConfig:
    return aps.SweepConfig(
        case_id="600",
        sweep_type=aps.SweepType.GRID,
        parameters=[
            aps.ParameterSpec(
                "R_value", default=2.0, min_val=1.0, max_val=2.0, step=1.0
            ),
            aps.ParameterSpec(
                "wall_thickness", default=0.15, min_val=0.10, max_val=0.20, step=0.05
            ),
        ],
        max_iterations=10,
        samples_per_param=4,
        trace_dir=None,
    )


def _fake_proc(returncode: int = 0, stdout: str = "", stderr: str = ""):
    """Mimics the shape of ``subprocess.CompletedProcess``."""
    obj = MagicMock()
    obj.returncode = returncode
    obj.stdout = stdout
    obj.stderr = stderr
    return obj


# ---------------------------------------------------------------------------
# parse_cargo_output — MAE/Pass-Rate regex (Issue #1847 Task 2 bullet 2).
# ---------------------------------------------------------------------------


def test_parse_cargo_output_simple_mae_header():
    out = """
test result: ok.
Mean Absolute Error: 4.21%
Case 600 : Heating=2.00 (Ref: 1.00-3.00), Cooling=1.00 (Ref: 0.50-1.50)
Pass Rate: 90.0% ... Passed: 18 ... Failed: 2
"""
    metrics = aps.parse_cargo_output(out)
    assert metrics["heating_mae"] == pytest.approx(4.21)
    assert metrics["overall_pass"] is True


def test_parse_cargo_output_no_match_yields_zero_metrics():
    metrics = aps.parse_cargo_output("nothing to see here")
    assert metrics == {
        "heating_mae": 0.0,
        "cooling_mae": 0.0,
        "peak_heating_mae": 0.0,
        "peak_cooling_mae": 0.0,
        "temperature_mae": 0.0,
        "overall_pass": False,
    }


def test_parse_cargo_output_low_pass_rate_marks_failure():
    out = "Mean Absolute Error: 5.0% Pass Rate: 60.0% Passed: 6 Failed: 4"
    metrics = aps.parse_cargo_output(out)
    assert metrics["overall_pass"] is False  # < 80% threshold


def test_parse_cargo_output_extracts_per_case_errors():
    out = """
Case 600 : Heating=2.00 (Ref: 1.00-3.00), Cooling=1.20 (Ref: 0.80-1.60)
Case 900 : Heating=5.10 (Ref: 4.50-5.50), Cooling=8.10 (Ref: 7.50-8.50)
"""
    metrics = aps.parse_cargo_output(out, case_filter="600")
    # heating_errors = |2.00-2.00|/2.00 = 0 (ref avg = 2.00, sim = 2.00)
    assert metrics["heating_mae"] == pytest.approx(0.0)
    # cooling: ref avg (0.80+1.60)/2 = 1.20, sim = 1.20, error = 0
    assert metrics["cooling_mae"] == pytest.approx(0.0)


def test_parse_cargo_output_case_filter_skips_non_matching():
    """``case_filter`` selects only the matching case; non-matching cases produce no errors."""
    out = """
Case 195 : Heating=0.50 (Ref: 0.00-1.00), Cooling=0.10 (Ref: -0.20-0.40)
Case 470 : Heating=2.00 (Ref: 1.00-3.00), Cooling=1.00 (Ref: 0.50-1.50)
"""
    metrics_195 = aps.parse_cargo_output(out, case_filter="195")
    metrics_470 = aps.parse_cargo_output(out, case_filter="470")
    # Different filter selects different case — heating midpoint for 195.
    assert metrics_195["heating_mae"] == pytest.approx(0.0)
    assert metrics_470["heating_mae"] == pytest.approx(0.0)
    assert metrics_195["cooling_mae"] == pytest.approx(0.0)
    assert metrics_470["cooling_mae"] == pytest.approx(0.0)


def test_parse_cargo_output_case_filter_only_counts_matching():
    """Filtering by ``"195"`` must aggregate errors only for the matching case."""
    out = """
Case 195 : Heating=1.00 (Ref: 0.00-2.00), Cooling=1.00 (Ref: 0.00-2.00)
Case 600 : Heating=5.00 (Ref: 1.00-3.00), Cooling=1.00 (Ref: 0.50-1.50)
"""
    metrics = aps.parse_cargo_output(out, case_filter="195")
    # Heating: ref avg = (0+2)/2 = 1.0, sim = 1.0, error = 0
    assert metrics["heating_mae"] == pytest.approx(0.0)
    # 600 should NOT have been aggregated (it's outside the filter).
    # Cooling for 195 also 0%. Total MAE stays 0.
    assert metrics["cooling_mae"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Parameter-space enumeration.
# ---------------------------------------------------------------------------


def test_generate_grid_points_grid_two_params():
    specs = [
        aps.ParameterSpec("a", default=1.0, min_val=0.0, max_val=2.0, step=1.0),
        aps.ParameterSpec("b", default=1.0, min_val=10.0, max_val=12.0, step=1.0),
    ]
    pts = aps.generate_grid_points(specs)
    assert len(pts) == 9
    assert pts[0] == {"a": 0.0, "b": 10.0}
    assert pts[-1] == {"a": 2.0, "b": 12.0}


def test_generate_random_points_respects_count_and_bounds():
    specs = [aps.ParameterSpec("x", default=0.0, min_val=-1.0, max_val=1.0, step=0.5)]
    points = aps.generate_random_points(specs, samples=128)
    assert len(points) == 128
    for p in points:
        assert -1.0 <= p["x"] <= 1.0


# ---------------------------------------------------------------------------
# Sweep state persistence — load_sweep_state, save_sweep_state.
# ---------------------------------------------------------------------------


def test_save_sweep_state_round_trip(tmp_path: Path):
    state = aps.SweepState(
        config={"case_id": "600"},
        results=[{"iteration": 1, "mae": 4.2}],
        best_parameters={"R_value": 2.5},
        best_mae=4.2,
        start_time="2026-01-01T00:00:00Z",
        status="running",
        iteration=1,
    )
    aps.save_sweep_state(tmp_path, state)
    loaded = aps.load_sweep_state(tmp_path)
    assert loaded is not None
    assert loaded.best_parameters == {"R_value": 2.5}
    assert loaded.best_mae == pytest.approx(4.2)
    assert loaded.iteration == 1
    assert loaded.status == "running"


def test_load_sweep_state_returns_none_when_missing(tmp_path: Path):
    assert aps.load_sweep_state(tmp_path) is None


def test_save_sweep_state_creates_directory(tmp_path: Path):
    nested = tmp_path / "deep" / "nested"
    state = aps.SweepState(
        config={},
        results=[],
        best_parameters={},
        best_mae=999.0,
        start_time="now",
        status="created",
        iteration=0,
    )
    aps.save_sweep_state(nested, state)
    assert (nested / "sweep_state.json").exists()


# ---------------------------------------------------------------------------
# run_sweep in --dry-run mode (avoids real cargo invocation).
# ---------------------------------------------------------------------------


def test_run_sweep_dry_run_writes_config_and_returns_state(
    sweep_config,
    temp_trace_dir,
):
    sweep_config.trace_dir = temp_trace_dir
    state = aps.run_sweep(sweep_config, dry_run=True)
    assert state.status == "running"
    assert state.iteration == 0
    assert (temp_trace_dir / "sweep_config.json").exists()
    cfg = json.loads((temp_trace_dir / "sweep_config.json").read_text())
    assert cfg["case_id"] == "600"
    assert cfg["max_iterations"] == 10


def test_run_sweep_dry_run_does_not_invoke_cargo(
    sweep_config,
    temp_trace_dir,
    fake_subprocess,
):
    sweep_config.trace_dir = temp_trace_dir
    # Configure subprocess.run to fail loudly if invoked (proves dry-run doesn't call it).
    fake_subprocess["install"](
        run_return=_pytest_fail("dry-run must not invoke subprocess"),
    )
    state = aps.run_sweep(sweep_config, dry_run=True)
    assert state is not None
    assert fake_subprocess["run_calls"] == []


# ---------------------------------------------------------------------------
# run_sweep random mode: fake a single cargo run with non-zero metrics.
# ---------------------------------------------------------------------------


def test_run_sweep_random_with_mocked_cargo(
    sweep_config,
    temp_trace_dir,
    fake_subprocess,
):
    sweep_config.sweep_type = aps.SweepType.RANDOM
    sweep_config.samples_per_param = 2
    sweep_config.max_iterations = 2
    sweep_config.trace_dir = temp_trace_dir
    # Tolerance far below the worst possible result (0% + 0% = 0%), so no early exit.
    sweep_config.tolerance_mae = 0.0

    fake_subprocess["install"](
        run_return=_fake_proc(
            0,
            stdout="Mean Absolute Error: 3.10%\n"
            "Case 600 : Heating=2.00 (Ref: 1.00-3.00), "
            "Cooling=1.20 (Ref: 0.80-1.60)\n"
            "Pass Rate: 92.0% ... Passed: 9 ... Failed: 1",
        )
    )

    state = aps.run_sweep(sweep_config, dry_run=False)
    # Both iterations run; with tolerance=0.0 the early-abort never triggers.
    assert state.iteration == 2
    assert len(state.results) == 2
    # log_result wrote JSONL
    jsonl = (
        (temp_trace_dir / "parameter_sweep_results.jsonl")
        .read_text()
        .strip()
        .splitlines()
    )
    assert len(jsonl) == 2
    first = json.loads(jsonl[0])
    assert first["case_id"] == "600"
    # convergence_log.csv exists
    conv = (temp_trace_dir / "convergence_log.csv").read_text().strip().splitlines()
    assert conv[0] == "iteration,mae,timestamp"
    assert len(conv) == 3  # header + 2 rows


def test_run_sweep_aborts_early_when_tolerance_met(
    sweep_config,
    temp_trace_dir,
    fake_subprocess,
):
    sweep_config.sweep_type = aps.SweepType.GRID
    sweep_config.max_iterations = 5
    sweep_config.tolerance_mae = 1.0  # tight threshold
    sweep_config.trace_dir = temp_trace_dir

    fake_subprocess["install"](
        run_return=_fake_proc(0, stdout="Mean Absolute Error: 0.50%\n")
    )

    state = aps.run_sweep(sweep_config, dry_run=False)
    # First iteration's heating+cooling = 0.50 <= 1.0 tolerance → abort.
    assert state.iteration == 1


def test_run_sweep_handles_cargo_timeout(
    sweep_config,
    temp_trace_dir,
    fake_subprocess,
):
    sweep_config.sweep_type = aps.SweepType.GRID
    sweep_config.max_iterations = 2
    sweep_config.tolerance_mae = 100.0
    sweep_config.trace_dir = temp_trace_dir

    fake_subprocess["install"](
        run_return=_pytest_fail(""),
    )
    # Force a TimeoutExpired-style return for our run:
    import subprocess as _subprocess

    fake_subprocess["install"](
        run_return=_raise_on_call(_subprocess.TimeoutExpired("cargo test", 60)),
    )

    state = aps.run_sweep(sweep_config, dry_run=False)
    # Each iteration should record an error result.
    assert all(r["error_message"] for r in state.results)
    assert state.status == "completed"


# ---------------------------------------------------------------------------
# apply_parameters — used to set env overrides.
# ---------------------------------------------------------------------------


def test_apply_parameters_writes_uppercase_env():
    env: dict[str, str] = {}
    aps.apply_parameters({"R_value": 1.5, "wall_thickness": 0.20}, env)
    assert env["FLUXION_PARAM_R_VALUE"] == "1.5"
    assert env["FLUXION_PARAM_WALL_THICKNESS"] == "0.2"


def test_apply_parameters_handles_empty_dict():
    env: dict[str, str] = {"existing": "x"}
    aps.apply_parameters({}, env)
    assert env == {"existing": "x"}


# ---------------------------------------------------------------------------
# log_result writes JSONL with one record per call.
# ---------------------------------------------------------------------------


def test_log_result_appends_jsonl(tmp_path: Path):
    trace = tmp_path / "trace"
    result = aps.SweepResult(
        run_id="abc12345",
        case_id="600",
        iteration=1,
        parameters={"R_value": 2.0},
        heating_mae=4.21,
        cooling_mae=3.10,
        peak_heating_mae=4.21,
        peak_cooling_mae=3.10,
        temperature_mae=4.21,
        overall_pass=True,
        timestamp="2026-01-01T00:00:00Z",
        duration_ms=1200,
    )
    aps.log_result(trace, result)
    aps.log_result(trace, result)
    lines = (trace / "parameter_sweep_results.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["run_id"] == "abc12345"


# ---------------------------------------------------------------------------
# log_convergence — CSV format with header creation on first call.
# ---------------------------------------------------------------------------


def test_log_convergence_writes_header_then_rows(tmp_path: Path):
    trace = tmp_path / "trace"
    aps.log_convergence(trace, 1, 4.21)
    aps.log_convergence(trace, 2, 3.10)
    rows = (trace / "convergence_log.csv").read_text().strip().splitlines()
    assert rows[0] == "iteration,mae,timestamp"
    assert rows[1].startswith("1,4.21,")
    assert rows[2].startswith("2,3.1,")


def test_log_convergence_does_not_duplicate_header(tmp_path: Path):
    trace = tmp_path / "trace"
    aps.log_convergence(trace, 1, 1.0)
    aps.log_convergence(trace, 2, 2.0)
    aps.log_convergence(trace, 3, 3.0)
    rows = (trace / "convergence_log.csv").read_text().strip().splitlines()
    assert sum(1 for r in rows if r.startswith("iteration")) == 1


# ---------------------------------------------------------------------------
# create_divergence_report renders best-pas markdown.
# ---------------------------------------------------------------------------


def test_create_divergence_report_emits_markdown(tmp_path: Path):
    trace = tmp_path / "trace"
    trace.mkdir(parents=True, exist_ok=True)  # ensure report can be written
    config = aps.SweepConfig(
        case_id="600",
        sweep_type=aps.SweepType.RANDOM,
        parameters=[
            aps.ParameterSpec(
                "R_value", default=2.0, min_val=1.0, max_val=5.0, step=0.5, unit="m²K/W"
            )
        ],
    )
    results = [
        aps.SweepResult(
            run_id="r1",
            case_id="600",
            iteration=1,
            parameters={"R_value": 2.0},
            heating_mae=4.0,
            cooling_mae=3.0,
            peak_heating_mae=4.0,
            peak_cooling_mae=3.0,
            temperature_mae=4.0,
            overall_pass=True,
            timestamp="t0",
            duration_ms=10,
        ),
        aps.SweepResult(
            run_id="r2",
            case_id="600",
            iteration=2,
            parameters={"R_value": 3.0},
            heating_mae=6.0,
            cooling_mae=5.0,
            peak_heating_mae=6.0,
            peak_cooling_mae=5.0,
            temperature_mae=6.0,
            overall_pass=False,
            timestamp="t0",
            duration_ms=10,
        ),
    ]
    aps.create_divergence_report(trace, config, results)
    text = (trace / "divergence_report.md").read_text()
    assert "Case:" in text and "600" in text
    assert "Pass Rate" in text
    assert "Best MAE" in text
    assert "50.0%" in text  # 1 of 2 passed
    assert "R_value" in text


def test_create_divergence_report_no_results_writes_nothing(tmp_path: Path):
    trace = tmp_path / "trace"
    trace.mkdir(parents=True, exist_ok=True)
    config = aps.SweepConfig(
        case_id="600",
        sweep_type=aps.SweepType.RANDOM,
        parameters=[
            aps.ParameterSpec("x", default=1.0, min_val=0.0, max_val=1.0, step=0.1)
        ],
    )
    aps.create_divergence_report(trace, config, [])
    assert not (trace / "divergence_report.md").exists()


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _pytest_fail(msg: str):
    """Wrap pytest.fail so the test process surfaces the assertion cleanly."""

    def _fail(*a, **kw):
        pytest.fail(msg)

    return _fail


def _raise_on_call(exc: BaseException):
    def _runner(*args, **kwargs):
        raise exc

    return _runner


# ---------------------------------------------------------------------------
# --brief YAML/JSON loader — Issue #2951.
# ---------------------------------------------------------------------------

YAML_BRIEF = """\
axes:
  - R_value
  - wall_thickness
ranges:
  R_value:
    min: 1.0
    max: 4.0
    step: 0.5
    default: 2.0
  wall_thickness:
    min: 0.05
    max: 0.25
    step: 0.05
objective: minimize_mae
seed: 42
"""

JSON_BRIEF = {
    "axes": ["thermal_mass", "h_tr_is"],
    "ranges": {
        "thermal_mass": {"min": 0.5, "max": 1.5, "step": 0.1, "default": 1.0},
        "h_tr_is": {"min": 5.0, "max": 12.0, "step": 1.0},
    },
    "objective": "minimize_peak_cooling",
    "seed": 7,
}


def test_brief_yaml_parsed(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """A well-formed YAML brief is parsed into a BriefSpec with all fields set."""
    path = tmp_path / "brief.yaml"
    path.write_text(YAML_BRIEF, encoding="utf-8")

    spec = aps.load_brief_spec(path)

    assert isinstance(spec, aps.BriefSpec)
    assert spec.axes == ["R_value", "wall_thickness"]
    assert spec.ranges["R_value"]["min"] == pytest.approx(1.0)
    assert spec.ranges["R_value"]["max"] == pytest.approx(4.0)
    assert spec.ranges["R_value"]["step"] == pytest.approx(0.5)
    assert spec.ranges["R_value"]["default"] == pytest.approx(2.0)
    assert spec.ranges["wall_thickness"]["min"] == pytest.approx(0.05)
    assert spec.ranges["wall_thickness"]["max"] == pytest.approx(0.25)
    assert spec.ranges["wall_thickness"]["step"] == pytest.approx(0.05)
    assert spec.objective == "minimize_mae"
    assert spec.seed == 42

    # Missing/malformed path does NOT emit a warning on success.
    captured = capsys.readouterr()
    assert "::warning::" not in captured.err


def test_brief_json_parsed(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """JSON briefs are accepted on any non-YAML extension."""
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(JSON_BRIEF), encoding="utf-8")

    spec = aps.load_brief_spec(path)

    assert spec.axes == ["thermal_mass", "h_tr_is"]
    assert spec.ranges["thermal_mass"]["min"] == pytest.approx(0.5)
    assert spec.ranges["thermal_mass"]["max"] == pytest.approx(1.5)
    assert spec.ranges["thermal_mass"]["step"] == pytest.approx(0.1)
    assert spec.ranges["thermal_mass"]["default"] == pytest.approx(1.0)
    # "default" key omitted in JSON for h_tr_is → not present in spec.
    assert "default" not in spec.ranges["h_tr_is"]
    assert spec.ranges["h_tr_is"]["step"] == pytest.approx(1.0)
    assert spec.objective == "minimize_peak_cooling"
    assert spec.seed == 7

    captured = capsys.readouterr()
    assert "::warning::" not in captured.err


def test_brief_missing_falls_back_to_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """A nonexistent brief path returns DEFAULT_BRIEF_SPEC and emits a warning."""
    missing = tmp_path / "does_not_exist.yaml"
    assert not missing.exists()

    spec = aps.load_brief_spec(missing)

    # Spec is the documented default — Issue #2951 acceptance: "missing → default".
    assert spec.axes == aps.DEFAULT_BRIEF_SPEC.axes
    assert spec.ranges == aps.DEFAULT_BRIEF_SPEC.ranges
    assert spec.objective == aps.DEFAULT_BRIEF_SPEC.objective
    assert spec.seed is None

    captured = capsys.readouterr()
    assert "::warning::" in captured.err
    assert "not found" in captured.err.lower()


def test_brief_malformed_falls_back_to_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """A malformed YAML/JSON brief returns defaults and emits a warning."""
    path = tmp_path / "broken.yaml"
    path.write_text("{ this is: not valid yaml: [}", encoding="utf-8")

    spec = aps.load_brief_spec(path)

    assert spec.axes == aps.DEFAULT_BRIEF_SPEC.axes
    assert spec.ranges == aps.DEFAULT_BRIEF_SPEC.ranges
    assert spec.objective == aps.DEFAULT_BRIEF_SPEC.objective
    assert spec.seed is None

    captured = capsys.readouterr()
    assert "::warning::" in captured.err


def test_brief_none_returns_default_silently(capsys: pytest.CaptureFixture[str]):
    """``load_brief_spec(None)`` returns the default brief without warning.

    This guards the ``--brief`` not-provided path so a missing CLI flag is
    indistinguishable from a deliberately-empty brief.
    """
    spec = aps.load_brief_spec(None)
    assert spec == aps.DEFAULT_BRIEF_SPEC
    captured = capsys.readouterr()
    assert captured.err == ""


def test_brief_to_parameter_specs_uses_brief_ranges():
    """``brief_to_parameter_specs`` honors brief-supplied bounds over defaults."""
    brief = aps.BriefSpec(
        axes=["R_value"],
        ranges={"R_value": {"min": 1.5, "max": 3.5, "step": 0.25, "default": 2.0}},
    )
    default_params = {
        "R_value": aps.ParameterSpec(
            "R_value", default=2.0, min_val=1.0, max_val=5.0, step=0.5, unit="m²K/W"
        )
    }

    specs = aps.brief_to_parameter_specs(brief, default_params)

    assert len(specs) == 1
    assert specs[0].name == "R_value"
    assert specs[0].min_val == pytest.approx(1.5)
    assert specs[0].max_val == pytest.approx(3.5)
    assert specs[0].step == pytest.approx(0.25)
    assert specs[0].default == pytest.approx(2.0)
    # Unit is inherited from the legacy default_params table.
    assert specs[0].unit == "m²K/W"


def test_build_parser_exposes_brief_flag():
    """``--brief`` is registered and parses a Path."""
    parser = aps.build_parser()
    args = parser.parse_args(
        ["--case", "600", "--brief", "/tmp/opencode/some_brief.yaml"]
    )
    assert args.brief == Path("/tmp/opencode/some_brief.yaml")

    # Default when not provided is None.
    args_default = parser.parse_args(["--case", "600"])
    assert args_default.brief is None
