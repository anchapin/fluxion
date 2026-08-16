"""
Tests for ``scripts/release_gate_checker.py`` -- Issue #505.

Regression guard for the release-gate evaluator. The script is the
release-time consumer of the gate budgets declared in
``release_gates.yaml``; the per-method ``check_*`` logic on
``ReleaseGateChecker`` is the load-bearing piece (the CLI + IO is thin).

Pattern: load the script as a fresh module via the shared
``load_script`` fixture, construct a ``ReleaseGateChecker`` with a
synthetic config dict, then drive each ``check_*`` method through
clean / violation scenarios. The CLI is exercised end-to-end with a
synthetic ``release_gates.yaml`` + ``validation_results.json`` planted
in ``tmp_path``.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

SCRIPT_NAME = "release_gate_checker"


@pytest.fixture
def checker(load_script):
    """Freshly-loaded copy of the release-gate checker."""
    return load_script(SCRIPT_NAME)


def _make_checker(checker, project_root: Path, config: dict):
    """Construct a ``ReleaseGateChecker`` against ``project_root``."""
    return checker.ReleaseGateChecker(config, project_root)


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------


_MINIMAL_VALIDATION_CONFIG = {
    "validation": {
        "min_pass_rate": 60.0,
        "max_mae": 50.0,
        "individual": {
            "max_deviation": 100.0,
            "extreme_deviation_limit": 2,
            "known_failures": ["600", "900"],
        },
    }
}


def _results(
    pass_rate: float = 50.0, mae: float = 30.0, cases: dict | None = None
) -> dict:
    """Build a synthetic ``validation_results`` payload."""
    return {
        "summary": {"pass_rate": pass_rate, "mae": mae},
        "cases": cases or {},
    }


def test_check_validation_gates_passes_when_pass_rate_above_floor(checker, tmp_path):
    """Pass rate 70% with min 60% → overall pass."""
    rg = _make_checker(checker, tmp_path, _MINIMAL_VALIDATION_CONFIG)
    results = rg.check_validation_gates(_results(pass_rate=70.0))
    by_name = {r.name: r for r in results}
    assert by_name["overall_pass_rate"].passed is True
    assert by_name["overall_pass_rate"].value == 70.0
    assert by_name["overall_pass_rate"].threshold == 60.0


def test_check_validation_gates_fails_when_pass_rate_below_floor(checker, tmp_path):
    """Pass rate 40% with min 60% → overall fail."""
    rg = _make_checker(checker, tmp_path, _MINIMAL_VALIDATION_CONFIG)
    results = rg.check_validation_gates(_results(pass_rate=40.0))
    by_name = {r.name: r for r in results}
    assert by_name["overall_pass_rate"].passed is False


def test_check_validation_gates_fails_when_mae_above_limit(checker, tmp_path):
    """MAE 60% with max 50% → MAE gate fails."""
    rg = _make_checker(checker, tmp_path, _MINIMAL_VALIDATION_CONFIG)
    results = rg.check_validation_gates(_results(pass_rate=70.0, mae=60.0))
    by_name = {r.name: r for r in results}
    assert by_name["max_mae"].passed is False


def test_check_validation_gates_excludes_known_failures(checker, tmp_path):
    """Known failures don't count toward ``extreme_deviations``."""
    rg = _make_checker(checker, tmp_path, _MINIMAL_VALIDATION_CONFIG)
    # Case 600 (known_fail) is wildly off; case 700 is in-band.
    cases = {
        "600": {
            "heating": 100.0,
            "heating_min": 4.0,
            "heating_max": 6.0,
            "cooling": 100.0,
            "cooling_min": 4.0,
            "cooling_max": 6.0,
        },
        "700": {
            "heating": 5.0,
            "heating_min": 4.0,
            "heating_max": 6.0,
            "cooling": 5.0,
            "cooling_min": 4.0,
            "cooling_max": 6.0,
        },
    }
    results = rg.check_validation_gates(_results(cases=cases))
    by_name = {r.name: r for r in results}
    # 600 is in known_failures -> excluded; 700 is in-band -> 0 extremes.
    assert by_name["extreme_deviations"].passed is True
    assert by_name["extreme_deviations"].value == 0
    assert "600" in by_name["extreme_deviations"].details["known_failures"]


def test_check_validation_gates_counts_extreme_cases(checker, tmp_path):
    """Non-known-fail cases that exceed max_deviation increment extreme_count."""
    rg = _make_checker(checker, tmp_path, _MINIMAL_VALIDATION_CONFIG)
    # extreme_deviation_limit = 2; plant 3 wildly off cases, none known.
    cases = {}
    for cid in ("100", "200", "300"):
        cases[cid] = {
            "heating": 100.0,
            "heating_min": 4.0,
            "heating_max": 6.0,
            "cooling": 5.0,
            "cooling_min": 4.0,
            "cooling_max": 6.0,
        }
    results = rg.check_validation_gates(_results(cases=cases))
    by_name = {r.name: r for r in results}
    assert by_name["extreme_deviations"].passed is False
    assert by_name["extreme_deviations"].value == 3


# ---------------------------------------------------------------------------
# Benchmark gates
# ---------------------------------------------------------------------------


_MINIMAL_BENCHMARK_CONFIG = {
    "benchmark": {
        "throughput": {"min_configs_per_sec": 150},
        "latency": {"max_ms_per_config": 10.0},
        "multi_zone": {"min_configs_per_sec": 10},
        "hybrid": {"min_configs_per_sec": 0},
        "hybrid_multi_zone": {"min_configs_per_sec": 0},
        "cross_validation": {"max_ms": 500},
        "absolute_min_throughput": 100,
    }
}


def _bench(
    throughput: float = 200.0,
    latency: float = 5.0,
    multi_zone: float = 20.0,
    hybrid: float = 0.0,
    hybrid_multi_zone: float = 0.0,
    cv_latency: float = 100.0,
    cold_start_ms: float = 0.0,
    warm_ms: float = 0.0,
    cold_warm_ratio: float = 0.0,
) -> dict:
    return {
        "metrics": {
            "throughput_configs_per_sec": throughput,
            "latency_ms_per_config": latency,
            "multi_zone_throughput": multi_zone,
            "hybrid_throughput": hybrid,
            "hybrid_multi_zone_throughput": hybrid_multi_zone,
            "cross_validation_latency_ms": cv_latency,
            "cold_start_ms": cold_start_ms,
            "warm_steady_state_ms": warm_ms,
            "cold_warm_ratio": cold_warm_ratio,
        }
    }


def test_check_benchmark_gates_passes_all_within_budget(checker, tmp_path):
    """All measured metrics above min / below max → all pass."""
    rg = _make_checker(checker, tmp_path, _MINIMAL_BENCHMARK_CONFIG)
    results = rg.check_benchmark_gates(_bench())
    by_name = {r.name: r for r in results}
    for r in results:
        assert r.passed, f"{r.name} unexpectedly failed: {r.message}"
    assert by_name["throughput"].passed is True
    assert by_name["latency"].passed is True


def test_check_benchmark_gates_fails_throughput_below_absolute_min(checker, tmp_path):
    """Throughput below absolute_min_throughput → throughput fails."""
    rg = _make_checker(checker, tmp_path, _MINIMAL_BENCHMARK_CONFIG)
    # min=150, abs_min=100 → 50 violates both.
    results = rg.check_benchmark_gates(_bench(throughput=50.0))
    by_name = {r.name: r for r in results}
    assert by_name["throughput"].passed is False
    assert "abs min: 100" in by_name["throughput"].message


def test_check_benchmark_gates_fails_latency_above_max(checker, tmp_path):
    """Latency 25 ms with max 10 ms → latency fails."""
    rg = _make_checker(checker, tmp_path, _MINIMAL_BENCHMARK_CONFIG)
    results = rg.check_benchmark_gates(_bench(latency=25.0))
    by_name = {r.name: r for r in results}
    assert by_name["latency"].passed is False


def test_check_benchmark_gates_passes_hybrid_when_floor_is_zero(checker, tmp_path):
    """Hybrid throughput with min_configs_per_sec=0 → pass-through (no floor)."""
    rg = _make_checker(checker, tmp_path, _MINIMAL_BENCHMARK_CONFIG)
    results = rg.check_benchmark_gates(_bench(hybrid=42.0))
    by_name = {r.name: r for r in results}
    assert by_name["hybrid_throughput"].passed is True


def test_check_benchmark_gates_gate_filter_restricts_results(checker, tmp_path):
    """``gate_filter={'throughput', 'latency'}`` skips multi_zone/cv/etc.

    Issue #2693 pattern: PR jobs evaluate just the absolute throughput
    + latency floors without multi-zone / cross-validation data.
    """
    rg = _make_checker(checker, tmp_path, _MINIMAL_BENCHMARK_CONFIG)
    results = rg.check_benchmark_gates(
        _bench(throughput=50.0),  # forces a throughput FAIL
        gate_filter={"throughput", "latency"},
    )
    names = {r.name for r in results}
    assert names == {"throughput", "latency"}
    # Multi-zone/cv/etc absent (would otherwise spuriously fail on
    # default 0 measurements).
    assert "multi_zone_throughput" not in names
    assert "cross_validation_latency" not in names


# ---------------------------------------------------------------------------
# Crate-size gates (Issue #2930)
# ---------------------------------------------------------------------------


def test_check_crate_size_gates_passes_when_under_limit(checker, tmp_path):
    """size_mb < max_mb → passed."""
    config = {"crate_size": {"max_mb": 10.0}}
    rg = _make_checker(checker, tmp_path, config)
    results = rg.check_crate_size_gates(
        {"size_bytes": 5 * 1024 * 1024, "size_mb": 5.0, "crate_path": "/x"}
    )
    assert len(results) == 1
    assert results[0].passed is True
    assert "within" in results[0].message


def test_check_crate_size_gates_fails_when_over_limit(checker, tmp_path):
    """size_mb > max_mb → passed=False with over-limit message."""
    config = {"crate_size": {"max_mb": 10.0}}
    rg = _make_checker(checker, tmp_path, config)
    results = rg.check_crate_size_gates(
        {"size_bytes": 12 * 1024 * 1024, "size_mb": 12.0, "crate_path": "/x"}
    )
    assert len(results) == 1
    assert results[0].passed is False
    assert "exceeds" in results[0].message


def test_check_crate_size_gates_returns_one_when_no_measurement(checker, tmp_path):
    """``crate_size_results=None`` → single FAIL with remediation."""
    config = {"crate_size": {"max_mb": 10.0}}
    rg = _make_checker(checker, tmp_path, config)
    results = rg.check_crate_size_gates(None)
    assert len(results) == 1
    assert results[0].passed is False
    assert "No crate-size measurement" in results[0].message


# ---------------------------------------------------------------------------
# Drift gates
# ---------------------------------------------------------------------------


_DRIFT_CONFIG = {
    "drift": {
        "enabled": True,
        "max_pass_rate_change": 2.0,
        "max_mae_change": 5.0,
        "max_pass_to_fail": 1,
        "max_fail_to_pass": 5,
    }
}


def test_check_drift_gates_passes_when_within_tolerance(checker, tmp_path):
    """Drift within ±2 pp pass-rate and ±5 pp MAE → all pass."""
    rg = _make_checker(checker, tmp_path, _DRIFT_CONFIG)
    current = {
        "summary": {"pass_rate": 50.0, "mae": 30.0},
        "cases": {},
    }
    baseline = {
        "summary": {"pass_rate": 51.0, "mae": 31.0},
        "cases": {},
    }
    results = rg.check_drift_gates(current, baseline)
    by_name = {r.name: r for r in results}
    assert by_name["pass_rate_drift"].passed is True
    assert by_name["mae_drift"].passed is True


def test_check_drift_gates_fails_when_pass_rate_drifts(checker, tmp_path):
    """Pass rate dropped 5 pp with max 2 pp → fail."""
    rg = _make_checker(checker, tmp_path, _DRIFT_CONFIG)
    current = {"summary": {"pass_rate": 45.0, "mae": 30.0}, "cases": {}}
    baseline = {"summary": {"pass_rate": 50.0, "mae": 30.0}, "cases": {}}
    results = rg.check_drift_gates(current, baseline)
    by_name = {r.name: r for r in results}
    assert by_name["pass_rate_drift"].passed is False


def test_check_drift_gates_handles_missing_baseline(checker, tmp_path):
    """No baseline + create_baseline_if_missing=False → baseline gate fails."""
    config = {
        **_DRIFT_CONFIG,
        "drift": {**_DRIFT_CONFIG["drift"], "create_baseline_if_missing": False},
    }
    rg = _make_checker(checker, tmp_path, config)
    current = {"summary": {"pass_rate": 50.0, "mae": 30.0}, "cases": {}}
    results = rg.check_drift_gates(current, baseline=None)
    by_name = {r.name: r for r in results}
    assert by_name["baseline"].passed is False


def test_check_drift_gates_disabled_returns_pass(checker, tmp_path):
    """``drift.enabled=False`` → single PASS gate, no baseline required."""
    config = {"drift": {"enabled": False}}
    rg = _make_checker(checker, tmp_path, config)
    results = rg.check_drift_gates({"summary": {}, "cases": {}}, baseline=None)
    assert len(results) == 1
    assert results[0].passed is True
    assert "disabled" in results[0].message.lower()


# ---------------------------------------------------------------------------
# Issue #2856: drift.baseline_file must exist on disk and stay fresh.
# ---------------------------------------------------------------------------
# Issue #2856: release_gates.yaml:drift.baseline_file pointed at a
# non-existent ``validation_baseline.json``. ``_load_baseline()`` silently
# returned ``None`` and the gate fell through ``create_baseline_if_missing:
# true`` with a PASS — drift detection was structurally blind. The fix
# commits the baseline file at the declared path; the regression tests
# below pin that contract and fail CI if anyone re-points the YAML to a
# missing or stale file. See ARCHITECTURE.md / release_gates.yaml for the
# canonical config; do NOT relax these tests without re-reading #2856.


def test_drift_baseline_file_path_resolves_in_release_gates_yaml(repo_root):
    """``release_gates.yaml:drift.baseline_file`` points at an existing file.

    Issue #2856 acceptance #1: the file at the YAML-declared path must
    exist on disk. Without this the drift gate short-circuits to a
    silent PASS via ``create_baseline_if_missing`` and never compares
    current results against history.
    """
    yaml_path = repo_root / "release_gates.yaml"
    assert yaml_path.is_file(), f"release_gates.yaml missing at {yaml_path}"

    with open(yaml_path) as f:
        yaml_config = yaml.safe_load(f)

    baseline_relpath = yaml_config.get("drift", {}).get("baseline_file", "").strip()
    assert baseline_relpath, (
        "release_gates.yaml:drift.baseline_file is empty or missing — "
        "the drift gate cannot function without a baseline reference."
    )

    baseline_path = repo_root / baseline_relpath
    assert baseline_path.is_file(), (
        f"Drift gate baseline file {baseline_path} (declared at "
        f"release_gates.yaml:drift.baseline_file) does NOT exist on disk. "
        "Issue #2856: this regresses the drift gate to a silent PASS via "
        "`create_baseline_if_missing: true`. Either commit the baseline "
        "file or repoint `drift.baseline_file` to an existing reference."
    )


def test_drift_baseline_file_has_required_schema(checker, repo_root):
    """The baseline JSON must parse and expose ``summary`` + ``cases``.

    Issue #2856 acceptance #2: ``check_drift_gates`` reads
    ``baseline['summary']`` for pass_rate/mae and ``baseline['cases']``
    for per-case pass/fail transitions. A baseline that parses as JSON
    but lacks those keys (e.g. a free-form ``_doc`` blob) would still
    short-circuit the drift logic and silently green the gate.
    """
    yaml_path = repo_root / "release_gates.yaml"
    with open(yaml_path) as f:
        yaml_config = yaml.safe_load(f)
    baseline_path = repo_root / yaml_config["drift"]["baseline_file"]
    assert baseline_path.is_file(), (
        "Pre-condition for schema test: baseline must exist. "
        "See test_drift_baseline_file_path_resolves_in_release_gates_yaml."
    )

    with open(baseline_path) as f:
        baseline = json.load(f)

    assert "summary" in baseline, (
        f"{baseline_path.name} must have a top-level `summary` block — "
        "check_drift_gates reads summary.pass_rate / summary.mae."
    )
    assert "cases" in baseline, (
        f"{baseline_path.name} must have a top-level `cases` dict — "
        "check_drift_gates reads it for pass_to_fail / fail_to_pass "
        "transition counts."
    )
    summary = baseline["summary"]
    assert "pass_rate" in summary, (
        f"{baseline_path.name}: `summary.pass_rate` is required for the "
        "pass_rate_drift gate."
    )
    assert "mae" in summary, (
        f"{baseline_path.name}: `summary.mae` is required for the " "mae_drift gate."
    )


def test_drift_baseline_file_is_fresh(checker, repo_root):
    """``captured_at`` must be within ``_stale_threshold_days`` of today.

    Issue #2856 acceptance #3: a baseline file that exists but is
    years out of date is structurally as bad as a missing one — the
    drift gate would silently PASS for any current run that happens
    to be within the recorded band. Mirrors the
    ``KNOWN_ISSUES.md`` 60-day stale check pattern
    (``scripts/check_known_issues_stale.py``).
    """
    yaml_path = repo_root / "release_gates.yaml"
    with open(yaml_path) as f:
        yaml_config = yaml.safe_load(f)
    baseline_path = repo_root / yaml_config["drift"]["baseline_file"]
    assert baseline_path.is_file(), (
        "Pre-condition: baseline must exist. See "
        "test_drift_baseline_file_path_resolves_in_release_gates_yaml."
    )

    with open(baseline_path) as f:
        baseline = json.load(f)

    captured_at_str = baseline.get("captured_at")
    assert captured_at_str, (
        f"{baseline_path.name} is missing `captured_at` — the file "
        "has no staleness signal. Add an ISO-8601 timestamp when the "
        "baseline is captured / refreshed."
    )

    # Parse ISO-8601 (allow trailing 'Z' or numeric offset).
    captured_at = datetime.fromisoformat(captured_at_str.replace("Z", "+00:00"))
    # Truncate to day boundary so a same-day capture does not race the
    # threshold (file stamped at 18:00 should not be considered 0.25
    # days old by a 06:00 CI run).
    captured_day = captured_at.date()
    threshold_days = int(baseline.get("_stale_threshold_days", 90))
    age_days = (date.today() - captured_day).days

    assert age_days <= threshold_days, (
        f"{baseline_path.name} `captured_at` {captured_at_str} is "
        f"{age_days} days old (threshold: {threshold_days} days). "
        "The drift gate has gone stale — refresh the baseline by "
        "running `cargo run --bin fluxion -- validate` and committing "
        "the updated validation_results.json / docs/ASHRAE140_RESULTS.md, "
        "then re-run `python3 scripts/release_gate_checker.py "
        "--update-baseline` to overwrite this file. See issue #2856."
    )


def test_drift_baseline_loads_via_release_gate_checker(checker, repo_root):
    """End-to-end: ``_load_baseline()`` returns a non-None dict from the real config.

    Pins the public loader contract against the real
    ``release_gates.yaml`` (not a tmp_path fixture) so a regression
    where the YAML-declared file goes missing or the path is broken
    is caught at the script's own entry point rather than only in CI.
    """
    yaml_path = repo_root / "release_gates.yaml"
    with open(yaml_path) as f:
        yaml_config = yaml.safe_load(f)
    drift_config = yaml_config.get("drift", {})
    assert drift_config.get("enabled", True), (
        "drift.enabled is False in release_gates.yaml — this test "
        "only applies when the drift gate is active."
    )

    rg = _make_checker(checker, repo_root, yaml_config)
    baseline = rg._load_baseline()
    assert baseline is not None, (
        "ReleaseGateChecker._load_baseline() returned None against the "
        "real repo_root + release_gates.yaml — the drift gate is "
        "structurally blind. Issue #2856."
    )
    assert isinstance(baseline, dict)
    assert "summary" in baseline and "cases" in baseline


def test_drift_gates_fire_against_committed_baseline(checker, repo_root):
    """Synthetic -5pp pass-rate regression vs committed baseline FAILS the gate.

    Closes the loop on issue #2856: before the fix, the drift gate
    silently PASSED because the baseline file was missing. After
    committing ``validation_baseline.json`` with a 14.3% pass-rate,
    any synthetic current run with pass_rate below 9.3% (i.e.
    baseline 14.3% − 2pp drift floor = 12.3%, plus the 5pp drop here)
    MUST trip the pass_rate_drift sub-gate. This is the regression
    test that proves the gate is wired up end-to-end.
    """
    yaml_path = repo_root / "release_gates.yaml"
    with open(yaml_path) as f:
        yaml_config = yaml.safe_load(f)

    rg = _make_checker(checker, repo_root, yaml_config)
    baseline = rg._load_baseline()
    assert baseline is not None, (
        "Pre-condition: baseline must load. See "
        "test_drift_baseline_loads_via_release_gate_checker."
    )

    current_summary = dict(baseline["summary"])
    # Drop pass_rate by 5pp — well past the default 2pp drift floor.
    current_summary["pass_rate"] = current_summary["pass_rate"] - 5.0
    current = {"summary": current_summary, "cases": dict(baseline["cases"])}

    results = rg.check_drift_gates(current, baseline)
    by_name = {r.name: r for r in results}
    assert by_name["pass_rate_drift"].passed is False, (
        "pass_rate_drift unexpectedly PASSED with a -5pp drop — the "
        "drift gate is not wired up to the committed baseline. "
        "Issue #2856."
    )


# ---------------------------------------------------------------------------
# Helper loaders
# ---------------------------------------------------------------------------


def test_load_validation_results_reads_json(checker, tmp_path):
    """Plant a ``validation_results.json`` → loader returns it."""
    payload = {"summary": {"pass_rate": 99.0}, "cases": {}}
    (tmp_path / "validation_results.json").write_text(json.dumps(payload))
    loaded = checker.load_validation_results(tmp_path)
    assert loaded == payload


def test_load_benchmark_results_reads_json(checker, tmp_path):
    """Plant a ``benchmark_results.json`` → loader returns it."""
    payload = {"metrics": {"throughput_configs_per_sec": 200.0}}
    (tmp_path / "benchmark_results.json").write_text(json.dumps(payload))
    loaded = checker.load_benchmark_results(tmp_path)
    assert loaded == payload


def test_load_crate_size_results_returns_size_dict(checker, tmp_path):
    """Plant a fake ``.crate`` file → loader returns size info."""
    target_dir = tmp_path / "target" / "package"
    target_dir.mkdir(parents=True)
    crate = target_dir / "fluxion-0.0.0.crate"
    crate.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MiB

    loaded = checker.load_crate_size_results(tmp_path)
    assert loaded is not None
    assert loaded["size_bytes"] == 2 * 1024 * 1024
    assert loaded["size_mb"] == pytest.approx(2.0, rel=1e-3)
    assert str(crate.resolve()) == loaded["crate_path"]


def test_load_crate_size_results_returns_none_when_missing(checker, tmp_path):
    """No ``target/package/fluxion-*.crate`` → ``None``."""
    assert checker.load_crate_size_results(tmp_path) is None


def test_load_crate_size_results_honors_explicit_path(checker, tmp_path):
    """``explicit_path`` overrides glob lookup."""
    crate = tmp_path / "external.crate"
    crate.write_bytes(b"y" * 1024)  # 1 KiB
    loaded = checker.load_crate_size_results(tmp_path, explicit_path=crate)
    assert loaded is not None
    assert loaded["size_bytes"] == 1024


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def test_generate_markdown_report_contains_pass_fail(checker, tmp_path):
    """Report includes overall status and per-gate details."""
    from datetime import datetime, timezone

    config = _MINIMAL_VALIDATION_CONFIG
    rg = _make_checker(checker, tmp_path, config)
    results = rg.check_validation_gates(_results(pass_rate=70.0))
    report = checker.GateReport(
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        overall_passed=all(r.passed for r in results),
        gates=results,
        summary={
            "total": len(results),
            "passed": len(results),
            "failed": 0,
            "by_category": {
                "validation": {
                    "passed": len(results),
                    "failed": 0,
                    "total": len(results),
                }
            },
        },
    )
    md = checker.generate_markdown_report(report)
    assert "# Release Gate Status" in md
    assert "PASSED" in md
    assert "Overall" in md


# ---------------------------------------------------------------------------
# CLI: main() end-to-end with synthetic config + results
# ---------------------------------------------------------------------------


def test_main_returns_zero_on_clean_config(checker, tmp_path, monkeypatch, capsys):
    """End-to-end: synthetic config + passing results → exit 0.

    The script reads ``project_root = Path(__file__).parent.parent`` and
    then resolves ``release_gates.yaml`` from that, so we monkey-patch
    the YAML loader to inject a synthetic config and pass
    ``--validation-results`` pointing at a tmp_path JSON file.
    """
    config = {
        "validation": {
            "min_pass_rate": 50.0,
            "max_mae": 50.0,
            "individual": {
                "max_deviation": 150.0,
                "extreme_deviation_limit": 5,
                "known_failures": [],
            },
        },
        "ci": {"fail_fast": True},
        "drift": {"enabled": False},
    }
    validation_path = tmp_path / "validation_results.json"
    validation_path.write_text(
        json.dumps(
            {
                "summary": {"pass_rate": 75.0, "mae": 30.0},
                "cases": {},
            }
        )
    )

    def fake_yaml_load(stream):
        if hasattr(stream, "name") and "release_gates.yaml" in str(stream.name):
            return config
        import yaml as _yaml  # type: ignore

        return _yaml.safe_load(stream)

    monkeypatch.setattr(checker.yaml, "safe_load", fake_yaml_load)
    monkeypatch.setattr(checker, "load_crate_size_results", lambda *a, **kw: None)

    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME, "--validation-results", str(validation_path)]
    try:
        rc = checker.main()
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 0
    finally:
        sys.argv[:] = saved

    out = capsys.readouterr().out
    assert rc == 0, f"expected exit 0, got {rc}\noutput:\n{out}"
    assert "PASSED" in out


def test_main_returns_one_when_validation_fails(checker, tmp_path, monkeypatch, capsys):
    """End-to-end: failing validation result → exit 1.

    Mirrors the redirect pattern of
    :func:`test_main_returns_zero_on_clean_config` — the script reads
    ``release_gates.yaml`` from its own ``project_root`` (the real repo),
    so we monkey-patch the YAML loader to inject the failing fixture.
    """
    config = {
        "validation": {
            "min_pass_rate": 90.0,
            "max_mae": 50.0,
            "individual": {
                "max_deviation": 150.0,
                "extreme_deviation_limit": 5,
                "known_failures": [],
            },
        },
        "ci": {"fail_fast": True},
        "drift": {"enabled": False},
    }
    validation_path = tmp_path / "validation_results.json"
    validation_path.write_text(
        json.dumps(
            {
                "summary": {"pass_rate": 50.0, "mae": 30.0},
                "cases": {},
            }
        )
    )

    def fake_yaml_load(stream):
        if hasattr(stream, "name") and "release_gates.yaml" in str(stream.name):
            return config
        import yaml as _yaml  # type: ignore

        return _yaml.safe_load(stream)

    monkeypatch.setattr(checker.yaml, "safe_load", fake_yaml_load)
    monkeypatch.setattr(checker, "load_crate_size_results", lambda *a, **kw: None)

    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME, "--validation-results", str(validation_path)]
    try:
        rc = checker.main()
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 0
    finally:
        sys.argv[:] = saved

    assert rc == 1


# ---------------------------------------------------------------------------
# Issue #2865: DEFAULTS dict must stay in sync with release_gates.yaml
# ---------------------------------------------------------------------------


def _collect_leaf_paths(tree: dict, prefix: str = "") -> list[tuple[str, object]]:
    """Walk a nested dict and return ``(dotted_path, leaf_value)`` tuples.

    Used by the drift test to enumerate every leaf key in ``DEFAULTS``
    and assert it has a matching leaf in ``release_gates.yaml``.
    """
    leaves: list[tuple[str, object]] = []
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            leaves.extend(_collect_leaf_paths(value, path))
        else:
            leaves.append((path, value))
    return leaves


def _walk_yaml(yaml_tree: dict, dotted_path: str) -> object:
    """Resolve ``dotted_path`` (``"a.b.c"``) against a nested YAML dict.

    Returns the leaf value if every segment exists, otherwise
    ``_MISSING`` (a sentinel) so the test can distinguish "key absent"
    from "key present but value differs".
    """
    node: object = yaml_tree
    for segment in dotted_path.split("."):
        if not isinstance(node, dict) or segment not in node:
            return _MISSING
        node = node[segment]
    return node


_MISSING = object()  # sentinel distinct from any real YAML value


def test_defaults_match_release_gates_yaml(checker, repo_root):
    """Every DEFAULTS leaf must equal the corresponding YAML leaf.

    Issue #2865 acceptance criterion: the script used to embed hard-
    coded fallback values (e.g. ``min_pass_rate=4.0``, ``min_configs_
    per_sec=800``, ``max_deviation=150``, ``extreme_deviation_limit=
    15``) that diverged from ``release_gates.yaml``. If a YAML key was
    renamed, the script silently fell back to the footgun literal with
    no warning — pass rate would still report "PASSED" at 4% pass rate
    (the literal) instead of 60% (the YAML). Hoisting every fallback
    into the single ``DEFAULTS`` dict makes the drift visible: this
    test loads the real ``release_gates.yaml`` and compares every leaf
    in ``DEFAULTS`` against it. A mismatch on either side (DEFAULTS
    leaf missing from YAML, or value drift) fails the test, blocking
    any PR that lets the two diverge again.

    The list ``KNOWN_YAML_EXTRAS`` lets the test tolerate YAML keys
    that legitimately exist in the config but are not gate thresholds
    (``description``, ``zones``, ``baseline_file``, ``triggered_by``,
    ``required_checks``, ``workflow_index``, ``release_requirements``,
    ``create_baseline_if_missing``, ``crate_glob``, ...) — those are
    metadata / path / policy values, not the numeric thresholds the
    script reads. Conversely the test does NOT tolerate DEFAULTS keys
    missing from the YAML: every DEFAULTS leaf is a threshold the
    script reads, so if YAML drops the key the script would fall back
    to the DEFAULTS value with no warning — exactly the regression
    this test exists to catch.
    """
    yaml_path = repo_root / "release_gates.yaml"
    assert yaml_path.is_file(), f"release_gates.yaml missing at {yaml_path}"

    with open(yaml_path) as f:
        yaml_config = yaml.safe_load(f)
    assert isinstance(yaml_config, dict), "release_gates.yaml must be a YAML mapping"

    defaults = checker.DEFAULTS
    assert isinstance(defaults, dict), "DEFAULTS must be a module-level dict"

    # Every DEFAULTS leaf must exist in the YAML with the same value.
    mismatches: list[str] = []
    missing_from_yaml: list[str] = []

    for dotted_path, default_value in _collect_leaf_paths(defaults):
        yaml_value = _walk_yaml(yaml_config, dotted_path)
        if yaml_value is _MISSING:
            missing_from_yaml.append(f"{dotted_path} (DEFAULTS={default_value!r})")
            continue
        if yaml_value != default_value:
            mismatches.append(
                f"{dotted_path}: DEFAULTS={default_value!r} vs YAML={yaml_value!r}"
            )

    assert not missing_from_yaml, (
        "DEFAULTS contains leaf keys not present in release_gates.yaml — "
        "every DEFAULTS leaf is a threshold the script reads, so YAML must "
        "declare them. Update release_gates.yaml or trim DEFAULTS:\n  - "
        + "\n  - ".join(missing_from_yaml)
    )
    assert not mismatches, (
        "DEFAULTS values diverge from release_gates.yaml — update DEFAULTS "
        "(scripts/release_gate_checker.py) to match the YAML, or update the "
        "YAML, so they stay in sync:\n  - " + "\n  - ".join(mismatches)
    )


def test_defaults_is_a_dict_at_module_level(checker):
    """Smoke test: ``DEFAULTS`` is importable as a dict from the script.

    Guards against accidental rename (e.g. to ``DEFAULT_GATES``) that
    would silently break the drift test above. The drift test itself
    only runs when the key is present, so without this regression
    guard the test could be deleted / renamed without failing CI.
    """
    assert isinstance(checker.DEFAULTS, dict)
    # The drift-relevant top-level sections must all be present.
    for section in ("validation", "benchmark", "crate_size", "drift", "ci"):
        assert (
            section in checker.DEFAULTS
        ), f"DEFAULTS is missing top-level section {section!r}"
