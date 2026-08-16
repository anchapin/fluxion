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
from pathlib import Path

import pytest

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


def _results(pass_rate: float = 50.0, mae: float = 30.0,
             cases: dict | None = None) -> dict:
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
        "600": {"heating": 100.0, "heating_min": 4.0, "heating_max": 6.0,
                "cooling": 100.0, "cooling_min": 4.0, "cooling_max": 6.0},
        "700": {"heating": 5.0, "heating_min": 4.0, "heating_max": 6.0,
                "cooling": 5.0, "cooling_min": 4.0, "cooling_max": 6.0},
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
            "heating": 100.0, "heating_min": 4.0, "heating_max": 6.0,
            "cooling": 5.0, "cooling_min": 4.0, "cooling_max": 6.0,
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


def _bench(throughput: float = 200.0, latency: float = 5.0,
           multi_zone: float = 20.0, hybrid: float = 0.0,
           hybrid_multi_zone: float = 0.0, cv_latency: float = 100.0,
           cold_start_ms: float = 0.0, warm_ms: float = 0.0,
           cold_warm_ratio: float = 0.0) -> dict:
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
    config = {**_DRIFT_CONFIG, "drift": {**_DRIFT_CONFIG["drift"],
                                          "create_baseline_if_missing": False}}
    rg = _make_checker(checker, tmp_path, config)
    current = {"summary": {"pass_rate": 50.0, "mae": 30.0}, "cases": {}}
    results = rg.check_drift_gates(current, baseline=None)
    by_name = {r.name: r for r in results}
    assert by_name["baseline"].passed is False


def test_check_drift_gates_disabled_returns_pass(checker, tmp_path):
    """``drift.enabled=False`` → single PASS gate, no baseline required."""
    config = {"drift": {"enabled": False}}
    rg = _make_checker(checker, tmp_path, config)
    results = rg.check_drift_gates(
        {"summary": {}, "cases": {}}, baseline=None
    )
    assert len(results) == 1
    assert results[0].passed is True
    assert "disabled" in results[0].message.lower()


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
        summary={"total": len(results), "passed": len(results),
                 "failed": 0, "by_category": {"validation": {"passed": len(results), "failed": 0, "total": len(results)}}},
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
        "validation": {"min_pass_rate": 50.0, "max_mae": 50.0,
                       "individual": {"max_deviation": 150.0,
                                      "extreme_deviation_limit": 5,
                                      "known_failures": []}},
        "ci": {"fail_fast": True},
        "drift": {"enabled": False},
    }
    validation_path = tmp_path / "validation_results.json"
    validation_path.write_text(json.dumps({
        "summary": {"pass_rate": 75.0, "mae": 30.0},
        "cases": {},
    }))

    def fake_yaml_load(stream):
        if hasattr(stream, "name") and "release_gates.yaml" in str(stream.name):
            return config
        import yaml as _yaml  # type: ignore
        return _yaml.safe_load(stream)

    monkeypatch.setattr(checker.yaml, "safe_load", fake_yaml_load)
    monkeypatch.setattr(
        checker, "load_crate_size_results", lambda *a, **kw: None
    )

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
        "validation": {"min_pass_rate": 90.0, "max_mae": 50.0,
                       "individual": {"max_deviation": 150.0,
                                      "extreme_deviation_limit": 5,
                                      "known_failures": []}},
        "ci": {"fail_fast": True},
        "drift": {"enabled": False},
    }
    validation_path = tmp_path / "validation_results.json"
    validation_path.write_text(json.dumps({
        "summary": {"pass_rate": 50.0, "mae": 30.0},
        "cases": {},
    }))

    def fake_yaml_load(stream):
        if hasattr(stream, "name") and "release_gates.yaml" in str(stream.name):
            return config
        import yaml as _yaml  # type: ignore
        return _yaml.safe_load(stream)

    monkeypatch.setattr(checker.yaml, "safe_load", fake_yaml_load)
    monkeypatch.setattr(
        checker, "load_crate_size_results", lambda *a, **kw: None
    )

    saved = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME, "--validation-results", str(validation_path)]
    try:
        rc = checker.main()
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 0
    finally:
        sys.argv[:] = saved

    assert rc == 1
