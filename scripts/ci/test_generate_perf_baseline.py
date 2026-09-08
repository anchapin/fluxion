from __future__ import annotations

import json

import pytest

SCRIPT_NAME = "generate_perf_baseline"


@pytest.fixture
def generator(load_script):
    return load_script(SCRIPT_NAME)


def test_one_run_extracts_last_metrics(generator, monkeypatch):
    class Result:
        returncode = 0
        stdout = "Throughput: 10.0 configs/sec\nThroughput: 12.5 configs/sec"
        stderr = "Latency per config: 4.0ms\nLatency per config: 3.5ms"

    monkeypatch.setattr(generator.subprocess, "run", lambda *args, **kwargs: Result())
    assert generator.one_run(1, 1) == {"throughput": 12.5, "latency": 3.5}


def test_main_writes_medians_and_report_metadata(generator, monkeypatch, tmp_path):
    output = tmp_path / "baseline.json"
    samples = iter([(10.0, 4.0), (20.0, 2.0), (15.0, 3.0)])
    monkeypatch.setattr(generator, "one_run", lambda idx, n: dict(zip(("throughput", "latency"), next(samples))))
    monkeypatch.setattr(generator.platform, "platform", lambda: "test")
    monkeypatch.setattr(generator.platform, "processor", lambda: "cpu")
    monkeypatch.setattr(generator.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(generator.sys, "argv", ["generate_perf_baseline.py", str(output), "3"])
    assert generator.main() == 0
    data = json.loads(output.read_text())
    assert data["throughput_analytical"] == 15.0
    assert data["latency_ms"] == 3.0
    assert data["_meta"]["enforcement"] == "report-only"
    assert data["_meta"]["n_runs"] == 3


def test_main_aborts_with_fewer_than_three_samples(generator, monkeypatch, tmp_path):
    output = tmp_path / "baseline.json"
    monkeypatch.setattr(generator, "one_run", lambda idx, n: {})
    monkeypatch.setattr(generator.sys, "argv", ["generate_perf_baseline.py", str(output), "3"])
    assert generator.main() == 2
    assert not output.exists()
