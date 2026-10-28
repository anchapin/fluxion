"""Unit tests for ``scripts/verify_weather_provenance.py`` (Issue #4284).

The script is the executable half of the Section 6 weather provenance chain:
it re-hashes the committed ``assets/weather/WD*.epw`` / ``WD*.tmy3`` files and
compares them against ``assets/weather/provenance-manifest.json``.

These tests drive the module-level API against hermetic ``tmp_path`` trees so
they never depend on the real weather files being present or unchanged, plus
one integration case that runs the CLI against the real committed manifest.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "verify_weather_provenance.py"
REAL_MANIFEST = REPO_ROOT / "assets" / "weather" / "provenance-manifest.json"


@pytest.fixture
def verifier(load_script):
    return load_script("verify_weather_provenance")


def _write_manifest(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps({"files": rows}), encoding="utf-8")
    return path


def _weather_tree(root: Path, name: str, payload: bytes) -> Path:
    weather = root / "assets" / "weather"
    weather.mkdir(parents=True, exist_ok=True)
    (weather / name).write_bytes(payload)
    return weather


def test_sha256_matches_hashlib(verifier, tmp_path):
    import hashlib

    target = tmp_path / "blob.bin"
    payload = b"section 6 weather driver payload"
    target.write_bytes(payload)
    assert verifier.sha256(target) == hashlib.sha256(payload).hexdigest()


def test_check_committed_passes_when_hashes_match(verifier, tmp_path, monkeypatch):
    payload = b"WD bytes"
    weather = _weather_tree(tmp_path, "WD100.epw", payload)
    monkeypatch.setattr(verifier, "WEATHER_DIR", weather)
    rows = [{
        "filename": "WD100.epw",
        "committed_sha256": verifier.sha256(weather / "WD100.epw"),
        "committed_bytes": len(payload),
    }]
    ok, problems = verifier.check_committed({"files": rows})
    assert ok == 1
    assert problems == []


def test_check_committed_reports_mismatch(verifier, tmp_path, monkeypatch):
    weather = _weather_tree(tmp_path, "WD100.epw", b"WD bytes")
    monkeypatch.setattr(verifier, "WEATHER_DIR", weather)
    rows = [{
        "filename": "WD100.epw",
        "committed_sha256": "0" * 64,
        "committed_bytes": len(b"WD bytes"),
    }]
    ok, problems = verifier.check_committed({"files": rows})
    assert ok == 0
    assert len(problems) == 1
    assert "MISMATCH" in problems[0]


def test_check_committed_reports_absent_file(verifier, tmp_path, monkeypatch):
    weather = tmp_path / "assets" / "weather"
    weather.mkdir(parents=True)
    monkeypatch.setattr(verifier, "WEATHER_DIR", weather)
    rows = [{"filename": "WD100.epw", "committed_sha256": "0" * 64}]
    ok, problems = verifier.check_committed({"files": rows})
    assert ok == 0
    assert "ABSENT" in problems[0]


def test_check_committed_reports_size_drift(verifier, tmp_path, monkeypatch):
    payload = b"WD bytes"
    weather = _weather_tree(tmp_path, "WD100.epw", payload)
    monkeypatch.setattr(verifier, "WEATHER_DIR", weather)
    rows = [{
        "filename": "WD100.epw",
        "committed_sha256": verifier.sha256(weather / "WD100.epw"),
        "committed_bytes": len(payload) + 1,
    }]
    ok, problems = verifier.check_committed({"files": rows})
    assert ok == 0
    assert "SIZE" in problems[0]


def test_check_publisher_compares_against_extracted_copy(verifier, tmp_path):
    payload = b"publisher bytes"
    publisher = tmp_path / "publisher"
    publisher.mkdir()
    (publisher / "WD100.epw").write_bytes(payload)
    rows = [{
        "filename": "WD100.epw",
        "publisher_sha256": verifier.sha256(publisher / "WD100.epw"),
    }]
    ok, problems = verifier.check_publisher({"files": rows}, publisher)
    assert ok == 1
    assert problems == []

    bad = [{
        "filename": "WD100.epw",
        "publisher_sha256": "0" * 64,
    }]
    ok, problems = verifier.check_publisher({"files": bad}, publisher)
    assert ok == 0
    assert "MISMATCH" in problems[0]


def test_load_manifest_rejects_malformed_documents(verifier, tmp_path):
    with pytest.raises(SystemExit):
        verifier.load_manifest(tmp_path / "missing.json")

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit):
        verifier.load_manifest(broken)

    empty = _write_manifest(tmp_path / "empty.json", [])
    with pytest.raises(SystemExit):
        verifier.load_manifest(empty)


def test_cli_exits_non_zero_on_manifest_mismatch(tmp_path):
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [{"filename": "WD100.epw", "committed_sha256": "0" * 64}],
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "FAIL" in result.stdout


def test_real_committed_manifest_verifies():
    """Integration: the committed Section 6 weather files match the manifest.

    This is the check the provenance chain exists for. If a committed weather
    file drifts away from its recorded hash, this test fails.
    """
    if not REAL_MANIFEST.is_file():
        pytest.skip("committed provenance manifest not present")
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout
