"""
Pytest harness for ``scripts/verify_h_tr_em_regression.py`` -- Issue #3063.

Mirrors the ``load_script`` + ``tmp_path`` mock-repo pattern from
``scripts/ci/test_verify_gauge_solver_regression.py`` (Issue #3070):

* load the script as a fresh module via the shared ``load_script`` fixture,
* plant per-case JSON + ``baseline_manifest.json`` in ``tmp_path`` for each
  scenario, then
* drive ``main()`` through clean (no drift), regression (drift > 0),
  placeholder (fail-closed), schema-drift, and tolerance-override scenarios.

The script's three key surfaces are pure functions / classes --
``SnapshotSet.is_placeholder``, ``compute_diff``, ``verify_fingerprints`` --
plus a CLI ``main()`` that consumes ``--before`` / ``--after`` paths.
Each test plants both inputs in ``tmp_path`` and invokes ``main()`` via
``sys.argv`` injection.

The metric schema mirrors the issue's Case 195 acceptance criterion
(annual heating kWh, annual cooling kWh, peak heating kW).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT_NAME = "verify_h_tr_em_regression"
SUPPORTED_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def verifier(load_script):
    """Freshly-loaded copy of the h_tr_em regression verifier."""
    return load_script(SCRIPT_NAME)


def _make_case_payload(
    *,
    case_id: str,
    heating: float | None,
    cooling: float | None,
    peak_h: float | None,
    captured_at: str | None = "2026-08-17T00:00:00Z",
    captured_commit: str | None = "deadbeefdeadbeef",
) -> dict[str, Any]:
    """Build a per-case JSON payload with the documented Case 195 schema."""
    return {
        "_doc": "synthetic",
        "case_id": case_id,
        "captured_at": captured_at,
        "captured_commit": captured_commit,
        "metrics": {
            "annual_heating_kwh": heating,
            "annual_cooling_kwh": cooling,
            "peak_heating_kw": peak_h,
        },
    }


def _make_manifest(
    case_payloads: dict[str, dict[str, Any]],
    *,
    captured_at: str | None = "2026-08-17T00:00:00Z",
    captured_commit: str | None = "deadbeefdeadbeef",
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a ``baseline_manifest.json`` payload referencing the supplied cases."""
    tolerances = tolerances or {
        "annual_heating_kwh": 0.0,
        "annual_cooling_kwh": 0.0,
        "peak_heating_kw": 0.0,
    }
    return {
        "_doc": "synthetic",
        "_schema_version": SUPPORTED_SCHEMA_VERSION,
        "captured_at": captured_at,
        "captured_commit": captured_commit,
        "cases": {
            case_key: {
                "path": f"{case_key}_baseline.json",
                "case_id": payload["case_id"],
                "description": f"synthetic {case_key}",
                "metrics": [
                    "annual_heating_kwh",
                    "annual_cooling_kwh",
                    "peak_heating_kw",
                ],
                "sha256": None,
            }
            for case_key, payload in case_payloads.items()
        },
        "verifier": {
            "path": "scripts/verify_h_tr_em_regression.py",
            "default_tolerance": tolerances,
        },
    }


def _write_snapshot_set(
    directory: Path,
    case_payloads: dict[str, dict[str, Any]],
    manifest_kwargs: dict[str, Any] | None = None,
) -> Path:
    """Plant a snapshot directory under ``tmp_path`` and return the dir."""
    directory.mkdir(parents=True, exist_ok=True)
    for case_key, payload in case_payloads.items():
        (directory / f"{case_key}_baseline.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    manifest = _make_manifest(case_payloads, **(manifest_kwargs or {}))
    (directory / "baseline_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return directory


def _populate_case_metric(
    case_payloads: dict[str, dict[str, Any]],
    case_key: str,
    metric: str,
    value: float,
) -> None:
    """Mutate one metric in the test fixture dict (helper for regression tests)."""
    case_payloads[case_key]["metrics"][metric] = value


def _invoke(verifier, before: Path, after: Path, *extra: str) -> tuple[int, str]:
    """Invoke ``verifier.main()`` with synthetic argv; return (rc, stdout)."""
    saved = sys.argv[:]
    sys.argv[:] = [
        SCRIPT_NAME,
        "--before", str(before),
        "--after", str(after),
        *extra,
    ]
    try:
        rc = verifier.main()
    finally:
        sys.argv[:] = saved
    return rc, ""


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_load_snapshot_set_parses_manifest(verifier, tmp_path):
    """A well-formed snapshot directory loads without error."""
    payloads = {
        "case_195": _make_case_payload(
            case_id="195",
            heating=6250.0, cooling=758.0,
            peak_h=1.05,
        ),
        "case_600": _make_case_payload(
            case_id="600",
            heating=4600.0, cooling=3300.0,
            peak_h=4.4,
        ),
        "case_620": _make_case_payload(
            case_id="620",
            heating=5000.0, cooling=2400.0,
            peak_h=3.2,
        ),
    }
    snapshot_dir = _write_snapshot_set(tmp_path, payloads)
    snapshot_set = verifier.load_snapshot_set(snapshot_dir)

    assert set(snapshot_set.cases.keys()) == {"case_195", "case_600", "case_620"}
    assert snapshot_set.cases["case_195"].metrics["annual_heating_kwh"] == 6250.0
    assert snapshot_set.cases["case_195"].case_id == "195"
    assert not snapshot_set.is_placeholder()


def test_load_snapshot_set_rejects_missing_manifest(verifier, tmp_path):
    """A directory without ``baseline_manifest.json`` raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError, match="missing manifest"):
        verifier.load_snapshot_set(tmp_path / "empty")


def test_load_snapshot_set_rejects_wrong_schema_version(verifier, tmp_path):
    """An older ``_schema_version`` raises ``ValueError`` (fail-closed on schema drift)."""
    payloads = {
        "case_195": _make_case_payload(
            case_id="195", heating=1.0, cooling=1.0, peak_h=1.0,
        ),
    }
    snapshot_dir = _write_snapshot_set(tmp_path, payloads)
    manifest_path = snapshot_dir / "baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["_schema_version"] = 999  # future version we don't know about
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported manifest schema_version"):
        verifier.load_snapshot_set(snapshot_dir)


def test_load_snapshot_set_skips_doc_keys(verifier, tmp_path):
    """Manifest entries whose key starts with ``_`` are documentation, not cases.

    Mirrors the layout of the placeholder manifest from
    ``scripts/verify_gauge_solver_regression.py`` (Issue #3070) which plants
    a ``cases._doc`` string next to the case entries. The loader must
    skip these without crashing (regression guard for the earlier
    ``'str' object has no attribute 'get'`` bug).
    """
    payloads = {
        "case_195": _make_case_payload(
            case_id="195", heating=1.0, cooling=1.0, peak_h=1.0,
        ),
    }
    snapshot_dir = _write_snapshot_set(tmp_path, payloads)
    manifest_path = snapshot_dir / "baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Inject a `_doc` key alongside the real cases (as the shipped
    # placeholder manifest does).
    manifest["cases"]["_doc"] = "Per-case snapshot file map."
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    snapshot_set = verifier.load_snapshot_set(snapshot_dir)
    assert "_doc" not in snapshot_set.cases
    assert set(snapshot_set.cases.keys()) == {"case_195"}


def test_snapshot_set_placeholder_detection(verifier, tmp_path):
    """A snapshot set with any null metric is a placeholder."""
    payload = _make_case_payload(
        case_id="195",
        heating=None, cooling=None, peak_h=None,
    )
    snapshot_dir = _write_snapshot_set(tmp_path, {"case_195": payload})

    snapshot_set = verifier.load_snapshot_set(snapshot_dir)
    assert snapshot_set.is_placeholder()
    assert snapshot_set.cases["case_195"].is_placeholder()


def test_compute_diff_no_drift(verifier, tmp_path):
    """Identical before/after → no deltas, no regressions."""
    payload = _make_case_payload(
        case_id="195", heating=6250.0, cooling=758.0, peak_h=1.05,
    )
    before_dir = _write_snapshot_set(tmp_path / "before", {"case_195": payload})
    after_dir = _write_snapshot_set(tmp_path / "after", {"case_195": payload})

    before_set = verifier.load_snapshot_set(before_dir)
    after_set = verifier.load_snapshot_set(after_dir)
    report = verifier.compute_diff(before_set, after_set)

    assert not report.has_regression
    assert report.deltas
    assert all(d.delta == 0.0 for d in report.deltas)
    assert all(d.within_tolerance for d in report.deltas)


def test_compute_diff_flags_cooling_regression_when_delta_exceeds_tolerance(verifier, tmp_path):
    """A 1.0 kWh cooling drift against tolerance 0.0 → regression flagged.

    This is the issue-#3063 diagnostic: the 220 kWh → 758 kWh Case 195
    cooling shift is the directional signal that the per-step recompute
    is missing. The verifier must flag this on a synthetic snapshot.
    """
    before_payload = _make_case_payload(
        case_id="195", heating=6250.0, cooling=220.0, peak_h=1.05,
    )
    after_payload = _make_case_payload(
        case_id="195", heating=6250.0, cooling=758.0, peak_h=1.05,
    )
    before_dir = _write_snapshot_set(tmp_path / "before", {"case_195": before_payload})
    after_dir = _write_snapshot_set(tmp_path / "after", {"case_195": after_payload})

    before_set = verifier.load_snapshot_set(before_dir)
    after_set = verifier.load_snapshot_set(after_dir)
    report = verifier.compute_diff(before_set, after_set)

    assert report.has_regression
    cooling_delta = next(d for d in report.deltas if d.metric == "annual_cooling_kwh")
    assert cooling_delta.delta == pytest.approx(538.0)
    assert not cooling_delta.within_tolerance


def test_compute_diff_respects_override_tolerance(verifier, tmp_path):
    """CLI-supplied tolerance override absorbs sub-tolerance drift → no regression."""
    before_payload = _make_case_payload(
        case_id="195", heating=6250.0, cooling=220.0, peak_h=1.05,
    )
    after_payload = _make_case_payload(
        case_id="195", heating=6250.0, cooling=400.0, peak_h=1.05,
    )
    before_dir = _write_snapshot_set(tmp_path / "before", {"case_195": before_payload})
    after_dir = _write_snapshot_set(tmp_path / "after", {"case_195": after_payload})

    before_set = verifier.load_snapshot_set(before_dir)
    after_set = verifier.load_snapshot_set(after_dir)
    report = verifier.compute_diff(
        before_set,
        after_set,
        override_tolerance={"annual_cooling_kwh": 250.0},
    )

    assert not report.has_regression
    cooling_delta = next(d for d in report.deltas if d.metric == "annual_cooling_kwh")
    assert cooling_delta.within_tolerance
    assert cooling_delta.tolerance == 250.0


def test_compute_diff_records_schema_drift_when_case_missing(verifier, tmp_path):
    """A case present in --after but missing from --before → schema_drift flagged."""
    before_dir = _write_snapshot_set(
        tmp_path / "before",
        {"case_195": _make_case_payload(case_id="195", heating=1.0, cooling=1.0, peak_h=1.0)},
    )
    after_dir = _write_snapshot_set(
        tmp_path / "after",
        {
            "case_195": _make_case_payload(case_id="195", heating=1.0, cooling=1.0, peak_h=1.0),
            "case_600": _make_case_payload(case_id="600", heating=1.0, cooling=1.0, peak_h=1.0),
        },
    )

    before_set = verifier.load_snapshot_set(before_dir)
    after_set = verifier.load_snapshot_set(after_dir)
    report = verifier.compute_diff(before_set, after_set)

    assert any("case 'case_600'" in s for s in report.schema_drift)


def test_verify_fingerprints_returns_empty_when_manifest_unstamped(verifier, tmp_path):
    """Fresh placeholder manifest (sha256 == null) → no fingerprint mismatches."""
    payload = _make_case_payload(
        case_id="195", heating=6250.0, cooling=758.0, peak_h=1.05,
    )
    snapshot_dir = _write_snapshot_set(tmp_path / "snap", {"case_195": payload})
    snapshot_set = verifier.load_snapshot_set(snapshot_dir)
    assert verifier.verify_fingerprints(snapshot_set) == []


def test_verify_fingerprints_flags_silent_edit(verifier, tmp_path):
    """A hand-edited case file (sha256 mismatch) → fingerprint mismatch."""
    payload = _make_case_payload(
        case_id="195", heating=6250.0, cooling=758.0, peak_h=1.05,
    )
    snapshot_dir = _write_snapshot_set(tmp_path / "snap", {"case_195": payload})
    manifest_path = snapshot_dir / "baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    # Stamp the manifest with a stale SHA-256 so a fresh file edit trips the check.
    manifest["cases"]["case_195"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    snapshot_set = verifier.load_snapshot_set(snapshot_dir)
    mismatches = verifier.verify_fingerprints(snapshot_set)
    assert len(mismatches) == 1
    assert "case_195" in mismatches[0]


# ---------------------------------------------------------------------------
# main() end-to-end scenarios
# ---------------------------------------------------------------------------


def test_main_exits_zero_when_no_drift(verifier, tmp_path, capsys):
    """Identical snapshots → exit 0; human-readable report printed."""
    payload = _make_case_payload(
        case_id="195", heating=6250.0, cooling=758.0, peak_h=1.05,
    )
    before_dir = _write_snapshot_set(tmp_path / "before", {"case_195": payload})
    after_dir = _write_snapshot_set(tmp_path / "after", {"case_195": payload})

    rc, _ = _invoke(verifier, before_dir, after_dir, "--allow-placeholder")
    out = capsys.readouterr().out
    assert rc == verifier.EXIT_OK
    assert "PASS" in out


def test_main_exits_one_on_cooling_regression(verifier, tmp_path, capsys):
    """A cooling shift on Case 195 (the issue's #3063 diagnostic signature) → exit 1."""
    before_payload = _make_case_payload(
        case_id="195", heating=6250.0, cooling=220.0, peak_h=1.05,
    )
    after_payload = _make_case_payload(
        case_id="195", heating=6250.0, cooling=758.0, peak_h=1.05,
    )
    before_dir = _write_snapshot_set(tmp_path / "before", {"case_195": before_payload})
    after_dir = _write_snapshot_set(tmp_path / "after", {"case_195": after_payload})

    rc, _ = _invoke(verifier, before_dir, after_dir, "--allow-placeholder")
    out = capsys.readouterr().out
    assert rc == verifier.EXIT_REGRESSION
    assert "FAIL" in out
    assert "annual_cooling_kwh" in out


def test_main_returns_two_when_snapshot_unpopulated(verifier, tmp_path, capsys):
    """Default contract: a placeholder snapshot set → exit 2 (fail-closed)."""
    placeholder = _make_case_payload(
        case_id="195", heating=None, cooling=None, peak_h=None,
        captured_at=None, captured_commit=None,
    )
    populated = _make_case_payload(
        case_id="195", heating=6250.0, cooling=758.0, peak_h=1.05,
    )
    before_dir = _write_snapshot_set(tmp_path / "before", {"case_195": placeholder})
    after_dir = _write_snapshot_set(tmp_path / "after", {"case_195": populated})

    rc, _ = _invoke(verifier, before_dir, after_dir)
    err = capsys.readouterr().err
    assert rc == verifier.EXIT_PLACEHOLDER
    assert "placeholder" in err.lower()


def test_main_returns_three_on_missing_manifest(verifier, tmp_path, capsys):
    """A non-existent --before path → exit 3 (usage error)."""
    rc, _ = _invoke(verifier, tmp_path / "no_such_dir", tmp_path / "after")
    err = capsys.readouterr().err
    assert rc == verifier.EXIT_USAGE
    assert "missing manifest" in err


def test_main_emits_json_shape(verifier, tmp_path, capsys):
    """--json flag emits a parseable JSON document with the documented fields."""
    payload = _make_case_payload(
        case_id="195", heating=6250.0, cooling=758.0, peak_h=1.05,
    )
    before_dir = _write_snapshot_set(tmp_path / "before", {"case_195": payload})
    after_dir = _write_snapshot_set(tmp_path / "after", {"case_195": payload})

    _invoke(verifier, before_dir, after_dir, "--allow-placeholder", "--json")
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "before" in parsed
    assert "after" in parsed
    assert "deltas" in parsed
    assert "schema_drift" in parsed
    assert "summary" in parsed
    assert "has_regression" in parsed["summary"]


def test_main_strict_exits_two_on_sha256_mismatch(verifier, tmp_path, capsys):
    """--strict mode: a hand-edited case file (sha256 mismatch) → exit 2."""
    payload = _make_case_payload(
        case_id="195", heating=6250.0, cooling=758.0, peak_h=1.05,
    )
    before_dir = _write_snapshot_set(tmp_path / "before", {"case_195": payload})
    after_dir = _write_snapshot_set(tmp_path / "after", {"case_195": payload})

    # Stamp the --before manifest with a stale sha256 so the file-vs-manifest
    # comparison trips. We must re-load the manifest to get past JSON caching.
    manifest_path = before_dir / "baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"]["case_195"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rc, _ = _invoke(
        verifier, before_dir, after_dir,
        "--allow-placeholder", "--strict",
    )
    err = capsys.readouterr().err
    assert rc == verifier.EXIT_PLACEHOLDER
    assert "sha256 mismatch" in err


def test_main_tolerance_override_cli(verifier, tmp_path, capsys):
    """``--tolerance`` CLI flag overrides manifest defaults per-metric."""
    before_payload = _make_case_payload(
        case_id="195", heating=6250.0, cooling=220.0, peak_h=1.05,
    )
    after_payload = _make_case_payload(
        case_id="195", heating=6250.0, cooling=400.0, peak_h=1.05,
    )
    before_dir = _write_snapshot_set(tmp_path / "before", {"case_195": before_payload})
    after_dir = _write_snapshot_set(tmp_path / "after", {"case_195": after_payload})

    rc, _ = _invoke(
        verifier, before_dir, after_dir,
        "--allow-placeholder",
        "--tolerance", "annual_cooling_kwh=250.0",
    )
    assert rc == verifier.EXIT_OK


def test_main_default_snapshot_dir_points_to_h_tr_em_baseline(verifier):
    """The default ``--before`` path is the future implementer's snapshot directory.

    Mirrors the default path choice in
    ``scripts/verify_gauge_solver_regression.py`` (which defaults to
    ``tests/reference_data/gauge_solver_baseline/``). The directory
    does not yet exist on disk (the future implementer creates it), so
    we only assert the path resolves to the documented location.
    """
    assert verifier.DEFAULT_SNAPSHOT_DIR.name == "h_tr_em_baseline"
    assert verifier.DEFAULT_SNAPSHOT_DIR.parent.name == "reference_data"
