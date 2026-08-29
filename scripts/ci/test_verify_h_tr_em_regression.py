"""
Pytest harness for ``scripts/verify_h_tr_em_regression.py`` -- Issue #3265.

Mirrors the ``load_script`` + ``tmp_path`` mock-repo pattern from
``scripts/ci/test_verify_gauge_solver_regression.py``:

* load the script as a fresh module via the shared ``load_script`` fixture,
* plant per-case JSON + ``baseline_manifest.json`` in ``tmp_path`` for each
  scenario, then
* drive ``main()`` through clean (no drift), regression (drift > 0),
  placeholder (fail-closed), schema-drift, tolerance-override, and
  SHA-256-fingerprint scenarios.

The h_tr_em verifier follows the gauge_solver contract
(ADR-0009 §2 / ADR-0008): same dataclass / dataclass surface
(``CaseSnapshot`` / ``SnapshotSet`` / ``MetricDelta`` / ``DiffReport``),
same ``EXIT_OK=0 / EXIT_REGRESSION=1 / EXIT_PLACEHOLDER=2 / EXIT_USAGE=3``
exit codes, same fail-closed default. The tests below exercise each contract
path so a future refactor that drifts from the documented behaviour trips
this harness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT_NAME = "verify_h_tr_em_regression"
SUPPORTED_SCHEMA_VERSION = 1

METRIC_KEYS = (
    "h_tr_em_w_k",
    "h_tr_em_south_w_k",
    "h_tr_ms_w_k",
    "h_tr_ms_no_south_w_k",
    "h_tr_is_w_k",
    "h_tr_is_no_south_w_k",
    "h_tr_w_w_k",
    "h_ve_w_k",
    "h_tr_floor_w_k",
    "cm_j_per_k",
)
DEFAULT_TOLERANCES = {m: 0.0 for m in METRIC_KEYS}


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
    metric_value: float | None,
    captured_at: str | None = "2026-08-17T00:00:00Z",
    captured_commit: str | None = "deadbeefdeadbeef",
) -> dict[str, Any]:
    """Build a per-case JSON payload with the documented schema."""
    metrics = {m: metric_value for m in METRIC_KEYS}
    return {
        "_doc": "synthetic",
        "case_id": case_id,
        "captured_at": captured_at,
        "captured_commit": captured_commit,
        "metrics": metrics,
    }


def _make_manifest(
    case_payloads: dict[str, dict[str, Any]],
    *,
    captured_at: str | None = "2026-08-17T00:00:00Z",
    captured_commit: str | None = "deadbeefdeadbeef",
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a ``baseline_manifest.json`` payload referencing the supplied cases."""
    tolerances = tolerances or DEFAULT_TOLERANCES
    return {
        "_doc": "synthetic",
        "_schema_version": SUPPORTED_SCHEMA_VERSION,
        "captured_at": captured_at,
        "captured_commit": captured_commit,
        "cases": {
            case_key: {
                "path": f"{case_key}.json",
                "case_id": payload["case_id"],
                "description": f"synthetic {case_key}",
                "metrics": list(METRIC_KEYS),
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
        (directory / f"{case_key}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    manifest = _make_manifest(case_payloads, **(manifest_kwargs or {}))
    (directory / "baseline_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return directory


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
        "case_195": _make_case_payload(case_id="195", metric_value=69.0645),
        "case_600": _make_case_payload(case_id="600", metric_value=59.2564),
        "case_620": _make_case_payload(case_id="620", metric_value=59.2564),
        "case_900": _make_case_payload(case_id="900", metric_value=48.8803),
    }
    snapshot_dir = _write_snapshot_set(tmp_path, payloads)
    snapshot_set = verifier.load_snapshot_set(snapshot_dir)

    assert set(snapshot_set.cases.keys()) == {"case_195", "case_600", "case_620", "case_900"}
    assert snapshot_set.cases["case_600"].metrics["h_tr_em_w_k"] == 59.2564
    assert snapshot_set.cases["case_195"].case_id == "195"
    assert not snapshot_set.is_placeholder()


def test_load_snapshot_set_rejects_missing_manifest(verifier, tmp_path):
    """A directory without ``baseline_manifest.json`` raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError, match="missing manifest"):
        verifier.load_snapshot_set(tmp_path / "empty")


def test_load_snapshot_set_rejects_wrong_schema_version(verifier, tmp_path):
    """An older ``_schema_version`` raises ``ValueError`` (fail-closed on schema drift)."""
    payloads = {
        "case_600": _make_case_payload(case_id="600", metric_value=50.0),
    }
    snapshot_dir = _write_snapshot_set(tmp_path, payloads)
    manifest_path = snapshot_dir / "baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["_schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported manifest schema_version"):
        verifier.load_snapshot_set(snapshot_dir)


def test_load_snapshot_set_skips_doc_keys(verifier, tmp_path):
    """Manifest entries whose key starts with ``_`` are documentation, not cases.

    Mirrors the layout of the shipped h_tr_em baseline manifest which
    plants doc-only keys alongside real cases. The loader must skip
    these without crashing.
    """
    payloads = {
        "case_600": _make_case_payload(case_id="600", metric_value=50.0),
    }
    snapshot_dir = _write_snapshot_set(tmp_path, payloads)
    manifest_path = snapshot_dir / "baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"]["_doc"] = "Per-case snapshot file map."
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    snapshot_set = verifier.load_snapshot_set(snapshot_dir)
    assert "_doc" not in snapshot_set.cases
    assert set(snapshot_set.cases.keys()) == {"case_600"}


def test_snapshot_set_placeholder_detection(verifier, tmp_path):
    """A snapshot set with any null metric is a placeholder."""
    payload = _make_case_payload(case_id="600", metric_value=None)
    snapshot_dir = _write_snapshot_set(tmp_path, {"case_600": payload})

    snapshot_set = verifier.load_snapshot_set(snapshot_dir)
    assert snapshot_set.is_placeholder()
    assert snapshot_set.cases["case_600"].is_placeholder()


def test_snapshot_set_placeholder_when_manifest_captured_at_null(verifier, tmp_path):
    """Manifest-level ``captured_at == null`` is itself a placeholder."""
    payload = _make_case_payload(case_id="600", metric_value=50.0)
    snapshot_dir = _write_snapshot_set(
        tmp_path,
        {"case_600": payload},
        manifest_kwargs={"captured_at": None},
    )

    snapshot_set = verifier.load_snapshot_set(snapshot_dir)
    assert snapshot_set.is_placeholder()


def test_compute_diff_no_drift(verifier, tmp_path):
    """Identical before/after - no deltas, no regressions."""
    payload = _make_case_payload(case_id="600", metric_value=50.0)
    before_dir = _write_snapshot_set(tmp_path / "before", {"case_600": payload})
    after_dir = _write_snapshot_set(tmp_path / "after", {"case_600": payload})

    before_set = verifier.load_snapshot_set(before_dir)
    after_set = verifier.load_snapshot_set(after_dir)
    report = verifier.compute_diff(before_set, after_set)

    assert not report.has_regression
    assert report.deltas
    assert all(d.delta == 0.0 for d in report.deltas)
    assert all(d.within_tolerance for d in report.deltas)


def test_compute_diff_flags_regression_when_delta_exceeds_tolerance(verifier, tmp_path):
    """A 0.001 drift against tolerance 0.0 - regression flagged."""
    before_payload = _make_case_payload(case_id="600", metric_value=50.0)
    after_payload = _make_case_payload(case_id="600", metric_value=50.001)
    before_dir = _write_snapshot_set(tmp_path / "before", {"case_600": before_payload})
    after_dir = _write_snapshot_set(tmp_path / "after", {"case_600": after_payload})

    before_set = verifier.load_snapshot_set(before_dir)
    after_set = verifier.load_snapshot_set(after_dir)
    report = verifier.compute_diff(before_set, after_set)

    assert report.has_regression
    em_delta = next(d for d in report.deltas if d.metric == "h_tr_em_w_k")
    assert em_delta.delta == pytest.approx(0.001)
    assert not em_delta.within_tolerance


def test_compute_diff_respects_override_tolerance(verifier, tmp_path):
    """CLI-supplied tolerance override absorbs sub-tolerance drift - no regression."""
    before_payload = _make_case_payload(case_id="600", metric_value=50.0)
    # Drift ONLY the h_tr_em_w_k metric; leave the rest identical so the
    # override just needs to cover that single metric.
    after_payload = _make_case_payload(case_id="600", metric_value=50.0)
    after_payload["metrics"]["h_tr_em_w_k"] = 50.0009
    before_dir = _write_snapshot_set(tmp_path / "before", {"case_600": before_payload})
    after_dir = _write_snapshot_set(tmp_path / "after", {"case_600": after_payload})

    before_set = verifier.load_snapshot_set(before_dir)
    after_set = verifier.load_snapshot_set(after_dir)
    report = verifier.compute_diff(
        before_set,
        after_set,
        override_tolerance={"h_tr_em_w_k": 0.001},
    )

    assert not report.has_regression
    em_delta = next(d for d in report.deltas if d.metric == "h_tr_em_w_k")
    assert em_delta.within_tolerance
    assert em_delta.tolerance == 0.001


def test_compute_diff_records_schema_drift_when_case_missing(verifier, tmp_path):
    """A case present in --after but missing from --before - schema_drift flagged."""
    before_dir = _write_snapshot_set(
        tmp_path / "before",
        {"case_600": _make_case_payload(case_id="600", metric_value=1.0)},
    )
    after_dir = _write_snapshot_set(
        tmp_path / "after",
        {
            "case_600": _make_case_payload(case_id="600", metric_value=1.0),
            "case_620": _make_case_payload(case_id="620", metric_value=1.0),
        },
    )

    before_set = verifier.load_snapshot_set(before_dir)
    after_set = verifier.load_snapshot_set(after_dir)
    report = verifier.compute_diff(before_set, after_set)

    assert any("case 'case_620'" in s for s in report.schema_drift)


def test_verify_fingerprints_returns_empty_when_manifest_unstamped(verifier, tmp_path):
    """Fresh placeholder manifest (sha256 == null) - no fingerprint mismatches."""
    payload = _make_case_payload(case_id="600", metric_value=50.0)
    snapshot_dir = _write_snapshot_set(tmp_path / "snap", {"case_600": payload})
    snapshot_set = verifier.load_snapshot_set(snapshot_dir)
    assert verifier.verify_fingerprints(snapshot_set) == []


def test_verify_fingerprints_flags_silent_edit(verifier, tmp_path):
    """A hand-edited case file (sha256 mismatch) - fingerprint mismatch."""
    payload = _make_case_payload(case_id="600", metric_value=50.0)
    snapshot_dir = _write_snapshot_set(tmp_path / "snap", {"case_600": payload})
    manifest_path = snapshot_dir / "baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    manifest["cases"]["case_600"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    snapshot_set = verifier.load_snapshot_set(snapshot_dir)
    mismatches = verifier.verify_fingerprints(snapshot_set)
    assert len(mismatches) == 1
    assert "case_600" in mismatches[0]


# ---------------------------------------------------------------------------
# main() end-to-end scenarios
# ---------------------------------------------------------------------------


def test_main_exits_zero_when_no_drift(verifier, tmp_path, capsys):
    """Identical snapshots - exit 0; human-readable report printed."""
    payload = _make_case_payload(case_id="600", metric_value=50.0)
    before_dir = _write_snapshot_set(tmp_path / "before", {"case_600": payload})
    after_dir = _write_snapshot_set(tmp_path / "after", {"case_600": payload})

    rc, _ = _invoke(verifier, before_dir, after_dir, "--allow-placeholder")
    out = capsys.readouterr().out
    assert rc == verifier.EXIT_OK
    assert "PASS" in out


def test_main_exits_one_on_regression(verifier, tmp_path, capsys):
    """Any per-metric drift > 0 against tolerance 0.0 - exit 1 (regression)."""
    before_payload = _make_case_payload(case_id="600", metric_value=50.0)
    after_payload = _make_case_payload(case_id="600", metric_value=50.001)
    before_dir = _write_snapshot_set(tmp_path / "before", {"case_600": before_payload})
    after_dir = _write_snapshot_set(tmp_path / "after", {"case_600": after_payload})

    rc, _ = _invoke(verifier, before_dir, after_dir, "--allow-placeholder")
    out = capsys.readouterr().out
    assert rc == verifier.EXIT_REGRESSION
    assert "FAIL" in out
    assert "h_tr_em_w_k" in out


def test_main_returns_two_when_snapshot_unpopulated(verifier, tmp_path, capsys):
    """Default contract: a placeholder snapshot set - exit 2 (fail-closed)."""
    placeholder = _make_case_payload(
        case_id="600", metric_value=None,
        captured_at=None, captured_commit=None,
    )
    populated = _make_case_payload(case_id="600", metric_value=50.0)
    before_dir = _write_snapshot_set(tmp_path / "before", {"case_600": placeholder})
    after_dir = _write_snapshot_set(tmp_path / "after", {"case_600": populated})

    rc, _ = _invoke(verifier, before_dir, after_dir)
    err = capsys.readouterr().err
    assert rc == verifier.EXIT_PLACEHOLDER
    assert "placeholder" in err.lower()


def test_main_returns_three_on_missing_manifest(verifier, tmp_path, capsys):
    """A non-existent --before path - exit 3 (usage error)."""
    rc, _ = _invoke(verifier, tmp_path / "no_such_dir", tmp_path / "after")
    err = capsys.readouterr().err
    assert rc == verifier.EXIT_USAGE
    assert "missing manifest" in err


def test_main_emits_json_shape(verifier, tmp_path, capsys):
    """--json flag emits a parseable JSON document with the documented fields."""
    payload = _make_case_payload(case_id="600", metric_value=50.0)
    before_dir = _write_snapshot_set(tmp_path / "before", {"case_600": payload})
    after_dir = _write_snapshot_set(tmp_path / "after", {"case_600": payload})

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
    """--strict mode: a hand-edited case file (sha256 mismatch) - exit 2."""
    payload = _make_case_payload(case_id="600", metric_value=50.0)
    before_dir = _write_snapshot_set(tmp_path / "before", {"case_600": payload})
    after_dir = _write_snapshot_set(tmp_path / "after", {"case_600": payload})

    manifest_path = before_dir / "baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"]["case_600"]["sha256"] = "0" * 64
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
    before_payload = _make_case_payload(case_id="600", metric_value=50.0)
    after_payload = _make_case_payload(case_id="600", metric_value=50.0)
    after_payload["metrics"]["h_tr_em_w_k"] = 50.0009
    before_dir = _write_snapshot_set(tmp_path / "before", {"case_600": before_payload})
    after_dir = _write_snapshot_set(tmp_path / "after", {"case_600": after_payload})

    rc, _ = _invoke(
        verifier, before_dir, after_dir,
        "--allow-placeholder",
        "--tolerance", "h_tr_em_w_k=0.001",
    )
    assert rc == verifier.EXIT_OK
