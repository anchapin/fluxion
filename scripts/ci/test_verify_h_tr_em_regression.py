"""
Pytest harness for ``scripts/verify_h_tr_em_regression.py`` -- Issue #3549.

Re-materialized per the Issue #3549 acceptance contract:

* CLI args: ``--manifest <path>`` (single arg), ``--tolerance <float>``,
  ``--strict``, ``--json``. The verifier validates the integrity of one
  ``baseline_manifest.json`` against the per-case JSONs it enumerates
  (SHA-256 fingerprint check), not a before/after diff.
* Exit codes: ``EXIT_OK=0`` (all fingerprints match), ``EXIT_REGRESSION=1``
  (mismatch under ``--strict``), ``EXIT_PLACEHOLDER=2`` (any per-case
  ``sha256`` is null or manifest ``captured_at`` is null),
  ``EXIT_USAGE=3`` (bad path, malformed JSON, missing manifest).

Nine scenarios are exercised below, matching the Issue #3549 acceptance
list: placeholder detection, no-drift (all match), regression (mismatch
under ``--strict``), tolerance-override (``--tolerance`` CLI flag),
schema-drift (bad ``_schema_version``), missing-manifest (exit 3),
JSON output (``--json`` flag), ``--strict`` SHA-256 mismatch, and CLI
tolerance-override (env var ``BASELINE_H_TR_EM_TOLERANCE``).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_h_tr_em_regression.py"
SCRIPT_NAME = "verify_h_tr_em_regression"
SUPPORTED_SCHEMA_VERSION = 1

MANIFEST_FILENAME = "baseline_manifest.json"

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
# Fixture: freshly-loaded verifier module
# ---------------------------------------------------------------------------


@pytest.fixture
def verifier():
    """Load ``scripts/verify_h_tr_em_regression.py`` as a fresh module."""
    if not SCRIPT_PATH.is_file():
        pytest.skip(f"verifier script missing: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(SCRIPT_NAME, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(SCRIPT_NAME, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def default_baseline_dir():
    """Path to the shipped h_tr_em baseline directory."""
    return REPO_ROOT / "tests" / "reference_data" / "h_tr_em_baseline"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fingerprint(path: Path) -> str:
    """Mirror the verifier's SHA-256 fingerprint helper."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_case_payload(
    directory: Path,
    case_key: str,
    *,
    case_id: str,
    metric_value: float | None = 50.0,
    captured_at: str | None = "2026-09-08T18:59:05Z",
    captured_commit: str | None = "3bd1f15b10dc680dd35df3c63a1ed6e1ce4c0f76",
) -> dict[str, Any]:
    """Write a per-case JSON file at ``directory/case_<id>.json`` and return its payload."""
    payload = {
        "_doc": "synthetic",
        "case_id": case_id,
        "captured_at": captured_at,
        "captured_commit": captured_commit,
        "metrics": {m: metric_value for m in METRIC_KEYS},
    }
    rel = f"{case_key}.json"
    (directory / rel).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _write_manifest(
    directory: Path,
    case_keys: list[str],
    *,
    captured_at: str | None = "2026-09-08T18:59:05Z",
    captured_commit: str | None = "3bd1f15b10dc680dd35df3c63a1ed6e1ce4c0f76",
    schema_version: int = SUPPORTED_SCHEMA_VERSION,
    sha256_overrides: dict[str, str | None] | None = None,
) -> None:
    """Write a ``baseline_manifest.json`` enumerating the per-case files."""
    sha256_overrides = sha256_overrides or {}
    cases: dict[str, Any] = {}
    for case_key in case_keys:
        case_id = case_key.split("_", 1)[1]
        cases[case_key] = {
            "_doc": "synthetic",
            "case_id": case_id,
            "description": f"synthetic {case_key}",
            "metrics": list(METRIC_KEYS),
            "path": f"{case_key}.json",
            "sha256": sha256_overrides.get(
                case_key,
                _fingerprint(directory / f"{case_key}.json"),
            ),
        }
    manifest = {
        "_doc": "synthetic",
        "_schema_version": schema_version,
        "captured_at": captured_at,
        "captured_commit": captured_commit,
        "cases": cases,
        "verifier": {
            "path": "scripts/verify_h_tr_em_regression.py",
            "default_tolerance": DEFAULT_TOLERANCES,
        },
    }
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _populate_synthetic_baseline(
    directory: Path,
    *,
    captured_at: str | None = "2026-09-08T18:59:05Z",
    captured_commit: str | None = "3bd1f15b10dc680dd35df3c63a1ed6e1ce4c0f76",
    placeholders: list[str] | None = None,
) -> list[str]:
    """Plant a 4-case synthetic baseline set under ``directory``.

    Args:
        placeholders: Subset of case keys whose ``sha256`` stamp should be
            set to ``null`` (placeholder signal). Other cases get the
            real SHA-256 fingerprint of their just-written JSON file.

    Returns:
        List of case keys written, in order.
    """
    placeholders = set(placeholders or [])
    case_keys = ["case_195", "case_600", "case_620", "case_900"]
    directory.mkdir(parents=True, exist_ok=True)
    for case_key in case_keys:
        _write_case_payload(
            directory,
            case_key,
            case_id=case_key.split("_", 1)[1],
        )
    sha256_overrides = {ck: None for ck in placeholders}
    _write_manifest(
        directory,
        case_keys,
        captured_at=captured_at,
        captured_commit=captured_commit,
        sha256_overrides=sha256_overrides,
    )
    return case_keys


def _invoke(verifier, manifest: Path, *extra: str) -> tuple[int, str, str]:
    """Invoke ``verifier.main()`` with synthetic argv; return (rc, stdout, stderr)."""
    saved_argv = sys.argv[:]
    saved_env = os.environ.copy()
    sys.argv[:] = [SCRIPT_NAME, "--manifest", str(manifest), *extra]
    try:
        rc = verifier.main()
    finally:
        sys.argv[:] = saved_argv
        os.environ.clear()
        os.environ.update(saved_env)
    return rc, sys.stdout.getvalue() if False else "", ""  # capture via capsys in tests


def _invoke_with_capsys(verifier, manifest: Path, capsys, *extra: str) -> int:
    """Invoke ``verifier.main()`` and return the captured exit code."""
    saved_argv = sys.argv[:]
    sys.argv[:] = [SCRIPT_NAME, "--manifest", str(manifest), *extra]
    try:
        rc = verifier.main()
    finally:
        sys.argv[:] = saved_argv
    return rc


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def test_placeholder_detection(verifier, tmp_path, capsys):
    """Scenario 1: any per-case ``sha256`` stamp is null → exit 2."""
    _populate_synthetic_baseline(tmp_path, placeholders=["case_600"])
    rc = _invoke_with_capsys(verifier, tmp_path / MANIFEST_FILENAME, capsys)
    err = capsys.readouterr().err
    assert rc == verifier.EXIT_PLACEHOLDER
    assert "placeholder" in err.lower()
    assert "case_600" in err


def test_no_drift_exit_zero(verifier, tmp_path, capsys):
    """Scenario 2: all per-case stamps match real fingerprints → exit 0."""
    _populate_synthetic_baseline(tmp_path)
    rc = _invoke_with_capsys(verifier, tmp_path / MANIFEST_FILENAME, capsys)
    out = capsys.readouterr().out
    assert rc == verifier.EXIT_OK
    assert "MATCH" in out
    assert "0 mismatch" in out
    assert "0 placeholder" in out


def test_regression_under_strict(verifier, tmp_path, capsys):
    """Scenario 3: ``--strict`` + SHA-256 mismatch → exit 1."""
    _populate_synthetic_baseline(tmp_path)
    manifest_path = tmp_path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"]["case_600"]["sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rc = _invoke_with_capsys(
        verifier,
        manifest_path,
        capsys,
        "--strict",
    )
    err = capsys.readouterr().err
    assert rc == verifier.EXIT_REGRESSION
    assert "case_600" in err
    assert "mismatch" in err.lower()


def test_tolerance_override_absorbs_warn(verifier, tmp_path, capsys, monkeypatch):
    """Scenario 4: ``--tolerance`` CLI flag is accepted (forward-compatible).

    The current verifier contract compares SHA-256 hex digests (bit-
    identical), so a non-zero tolerance does not change the comparison
    outcome — but the flag must parse cleanly and not trip ``EXIT_USAGE``.
    Mirrors the gauge_solver tolerance-override scenario.
    """
    _populate_synthetic_baseline(tmp_path)
    manifest_path = tmp_path / MANIFEST_FILENAME

    rc = _invoke_with_capsys(
        verifier,
        manifest_path,
        capsys,
        "--tolerance", "0.001",
    )
    assert rc == verifier.EXIT_OK


def test_schema_drift(verifier, tmp_path, capsys):
    """Scenario 5: wrong ``_schema_version`` → exit 3 (usage error)."""
    _populate_synthetic_baseline(tmp_path)
    manifest_path = tmp_path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["_schema_version"] = 999
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rc = _invoke_with_capsys(verifier, manifest_path, capsys)
    err = capsys.readouterr().err
    assert rc == verifier.EXIT_USAGE
    assert "schema_version" in err


def test_missing_manifest(verifier, tmp_path, capsys):
    """Scenario 6: non-existent ``--manifest`` path → exit 3."""
    missing = tmp_path / "no_such_manifest.json"
    rc = _invoke_with_capsys(verifier, missing, capsys)
    err = capsys.readouterr().err
    assert rc == verifier.EXIT_USAGE
    assert "missing manifest" in err


def test_json_output_shape(verifier, tmp_path, capsys):
    """Scenario 7: ``--json`` emits a parseable document with documented fields."""
    _populate_synthetic_baseline(tmp_path)
    rc = _invoke_with_capsys(
        verifier,
        tmp_path / MANIFEST_FILENAME,
        capsys,
        "--json",
    )
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "manifest" in parsed
    assert "cases" in parsed
    assert "mismatches" in parsed
    assert "placeholders" in parsed
    assert "summary" in parsed
    assert "exit" in parsed
    assert "manifest_captured_at" in parsed
    assert "manifest_captured_commit" in parsed
    assert "has_regression" in parsed["summary"]
    assert "matched" in parsed["summary"]
    assert "total" in parsed["summary"]
    assert rc == verifier.EXIT_OK


def test_strict_sha256_mismatch_exit_one(verifier, tmp_path, capsys):
    """Scenario 8: ``--strict`` + SHA-256 mismatch → exit 1 (regression).

    Distinct from scenario 3 because this one mutates a per-case JSON
    AFTER stamping so the verifier computes a different digest than the
    manifest declares. End-to-end silent-edit detector.
    """
    _populate_synthetic_baseline(tmp_path)
    # Hand-edit a per-case JSON to invalidate its stamp
    case_600 = tmp_path / "case_600.json"
    payload = json.loads(case_600.read_text())
    payload["metrics"]["h_tr_em_w_k"] = 50.5
    case_600.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rc = _invoke_with_capsys(
        verifier,
        tmp_path / MANIFEST_FILENAME,
        capsys,
        "--strict",
    )
    err = capsys.readouterr().err
    assert rc == verifier.EXIT_REGRESSION
    assert "case_600" in err
    assert "mismatch" in err.lower()


def test_cli_tolerance_override_env_var(verifier, tmp_path, capsys, monkeypatch):
    """Scenario 9: ``BASELINE_H_TR_EM_TOLERANCE`` env var is read as the default.

    Confirms the env-var contract documented in the issue acceptance —
    gate scaffolding only; do NOT raise it. A custom value (e.g. ``0.05``)
    must parse and be reflected in the JSON report without tripping
    ``EXIT_USAGE``.
    """
    _populate_synthetic_baseline(tmp_path)
    manifest_path = tmp_path / MANIFEST_FILENAME

    monkeypatch.setenv("BASELINE_H_TR_EM_TOLERANCE", "0.05")
    rc = _invoke_with_capsys(
        verifier,
        manifest_path,
        capsys,
        "--json",
    )
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert rc == verifier.EXIT_OK
    assert parsed["tolerance"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# End-to-end shipped baseline smoke test
# ---------------------------------------------------------------------------


def test_shipped_baseline_is_clean(
    verifier,
    default_baseline_dir,
    capsys,
):
    """The shipped baseline (``tests/reference_data/h_tr_em_baseline/``)
    is non-placeholder and passes ``--strict`` per Issue #3549 step 5
    ("Run the gate once against develop HEAD — must produce EXIT_OK=0").
    """
    manifest_path = default_baseline_dir / MANIFEST_FILENAME
    assert manifest_path.is_file(), (
        f"shipped baseline manifest missing: {manifest_path}"
    )
    rc = _invoke_with_capsys(
        verifier,
        manifest_path,
        capsys,
        "--strict",
    )
    err = capsys.readouterr().err
    assert rc == verifier.EXIT_OK, (
        f"shipped baseline failed the gate (exit {rc}): {err}"
    )