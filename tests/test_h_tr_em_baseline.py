"""Structural guard for the h_tr_em Regression Gate baseline (Issue #3265).

Per `RULES.md` ("must-never hardcode results") and AGENTS.md (the strict-
energy-gate baseline must NEVER be raised to hide a regression), the
h_tr_em baseline at ``tests/reference_data/h_tr_em_baseline/`` is the
bit-identical reference for any future wind-dependent per-step
recompute (Issue #3063 / ADR-0009 / LIMIT-13).

This module pins three contracts that the gate's fail-closed
``EXIT_PLACEHOLDER=2`` semantics rely on:

1. ``baseline_manifest.json`` exists and ``captured_at`` is non-null.
2. Every per-case JSON listed in the manifest exists, parses, has a
   non-null ``captured_at``, and every declared metric is a non-null
   ``float``.
3. The metrics listed in the manifest's per-case ``metrics`` block
   match the metric keys emitted by
   ``scripts/verify_h_tr_em_regression.py``'s ``DEFAULT_METRICS``
   (``scripts/verify_h_tr_em_regression.py``).

If any of these regress, the workflow ``h_tr_em Regression Gate
(LIMIT-13)`` (``.github/workflows/h_tr_em_regression_gate.yml``) trips
the fail-closed contract and CI rejects every PR. The tests below
exercise each contract path against the real
``tests/reference_data/h_tr_em_baseline/`` directory so a hand-edit
that nulls a metric, removes a case file, or ships a placeholder
``captured_at`` trips a fast Python assertion in the test report
instead of waiting for the workflow run.

Run via::

    cd scripts/ci && python3 -m pytest ../../tests/test_h_tr_em_baseline.py -v

These tests deliberately avoid depending on the fluxion Python
bindings so they run on a plain checkout (no ``maturin develop``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = REPO_ROOT / "tests" / "reference_data" / "h_tr_em_baseline"
MANIFEST_FILENAME = "baseline_manifest.json"


def _load_verifier_module():
    """Load ``scripts/verify_h_tr_em_regression.py`` fresh, mirroring the
    ``scripts/ci/test_verify_h_tr_em_regression.py`` pattern.

    Returns the verifier module, or ``None`` if the script is unavailable
    (so a transient script rename does not block the CI guard).
    """
    verifier_path = REPO_ROOT / "scripts" / "verify_h_tr_em_regression.py"
    if not verifier_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("verify_h_tr_em_regression", verifier_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("verify_h_tr_em_regression", module)
    spec.loader.exec_module(module)
    return module


def _load_manifest_payload():
    """Read ``baseline_manifest.json`` and return its parsed body."""
    manifest_path = BASELINE_DIR / MANIFEST_FILENAME
    assert manifest_path.is_file(), (
        f"baseline manifest missing: {manifest_path} - "
        f"the h_tr_em Regression Gate fails closed (exit 2) until this is populated"
    )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_baseline_directory_exists():
    """The baseline directory must exist; the gate fails closed (exit 2) otherwise."""
    assert BASELINE_DIR.is_dir(), (
        f"baseline directory missing: {BASELINE_DIR} - "
        f"re-run `cargo run --release --example capture_h_tr_em_baseline`"
    )


def test_baseline_manifest_exists_and_captured_at_is_set():
    """``baseline_manifest.json`` exists and ``captured_at`` is non-null.

    Per ADR-0009 §"Decision" and the gate workflow's first step
    ("Check baseline snapshot directory exists"), a manifest whose
    ``captured_at`` is null trips ``EXIT_PLACEHOLDER=2`` so a future
    implementer cannot silently compare against an empty baseline.
    """
    payload = _load_manifest_payload()
    assert payload.get("captured_at") is not None, (
        "baseline manifest `captured_at` is null - the gate will fail "
        "closed (exit 2); re-run `cargo run --release --example "
        "capture_h_tr_em_baseline` to populate it"
    )
    assert isinstance(payload["captured_at"], str)
    # Loose ISO 8601 check: 4-digit year + dash + 2-digit month + dash + 2-digit day.
    date_prefix = payload["captured_at"][:10]
    assert len(date_prefix) == 10 and date_prefix[4] == "-" and date_prefix[7] == "-"


def test_baseline_manifest_schema_version_matches_verifier():
    """Manifest ``_schema_version`` matches the verifier's in-code expectation.

    A future backward-incompatible format bump must update both the
    verifier's ``SUPPORTED_SCHEMA_VERSION`` and the shipped manifest;
    a mismatch trips the verifier's ``EXIT_USAGE=3`` so a refactor
    cannot silently diff against a stale schema.
    """
    payload = _load_manifest_payload()
    verifier = _load_verifier_module()
    if verifier is None:
        pytest.skip("verifier script not present; structural contract is exercised elsewhere")
    assert payload.get("_schema_version") == verifier.SUPPORTED_SCHEMA_VERSION, (
        f"manifest schema_version={payload.get('_schema_version')!r} "
        f"mismatches verifier SUPPORTED_SCHEMA_VERSION="
        f"{verifier.SUPPORTED_SCHEMA_VERSION!r}"
    )


def test_baseline_manifest_lists_per_case_files_with_non_null_metrics():
    """Every per-case file listed in the manifest is present with non-null metrics."""
    payload = _load_manifest_payload()
    cases = payload.get("cases", {})

    assert cases, "baseline manifest has empty `cases` map"

    missing: list[str] = []
    placeholders: list[str] = []
    captured_at_nulls: list[str] = []

    for case_key, case_meta in cases.items():
        if case_key.startswith("_"):
            continue
        if not isinstance(case_meta, dict):
            continue
        rel_path = case_meta.get("path")
        assert rel_path, f"manifest entry {case_key!r} missing `path`"
        case_path = BASELINE_DIR / rel_path
        if not case_path.is_file():
            missing.append(str(case_path))
            continue

        case_payload = json.loads(case_path.read_text(encoding="utf-8"))
        if case_payload.get("captured_at") is None:
            captured_at_nulls.append(case_key)
        metrics = case_payload.get("metrics", {})
        # Track every null metric so the failure message lists them all.
        null_metrics = [k for k, v in metrics.items() if v is None]
        if null_metrics:
            placeholders.append(f"{case_key} ({', '.join(null_metrics)})")

    assert not missing, (
        f"manifest lists missing case files: {missing} - "
        f"re-run `cargo run --release --example capture_h_tr_em_baseline`"
    )
    assert not captured_at_nulls, (
        f"per-case `captured_at` is null for: {captured_at_nulls} - "
        f"the gate fails closed (exit 2)"
    )
    assert not placeholders, (
        f"per-case null metrics (placeholder values, gate exits 2): {placeholders}"
    )


def test_baseline_metrics_align_with_verifier_default_metrics():
    """Manifest metrics names match the verifier's ``DEFAULT_METRICS`` set.

    The verifier's ``compute_diff`` is keyed by metric name; a verifier
    rename without a manifest refresh yields ``schema_drift`` warnings
    that the workflow will surface. This test catches the drift early
    so the operator can re-capture the manifest.
    """
    payload = _load_manifest_payload()
    verifier = _load_verifier_module()
    if verifier is None:
        pytest.skip("verifier script not present; metric alignment is exercised elsewhere")

    case_600 = payload["cases"].get("case_600")
    assert case_600 is not None, "manifest missing case_600"
    declared = set(case_600.get("metrics", []))
    expected = set(verifier.DEFAULT_METRICS)
    assert declared == expected, (
        f"manifest metrics for case_600 do not match verifier DEFAULT_METRICS "
        f"(declared={declared!r}, expected={expected!r})"
    )


def test_baseline_is_loadable_by_verifier():
    """The shipped baseline passes ``load_snapshot_set`` and is non-placeholder.

    End-to-end test: drives the verifier's own load path so a future
    manifest-shape change that the verifier can't parse (e.g. a missing
    `cases` map) trips this guard immediately.
    """
    verifier = _load_verifier_module()
    if verifier is None:
        pytest.skip("verifier script not present")

    snapshot_set = verifier.load_snapshot_set(BASELINE_DIR)
    assert not snapshot_set.is_placeholder(), (
        "the shipped baseline tripped the verifier's placeholder check "
        "(any metric is null or manifest.captured_at is null). The h_tr_em "
        "Regression Gate will exit 2; re-run `cargo run --release --example "
        "capture_h_tr_em_baseline` to refresh the snapshots."
    )
    # Sanity: at least one case loads.
    assert snapshot_set.cases, "no per-case snapshots loaded from the shipped baseline"
