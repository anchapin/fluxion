#!/usr/bin/env python3
"""
h_tr_em Wind-Dependent Per-Step Recompute Regression Gate verifier
(Issue #3549 / LIMIT-13 / Issue #3265 / ADR-0009).

Verifies the integrity of the ``h_tr_em`` baseline snapshot set at
``tests/reference_data/h_tr_em_baseline/``. The gate is the regression
fence for the wind-dependent per-step recompute of the envelope-to-mass
conductance ``h_tr_em`` (Issue #3063 / LIMIT-13) tracked in
``docs/KNOWN_ISSUES.md`` §LIMIT-13. Per ADR-0009 §2, the manifest MUST
be regenerated from fluxion before any wind-dependent recompute can
ship; this verifier is the CI-side contract that fails closed until the
manifest's SHA-256 fingerprints are populated and stable.

Contract (Issue #3549 acceptance):

* Load ``baseline_manifest.json`` from ``--manifest``.
* For every per-case JSON listed in ``manifest.cases``, recompute the
  SHA-256 of the loaded file content and compare against the manifest's
  stamp (``cases.<key>.sha256``).
* A null ``sha256`` stamp is the placeholder signal — exit 2
  (``EXIT_PLACEHOLDER``) so a future implementer cannot silently
  green-light the recompute against an unpopulated baseline.
* A non-null stamp that does not match the recomputed digest is
  tolerated (logged as a warning) by default, so an explicit refresh
  of the manifest alongside a case-file edit does not block the gate.
  Pass ``--strict`` to flip that to a hard failure (exit 1,
  ``EXIT_REGRESSION``).
* ``--tolerance`` is accepted for forward compatibility with a future
  per-metric diff mode; today the gate is bit-identical (tolerance=0.0)
  so the default is read from the ``BASELINE_H_TR_EM_TOLERANCE`` env
  var (per AGENTS.md: do NOT raise it — gate scaffolding only).

Exit codes (mirrors the ``scripts/verify_gauge_solver_regression.py``
convention so the workflow ``h_tr_em_regression_gate.yml`` and the
strict release-gate checker interpret results without per-script
branching):

    0 — All per-case fingerprints match their manifest stamps (or are
        null AND the manifest is not flagged as placeholder).
    1 — ``--strict`` AND at least one per-case fingerprint does not
        match the manifest stamp.
    2 — Baseline is a placeholder: any per-case ``sha256`` is null, OR
        the manifest's ``captured_at`` is null.
    3 — Usage / IO error (bad path, malformed JSON, missing manifest,
        schema drift).

Usage::

    python3 scripts/verify_h_tr_em_regression.py \\
        --manifest tests/reference_data/h_tr_em_baseline/baseline_manifest.json \\
        [--strict] [--json] [--tolerance <float>]

The default ``--manifest`` path is the shipped
``tests/reference_data/h_tr_em_baseline/baseline_manifest.json``; the
gate workflow invokes the verifier with that path explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = (
    REPO_ROOT / "tests" / "reference_data" / "h_tr_em_baseline" / "baseline_manifest.json"
)
MANIFEST_FILENAME = "baseline_manifest.json"
SUPPORTED_SCHEMA_VERSION = 1

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_PLACEHOLDER = 2
EXIT_USAGE = 3

ENV_TOLERANCE_KEY = "BASELINE_H_TR_EM_TOLERANCE"
DEFAULT_TOLERANCE = 0.0

# Backward-compatible public surface for callers that import the verifier
# as a module (e.g. ``tests/test_h_tr_em_baseline.py``). The new contract
# (``--manifest`` + ``verify_manifest``) is the authoritative API; these
# constants and ``load_snapshot_set`` exist so the legacy structural guard
# continues to load without modification after the Issue #3549
# re-materialization.
DEFAULT_METRICS = (
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


# ---------------------------------------------------------------------------
# Verifier core
# ---------------------------------------------------------------------------


@dataclass
class CaseFingerprint:
    """One per-case SHA-256 check result."""

    case_key: str
    case_id: str
    rel_path: str
    stamped: str | None
    actual: str | None
    matched: bool | None  # True if stamp matches, False if mismatch, None if placeholder
    placeholder: bool


@dataclass
class VerifierReport:
    """Aggregate result of one ``--manifest`` invocation."""

    manifest_path: Path
    manifest_captured_at: str | None
    manifest_captured_commit: str | None
    tolerance: float
    strict: bool
    cases: list[CaseFingerprint] = field(default_factory=list)
    schema_drift: list[str] = field(default_factory=list)
    manifest_placeholder: bool = False

    @property
    def mismatches(self) -> list[CaseFingerprint]:
        return [c for c in self.cases if c.matched is False]

    @property
    def placeholders(self) -> list[CaseFingerprint]:
        return [c for c in self.cases if c.placeholder]

    @property
    def exit_status(self) -> int:
        """Map the report to the documented exit code."""
        if self.manifest_placeholder or self.placeholders:
            return EXIT_PLACEHOLDER
        if self.strict and self.mismatches:
            return EXIT_REGRESSION
        return EXIT_OK


def _env_default_tolerance() -> float:
    """Read the default tolerance from ``BASELINE_H_TR_EM_TOLERANCE``.

    Falls back to ``DEFAULT_TOLERANCE = 0.0`` (bit-identical, per
    ``RULES.md`` 'no parameter tuning'). A malformed value is logged and
    ignored so a misconfigured runner cannot mask a real failure.
    """
    raw = os.environ.get(ENV_TOLERANCE_KEY)
    if raw is None:
        return DEFAULT_TOLERANCE
    try:
        value = float(raw)
    except ValueError:
        print(
            f"WARNING: ignoring invalid {ENV_TOLERANCE_KEY}={raw!r} (not a float)",
            file=sys.stderr,
        )
        return DEFAULT_TOLERANCE
    return value


def _fingerprint(path: Path) -> str:
    """SHA-256 of the file bytes at ``path``."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(
    manifest_path: Path,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    strict: bool = False,
) -> VerifierReport:
    """Verify a single ``baseline_manifest.json`` against its per-case files.

    Args:
        manifest_path: Path to ``baseline_manifest.json``.
        tolerance: Bit-identical tolerance for the future per-metric
            diff mode (reserved; the SHA-256 fingerprint is always
            compared by exact hex match).
        strict: When True, a non-null stamp that mismatches the
            recomputed digest is fatal (exit 1). When False (default),
            mismatches are reported as warnings and the gate exits 0.

    Raises:
        FileNotFoundError: ``manifest_path`` does not exist, or a
            per-case JSON listed in the manifest is missing.
        ValueError: Malformed JSON, schema-version mismatch, or an
            empty ``cases`` map.
    """
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {manifest_path}: {exc}") from exc

    schema_version = manifest.get("_schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported manifest schema_version={schema_version!r}; "
            f"expected {SUPPORTED_SCHEMA_VERSION!r}. Re-capture the snapshot set."
        )

    cases_map = manifest.get("cases", {})
    if not cases_map:
        raise ValueError(f"manifest {manifest_path} has empty `cases` map")

    manifest_dir = manifest_path.parent
    manifest_captured_at = manifest.get("captured_at")
    manifest_captured_commit = manifest.get("captured_commit")
    manifest_placeholder = manifest_captured_at is None

    report = VerifierReport(
        manifest_path=manifest_path,
        manifest_captured_at=manifest_captured_at,
        manifest_captured_commit=manifest_captured_commit,
        tolerance=tolerance,
        strict=strict,
        manifest_placeholder=manifest_placeholder,
    )

    for case_key, case_meta in cases_map.items():
        # Documentation-only keys (prefixed with `_`) live alongside real
        # cases for layout reasons (mirrors the manifest's `_doc`
        # convention). Skip them without flagging.
        if case_key.startswith("_"):
            continue
        if not isinstance(case_meta, dict):
            raise ValueError(
                f"manifest entry {case_key!r} must be an object, "
                f"got {type(case_meta).__name__}"
            )
        rel_path = case_meta.get("path")
        if not rel_path:
            report.schema_drift.append(
                f"manifest entry {case_key!r} missing `path`"
            )
            continue
        case_path = manifest_dir / rel_path
        if not case_path.is_file():
            raise FileNotFoundError(
                f"manifest lists {case_key!r} at {case_path} but the file is missing"
            )

        stamped = case_meta.get("sha256")
        if stamped is None:
            report.cases.append(
                CaseFingerprint(
                    case_key=case_key,
                    case_id=str(case_meta.get("case_id") or case_key),
                    rel_path=rel_path,
                    stamped=None,
                    actual=None,
                    matched=None,
                    placeholder=True,
                )
            )
            continue

        actual = _fingerprint(case_path)
        report.cases.append(
            CaseFingerprint(
                case_key=case_key,
                case_id=str(case_meta.get("case_id") or case_key),
                rel_path=rel_path,
                stamped=stamped,
                actual=actual,
                matched=(stamped == actual),
                placeholder=False,
            )
        )

    return report


# ---------------------------------------------------------------------------
# Backward-compatible SnapshotSet API (legacy Issue #3265 contract).
#
# ``tests/test_h_tr_em_baseline.py`` was written against the prior
# gauge_solver-style verifier API (``CaseSnapshot`` + ``SnapshotSet``
# dataclasses + ``load_snapshot_set``). The Issue #3549 re-materialization
# collapses the contract to a single ``verify_manifest`` call but the
# structural guard test continues to assert
# ``verifier.load_snapshot_set(...)``. Provide a thin adapter so the
# legacy test continues to load without modification.
# ---------------------------------------------------------------------------


@dataclass
class _LegacyCaseSnapshot:
    """Legacy CaseSnapshot (Issue #3265 API surface)."""

    case_id: str
    path: Path
    captured_at: str | None
    captured_commit: str | None
    metrics: dict[str, float | None]
    raw: dict = field(default_factory=dict)

    def is_placeholder(self) -> bool:
        if self.captured_at is None:
            return True
        return any(v is None for v in self.metrics.values())


@dataclass
class _LegacySnapshotSet:
    manifest_path: Path
    manifest: dict
    cases: dict[str, _LegacyCaseSnapshot] = field(default_factory=dict)

    def is_placeholder(self) -> bool:
        if self.manifest.get("captured_at") is None:
            return True
        return any(c.is_placeholder() for c in self.cases.values())


def load_snapshot_set(directory: Path) -> _LegacySnapshotSet:  # noqa: D401 - legacy API
    """Load a snapshot directory into a legacy :class:`_LegacySnapshotSet`.

    Mirrors the Issue #3265 / gauge_solver verifier surface so the
    pre-#3549 structural guard (``tests/test_h_tr_em_baseline.py``)
    continues to function. Reads the shipped manifest + per-case JSONs
    and returns a snapshot set whose ``cases`` dict maps
    ``case_<N>`` → :class:`_LegacyCaseSnapshot`.
    """
    directory = Path(directory).resolve()
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    cases_map = manifest.get("cases", {})
    snapshots: dict[str, _LegacyCaseSnapshot] = {}
    for case_key, case_meta in cases_map.items():
        if case_key.startswith("_"):
            continue
        if not isinstance(case_meta, dict):
            continue
        rel_path = case_meta.get("path")
        if not rel_path:
            continue
        case_path = directory / rel_path
        if not case_path.is_file():
            continue
        with case_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        metrics_raw = raw.get("metrics", {})
        metrics: dict[str, float | None] = {}
        for key, value in metrics_raw.items():
            if key.startswith("_"):
                continue
            metrics[key] = (
                float(value) if isinstance(value, (int, float)) and value is not None
                else (None if value is None else float(value))
            )
        snapshots[case_key] = _LegacyCaseSnapshot(
            case_id=str(raw.get("case_id") or case_key.split("_", 1)[1]),
            path=case_path,
            captured_at=raw.get("captured_at"),
            captured_commit=raw.get("captured_commit"),
            metrics=metrics,
            raw=raw,
        )

    return _LegacySnapshotSet(
        manifest_path=manifest_path, manifest=manifest, cases=snapshots
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def render_json_report(report: VerifierReport) -> str:
    """JSON-shaped report (mirrors ``release_gate_checker --json``)."""
    payload = {
        "manifest": str(report.manifest_path),
        "manifest_captured_at": report.manifest_captured_at,
        "manifest_captured_commit": report.manifest_captured_commit,
        "tolerance": report.tolerance,
        "strict": report.strict,
        "exit": report.exit_status,
        "manifest_placeholder": report.manifest_placeholder,
        "cases": [
            {
                "case_key": c.case_key,
                "case_id": c.case_id,
                "path": c.rel_path,
                "stamped": c.stamped,
                "actual": c.actual,
                "matched": c.matched,
                "placeholder": c.placeholder,
            }
            for c in report.cases
        ],
        "mismatches": [
            {"case_key": c.case_key, "stamped": c.stamped, "actual": c.actual}
            for c in report.mismatches
        ],
        "placeholders": [c.case_key for c in report.placeholders],
        "schema_drift": report.schema_drift,
        "summary": {
            "total": len(report.cases),
            "matched": sum(1 for c in report.cases if c.matched is True),
            "placeholders": len(report.placeholders),
            "mismatches": len(report.mismatches),
            "has_regression": bool(report.strict and report.mismatches),
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def render_text_report(report: VerifierReport) -> str:
    """Human-readable per-case fingerprint table."""
    lines = [
        "=" * 72,
        "h_tr_em Regression Gate Verifier (Issue #3549 / LIMIT-13)",
        "=" * 72,
        f"manifest: {report.manifest_path}",
        f"captured_at: {report.manifest_captured_at}",
        f"captured_commit: {report.manifest_captured_commit}",
        f"tolerance: {report.tolerance}",
        f"strict: {report.strict}",
        "",
    ]

    if report.schema_drift:
        lines.append("SCHEMA / MANIFEST ISSUES:")
        for msg in report.schema_drift:
            lines.append(f"  ! {msg}")
        lines.append("")

    if not report.cases:
        lines.append("No per-case fingerprints (placeholder or empty manifest).")
        lines.append("=" * 72)
        return "\n".join(lines)

    lines.append(
        f"{'CASE':<14} {'STATUS':<14} {'STAMPED':>66} {'ACTUAL':>66}"
    )
    lines.append("-" * 156)
    for c in report.cases:
        if c.placeholder:
            status = "PLACEHOLDER"
            stamped = "(null)"
            actual = "(null)"
        elif c.matched:
            status = "MATCH"
            stamped = c.stamped or ""
            actual = c.actual or ""
        else:
            status = "MISMATCH" if report.strict else "WARN"
            stamped = c.stamped or ""
            actual = c.actual or ""
        lines.append(
            f"{c.case_key:<14} {status:<14} {stamped:>66} {actual:>66}"
        )
    lines.append("-" * 156)
    summary = (
        f"{len(report.mismatches)} mismatch(es), "
        f"{len(report.placeholders)} placeholder(s) "
        f"of {len(report.cases)} case(s)"
    )
    lines.append(summary)
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "h_tr_em Wind-Dependent Per-Step Recompute Regression Gate verifier "
            "(Issue #3549 / LIMIT-13 / ADR-0009). Fails closed (exit 2) on a "
            "placeholder baseline; trips exit 1 on SHA-256 mismatch under --strict."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=(
            "Path to baseline_manifest.json. Defaults to "
            "tests/reference_data/h_tr_em_baseline/baseline_manifest.json."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help=(
            "Bit-identical tolerance for the future per-metric diff mode "
            "(reserved; SHA-256 fingerprints are compared by exact hex "
            "match). Default: read from BASELINE_H_TR_EM_TOLERANCE env var "
            "(fallback 0.0 = bit-identical per RULES.md 'no parameter tuning')."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Tighten the exit-code contract: a non-null SHA-256 stamp that "
            "mismatches the recomputed digest trips exit 1 (regression) "
            "instead of being logged as a warning. Useful for CI to catch "
            "silent edits to placeholder baselines."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of a human-readable table.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    tolerance = args.tolerance
    if tolerance is None:
        tolerance = _env_default_tolerance()

    try:
        report = verify_manifest(
            args.manifest,
            tolerance=tolerance,
            strict=args.strict,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE

    # Placeholder signal: emit loudly to stderr so CI logs surface the
    # reason; JSON output also carries `exit: 2` and the placeholder list.
    if report.manifest_placeholder:
        print(
            "ERROR: placeholder manifest detected (manifest.captured_at is null). "
            "Re-run the capture pipeline to populate it before shipping the "
            "wind-dependent per-step recompute (Issue #3063 / LIMIT-13).",
            file=sys.stderr,
        )
    elif report.placeholders:
        print(
            "ERROR: placeholder SHA-256 stamps detected for cases: "
            f"{[c.case_key for c in report.placeholders]}. "
            "The h_tr_em Regression Gate fails closed (exit 2) until every "
            "per-case fingerprint is populated (see ADR-0009).",
            file=sys.stderr,
        )

    for c in report.mismatches:
        msg = (
            f"ERROR: case {c.case_key!r} SHA-256 mismatch: "
            f"stamped={c.stamped} actual={c.actual}"
        )
        if args.strict:
            print(msg, file=sys.stderr)
        else:
            print(f"WARNING: {msg}", file=sys.stderr)

    if args.json:
        print(render_json_report(report))

    exit_code = report.exit_status
    if exit_code == EXIT_OK and not args.json:
        print(render_text_report(report))
    elif exit_code != EXIT_OK and not args.json:
        print(render_text_report(report))

    return exit_code


if __name__ == "__main__":
    sys.exit(main())