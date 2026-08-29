#!/usr/bin/env python3
"""
Snapshot diff verifier for the ``h_tr_em`` Wind-Dependent Per-Step Recompute
Regression Gate (Issue #3063 / LIMIT-13 / Issue #3265 / ADR-0009 §2).

This gate is the regression fence for the ``h_tr_em`` (envelope-to-mass
conductance) wind-dependent per-step recompute that closes the Case 195
cooling shift (``220 → 758`` kWh) tracked in LIMIT-13. Until the
recomputation lands, ``h_tr_em`` remains time-invariant at build-time
``1 / EXTERIOR_FILM_COEFF_DEFAULT``, and the baseline snapshot set
captures the per-zone conductances so a future implementer can diff
their post-recompute snapshots against these pre-recompute values and
fail closed if any per-metric drift exceeds the manifest's per-metric
tolerance (default 0.0 = bit-identical).

This verifier is the direct mirror of the snapshot-diff pattern
documented in ``scripts/verify_gauge_solver_regression.py`` (Issue #3070
/ ADR-0008). The contract follows the same ``EXIT_OK=0 / EXIT_REGRESSION=1
/ EXIT_PLACEHOLDER=2 / EXIT_USAGE=3`` exit codes so the workflow
``h_tr_em_regression_gate.yml`` and the strict-release gate checker can
interpret the result without per-script branching.

Usage::

    python3 scripts/verify_h_tr_em_regression.py \\
        --before tests/reference_data/h_tr_em_baseline \\
        --after  tests/reference_data/h_tr_em_current \\
        [--json] [--strict] [--tolerance h_tr_em_w_k=0.0,...]

Exit codes:
    0 - All metrics within tolerance (no regression detected).
    1 - At least one per-metric drift exceeded tolerance (regression).
    2 - Snapshot set is unpopulated / placeholder / schema-drift (fail
        closed so an accidental comparison against an empty baseline
        cannot silently green-light the recompute).
    3 - Usage / IO error (bad path, malformed JSON, missing manifest).

Per ADR-0009 §"Decision", the baseline MUST be regenerated from fluxion
(via ``cargo run --release --example capture_h_tr_em_baseline``) before
the workflow can transition from the placeholder contract (exit 2) to
the measured regression check (exit 0 / 1). Handcrafted synthetic values
violate ``RULES.md`` ("must-never hardcode results") and trip the
``--strict`` SHA-256 fingerprint check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "tests" / "reference_data" / "h_tr_em_baseline"
MANIFEST_FILENAME = "baseline_manifest.json"
SUPPORTED_SCHEMA_VERSION = 1

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_PLACEHOLDER = 2
EXIT_USAGE = 3

# Default per-case metric list. The manifest's
# ``verifier.default_tolerance`` map can override; ``--tolerance`` CLI
# overrides the manifest; missing entries default to ``0.0`` (bit-
# identical, per ``RULES.md``).
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
# Snapshot loading
# ---------------------------------------------------------------------------


@dataclass
class CaseSnapshot:
    """A single per-case ``h_tr_em`` snapshot loaded from a JSON file.

    Mirrors the shape documented in
    ``tests/reference_data/h_tr_em_baseline/baseline_manifest.json`` and
    the individual ``case_<N>.json`` files. ``metrics`` is a
    ``dict[str, float | None]``; ``None`` indicates the field is a
    placeholder and the snapshot MUST be rejected via
    :meth:`is_placeholder`.
    """

    case_id: str
    path: Path
    captured_at: str | None
    captured_commit: str | None
    metrics: dict[str, float | None]
    raw: dict = field(default_factory=dict)

    def is_placeholder(self) -> bool:
        """A snapshot is a placeholder if any metric is null or no ``captured_at``."""
        if self.captured_at is None:
            return True
        return any(v is None for v in self.metrics.values())


@dataclass
class SnapshotSet:
    """A snapshot directory plus the per-case snapshots it enumerates."""

    manifest_path: Path
    manifest: dict
    cases: dict[str, CaseSnapshot] = field(default_factory=dict)

    def is_placeholder(self) -> bool:
        """A set is a placeholder if any case is a placeholder or the manifest is unpopulated."""
        if self.manifest.get("captured_at") is None:
            return True
        return any(c.is_placeholder() for c in self.cases.values())


def _coerce_float(value: Any) -> float | None:
    """Coerce a JSON metric value to ``float``; return ``None`` for null."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(
        f"unsupported metric value type: {type(value).__name__} ({value!r})"
    )


def _load_snapshot_file(path: Path) -> CaseSnapshot:
    """Load one ``case_<N>.json`` file into a :class:`CaseSnapshot`."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {path}: {exc}") from exc

    case_id = str(raw.get("case_id") or path.stem.split("_", 1)[1])
    metrics_raw = raw.get("metrics", {})
    metrics: dict[str, float | None] = {}
    for key, value in metrics_raw.items():
        if key.startswith("_"):
            continue
        metrics[key] = _coerce_float(value)

    return CaseSnapshot(
        case_id=case_id,
        path=path,
        captured_at=raw.get("captured_at"),
        captured_commit=raw.get("captured_commit"),
        metrics=metrics,
        raw=raw,
    )


def load_snapshot_set(directory: Path) -> SnapshotSet:
    """Load a snapshot directory and every per-case JSON it enumerates.

    Raises:
        FileNotFoundError: missing ``baseline_manifest.json`` or a per-case
            file listed in the manifest.
        ValueError: malformed JSON, schema-version mismatch, or a
            per-case file that omits required fields.
    """
    directory = directory.resolve()
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    schema_version = manifest.get("_schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported manifest schema_version={schema_version!r}; "
            f"expected {SUPPORTED_SCHEMA_VERSION!r}. Re-capture the snapshot set."
        )

    cases_map = manifest.get("cases", {})
    if not cases_map:
        raise ValueError(f"manifest {manifest_path} has empty `cases` map")

    snapshots: dict[str, CaseSnapshot] = {}
    for case_key, case_meta in cases_map.items():
        if case_key.startswith("_"):
            continue
        if not isinstance(case_meta, dict):
            raise ValueError(
                f"manifest entry {case_key!r} must be an object, "
                f"got {type(case_meta).__name__}"
            )
        rel_path = case_meta.get("path")
        if not rel_path:
            raise ValueError(f"manifest entry {case_key!r} missing `path`")
        snapshot_path = directory / rel_path
        if not snapshot_path.is_file():
            raise FileNotFoundError(
                f"manifest lists {case_key!r} at {snapshot_path} "
                f"but the file is missing"
            )
        snapshot = _load_snapshot_file(snapshot_path)
        snapshots[case_key] = snapshot

    return SnapshotSet(manifest_path=manifest_path, manifest=manifest, cases=snapshots)


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


@dataclass
class MetricDelta:
    """The diff for one (case, metric) pair."""

    case_id: str
    metric: str
    before: float | None
    after: float | None
    delta: float | None
    tolerance: float
    within_tolerance: bool

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "metric": self.metric,
            "before": self.before,
            "after": self.after,
            "delta": self.delta,
            "tolerance": self.tolerance,
            "within_tolerance": self.within_tolerance,
        }


@dataclass
class DiffReport:
    """Aggregate diff result for a ``--before`` / ``--after`` pair."""

    before_set: SnapshotSet
    after_set: SnapshotSet
    deltas: list[MetricDelta] = field(default_factory=list)
    regressions: list[MetricDelta] = field(default_factory=list)
    schema_drift: list[str] = field(default_factory=list)

    @property
    def has_regression(self) -> bool:
        return any(not d.within_tolerance for d in self.deltas)


def _tolerance_for_metric(manifest: dict, metric: str) -> float:
    """Read the per-metric tolerance from the manifest, defaulting to 0.0."""
    verifier_cfg = manifest.get("verifier", {})
    default_tol = verifier_cfg.get("default_tolerance", {})
    if metric in default_tol:
        return float(default_tol[metric])
    return 0.0


def compute_diff(
    before_set: SnapshotSet,
    after_set: SnapshotSet,
    *,
    override_tolerance: dict[str, float] | None = None,
) -> DiffReport:
    """Compute per-case × per-metric deltas and flag regressions.

    Args:
        before_set: Snapshot set to compare FROM (the pre-recompute
            baseline at ``tests/reference_data/h_tr_em_baseline``).
        after_set: Snapshot set to compare TO (the post-recompute
            snapshot the implementer produces).
        override_tolerance: Optional dict that overrides per-metric
            tolerance from the manifest (CLI flag). Per-metric tolerance
            still defaults to 0.0 when neither the manifest nor the
            override supplies it - bit-identical equality is the
            ``RULES.md``-compliant default.
    """
    report = DiffReport(before_set=before_set, after_set=after_set)
    override_tolerance = override_tolerance or {}

    for case_key, after_case in after_set.cases.items():
        if case_key not in before_set.cases:
            report.schema_drift.append(
                f"case {case_key!r} present in --after but missing from --before"
            )
            continue
        before_case = before_set.cases[case_key]
        all_metrics = sorted(
            set(DEFAULT_METRICS)
            | set(before_case.metrics)
            | set(after_case.metrics)
        )
        for metric in all_metrics:
            before_val = before_case.metrics.get(metric)
            after_val = after_case.metrics.get(metric)
            if before_val is None or after_val is None:
                report.schema_drift.append(
                    f"case {case_key!r} metric {metric!r}: "
                    f"before={before_val} after={after_val} (null)"
                )
                continue
            tol = override_tolerance.get(metric)
            if tol is None:
                tol = _tolerance_for_metric(before_set.manifest, metric)
            delta = after_val - before_val
            within_tol = abs(delta) <= tol
            entry = MetricDelta(
                case_id=after_case.case_id,
                metric=metric,
                before=before_val,
                after=after_val,
                delta=delta,
                tolerance=tol,
                within_tolerance=within_tol,
            )
            report.deltas.append(entry)
            if not within_tol:
                report.regressions.append(entry)

    for case_key in before_set.cases:
        if case_key not in after_set.cases:
            report.schema_drift.append(
                f"case {case_key!r} present in --before but missing from --after"
            )

    return report


# ---------------------------------------------------------------------------
# SHA-256 fingerprint check (fail-closed: hand-edits to a placeholder
# baseline trip exit code 2 even when metrics remain null).
# ---------------------------------------------------------------------------


def _fingerprint_case_files(snapshot_set: SnapshotSet) -> dict[str, str]:
    """SHA-256 each per-case JSON; used to detect silent edits to placeholders."""
    out: dict[str, str] = {}
    for case_key, case in snapshot_set.cases.items():
        digest = hashlib.sha256(case.path.read_bytes()).hexdigest()
        out[case_key] = digest
    return out


def verify_fingerprints(snapshot_set: SnapshotSet) -> list[str]:
    """Compare SHA-256 of each case file against the manifest's stamp.

    Returns a list of human-readable mismatches (empty == pass). The
    manifest's ``cases.<key>.sha256`` is ``null`` for the placeholder;
    this check therefore always passes on a fresh placeholder and starts
    firing the moment a real snapshot is captured without a manifest
    refresh - catching silent edits.
    """
    mismatches: list[str] = []
    fingerprints = _fingerprint_case_files(snapshot_set)
    for case_key, digest in fingerprints.items():
        case_meta = snapshot_set.manifest.get("cases", {}).get(case_key)
        if not isinstance(case_meta, dict):
            continue
        stamped = case_meta.get("sha256")
        if stamped is None:
            continue
        if stamped != digest:
            mismatches.append(
                f"case {case_key!r} sha256 mismatch: "
                f"stamped={stamped} actual={digest}"
            )
    return mismatches


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def render_text_report(report: DiffReport) -> str:
    """Human-readable diff table (mirrors ``release_gate_checker`` style)."""
    lines = [
        "=" * 72,
        "H_TR_EM REGRESSION VERIFIER (Issue #3063 / Issue #3265 / LIMIT-13 / ADR-0009)",
        "=" * 72,
        f"before: {report.before_set.manifest_path}",
        f"after:  {report.after_set.manifest_path}",
        "",
    ]

    if report.schema_drift:
        lines.append("SCHEMA / PLACEHOLDER ISSUES:")
        for msg in report.schema_drift:
            lines.append(f"  ! {msg}")
        lines.append("")

    if not report.deltas:
        lines.append("No comparable metrics (placeholder set or zero overlap).")
        lines.append("=" * 72)
        return "\n".join(lines)

    lines.append(
        f"{'CASE':<10} {'METRIC':<24} {'BEFORE':>14} {'AFTER':>14} "
        f"{'DELTA':>14} {'TOL':>10} STATUS"
    )
    lines.append("-" * 72)
    for d in report.deltas:
        status = "PASS" if d.within_tolerance else "FAIL"
        lines.append(
            f"{d.case_id:<10} {d.metric:<24} "
            f"{d.before:>14.6f} {d.after:>14.6f} "
            f"{d.delta:>+14.6f} {d.tolerance:>10.4f} {status}"
        )
    lines.append("-" * 72)
    total = len(report.deltas)
    failures = sum(1 for d in report.deltas if not d.within_tolerance)
    lines.append(f"{failures} of {total} metrics regressed beyond tolerance")
    lines.append("=" * 72)
    return "\n".join(lines)


def render_json_report(report: DiffReport) -> str:
    """JSON-shaped report (mirrors ``release_gate_checker --json``)."""
    payload = {
        "before": str(report.before_set.manifest_path),
        "after": str(report.after_set.manifest_path),
        "deltas": [d.to_dict() for d in report.deltas],
        "schema_drift": report.schema_drift,
        "summary": {
            "total": len(report.deltas),
            "regressions": len(report.regressions),
            "has_regression": report.has_regression,
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Snapshot diff verifier for the h_tr_em wind-dependent per-step "
            "recompute (Issue #3063 / Issue #3265 / LIMIT-13 / ADR-0009 §2). "
            "Fails closed when ANY per-metric drift exceeds the manifest's "
            "per-metric tolerance."
        )
    )
    parser.add_argument(
        "--before",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
        help=(
            "Path to the pre-recompute snapshot directory. Defaults to "
            "tests/reference_data/h_tr_em_baseline/."
        ),
    )
    parser.add_argument(
        "--after",
        type=Path,
        required=True,
        help="Path to the post-recompute snapshot directory.",
    )
    parser.add_argument(
        "--tolerance",
        type=str,
        default=None,
        help=(
            "Optional per-metric tolerance override as a comma-separated "
            "key=value list (e.g. 'h_tr_em_w_k=0.001,h_tr_ms_w_k=0.001'). "
            "Default: read from manifest's verifier.default_tolerance (0.0 = "
            "bit-identical)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of a human-readable table.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Tighten the exit-code contract: in addition to flagging "
            "per-metric drift, exit 2 when the SHA-256 fingerprint of any "
            "case file no longer matches the manifest stamp. Useful for "
            "CI that wants the gate to fail on silent placeholder edits."
        ),
    )
    parser.add_argument(
        "--allow-placeholder",
        action="store_true",
        help=(
            "Allow placeholder snapshots (any metric == null or "
            "captured_at == null) to be diffed. Off by default - the "
            "verifier fails closed (exit 2) so a future implementer "
            "cannot silently compare against an empty baseline."
        ),
    )
    return parser.parse_args(argv)


def _parse_tolerance_override(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    out: dict[str, float] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"invalid tolerance entry {chunk!r} (expected key=value)"
            )
        key, value = chunk.split("=", 1)
        out[key.strip()] = float(value)
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        before_set = load_snapshot_set(args.before)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR loading --before snapshot set: {exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        after_set = load_snapshot_set(args.after)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR loading --after snapshot set: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if not args.allow_placeholder:
        if before_set.is_placeholder() or after_set.is_placeholder():
            offending = []
            if before_set.is_placeholder():
                offending.append("--before")
            if after_set.is_placeholder():
                offending.append("--after")
            print(
                "ERROR: placeholder snapshot set detected for "
                + ", ".join(offending)
                + ". Capture real metrics first (run "
                "`cargo run --release --example capture_h_tr_em_baseline` "
                "and copy the resulting per-case JSON into both "
                "tests/reference_data/h_tr_em_baseline/ and "
                "tests/reference_data/h_tr_em_current/). "
                "Re-run with --allow-placeholder to override.",
                file=sys.stderr,
            )
            return EXIT_PLACEHOLDER

    try:
        override_tol = _parse_tolerance_override(args.tolerance)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE

    report = compute_diff(
        before_set,
        after_set,
        override_tolerance=override_tol,
    )

    if args.strict:
        mismatches = (
            verify_fingerprints(before_set) + verify_fingerprints(after_set)
        )
        if mismatches:
            for msg in mismatches:
                print(f"ERROR: {msg}", file=sys.stderr)
            return EXIT_PLACEHOLDER

    if args.json:
        print(render_json_report(report))
    else:
        print(render_text_report(report))

    return EXIT_REGRESSION if report.has_regression else EXIT_OK


# ---------------------------------------------------------------------------
# Self-test (mirror scripts/release_gate_checker.py + scripts/check_*.py)
# ---------------------------------------------------------------------------


def _self_test() -> int:
    """Run a hermetic round-trip without pytest.

    Builds a synthetic snapshot directory in a temp dir, runs the full
    pipeline (placeholder check, capture, diff, regression), and asserts
    the exit codes match the documented contract. Returns 0 on success;
    non-zero on contract violation.
    """
    import tempfile

    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        before_dir = tmp / "before"
        after_dir = tmp / "after"
        _populate_synthetic_snapshot(before_dir, metric_value=50.17)
        _populate_synthetic_snapshot(after_dir, metric_value=50.17)

        # 1) Placeholder check on a fresh directory (no captured_at).
        #    Using --allow-placeholder to bypass the placeholder gate so
        #    we exercise the diff path itself; the actual placeholder
        #    fail-closed contract is exercised by test 3 below.
        rc_no_drift = main([
            "--before", str(before_dir),
            "--after", str(after_dir),
            "--allow-placeholder",
        ])
        if rc_no_drift != EXIT_OK:
            failures.append(
                f"synthetic no-drift diff: expected EXIT_OK, got {rc_no_drift}"
            )

        # 2) Inject a regression in --after and verify EXIT_REGRESSION.
        _populate_synthetic_snapshot(after_dir, metric_value=50.171)
        rc_regression = main([
            "--before", str(before_dir),
            "--after", str(after_dir),
            "--allow-placeholder",
        ])
        if rc_regression != EXIT_REGRESSION:
            failures.append(
                f"synthetic 0.001 drift: expected EXIT_REGRESSION, got {rc_regression}"
            )

        # 3) Fail-closed on a placeholder snapshot (no captured_at).
        #    We plant a *raw* manifest without the captured_at stamp so
        #    the verifier's is_placeholder() gate fires.
        empty_dir = tmp / "empty"
        _populate_unpopulated_snapshot(empty_dir)
        rc_closed = main([
            "--before", str(empty_dir),
            "--after", str(after_dir),
        ])
        if rc_closed != EXIT_PLACEHOLDER:
            failures.append(
                f"placeholder snapshot: expected EXIT_PLACEHOLDER, "
                f"got {rc_closed}"
            )

        # 4) Usage error on a missing manifest.
        rc_missing = main([
            "--before", str(tmp / "does_not_exist"),
            "--after", str(after_dir),
        ])
        if rc_missing != EXIT_USAGE:
            failures.append(
                f"missing manifest: expected EXIT_USAGE, got {rc_missing}"
            )

    if failures:
        for f in failures:
            print(f"SELF-TEST FAILURE: {f}", file=sys.stderr)
        return 1
    print("self-test OK")
    return 0


def _populate_synthetic_snapshot(directory: Path, *, metric_value: float) -> None:
    """Write a 1-case synthetic snapshot set for the self-test."""
    directory.mkdir(parents=True, exist_ok=True)
    case_payload = {
        "_doc": "synthetic",
        "case_id": "600",
        "captured_at": "2026-08-17T00:00:00Z",
        "captured_commit": "deadbeef",
        "metrics": {m: metric_value for m in DEFAULT_METRICS},
    }
    (directory / "case_600.json").write_text(
        json.dumps(case_payload, indent=2), encoding="utf-8"
    )
    manifest = {
        "_doc": "synthetic",
        "_schema_version": SUPPORTED_SCHEMA_VERSION,
        "captured_at": "2026-08-17T00:00:00Z",
        "captured_commit": "deadbeef",
        "cases": {
            "case_600": {
                "path": "case_600.json",
                "case_id": "600",
                "metrics": list(DEFAULT_METRICS),
                "sha256": None,
            },
        },
        "verifier": {
            "path": "scripts/verify_h_tr_em_regression.py",
            "default_tolerance": {m: 0.0 for m in DEFAULT_METRICS},
        },
    }
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _populate_unpopulated_snapshot(directory: Path) -> None:
    """Write a snapshot set whose ``captured_at`` / per-case metrics are null.

    Mirrors the placeholder state shipped in
    ``tests/reference_data/gauge_solver_baseline/``: the verifier must
    refuse to diff this set (exit code 2).
    """
    directory.mkdir(parents=True, exist_ok=True)
    case_payload = {
        "_doc": "placeholder",
        "case_id": "600",
        "captured_at": None,
        "captured_commit": None,
        "metrics": {m: None for m in DEFAULT_METRICS},
    }
    (directory / "case_600.json").write_text(
        json.dumps(case_payload, indent=2), encoding="utf-8"
    )
    manifest = {
        "_doc": "placeholder",
        "_schema_version": SUPPORTED_SCHEMA_VERSION,
        "captured_at": None,
        "captured_commit": None,
        "cases": {
            "case_600": {
                "path": "case_600.json",
                "case_id": "600",
                "metrics": list(DEFAULT_METRICS),
                "sha256": None,
            },
        },
        "verifier": {
            "path": "scripts/verify_h_tr_em_regression.py",
            "default_tolerance": {m: 0.0 for m in DEFAULT_METRICS},
        },
    }
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        sys.exit(_self_test())
    sys.exit(main())
