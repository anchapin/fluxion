#!/usr/bin/env python3
"""
β-phase 30-day soak window gate (Issue #3286).

The GaugeSolver production-path flip (Phase A8, Issue #3291, PR #3482)
intentionally retains the ``gauge-solver`` cargo feature as the
production-path gate pending §LOW-21 closure. The gate is the β-soak
program: 30 consecutive successful nightly runs of the seven ADR-0007
acceptance criteria must be GREEN before the default flip is unblocked
(see ``docs/adr/0007-gauge-solver-structural-work.md``).

This script is the **single source of truth** for the gate contract:

1. It defines the ``beta-soak-state.json`` schema (mirrors the artifact
   uploaded by ``.github/workflows/nightly-ashrae-140-gauge.yml``).
2. It validates a state file against that schema, computes the
   ``unblocked`` / ``remaining`` status, and reports it.
3. In ``--gate enforce`` mode it exits **non-zero** unless
   ``streak >= target`` — the contract a PR that touches the gauge
   default must satisfy.
4. In ``--write-template`` mode it writes a fresh template state file
   (streak=0, target=30) so the schema is reproducible from this
   script alone — the canonical artefact is the schema definition
   below, not the workflow's inline JSON construction.

Counting convention (Issue #3286 + Issue #3354):
  * Only completed runs on ``develop`` count.
  * Any failure resets the streak to zero.
  * The in-flight run is not counted (one-run-conservative — tonight's
    success counts from tomorrow).

Usage
-----

From the repository root::

    # Report the current streak (informational, exits 0).
    python3 scripts/check_beta_soak_gate.py

    # Emit the streak as JSON.
    python3 scripts/check_beta_soak_gate.py --json

    # CI gate: fail unless streak >= 30/30.
    python3 scripts/check_beta_soak_gate.py --gate enforce

    # Initialise a fresh state file (used by the nightly workflow's
    # idempotency boot path — Issue #3286 acceptance criterion "schema
    # visible").
    python3 scripts/check_beta_soak_gate.py --write-template ./beta-soak-state.json

Exit codes:

  0 — gate open (streak >= target) OR --gate not requested AND the
      state file is valid.
  1 — gate closed (streak < target) in ``--gate enforce`` mode.
  2 — script error: state file missing/invalid (in ``--gate`` mode),
      or unparseable CLI arguments.

See ``docs/agents/beta-soak-state-schema.md`` for the full schema,
field semantics, and the gate contract rationale.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Issue #3286: the β-soak target is hard-coded to 30 nightly runs.
DEFAULT_TARGET = 30
# State-schema version. Bumped when a field is added/removed/renamed.
# The nightly workflow (``.github/workflows/nightly-ashrae-140-gauge.yml``)
# writes schema_version when called via ``check_beta_soak_gate.py`` —
# pre-#3286 artefacts lack it and are accepted as version "1" by the
# validator below (back-compat).
SCHEMA_VERSION = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE_PATH = REPO_ROOT / "beta-soak-state.json"


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------
#
# The canonical schema lives in this script. The
# ``docs/agents/beta-soak-state-schema.md`` documentation mirrors these
# fields; both must be updated together (the docs-summaries check + the
# validator below will catch drift on the next CI run).
#
# The state file is the durable artefact produced by the nightly
# workflow (``.github/workflows/nightly-ashrae-140-gauge.yml``, "Compute
# β-soak streak" step) and uploaded as a 90-day artifact. It is also
# the input to the gate check below.

REQUIRED_FIELDS_V1: frozenset[str] = frozenset(
    {
        "streak",
        "target",
        "unblocked",
        "generated_at",
    }
)
OPTIONAL_FIELDS_V1: frozenset[str] = frozenset(
    {
        "schema_version",
        "remaining",
        "runs_considered",
        "last_run_id",
        "first_run_id",
        "oldest_run_at",
    }
)
ALLOWED_FIELDS_V1: frozenset[str] = REQUIRED_FIELDS_V1 | OPTIONAL_FIELDS_V1


@dataclass
class BetaSoakState:
    """Validated ``beta-soak-state.json`` payload (Issue #3286)."""

    streak: int
    target: int
    unblocked: bool
    generated_at: str
    schema_version: str = SCHEMA_VERSION
    remaining: Optional[int] = None
    runs_considered: Optional[int] = None
    last_run_id: Optional[str] = None
    first_run_id: Optional[str] = None
    oldest_run_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict, dropping None optionals."""
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "streak": self.streak,
            "target": self.target,
            "unblocked": self.unblocked,
            "generated_at": self.generated_at,
        }
        if self.remaining is not None:
            out["remaining"] = self.remaining
        if self.runs_considered is not None:
            out["runs_considered"] = self.runs_considered
        if self.last_run_id is not None:
            out["last_run_id"] = self.last_run_id
        if self.first_run_id is not None:
            out["first_run_id"] = self.first_run_id
        if self.oldest_run_at is not None:
            out["oldest_run_at"] = self.oldest_run_at
        return out


@dataclass
class ValidationFailure(Exception):
    """Raised when a state file fails schema validation."""

    reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return "beta-soak-state.json invalid: " + "; ".join(self.reasons)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


_ISO_8601_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+\-]\d{2}:\d{2})$"
)


def _validate_int_in_range(value: Any, name: str, lo: int, hi: int) -> Optional[str]:
    if not isinstance(value, int) or isinstance(value, bool):
        return f"{name} must be an integer (got {type(value).__name__})"
    if value < lo or value > hi:
        return f"{name}={value} outside [{lo}, {hi}]"
    return None


def _validate_str(value: Any, name: str) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return f"{name} must be a non-empty string (got {type(value).__name__})"
    return None


def _validate_optional_int(value: Any, name: str, lo: int, hi: int) -> Optional[str]:
    if value is None:
        return None
    return _validate_int_in_range(value, name, lo, hi)


def _validate_optional_str(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _validate_str(value, name)


def _require_int(
    payload: dict[str, Any], name: str, lo: int, hi: int, reasons: list[str]
) -> Optional[int]:
    """Validate that ``payload[name]`` is an int in [lo, hi] or append
    a diagnostic to ``reasons``. Returns the validated int (narrowed)
    or None when validation failed."""
    raw = payload.get(name)
    msg = _validate_int_in_range(raw, name, lo, hi)
    if msg is not None:
        reasons.append(msg)
        return None
    assert isinstance(raw, int) and not isinstance(raw, bool)
    return raw


def _require_bool(
    payload: dict[str, Any], name: str, reasons: list[str]
) -> Optional[bool]:
    """Validate that ``payload[name]`` is a bool or append a diagnostic.

    Returns the validated bool (narrowed) or None on failure.
    """
    raw = payload.get(name)
    if not isinstance(raw, bool):
        reasons.append(f"{name} must be a boolean (got {type(raw).__name__})")
        return None
    return raw


def _require_str(
    payload: dict[str, Any], name: str, reasons: list[str]
) -> Optional[str]:
    """Validate that ``payload[name]`` is a non-empty str or append a
    diagnostic. Returns the validated string (narrowed) or None on failure."""
    raw = payload.get(name)
    msg = _validate_str(raw, name)
    if msg is not None:
        reasons.append(msg)
        return None
    assert isinstance(raw, str) and raw
    return raw


def validate_state(payload: Any) -> BetaSoakState:
    """Validate ``payload`` against the v1 schema; raise on failure.

    Back-compat: pre-#3286 state files do not carry ``schema_version`` —
    they are accepted as v1 if every required field is present. The
    nightly workflow's inline construction (current as of this commit)
    omits ``schema_version`` and so relies on this back-compat path; the
    ``--write-template`` flow writes ``schema_version`` explicitly.
    """
    if not isinstance(payload, dict):
        raise ValidationFailure(
            [f"top-level must be an object, got {type(payload).__name__}"]
        )

    reasons: list[str] = []

    extra = set(payload.keys()) - ALLOWED_FIELDS_V1
    if extra:
        reasons.append(f"unknown fields: {sorted(extra)}")

    missing = REQUIRED_FIELDS_V1 - set(payload.keys())
    if missing:
        reasons.append(f"missing required fields: {sorted(missing)}")

    streak = _require_int(payload, "streak", 0, 10_000, reasons)
    target = _require_int(payload, "target", 1, 10_000, reasons)
    unblocked = _require_bool(payload, "unblocked", reasons)
    generated_at = _require_str(payload, "generated_at", reasons)

    if generated_at is not None and not _ISO_8601_RE.match(generated_at):
        reasons.append(
            f"generated_at={generated_at!r} is not a valid ISO-8601 timestamp"
        )

    remaining_raw = payload.get("remaining")
    msg = _validate_optional_int(remaining_raw, "remaining", 0, 10_000)
    if msg:
        reasons.append(msg)
    runs_considered_raw = payload.get("runs_considered")
    msg = _validate_optional_int(runs_considered_raw, "runs_considered", 0, 10_000)
    if msg:
        reasons.append(msg)
    last_run_id_raw = payload.get("last_run_id")
    msg = _validate_optional_str(last_run_id_raw, "last_run_id")
    if msg:
        reasons.append(msg)
    first_run_id_raw = payload.get("first_run_id")
    msg = _validate_optional_str(first_run_id_raw, "first_run_id")
    if msg:
        reasons.append(msg)
    oldest_run_at_raw = payload.get("oldest_run_at")
    msg = _validate_optional_str(oldest_run_at_raw, "oldest_run_at")
    if msg:
        reasons.append(msg)

    sv = payload.get("schema_version")
    if sv is not None and sv != SCHEMA_VERSION:
        reasons.append(f"schema_version={sv!r} not supported (only {SCHEMA_VERSION!r})")

    # Cross-field invariants — only enforce when the underlying values
    # are themselves well-typed, so the diagnostic isolates the field
    # error from the cross-field error.
    if (
        streak is not None
        and target is not None
        and unblocked is not None
        and generated_at is not None
    ):
        if unblocked is not (streak >= target):
            reasons.append(
                f"unblocked={unblocked} inconsistent with "
                f"streak={streak} >= target={target}"
            )
        if isinstance(remaining_raw, int) and not isinstance(remaining_raw, bool):
            expected_remaining = max(0, target - streak)
            if remaining_raw != expected_remaining:
                reasons.append(
                    f"remaining={remaining_raw} inconsistent with "
                    f"max(0, target={target} - streak={streak})={expected_remaining}"
                )

    if reasons:
        raise ValidationFailure(reasons)

    assert generated_at is not None  # required field check above
    return BetaSoakState(
        schema_version=SCHEMA_VERSION,
        streak=streak if streak is not None else 0,
        target=target if target is not None else DEFAULT_TARGET,
        unblocked=unblocked if unblocked is not None else False,
        generated_at=generated_at,
        remaining=remaining_raw if isinstance(remaining_raw, int) else None,
        runs_considered=(
            runs_considered_raw if isinstance(runs_considered_raw, int) else None
        ),
        last_run_id=last_run_id_raw if isinstance(last_run_id_raw, str) else None,
        first_run_id=(first_run_id_raw if isinstance(first_run_id_raw, str) else None),
        oldest_run_at=(
            oldest_run_at_raw if isinstance(oldest_run_at_raw, str) else None
        ),
    )


def load_state(path: Path) -> BetaSoakState:
    """Read and validate the state file at ``path``.

    Surfaces a precise diagnostic for the common case where the file is
    missing or its schema does not match — Issue #3286 acceptance
    criterion "schema visible".
    """
    if not path.exists():
        raise ValidationFailure(
            [
                f"state file missing: {path}. Run "
                f"`python3 scripts/check_beta_soak_gate.py --write-template {path}` "
                f"to initialise a fresh state template (streak=0/{DEFAULT_TARGET})."
            ]
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationFailure([f"could not read {path}: {exc}"]) from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationFailure(
            [f"invalid JSON: {exc.msg} (line {exc.lineno})"]
        ) from exc

    return validate_state(payload)


def write_template(path: Path, target: int = DEFAULT_TARGET) -> BetaSoakState:
    """Write a fresh state template (streak=0, target=DEFAULT_TARGET).

    Used by the nightly workflow's idempotency boot path and by humans
    inspecting the schema for the first time. Returns the state that
    was written so the caller can echo it.
    """
    now = datetime.now(timezone.utc).isoformat()
    state = BetaSoakState(
        streak=0,
        target=target,
        unblocked=False,
        generated_at=now,
        remaining=target,
        runs_considered=0,
        last_run_id=None,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2) + "\n", encoding="utf-8")
    return state


# ---------------------------------------------------------------------------
# Gate contract
# ---------------------------------------------------------------------------


def gate_status(state: BetaSoakState) -> dict[str, Any]:
    """Compute the canonical gate-status payload for ``state``.

    Returns a dict with the streak / target / unblocked / remaining
    keys, suitable for JSON emission. The ``gate_open`` field is the
    boolean the CI gate fires on (``--gate enforce`` mode).
    """
    return {
        "schema_version": state.schema_version,
        "streak": state.streak,
        "target": state.target,
        "gate_open": state.unblocked,
        "unblocked": state.unblocked,
        "remaining": max(0, state.target - state.streak),
        "runs_considered": state.runs_considered,
        "last_run_id": state.last_run_id,
        "generated_at": state.generated_at,
    }


def gate_open(state: BetaSoakState) -> bool:
    """Return True iff the soak has reached the target streak."""
    return state.unblocked


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="β-phase 30-day soak window gate (Issue #3286).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes:\n"
            "  0 — gate open OR --gate not requested AND the state file is valid.\n"
            "  1 — gate closed (streak < target) in --gate enforce mode.\n"
            "  2 — script error: state file missing/invalid in --gate mode,\n"
            "      or unparseable CLI arguments.\n"
            "\n"
            "Schema documentation: docs/agents/beta-soak-state-schema.md\n"
        ),
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help="Path to beta-soak-state.json (default: %(default)s)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help="Override the soak target (default: %(default)s)",
    )
    parser.add_argument(
        "--gate",
        choices=["enforce", "report"],
        default="report",
        help=(
            "Gate mode. 'enforce' exits 1 when streak < target "
            "(use this in CI for PRs that touch the gauge default); "
            "'report' always exits 0 (default)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the gate status as JSON on stdout.",
    )
    parser.add_argument(
        "--write-template",
        type=Path,
        metavar="PATH",
        help=(
            "Write a fresh state template (streak=0, target=--target) "
            "to PATH and exit. Useful for bootstrapping the workflow's "
            "idempotency boot path."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.target < 1:
        print(f"ERROR: --target must be >= 1, got {args.target}", file=sys.stderr)
        return 2

    if args.write_template is not None:
        state = write_template(args.write_template, target=args.target)
        if args.json:
            print(json.dumps(gate_status(state), indent=2))
        else:
            print(
                f"Wrote β-soak state template to {args.write_template} "
                f"(streak=0/{state.target}, generated_at={state.generated_at})"
            )
        return 0

    try:
        state = load_state(args.state)
    except ValidationFailure as exc:
        if args.gate == "enforce":
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(
                json.dumps(
                    {"valid": False, "error": str(exc), "path": str(args.state)},
                    indent=2,
                )
            )
        else:
            print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    status = gate_status(state)
    if args.json:
        print(json.dumps({"valid": True, **status}, indent=2))
    else:
        verdict = "OPEN — flip PR unblocked" if status["gate_open"] else "CLOSED"
        print(
            f"β-soak streak: {status['streak']}/{status['target']} "
            f"({status['remaining']} to go) — gate {verdict}"
        )
        if status["last_run_id"]:
            print(f"  last counted run: {status['last_run_id']}")
        if status["generated_at"]:
            print(f"  generated_at:     {status['generated_at']}")

    if args.gate == "enforce" and not gate_open(state):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
