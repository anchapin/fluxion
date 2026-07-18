"""Command-line entry point for the Fluxion Python package.

Subcommands
-----------
- ``fluxion apply-measures`` — AOT measure runner (Issue #1814). Loads a
  base model, runs each discovered :class:`fluxion.FluxionMeasure` subclass
  in sequence, and serializes the mutated model to JSON (or msgpack).

The CLI uses ``argparse`` to match the conventions of ``scripts/*.py`` in
this repository. It is intentionally lightweight — the goal is to provide a
deterministic, scriptable runner for CI; richer argument parsing (choices,
range validation, etc.) lives in the individual measures via
:meth:`FluxionMeasure.arguments`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from fluxion.measures import (
    FluxionMeasure,
    apply_measures,
    discover_measures,
    load_model,
    save_model,
)


logger = logging.getLogger("fluxion.cli")


# =============================================================================
# Argument parsing
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fluxion",
        description=(
            "Fluxion — Neuro-Symbolic Building Energy Modeling CLI. "
            "The Python subcommands live alongside the Rust ones."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (can be repeated).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    apply_p = sub.add_parser(
        "apply-measures",
        help=(
            "Run AOT Python measures against a base model. "
            "Measures mutate the model once and serialize the result."
        ),
        description=(
            "Loads --model, discovers FluxionMeasure subclasses in --measures, "
            "runs each in sequence, and writes the mutated model to --output. "
            "Measures are pre-processors and MUST NOT run inside the "
            "timestepping loop (see docs/measures.md)."
        ),
    )
    apply_p.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to the base model JSON (or .msgpack if msgpack is installed).",
    )
    apply_p.add_argument(
        "--measures",
        required=True,
        type=Path,
        help="Directory containing FluxionMeasure subclasses (*.py files).",
    )
    apply_p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("model.applied.json"),
        help="Output path for the serialized mutated model (default: model.applied.json).",
    )
    apply_p.add_argument(
        "--measure-args",
        type=Path,
        default=None,
        help=(
            "Optional JSON file mapping measure name -> arguments dict. "
            "Missing measures receive an empty dict."
        ),
    )
    apply_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover measures and report what would run, but do not mutate or write.",
    )
    apply_p.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="Discover and list measure classes; do not run them.",
    )

    return parser


# =============================================================================
# Subcommand handlers
# =============================================================================


def _handle_apply_measures(args: argparse.Namespace) -> int:
    """Implement ``fluxion apply-measures``.

    Returns the process exit code (0 on success, non-zero on error).
    """
    # ``--list`` and ``--dry-run`` modes only need the measures directory,
    # not the model — handle them first so the model check doesn't fire.
    measure_classes: list[type[FluxionMeasure]] = []
    if args.measures.exists() and args.measures.is_dir():
        measure_classes = discover_measures(args.measures)
        if not measure_classes:
            logger.warning(
                "No FluxionMeasure subclasses discovered in %s. "
                "Check that your measure files define `class X(FluxionMeasure)` "
                "and override `apply()`.",
                args.measures,
            )
    elif not args.measures.exists():
        logger.error("Measures directory not found: %s", args.measures)
        return 2
    else:
        logger.error("--measures must be a directory, got file: %s", args.measures)
        return 2

    if args.list_only:
        print(json.dumps([c.__name__ for c in measure_classes], indent=2))
        return 0

    # From here on we need the model. Check it before parsing args JSON.
    if not args.model.exists():
        logger.error("Base model not found: %s", args.model)
        return 2

    # Parse optional measure-args JSON
    measure_args: dict[str, dict[str, Any]] = {}
    if args.measure_args is not None:
        if not args.measure_args.exists():
            logger.error("--measure-args file not found: %s", args.measure_args)
            return 2
        try:
            measure_args = json.loads(args.measure_args.read_text())
        except json.JSONDecodeError as e:
            logger.error("Failed to parse --measure-args JSON: %s", e)
            return 2

    if args.dry_run:
        # Instantiate to capture declared arguments.
        plan = []
        for cls in measure_classes:
            instance = cls()
            plan.append(
                {
                    "name": instance.name,
                    "class": cls.__name__,
                    "arguments": instance.arguments(),
                }
            )
        print(json.dumps(plan, indent=2, default=str))
        return 0

    # Load the base model.
    logger.info("Loading base model from %s", args.model)
    try:
        model = load_model(args.model)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return 1

    # Apply each measure in order.
    try:
        applied = apply_measures(model, measure_classes, measure_args)
    except Exception as e:
        logger.error("Measure execution failed: %s", e)
        return 1

    # Serialize the result.
    logger.info("Writing mutated model to %s", args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_model(model, args.output)
    except Exception as e:
        logger.error("Failed to write model: %s", e)
        return 1

    # Append a run-summary section to the output file so downstream
    # tooling can introspect what happened without re-running.
    summary = {
        "applied": applied,
        "output": str(args.output),
        "model": {
            "num_zones": model.num_zones(),
            "surface_count": len(model.surfaces()),
        },
    }
    try:
        existing = json.loads(args.output.read_text())
        if isinstance(existing, dict):
            existing["_fluxion_run"] = summary
            args.output.write_text(json.dumps(existing, indent=2, sort_keys=True))
    except (OSError, json.JSONDecodeError):
        # Best-effort — never fail the CLI on metadata writing.
        pass

    print(json.dumps(summary, indent=2))
    return 0


# =============================================================================
# Entry point
# =============================================================================


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "apply-measures":
        return _handle_apply_measures(args)

    parser.error(f"Unknown command: {args.command}")
    return 2  # unreachable


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
