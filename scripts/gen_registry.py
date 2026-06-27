#!/usr/bin/env python3
"""Generate `tests/surrogate_models/registry.json`.

Iterates over a directory of `.onnx` files and emits one entry per
file with version + sha256 + opset + training_data_hash + metadata.

Usage:
    python3 scripts/gen_registry.py <models-dir> [--output path]

This script is intentionally permissive: it does NOT refuse to emit
placeholders, because the workflow may stage the registry *before* the
real model artefact is in place. The `SurrogateManager::load_version`
path is what enforces hash correctness at runtime — this script only
handles the bookkeeping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models_dir",
        type=Path,
        help="directory containing one or more .onnx files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/surrogate_models/registry.json"),
        help="output registry path (default: tests/surrogate_models/registry.json)",
    )
    parser.add_argument(
        "--trained-on",
        default="1970-01-01",
        help="ISO-8601 trained-on date",
    )
    parser.add_argument(
        "--training-data-summary",
        default="(unset)",
        help="one-line training data summary",
    )
    args = parser.parse_args()

    if not args.models_dir.is_dir():
        print(f"models dir not found: {args.models_dir}", file=sys.stderr)
        return 1

    versions: list[dict] = []
    for path in sorted(args.models_dir.glob("*.onnx")):
        # Derive version from filename stem (strip .onnx). CI is expected
        # to rename artefacts to e.g. `surrogate_v3.1.0.onnx` and place
        # only the canonical one in the dir at release time.
        stem = path.stem
        version = stem.replace("surrogate_", "").replace("surrogate-", "")
        versions.append(
            {
                "version": version,
                "model_sha256": sha256_file(path),
                "onnx_opset_version": 17,
                "training_data_hash": "0" * 64,
                "trained_on": args.trained_on,
                "training_data_summary": args.training_data_summary,
                "expected_accuracy": 0.0,
                "model_path": str(path),
            }
        )

    payload = {
        "_meta": {
            "schema": "fluxion-surrogate-registry/v1",
            "description": (
                "Pinned surrogate model versions for SurrogateManager. "
                "ONNX files themselves are NOT committed to git; see "
                "docs/adr/0004-onnx-model-versioning.md for the deployment contract."
            ),
            "regenerate_with": "python3 scripts/gen_registry.py <path-to-models-dir>",
        },
        "versions": versions,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output} ({len(versions)} versions)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())