#!/usr/bin/env python3
"""Verify the committed Section 6 weather files against the provenance manifest.

Issue #4284. Reads ``assets/weather/provenance-manifest.json`` and re-hashes
every committed Section 6 weather file (WD100-WD600, ``.epw`` and ``.tmy3``).
The check is deliberately split into two independent hash fields:

* ``committed_sha256`` — what this repository stores.
* ``publisher_sha256`` — what ASHRAE published in
  ``140-2023-AccompanyingFiles.zip``.

A committed file that drifts away from its recorded committed hash fails the
run. With ``--publisher-dir`` the run additionally re-hashes a locally
extracted publisher copy and compares it against ``publisher_sha256``, so the
publisher claim can be re-verified end to end by anyone holding the package.

The script never writes to the weather files and never modifies the manifest.

Usage
-----
    python3 scripts/verify_weather_provenance.py
    python3 scripts/verify_weather_provenance.py --publisher-dir "<extracted>/Std140_WD_Files/Normative Materials"
    python3 scripts/verify_weather_provenance.py --manifest path/to/manifest.json

Exit codes
----------
    0 — every committed file matches its recorded committed hash (and, when
        ``--publisher-dir`` is given, its recorded publisher hash).
    1 — a mismatch, a missing file, or a manifest inconsistency was found.
    2 — the manifest could not be read or is malformed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "assets" / "weather" / "provenance-manifest.json"
WEATHER_DIR = REPO_ROOT / "assets" / "weather"


def sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    """Load and shape-check the provenance manifest."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"FAIL: manifest not found: {path}")
    except json.JSONDecodeError as exc:
        sys.exit(f"FAIL: manifest is not valid JSON: {path}: {exc}")
    if not isinstance(doc, dict) or not isinstance(doc.get("files"), list):
        sys.exit(f"FAIL: manifest has no 'files' list: {path}")
    if not doc["files"]:
        sys.exit(f"FAIL: manifest 'files' list is empty: {path}")
    return doc


def check_committed(manifest: dict) -> tuple[int, list[str]]:
    """Re-hash every committed weather file and compare to the manifest."""
    problems: list[str] = []
    ok = 0
    for entry in manifest["files"]:
        name = entry.get("filename")
        recorded = entry.get("committed_sha256")
        if not name or not recorded:
            problems.append(f"manifest row missing 'filename'/'committed_sha256': {entry!r}")
            continue
        path = WEATHER_DIR / name
        if not path.is_file():
            problems.append(f"ABSENT    {name}: not found at {path}")
            continue
        actual = sha256(path)
        if actual != recorded:
            problems.append(
                f"MISMATCH {name}:\n"
                f"           manifest committed_sha256 {recorded}\n"
                f"           actual committed bytes    {actual}"
            )
            continue
        expected_bytes = entry.get("committed_bytes")
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            problems.append(
                f"SIZE      {name}: manifest says {expected_bytes} bytes, "
                f"file is {path.stat().st_size} bytes"
            )
            continue
        ok += 1
    return ok, problems


def check_publisher(manifest: dict, publisher_dir: Path) -> tuple[int, list[str]]:
    """Re-hash an extracted publisher copy and compare to `publisher_sha256`."""
    problems: list[str] = []
    ok = 0
    for entry in manifest["files"]:
        name = entry.get("filename")
        recorded = entry.get("publisher_sha256")
        if not name or not recorded:
            problems.append(
                f"manifest row missing 'filename'/'publisher_sha256': {entry!r}"
            )
            continue
        path = publisher_dir / name
        if not path.is_file():
            problems.append(f"ABSENT    {name}: not found in publisher dir at {path}")
            continue
        actual = sha256(path)
        if actual != recorded:
            problems.append(
                f"MISMATCH {name} (publisher):\n"
                f"           manifest publisher_sha256 {recorded}\n"
                f"           actual publisher bytes    {actual}"
            )
            continue
        ok += 1
    return ok, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"provenance manifest to check (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--publisher-dir",
        type=Path,
        default=None,
        help=(
            "directory holding an extracted ASHRAE copy of the Section 6 "
            "normative weather files; when given, publisher hashes are "
            "re-verified against it"
        ),
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    total = len(manifest["files"])
    print("=== Section 6 weather provenance verification ===")
    print(f"manifest: {args.manifest}")
    print(f"files:    {total}")
    print()

    ok, problems = check_committed(manifest)
    for problem in problems:
        print(f"  {problem}")
    print(f"\ncommitted bytes: {ok}/{total} match their recorded committed_sha256")

    if args.publisher_dir is not None:
        print(f"publisher dir: {args.publisher_dir}")
        if not args.publisher_dir.is_dir():
            print(f"  FAIL: publisher dir does not exist: {args.publisher_dir}")
            return 1
        pub_ok, pub_problems = check_publisher(manifest, args.publisher_dir)
        for problem in pub_problems:
            print(f"  {problem}")
        print(f"publisher bytes: {pub_ok}/{total} match their recorded publisher_sha256")
        if pub_problems:
            problems.extend(pub_problems)

    if problems:
        print("\nFAIL: the committed Section 6 weather files do not match the record.")
        return 1

    print("\nPASS: the committed Section 6 weather files match the provenance record.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
