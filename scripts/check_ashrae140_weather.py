#!/usr/bin/env python3
"""CI provenance gate for the ASHRAE 140 canonical-weather binding (round 41).

Asserts, before the strict suites run:
  1. tests/reference_data/weather/ashrae140_canonical_checksums.json exists
     and parses.
  2. The installed canonical files in assets/weather/ashrae140/ match the
     recorded SHA-256 checksums (or the converted EPW exists alongside them).

Exits 0 when the canonical weather route is satisfiable. Exit 1 with a
remedy message otherwise. Run scripts/fetch_ashrae140_weather.py first.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKSUM_FILE = (
    REPO_ROOT
    / "tests"
    / "reference_data"
    / "weather"
    / "ashrae140_canonical_checksums.json"
)
DEST_DIR = REPO_ROOT / "assets" / "weather" / "ashrae140"
EPW_PATH = DEST_DIR / "DRYCOLD_TMY725650TY.epw"


def main() -> int:
    if not CHECKSUM_FILE.is_file():
        print(
            "FATAL: canonical-weather checksum file missing: "
            f"{CHECKSUM_FILE.relative_to(REPO_ROOT)}",
            file=sys.stderr,
        )
        return 1
    guard = json.loads(CHECKSUM_FILE.read_text())

    missing = [n for n in guard["files"] if not (DEST_DIR / n).is_file()]
    if missing:
        print(
            "FATAL: canonical ASHRAE 140 weather files not installed: "
            f"{missing}. Run: python3 scripts/fetch_ashrae140_weather.py "
            "(CI restores them via the actions cache keyed on the checksum "
            "file).",
            file=sys.stderr,
        )
        return 1
    for name, meta in guard["files"].items():
        actual = hashlib.sha256((DEST_DIR / name).read_bytes()).hexdigest()
        if actual != meta["sha256"]:
            print(
                f"FATAL: {name} sha256 mismatch — downloaded file differs from "
                "the recorded provenance. Do not proceed on an unverified "
                "weather input.",
                file=sys.stderr,
            )
            return 1
    if not EPW_PATH.is_file():
        print(
            f"FATAL: converted EPW missing ({EPW_PATH.relative_to(REPO_ROOT)}). "
            "Run: python3 scripts/convert_tmy3_to_epw.py",
            file=sys.stderr,
        )
        return 1
    print("ashrae140 canonical weather: provenance OK (sha256 verified, EPW present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
