#!/usr/bin/env python3
"""Fetch the ASHRAE 140-2023 canonical weather inputs (loop round 41).

Downloads ASHRAE's free supplemental-files package, extracts the two
canonical weather inputs for the §5.2.1–5.2.3 validation program, verifies
their SHA-256 against the in-repo provenance guard, and installs them into
the git-ignored cache directory ``assets/weather/ashrae140/``:

- ``DRYCOLD.TMY725650TY.CSV``  (normative weather, §5.2.1.1.1.1 / A1.1.1.1)
- ``Tsky-Informative.xlsx``    (informative modified Martin–Berdahl Tsky)

The files are fetched at build/test time and never committed (the underlying
TMY3 data are NREL public domain; the ASHRAE packaging is distributed freely
via ASHRAE's supplemental-files host — no membership gate, no robots
restriction observed on the direct package URL at adoption time, 2026-10-15).

Usage:
    python3 scripts/fetch_ashrae140_weather.py [--skip-download]

``--skip-download`` re-verifies already-installed files (offline CI restore
path); the download source is a 704 MB zip, so CI caches the extracted
outputs keyed on the checksum file and re-verifies only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
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
PACKAGE_URL = "https://xp20.ashrae.org/supplementalfiles/140-2023-AccompanyingFiles.zip"

# Member paths inside 140-2023-AccompanyingFiles.zip (note the upstream
# directory typo "FIles" — verbatim from the archive).
ZIP_MEMBERS = {
    "DRYCOLD.TMY725650TY.CSV": (
        "Accompanying FIles/Std140_TF_Files/Normative Materials/725650TY.csv"
    ),
    "Tsky-Informative.xlsx": (
        "Accompanying FIles/Std140_TF_Files/Informative Materials/Tsky-Informative.xlsx"
    ),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_installed() -> list[str]:
    """Verify DEST_DIR contents against the checksum file; return problems."""
    guard = json.loads(CHECKSUM_FILE.read_text())
    problems: list[str] = []
    for name, meta in guard["files"].items():
        path = DEST_DIR / name
        if not path.is_file():
            problems.append(f"{name}: missing from {DEST_DIR}")
            continue
        actual = sha256(path.read_bytes())
        if actual != meta["sha256"]:
            problems.append(
                f"{name}: sha256 {actual} != recorded {meta['sha256']}"
            )
    return problems


def fetch() -> None:
    guard = json.loads(CHECKSUM_FILE.read_text())
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {PACKAGE_URL} (this is a ~704 MB package)...")
    tmp = Path("/tmp/ashrae140-accompanying.zip")
    with urllib.request.urlopen(PACKAGE_URL, timeout=600) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    with zipfile.ZipFile(tmp) as zf:
        names = set(zf.namelist())
        for name, member in ZIP_MEMBERS.items():
            if member not in names:
                raise SystemExit(
                    f"FATAL: expected member {member!r} not present in the package; "
                    "ASHRAE may have repackaged it — re-verify and update "
                    f"{CHECKSUM_FILE} with a dated provenance note."
                )
            data = zf.read(member)
            recorded = guard["files"][name]["sha256"]
            actual = sha256(data)
            if actual != recorded:
                raise SystemExit(
                    f"FATAL: {name} sha256 {actual} != recorded {recorded}. "
                    "Do NOT update the checksum silently — verify the new file "
                    "against ASHRAE's site and record the change with dated "
                    "provenance."
                )
            (DEST_DIR / name).write_bytes(data)
            print(f"installed {DEST_DIR / name} ({len(data)} bytes, sha256 OK)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="verify already-installed files only (offline cache-restore path)",
    )
    args = ap.parse_args()

    if not CHECKSUM_FILE.is_file():
        print(f"FATAL: provenance guard {CHECKSUM_FILE} missing", file=sys.stderr)
        return 2

    problems = verify_installed()
    if not problems:
        print("canonical ASHRAE 140 weather files present and sha256-verified")
        return 0
    if args.skip_download:
        for p in problems:
            print(f"PROBLEM: {p}", file=sys.stderr)
        print(
            "FATAL: canonical weather cache incomplete after restore; run "
            "scripts/fetch_ashrae140_weather.py without --skip-download",
            file=sys.stderr,
        )
        return 1

    fetch()
    problems = verify_installed()
    if problems:
        for p in problems:
            print(f"PROBLEM: {p}", file=sys.stderr)
        return 1
    print("canonical ASHRAE 140 weather files installed and sha256-verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
