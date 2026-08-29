#!/usr/bin/env python3
"""
Fetch ASHRAE 140 EPW weather files required for validation.

Usage:
    python3 scripts/fetch_ashrae140_epw.py [--dry-run] [--check-only]

This script downloads the EPW files required to run the full ASHRAE 140
validation test suite.  The files are fetched from the EnergyPlus
weather data portal (https://energyplus.net/weather).

After downloading, set the environment variable:
    export FLUXION_EPW_DIR=/path/to/weather/data

Or place the files in the default location `assets/weather/` relative
to the repository root.

Required files for ASHRAE 140 validation:
  - USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw  (Case 600/900 reference)
  - WD600.epw                                   (Test case weather)

Optional files for multi-climate validation:
  - USA_FL_Miami.Intl.AP.722020_TMY3.epw
  - USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw
  - USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw
  - USA_CO_Golden-NREL.724666_TMY3.epw
  - USA_TX_Houston.Intercontinental.AP.722430_TMY3.epw
  - (see src/ai/sweeps/weather.rs for full list)

DDY design-day files are also fetched alongside their EPW counterparts.

Closes #N (tracking issue for this refactor).
"""

import argparse
import os
import sys
import urllib.request
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_EPW_DIR = REPO_ROOT / "assets" / "weather"
ASHRAE140_EPW_DIR = REPO_ROOT / "assets" / "weather"

# EnergyPlus weather data portal base URL
# Files follow the pattern: region/country/USA/USA_CO/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw
EPW_BASE_URL = "https://energyplus.net/sites/default/files/weatherdownload/"

# Required files for ASHRAE 140 (Case 600/900)
REQUIRED_FILES = [
    "USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw",
    "USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.ddy",
    "WD600.epw",
]

# Optional files for extended validation
OPTIONAL_FILES = [
    "USA_FL_Miami.Intl.AP.722020_TMY3.epw",
    "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
    "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
    "USA_CO_Golden-NREL.724666_TMY3.epw",
    "USA_TX_Houston.Intercontinental.AP.722430_TMY3.epw",
    "USA_GA_Atlanta-Hartsfield.Jackson.Intl.AP.722190_TMY3.epw",
    "USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3.epw",
    "USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3.epw",
    "USA_NM_Albuquerque.Intl.AP.723650_TMY3.epw",
    "USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3.epw",
    "USA_MA_Boston-Logan.Intl.AP.725045_TMY3.epw",
    "USA_CO_Denver.Intl.AP.725650_TMY3.epw",
    "USA_OR_Portland.Intl.AP.726980_TMY3.epw",
    "USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3.epw",
    "USA_MT_Helena.Rgnl.AP.727725_TMY3.epw",
    "USA_MN_Duluth.Intl.AP.727450_TMY3.epw",
    "USA_AK_Fairbanks.Intl.AP.702610_TMY3.epw",
    "WD100.epw",
    "WD200.epw",
    "WD300.epw",
    "WD400.epw",
    "WD500.epw",
    "WD600.epw",
]

ALL_FILES = REQUIRED_FILES + OPTIONAL_FILES


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def file_exists_and_valid(path: Path, min_size: int = 1000) -> bool:
    """Check if an EPW file exists and is plausibly valid (minimum size)."""
    if not path.exists():
        return False
    if path.stat().st_size < min_size:
        return False
    return True


def download_file(url: str, dest: Path, dry_run: bool = False) -> bool:
    """Download a single file. Returns True on success."""
    if dry_run:
        print(f"  [dry-run] would download {url}\n         -> {dest}")
        return True

    try:
        ensure_dir(dest.parent)
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
            dest.write_bytes(content)
        size_kb = len(content) / 1024
        print(f"  OK  {dest.name} ({size_kb:.0f} KB)")
        return True
    except urllib.error.URLError as e:
        print(f"  ERR {dest.name}: {e}")
        return False
    except OSError as e:
        print(f"  ERR {dest.name}: {e}")
        return False


def fetch_epw(epw_dir: Path, dry_run: bool = False, check_only: bool = False) -> int:
    """Fetch all required EPW files. Returns 0 on success, non-zero on failure."""
    print(f"\nASHRAE 140 EPW Weather File Fetcher")
    print(f"=" * 50)
    print(f"Target directory: {epw_dir}")
    print(f"Mode: {'check-only' if check_only else ('dry-run' if dry_run else 'download')}")
    print()

    if check_only:
        missing = []
        for fname in REQUIRED_FILES:
            path = epw_dir / fname
            if not file_exists_and_valid(path):
                missing.append(str(path.relative_to(REPO_ROOT)))
        if missing:
            print(f"MISSING {len(missing)} required file(s):")
            for m in missing:
                print(f"  - {m}")
            print()
            print("Run: python3 scripts/fetch_ashrae140_epw.py")
            return 1
        else:
            print("All required EPW files present.")
            return 0

    ensure_dir(epw_dir)

    # EnergyPlus organizes files by region. We map each file to its URL.
    # URL pattern: https://energyplus.net/sites/default/files/weatherdownload/[region]-[country]-[state]-[loc].epw
    # The easiest reliable source is the EnergyPlus website directly.
    url_map = _build_url_map()

    success_count = 0
    fail_count = 0

    for fname in ALL_FILES:
        dest = epw_dir / fname
        if file_exists_and_valid(dest):
            print(f"  SKIP {fname} (already present)")
            success_count += 1
            continue

        url = url_map.get(fname)
        if url is None:
            print(f"  SKIP {fname} (no known URL — place manually if needed)")
            continue

        ok = download_file(url, dest, dry_run=dry_run)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print()
    print(f"Result: {success_count} present, {fail_count} failed")
    if fail_count > 0:
        print(f"\nFailed files can be manually downloaded from:\n  https://energyplus.net/weather")
        print(f"\nPlace manually in: {epw_dir}")
    return 1 if fail_count > 0 else 0


def _build_url_map() -> dict:
    """
    Build URL map for EnergyPlus weather file downloads.

    EnergyPlus hosts TMY3 files at energyplus.net. The URL structure is:
    https://energyplus.net/sites/default/files/weatherdownload/[normalized-name].epw

    Where [normalized-name] is derived from the location code.
    """
    # EnergyPlus weather download URLs — derived from the official distribution.
    # Format: https://energyplus.net/sites/default/files/weatherdownload/[file]
    base = EPW_BASE_URL

    # fmt: off
    url_map = {
        # ASHRAE 140 reference weather (Case 600/900)
        "USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw":
            f"{base}USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw",
        "USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.ddy":
            f"{base}USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.ddy",

        # WD files (EnergyPlus test weather files)
        "WD600.epw": f"{base}WD600.epw",
        "WD100.epw": f"{base}WD100.epw",
        "WD200.epw": f"{base}WD200.epw",
        "WD300.epw": f"{base}WD300.epw",
        "WD400.epw": f"{base}WD400.epw",
        "WD500.epw": f"{base}WD500.epw",

        # TMY3 files by city
        "USA_FL_Miami.Intl.AP.722020_TMY3.epw":
            f"{base}USA_FL_Miami.Intl.AP.722020_TMY3.epw",
        "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw":
            f"{base}USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw",
        "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw":
            f"{base}USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw",
        "USA_CO_Golden-NREL.724666_TMY3.epw":
            f"{base}USA_CO_Golden-NREL.724666_TMY3.epw",
        "USA_TX_Houston.Intercontinental.AP.722430_TMY3.epw":
            f"{base}USA_TX_Houston.Intercontinental.AP.722430_TMY3.epw",
        "USA_GA_Atlanta-Hartsfield.Jackson.Intl.AP.722190_TMY3.epw":
            f"{base}USA_GA_Atlanta-Hartsfield.Jackson.Intl.AP.722190_TMY3.epw",
        "USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3.epw":
            f"{base}USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3.epw",
        "USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3.epw":
            f"{base}USA_MD_Baltimore-Washington.Intl.AP.724060_TMY3.epw",
        "USA_NM_Albuquerque.Intl.AP.723650_TMY3.epw":
            f"{base}USA_NM_Albuquerque.Intl.AP.723650_TMY3.epw",
        "USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3.epw":
            f"{base}USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3.epw",
        "USA_MA_Boston-Logan.Intl.AP.725045_TMY3.epw":
            f"{base}USA_MA_Boston-Logan.Intl.AP.725045_TMY3.epw",
        "USA_CO_Denver.Intl.AP.725650_TMY3.epw":
            f"{base}USA_CO_Denver.Intl.AP.725650_TMY3.epw",
        "USA_OR_Portland.Intl.AP.726980_TMY3.epw":
            f"{base}USA_OR_Portland.Intl.AP.726980_TMY3.epw",
        "USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3.epw":
            f"{base}USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3.epw",
        "USA_MT_Helena.Rgnl.AP.727725_TMY3.epw":
            f"{base}USA_MT_Helena.Rgnl.AP.727725_TMY3.epw",
        "USA_MN_Duluth.Intl.AP.727450_TMY3.epw":
            f"{base}USA_MN_Duluth.Intl.AP.727450_TMY3.epw",
        "USA_AK_Fairbanks.Intl.AP.702610_TMY3.epw":
            f"{base}USA_AK_Fairbanks.Intl.AP.702610_TMY3.epw",
    }
    # fmt: on
    return url_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ASHRAE 140 EPW weather files.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check if required files are present, exit 0 if yes, 1 if missing.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help=f"Target directory (default: {DEFAULT_EPW_DIR})",
    )
    args = parser.parse_args()

    epw_dir = args.dir or DEFAULT_EPW_DIR
    epw_dir = epw_dir.resolve()

    sys.exit(fetch_epw(epw_dir, dry_run=args.dry_run, check_only=args.check_only))
