#!/usr/bin/env python3
"""Convert the ASHRAE 140 canonical DRYCOLD TMY3 CSV + informative Tsky
series into the EPW file the strict gate reads (loop round 41).

Inputs (installed by scripts/fetch_ashrae140_weather.py):
    assets/weather/ashrae140/DRYCOLD.TMY725650TY.CSV   (TMY3, station 725650)
    assets/weather/ashrae140/Tsky-Informative.xlsx     (informative Tsky)

Output:
    assets/weather/ashrae140/DRYCOLD_TMY725650TY.epw

The engine reads EPW fields 6 (dry bulb), 8 (RH), 12 (horizontal IR), 13
(GHI), 14 (DNI), 15 (DHI) and 21 (wind speed). The TMY3 CSV carries no
horizontal-IR column, so the converter emits HIR = sigma * Tsky^4 from the
informative modified Martin–Berdahl series — physically identical to the
engine reading Tsky directly (the single sky path is preserved; see
docs/investigations/sky-longwave-input-provenance-and-canonical-weather.md,
round-40 spec step 2).

TMY3 CSV column layout (0-based): GHI 4, DNI 7, DHI 10, dry-bulb 31,
dew-point 34, RHum 37, pressure 40 (mbar), wind dir 43, wind speed 46.
"""

from __future__ import annotations

import csv
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "assets" / "weather" / "ashrae140"
CSV_PATH = SRC_DIR / "DRYCOLD.TMY725650TY.CSV"
XLSX_PATH = SRC_DIR / "Tsky-Informative.xlsx"
OUT_PATH = SRC_DIR / "DRYCOLD_TMY725650TY.epw"

SIGMA = 5.6697e-8  # Stefan–Boltzmann, W/(m^2 K^4)

# 0-based TMY3 column indices (measure column of each source/uncert triplet).
COL_GHI = 4
COL_DNI = 7
COL_DHI = 10
COL_DRY_BULB = 31
COL_DEW_POINT = 34
COL_RHUM = 37
COL_PRESSURE = 40
COL_WDIR = 43
COL_WSPD = 46

SOLAR_SENTINEL = 9999  # TMY3 missing-solar flag; EPW parser coerces to 0

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_tmy3(path: Path) -> list[list[str]]:
    with open(path, newline="") as f:
        rd = csv.reader(f)
        next(rd)  # station header line
        next(rd)  # column header line
        return [row for row in rd if row]


def read_tsky(path: Path) -> list[float]:
    """Hourly Tsky (°C) from the 'Informative-Tsky' sheet, column D."""
    with zipfile.ZipFile(path) as zf:
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    values: list[float] = []
    for row in sheet.findall(".//m:row", NS):
        cells = row.findall("m:c", NS)
        if len(cells) < 4:
            continue
        # Data rows carry numeric date (A), time-fraction (B) and Tsky (D);
        # trailing summary rows lack the date/time pair.
        try:
            float(cells[0].find("m:v", NS).text)
            float(cells[1].find("m:v", NS).text)
            v = cells[3].find("m:v", NS)
            v.text = v.text  # keep type-checkers honest
        except (AttributeError, ValueError):
            continue
        if cells[3].attrib.get("t") == "s" or v is None:
            continue
        try:
            values.append(float(v.text))
        except ValueError:
            continue
    return values


def f3(x: float) -> str:
    return f"{x:.1f}"


def main() -> int:
    if not CSV_PATH.is_file() or not XLSX_PATH.is_file():
        print(
            "FATAL: canonical weather files missing; run "
            "scripts/fetch_ashrae140_weather.py first",
            file=sys.stderr,
        )
        return 2

    rows = read_tmy3(CSV_PATH)
    tsky = read_tsky(XLSX_PATH)
    if len(rows) != 8760 or len(tsky) != 8760:
        print(
            f"FATAL: expected 8760 hourly rows, got TMY3={len(rows)} "
            f"Tsky={len(tsky)}",
            file=sys.stderr,
        )
        return 1

    lines = [
        # 1 LOCATION
        "LOCATION,725650,DENVER INTL AP,CO,USA,TMY3,39.833,-104.650,-7.0,1650.0",
        # 2 DESIGN CONDITIONS
        "DESIGN CONDITIONS,0",
        # 3 TYPICAL/EXTREME PERIODS
        "TYPICAL/EXTREME PERIODS,0",
        # 4 GROUND TEMPERATURES (kept as a header so is_epw_header_line filters it)
        "GROUND TEMPERATURES,0",
        # 5 HOLIDAYS/DAYLIGHT SAVINGS
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
        # 6-7 COMMENTS
        "COMMENTS 1,ASHRAE 140-2023 canonical DRYCOLD weather (TMY3 station 725650),"
        " converted from DRYCOLD.TMY725650TY.CSV by scripts/convert_tmy3_to_epw.py",
        "COMMENTS 2,Field 12 horizontal IR = sigma*Tsky^4 from the informative"
        " modified Martin-Berdahl series (Tsky-Informative.xlsx); round-41 binding",
        # 8 DATA PERIODS
        "DATA PERIODS,1,1,Data,Sun-3,1/1,12/31,hourly,1,00:00:00,01:00:00",
    ]

    for i, row in enumerate(rows):
        month, day = row[0].split("/")[:2]
        hour = i % 24 + 1  # EPW hour convention 1..24; TMY3 rows are hour-sequential
        tdb = float(row[COL_DRY_BULB])
        dew = float(row[COL_DEW_POINT])
        rh = float(row[COL_RHUM])
        pres_pa = float(row[COL_PRESSURE]) * 100.0  # mbar -> Pa
        ghi = float(row[COL_GHI])
        dni = float(row[COL_DNI])
        dhi = float(row[COL_DHI])
        wdir = float(row[COL_WDIR])
        wspd = float(row[COL_WSPD])
        hir = SIGMA * (tsky[i] + 273.15) ** 4

        fields = [
            "1995",  # year placeholder; parser treats it as informational
            str(int(month)),
            str(int(day)),
            str(hour),
            "0",  # minute
            "0",  # data source/uncertainty flags
            f3(tdb),      # 6  dry bulb
            f3(dew),      # 7  dew point
            f3(rh),       # 8  relative humidity
            f3(pres_pa),  # 9  station pressure
            "0",          # 10 extraterrestrial horizontal radiation
            "0",          # 11 extraterrestrial direct normal radiation
            f3(hir),      # 12 horizontal infrared (from informative Tsky)
            f3(ghi),      # 13 global horizontal radiation
            f3(dni),      # 14 direct normal radiation
            f3(dhi),      # 15 diffuse horizontal radiation
            "0", "0",     # 16-17 horizontal/placeholder illuminance
            "0", "0",     # 18-19 placeholder illuminance fields
            f3(wdir),     # 20 wind direction
            f3(wspd),     # 21 wind speed
            "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0",
        ]
        assert len(fields) == 35, len(fields)
        lines.append(",".join(fields))

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_PATH} ({len(lines) - 8} hourly rows)")

    # Self-check: annual Tsky-Tdb depression, compared with the round-40
    # measured provenance (-16.78 K annual mean). A mismatch means the Tsky
    # series did not parse in row order — stop rather than emit a shifted EPW.
    deps = [
        tsky[i] - float(rows[i][COL_DRY_BULB]) for i in range(8760)
    ]
    annual = sum(deps) / len(deps)
    print(f"Tsky-Tdb annual mean: {annual:.2f} K (round-40 provenance: -16.78 K)")
    if abs(annual - (-16.78)) > 0.05:
        print(
            "FATAL: Tsky/Tdb alignment mismatch vs round-40 provenance measurement",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
