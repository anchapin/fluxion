#!/usr/bin/env python3
"""
Extract hourly per-tilt / per-orientation incident-solar data for the 5
sun-exposed ASHRAE 140 surfaces from EnergyPlus output.

Implements issue #1330 — closes the per-surface data-extraction gap that
prevents Module 2 (Solar) of fluxion from being validated against a
tilt-resolved EnergyPlus reference. The companion IDF
``tests/reference_data/energyplus_models/ashrae_140_solar_gain.idf``
already declares the 15 Output:Variable lines (5 surfaces × 3 components),
so this script simply runs E+, opens ``eplusout.sql``, and pivots the
hourly ReportData table into a single wide CSV.

CSV format
----------
``tests/reference_data/solar/ashrae_140_surface_incident_solar.csv``:

::

    # EnergyPlus Version: 25.2.0
    # Model: ASHRAE 140 box geometry (6×8×2.7m, lightweight walls)
    # EPW: USA_CO_Golden-NREL.724666_TMY3.epw
    # ...
    hour(1-8760),roof_beam,roof_sky,roof_ground,southwall_beam,southwall_sky,southwall_ground,northwall_beam,northwall_sky,northwall_ground,eastwall_beam,eastwall_sky,eastwall_ground,westwall_beam,westwall_sky,westwall_ground
    1,...
    ...
    8760,...

A companion summary CSV ``ashrae_140_surface_incident_solar_summary.csv``
holds annual mean / total per (surface, component) at 3 sig figs — the
1% tolerance check column referenced by ARCHITECTURE.md §Module 2.

Usage
-----
Requires EnergyPlus 25.2.0 on PATH (or at /usr/local/bin/energyplus).
From the repository root::

    python3 tests/reference_data/solar/extract_ashrae_140_per_surface.py

If E+ is unavailable, the script aborts with a clear error — there is no
deterministic eplusout.sql checked in, and the reference data depends on
a fresh E+ run against the gold-standard IDF.

Verification path (per issue #1330 acceptance criteria)
------------------------------------------------------
1. CSV exists, 8760 rows + header, 16 columns (hour + 15 data).
2. No missing / NaN cells (every E+ TimeIndex 1..8760 reported for all 15 series).
3. Annual integrated beam + sky + ground per surface to 3 sig figs in
   the summary CSV.
4. SouthWall beam column within ±5 % of the pre-existing
   ``surface_irradiance_south.csv`` reference (sanity check that this
   extractor matches the already-validated south path).
5. Script is runnable standalone and reproducible from the committed
   IDF / EPW (E+ 25.2.0 is the pinned runtime).
"""

from __future__ import annotations

import csv
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
IDF_PATH = REPO_ROOT / "tests/reference_data/energyplus_models/ashrae_140_solar_gain.idf"
EPW_PATH = Path("/usr/local/EnergyPlus-25-2-0/WeatherData/USA_CO_Golden-NREL.724666_TMY3.epw")
SOLAR_DIR = REPO_ROOT / "tests/reference_data/solar"
WORK_DIR = Path("/tmp/eplus_issue_1330")

# The 5 sun-exposed surfaces from the ASHRAE 140 lightweight box IDF.
SURFACES: tuple[str, ...] = ("Roof", "SouthWall", "NorthWall", "EastWall", "WestWall")
# The 3 incident-solar components emitted as hourly Output:Variables.
COMPONENTS: tuple[str, ...] = ("beam", "sky", "ground")

# Map column shortname -> exact E+ variable Name (E+ 25.2 stores these verbatim).
VARIABLE_NAMES: dict[str, str] = {
    "beam": "Surface Outside Face Incident Beam Solar Radiation Rate per Area",
    "sky": "Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area",
    "ground": "Surface Outside Face Incident Ground Diffuse Solar Radiation Rate per Area",
}

# Map surface -> E+ KeyValue (uppercased in 25.2 schema).
KEY_VALUES: dict[str, str] = {s: s.upper() for s in SURFACES}

OUT_CSV = SOLAR_DIR / "ashrae_140_surface_incident_solar.csv"
SUMMARY_CSV = SOLAR_DIR / "ashrae_140_surface_incident_solar_summary.csv"
REFERENCE_SOUTH_CSV = SOLAR_DIR / "surface_irradiance_south.csv"

EP_BIN_CANDIDATES: tuple[Path, ...] = (
    Path("/usr/local/bin/energyplus"),
    Path("/usr/local/EnergyPlus-25-2-0/energyplus"),
)


# ────────────────────────────────────────────────────────────────────────────
# EnergyPlus runner
# ────────────────────────────────────────────────────────────────────────────


def find_energyplus() -> Path | None:
    for p in EP_BIN_CANDIDATES:
        if p.exists():
            return p
    on_path = shutil.which("energyplus")
    return Path(on_path) if on_path else None


def run_energyplus(ep_bin: Path) -> Path:
    """Run EnergyPlus against the ASHRAE 140 solar-gain IDF.

    Returns the path to the generated eplusout.sql. Aborts on failure.
    """
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    cmd = [
        str(ep_bin),
        "-w", str(EPW_PATH),
        "-d", str(WORK_DIR),
        "-p", "eplus",
        str(IDF_PATH),
    ]
    print(f"Running EnergyPlus: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("EnergyPlus FAILED.")
        print("STDOUT (tail):\n" + result.stdout[-2000:])
        print("STDERR (tail):\n" + result.stderr[-1000:])
        sys.exit(1)

    sql_path = WORK_DIR / "eplusout.sql"
    if not sql_path.exists():
        print(f"EnergyPlus did not produce {sql_path}")
        sys.exit(1)
    return sql_path


# ────────────────────────────────────────────────────────────────────────────
# SQL extraction (EnergyPlus 25.2 schema: ReportDataDictionary / ReportData)
# ────────────────────────────────────────────────────────────────────────────


def extract_per_surface(sql_path: Path) -> dict[tuple[str, str], dict[int, float]]:
    """Return {(surface, component): {TimeIndex: W/m2}} for the 5×3 grid.

    Verifies that all 15 series have exactly 8760 values — partial
    extraction would corrupt the downstream tolerance tests.
    """
    conn = sqlite3.connect(sql_path)
    try:
        cur = conn.cursor()

        # 1. TimeIndex 1..N must be contiguous. E+ 25.2 emits exactly 8760.
        cur.execute("SELECT MIN(TimeIndex), MAX(TimeIndex), COUNT(*) FROM Time")
        lo, hi, n = cur.fetchone()
        if (lo, hi, n) != (1, 8760, 8760):
            raise RuntimeError(
                f"Unexpected Time table: min={lo}, max={hi}, count={n} (need 1, 8760, 8760)"
            )

        # 2. Build {(surface, component): dict_index} from ReportDataDictionary.
        want_keys = set(KEY_VALUES.values())
        want_names = set(VARIABLE_NAMES.values())
        idx_for: dict[tuple[str, str], int] = {}
        cur.execute(
            'SELECT "ReportDataDictionaryIndex", "KeyValue", "Name" '
            'FROM "ReportDataDictionary"'
        )
        for dict_idx, key_value, name in cur.fetchall():
            if key_value not in want_keys or name not in want_names:
                continue
            # Reverse lookup.
            surface = next(s for s, k in KEY_VALUES.items() if k == key_value)
            component = next(c for c, n in VARIABLE_NAMES.items() if n == name)
            idx_for[(surface, component)] = dict_idx

        missing = {
            (s, c)
            for s in SURFACES
            for c in COMPONENTS
            if (s, c) not in idx_for
        }
        if missing:
            raise RuntimeError(
                f"Missing ReportDataDictionary entries for {sorted(missing)}; "
                "check Output:Variable lines in the IDF."
            )

        # 3. Pull all ReportData rows for the 15 series.
        series: dict[tuple[str, str], dict[int, float]] = {
            (s, c): {} for s in SURFACES for c in COMPONENTS
        }
        # One IN clause covers all 15 dict indices.
        placeholders = ",".join("?" * len(idx_for))
        flat_idx = list(idx_for.values())
        cur.execute(
            f'SELECT "ReportDataDictionaryIndex", "TimeIndex", "Value" '
            f'FROM "ReportData" WHERE "ReportDataDictionaryIndex" IN ({placeholders})',
            flat_idx,
        )
        for dict_idx, time_index, value in cur.fetchall():
            (surf_comp,) = [(s, c) for (s, c), di in idx_for.items() if di == dict_idx]
            series[surf_comp][time_index] = value

        # 4. Verify completeness — every cell filled.
        for (s, c), data in series.items():
            if len(data) != 8760:
                missing_hours = sorted(set(range(1, 8761)) - set(data.keys()))[:5]
                raise RuntimeError(
                    f"{s}/{c}: {len(data)} hourly values (expected 8760); "
                    f"missing first 5 = {missing_hours}"
                )
            if any(v is None for v in data.values()):
                raise RuntimeError(f"{s}/{c}: NaN/None encountered in ReportData")

        return series
    finally:
        conn.close()


# ────────────────────────────────────────────────────────────────────────────
# CSV writers
# ────────────────────────────────────────────────────────────────────────────


def _surface_slug(s: str) -> str:
    return s.lower()


def write_wide_csv(series: dict[tuple[str, str], dict[int, float]], out_path: Path) -> None:
    """Write the wide-format 8760-row × 16-column CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Column order: hour, then (Roof, S, N, E, W) × (beam, sky, ground).
    column_order = ["hour"] + [
        f"{_surface_slug(s)}_{c}" for s in SURFACES for c in COMPONENTS
    ]

    with open(out_path, "w", newline="") as f:
        f.write(
            "# EnergyPlus Version: 25.2.0\n"
            f"# Model: ASHRAE 140 box geometry (6x8x2.7m, lightweight walls, "
            "south wall = 16.2 m^2, N = 16.2 m^2, E = 21.6 m^2, W = 21.6 m^2, "
            "Roof = 48 m^2)\n"
            f"# EPW: {EPW_PATH.name}\n"
            f"# IDF: {IDF_PATH.relative_to(REPO_ROOT)}\n"
            f"# Output:Variable keys: "
            + ", ".join(f"{s}/{c}" for s in SURFACES for c in COMPONENTS)
            + "\n"
            f"# Generated: {datetime.now(timezone.utc).isoformat()}\n"
            f"# Rows: 8760\n"
            f"# Columns: {','.join(column_order)}\n"
            f"# Units: all irradiance columns in W/m^2 (hourly mean)\n"
        )
        w = csv.writer(f)
        w.writerow(column_order)
        for h in range(1, 8761):
            row = [h]
            for s in SURFACES:
                for c in COMPONENTS:
                    row.append(f"{series[(s, c)][h]:.4f}")
            w.writerow(row)
    print(f"Wrote {out_path} (8760 rows × {len(column_order)} columns)")


def write_summary_csv(series: dict[tuple[str, str], dict[int, float]], out_path: Path) -> None:
    """Write per-surface annual mean + integrated total to 3 sig figs.

    Issue #1330 acceptance criterion 3: 'Annual integrated beam+diffuse+ground
    per surface reported to 3 sig figs in the verification script.'
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in SURFACES:
        b_mean = sum(series[(s, "beam")].values()) / 8760.0
        s_mean = sum(series[(s, "sky")].values()) / 8760.0
        g_mean = sum(series[(s, "ground")].values()) / 8760.0
        b_sum = sum(series[(s, "beam")].values())
        s_sum = sum(series[(s, "sky")].values())
        g_sum = sum(series[(s, "ground")].values())
        t_mean = b_mean + s_mean + g_mean
        t_sum = b_sum + s_sum + g_sum
        rows.append(
            {
                "surface": s,
                "annual_mean_beam": b_mean,
                "annual_mean_sky": s_mean,
                "annual_mean_ground": g_mean,
                "annual_mean_total": t_mean,
                "annual_total_beam": b_sum,
                "annual_total_sky": s_sum,
                "annual_total_ground": g_sum,
                "annual_total_all": t_sum,
                "peak_hour_total": max(
                    series[(s, "beam")][h]
                    + series[(s, "sky")][h]
                    + series[(s, "ground")][h]
                    for h in range(1, 8761)
                ),
            }
        )

    with open(out_path, "w", newline="") as f:
        f.write(
            "# EnergyPlus Version: 25.2.0\n"
            "# Annual integrated per-surface incident solar for the ASHRAE 140 box.\n"
            "# Units: means in W/m^2 (hourly average); totals in Wh/m^2 (sum of hourly W/m^2).\n"
            "# peak_hour_total in W/m^2 — max over hours 1..8760.\n"
            f"# Generated: {datetime.now(timezone.utc).isoformat()}\n"
        )
        w = csv.writer(f)
        w.writerow(
            [
                "surface",
                "annual_mean_beam(W/m2)",
                "annual_mean_sky(W/m2)",
                "annual_mean_ground(W/m2)",
                "annual_mean_total(W/m2)",
                "annual_total_beam(Wh/m2)",
                "annual_total_sky(Wh/m2)",
                "annual_total_ground(Wh/m2)",
                "annual_total_all(Wh/m2)",
                "peak_hour_total(W/m2)",
            ]
        )
        def sig3(x: float) -> str:
            # 3 significant figures (issue #1330 AC 3).
            return f"{x:.3g}"

        for r in rows:
            w.writerow(
                [
                    r["surface"],
                    sig3(r["annual_mean_beam"]),
                    sig3(r["annual_mean_sky"]),
                    sig3(r["annual_mean_ground"]),
                    sig3(r["annual_mean_total"]),
                    sig3(r["annual_total_beam"]),
                    sig3(r["annual_total_sky"]),
                    sig3(r["annual_total_ground"]),
                    sig3(r["annual_total_all"]),
                    sig3(r["peak_hour_total"]),
                ]
            )
    print(f"Wrote {out_path}")
    return rows


# ────────────────────────────────────────────────────────────────────────────
# Acceptance-criteria checks (issue #1330)
# ────────────────────────────────────────────────────────────────────────────


def load_existing_south_beam() -> dict[int, float]:
    """Read the previously-validated surface_irradiance_south.csv beam column."""
    out: dict[int, float] = {}
    with open(REFERENCE_SOUTH_CSV) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            try:
                h = int(parts[0])
                beam = float(parts[1])
            except ValueError:
                continue
            out[h] = beam
    return out


def cross_check_south(south_series: dict[int, float]) -> tuple[float, float, float]:
    """Return (mean_ratio, min_ratio, max_ratio, pct_within_5pct).

    Issue #1330 AC 4: SouthWall beam column within +/-5 % of the
    pre-existing reference.
    """
    ref = load_existing_south_beam()
    ratios: list[float] = []
    for h, beam_new in south_series.items():
        if h not in ref:
            continue
        ref_v = ref[h]
        # Skip hours where the reference itself is ~0 (sun below horizon).
        # Comparing to a near-zero reference would amplify any tiny numerical
        # difference into a huge ratio.
        if abs(ref_v) < 1.0:
            continue
        ratios.append(beam_new / ref_v)
    mean_r = sum(ratios) / len(ratios)
    min_r = min(ratios)
    max_r = max(ratios)
    pct = 100.0 * sum(1 for r in ratios if 0.95 <= r <= 1.05) / len(ratios)
    return mean_r, min_r, max_r, pct


def assert_invariants(series: dict[tuple[str, str], dict[int, float]]) -> None:
    """Run the issue-1330 acceptance-criteria sanity checks."""
    # AC 1 + 2: 8760 rows × 16 columns, no missing/NaN.
    for s in SURFACES:
        for c in COMPONENTS:
            vals = series[(s, c)]
            assert len(vals) == 8760, f"{s}/{c}: {len(vals)} values (need 8760)"
            assert all(v is not None for v in vals.values()), f"{s}/{c}: NaN found"

    # AC 4: SouthWall beam cross-check.
    mean_r, min_r, max_r, pct = cross_check_south(series[("SouthWall", "beam")])
    assert 0.95 <= mean_r <= 1.05, (
        f"SouthWall beam mean ratio {mean_r:.4f} outside +/-5 %"
    )
    assert min_r >= 0.90, f"SouthWall beam min ratio {min_r:.4f} below 0.90"
    assert max_r <= 1.10, f"SouthWall beam max ratio {max_r:.4f} above 1.10"
    assert pct >= 95.0, (
        f"Only {pct:.1f}% of SouthWall beam hours within +/-5 % (need >= 95 %)"
    )

    # Physics sanity 1: peak per-hour total <= ~1100 W/m^2 (clear-sky envelope).
    peak_total = max(
        series[(s, c)][h]
        for s in SURFACES
        for c in COMPONENTS
        for h in range(1, 8761)
    )
    assert peak_total <= 1100.0, (
        f"Peak total irradiance {peak_total:.2f} W/m^2 exceeds 1100 W/m^2 envelope"
    )

    # Physics sanity 2: roof annual total > south-wall annual total
    # (horizontal roof receives beam at all hours the sun is up; south wall
    # only receives strong beam when sun is in the southern sky).
    roof_total = sum(
        series[("Roof", c)][h] for c in COMPONENTS for h in range(1, 8761)
    )
    south_total = sum(
        series[("SouthWall", c)][h] for c in COMPONENTS for h in range(1, 8761)
    )
    assert roof_total > south_total, (
        f"Roof annual total {roof_total:.0f} Wh/m^2 should exceed SouthWall "
        f"total {south_total:.0f} Wh/m^2"
    )

    # Physics sanity 3: at solar noon (sun due south), East and West beam
    # should both be near zero (sun behind both vertical walls). Use the
    # companion solar-position CSV to find the right TimeIndex.
    sol_pos_csv = SOLAR_DIR / "solar_position_denver.csv"
    if sol_pos_csv.exists():
        _check_solar_noon_east_west_near_zero(series, sol_pos_csv)


def _check_solar_noon_east_west_near_zero(
    series: dict[tuple[str, str], dict[int, float]],
    sol_pos_csv: Path,
) -> None:
    """At near-noon hours (|azimuth - 180| <= 5°), East + West beam must be << Roof.

    Geometric invariant: when the sun's azimuth is close to due south
    (true solar noon), the horizontal roof receives cos(zenith) ≈ 1 of
    DNI while east and west vertical walls receive only the small
    residual cos of the sun's offset from due south. So at true solar
    noon, Roof beam dominates.

    (At hours far from solar noon — early morning or late afternoon —
    the sun is in the east or west and the vertical wall facing the
    sun can receive more beam than the roof. That's geometrically
    correct and excluded here by the azimuth filter.)
    """
    azimuth_by_hour: dict[int, float] = {}
    with open(sol_pos_csv) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            try:
                azimuth_by_hour[int(parts[0])] = float(parts[2])
            except ValueError:
                continue

    # Find hours where sun is high AND azimuth is within ±5° of due south.
    # 5° envelope accommodates TMY3's LST clock + equation-of-time effects.
    near_noon_hours = [
        h for h in range(1, 8761)
        if abs(azimuth_by_hour.get(h, 999.0) - 180.0) <= 5.0
        and series[("Roof", "beam")].get(h, 0.0) > 100.0
    ]
    violations = []
    for h in near_noon_hours:
        e = series[("EastWall", "beam")].get(h, 0.0)
        w = series[("WestWall", "beam")].get(h, 0.0)
        r = series[("Roof", "beam")].get(h, 0.0)
        # At azimuth=175° (5° east of south), the east wall receives
        # cos(5°) ≈ 1.0 of sin(altitude) of DNI, while the roof receives
        # cos(zenith) = sin(altitude). So east/roof ≈ 1.0 only when sun
        # is due east; at ±5° of south, east/roof ≈ sin(5°)/cos(zenith)
        # which is small for high-sun hours.
        if e + w > 0.30 * r:  # 30% envelope accommodates 5° offset + TMY3 noise
            violations.append(
                (h, e, w, r, azimuth_by_hour[h])
            )
    assert not violations, (
        f"{len(violations)} near-noon hours (|az-180|<=5°, roof beam>100 W/m^2) "
        f"where East+West beam > 30% of Roof beam; first 5 = {violations[:5]}"
    )


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────


def main() -> int:
    print("=" * 72)
    print("Issue #1330 — Per-tilt/per-orientation incident-solar CSV extractor")
    print("=" * 72)

    ep_bin = find_energyplus()
    if ep_bin is None:
        print(
            "EnergyPlus not found (looked at /usr/local/bin/energyplus, "
            "/usr/local/EnergyPlus-25-2-0/energyplus, and $PATH). "
            "Install E+ 25.2.0 to regenerate this reference data."
        )
        return 1
    print(f"EnergyPlus binary: {ep_bin}")
    if not EPW_PATH.exists():
        print(f"EPW missing: {EPW_PATH}")
        return 1
    if not IDF_PATH.exists():
        print(f"IDF missing: {IDF_PATH}")
        return 1

    sql_path = run_energyplus(ep_bin)
    print(f"EnergyPlus produced {sql_path}")

    series = extract_per_surface(sql_path)
    assert_invariants(series)
    print("Acceptance-criteria checks passed.")

    write_wide_csv(series, OUT_CSV)
    summary_rows = write_summary_csv(series, SUMMARY_CSV)

    print("\nAnnual integrated totals (Wh/m^2):")
    print(f"  {'surface':<12} {'beam':>10} {'sky':>10} {'ground':>10} {'total':>10}")
    for r in summary_rows:
        print(
            f"  {r['surface']:<12} "
            f"{r['annual_total_beam']:>10.0f} "
            f"{r['annual_total_sky']:>10.0f} "
            f"{r['annual_total_ground']:>10.0f} "
            f"{r['annual_total_all']:>10.0f}"
        )

    peak_total = max(
        series[(s, "beam")][h]
        + series[(s, "sky")][h]
        + series[(s, "ground")][h]
        for s in SURFACES
        for h in range(1, 8761)
    )
    roof_total = summary_rows[0]["annual_total_all"]
    south_total = summary_rows[1]["annual_total_all"]
    print(f"\nPhysics sanity:")
    print(f"  Max per-hour total (beam+sky+ground): {peak_total:>7.2f} W/m^2 (<= 1100 envelope)")
    print(f"  Roof > SouthWall annual total:       {roof_total > south_total} "
          f"({roof_total:.0f} > {south_total:.0f} Wh/m^2)")

    mean_r, min_r, max_r, pct = cross_check_south(series[("SouthWall", "beam")])
    print(
        f"\nSouthWall beam cross-check vs surface_irradiance_south.csv: "
        f"mean_ratio={mean_r:.4f}, min={min_r:.4f}, max={max_r:.4f}, "
        f"hours_within_5pct={pct:.1f}%"
    )

    print(
        "\nAcceptance criteria status:"
        "\n  [x] AC1: 8760 rows + header, 1 hour col + 15 (surface x component) numeric cols"
        "\n  [x] AC2: no missing-key NaNs (every TimeIndex 1..8760 reported)"
        "\n  [x] AC3: annual integrated per (surface, component) to 3 sig figs in summary CSV"
        "\n  [x] AC4: SouthWall beam within +/-5% of surface_irradiance_south.csv reference"
        "\n  [x] AC5: standalone runnable script; reproduces output via E+ 25.2.0"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())