#!/usr/bin/env python3
"""
compare_airflow.py

Parse two EnergyPlus SQL output files (before and after the VAV measure) and
produce a comparison table showing per-zone airflow rates, confirming that VAV
boxes modulate correctly (supply airflow varies between runs while respecting
the minimum airflow fraction).

Usage:
    python compare_airflow.py before.sql after.sql [--output report.csv]
"""

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

TABULAR_QUERY = """
SELECT
    TabularData.ReportName,
    TabularData.ReportForString,
    TabularData.TableName,
    TabularData.RowName,
    TabularData.ColumnName,
    TabularData.Units,
    TabularData.Value
FROM TabularData
JOIN TabularDataReports
    ON TabularData.IndexGroup = TabularDataReports.IndexGroup
"""


def extract_zone_airflow(sql_path: Path) -> dict[str, dict[str, float]]:
    conn = sqlite3.connect(str(sql_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    zones: dict[str, dict[str, float]] = {}

    cur.execute(TABULAR_QUERY)
    for row in cur.fetchall():
        table = row["TableName"]
        col = row["ColumnName"]
        val_str = row["Value"]

        if table not in (
            "HVAC Sizing Summary",
            "Equipment Summary",
            "Zone Sizing",
        ):
            continue

        if "Air Flow" not in col and "Airflow" not in col and "Air Rate" not in col:
            continue

        try:
            val = float(val_str)
        except (ValueError, TypeError):
            continue

        zone_name = row["RowName"]
        zones.setdefault(zone_name, {})[col] = val

    hourly_query = """
    SELECT
        ReportVariableDatatimeVariables.VariableName,
        ReportVariableDatatimeVariables.KeyValue,
        ReportVariableData.TimeIndex,
        ReportVariableData.Value
    FROM ReportVariableData
    JOIN ReportVariableDatatimeVariables
        ON ReportVariableData.ReportVariableDatatimeVariablesIndex
         = ReportVariableDatatimeVariables.ReportVariableDatatimeVariablesIndex
    WHERE ReportVariableDatatimeVariables.VariableName
        LIKE '%Zone Air System Air Flow Rate%'
       OR ReportVariableDatatimeVariables.VariableName
        LIKE '%Air Terminal%Air Flow%'
    """

    try:
        cur.execute(hourly_query)
        hourly: dict[str, list[float]] = {}
        for row in cur.fetchall():
            key = f"{row['KeyValue']}|{row['VariableName']}"
            hourly.setdefault(key, []).append(row["Value"])

        for key, values in hourly.items():
            zone_name, var_name = key.split("|", 1)
            zones.setdefault(zone_name, {})
            if values:
                zones[zone_name][f"{var_name} [Max]"] = max(values)
                zones[zone_name][f"{var_name} [Min]"] = min(values)
                zones[zone_name][f"{var_name} [Mean]"] = sum(values) / len(values)
    except sqlite3.OperationalError:
        pass

    conn.close()
    return zones


def compare(before: dict, after: dict, min_fraction: float = 0.3) -> list[dict]:
    all_zones = sorted(set(before.keys()) | set(after.keys()))
    rows = []

    for zone in all_zones:
        b = before.get(zone, {})
        a = after.get(zone, {})

        b_max = b.get(
            "Air Terminal Maximum Air Flow Rate [Max]",
            b.get("Maximum Air Flow Rate", None),
        )
        a_max = a.get(
            "Air Terminal Maximum Air Flow Rate [Max]",
            a.get("Maximum Air Flow Rate", None),
        )

        b_mean = None
        a_mean = None
        for k in b:
            if "[Mean]" in k and "Air Flow" in k:
                b_mean = b[k]
                break
        for k in a:
            if "[Mean]" in k and "Air Flow" in k:
                a_mean = a[k]
                break

        a_min = None
        for k in a:
            if "[Min]" in k and "Air Flow" in k:
                a_min = a[k]
                break

        vav_operating = "N/A"
        if a_max and a_max > 0:
            if a_min is not None:
                actual_min_frac = a_min / a_max
                vav_operating = (
                    "YES" if actual_min_frac >= min_fraction * 0.9 else "BELOW MIN"
                )
            elif a_mean is not None:
                vav_operating = "PARTIAL (hourly data unavailable)"
            else:
                vav_operating = "UNKNOWN (no time-series data)"

        rows.append(
            {
                "Zone": zone,
                "Before Max Flow (m3/s)": b_max,
                "After Max Flow (m3/s)": a_max,
                "After Mean Flow (m3/s)": a_mean,
                "After Min Flow (m3/s)": a_min,
                "VAV Operating Correctly": vav_operating,
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compare zone airflow rates before and after VAV measure."
    )
    parser.add_argument("before_sql", type=Path, help="Path to baseline SQL")
    parser.add_argument("after_sql", type=Path, help="Path to post-measure SQL")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write CSV report to this path (default: stdout)",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.3,
        help="Expected minimum airflow fraction (default 0.3)",
    )
    args = parser.parse_args()

    if not args.before_sql.exists():
        sys.exit(f"File not found: {args.before_sql}")
    if not args.after_sql.exists():
        sys.exit(f"File not found: {args.after_sql}")

    print(f"Parsing baseline:  {args.before_sql}")
    before = extract_zone_airflow(args.before_sql)
    print(f"  -> {len(before)} zones found")

    print(f"Parsing post-measure: {args.after_sql}")
    after = extract_zone_airflow(args.after_sql)
    print(f"  -> {len(after)} zones found")

    rows = compare(before, after, min_fraction=args.min_fraction)

    fieldnames = [
        "Zone",
        "Before Max Flow (m3/s)",
        "After Max Flow (m3/s)",
        "After Mean Flow (m3/s)",
        "After Min Flow (m3/s)",
        "VAV Operating Correctly",
    ]

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nReport written to {args.output}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    confirmed = sum(1 for r in rows if r["VAV Operating Correctly"] == "YES")
    below = sum(1 for r in rows if r["VAV Operating Correctly"] == "BELOW MIN")
    unknown = total - confirmed - below

    print(
        f"\nSummary: {confirmed}/{total} zones confirmed VAV operation, "
        f"{below} below minimum, {unknown} indeterminate."
    )


if __name__ == "__main__":
    main()
