"""
Compare zone airflow rates before and after the CV-to-VAV measure.

Parses two EnergyPlus SQLite output databases (before and after the measure)
and reports zone-level airflow comparison to verify VAV boxes are operating
correctly — i.e., minimum airflow fractions are respected at part load while
peak flows are maintained.

Reads from:
  - TabularDataWithStrings  (HVAC Sizing Summary, Zone Equipment Summary)
  - ZoneSizes               (design cooling/heating air flow rates)
  - ComponentSizes          (terminal box max and min flow rates)

Usage:
    python compare_airflow.py --before baseline.sql --after post_measure.sql [--output report.csv]
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


def query_zone_sizes(db_path: str) -> pd.DataFrame:
    """Query ZoneSizes table for design airflow rates.

    Returns DataFrame with columns:
        zone_name, calc_cooling_airflow, calc_heating_airflow, user_cooling_airflow, user_heating_airflow
    All airflow values in m3/s.
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT
                ZoneName       AS zone_name,
                ZoneCoolingDesignAirFlow    AS calc_cooling_airflow,
                ZoneHeatingDesignAirFlow    AS calc_heating_airflow,
                ZoneCoolingDesignAirFlowRate AS user_cooling_airflow,
                ZoneHeatingDesignAirFlowRate AS user_heating_airflow
            FROM ZoneSizes
            """,
            conn,
        )
    finally:
        conn.close()
    return df


def query_component_sizes(db_path: str) -> pd.DataFrame:
    """Query ComponentSizes for air terminal flow data.

    Returns DataFrame with columns:
        component_name, component_type, description, value
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT
                CompName      AS component_name,
                CompType      AS component_type,
                Description   AS description,
                Value         AS value
            FROM ComponentSizes
            WHERE CompType LIKE 'AirTerminal%'
               OR Description LIKE '%Air Flow%'
               OR Description LIKE '%Minimum Air Flow%'
            """,
            conn,
        )
    finally:
        conn.close()
    return df


def query_tabular_airflow(db_path: str) -> pd.DataFrame:
    """Query TabularDataWithStrings for zone design airflow from sizing summary.

    Looks for 'HVAC Sizing Summary' or 'Equipment Summary' reports.
    Returns DataFrame with columns: zone_name, design_airflow, report_name
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT
                ReportName      AS report_name,
                ReportForString AS report_for,
                TableName       AS table_name,
                RowName         AS zone_name,
                ColumnName      AS column_name,
                Value           AS value
            FROM TabularDataWithStrings
            WHERE (ReportName LIKE '%Sizing%'
                   OR ReportName LIKE '%Equipment%')
              AND (ColumnName LIKE '%Air Flow%'
                   OR ColumnName LIKE '%Airflow%'
                   OR RowName LIKE '%Air%')
            """,
            conn,
        )
    finally:
        conn.close()
    return df


def compare_zone_airflows(before_path: str, after_path: str) -> pd.DataFrame:
    """Compare zone design airflows between baseline and post-measure databases.

    Returns a merged DataFrame with columns:
        zone_name, before_cooling_af, after_cooling_af,
        before_heating_af, after_heating_af,
        cooling_diff, heating_diff, cooling_pct_change, heating_pct_change
    """
    before = query_zone_sizes(before_path)
    after = query_zone_sizes(after_path)

    before = before.rename(
        columns={
            "calc_cooling_airflow": "before_cooling_af",
            "calc_heating_airflow": "before_heating_af",
        }
    )
    after = after.rename(
        columns={
            "calc_cooling_airflow": "after_cooling_af",
            "calc_heating_airflow": "after_heating_af",
        }
    )

    merged = pd.merge(
        before[["zone_name", "before_cooling_af", "before_heating_af"]],
        after[["zone_name", "after_cooling_af", "after_heating_af"]],
        on="zone_name",
        how="outer",
        indicator=True,
    )

    merged["cooling_diff"] = merged["after_cooling_af"] - merged["before_cooling_af"]
    merged["heating_diff"] = merged["after_heating_af"] - merged["before_heating_af"]

    merged["cooling_pct_change"] = (
        (merged["after_cooling_af"] - merged["before_cooling_af"])
        / merged["before_cooling_af"].replace(0, float("nan"))
        * 100
    ).round(2)

    merged["heating_pct_change"] = (
        (merged["after_heating_af"] - merged["before_heating_af"])
        / merged["before_heating_af"].replace(0, float("nan"))
        * 100
    ).round(2)

    return merged


def check_vav_minimums(after_path: str, min_fraction: float = 0.3) -> pd.DataFrame:
    """Check that VAV terminal minimum airflows respect the configured fraction.

    Queries ComponentSizes for terminal box data and verifies that the
    minimum airflow is >= min_fraction * maximum airflow for each VAV box.

    Returns DataFrame with columns:
        component_name, max_airflow, min_airflow, actual_fraction,
        required_fraction, compliant
    """
    comp = query_component_sizes(after_path)

    pivoted = comp.pivot_table(
        index="component_name",
        columns="description",
        values="value",
        aggfunc="first",
    ).reset_index()

    results = []
    for _, row in pivoted.iterrows():
        name = row["component_name"]
        max_af = row.get(
            "Maximum Air Flow Rate", row.get("Design Size Maximum Air Flow Rate", None)
        )
        min_af = row.get(
            "Minimum Air Flow Rate", row.get("Design Size Minimum Air Flow Rate", None)
        )

        if max_af is None or min_af is None:
            continue

        try:
            max_val = float(max_af)
            min_val = float(min_af)
        except (ValueError, TypeError):
            continue

        if max_val <= 0:
            continue

        actual_frac = min_val / max_val
        results.append(
            {
                "component_name": name,
                "max_airflow_m3s": max_val,
                "min_airflow_m3s": min_val,
                "actual_fraction": round(actual_frac, 4),
                "required_fraction": min_fraction,
                "compliant": actual_frac >= min_fraction - 0.01,
            }
        )

    return pd.DataFrame(results)


def generate_report(
    before_path: str,
    after_path: str,
    min_fraction: float = 0.3,
    output_path: Optional[str] = None,
) -> str:
    """Generate a human-readable comparison report.

    Returns the report as a string. Optionally writes to output_path.
    """
    zone_comparison = compare_zone_airflows(before_path, after_path)
    vav_check = check_vav_minimums(after_path, min_fraction)

    lines = []
    lines.append("=" * 72)
    lines.append("ZONE AIRFLOW COMPARISON: BEFORE vs. AFTER CV->VAV MEASURE")
    lines.append("=" * 72)
    lines.append("")

    lines.append(f"Before database: {before_path}")
    lines.append(f"After database:  {after_path}")
    lines.append(f"VAV min fraction: {min_fraction}")
    lines.append("")

    # Zone airflow table
    lines.append("-" * 72)
    lines.append("ZONE DESIGN AIRFLOW COMPARISON (m3/s)")
    lines.append("-" * 72)
    if zone_comparison.empty:
        lines.append("  No zone sizing data found.")
    else:
        lines.append(
            f"{'Zone':<30} {'Before Clg':>10} {'After Clg':>10} "
            f"{'%Chg':>8} {'Before Htg':>10} {'After Htg':>10} {'%Chg':>8}"
        )
        lines.append("-" * 72)
        for _, r in zone_comparison.iterrows():
            lines.append(
                f"{r['zone_name']:<30} "
                f"{r['before_cooling_af']:>10.4f} {r['after_cooling_af']:>10.4f} "
                f"{r['cooling_pct_change']:>7.1f}% "
                f"{r['before_heating_af']:>10.4f} {r['after_heating_af']:>10.4f} "
                f"{r['heating_pct_change']:>7.1f}%"
            )

        new_zones = zone_comparison[zone_comparison["_merge"] == "right_only"]
        removed_zones = zone_comparison[zone_comparison["_merge"] == "left_only"]
        if not new_zones.empty:
            lines.append("")
            lines.append(
                f"  New zones in post-measure: {', '.join(new_zones['zone_name'].tolist())}"
            )
        if not removed_zones.empty:
            lines.append(
                f"  Removed zones: {', '.join(removed_zones['zone_name'].tolist())}"
            )

    lines.append("")

    # VAV minimum compliance table
    lines.append("-" * 72)
    lines.append("VAV MINIMUM AIRFLOW COMPLIANCE CHECK")
    lines.append("-" * 72)
    if vav_check.empty:
        lines.append("  No VAV terminal sizing data found in post-measure database.")
    else:
        compliant_count = vav_check["compliant"].sum()
        total_count = len(vav_check)
        lines.append(f"  Compliant: {compliant_count}/{total_count} terminals")
        lines.append("")
        lines.append(
            f"{'Terminal':<40} {'Max':>8} {'Min':>8} {'Fraction':>9} {'OK':>4}"
        )
        lines.append("-" * 72)
        for _, r in vav_check.iterrows():
            status = "YES" if r["compliant"] else "FAIL"
            lines.append(
                f"{r['component_name']:<40} "
                f"{r['max_airflow_m3s']:>8.4f} {r['min_airflow_m3s']:>8.4f} "
                f"{r['actual_fraction']:>9.4f} {status:>4}"
            )

        failures = vav_check[~vav_check["compliant"]]
        if not failures.empty:
            lines.append("")
            lines.append(
                "  WARNING: The following terminals have min fractions below the target:"
            )
            for _, r in failures.iterrows():
                lines.append(
                    f"    {r['component_name']}: actual={r['actual_fraction']:.4f} "
                    f"< required={r['required_fraction']:.4f}"
                )

    lines.append("")
    lines.append("=" * 72)

    report = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Compare zone airflow rates before and after CV-to-VAV measure."
    )
    parser.add_argument(
        "--before",
        required=True,
        help="Path to baseline EnergyPlus SQLite output database (eplusout.sql).",
    )
    parser.add_argument(
        "--after",
        required=True,
        help="Path to post-measure EnergyPlus SQLite output database.",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.3,
        help="Expected VAV minimum airflow fraction (default: 0.3).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the text report.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to write zone comparison as CSV.",
    )

    args = parser.parse_args()

    report = generate_report(args.before, args.after, args.min_fraction, args.output)
    print(report)

    if args.csv:
        zone_comparison = compare_zone_airflows(args.before, args.after)
        zone_comparison.drop(columns=["_merge"], inplace=True, errors="ignore")
        zone_comparison.to_csv(args.csv, index=False)
        print(f"\nZone comparison CSV written to: {args.csv}")


if __name__ == "__main__":
    main()
