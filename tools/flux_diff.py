#!/usr/bin/env python3
"""
flux-diff: Numerical Regression Tracking Tool

Compares two Fluxion output CSV/JSON files and reports numerical drift
with configurable tolerance thresholds per metric type.
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToleranceConfig:
    temperature: float = 0.05
    energy: float = 0.01
    power: float = 0.01
    flow: float = 0.01
    generic: float = 0.001


@dataclass
class ColumnDiff:
    column: str
    baseline_value: float
    current_value: float
    delta: float
    percent_diff: float
    exceeds_tolerance: bool
    tolerance: float
    tolerance_type: str


@dataclass
class RowDiff:
    row_index: int
    column_diffs: list[ColumnDiff]
    has_violation: bool = False


@dataclass
class DiffReport:
    baseline_file: str
    current_file: str
    total_rows: int
    rows_with_violations: int
    columns_analyzed: int
    tolerance_config: ToleranceConfig
    row_diffs: list[RowDiff] = field(default_factory=list)
    violations: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "baseline_file": self.baseline_file,
                "current_file": self.current_file,
                "total_rows": self.total_rows,
                "rows_with_violations": self.rows_with_violations,
                "columns_analyzed": self.columns_analyzed,
                "tolerance_config": {
                    "temperature": self.tolerance_config.temperature,
                    "energy": self.tolerance_config.energy,
                    "power": self.tolerance_config.power,
                    "flow": self.tolerance_config.flow,
                    "generic": self.tolerance_config.generic,
                },
                "violations": self.violations,
                "summary": {
                    "status": "FAIL" if self.rows_with_violations > 0 else "PASS",
                    "violation_rate": f"{self.rows_with_violations / max(self.total_rows, 1) * 100:.2f}%",
                },
            },
            indent=2,
        )

    def to_text(self) -> str:
        lines = [
            "flux-diff Report",
            "=" * 60,
            f"Baseline: {self.baseline_file}",
            f"Current:  {self.current_file}",
            "",
            "Tolerance Configuration:",
            f"  Temperature: ±{self.tolerance_config.temperature}°C",
            f"  Energy:     ±{self.tolerance_config.energy * 100:.2f}%",
            f"  Power:      ±{self.tolerance_config.power * 100:.2f}%",
            f"  Flow:       ±{self.tolerance_config.flow * 100:.2f}%",
            f"  Generic:    ±{self.tolerance_config.generic * 100:.2f}%",
            "",
            f"Total rows analyzed: {self.total_rows}",
            f"Rows with violations: {self.rows_with_violations}",
            f"Status: {'FAIL' if self.rows_with_violations > 0 else 'PASS'}",
            "",
        ]

        if self.violations:
            lines.append("Violations:")
            lines.append("-" * 60)
            for v in self.violations[:50]:
                lines.append(
                    f"  Row {v['row_index']}, Col '{v['column']}': "
                    f"{v['baseline_value']:.6g} -> {v['current_value']:.6g} "
                    f"(Δ={v['delta']:+.6g}, {v['percent_diff']:+.2f}%, "
                    f"tol={v['tolerance']}, type={v['tolerance_type']})"
                )
            if len(self.violations) > 50:
                lines.append(f"  ... and {len(self.violations) - 50} more violations")

        return "\n".join(lines)


def infer_tolerance_type(column: str) -> str:
    col_lower = column.lower()
    if any(x in col_lower for x in ["temp", "outdoor", "zone", "indoor", "ambient"]):
        return "temperature"
    if any(
        x in col_lower for x in ["energy", "heating", "cooling", "load", "consumption"]
    ):
        return "energy"
    if any(x in col_lower for x in ["power", "watt", "kw", "mw"]):
        return "power"
    if any(x in col_lower for x in ["flow", "cfm", "m3", "l/s", "velocity"]):
        return "flow"
    return "generic"


def get_tolerance(column: str, config: ToleranceConfig) -> tuple[float, str]:
    tol_type = infer_tolerance_type(column)
    tol_map = {
        "temperature": config.temperature,
        "energy": config.energy,
        "power": config.power,
        "flow": config.flow,
        "generic": config.generic,
    }
    return tol_map[tol_type], tol_type


def parse_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames) if reader.fieldnames else []
        rows = list(reader)
    return headers, rows


def parse_json(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "timeseries" in data:
        data = data["timeseries"]
    elif isinstance(data, dict) and "data" in data:
        data = data["data"]

    if not isinstance(data, list):
        data = [data]

    if not data:
        return [], []

    if isinstance(data[0], dict):
        headers = list(data[0].keys())
        rows = [{str(k): str(v) for k, v in row.items()} for row in data]
        return headers, rows

    return [], []


def compute_diff(
    baseline_file: str,
    current_file: str,
    baseline_data: tuple[list[str], list[dict[str, str]]],
    current_data: tuple[list[str], list[dict[str, str]]],
    config: ToleranceConfig,
) -> DiffReport:
    baseline_headers, baseline_rows = baseline_data
    current_headers, current_rows = current_data

    common_columns = [col for col in baseline_headers if col in current_headers]
    num_rows = min(len(baseline_rows), len(current_rows))

    report = DiffReport(
        baseline_file=baseline_file,
        current_file=current_file,
        total_rows=num_rows,
        rows_with_violations=0,
        columns_analyzed=len(common_columns),
        tolerance_config=config,
    )

    for row_idx in range(num_rows):
        b_row = baseline_rows[row_idx]
        c_row = current_rows[row_idx]
        row_diffs: list[ColumnDiff] = []
        has_violation = False

        for col in common_columns:
            b_val_str = b_row.get(col, "")
            c_val_str = c_row.get(col, "")

            try:
                b_val = float(b_val_str)
            except (ValueError, TypeError):
                continue

            try:
                c_val = float(c_val_str)
            except (ValueError, TypeError):
                continue

            delta = c_val - b_val
            percent_diff = (delta / abs(b_val) * 100.0) if b_val != 0 else 0.0

            tolerance, tol_type = get_tolerance(col, config)
            is_absolute = tol_type == "temperature"

            if is_absolute:
                exceeds = abs(delta) > tolerance
            else:
                exceeds = abs(percent_diff / 100.0) > tolerance

            col_diff = ColumnDiff(
                column=col,
                baseline_value=b_val,
                current_value=c_val,
                delta=delta,
                percent_diff=percent_diff,
                exceeds_tolerance=exceeds,
                tolerance=tolerance,
                tolerance_type=tol_type,
            )
            row_diffs.append(col_diff)

            if exceeds:
                has_violation = True
                report.violations.append(
                    {
                        "row_index": row_idx,
                        "column": col,
                        "baseline_value": b_val,
                        "current_value": c_val,
                        "delta": delta,
                        "percent_diff": percent_diff,
                        "tolerance": tolerance,
                        "tolerance_type": tol_type,
                    }
                )

        report.row_diffs.append(
            RowDiff(
                row_index=row_idx,
                column_diffs=row_diffs,
                has_violation=has_violation,
            )
        )
        if has_violation:
            report.rows_with_violations += 1

    return report


def main():
    parser = argparse.ArgumentParser(
        prog="flux-diff",
        description="Numerical regression tracking for Fluxion output files",
    )
    parser.add_argument("baseline", type=Path, help="Baseline CSV/JSON file")
    parser.add_argument("current", type=Path, help="Current PR output CSV/JSON file")
    parser.add_argument(
        "--temperature-tol",
        type=float,
        default=0.05,
        help="Temperature tolerance in °C (default: 0.05)",
    )
    parser.add_argument(
        "--energy-tol",
        type=float,
        default=0.01,
        help="Energy tolerance as fraction (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--power-tol",
        type=float,
        default=0.01,
        help="Power tolerance as fraction (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--flow-tol",
        type=float,
        default=0.01,
        help="Flow tolerance as fraction (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--generic-tol",
        type=float,
        default=0.001,
        help="Generic tolerance as fraction (default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument("--output", type=Path, help="Output file (default: stdout)")

    args = parser.parse_args()

    if not args.baseline.exists():
        print(f"Error: Baseline file not found: {args.baseline}", file=sys.stderr)
        sys.exit(1)
    if not args.current.exists():
        print(f"Error: Current file not found: {args.current}", file=sys.stderr)
        sys.exit(1)

    config = ToleranceConfig(
        temperature=args.temperature_tol,
        energy=args.energy_tol,
        power=args.power_tol,
        flow=args.flow_tol,
        generic=args.generic_tol,
    )

    if args.baseline.suffix.lower() == ".json":
        baseline_data = parse_json(args.baseline)
    else:
        baseline_data = parse_csv(args.baseline)

    if args.current.suffix.lower() == ".json":
        current_data = parse_json(args.current)
    else:
        current_data = parse_csv(args.current)

    report = compute_diff(
        str(args.baseline),
        str(args.current),
        baseline_data,
        current_data,
        config,
    )

    output = report.to_json() if args.format == "json" else report.to_text()

    if args.output:
        args.output.write_text(output)
        print(f"Report written to: {args.output}")
    else:
        print(output)

    sys.exit(1 if report.rows_with_violations > 0 else 0)


if __name__ == "__main__":
    main()
