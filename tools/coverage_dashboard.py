#!/usr/bin/env python3
"""
Coverage Dashboard Generator for Fluxion Physics Modules

Generates HTML dashboard showing:
- Line coverage by module
- Branch coverage by function
- Trend comparison (if baseline available)
- Missing coverage areas

Usage:
    python tools/coverage_dashboard.py [--baseline coverage/baseline/html/index.html]
    python tools/coverage_dashboard.py --output coverage/dashboard.html
"""

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

TARGET_COVERAGE = 90.0


@dataclass
class ModuleCoverage:
    """Coverage data for a single module."""

    name: str
    file_path: str
    line_coverage: float
    line_covered: int
    line_total: int
    branch_coverage: float = 0.0
    branch_covered: int = 0
    branch_total: int = 0
    functions_covered: int = 0
    functions_total: int = 0

    @property
    def gap(self) -> float:
        """Coverage gap from target."""
        return max(0.0, TARGET_COVERAGE - self.line_coverage)

    @property
    def status(self) -> str:
        """Status indicator."""
        if self.line_coverage >= TARGET_COVERAGE:
            return "✅ PASS"
        elif self.line_coverage >= 70.0:
            return "⚠️ WARN"
        else:
            return "❌ FAIL"

    @property
    def status_class(self) -> str:
        """CSS class for status."""
        if self.line_coverage >= TARGET_COVERAGE:
            return "status-pass"
        elif self.line_coverage >= 70.0:
            return "status-warn"
        else:
            return "status-fail"


@dataclass
class BaselineCoverage:
    """Baseline coverage data for comparison."""

    modules: Dict[str, float]
    timestamp: str


def get_tarpaulin_coverage() -> List[ModuleCoverage]:
    """
    Parse tarpaulin cobertura XML coverage report.

    Returns:
        List of ModuleCoverage objects for physics modules.
    """
    try:
        subprocess.run(
            [
                "cargo",
                "tarpaulin",
                "--out",
                "Xml",
                "--output-dir",
                "coverage/tarpaulin_temp",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print("Error: tarpaulin timed out after 10 minutes")
        return []
    except FileNotFoundError:
        print(
            "Error: cargo-tarpaulin not found. Install with: cargo install cargo-tarpaulin"
        )
        return []

    # Parse cobertura.xml
    xml_path = Path("coverage/tarpaulin_temp/cobertura.xml")
    if not xml_path.exists():
        print(f"Error: Coverage XML not found at {xml_path}")
        print("Available files:")
        for p in Path("coverage/tarpaulin_temp").glob("*"):
            print(f"  {p}")
        return []

    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    modules: Dict[str, ModuleCoverage] = {}

    # Find all classes in coverage report
    for cls in root.findall(".//class"):
        filename = cls.get("filename", "")
        if "src/physics/" not in filename:
            continue

        # Extract module name from filename
        module_match = re.search(r"src/physics/([^/]+)\.rs", filename)
        if not module_match:
            continue

        module_name = module_match.group(1)

        # Get coverage metrics
        line_rate = float(cls.get("line-rate", "0.0"))
        lines_covered = int(cls.get("lines-covered", "0"))
        lines_valid = int(cls.get("lines-valid", "0"))

        # Aggregate coverage by module
        if module_name in modules:
            # Combine with existing module data
            existing = modules[module_name]
            total_lines = existing.line_total + lines_valid
            total_covered = existing.line_covered + lines_covered
            modules[module_name] = ModuleCoverage(
                name=module_name,
                file_path=filename,
                line_coverage=(
                    (total_covered / total_lines * 100) if total_lines > 0 else 0.0
                ),
                line_covered=int(total_covered),
                line_total=total_lines,
            )
        else:
            modules[module_name] = ModuleCoverage(
                name=module_name,
                file_path=filename,
                line_coverage=line_rate * 100,
                line_covered=lines_covered,
                line_total=lines_valid,
            )

    return list(modules.values())


def get_llvm_cov_coverage() -> List[ModuleCoverage]:
    """
    Parse cargo llvm-cov JSON coverage report.

    Returns:
        List of ModuleCoverage objects for physics modules.
    """
    try:
        result = subprocess.run(
            ["cargo", "llvm-cov", "report", "--json", "--output-path", "-"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        coverage_data = json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        print("Error: llvm-cov timed out after 10 minutes")
        return []
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Failed to get llvm-cov data: {e}")
        return []

    modules: Dict[str, ModuleCoverage] = {}

    for source in coverage_data.get("source", []):
        filename = source.get("name", "")
        if "src/physics/" not in filename:
            continue

        module_match = re.search(r"src/physics/([^/]+)\.rs", filename)
        if not module_match:
            continue

        module_name = module_match.group(1)

        # Calculate line coverage
        total_lines = 0
        covered_lines = 0

        for function in source.get("functions", []):
            for line in function.get("lines", []):
                total_lines += 1
                if line.get("execution_count", 0) > 0:
                    covered_lines += 1

        modules[module_name] = ModuleCoverage(
            name=module_name,
            file_path=filename,
            line_coverage=(
                (covered_lines / total_lines * 100) if total_lines > 0 else 0.0
            ),
            line_covered=covered_lines,
            line_total=total_lines,
        )

    return list(modules.values())


def parse_baseline(baseline_path: Path) -> Optional[BaselineCoverage]:
    """
    Parse baseline coverage report for trend comparison.

    Args:
        baseline_path: Path to baseline HTML or JSON report

    Returns:
        BaselineCoverage or None if parsing fails.
    """
    # For now, return None - trend tracking can be added later
    return None


def generate_html_dashboard(
    modules: List[ModuleCoverage],
    baseline: Optional[BaselineCoverage] = None,
    output_path: Path = Path("coverage/dashboard.html"),
) -> None:
    """
    Generate HTML dashboard.

    Args:
        modules: List of module coverage data
        baseline: Optional baseline for comparison
        output_path: Path to output HTML file
    """
    # Sort modules by name
    modules_sorted = sorted(modules, key=lambda m: m.name)

    # Calculate summary stats
    avg_coverage = (
        sum(m.line_coverage for m in modules) / len(modules) if modules else 0.0
    )
    total_lines = sum(m.line_total for m in modules)
    total_covered = sum(m.line_covered for m in modules)
    overall_coverage = (total_covered / total_lines * 100) if total_lines > 0 else 0.0
    passing_count = sum(1 for m in modules if m.line_coverage >= TARGET_COVERAGE)
    warning_count = sum(1 for m in modules if 70.0 <= m.line_coverage < TARGET_COVERAGE)
    failing_count = sum(1 for m in modules if m.line_coverage < 70.0)

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fluxion Physics Coverage Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
            font-weight: 700;
        }}
        .header p {{
            margin: 10px 0 0;
            opacity: 0.9;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-card h3 {{
            color: #6c757d;
            font-size: 14px;
            margin-bottom: 10px;
            text-transform: uppercase;
        }}
        .summary-card .value {{
            font-size: 36px;
            font-weight: 700;
            color: #667eea;
        }}
        .summary-card .sub {{
            font-size: 14px;
            color: #6c757d;
            margin-top: 5px;
        }}
        .section {{
            padding: 30px;
        }}
        .section h2 {{
            color: #2d3748;
            margin-bottom: 20px;
            font-size: 24px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
        }}
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #4a5568;
            font-size: 14px;
            text-transform: uppercase;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .status-pass {{ color: #28a745; font-weight: 600; }}
        .status-warn {{ color: #ffc107; font-weight: 600; }}
        .status-fail {{ color: #dc3545; font-weight: 600; }}
        .coverage-bar {{
            width: 100px;
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
        }}
        .coverage-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        .coverage-fill.pass {{ background: #28a745; }}
        .coverage-fill.warn {{ background: #ffc107; }}
        .coverage-fill.fail {{ background: #dc3545; }}
        .percentage {{
            font-weight: 600;
            min-width: 60px;
        }}
        .timestamp {{
            text-align: center;
            padding: 20px;
            color: #6c757d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Fluxion Physics Coverage Dashboard</h1>
            <p>Target: {TARGET_COVERAGE}% | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <div class="summary">
            <div class="summary-card">
                <h3>Overall Coverage</h3>
                <div class="value">{overall_coverage:.2f}%</div>
                <div class="sub">{total_covered}/{total_lines} lines</div>
            </div>
            <div class="summary-card">
                <h3>Average Coverage</h3>
                <div class="value">{avg_coverage:.2f}%</div>
                <div class="sub">{len(modules)} modules</div>
            </div>
            <div class="summary-card">
                <h3>✅ Passing</h3>
                <div class="value">{passing_count}</div>
                <div class="sub">≥{TARGET_COVERAGE}%</div>
            </div>
            <div class="summary-card">
                <h3>⚠️ Warning</h3>
                <div class="value">{warning_count}</div>
                <div class="sub">70-89%</div>
            </div>
            <div class="summary-card">
                <h3>❌ Failing</h3>
                <div class="value">{failing_count}</div>
                <div class="sub">&lt;70%</div>
            </div>
        </div>

        <div class="section">
            <h2>Module Coverage</h2>
            <table>
                <thead>
                    <tr>
                        <th>Module</th>
                        <th>Coverage</th>
                        <th>Progress</th>
                        <th>Lines</th>
                        <th>Status</th>
                        <th>Gap</th>
                    </tr>
                </thead>
                <tbody>
"""

    # Add module rows
    for module in modules_sorted:
        fill_class = (
            "pass"
            if module.line_coverage >= TARGET_COVERAGE
            else "warn" if module.line_coverage >= 70 else "fail"
        )
        gap_str = f"{module.gap:.2f}%" if module.gap > 0 else "-"

        html += f"""
                    <tr>
                        <td><strong>{module.name}</strong></td>
                        <td class="percentage">{module.line_coverage:.2f}%</td>
                        <td>
                            <div class="coverage-bar">
                                <div class="coverage-fill {fill_class}" style="width: {module.line_coverage}%"></div>
                            </div>
                        </td>
                        <td>{module.line_covered}/{module.line_total}</td>
                        <td class="{module.status_class}">{module.status}</td>
                        <td style="color: #dc3545; font-weight: 600;">{gap_str}</td>
                    </tr>
"""

    # Add footer
    html += """
                </tbody>
            </table>
        </div>

        <div class="timestamp">
            <p>Generated by <code>tools/coverage_dashboard.py</code></p>
            <p>Regenerate with: <code>python tools/coverage_dashboard.py</code></p>
        </div>
    </div>
</body>
</html>
"""

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Dashboard written to: {output_path}")
    print(f"Overall coverage: {overall_coverage:.2f}%")
    print(
        f"Passing: {passing_count}, Warning: {warning_count}, Failing: {failing_count}"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate HTML coverage dashboard for Fluxion physics modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dashboard from tarpaulin
  python tools/coverage_dashboard.py --method tarpaulin

  # Generate dashboard from llvm-cov
  python tools/coverage_dashboard.py --method llvm-cov

  # Specify custom output
  python tools/coverage_dashboard.py --output coverage/custom-dashboard.html
        """,
    )

    parser.add_argument(
        "--method",
        choices=["tarpaulin", "llvm-cov"],
        default="tarpaulin",
        help="Coverage tool to use (default: tarpaulin)",
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        help="Path to baseline report for trend comparison",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("coverage/dashboard.html"),
        help="Output HTML file path (default: coverage/dashboard.html)",
    )

    args = parser.parse_args()

    # Get coverage data
    print(f"Generating coverage dashboard using {args.method}...")
    if args.method == "tarpaulin":
        modules = get_tarpaulin_coverage()
    else:
        modules = get_llvm_cov_coverage()

    if not modules:
        print("Error: No physics modules found in coverage data")
        return 1

    # Parse baseline if provided
    baseline = None
    if args.baseline and args.baseline.exists():
        baseline = parse_baseline(args.baseline)

    # Generate dashboard
    generate_html_dashboard(modules, baseline, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
