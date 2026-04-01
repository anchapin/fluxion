"""
Component-Level Comparison: Fluxion vs EnergyPlus.

This module generates comprehensive comparison reports between Fluxion and
EnergyPlus simulation results at the component level.

Features:
- Time series plots (Fluxion vs EnergyPlus)
- Scatter plots with R² and regression line
- Error distribution histograms
- Pass/Fail metrics table per component
- HTML report generation

Usage:
    python -m tools.compare_components \
        --fluxion fluxion_components.csv \
        --energyplus energyplus_components.csv \
        --output component_comparison_report.html
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .statistical_metrics import (
    MetricsResult,
    calculate_all_metrics,
)


@dataclass
class ComponentComparison:
    """Comparison results for a single component."""

    component_type: str
    key_value: str  # Surface/zone name or "*"
    fluxion_data: np.ndarray
    energyplus_data: np.ndarray
    timestamps: np.ndarray
    units: str
    metrics: MetricsResult
    passes_criteria: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "component_type": self.component_type,
            "key_value": self.key_value,
            "units": self.units,
            "metrics": self.metrics.to_dict(),
            "passes_criteria": self.passes_criteria,
            "n_points": len(self.fluxion_data),
        }


class ComponentComparator:
    """Compare Fluxion and EnergyPlus component data."""

    # Pass/fail criteria per component type
    CRITERIA = {
        "ctf_flux": {"rmse_max": 10.0, "nmbe_max": 5.0, "r_squared_min": 0.95},
        "convective_flux": {"rmse_max": 15.0, "nmbe_max": 10.0, "r_squared_min": 0.90},
        "solar_flux": {"rmse_max": 20.0, "nmbe_max": 5.0, "r_squared_min": 0.95},
        "hvac_power": {"rmse_max": 200.0, "nmbe_max": 10.0, "r_squared_min": 0.90},
        "surface_temp": {"rmse_max": 1.0, "nmbe_max": 2.0, "r_squared_min": 0.98},
        "zone_temp": {"rmse_max": 1.0, "nmbe_max": 2.0, "r_squared_min": 0.99},
        "infiltration": {"rmse_max": 50.0, "nmbe_max": 5.0, "r_squared_min": 0.95},
        "zone_balance": {"rmse_max": 1.0, "nmbe_max": None, "r_squared_min": None},
    }

    def __init__(self, fluxion_csv: Path, energyplus_csv: Path):
        """Initialize comparator.

        Args:
            fluxion_csv: Path to Fluxion components CSV
            energyplus_csv: Path to EnergyPlus components CSV
        """
        self.fluxion_csv = fluxion_csv
        self.energyplus_csv = energyplus_csv
        self.comparisons: List[ComponentComparison] = []
        self.summary: Dict = {}

    def load_data(self) -> Tuple[Dict, Dict]:
        """Load CSV data into dictionaries.

        Returns:
            (fluxion_data, energyplus_data) dictionaries
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for data loading")

        # Load Fluxion data
        print(f"Loading Fluxion data: {self.fluxion_csv}")
        fluxion_df = pd.read_csv(self.fluxion_csv)

        # Load EnergyPlus data
        print(f"Loading EnergyPlus data: {self.energyplus_csv}")
        energyplus_df = pd.read_csv(self.energyplus_csv)

        # Group by component type and key
        fluxion_grouped = {}
        for comp_type in fluxion_df["component_type"].unique():
            comp_data = fluxion_df[fluxion_df["component_type"] == comp_type]
            for key_val in comp_data["key_value"].unique():
                key_data = comp_data[comp_data["key_value"] == key_val]
                key = f"{comp_type}:{key_val}"
                fluxion_grouped[key] = {
                    "timestep": key_data["timestep"].values,
                    "value": key_data["value"].values,
                    "units": key_data["units"].iloc[0],
                }

        # Group EnergyPlus data similarly
        energyplus_grouped = {}
        for comp_type in energyplus_df["component_type"].unique():
            comp_data = energyplus_df[energyplus_df["component_type"] == comp_type]
            for key_val in comp_data["key_value"].unique():
                key_data = comp_data[comp_data["key_value"] == key_val]
                key = f"{comp_type}:{key_val}"
                energyplus_grouped[key] = {
                    "timestep": key_data["timestep"].values,
                    "value": key_data["value"].values,
                    "units": key_data["units"].iloc[0],
                }

        return fluxion_grouped, energyplus_grouped

    def compare(self) -> List[ComponentComparison]:
        """Compare all matching components.

        Returns:
            List of ComponentComparison objects
        """
        fluxion_data, energyplus_data = self.load_data()

        # Find matching keys
        common_keys = set(fluxion_data.keys()) & set(energyplus_data.keys())

        if not common_keys:
            print("Warning: No matching components found!")
            print(f"Fluxion components: {list(fluxion_data.keys())[:10]}...")
            print(f"EnergyPlus components: {list(energyplus_data.keys())[:10]}...")
            return []

        print(f"Comparing {len(common_keys)} components...")

        for key in sorted(common_keys):
            comp_type, key_val = key.split(":", 1)

            flux = fluxion_data[key]
            ep = energyplus_data[key]

            # Align timesteps
            flux_ts = set(flux["timestep"])
            ep_ts = set(ep["timestep"])
            common_ts = sorted(flux_ts & ep_ts)

            if len(common_ts) < 10:
                print(
                    f"  Skipping {key}: insufficient common timesteps ({len(common_ts)})"
                )
                continue

            # Extract aligned data
            flux_values = flux["value"][np.isin(flux["timestep"], common_ts)]
            ep_values = ep["value"][np.isin(ep["timestep"], common_ts)]
            timestamps = np.array(common_ts)

            # Calculate metrics
            metrics = calculate_all_metrics(ep_values, flux_values, units=flux["units"])

            # Check pass/fail
            criteria = self.CRITERIA.get(comp_type, {})
            passes = metrics.passes_criteria(
                rmse_threshold=criteria.get("rmse_max"),
                nmbe_threshold=criteria.get("nmbe_max"),
                r_squared_threshold=criteria.get("r_squared_min"),
            )

            comparison = ComponentComparison(
                component_type=comp_type,
                key_value=key_val,
                fluxion_data=flux_values,
                energyplus_data=ep_values,
                timestamps=timestamps,
                units=flux["units"],
                metrics=metrics,
                passes_criteria=passes,
            )

            self.comparisons.append(comparison)

            status = "✓ PASS" if passes else "✗ FAIL"
            print(
                f"  {status} {key}: RMSE={metrics.rmse:.2f} {metrics.rmse_units}, "
                f"NMBE={metrics.nmbe:.1f}%, R²={metrics.r_squared:.3f}"
            )

        # Generate summary
        self._generate_summary()

        return self.comparisons

    def _generate_summary(self):
        """Generate summary statistics."""
        total = len(self.comparisons)
        passed = sum(1 for c in self.comparisons if c.passes_criteria)

        by_type = {}
        for comp in self.comparisons:
            if comp.component_type not in by_type:
                by_type[comp.component_type] = {"total": 0, "passed": 0}
            by_type[comp.component_type]["total"] += 1
            if comp.passes_criteria:
                by_type[comp.component_type]["passed"] += 1

        self.summary = {
            "total_components": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": (passed / total * 100) if total > 0 else 0.0,
            "by_type": by_type,
            "timestamp": datetime.now().isoformat(),
        }

    def generate_html_report(self, output_path: Path):
        """Generate HTML comparison report.

        Args:
            output_path: Output HTML file path
        """
        try:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
        except ImportError as e:
            raise ImportError(f"Required library not found: {e.name}")

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate plots
        plots_dir = output_path.parent / "plots"
        plots_dir.mkdir(exist_ok=True)

        plot_files = []
        for i, comp in enumerate(self.comparisons[:10]):  # Limit to first 10
            plot_path = self._generate_component_plots(comp, plots_dir, i, output_path)
            if plot_path:
                plot_files.append(plot_path)

        # Generate HTML
        html = self._build_html_report(plot_files)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    def _generate_component_plots(
        self, comp: ComponentComparison, plots_dir: Path, idx: int, output_path: Path
    ) -> Optional[Path]:
        """Generate plots for a single component.

        Args:
            comp: Component comparison
            plots_dir: Directory for plot files
            idx: Index for filename
            output_path: Root output path for relative path calculation

        Returns:
            Path to generated plot file
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{comp.component_type} ({comp.key_value}) - {comp.units}")

        # Plot 1: Time series
        ax = axes[0, 0]
        ax.plot(
            comp.timestamps, comp.energyplus_data, "b-", label="EnergyPlus", alpha=0.7
        )
        ax.plot(comp.timestamps, comp.fluxion_data, "r--", label="Fluxion", alpha=0.7)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(comp.units)
        ax.legend()
        ax.set_title("Time Series Comparison")
        ax.grid(True, alpha=0.3)

        # Plot 2: Scatter plot
        ax = axes[0, 1]
        ax.scatter(comp.energyplus_data, comp.fluxion_data, alpha=0.5, s=10)

        # Add 1:1 line
        min_val = min(comp.energyplus_data.min(), comp.fluxion_data.min())
        max_val = max(comp.energyplus_data.max(), comp.fluxion_data.max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", label="1:1 line")

        # Add regression line
        z = np.polyfit(comp.energyplus_data, comp.fluxion_data, 1)
        p = np.poly1d(z)
        ax.plot(
            comp.energyplus_data,
            p(comp.energyplus_data),
            "r-",
            label=f"Fit: y={z[0]:.2f}x+{z[1]:.2f}",
        )

        ax.set_xlabel("EnergyPlus")
        ax.set_ylabel("Fluxion")
        ax.legend()
        ax.set_title(f"Scatter Plot (R² = {comp.metrics.r_squared:.3f})")
        ax.grid(True, alpha=0.3)

        # Plot 3: Error distribution
        ax = axes[1, 0]
        errors = comp.fluxion_data - comp.energyplus_data
        ax.hist(errors, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(x=0, color="r", linestyle="--", label="Zero error")
        ax.axvline(
            x=errors.mean(),
            color="g",
            linestyle="-",
            label=f"Mean: {errors.mean():.2f}",
        )
        ax.set_xlabel(f"Error ({comp.units})")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.set_title("Error Distribution")
        ax.grid(True, alpha=0.3)

        # Plot 4: Hourly error profile
        ax = axes[1, 1]
        hours = comp.timestamps / 6.0  # Convert to hours
        ax.plot(hours, errors, "g-", alpha=0.5)
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("Hour")
        ax.set_ylabel(f"Error ({comp.units})")
        ax.set_title("Hourly Error Profile")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = plots_dir / f"component_{idx:03d}_{comp.component_type}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        return plot_path.relative_to(output_path.parent.parent)

    def _build_html_report(self, plot_files: List[Path]) -> str:
        """Build HTML report string.

        Args:
            plot_files: List of plot file paths

        Returns:
            HTML string
        """
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Component-Level Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .pass {{ color: green; font-weight: bold; }}
        .fail {{ color: red; font-weight: bold; }}
        .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
        img {{ max-width: 100%; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Component-Level Comparison Report</h1>
    <p><strong>Generated:</strong> {self.summary["timestamp"]}</p>
    <p><strong>Fluxion Data:</strong> {self.fluxion_csv.name}</p>
    <p><strong>EnergyPlus Data:</strong> {self.energyplus_csv.name}</p>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Components:</strong> {self.summary["total_components"]}</p>
        <p><strong>Passed:</strong> <span class="pass">{self.summary["passed"]}</span></p>
        <p><strong>Failed:</strong> <span class="fail">{self.summary["failed"]}</span></p>
        <p><strong>Pass Rate:</strong> {self.summary["pass_rate"]:.1f}%</p>
    </div>

    <h2>Results by Component Type</h2>
    <table>
        <tr>
            <th>Component Type</th>
            <th>Total</th>
            <th>Passed</th>
            <th>Failed</th>
            <th>Pass Rate</th>
        </tr>
"""

        for comp_type, stats in sorted(self.summary["by_type"].items()):
            pass_rate = (
                (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            )
            html += f"""        <tr>
            <td>{comp_type}</td>
            <td>{stats["total"]}</td>
            <td><span class="pass">{stats["passed"]}</span></td>
            <td><span class="fail">{stats["failed"]}</span></td>
            <td>{pass_rate:.1f}%</td>
        </tr>
"""

        html += """    </table>

    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Component</th>
            <th>Key</th>
            <th>Units</th>
            <th>RMSE</th>
            <th>NMBE (%)</th>
            <th>CV(RMSE) (%)</th>
            <th>R²</th>
            <th>Status</th>
        </tr>
"""

        for comp in self.comparisons:
            status_class = "pass" if comp.passes_criteria else "fail"
            status_text = "✓ PASS" if comp.passes_criteria else "✗ FAIL"
            html += f"""        <tr>
            <td>{comp.component_type}</td>
            <td>{comp.key_value}</td>
            <td>{comp.units}</td>
            <td>{comp.metrics.rmse:.3f}</td>
            <td>{comp.metrics.nmbe:.2f}</td>
            <td>{comp.metrics.cv_rmse:.2f}</td>
            <td>{comp.metrics.r_squared:.4f}</td>
            <td class="{status_class}">{status_text}</td>
        </tr>
"""

        html += """    </table>

    <h2>Plots</h2>
"""

        for plot_file in plot_files:
            html += f"""    <img src="{plot_file}" alt="Component Plot"><br>
"""

        html += """
</body>
</html>
"""

        return html


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare Fluxion and EnergyPlus component data"
    )
    parser.add_argument(
        "--fluxion",
        "-f",
        type=Path,
        required=True,
        help="Path to Fluxion components CSV",
    )
    parser.add_argument(
        "--energyplus",
        "-e",
        type=Path,
        required=True,
        help="Path to EnergyPlus components CSV",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default="component_comparison_report.html",
        help="Output HTML report path",
    )
    parser.add_argument("--json", type=Path, help="Optional JSON output path")
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip plot generation (faster)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose output"
    )

    args = parser.parse_args()

    # Check input files exist
    for path in [args.fluxion, args.energyplus]:
        if not path.exists():
            print(f"Error: File not found: {path}")
            return 1

    # Run comparison
    comparator = ComponentComparator(args.fluxion, args.energyplus)
    comparisons = comparator.compare()

    if not comparisons:
        print("No comparisons to generate")
        return 1

    # Generate outputs
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "summary": comparator.summary,
                    "results": [c.to_dict() for c in comparisons],
                },
                f,
                indent=2,
            )
        print(f"Generated JSON: {args.json}")

    if not args.no_plots:
        comparator.generate_html_report(args.output)
    else:
        # Generate simple text report
        print("\n" + "=" * 60)
        print("COMPONENT COMPARISON SUMMARY")
        print("=" * 60)
        print(f"Total: {comparator.summary['total_components']}")
        print(f"Passed: {comparator.summary['passed']}")
        print(f"Failed: {comparator.summary['failed']}")
        print(f"Pass Rate: {comparator.summary['pass_rate']:.1f}%")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
