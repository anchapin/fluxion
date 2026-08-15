#!/usr/bin/env python3
"""
Release Gate Checker for Fluxion

Issue #505: Define release gates for accuracy and performance

This script evaluates all release gates defined in release_gates.yaml and
reports whether each gate has passed or failed. It is used in CI to fail
builds when thresholds are breached and generates status reports.

Usage:
    python scripts/release_gate_checker.py [--verbose] [--json] [--update-baseline]

Exit codes:
    0 - All gates passed
    1 - One or more gates failed
    2 - Could not evaluate gates (missing data)
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml  # type: ignore


@dataclass
class GateResult:
    """Result of a single gate evaluation."""

    name: str
    category: str
    passed: bool
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    details: dict = field(default_factory=dict)


@dataclass
class GateReport:
    """Complete report of all gate evaluations."""

    timestamp: str
    overall_passed: bool
    gates: list[GateResult]
    summary: dict


class ReleaseGateChecker:
    """Evaluates release gates against current project state."""

    def __init__(self, gates_config: dict, project_root: Path):
        self.config = gates_config
        self.project_root = project_root
        self.results: list[GateResult] = []

    def check_validation_gates(self, validation_results: dict) -> list[GateResult]:
        """Check all validation gates."""
        results = []
        validation_config = self.config.get("validation", {})
        summary = validation_results.get("summary", {})
        cases = validation_results.get("cases", {})

        # Overall pass rate
        pass_rate = summary.get("pass_rate", 0.0)
        min_pass_rate = validation_config.get("min_pass_rate", 4.0)
        results.append(
            GateResult(
                name="overall_pass_rate",
                category="validation",
                passed=pass_rate >= min_pass_rate,
                message=f"Pass rate {pass_rate:.1f}% vs required {min_pass_rate}%",
                value=pass_rate,
                threshold=min_pass_rate,
            )
        )

        # MAE check
        mae = summary.get("mae", 0.0)
        max_mae = validation_config.get("max_mae", 30.0)
        results.append(
            GateResult(
                name="max_mae",
                category="validation",
                passed=mae <= max_mae,
                message=f"MAE {mae:.2f}% vs max allowed {max_mae}%",
                value=mae,
                threshold=max_mae,
            )
        )

        # Individual case deviations
        individual_config = validation_config.get("individual", {})
        max_deviation = individual_config.get("max_deviation", 150.0)
        extreme_limit = individual_config.get("extreme_deviation_limit", 15)
        known_failures = set(individual_config.get("known_failures", []))

        extreme_count = 0
        for case_id, case_data in cases.items():
            heating = case_data.get("heating", 0)
            heating_min = case_data.get("heating_min", 0)
            heating_max = case_data.get("heating_max", 0)
            cooling = case_data.get("cooling", 0)
            cooling_min = case_data.get("cooling_min", 0)
            cooling_max = case_data.get("cooling_max", 0)

            if case_id in known_failures:
                continue

            # Calculate deviations
            if heating_min > 0:
                h_dev = (
                    abs(heating - (heating_min + heating_max) / 2)
                    / ((heating_min + heating_max) / 2)
                    * 100
                )
            else:
                h_dev = 0

            if cooling_min > 0:
                c_dev = (
                    abs(cooling - (cooling_min + cooling_max) / 2)
                    / ((cooling_min + cooling_max) / 2)
                    * 100
                )
            else:
                c_dev = 0

            if h_dev > max_deviation or c_dev > max_deviation:
                extreme_count += 1

        results.append(
            GateResult(
                name="extreme_deviations",
                category="validation",
                passed=extreme_count <= extreme_limit,
                message=f"{extreme_count} cases exceed {max_deviation}% deviation (limit: {extreme_limit}; {len(known_failures)} known failures excluded: {sorted(known_failures)})",
                value=extreme_count,
                threshold=extreme_limit,
                details={
                    "max_deviation": max_deviation,
                    "known_failures": sorted(known_failures),
                },
            )
        )

        return results

    def check_benchmark_gates(
        self,
        benchmark_results: dict,
        gate_filter: Optional[set[str]] = None,
    ) -> list[GateResult]:
        """Check all benchmark gates.

        If ``gate_filter`` is provided, only benchmark gates whose ``name``
        is in the set are evaluated and returned. This lets lightweight PR
        jobs (issue #2693) evaluate just the throughput + latency absolute
        floors without needing multi-zone / cross-validation measurements
        that the heavy criterion sweep produces. Unset (default) evaluates
        every benchmark gate — preserving the release-time behaviour.
        """
        results = []
        benchmark_config = self.config.get("benchmark", {})
        throughput_config = benchmark_config.get("throughput", {})
        latency_config = benchmark_config.get("latency", {})
        multi_zone_config = benchmark_config.get("multi_zone", {})
        cv_config = benchmark_config.get("cross_validation", {})

        metrics = benchmark_results.get("metrics", {})

        # Throughput check
        throughput = metrics.get("throughput_configs_per_sec", 0.0)
        min_throughput = throughput_config.get("min_configs_per_sec", 800)
        abs_min = benchmark_config.get("absolute_min_throughput", 100)

        # Must pass both minimum and absolute minimum
        throughput_passed = throughput >= min_throughput and throughput >= abs_min

        results.append(
            GateResult(
                name="throughput",
                category="benchmark",
                passed=throughput_passed,
                message=f"Throughput {throughput:.0f} configs/sec vs required {min_throughput} (abs min: {abs_min})",
                value=throughput,
                threshold=min_throughput,
                details={"absolute_min": abs_min},
            )
        )

        # Latency check
        latency = metrics.get("latency_ms_per_config", 0.0)
        max_latency = latency_config.get("max_ms_per_config", 10.0)

        results.append(
            GateResult(
                name="latency",
                category="benchmark",
                passed=latency <= max_latency,
                message=f"Latency {latency:.2f}ms vs max allowed {max_latency}ms",
                value=latency,
                threshold=max_latency,
            )
        )

        # Multi-zone check
        multi_zone = metrics.get("multi_zone_throughput", 0.0)
        min_multi_zone = multi_zone_config.get("min_configs_per_sec", 500)

        results.append(
            GateResult(
                name="multi_zone_throughput",
                category="benchmark",
                passed=multi_zone >= min_multi_zone,
                message=f"Multi-zone throughput {multi_zone:.0f} vs required {min_multi_zone}",
                value=multi_zone,
                threshold=min_multi_zone,
            )
        )

        # Cross-validation latency
        cv_latency = metrics.get("cross_validation_latency_ms", 0.0)
        max_cv_latency = cv_config.get("max_ms", 500)

        results.append(
            GateResult(
                name="cross_validation_latency",
                category="benchmark",
                passed=cv_latency <= max_cv_latency,
                message=f"Cross-validation latency {cv_latency:.0f}ms vs max {max_cv_latency}ms",
                value=cv_latency,
                threshold=max_cv_latency,
            )
        )

        # Regression check (if baseline exists)
        baseline_throughput = benchmark_results.get("baseline_throughput")
        if baseline_throughput and baseline_throughput > 0:
            regression_threshold = benchmark_config.get("regression_threshold", 5.0)
            change_pct = (
                abs(throughput - baseline_throughput) / baseline_throughput * 100
            )
            regression_passed = change_pct <= regression_threshold

            results.append(
                GateResult(
                    name="regression",
                    category="benchmark",
                    passed=regression_passed,
                    message=f"Change {change_pct:.1f}% vs threshold {regression_threshold}% (baseline: {baseline_throughput:.0f})",
                    value=change_pct,
                    threshold=regression_threshold,
                    details={"baseline": baseline_throughput},
                )
            )

        # Issue #2693: restrict to a named subset of benchmark gates so a
        # lightweight PR job can evaluate just the absolute throughput +
        # latency floors without multi-zone / cross-validation data (which
        # default to 0 and would spuriously fail). Unset ⇒ all gates.
        if gate_filter:
            results = [r for r in results if r.name in gate_filter]

        return results

    def check_crate_size_gates(
        self,
        crate_size_results: Optional[dict],
        gate_filter: Optional[set[str]] = None,
    ) -> list[GateResult]:
        """Check crate-size gate (Issue #2930 / Goal #10: <10 MB published).

        Compares the packaged ``fluxion`` crate size against the
        ``crate_size.max_mb`` threshold declared in ``release_gates.yaml``.

        ``crate_size_results`` is expected to be a dict of the form::

            {
                "size_bytes": <int>,
                "size_mb": <float>,
                "crate_path": "<absolute path>",
            }

        If ``None``, a single result is returned with ``passed=False`` and
        a message that no measurement was supplied. The lightweight PR
        job (`.github/workflows/crate-size.yml`) handles the actual
        `cargo package` + measurement; this method is the release-audit
        re-validation that mirrors the same threshold.

        ``gate_filter`` (issue #2930, mirroring the #2693 benchmark-filter
        pattern) lets a caller restrict to a single named gate. Unset
        (default) evaluates every crate-size gate — preserving the
        release-time behaviour.
        """
        results = []
        crate_config = self.config.get("crate_size", {})
        max_mb = float(crate_config.get("max_mb", 10.0))
        max_bytes = int(max_mb * 1024 * 1024)

        if gate_filter and "crate_size" not in gate_filter:
            return results

        if not crate_size_results:
            results.append(
                GateResult(
                    name="crate_size",
                    category="crate_size",
                    passed=False,
                    message=(
                        f"No crate-size measurement supplied — run `cargo package "
                        f"--allow-dirty --no-verify` and re-invoke with "
                        f"`--crate-size-results <path>` (or place the result at "
                        f"`target/package/fluxion-*.crate`). Limit: {max_mb} MiB."
                    ),
                    threshold=max_mb,
                )
            )
            return results

        size_bytes = int(crate_size_results.get("size_bytes", 0))
        size_mb = float(crate_size_results.get("size_mb", size_bytes / (1024 * 1024)))
        crate_path = crate_size_results.get("crate_path", "<unknown>")
        passed = size_bytes <= max_bytes

        if passed:
            headroom_mb = round((max_bytes - size_bytes) / (1024 * 1024), 3)
            message = (
                f"Crate size {size_mb:.3f} MiB ({size_bytes} bytes) is within "
                f"the {max_mb} MiB limit ({crate_path}); headroom {headroom_mb} MiB."
            )
        else:
            over_mb = round((size_bytes - max_bytes) / (1024 * 1024), 3)
            pct = round(size_bytes / max_bytes * 100, 1) if max_bytes else 0.0
            message = (
                f"Crate size {size_mb:.3f} MiB ({size_bytes} bytes) exceeds the "
                f"{max_mb} MiB Goal #10 limit by {over_mb} MiB "
                f"({pct}% of limit) — path: {crate_path}."
            )

        results.append(
            GateResult(
                name="crate_size",
                category="crate_size",
                passed=passed,
                message=message,
                value=size_mb,
                threshold=max_mb,
                details={
                    "size_bytes": size_bytes,
                    "crate_path": crate_path,
                    "limit_bytes": max_bytes,
                },
            )
        )

        return results

    def check_drift_gates(
        self, current_results: dict, baseline: Optional[dict]
    ) -> list[GateResult]:
        """Check drift gates by comparing to baseline."""
        results = []
        drift_config = self.config.get("drift", {})

        if not drift_config.get("enabled", True):
            results.append(
                GateResult(
                    name="drift_detection",
                    category="drift",
                    passed=True,
                    message="Drift detection disabled",
                )
            )
            return results

        if not baseline:
            if drift_config.get("create_baseline_if_missing", False):
                results.append(
                    GateResult(
                        name="baseline",
                        category="drift",
                        passed=True,
                        message="No baseline - would create new baseline",
                    )
                )
            else:
                results.append(
                    GateResult(
                        name="baseline",
                        category="drift",
                        passed=False,
                        message="No baseline found and create_baseline_if_missing is False",
                    )
                )
            return results

        # Pass rate drift
        current_summary = current_results.get("summary", {})
        baseline_summary = baseline.get("summary", {})

        current_pass_rate = current_summary.get("pass_rate", 0.0)
        baseline_pass_rate = baseline_summary.get("pass_rate", 0.0)
        max_pass_rate_change = drift_config.get("max_pass_rate_change", 2.0)

        pass_rate_change = current_pass_rate - baseline_pass_rate
        pass_rate_drift_passed = abs(pass_rate_change) <= max_pass_rate_change

        results.append(
            GateResult(
                name="pass_rate_drift",
                category="drift",
                passed=pass_rate_drift_passed,
                message=f"Pass rate change {pass_rate_change:+.1f}pp vs max {max_pass_rate_change}pp (baseline: {baseline_pass_rate:.1f}%)",
                value=pass_rate_change,
                threshold=max_pass_rate_change,
                details={"baseline": baseline_pass_rate, "current": current_pass_rate},
            )
        )

        # MAE drift
        current_mae = current_summary.get("mae", 0.0)
        baseline_mae = baseline_summary.get("mae", 0.0)
        max_mae_change = drift_config.get("max_mae_change", 5.0)

        mae_change = current_mae - baseline_mae
        mae_drift_passed = abs(mae_change) <= max_mae_change

        results.append(
            GateResult(
                name="mae_drift",
                category="drift",
                passed=mae_drift_passed,
                message=f"MAE change {mae_change:+.2f}pp vs max {max_mae_change}pp",
                value=mae_change,
                threshold=max_mae_change,
                details={"baseline": baseline_mae, "current": current_mae},
            )
        )

        # Pass/fail transitions
        current_cases = current_results.get("cases", {})
        baseline_cases = baseline.get("cases", {})

        max_p2f = drift_config.get("max_pass_to_fail", 1)
        max_f2p = drift_config.get("max_fail_to_pass", 5)

        pass_to_fail = 0
        fail_to_pass = 0

        def case_passed(case):
            h = case.get("heating", 0)
            h_min = case.get("heating_min", 0)
            h_max = case.get("heating_max", 0)
            c = case.get("cooling", 0)
            c_min = case.get("cooling_min", 0)
            c_max = case.get("cooling_max", 0)
            return (h_min <= h <= h_max) and (c_min <= c <= c_max)

        all_cases = set(list(current_cases.keys()) + list(baseline_cases.keys()))
        for case_id in all_cases:
            curr = current_cases.get(case_id, {})
            base = baseline_cases.get(case_id, {})

            if not curr or not base:
                continue

            curr_pass = case_passed(curr)
            base_pass = case_passed(base)

            if base_pass and not curr_pass:
                pass_to_fail += 1
            elif not base_pass and curr_pass:
                fail_to_pass += 1

        results.append(
            GateResult(
                name="pass_to_fail_transitions",
                category="drift",
                passed=pass_to_fail <= max_p2f,
                message=f"{pass_to_fail} cases changed from pass to fail (limit: {max_p2f})",
                value=pass_to_fail,
                threshold=max_p2f,
            )
        )

        results.append(
            GateResult(
                name="fail_to_pass_transitions",
                category="drift",
                passed=fail_to_pass <= max_f2p,
                message=f"{fail_to_pass} cases changed from fail to pass (limit: {max_f2p})",
                value=fail_to_pass,
                threshold=max_f2p,
            )
        )

        return results

    def check_all_gates(
        self,
        validation_results: Optional[dict] = None,
        benchmark_results: Optional[dict] = None,
        update_baseline: bool = False,
        benchmark_gate_filter: Optional[set[str]] = None,
        crate_size_results: Optional[dict] = None,
        crate_size_gate_filter: Optional[set[str]] = None,
    ) -> GateReport:
        """Check all gates and return a comprehensive report.

        ``benchmark_gate_filter`` restricts the benchmark sub-gates evaluated
        (issue #2693); see :meth:`check_benchmark_gates`.

        ``crate_size_results`` and ``crate_size_gate_filter`` (issue #2930)
        thread the same pattern through the new crate-size gate — see
        :meth:`check_crate_size_gates`.

        When ``ci.fail_fast: true`` is set in the YAML config (default), the
        checker stops evaluating further gate categories after the first
        failure. Drift and benchmark gates are skipped if validation gates
        already produced a failure. Issue #2886.
        """
        self.results = []

        # Validation gates
        validation_failed = False
        if validation_results:
            validation_results_list = self.check_validation_gates(validation_results)
            self.results.extend(validation_results_list)
            validation_failed = any(not r.passed for r in validation_results_list)

            # Drift gates (need both current and baseline) — skip on fail_fast
            if self.config.get("drift", {}).get("enabled", True):
                if validation_failed and self.config.get("ci", {}).get("fail_fast", True):
                    sys.stderr.write(
                        "::warning::fail_fast: skipping drift gates after validation failure\n"
                    )
                else:
                    baseline = self._load_baseline()
                    self.results.extend(
                        self.check_drift_gates(validation_results, baseline)
                    )

        # Benchmark gates — skip on fail_fast
        if benchmark_results:
            if validation_failed and self.config.get("ci", {}).get("fail_fast", True):
                sys.stderr.write(
                    "::warning::fail_fast: skipping benchmark gates after validation failure\n"
                )
            else:
                self.results.extend(
                    self.check_benchmark_gates(benchmark_results, benchmark_gate_filter)
                )

        # Crate-size gate (Issue #2930 / Goal #10). Always evaluated when a
        # measurement is supplied — this gate is orthogonal to the
        # validation/benchmark/drift triad (it's a packaging contract, not
        # a runtime property) and must not be skipped under fail_fast.
        if crate_size_results is not None or "crate_size" in self.config:
            self.results.extend(
                self.check_crate_size_gates(crate_size_results, crate_size_gate_filter)
            )

        # Calculate overall
        overall_passed = all(r.passed for r in self.results)

        summary: dict = {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "by_category": {},
        }

        for r in self.results:
            if r.category not in summary["by_category"]:
                summary["by_category"][r.category] = {"passed": 0, "failed": 0}
            if r.passed:
                summary["by_category"][r.category]["passed"] += 1
            else:
                summary["by_category"][r.category]["failed"] += 1

        return GateReport(
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            overall_passed=overall_passed,
            gates=self.results,
            summary=summary,
        )

    def _load_baseline(self) -> Optional[dict]:
        """Load baseline validation results if available."""
        baseline_path = self.project_root / self.config.get("drift", {}).get(
            "baseline_file", "validation_baseline.json"
        )
        if baseline_path.exists():
            with open(baseline_path) as f:
                return json.load(f)
        return None


def load_validation_results(project_root: Path) -> Optional[dict]:
    """Load validation results from standard locations."""
    possible_paths = [
        project_root / "validation_results.json",
        project_root / "ASHRAE_140_VALIDATION_REPORT.md",
    ]

    for path in possible_paths:
        if path.name.endswith(".json") and path.exists():
            with open(path) as f:
                return json.load(f)

    return None


def load_benchmark_results(project_root: Path) -> Optional[dict]:
    """Load benchmark results from standard locations."""
    possible_paths = [
        project_root / "benchmark_results.json",
        project_root / "target" / "criterion" / "benchmark_results.json",
    ]

    for path in possible_paths:
        if path.exists():
            with open(path) as f:
                return json.load(f)

    return None


def load_crate_size_results(
    project_root: Path,
    crate_glob: str = "target/package/fluxion-*.crate",
    explicit_path: Optional[Path] = None,
) -> Optional[dict]:
    """Load packaged-crate size from the standard cargo-package output path.

    Issue #2930 / Goal #10: returns a dict of the form::

        {
            "size_bytes": <int>,
            "size_mb": <float>,
            "crate_path": "<absolute path>",
        }

    Resolves the first matching crate file under ``crate_glob`` (relative to
    ``project_root``) and reports its on-disk size. Returns ``None`` when no
    crate file is present — callers should treat that as "no measurement
    supplied" (the gate will surface a clear failure in that case).

    Cargo writes the package artifact to ``target/package/tmp-crate/`` during
    the manifest-normalization step (before the verify step runs) and only
    promotes it to ``target/package/`` after verify succeeds. When verify
    errors out (e.g. on optional path-only deps with a `version = "X"`
    constraint that isn't on crates.io — see issue #2930) the artifact
    remains in the scratch dir, so we glob both locations to make the gate
    robust to that case.

    ``explicit_path`` overrides the glob lookup (used when the caller passes
    ``--crate-size-results <path>`` from the CLI).
    """
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    else:
        # Primary location (post-verify) and scratch location (pre-verify).
        # The scratch dir is checked first because for in-flight cargo
        # package runs the artifact is more reliably present there.
        scratch_glob = str(Path(crate_glob).parent / "tmp-crate" / Path(crate_glob).name)
        candidates.extend(sorted(project_root.glob(scratch_glob)))
        candidates.extend(sorted(project_root.glob(crate_glob)))

    for path in candidates:
        if path.is_file():
            size_bytes = path.stat().st_size
            return {
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 6),
                "crate_path": str(path.resolve()),
            }

    return None


def generate_markdown_report(report: GateReport) -> str:
    """Generate markdown-formatted gate status report."""
    lines = [
        "# Release Gate Status",
        "",
        f"**Timestamp:** {report.timestamp}",
        f"**Overall Status:** {'✅ PASSED' if report.overall_passed else '❌ FAILED'}",
        "",
        "## Summary",
        "",
        f"- **Total Gates:** {report.summary['total']}",
        f"- **Passed:** {report.summary['passed']}",
        f"- **Failed:** {report.summary['failed']}",
        "",
    ]

    if "by_category" in report.summary:
        lines.append("### By Category")
        lines.append("")
        for category, counts in report.summary["by_category"].items():
            status = "✅" if counts["failed"] == 0 else "❌"
            lines.append(
                f"- **{category.title()}:** {status} {counts['passed']}/{counts['total']}"
            )
        lines.append("")

    lines.append("## Gate Details")
    lines.append("")

    current_category = None
    for result in report.gates:
        if result.category != current_category:
            current_category = result.category
            lines.append(f"### {current_category.title()}")
            lines.append("")

        status = "✅" if result.passed else "❌"
        lines.append(f"{status} **{result.name}**")
        lines.append(f"> {result.message}")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by release_gate_checker.py (Issue #505)*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Check release gates for Fluxion")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--markdown", action="store_true", help="Output as Markdown")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Update baseline with current results",
    )
    parser.add_argument(
        "--validation-results", type=Path, help="Path to validation results JSON"
    )
    parser.add_argument(
        "--benchmark-results", type=Path, help="Path to benchmark results JSON"
    )
    parser.add_argument(
        "--benchmark-gates",
        type=str,
        default=None,
        help=(
            "Comma-separated benchmark gate names to evaluate (issue #2693). "
            "Default: all benchmark gates. Example: 'throughput,latency' to "
            "evaluate only the absolute throughput + latency floors on a PR."
        ),
    )
    parser.add_argument(
        "--crate-size-results",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file with `{size_bytes, size_mb, crate_path}` "
            "(issue #2930 / Goal #10). Defaults to auto-detecting the first "
            "`target/package/fluxion-*.crate` under the project root."
        ),
    )
    parser.add_argument(
        "--crate-size-gates",
        type=str,
        default=None,
        help=(
            "Comma-separated crate-size gate names to evaluate (issue #2930). "
            "Default: all crate-size gates. Currently only `crate_size` is "
            "defined; the flag mirrors `--benchmark-gates` for forward "
            "compatibility."
        ),
    )
    parser.add_argument("--output", "-o", type=Path, help="Output file path")
    args = parser.parse_args()

    # Parse the optional benchmark-gate filter into a set (issue #2693).
    benchmark_gate_filter: Optional[set[str]] = None
    if args.benchmark_gates:
        benchmark_gate_filter = {
            g.strip() for g in args.benchmark_gates.split(",") if g.strip()
        }

    # Parse the optional crate-size-gate filter into a set (issue #2930).
    crate_size_gate_filter: Optional[set[str]] = None
    if args.crate_size_gates:
        crate_size_gate_filter = {
            g.strip() for g in args.crate_size_gates.split(",") if g.strip()
        }

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Load config
    config_path = project_root / "release_gates.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load results
    validation_results = None
    if args.validation_results:
        with open(args.validation_results) as f:
            validation_results = json.load(f)
    else:
        validation_results = load_validation_results(project_root)

    benchmark_results = None
    if args.benchmark_results:
        with open(args.benchmark_results) as f:
            benchmark_results = json.load(f)
    else:
        benchmark_results = load_benchmark_results(project_root)

    # Crate-size results (issue #2930 / Goal #10). Accept either an explicit
    # JSON path or auto-detect `target/package/fluxion-*.crate` (the standard
    # `cargo package` output). Only attempt auto-detection when the YAML
    # declares a `crate_size:` section — otherwise the gate is opt-in and a
    # missing measurement must not silently fail the run.
    crate_size_results: Optional[dict] = None
    if args.crate_size_results is not None:
        if args.crate_size_results.exists():
            with open(args.crate_size_results) as f:
                crate_size_results = json.load(f)
        else:
            sys.stderr.write(
                f"::warning::--crate-size-results path does not exist: {args.crate_size_results}\n"
            )
    elif "crate_size" in config:
        crate_glob = config.get("crate_size", {}).get(
            "crate_glob", "target/package/fluxion-*.crate"
        )
        crate_size_results = load_crate_size_results(project_root, crate_glob)

    # Check gates
    checker = ReleaseGateChecker(config, project_root)
    report = checker.check_all_gates(
        validation_results=validation_results,
        benchmark_results=benchmark_results,
        update_baseline=args.update_baseline,
        benchmark_gate_filter=benchmark_gate_filter,
        crate_size_results=crate_size_results,
        crate_size_gate_filter=crate_size_gate_filter,
    )

    # Update baseline if requested
    if args.update_baseline and validation_results:
        baseline_path = project_root / config.get("drift", {}).get(
            "baseline_file", "validation_baseline.json"
        )
        with open(baseline_path, "w") as f:
            json.dump(validation_results, f, indent=2)
        print(f"Updated baseline at {baseline_path}")

    # Output
    output = None
    if args.json:
        output = json.dumps(
            {
                "timestamp": report.timestamp,
                "overall_passed": report.overall_passed,
                "summary": report.summary,
                "gates": [
                    {
                        "name": r.name,
                        "category": r.category,
                        "passed": r.passed,
                        "message": r.message,
                        "value": r.value,
                        "threshold": r.threshold,
                    }
                    for r in report.gates
                ],
            },
            indent=2,
        )
    elif args.markdown:
        output = generate_markdown_report(report)
    else:
        lines = [
            "=" * 60,
            "RELEASE GATE STATUS",
            "=" * 60,
            f"Timestamp: {report.timestamp}",
            f"Overall: {'✅ PASSED' if report.overall_passed else '❌ FAILED'}",
            "",
            f"Total: {report.summary['total']} | Passed: {report.summary['passed']} | Failed: {report.summary['failed']}",
            "",
            "-" * 60,
        ]

        for result in report.gates:
            status = "✅" if result.passed else "❌"
            lines.append(
                f"{status} [{result.category.upper()}] {result.name}: {result.message}"
            )

        lines.append("-" * 60)
        output = "\n".join(lines)

    # Write or print
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Gate status written to {args.output}")
    else:
        print(output)

    # Exit code
    sys.exit(0 if report.overall_passed else 1)


if __name__ == "__main__":
    main()
