"""
ASHRAE 140 Test Harness.

This module provides a comprehensive test runner for ASHRAE 140 validation,
including:
- Input validation test execution
- Output comparison with EnergyPlus
- Diagnostic test execution
- Results aggregation and reporting

Usage:
    python -m tools.ashrae_140_test_harness --help
    python -m tools.ashrae_140_test_harness --input-validation
    python -m tools.ashrae_140_test_harness --output-comparison
    python -m tools.ashrae_140_test_harness --diagnostics
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TestResult:
    """Result of a single test case."""

    test_name: str
    passed: bool
    message: str = ""
    error: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    duration_sec: float = 0.0


@dataclass
class TestSuiteResult:
    """Result of a test suite execution."""

    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration_sec: float
    results: List[TestResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "suite_name": self.suite_name,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "duration_sec": self.duration_sec,
            "pass_rate": self.pass_rate,
            "results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "message": r.message,
                    "error": r.error,
                    "metrics": r.metrics,
                    "duration_sec": r.duration_sec,
                }
                for r in self.results
            ],
        }


class ASHRAE140TestHarness:
    """Test harness for ASHRAE 140 validation."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize test harness.

        Args:
            project_root: Root directory of the project. Defaults to current directory.
        """
        self.project_root = project_root or Path.cwd()
        self.test_results: List[TestSuiteResult] = []

    def run_rust_tests(
        self, test_pattern: str, verbose: bool = False
    ) -> TestSuiteResult:
        """Run Rust tests matching a pattern.

        Args:
            test_pattern: Cargo test pattern (e.g., "ashrae_140::geometry")
            verbose: Whether to show detailed output

        Returns:
            TestSuiteResult with test execution results
        """
        print(f"\n{'=' * 60}")
        print(f"Running Rust tests: {test_pattern}")
        print(f"{'=' * 60}")

        start_time = datetime.now()

        # Build cargo test command
        cmd = ["cargo", "test", test_pattern, "--", "--nocapture"]

        if not verbose:
            cmd.append("--quiet")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            duration = (datetime.now() - start_time).total_seconds()

            # Parse test output
            total, passed, failed, skipped = self._parse_cTest_output(result.stdout)

            suite_result = TestSuiteResult(
                suite_name=test_pattern,
                total_tests=total,
                passed=passed,
                failed=failed,
                skipped=skipped,
                duration_sec=duration,
            )

            # Add individual test results
            for line in result.stdout.split("\n"):
                if "test ... " in line:
                    parts = line.split("test ... ")
                    if len(parts) == 2:
                        test_name, status = parts[0].strip(), parts[1].strip()
                        test_result = TestResult(
                            test_name=test_name,
                            passed="ok" in status,
                            message=status,
                        )
                        suite_result.results.append(test_result)

            self.test_results.append(suite_result)
            return suite_result

        except subprocess.TimeoutExpired:
            return TestSuiteResult(
                suite_name=test_pattern,
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                duration_sec=(datetime.now() - start_time).total_seconds(),
                results=[
                    TestResult(
                        test_name=test_pattern,
                        passed=False,
                        error="Test timed out after 10 minutes",
                    )
                ],
            )
        except Exception as e:
            return TestSuiteResult(
                suite_name=test_pattern,
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                duration_sec=0.0,
                results=[
                    TestResult(
                        test_name=test_pattern,
                        passed=False,
                        error=str(e),
                    )
                ],
            )

    def _parse_cargo_test_output(self, output: str) -> Tuple[int, int, int, int]:
        """Parse cargo test output to extract counts.

        Returns:
            Tuple of (total, passed, failed, skipped)
        """
        total = passed = failed = skipped = 0

        for line in output.split("\n"):
            if "test result:" in line:
                # Parse: test result: ok. 15 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out
                parts = line.split(";")
                for part in parts:
                    if "passed" in part:
                        passed = int(part.split()[0])
                    elif "failed" in part:
                        failed = int(part.split()[0])
                    elif "ignored" in part:
                        skipped = int(part.split()[0])

                total = passed + failed + skipped
                break

        return total, passed, failed, skipped

    def run_input_validation(self, verbose: bool = False) -> TestSuiteResult:
        """Run all input validation tests."""
        return self.run_rust_tests("ashrae_140_input_validation", verbose)

    def run_output_comparison(self, verbose: bool = False) -> TestSuiteResult:
        """Run all output comparison tests."""
        return self.run_rust_tests("ashrae_140_output_validation", verbose)

    def run_diagnostics(self, verbose: bool = False) -> TestSuiteResult:
        """Run all diagnostic tests."""
        return self.run_rust_tests("ashrae_140_diagnostics", verbose)

    def run_all_tests(self, verbose: bool = False) -> List[TestSuiteResult]:
        """Run all ASHRAE 140 validation tests."""
        results = []

        results.append(self.run_input_validation(verbose))
        results.append(self.run_output_comparison(verbose))
        results.append(self.run_diagnostics(verbose))

        return results

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate a test report.

        Args:
            output_path: Optional path to save JSON report

        Returns:
            JSON report string
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "suites": [result.to_dict() for result in self.test_results],
            "summary": {
                "total_suites": len(self.test_results),
                "total_tests": sum(r.total_tests for r in self.test_results),
                "total_passed": sum(r.passed for r in self.test_results),
                "total_failed": sum(r.failed for r in self.test_results),
                "total_skipped": sum(r.skipped for r in self.test_results),
                "total_duration_sec": sum(r.duration_sec for r in self.test_results),
            },
        }

        report_json = json.dumps(report, indent=2)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_json)
            print(f"\nReport saved to: {output_path}")

        return report_json

    def print_summary(self):
        """Print test summary to console."""
        print("\n" + "=" * 60)
        print("ASHRAE 140 VALIDATION TEST SUMMARY")
        print("=" * 60)

        for suite in self.test_results:
            status = "✅ PASS" if suite.failed == 0 else "❌ FAIL"
            print(f"\n{suite.suite_name}: {status}")
            print(
                f"  Tests: {suite.total_tests} | "
                f"Passed: {suite.passed} | "
                f"Failed: {suite.failed} | "
                f"Skipped: {suite.skipped}"
            )
            print(
                f"  Pass Rate: {suite.pass_rate:.1f}% | "
                f"Duration: {suite.duration_sec:.1f}s"
            )

        # Overall summary
        total_tests = sum(r.total_tests for r in self.test_results)
        total_passed = sum(r.passed for r in self.test_results)
        sum(r.failed for r in self.test_results)
        overall_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        print("\n" + "-" * 60)
        print(
            f"OVERALL: {total_passed}/{total_tests} tests passed ({overall_rate:.1f}%)"
        )
        print("=" * 60)


def main():
    """Main entry point for test harness."""
    parser = argparse.ArgumentParser(description="ASHRAE 140 Validation Test Harness")
    parser.add_argument(
        "--input-validation",
        action="store_true",
        help="Run input validation tests",
    )
    parser.add_argument(
        "--output-comparison",
        action="store_true",
        help="Run output comparison tests",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run diagnostic tests",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed test output",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Save JSON report to specified path",
    )

    args = parser.parse_args()

    # Default to running all tests if no specific option selected
    if not any(
        [args.input_validation, args.output_comparison, args.diagnostics, args.all]
    ):
        args.all = True

    harness = ASHRAE140TestHarness()

    if args.input_validation:
        harness.run_input_validation(args.verbose)

    if args.output_comparison:
        harness.run_output_comparison(args.verbose)

    if args.diagnostics:
        harness.run_diagnostics(args.verbose)

    if args.all:
        harness.run_all_tests(args.verbose)

    harness.print_summary()

    if args.report:
        harness.generate_report(args.report)

    # Exit with error code if any tests failed
    total_failed = sum(r.failed for r in harness.test_results)
    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    main()
