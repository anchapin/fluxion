#!/usr/bin/env python3
"""
Fluxion Test False Confidence Audit

Audits tests for false confidence issues:
- No-op tests (empty or trivially passing)
- Always-pass tests (missing assertions)
- Wrong tolerance tests (tolerance too loose)
- Mock overkill tests (over-mocked)
- Wrong assertion tests (assertion doesn't match intent)
- Flaky pass tests (race conditions, timing dependencies)

Usage:
    python scripts/audit_false_confidence.py [path]
    python scripts/audit_false_confidence.py src/physics/
    python scripts/audit_false_confidence.py --fix  # Auto-fix some issues

Exit codes:
    0 - All tests pass confidence audit
    1 - Issues found (see report)
    2 - Error during analysis
"""

import argparse
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class IssueSeverity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class IssueType(Enum):
    NO_OP_TEST = "no_op_test"
    ALWAYS_PASS = "always_pass"
    WRONG_TOLERANCE = "wrong_tolerance"
    MOCK_OVERKILL = "mock_overkill"
    WRONG_ASSERTION = "wrong_assertion"
    FLAKY_PASS = "flaky_pass"
    MISSING_ASSERT = "missing_assert"


@dataclass
class Issue:
    severity: IssueSeverity
    issue_type: IssueType
    file: str
    line: int
    function: str
    message: str
    suggestion: str = ""


@dataclass
class TestInfo:
    name: str
    file: str
    line: int
    end_line: int
    body: list[str]
    has_assert: bool
    assertions: list[str]
    mocks: list[str]
    docstring: str = ""


class FalseConfidenceAuditor:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.issues: list[Issue] = []
        self.tests: list[TestInfo] = []

    def find_test_files(self, path: Optional[str] = None) -> list[Path]:
        """Find all Rust test files in the given path."""
        if path:
            search_path = self.repo_root / path
        else:
            search_path = self.repo_root / "src"

        test_files = []
        for pattern in ["**/*.rs"]:
            test_files.extend(search_path.glob(pattern))
        return [f for f in test_files if f.is_file()]

    def extract_tests(self, content: str, file_path: Path) -> list[TestInfo]:
        """Extract test information from Rust source."""
        tests = []
        lines = content.split("\n")

        # Find #[test] and #[tokio::test] annotated functions
        test_pattern = re.compile(r"#\[(test|tokio::test|test_strategy|proptest)\]")

        i = 0
        while i < len(lines):
            line = lines[i]
            if test_pattern.search(line):
                # Find function name
                func_match = re.search(r"fn\s+(\w+)", lines[i + 1] if i + 1 < len(lines) else "")
                if func_match:
                    func_name = func_match.group(1)
                    start_line = i + 1

                    # Find function end (matching brace)
                    depth = 0
                    end_line = i + 1
                    in_func = False
                    for j in range(i + 1, len(lines)):
                        if "{" in lines[j]:
                            depth += lines[j].count("{")
                            in_func = True
                        if "}" in lines[j]:
                            depth -= lines[j].count("}")
                        if in_func and depth == 0:
                            end_line = j
                            break

                    # Extract body
                    body_lines = lines[start_line:end_line + 1]
                    body = "\n".join(body_lines)

                    # Find assertions
                    assertions = re.findall(r"assert[!?]?_?(?:eq|ne|ion|true|false)?!\s*\([^)]+\)", body)

                    # Find mocks
                    mocks = re.findall(r"(Mock\w+|mock\(\)|when\(|\.verify\(\))", body)

                    tests.append(TestInfo(
                        name=func_name,
                        file=str(file_path),
                        line=start_line + 1,
                        end_line=end_line + 1,
                        body=body_lines,
                        has_assert=len(assertions) > 0,
                        assertions=assertions,
                        mocks=mocks,
                    ))
            i += 1

        return tests

    def check_no_op_test(self, test: TestInfo) -> Optional[Issue]:
        """Check for no-op tests (empty or trivially passing)."""
        # Remove comments, strings, and whitespace
        code_lines = []
        in_string = False
        for line in test.body:
            stripped = ""
            for char in line:
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string and not char.startswith("//"):
                    stripped += char
            code_lines.append(stripped)

        code = "".join(code_lines)

        # Check if test body is essentially empty
        significant_code = re.sub(r"\s+", "", code)

        # Empty test or only has docstring
        if len(significant_code) < 10:
            return Issue(
                severity=IssueSeverity.CRITICAL,
                issue_type=IssueType.NO_OP_TEST,
                file=test.file,
                line=test.line,
                function=test.name,
                message=f"Test '{test.name}' has no body or is trivially empty",
                suggestion="Remove this test or add meaningful assertions",
            )

        # Check for body that just returns Ok
        if re.match(r"^\s*fn\s+\w+\s*\([^)]*\)\s*->\s*\w+\s*\{\s*\}?$", code):
            return Issue(
                severity=IssueSeverity.CRITICAL,
                issue_type=IssueType.NO_OP_TEST,
                file=test.file,
                line=test.line,
                function=test.name,
                message=f"Test '{test.name}' body is empty or just returns",
                suggestion="Add assertions to verify expected behavior",
            )

        return None

    def check_always_pass(self, test: TestInfo) -> Optional[Issue]:
        """Check for tests that always pass."""
        if not test.has_assert:
            return Issue(
                severity=IssueSeverity.HIGH,
                issue_type=IssueType.ALWAYS_PASS,
                file=test.file,
                line=test.line,
                function=test.name,
                message=f"Test '{test.name}' has no assertions",
                suggestion="Add assertions to verify expected behavior",
            )

        # Check for assertion that always passes
        for assertion in test.assertions:
            # assert!(true) or assert_eq!(x, x)
            if re.search(r"assert!\s*\(\s*(true|1|Ok|Some)", assertion):
                return Issue(
                    severity=IssueSeverity.HIGH,
                    issue_type=IssueType.ALWAYS_PASS,
                    file=test.file,
                    line=test.line,
                    function=test.name,
                    message=f"Test '{test.name}' contains an assertion that always passes",
                    suggestion="Replace with a meaningful assertion",
                )

        return None

    def check_wrong_tolerance(self, test: TestInfo) -> Optional[Issue]:
        """Check for tests with wrong tolerance."""
        # Look for tolerance values
        tolerance_pattern = re.compile(r"(epsilon|rtol|atol|tolerance|1e-\d+)")

        for assertion in test.assertions:
            if "approx" in assertion or "relative" in assertion.lower():
                # Check for very loose tolerances
                loose_tolerance = re.search(r"1e-([1-5])", assertion)
                if loose_tolerance:
                    exp = int(loose_tolerance.group(1))
                    if exp <= 5:
                        return Issue(
                            severity=IssueSeverity.MEDIUM,
                            issue_type=IssueType.WRONG_TOLERANCE,
                            file=test.file,
                            line=test.line,
                            function=test.name,
                            message=f"Test '{test.name}' uses loose tolerance 1e-{exp} (should be 1e-6 or tighter for physics)",
                            suggestion="Use tolerance of 1e-6 or tighter for energy balance validation",
                        )

        return None

    def check_mock_overkill(self, test: TestInfo) -> Optional[Issue]:
        """Check for tests that over-mock (mocking too much)."""
        # Count ratio of mock code to assertion code
        mock_lines = sum(1 for line in test.body if "mock" in line.lower() or "when" in line.lower())
        assertion_lines = sum(1 for line in test.body if "assert" in line.lower())

        if mock_lines > 0 and assertion_lines == 0:
            return Issue(
                severity=IssueSeverity.MEDIUM,
                issue_type=IssueType.MOCK_OVERKILL,
                file=test.file,
                line=test.line,
                function=test.name,
                message=f"Test '{test.name}' has {mock_lines} mock lines but no assertions",
                suggestion="Add assertions to verify the mock was called correctly",
            )

        if mock_lines > 20 and assertion_lines < 3:
            return Issue(
                severity=IssueSeverity.LOW,
                issue_type=IssueType.MOCK_OVERKILL,
                file=test.file,
                line=test.line,
                function=test.name,
                message=f"Test '{test.name}' has heavy mocking ({mock_lines} lines) with few assertions ({assertion_lines})",
                suggestion="Consider using integration tests with real components instead of extensive mocking",
            )

        return None

    def check_wrong_assertion(self, test: TestInfo) -> Optional[Issue]:
        """Check for tests with wrong assertions."""
        for assertion in test.assertions:
            # assert_eq!(a, b) where a and b are the same expression
            match = re.search(r"assert_eq!\s*\(\s*([^,]+),\s*\1\s*\)", assertion)
            if match:
                return Issue(
                    severity=IssueSeverity.HIGH,
                    issue_type=IssueType.WRONG_ASSERTION,
                    file=test.file,
                    line=test.line,
                    function=test.name,
                    message=f"Test '{test.name}' uses assert_eq! on identical expressions",
                    suggestion="assert_eq! should compare different values",
                )

            # assert!(a == b) where a and b are the same
            match = re.search(r"assert!\s*\(\s*([^=]+)\s*==\s*\1\s*\)", assertion)
            if match:
                return Issue(
                    severity=IssueSeverity.HIGH,
                    issue_type=IssueType.WRONG_ASSERTION,
                    file=test.file,
                    line=test.line,
                    function=test.name,
                    message=f"Test '{test.name}' asserts identical values are equal",
                    suggestion="Compare different values or use assert! with a meaningful condition",
                )

        return None

    def check_flaky_pass(self, test: TestInfo) -> Optional[Issue]:
        """Check for potentially flaky tests."""
        body = "\n".join(test.body)

        # Check for timing-dependent code
        if "sleep" in body or "timeout" in body or "delay" in body:
            return Issue(
                severity=IssueSeverity.MEDIUM,
                issue_type=IssueType.FLAKY_PASS,
                file=test.file,
                line=test.line,
                function=test.name,
                message=f"Test '{test.name}' contains timing-dependent code (sleep/timeout/delay)",
                suggestion="Use event-driven synchronization instead of timing delays",
            )

        # Check for thread::sleep without proper synchronization
        if "thread::sleep" in body and "std::time::Duration" in body:
            return Issue(
                severity=IssueSeverity.MEDIUM,
                issue_type=IssueType.FLAKY_PASS,
                file=test.file,
                line=test.line,
                function=test.name,
                message=f"Test '{test.name}' uses thread::sleep which can cause flakiness",
                suggestion="Use proper synchronization primitives (channels, barriers, etc.)",
            )

        # Check for random number generation without seed
        if "rand::" in body and "seed_from_u64" not in body and "Rng::seed_from_u64" not in body:
            if "SmallRng" not in body or "from_entropy" in body:
                return Issue(
                    severity=IssueSeverity.LOW,
                    issue_type=IssueType.FLAKY_PASS,
                    file=test.file,
                    line=test.line,
                    function=test.name,
                    message=f"Test '{test.name}' uses random numbers without a fixed seed",
                    suggestion="Use SmallRng::seed_from_u64(42) for deterministic tests",
                )

        return None

    def audit_test(self, test: TestInfo):
        """Run all checks on a single test."""
        checks = [
            self.check_no_op_test,
            self.check_always_pass,
            self.check_wrong_tolerance,
            self.check_mock_overkill,
            self.check_wrong_assertion,
            self.check_flaky_pass,
        ]

        for check in checks:
            issue = check(test)
            if issue:
                self.issues.append(issue)

    def run_dynamic_analysis(self, test: TestInfo) -> Optional[Issue]:
        """Run dynamic analysis by injecting synthetic bugs."""
        # This is a simplified version - full dynamic analysis would require
        # actually running the tests with injected faults

        # For now, we check for patterns that would be fragile under dynamic analysis
        body = "\n".join(test.body)

        # Check for tests that don't use result unwrapping properly
        if ".unwrap()" in body and "assert" not in body.lower():
            return Issue(
                severity=IssueSeverity.LOW,
                issue_type=IssueType.MISSING_ASSERT,
                file=test.file,
                line=test.line,
                function=test.name,
                message=f"Test '{test.name}' uses .unwrap() without assertions - will panic instead of failing gracefully",
                suggestion="Use ? operator or map_err and provide meaningful assertions",
            )

        return None

    def audit(self, path: Optional[str] = None) -> list[Issue]:
        """Run the full audit on all test files."""
        test_files = self.find_test_files(path)

        print(f"Found {len(test_files)} Rust files to analyze")

        for file_path in test_files:
            try:
                content = file_path.read_text()
                tests = self.extract_tests(content, file_path)
                self.tests.extend(tests)

                for test in tests:
                    self.audit_test(test)
                    dynamic_issue = self.run_dynamic_analysis(test)
                    if dynamic_issue:
                        self.issues.append(dynamic_issue)

            except Exception as e:
                print(f"Error analyzing {file_path}: {e}", file=sys.stderr)

        return self.issues

    def generate_report(self) -> str:
        """Generate a markdown report of findings."""
        lines = [
            "# Fluxion False Confidence Audit Report",
            "",
            f"**Generated:** {Path().cwd()}",
            f"**Total tests analyzed:** {len(self.tests)}",
            f"**Issues found:** {len(self.issues)}",
            "",
        ]

        # Group by severity
        by_severity = {}
        for issue in self.issues:
            if issue.severity not in by_severity:
                by_severity[issue.severity] = []
            by_severity[issue.severity].append(issue)

        for severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH, IssueSeverity.MEDIUM, IssueSeverity.LOW]:
            if severity not in by_severity:
                continue

            lines.append(f"## {severity.value}")
            lines.append("")

            for issue in by_severity[severity]:
                lines.append(f"### `{issue.function}` at {issue.file}:{issue.line}")
                lines.append("")
                lines.append(f"**Type:** `{issue.issue_type.value}`")
                lines.append("")
                lines.append(f"**Message:** {issue.message}")
                lines.append("")
                if issue.suggestion:
                    lines.append(f"**Suggestion:** {issue.suggestion}")
                    lines.append("")
                lines.append("---")
                lines.append("")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit tests for false confidence issues")
    parser.add_argument("path", nargs="?", help="Path to analyze (default: src/)")
    parser.add_argument("--fix", action="store_true", help="Auto-fix some issues")
    parser.add_argument("--output", "-o", help="Output file for report")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    auditor = FalseConfidenceAuditor(repo_root)

    print(f"Analyzing tests in: {args.path or 'src/'}")
    issues = auditor.audit(args.path)

    report = auditor.generate_report()

    if args.output:
        Path(args.output).write_text(report)
        print(f"Report saved to: {args.output}")
    else:
        print()
        print(report)

    # Summary
    print()
    print("=" * 60)
    print(f"Summary: {len(issues)} issues found in {len(auditor.tests)} tests")
    print("=" * 60)

    by_type = {}
    for issue in issues:
        if issue.issue_type not in by_type:
            by_type[issue.issue_type] = 0
        by_type[issue.issue_type] += 1

    print("\nIssues by type:")
    for itype, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {itype.value}: {count}")

    if issues:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
