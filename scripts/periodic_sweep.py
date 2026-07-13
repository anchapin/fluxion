#!/usr/bin/env python3
"""
periodic_sweep.py — monthly agent sweep through recent commits

Scans recent commits looking for:
- Regressions in numerical accuracy across physics modules
- Architectural drift (fluxion-core cycle violations)
- Performance degradation
- New patterns that violate coding conventions

Intended to run monthly via cron or CI scheduled job.
"""

import subprocess
import sys
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent

DAYS_TO_SCAN = 30
REPORT_FILE = Path(__file__).parent / ".sweep_report.md"


def run_command(cmd: List[str], timeout: int = 120) -> Tuple[str, int]:
    """Run command and return (stdout, returncode)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", 124


def get_recent_commits(days: int = 30) -> List[Dict]:
    """Get commits from last N days."""
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    stdout, _ = run_command([
        "git", "log", f"--since={cutoff_str}", "--format=%H|%s|%an|%ad",
        "--date=iso", "--no-merges"
    ])

    commits = []
    for line in stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split("|", 3)
        if len(parts) >= 4:
            commits.append({
                "hash": parts[0],
                "subject": parts[1],
                "author": parts[2],
                "date": parts[3],
            })
    return commits


def check_numerical_accuracy(commit_hash: str) -> List[str]:
    """Check for numerical accuracy regressions in physics modules."""
    issues = []

    stdout, _ = run_command([
        "git", "show", commit_hash, "--stat", "--name-only"
    ])

    physics_files = [
        "solar.rs", "ventilation.rs", "solver_trait.rs",
        "thermal_model.rs", "weather.rs", "epw.rs"
    ]

    changed_phys_files = []
    for line in stdout.splitlines():
        for pf in physics_files:
            if pf in line:
                changed_phys_files.append(pf)

    if changed_phys_files:
        run_command(["git", "checkout", commit_hash, "--"] + changed_phys_files)
        test_result, _ = run_command(["cargo", "test", "--lib", "--", "-q"])
        if "test result: FAILED" in test_result or "FAILED" in test_result:
            issues.append(f"  [{commit_hash[:7]}] Numerical test failures in: {', '.join(changed_phys_files)}")
        run_command(["git", "checkout", "HEAD", "--"] + changed_phys_files)

    return issues


def check_architecture_drift() -> List[str]:
    """Check for architectural drift (fluxion-core cycle violations)."""
    issues = []

    arch_check = Path(PROJECT_ROOT) / "scripts" / "check_architecture_drift.py"
    if arch_check.exists():
        output, rc = run_command([sys.executable, str(arch_check)])
        if rc != 0:
            issues.append(f"  Architecture drift detected:\n{output}")
    else:
        dep_check = Path(PROJECT_ROOT) / "ARCHITECTURE.md"
        if dep_check.exists():
            issues.append("  Architecture doc exists but check_architecture_drift.py missing")

    return issues


def check_performance_degradation() -> List[str]:
    """Check for performance degradation via benchmark comparison."""
    issues = []

    perf_gate = Path(PROJECT_ROOT) / "scripts" / "performance_gate.py"
    if perf_gate.exists():
        output, rc = run_command([sys.executable, str(perf_gate)])
        if rc != 0:
            issues.append(f"  Performance regression detected:\n{output[-500:]}")
    else:
        issues.append("  performance_gate.py not found — cannot check performance")

    return issues


def check_coding_conventions() -> List[str]:
    """Check for new patterns that violate coding conventions."""
    issues = []

    stdout, rc = run_command(["cargo", "clippy", "--", "-D", "warnings"])
    if rc != 0:
        issues.append(f"  Clippy warnings found:\n{stdout[-1000:]}")

    fmt_result, _ = run_command(["cargo", "fmt", "--check"])
    if fmt_result:
        issues.append("  Formatting violations found")

    return issues


def check_ashrae_regressions() -> List[str]:
    """Check ASHRAE case regressions."""
    issues = []

    ashrae_check = Path(PROJECT_ROOT) / "scripts" / "check_ashrae_cases_cycle.py"
    if ashrae_check.exists():
        output, rc = run_command([sys.executable, str(ashrae_check)])
        if rc != 0:
            issues.append(f"  ASHRAE case regression:\n{output[-500:]}")
    else:
        issues.append("  check_ashrae_cases_cycle.py not found")

    return issues


def generate_report(
    commits: List[Dict],
    numerical_issues: List[str],
    arch_issues: List[str],
    perf_issues: List[str],
    convention_issues: List[str],
    ashrae_issues: List[str]
) -> str:
    """Generate markdown report."""
    total_issues = (
        len(numerical_issues) + len(arch_issues) +
        len(perf_issues) + len(convention_issues) +
        len(ashrae_issues)
    )

    report = f"""# Periodic Sweep Report

**Generated:** {datetime.now().isoformat()}
**Scanned commits:** {len(commits)} (last {DAYS_TO_SCAN} days)
**Total issues found:** {total_issues}

## Commit Summary

| Hash | Subject | Author | Date |
|------|---------|--------|------|
"""

    for c in commits[:20]:
        report += f"| `{c['hash'][:7]}` | {c['subject']} | {c['author']} | {c['date'][:10]} |\n"

    if numerical_issues:
        report += f"\n## Numerical Accuracy Issues ({len(numerical_issues)})\n\n"
        report += "\n".join(numerical_issues) + "\n"

    if arch_issues:
        report += f"\n## Architectural Drift ({len(arch_issues)})\n\n"
        report += "\n".join(arch_issues) + "\n"

    if perf_issues:
        report += f"\n## Performance Degradation ({len(perf_issues)})\n\n"
        report += "\n".join(perf_issues) + "\n"

    if convention_issues:
        report += f"\n## Coding Convention Violations ({len(convention_issues)})\n\n"
        report += "\n".join(convention_issues) + "\n"

    if ashrae_issues:
        report += f"\n## ASHRAE Case Regressions ({len(ashrae_issues)})\n\n"
        report += "\n".join(ashrae_issues) + "\n"

    if total_issues == 0:
        report += "\n## Status: PASSED\n\nNo issues found in this sweep.\n"
    else:
        report += f"\n## Status: NEEDS ATTENTION\n\n{total_issues} issue(s) require review.\n"

    return report


def main():
    print("=== Periodic Sweep ===")
    print(f"Scanning last {DAYS_TO_SCAN} days of commits...")

    commits = get_recent_commits(DAYS_TO_SCAN)
    print(f"Found {len(commits)} commits")

    print("Checking architecture drift...")
    arch_issues = check_architecture_drift()

    print("Checking performance degradation...")
    perf_issues = check_performance_degradation()

    print("Checking coding conventions...")
    convention_issues = check_coding_conventions()

    print("Checking ASHRAE regressions...")
    ashrae_issues = check_ashrae_regressions()

    numerical_issues = []
    if commits:
        print("Checking numerical accuracy...")
        for c in commits[:10]:
            issues = check_numerical_accuracy(c["hash"])
            numerical_issues.extend(issues)

    report = generate_report(
        commits, numerical_issues, arch_issues,
        perf_issues, convention_issues, ashrae_issues
    )

    with open(REPORT_FILE, "w") as f:
        f.write(report)

    print(f"\nReport written to: {REPORT_FILE}")
    print(report)

    total_issues = (
        len(numerical_issues) + len(arch_issues) +
        len(perf_issues) + len(convention_issues) +
        len(ashrae_issues)
    )
    sys.exit(1 if total_issues > 0 else 0)


if __name__ == "__main__":
    main()
