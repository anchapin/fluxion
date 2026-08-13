#!/usr/bin/env python3
"""
performance_gate.py — flags performance regressions >10% on PRs

Runs benchmarks and compares against main branch baseline.
Fails if any benchmark degrades by more than 10%.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

BENCHMARK_THRESHOLD = 0.10  # 10% degradation allowed
BASELINE_FILE = Path(__file__).parent / ".perf_baseline.json"
PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd: List[str], timeout: int = 300) -> Tuple[str, int]:
    """Run command and return (stdout, returncode)."""
    try:
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", 124


def get_git_branch() -> str:
    """Get current git branch name."""
    stdout, _ = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return stdout.strip()


def get_main_branch_baseline() -> Dict[str, float]:
    """Get baseline from main branch."""
    main_baseline = {}
    if not BASELINE_FILE.exists():
        return main_baseline

    stdout, rc = run_command(["git", "stash"])
    if rc == 0:
        run_command(["git", "checkout", "main"])

    import json

    try:
        with open(BASELINE_FILE) as f:
            main_baseline = json.load(f)
    finally:
        run_command(["git", "checkout", "-"])
        run_command(["git", "stash", "pop"])

    return main_baseline


def run_benchmarks() -> Dict[str, float]:
    """Run cargo bench and extract results."""
    benchmarks: Dict[str, float] = {}

    stdout, rc = run_command(["cargo", "bench", "--", "--json", "--format", "json"])
    if rc != 0:
        print(f"Benchmark command exited with {rc}")
        print(stdout[-2000:] if len(stdout) > 2000 else stdout)
        return benchmarks

    for line in stdout.splitlines():
        try:
            data = json.loads(line)
            if data.get("type") == "benchmark-complete":
                name = data.get("name", "unknown")
                ns_per_iter = data.get(" median_time_ns", data.get("mean_time_ns", 0))
                if ns_per_iter:
                    benchmarks[name] = ns_per_iter / 1e9  # Convert to seconds
        except json.JSONDecodeError:
            continue

    if not benchmarks:
        for line in stdout.splitlines():
            m = re.search(
                r"(?P<name>[\w_]+)\s+time:\s+(?P<time>[\d.]+)\s*(?P<unit>\w+)", line
            )
            if m:
                name = m.group("name")
                time_val = float(m.group("time"))
                unit = m.group("unit")
                if unit == "ms":
                    benchmarks[name] = time_val / 1000
                elif unit == "us":
                    benchmarks[name] = time_val / 1e6
                elif unit == "ns":
                    benchmarks[name] = time_val / 1e9
                else:
                    benchmarks[name] = time_val

    return benchmarks


def check_perf_regression(
    current: Dict[str, float], baseline: Dict[str, float]
) -> List[str]:
    """Check for regressions > threshold."""
    regressions = []
    for name, current_time in current.items():
        if name not in baseline:
            continue
        baseline_time = baseline[name]
        if baseline_time <= 0:
            continue
        pct_change = (current_time - baseline_time) / baseline_time
        if pct_change > BENCHMARK_THRESHOLD:
            regressions.append(
                f"  {name}: {baseline_time:.4f}s -> {current_time:.4f}s "
                f"(+{pct_change * 100:.1f}%)"
            )
    return regressions


def save_baseline(benchmarks: Dict[str, float]):
    """Save current benchmarks as new baseline."""
    with open(BASELINE_FILE, "w") as f:
        json.dump(benchmarks, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Fluxion Performance Gate")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run benchmark check against baseline and exit with appropriate code",
    )
    args = parser.parse_args()

    print("=== Performance Gate ===")
    print(f"Threshold: {BENCHMARK_THRESHOLD * 100}% max regression")

    branch = get_git_branch()
    is_main = branch == "main"

    if is_main and not args.check:
        print("Running on main branch — saving baseline")
        benchmarks = run_benchmarks()
        if benchmarks:
            save_baseline(benchmarks)
            print(f"Saved {len(benchmarks)} benchmark results to {BASELINE_FILE}")
        else:
            print("No benchmarks found")
            sys.exit(1)
        sys.exit(0)

    if args.check:
        print("Running in check mode — comparing against baseline")
    else:
        print(f"Running on branch '{branch}' — checking against baseline")

    baseline = get_main_branch_baseline()
    if not baseline:
        print("No baseline found — run on main first to establish baseline")
        sys.exit(0)

    current = run_benchmarks()
    if not current:
        print("No benchmarks produced")
        sys.exit(1)

    regressions = check_perf_regression(current, baseline)
    if regressions:
        print("\nPERFORMANCE REGRESSIONS DETECTED:")
        for r in regressions:
            print(r)
        print(
            f"\nFailed: {len(regressions)} benchmark(s) exceed {BENCHMARK_THRESHOLD * 100}% threshold"
        )
        sys.exit(1)

    print(f"All {len(current)} benchmarks within threshold")
    sys.exit(0)


if __name__ == "__main__":
    main()
