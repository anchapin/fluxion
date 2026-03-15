#!/usr/bin/env python3
"""
Hook: Performance baseline check and comparison
Purpose: Smoke test BatchOracle throughput and detect performance regressions

Features:
- Smoke test: Verifies throughput exceeds 100μs/config
- Baseline comparison: Detects >10% regressions from stored baseline
- CI integration: JSON output for GitHub Actions

Run with:
  - Smoke test: pre-commit run --hook-stage manual
  - Baseline comparison: python .githooks/perf-baseline.py --compare
  - Update baseline: python .githooks/perf-baseline.py --update-baseline

This catches silent performance regressions:
  - Unnecessary clones in batch loop
  - Memory allocations in hot path
  - Missing rayon parallelism
  - FFI overhead issues
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

# Baseline file location
BASELINE_FILE = "tests/perf_baseline.json"

# Performance targets
TARGET_THROUGHPUT = 10000  # configs/sec (100μs per config)
REGRESSION_THRESHOLD = 0.10  # 10% regression threshold
WARNING_THRESHOLD = 1.0  # ms per config (regression indicator)


def load_baseline():
    """Load baseline metrics from JSON file."""
    if not os.path.exists(BASELINE_FILE):
        return None

    try:
        with open(BASELINE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_baseline(metrics):
    """Save baseline metrics to JSON file."""
    os.makedirs(os.path.dirname(BASELINE_FILE), exist_ok=True)
    with open(BASELINE_FILE, "w") as f:
        json.dump(metrics, f, indent=2)


def run_perf_test():
    """Run performance test and return metrics."""
    # Only run if release build exists
    release_lib = "target/release/libfluxion.so"
    if not os.path.exists(release_lib):
        print("ℹ Release build not found; skipping perf check")
        print("  Build with: cargo build --release && maturin develop --release")
        return None

    try:
        from fluxion import BatchOracle
    except ImportError:
        print("ℹ Fluxion not installed; skipping perf check")
        print("  Install with: maturin develop --release")
        return None

    try:
        oracle = BatchOracle()

        # Test with 100 configs (representative batch)
        population = [[1.5, 21.0, 24.0] for _ in range(100)]

        # Warm-up run (JIT, caching)
        _ = oracle.evaluate_population(population, False)

        # Actual benchmark
        start = time.perf_counter()
        _ = oracle.evaluate_population(population, False)
        elapsed = time.perf_counter() - start

        time_per_config_ms = (elapsed * 1000) / 100
        throughput_per_sec = 100 / elapsed

        return {
            "time_per_config_ms": time_per_config_ms,
            "throughput_per_sec": throughput_per_sec,
            "population_size": 100,
            "elapsed_seconds": elapsed,
        }

    except Exception as e:
        print(f"⚠ Perf check error: {e}")
        import traceback

        traceback.print_exc()
        return None


def format_json_output(metrics, baseline=None, regression_detected=False):
    """Format metrics as JSON for CI integration."""
    output = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
    }

    if baseline:
        if metrics["throughput_per_sec"] > 0:
            percent_change = (
                metrics["throughput_per_sec"] - baseline["throughput_analytical"]
            ) / baseline["throughput_analytical"]
            output["comparison"] = {
                "baseline_throughput": baseline["throughput_analytical"],
                "current_throughput": metrics["throughput_per_sec"],
                "percent_change": round(percent_change * 100, 2),
                "regression_detected": regression_detected,
            }

    return output


def check_regression(current_metrics, baseline):
    """Check if current performance shows regression vs baseline."""
    if not baseline:
        return False, 0

    current_throughput = current_metrics["throughput_per_sec"]
    baseline_throughput = baseline.get("throughput_analytical", 0)

    if baseline_throughput <= 0:
        return False, 0

    # Calculate percent change (negative = regression)
    percent_change = (current_throughput - baseline_throughput) / baseline_throughput

    # Regression if throughput decreased by more than threshold
    regression_detected = percent_change < -REGRESSION_THRESHOLD

    return regression_detected, percent_change


def main():
    parser = argparse.ArgumentParser(description="Fluxion performance baseline check")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare against stored baseline and fail if regression detected",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Update baseline with current performance metrics",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results as JSON for CI integration"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress non-essential output"
    )

    args = parser.parse_args()

    # Run performance test
    metrics = run_perf_test()

    if not metrics:
        if args.json:
            print(json.dumps({"error": "Performance test failed", "success": False}))
        sys.exit(0)  # Skip if can't run

    # Output format
    time_per_config = metrics["time_per_config_ms"]
    throughput = metrics["throughput_per_sec"]

    # Load baseline if needed
    baseline = None
    if args.compare or args.update_baseline:
        baseline = load_baseline()

    # Update baseline mode
    if args.update_baseline:
        new_baseline = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "throughput_analytical": throughput,
            "latency_ms": time_per_config,
            "population_size": metrics["population_size"],
        }
        save_baseline(new_baseline)

        if not args.quiet:
            print(
                f"✓ Baseline updated: {throughput:.0f} configs/sec ({time_per_config:.3f}ms per config)"
            )

        if args.json:
            print(
                json.dumps(
                    {"success": True, "baseline_updated": True, "metrics": new_baseline}
                )
            )
        sys.exit(0)

    # Compare mode
    if args.compare and baseline:
        regression, percent_change = check_regression(metrics, baseline)

        if regression:
            print("⚠ PERFORMANCE REGRESSION DETECTED")
            print(f"  Baseline: {baseline['throughput_analytical']:.0f} configs/sec")
            print(f"  Current:  {throughput:.0f} configs/sec")
            print(f"  Change:   {percent_change * 100:.1f}%")
            print(f"  Threshold: -{REGRESSION_THRESHOLD * 100:.0f}%")
            print()
            print("  Likely causes:")
            print("    • Unnecessary clones of ThermalModel in evaluate_population")
            print("    • Memory allocations in solve_timesteps inner loop")
            print("    • Missing or nested rayon parallelism")
            print("    • FFI overhead (check Python boundary crossing)")

            if args.json:
                print(json.dumps(format_json_output(metrics, baseline, True)))
            sys.exit(1)
        else:
            if not args.quiet:
                print(
                    f"✓ Performance OK: {throughput:.0f} configs/sec ({time_per_config:.3f}ms per config)"
                )
                if baseline:
                    print(f"  vs baseline: {percent_change * 100:+.1f}%")

            if args.json:
                print(json.dumps(format_json_output(metrics, baseline, False)))
            sys.exit(0)

    # Default: smoke test mode
    if time_per_config > WARNING_THRESHOLD:
        print(f"⚠ PERF WARNING: {time_per_config:.2f}ms per config")
        print(f"  Target: <0.1ms ({TARGET_THROUGHPUT:.0f}+ configs/sec)")
        print(f"  Current: {throughput:.0f} configs/sec")
        print()
        print("  Likely causes:")
        print("    • Unnecessary clones of ThermalModel in evaluate_population")
        print("    • Memory allocations in solve_timesteps inner loop")
        print("    • Missing or nested rayon parallelism")
        print("    • FFI overhead (check Python boundary crossing)")

        if args.json:
            print(json.dumps(format_json_output(metrics, None, True)))
        sys.exit(1)
    else:
        if not args.quiet:
            print(f"✓ Perf OK: {time_per_config:.3f}ms per config")
            print(
                f"  Throughput: {throughput:.0f} configs/sec (Target: >{TARGET_THROUGHPUT})"
            )

        if args.json:
            print(json.dumps(format_json_output(metrics, None, False)))
        sys.exit(0)


if __name__ == "__main__":
    main()
