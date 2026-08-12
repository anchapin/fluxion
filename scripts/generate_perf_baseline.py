#!/usr/bin/env python3
"""
generate_perf_baseline.py — generate/regenerate tests/perf_baseline.json.

This is the canonical "regenerate" command referenced by the fail-loud guard
in tests/performance_regression_test.rs (issue #2680). It runs the SAME
harness CI uses — `cargo test --test performance_regression_test --release
test_performance_regression` (population=100) — N times, parses the printed
`Throughput:` / `Latency per config:` lines, and writes the median to
`tests/perf_baseline.json` (computed via Python statistics.median, per
RULES.md constraint #0). Values are MEASURED, never invented.

Usage:
    python3 scripts/generate_perf_baseline.py                 # median-of-7 -> tests/perf_baseline.json
    python3 scripts/generate_perf_baseline.py tests/perf_baseline.json 5
    python3 scripts/generate_perf_baseline.py tests/perf_baseline.json 7 --hard-gate

The emitted baseline is `report-only` by default because it is usually
generated on a DEV machine, not the GitHub-hosted `ubuntu-latest` runner that
CI's absolute-perf-gate (#2693) and regression-check (#1618) jobs use. Runner
throughput differs far more than the 5% regression threshold across hardware,
so a dev baseline cannot be a hard cross-runner gate — passing `--hard-gate`
opts into a panic-on-regression baseline (use only when generating on the
same runner class the gate runs on, e.g. ubuntu-latest).
"""
import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "tests" / "perf_baseline.json"

THROUGHPUT_RE = re.compile(r"Throughput:\s*([\d.]+)\s*configs/sec")
LATENCY_RE = re.compile(r"Latency per config:\s*([\d.]+)ms")

CMD = [
    "cargo", "test", "--test", "performance_regression_test",
    "--release", "test_performance_regression",
    "--", "--nocapture",
]


def one_run(idx: int, n: int) -> dict:
    print(f"=== measurement run {idx}/{n} ===", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(CMD, capture_output=True, text=True, timeout=600,
                          cwd=str(REPO_ROOT))
    dt = time.perf_counter() - t0
    out = proc.stdout + "\n--stderr--\n" + proc.stderr
    th = THROUGHPUT_RE.findall(out)
    la = LATENCY_RE.findall(out)
    if not th or not la:
        print(f"  WARNING: run {idx} produced no parseable metrics "
              f"(rc={proc.returncode}). Excerpt:", flush=True)
        print("    " + out[-800:].replace("\n", "\n    "), flush=True)
        return {}
    throughput = float(th[-1])
    latency = float(la[-1])
    print(f"  run {idx}: throughput={throughput:.1f} cfg/s, "
          f"latency={latency:.4f} ms/cfg  (wall={dt:.1f}s, rc={proc.returncode})",
          flush=True)
    return {"throughput": throughput, "latency": latency}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", nargs="?", default=str(DEFAULT_OUT),
                    help=f"Output JSON path (default: {DEFAULT_OUT})")
    ap.add_argument("n", nargs="?", type=int, default=7,
                    help="Number of measurement runs (default: 7)")
    ap.add_argument("--hard-gate", action="store_true",
                    help="Mark the baseline enforcement=hard-gate (panic on "
                         "regression). Only use when generating on the SAME "
                         "runner class the gate runs on.")
    args = ap.parse_args()

    n = max(1, args.n)
    enforcement = "hard-gate" if args.hard_gate else "report-only"

    samples = [s for i in range(1, n + 1) if (s := one_run(i, n))]

    if len(samples) < 3:
        print(f"\nERROR: only {len(samples)}/{n} runs produced metrics; cannot "
              f"compute a stable median. Aborting without writing a baseline.",
              file=sys.stderr)
        return 2

    throughputs = [s["throughput"] for s in samples]
    latencies = [s["latency"] for s in samples]
    med_t = statistics.median(throughputs)
    med_l = statistics.median(latencies)

    print(f"\nSamples ({len(samples)}):", flush=True)
    print(f"  throughput cfg/s : {[round(x, 1) for x in throughputs]}", flush=True)
    print(f"  latency    ms/cfg: {[round(x, 4) for x in latencies]}", flush=True)
    print(f"  mean throughput  : {statistics.mean(throughputs):.1f}", flush=True)
    print(f"  stdev throughput : {statistics.pstdev(throughputs):.1f}", flush=True)
    print(f"MEDIAN throughput  : {med_t:.1f} configs/sec", flush=True)
    print(f"MEDIAN latency     : {med_l:.4f} ms/config", flush=True)

    # Format expected by performance_regression_test.rs::load_baseline.
    baseline = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "throughput_analytical": round(med_t, 3),
        "latency_ms": round(med_l, 6),
        "population_size": 100,
        "_meta": {
            "measured_at": datetime.now(timezone.utc).date().isoformat(),
            "methodology": (
                "Median of N runs of `cargo test --test performance_regression_test "
                "--release test_performance_regression` (population=100, 1 warmup + "
                "1 measured per run). Same harness CI's absolute-perf-gate (#2693) "
                "and performance.yml use. Computed via Python statistics.median "
                "(RULES.md constraint #0)."
            ),
            "n_runs": len(samples),
            "strategy": "median",
            "samples": {
                "throughput_configs_per_sec": [round(x, 3) for x in throughputs],
                "latency_ms_per_config": [round(x, 6) for x in latencies],
            },
            "machine": {
                "platform": platform.platform(),
                "processor": platform.processor() or "unknown",
                "cpu_count": os.cpu_count(),
            },
            "enforcement": enforcement,
            "runner_class": "ci-ubuntu-latest" if args.hard_gate else "dev-local",
            "stale_after_days": 90,
            "regression_threshold_pct_source":
                "release_gates.yaml benchmark.regression_threshold",
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)
        f.write("\n")
    print(f"\nWrote {out_path} (enforcement={enforcement})", flush=True)
    if enforcement == "report-only":
        print("NOTE: report-only baselines print a loud WARNING on regression "
              "but do not fail the test (cross-runner throughput is not "
              "comparable within 5%). Use --hard-gate only when generating on "
              "ubuntu-latest.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
