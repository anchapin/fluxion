#!/usr/bin/env python3
"""First-run ASHRAE 140 demo: a real validation check in ~5 seconds.

Runs the repo's ASHRAE 140-2023 pipeline in its fast mode
(``--skip-engine``): it verifies the licensed suite files against the
ASHRAE publisher hashes, runs the strict ±15% annual-energy gate, and
runs the Python honesty guard that checks every recorded engine value
against the ASHRAE-published bands. No Rust toolchain and no engine
build are needed.

What this is and is not
-----------------------
It compares *recorded* suite values against the normative ASHRAE
reference bands — a real ASHRAE 140 comparison, not mock data. It does
not re-run the physics engine; the full end-to-end run (engine build +
case series, ~10 minutes) is:

    python3 scripts/ashrae_140_pipeline.py

Usage
-----
    python3 examples/run_ashrae_check.py
    python3 examples/run_ashrae_check.py --output report.json

Exit codes mirror the pipeline: 0 all stages pass, 1 validation
failure, 2 a stage could not run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = os.path.join(REPO, "scripts", "ashrae_140_pipeline.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", help="write the pipeline's JSON report here")
    args = ap.parse_args()

    cmd = [sys.executable, PIPELINE, "--skip-engine"]
    if args.output:
        cmd += ["--output", args.output]
    print("Running the ASHRAE 140-2023 validation pipeline (fast mode)...", flush=True)
    print(f"  {' '.join(os.path.basename(c) if i else c for i, c in enumerate(cmd))}\n", flush=True)
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode == 0:
        print(
            "\nAll fast stages passed. For the full end-to-end run "
            "(engine build + case series, ~10 min):\n"
            "  python3 scripts/ashrae_140_pipeline.py\n"
            "Current pass rate and known structural failures: "
            "docs/ASHRAE140_RESULTS.md and docs/KNOWN_ISSUES.md"
        )
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
