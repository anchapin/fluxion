#!/usr/bin/env python3
"""
Issue #3339 — OpenEvolve adapter for the BDF DAE evolver.

OpenEvolve is the reference evolver (per `tools/evolution/README.md`).
It mutates source files sliced by `// EVOLVE-BLOCK-START` /
`// EVOLVE-BLOCK-END` markers, recompiles the candidate, and feeds the
resulting binary's stdout JSON back to its population DB.

The evolver fragment in the seed returns a Python DICT, not a Rust
file. OpenEvolve's evaluator expects a "candidate file" that is
*executable* (a Rust source or a shell-style program). To bridge the
two, this adapter:

  1. Writes the seed_controller.py to a temp directory (or accepts a
     path to one already-populated by OpenEvolve).
  2. Calls `python3 seed_controller.py --seed <N>` to produce the
     candidate's strategy JSON.
  3. Hands that JSON to `./target/release/bdf_evaluator --strategy-file ...`
     (a fast Rust re-evaluation, no recompile).
  4. Parses the Summary JSON stdout and returns it to OpenEvolve.

This adapter is INTENTIONALLY A SHIM. The real evolution stays in
OpenEvolve's sandbox; this script makes the `bdf_evaluator` binary
look like an OpenEvolve-compatible evaluator function.

CLI exit codes match OpenEvolve's contract:
  0 — candidate evaluated; consult stdout JSON for `score`.
  2 — compile failure (or strategy invalid)
  3 — invariant hard fail
  4 — timeout / cap
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-file", required=True,
                    help="Path to the OpenEvolve-mutated seed file (Python source).")
    ap.add_argument("--generation", type=int, default=0,
                    help="Generation index (echoed into the Summary).")
    ap.add_argument("--candidate-id", default=None,
                    help="Stable identifier; default = basename of --candidate-file.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed for the candidate's deterministic RNG step.")
    ap.add_argument("--bin", default="target/release/bdf_evaluator",
                    help="Path to the bdf_evaluator binary.")
    args = ap.parse_args()

    candidate_file = Path(args.candidate_file).resolve()
    if not candidate_file.exists():
        # OpenEvolve may not have written the file yet if the eval ran
        # against the seed baseline. Fail closed with exit 2.
        emit_and_exit(2, "compile_failure", f"candidate file not found: {candidate_file}",
                      generation=args.generation, candidate_id=args.candidate_id or candidate_file.stem)

    cid = args.candidate_id or candidate_file.stem

    # 1. Run the (OpenEvolve-mutated) seed to obtain the strategy dict.
    cmd_seed = ["python3", str(candidate_file)]
    if args.seed is not None:
        cmd_seed += ["--seed", str(args.seed)]
    # else: default branch returns the byte-equivalent baseline.
    try:
        seed_out = subprocess.check_output(cmd_seed, text=True, timeout=30)
    except subprocess.CalledProcessError as e:
        emit_and_exit(2, "compile_failure", f"seed exec failed: {e}",
                      generation=args.generation, candidate_id=cid)
    except subprocess.TimeoutExpired:
        emit_and_exit(4, "resource_cap", "seed exec timed out",
                      generation=args.generation, candidate_id=cid)

    try:
        strategy = json.loads(seed_out.splitlines()[0])
    except json.JSONDecodeError as e:
        emit_and_exit(2, "compile_failure", f"strategy JSON parse failed: {e}\n{seed_out[:200]}",
                      generation=args.generation, candidate_id=cid)

    # 2. Hand the strategy to the bdf_evaluator binary.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as sf:
        json.dump(strategy, sf, indent=2, sort_keys=True)
        strategy_path = sf.name
    summary_path = None

    try:
        cmd_eval = [
            args.bin,
            "--candidate-id", cid,
            "--strategy-file", strategy_path,
            "--generation", str(args.generation),
        ]
        proc = subprocess.run(cmd_eval, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        emit_and_exit(4, "resource_cap", "evaluator subprocess timed out",
                      generation=args.generation, candidate_id=cid)
    finally:
        os.unlink(strategy_path)
        if summary_path and os.path.exists(summary_path):
            os.unlink(summary_path)

    # The bdf_evaluator emits a Schema-v1 Summary on stdout. We
    # forward it verbatim (OpenEvolve reads `fitness`,
    # `invariants_passed`, etc. directly).
    if proc.returncode not in (0, 3):
        # 2 = strategy parse fail, 4 = cap. Surface the error so
        # OpenEvolve records the attempt.
        try:
            summary = json.loads(proc.stdout)
        except json.JSONDecodeError:
            summary = {
                "schema_version": 1,
                "candidate_id": cid,
                "generation": args.generation,
                "fitness": 0.0,
                "compiled": False,
                "invariants_passed": False,
                "max_error": None,
                "eval_latency_ns": None,
                "eval_latency_spread_ns": None,
                "determinism_digest": None,
                "outcome": "compile_failure",
                "invariant_violations": [],
                "error": proc.stderr or f"exit {proc.returncode}",
                "min_invariant_margin": 0.0,
            }
        sys.stdout.write(json.dumps(summary))
        sys.stdout.write("\n")
        sys.exit(proc.returncode)

    # Normalise: the bdf_evaluator also returns a JSON on its stdout
    # when --output is absent (which is our default).
    sys.stdout.write(proc.stdout)
    sys.stdout.write("\n")
    sys.exit(proc.returncode)


def emit_and_exit(code: int, outcome: str, error: str, *,
                  generation: int, candidate_id: str):
    """Emit a Schema-v1 failure Summary and exit with `code`."""
    summary = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "generation": generation,
        "fitness": 0.0,
        "compiled": False,
        "invariants_passed": False,
        "max_error": None,
        "eval_latency_ns": None,
        "eval_latency_spread_ns": None,
        "determinism_digest": None,
        "outcome": outcome,
        "invariant_violations": [],
        "error": error,
        "min_invariant_margin": 0.0,
    }
    sys.stdout.write(json.dumps(summary))
    sys.stdout.write("\n")
    sys.exit(code)


if __name__ == "__main__":
    main()
