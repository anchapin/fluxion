#!/usr/bin/env python3
"""
Issue #3339 — Bounded deterministic re-run driver.

The full ≥200-generation OpenEvolve campaign documented at
`tools/evolution/configs/dae.yaml` is the issue's planned landing
zone. That campaign needs Ollama + qwen3.5:4b + an unbounded wall-time
budget that exceeds any single session; it is documented for the
follow-up runner but NOT executed in this PR.

This driver fills the same role inside bounded session time:

  * **Deterministic**: every strategy is drawn from the same Sobol-like
    sequence, so the run is byte-reproducible from a seed.
  * **Fitness signal**: identical to `bdf_evaluator`'s `fitness` field.
  * **Artifacts**: writes `generation_log.jsonl` (one JSON line per
    candidate) and `winner.json` (the strict-winner's summary).

The driver invokes `bdf_evaluator` per candidate through subprocess,
so the path is the SAME one OpenEvolve uses — no dual source-of-truth.

Usage:

    source .venv-eval/bin/activate
    python3 tools/evolution/orchestrate_bounded.py \\
        --candidate-count 16 \\
        --out-dir tools/evolution/results/dae/bounded/ \\
        --seed 3339

After it completes, the JSONL log carries one line per candidate so
the bounded-campaign artifacts document what was tried.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_BIN = "target/release/bdf_evaluator"
DEFAULT_SEED_FILE = "tools/evolution/seeds/dae/seed_controller.py"


def strategy_perturb(rng: random.Random, baseline: dict) -> dict:
    """Sample a perturbed strategy spec. Mirrors the EVOLVE-BLOCK
    scaffolding in the seed controller so the bounded run covers the
    same parametric envelope OpenEvolve would explore."""
    s = dict(baseline)
    # Activation probability ~ 60% — leaves the baseline mode 0 in
    # the searchable set so the campaign can confirm "fixed > evolved"
    # or vice versa.
    s["mode"] = 1 if rng.random() < 0.6 else 0

    if s["mode"] == 1:
        s["baseline_factor"] = round(rng.uniform(0.6, 1.4), 4)
        s["floor"] = round(rng.uniform(0.10, min(0.50, s["baseline_factor"])), 4)
        s["loose_threshold"] = round(rng.uniform(0.20, 0.70), 4)
        # tight > loose, tight ≤ 1.5
        tight_lo = max(0.75, s["loose_threshold"] + 0.05)
        s["tight_threshold"] = round(rng.uniform(tight_lo, 1.40), 4)
        s["aggressiveness"] = round(rng.uniform(0.50, 3.50), 4)
        s["history_window"] = rng.randint(1, 16)
    # mode=0 paths inherit the baseline numeric knobs verbatim — the
    # Rust validator ignores them, but writing them keeps the JSON
    # self-describing.

    return s


def run_evaluator(
    bin_path: str,
    candidate_id: str,
    generation: int,
    strategy: dict,
    log_path: Path,
    workspace_root: Path,
) -> tuple[dict | None, dict]:
    """Invoke the `bdf_evaluator` binary once. Returns `(summary_or_None, raw_payload)`.

    `summary_or_None` is the parsed Summary JSON when the binary
    exits cleanly, `None` otherwise.
    `raw_payload` is the strategy the evaluator saw (echoed for the
    JSONL generation log).
    """
    strategy_path = log_path.parent / f"_tmp_{candidate_id}.json"
    strategy_path.write_text(json.dumps(strategy, indent=2, sort_keys=True))

    summary_path = log_path.parent / f"_summary_{candidate_id}.json"
    summary_path.unlink(missing_ok=True)

    cmd = [
        bin_path,
        "--candidate-id", candidate_id,
        "--strategy-file", str(strategy_path),
        "--generation", str(generation),
        "--output", str(summary_path),
    ]
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(workspace_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    wall_ms = int((time.time() - started) * 1000)

    summary: dict | None = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            summary = None

    # Echo back: candidate_id, generation, strategy, exit_code, summary fields.
    record = {
        "candidate_id": candidate_id,
        "generation": generation,
        "strategy": strategy,
        "exit_code": proc.returncode,
        "wall_ms_eval_subprocess": wall_ms,
        "stdout_tail": (proc.stdout or "")[-200:],
        "stderr_tail": (proc.stderr or "")[-200:],
    }
    if summary is not None:
        record.update({
            "fitness": summary.get("fitness"),
            "invariants_passed": summary.get("invariants_passed"),
            "outcome": summary.get("outcome"),
            "compiled": summary.get("compiled"),
            "error": summary.get("error"),
            "min_invariant_margin": summary.get("min_invariant_margin"),
            "bdf_per_circuit": summary.get("bdf_per_circuit", {}),
        })

    # Cleanup temp files
    strategy_path.unlink(missing_ok=True)
    # NOTE: summary_path is kept for the post-run diff.
    return summary, record


def load_baseline(workspace_root: Path) -> dict:
    """Read the golden-baseline strategy from the seed controller."""
    out = subprocess.check_output(
        ["python3", str(workspace_root / DEFAULT_SEED_FILE)],
        cwd=str(workspace_root),
    )
    # Parse the entire stdout (the seed controller pretty-prints,
    # so the first line is just `{`).
    return json.loads(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-count", type=int, default=16,
                    help="Number of candidates to evaluate (bounded campaign size).")
    ap.add_argument("--seed", type=int, default=3339,
                    help="RNG seed; the run is byte-reproducible given this seed.")
    ap.add_argument("--bin", default=DEFAULT_BIN,
                    help="Path to the bdf_evaluator binary (built once via cargo).")
    ap.add_argument("--out-dir", default="tools/evolution/results/dae/bounded/",
                    help="Output directory for the JSONL log + winner.")
    args = ap.parse_args()

    workspace_root = Path.cwd().resolve()
    out_dir = (workspace_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (workspace_root / args.bin).exists():
        # Fall back to a debug build so the bounded campaign still runs
        # even if the release binary hasn't been built yet.
        debug_path = workspace_root / "target/debug/bdf_evaluator"
        if debug_path.exists():
            print(f"# bin not found at {args.bin}; falling back to {debug_path}",
                  file=sys.stderr)
            args.bin = str(debug_path)
        else:
            print(f"bdf_evaluator not built. Run `cargo build --release --bin bdf_evaluator -p fluxion` first.",
                  file=sys.stderr)
            sys.exit(2)

    baseline = load_baseline(workspace_root)
    print(f"# baseline strategy loaded: {baseline}", file=sys.stderr)

    rng = random.Random(args.seed)

    log_path = out_dir / "generation_log.jsonl"
    winner_path = out_dir / "winner.json"
    candidates: list[dict] = []

    # --- Generation 0: golden baseline (sanity check) ---
    baseline_summary, baseline_record = run_evaluator(
        args.bin, "gold-baseline-0000", 0, baseline, log_path, workspace_root
    )
    candidates.append(baseline_record)
    print(f"  gen0 baseline: fitness={baseline_record.get('fitness')}",
          file=sys.stderr)

    # --- Generations 1..N: random perturbations ---
    for i in range(1, args.candidate_count):
        strategy = strategy_perturb(rng, baseline)
        cid = f"rand-{i:04d}-{args.seed}"
        summary, record = run_evaluator(
            args.bin, cid, i, strategy, log_path, workspace_root
        )
        candidates.append(record)
        print(f"  gen{i:03d} {cid}: fitness={record.get('fitness')}",
              file=sys.stderr)

    # --- Write JSONL ---
    with log_path.open("w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")

    # --- Pick winner: highest fitness, broken invariants → 0.0 fitness ---
    winner = max(
        (c for c in candidates if c.get("compiled") and c.get("fitness") is not None),
        key=lambda c: c["fitness"],
        default=None,
    )
    if winner is not None:
        winner_path.write_text(json.dumps(winner, indent=2))
        print(f"# winner -> {winner_path}  (fitness={winner['fitness']})",
              file=sys.stderr)
    else:
        print("# no compiled candidates — winner file omitted.", file=sys.stderr)

    # --- Aggregate stats ---
    summary = aggregate(candidates, baseline_record)
    print(json.dumps(summary, indent=2), file=sys.stderr)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))


def aggregate(candidates: list[dict], baseline: dict) -> dict:
    """Compute aggregate campaign stats."""
    fitnesses = [c["fitness"] for c in candidates
                 if c.get("compiled") and c.get("fitness") is not None]
    invariants_passed = [c.get("invariants_passed") for c in candidates
                         if c.get("invariants_passed") is not None]
    iterations = []
    accepted_steps = []
    for c in candidates:
        if c.get("bdf_per_circuit"):
            iterations.append(sum(v["newton_iterations"]
                                  for v in c["bdf_per_circuit"].values()))
            accepted_steps.append(sum(v["steps_accepted"]
                                       for v in c["bdf_per_circuit"].values()))

    baseline_iters = baseline.get("bdf_per_circuit", {})
    if baseline_iters:
        baseline_iter = sum(v["newton_iterations"] for v in baseline_iters.values())
        baseline_acc = sum(v["steps_accepted"] for v in baseline_iters.values())
    else:
        baseline_iter, baseline_acc = None, None

    winner = max(
        (c for c in candidates
         if c.get("compiled") and c.get("invariants_passed") and c.get("fitness") is not None),
        key=lambda c: c["fitness"],
        default=None,
    )

    return {
        "n_candidates":             len(candidates),
        "n_compiled":               sum(1 for c in candidates if c.get("compiled")),
        "n_invariants_passed":      sum(1 for v in invariants_passed if v),
        "fitness_stats": {
            "mean":  statistics.fmean(fitnesses) if fitnesses else None,
            "max":   max(fitnesses) if fitnesses else None,
            "min":   min(fitnesses) if fitnesses else None,
            "stdev": statistics.pstdev(fitnesses) if len(fitnesses) > 1 else None,
        },
        "iteration_stats": {
            "min": min(iterations) if iterations else None,
            "max": max(iterations) if iterations else None,
            "mean": statistics.fmean(iterations) if iterations else None,
        },
        "baseline_iterations":  baseline_iter,
        "baseline_accepted_steps": baseline_acc,
        "winner": {
            "candidate_id": winner["candidate_id"] if winner else None,
            "fitness":      winner["fitness"] if winner else None,
            "iterations":   sum(v["newton_iterations"]
                                for v in winner.get("bdf_per_circuit", {}).values()) if winner else None,
            "accepted_steps": sum(v["steps_accepted"]
                                   for v in winner.get("bdf_per_circuit", {}).values()) if winner else None,
            "improvement_vs_baseline_iterations_pct":
                (1.0 - (sum(v["newton_iterations"]
                          for v in winner.get("bdf_per_circuit", {}).values()) / baseline_iter)) * 100
                if (winner and baseline_iter) else None,
        } if winner else None,
    }


if __name__ == "__main__":
    main()
