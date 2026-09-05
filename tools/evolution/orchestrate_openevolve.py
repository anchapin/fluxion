#!/usr/bin/env python3
"""
Issue #3339 — OpenEvolve ≥200-gen reference driver.

This is the FULL unbounded campaign as the issue intends. It shells
out to OpenEvolve's high-level `run_evolution` API via Python and
points it at:

  • The seed file         `tools/evolution/seeds/dae/seed_controller.py`
  • The eval adapter      `tools/evolution/openevolve_adapter.py`
  • The OpenEvolve config `tools/evolution/configs/dae.yaml`

The bounded-campaign orchestrator
(`tools/evolution/orchestrate_bounded.py`) is the in-session driver;
this one is the unbounded runner that lands the issue's
acceptance-criteria winner when a wall-time budget is available.

Usage (requires the local Ollama daemon at `localhost:11434/v1`):

    source .venv-eval/bin/activate
    python3 tools/evolution/orchestrate_openevolve.py \\
        --config tools/evolution/configs/dae.yaml \\
        --seed   tools/evolution/seeds/dae/seed_controller.py \\
        --out    tools/evolution/results/dae/full/

Wall-time projection (per the local LLM benchmark in
`tools/evolution/README.md`):
  * `qwen3.5:4b` at ~6 tok/s on CPU; ~500 prompts per generation;
    ~50–80 s per generation.
  * 200 generations × 80 s ≈ 4.5 hours wall-time.

Required environment:

    pip install openevolve        # already in .venv-eval
    ollama pull qwen3.5:4b
    ollama serve                  # serves http://localhost:11434/v1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="Path to the OpenEvolve YAML config.")
    ap.add_argument("--seed", required=True,
                    help="Path to the seed file (with EVOLVE-BLOCK markers).")
    ap.add_argument("--out", default="tools/evolution/results/dae/full/",
                    help="Output directory for the checkpoint DB + logs.")
    ap.add_argument("--iterations", type=int, default=200,
                    help="How many generations to run (issue baseline: 200).")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    seed_path = Path(args.seed).resolve()
    out_path = (Path.cwd() / args.out).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        print(f"config not found: {config_path}", file=sys.stderr); sys.exit(1)
    if not seed_path.exists():
        print(f"seed not found: {seed_path}", file=sys.stderr); sys.exit(1)

    # Drive OpenEvolve programmatically. The `openevolve` package
    # doesn't ship a CLI binary (`pip install openevolve` only adds
    # the Python module), so we invoke its `OpenEvolve` class via
    # a small Python script.
    driver_path = out_path / "_openevolve_driver.py"
    driver_path.write_text(textwrap.dedent("""
        import json, os, sys, time
        from pathlib import Path

        # Import OpenEvolve lazily so missing-dep errors surface here.
        try:
            from openevolve import OpenEvolve
            from openevolve.config import Config
        except ImportError as e:
            print(json.dumps({"phase": "import", "ok": False, "error": str(e)}),
                  file=sys.stderr); sys.exit(2)

        cfg_path = Path(sys.argv[1])
        seed_path = Path(sys.argv[2])
        out_path = Path(sys.argv[3])
        iterations = int(sys.argv[4])

        # The `Config` class loads a YAML file. OpenEvolve's
        # `run_evolution` (the high-level helper) takes the same
        # config; we use the lower-level API to expose runtime hooks.
        cfg = Config(str(cfg_path))
        cfg.iterations = iterations

        # OpenEvolve's OpenEvolve class expects an "evaluator"
        # callable that returns a dict with a `score` field. Wrap
        # our adapter through subprocess so the LLM-driven
        # campaign can re-emit Schema-v1 Summary JSON.
        import subprocess as sp
        def evaluator(candidate_source: str, generation: int):
            # Write the candidate to a temp file and shell out
            # to the adapter.
            with open(out_path / f"candidate_gen{generation}.py", "w") as f:
                f.write(candidate_source)
            proc = sp.run(
                [
                    "python3", str(Path.cwd() / "tools/evolution/openevolve_adapter.py"),
                    "--candidate-file", str(out_path / f"candidate_gen{generation}.py"),
                    "--generation",     str(generation),
                    "--candidate-id",   f"openevolve-gen{generation}",
                ],
                capture_output=True, text=True, timeout=120,
            )
            try:
                summary = json.loads(proc.stdout.splitlines()[0])
            except (json.JSONDecodeError, IndexError):
                summary = {"fitness": 0.0, "compiled": False,
                           "error": proc.stderr or "adapter exec failed"}
            fitness = float(summary.get("fitness", 0.0)) if summary.get("compiled") else 0.0
            return {
                "score": fitness,
                "summary": summary,
                "stdout": proc.stdout[:1000],
                "stderr": proc.stderr[:1000],
            }

        try:
            evolver = OpenEvolve(
                initial_code=seed_path.read_text(),
                evaluator=evaluator,
                config=cfg,
                checkpoint_dir=str(out_path / "checkpoints"),
                log_dir=str(out_path / "logs"),
            )
            print(json.dumps({"phase": "ready", "ok": True,
                              "iterations": iterations,
                              "out_dir": str(out_path)}), file=sys.stderr)
            result = evolver.run(iterations=iterations)
            print(json.dumps(result, default=str))
        except Exception as e:
            print(json.dumps({"phase": "run", "ok": False, "error": repr(e)}),
                  file=sys.stderr); sys.exit(3)
    """).strip())

    cmd = [sys.executable, str(driver_path),
           str(config_path), str(seed_path), str(out_path), str(args.iterations)]
    print("+", " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, env={**__import__("os").environ})
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
