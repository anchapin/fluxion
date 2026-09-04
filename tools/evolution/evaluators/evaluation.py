"""OpenEvolve evaluator module — exposes `evaluate` callable.

OpenEvolve imports this file via importlib and calls
`evaluate(candidate_path)` to score each candidate. The function
reads the candidate source from `candidate_path` (a Rust file), runs
the per-candidate build-and-eval pipeline via the
`ctf_evaluator.build_and_run` driver, and returns the schema-v1
Score dict that OpenEvolve uses for selection.

Score convention (per OpenEvolve's `EvaluationResult`):

  {
    "score": <float in [0, 1]>,             # primary fitness
    "metrics": {                            # per-metric breakdown
      "wall_library_size": <int>,
      "max_error": <float>,
      "mean_error": <float>,
      "state_count_max": <int>,
      "dc_gain_margin_min": <float>,
      "monotonic_pass": <bool>,
      "all_finite": <bool>,
      "compile_ok": <bool>,
    },
    "valid": <bool>,                        # whether the score is meaningful
    "error": <str | None>,                  # diagnostic on failure
  }

The harness-level Summary JSON is also returned in the dict for
traceability.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# Make sure the ctf_evaluator module is importable when OpenEvolve
# loads this file via importlib.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent.parent  # tools/evolution/evaluators → repo
sys.path.insert(0, str(_REPO_ROOT / "tools" / "evolution" / "evaluators"))

import ctf_evaluator  # noqa: E402  (import-after-path-adjustment)


def evaluate(candidate_path: str, *args, **kwargs) -> dict:
    """Score a candidate Rust file against the CTF wall library.

    Parameters
    ----------
    candidate_path : str
        Path to the candidate Rust source file (the kernel.rs equivalent
        of the seed — declares `pub struct Candidate` and `impl Kernel`).
    """
    try:
        src = Path(candidate_path).read_text()
        references = ctf_evaluator.load_references(ctf_evaluator.REFERENCE_DIR)
        if not references:
            return {
                "score": 0.0,
                "metrics": {"wall_library_size": 0, "compile_ok": False},
                "valid": False,
                "error": "no reference data found",
            }
        # Extract candidate id and generation from kwargs (OpenEvolve
        # convention; default to a generic id and no generation).
        candidate_id = kwargs.get("candidate_id", "ctf-candidate")
        generation = kwargs.get("generation", None)
        summary = ctf_evaluator.build_and_run(
            src, references, candidate_id, generation,
        )
        # The driver returns the schema-v1 Summary JSON as a dict.
        score = float(summary.get("fitness", 0.0))
        valid = bool(summary.get("compiled", False))
        # Per-metric breakdown.
        metrics = {
            "wall_library_size": len(references),
            "max_error": summary.get("max_error"),
            "min_invariant_margin": summary.get("min_invariant_margin"),
            "state_count_max": None,
            "monotonic_pass": valid,
            "all_finite": valid,
            "compile_ok": valid,
            "invariants_passed": summary.get("invariants_passed", False),
            "outcome": summary.get("outcome"),
            "determinism_digest": summary.get("determinism_digest"),
            "_wallclock_s": summary.get("_eval_wallclock_s"),
            # OpenEvolve uses `combined_score` as the primary ranking
            # metric for MAP-Elites selection. Without it, OpenEvolve
            # falls back to the average of all numeric metrics, which
            # overshoots when many metrics are saturated at 1.0. We
            # alias combined_score to the schema-v1 fitness.
            "combined_score": score,
        }
        error = summary.get("error")
        return {
            "score": score,
            "metrics": metrics,
            "valid": valid and not error,
            "error": error,
            # Echo the schema-v1 summary for traceability (OpenEvolve
            # ignores unknown keys; downstream tooling can grep for
            # them).
            "summary": summary,
        }
    except Exception as e:
        return {
            "score": 0.0,
            "metrics": {"compile_ok": False},
            "valid": False,
            "error": f"evaluator crashed: {e!r}\n{traceback.format_exc()[:500]}",
        }
