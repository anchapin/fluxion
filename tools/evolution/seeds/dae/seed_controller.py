"""
Issue #3339 — Seed controller for evolving the BDF DAE engine's
adaptive damping strategy.

This file is the **Python seed** OpenEvolve mutates. OpenEvolve reads
the source of this file, slices the region between the
`# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` markers, asks the LLM
to produce a replacement, splices it back, and reruns the evaluation.

The SCRIPT-WIDE INVARIANTS the evolver must respect:

- **Byte-equivalent baseline**: the `golden_baseline_strategy()`
  function returns the exact numeric knobs the existing
  `NewtonRaphsonConfig::default()` flow uses
  (`mode=0, baseline_factor=1.0`, all other fields arbitrary because
  they are ignored when `mode==0`). The regression test
  `tests/golden_traces_regression.py` runs the binary with this
  baseline strategy and confirms it reproduces the byte-identical
  Summary JSON committed under
  `tools/evolution/results/dae/golden/baseline.json`.

- **Schema v1 contract**: the strategy dict must satisfy the
  `StrategySpec` schema the `bdf_evaluator` binary parses. Any
  missing field rejects the candidate (`fitness = 0`, exit 2).

- **Single-method heuristic**: the EVOLVE block contains ONE function
  (`sample_strategy()`) that returns the dict. The harness drives
  this function once per candidate.

- **Determinism**: the strategy generation must be pure (no RNG,
  no wall-clock, no network). All randomness is injected explicitly
  via the `rng` parameter, which OpenEvolve seeds deterministically.

The full unbounded campaign is run with:

```bash
python3 tools/evolution/orchestrate_openevolve.py \\
    --config tools/evolution/configs/dae.yaml \\
    --seed  tools/evolution/seeds/dae/seed_controller.py \\
    --out   tools/evolution/results/dae/full/
```

A short deterministic re-run lives at
`tools/evolution/orchestrate_bounded.py` and targets 10–30
generations (used to populate `tools/evolution/results/dae/bounded/`).
"""

# ---------------------------------------------------------------------------
# Schema-v1 StrategySpec fields (must match `src/bin/bdf_evaluator.rs`).
#
# Keeping this schema declaration here, co-located with the seed, is the
# cheapest way to ensure the Python seed and the Rust validator agree.
# Any field rename / value-range change requires updating BOTH sides.
# ---------------------------------------------------------------------------
SCHEMA = {
    "mode":              {"type": int,  "values": {0, 1},                        "required": False, "default": 0},
    "baseline_factor":   {"type": float, "range": (0.0, 2.0),                    "required": True, "default": 1.0},
    "floor":             {"type": float, "range": (0.0, 2.0),                    "required": True, "default": 0.25},
    "loose_threshold":   {"type": float, "range": (0.0, 1.0),                    "required": True, "default": 0.5},
    "tight_threshold":   {"type": float, "range": (0.0, 1.5),                    "required": True, "default": 0.95},
    "aggressiveness":    {"type": float, "range": (0.0, 4.0),                    "required": True, "default": 1.0},
    "history_window":    {"type": int,   "range": (0, 32),                       "required": False, "default": 4},
    "max_steps":         {"type": int,   "range": (1_000, 200_000),              "required": False, "default": 50_000},
}


def golden_baseline_strategy():
    """The byte-equivalent baseline strategy.

    Returns the *exact* numeric knobs the existing
    `NewtonRaphsonConfig::default()` flow uses. The Rust validator's
    `mode==0` branch ignores every field except `baseline_factor`,
    so the other numeric values are placeholders; their defaults
    mirror `DampingPolicy::default()`.

    The regression test
    `tests/golden_traces_regression.py` (see
    `tools/evolution/results/dae/golden/baseline.json`) confirms the
    Summary JSON bytes match this strategy input verbatim.

    Returns:
        dict: schema-v1 strategy spec.
    """
    return {
        "mode":            0,
        "baseline_factor": 1.0,
        "floor":           0.25,
        "loose_threshold": 0.5,
        "tight_threshold": 0.95,
        "aggressiveness":  1.0,
        "history_window":  4,
        "max_steps":       50_000,
    }


# EVOLVE-BLOCK-START
# ===========================================================================
# === OpenEvolve evolution surface ===
#
# The function below is the SINGLE-METHOD HEURISTIC the evolver mutates.
# It returns a Schema-v1 strategy dict; OpenEvolve replaces this body with
# its LLM-proposed candidate, then `orchestrate_openevolve.py` calls
# `bdf_evaluator` with the returned dict and scores the run.
#
# Fitness signal (from `bdf_evaluator`):
#   fitness = 1 / (1 + total_newton_iters · 1e-4 + total_accepted_steps · 1e-4)
#
# Hard invariants that FLIP THE SCORE TO ZERO:
#   • any circuit has NaN / Inf in the final state (NaN/Inf reject)
#   • any circuit accumulates conservation-violation events (>1e-7 rel.)
#   • any `driver.run()` failed to converge to its `t_end` within `max_steps`
#
# The strategy knobs the evolver is expected to tweak:
#   • mode (0 = fixed / 1 = residual-ratio-aware)
#   • baseline_factor (the ceiling — full step)
#   • floor (lower bound on the ratio-aware factor)
#   • loose_threshold / tight_threshold (band limits on the residual ratio)
#   • aggressiveness (slope of the linear interpolation in the band)
#   • history_window (memory for the ratio; capped at 32)
#
# The 5 stiff benchmark circuits (in `src/physics/bdf_benchmarks.rs`)
# are deliberately chosen so no single-line tweak wins them all:
#   • Mixing-valve closure (mass conservation at a 1-second ramp)
#   • Pump frequency ramp (continuous forcing)
#   • AHU cooling-coil wet-surface (dehumidification latent-load pulse)
#   • Primary/secondary decoupling loop (2-state tight coupling)
#   • Heat-pump entering-fluid step (C⁰ COP curve discontinuity)
#
# A candidate that lands 25%-fewer iterations than baseline (across the
# suite) AND keeps all invariants clean is the issue's `resolve` bar.
# ===========================================================================
def sample_strategy(rng=None):
    """Sample one candidate strategy spec.

    OpenEvolve calls this once per candidate. The function MUST be
    deterministic given `(None,)` or `(rng with fixed seed)`.

    Args:
        rng: An injectable random-number generator. The OpenEvolve
             adapter seeds `random.Random(seed)` deterministically per
             candidate; advanced candidates can use `rng.uniform()`,
             `rng.gauss()`, etc. Passing `None` is allowed and means
             "no randomness — return a deterministic default".

    Returns:
        dict: schema-v1 strategy spec.
    """
    if rng is None:
        # Deterministic fallback. Mirror the baseline; OpenEvolve's
        # first generation over this seed must produce a Summary
        # byte-identical to the golden baseline trace.
        return golden_baseline_strategy()

    # LLM-mutated body lives here. This scaffolding is the **seed**:
    # the evolver replaces everything between this and EVOLVE-BLOCK-END.
    # The expected edit is a residual-ratio schedule — e.g.:
    #
    #   mode = 1                                # enable residual-ratio
    #   ratio = current_residual / previous    # bounded in [0, 2]
    #   if ratio <= loose_threshold:
    #       factor = baseline_factor            # full step
    #   elif ratio >= tight_threshold:
    #       factor = floor                      # conservative
    #   else:
    #       t = (ratio - loose_threshold) / (tight - loose)
    #       factor = baseline + aggressiveness * t * (floor - baseline)
    #   factor = clamp(factor, floor, baseline)
    #
    # but the LLM is free to mutate the body however it likes.
    strategy = golden_baseline_strategy()

    # Tiny perturbation to seed OpenEvolve's MAP-Elites island with a
    # slightly non-baseline candidate on the very first generation
    # (it would otherwise have zero diversity).
    strategy["baseline_factor"] = float(rng.uniform(0.95, 1.05))
    strategy["floor"]           = float(rng.uniform(0.20, 0.40))
    strategy["loose_threshold"] = float(rng.uniform(0.30, 0.70))
    strategy["tight_threshold"] = float(rng.uniform(0.85, 1.10))
    strategy["aggressiveness"]  = float(rng.uniform(0.50, 2.00))
    # Keep loose < tight by construction:
    if strategy["loose_threshold"] >= strategy["tight_threshold"]:
        strategy["tight_threshold"] = min(1.5, strategy["loose_threshold"] + 0.2)
    return strategy
# EVOLVE-BLOCK-END


def main():
    """CLI wrapper for the seed controller.

    Usage:
        python3 seed_controller.py > strategy.json

    Writes the baseline strategy to stdout. Use `--perturbed` (or
    pass `--seed N`) to get a randomised sample; the OpenEvolve
    adapter uses this CLI mode to evaluate the *zero-th* candidate.
    """
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perturbed", action="store_true",
                        help="Apply the random-perturbation branch of sample_strategy.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the underlying PRNG (overrides --perturbed).")
    args = parser.parse_args()

    if args.seed is not None:
        import random
        rng = random.Random(args.seed)
        out = sample_strategy(rng)
    elif args.perturbed:
        import random
        rng = random.Random(0)
        out = sample_strategy(rng)
    else:
        out = golden_baseline_strategy()

    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
