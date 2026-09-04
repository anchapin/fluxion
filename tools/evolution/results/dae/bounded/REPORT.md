# Issue #3339 — Bounded-campaign summary

**Decision: `Refs` (keep-open)** — issue remains open pending an
unbounded ≥200-generation campaign on the production host. This
document records what the in-session bounded run delivered.

## What this PR lands

- `src/physics/bdf_engine.rs` extended with `DampingPolicy`,
  `BdfDriver`, `DriverStats`; `NewtonRaphsonConfig` now carries a
  configurable damping schedule (issue spec: "residual-ratio-aware
  damping strategy is in scope").
- `src/physics/bdf_benchmarks.rs` introduces the **5 stiff
  benchmark circuits** the issue calls for.
- `src/bin/bdf_evaluator.rs` is the **bounded-campaign fitness
  oracle**: reads a Schema-v1 `DampingPolicy`, runs all 5 circuits
  through `BdfDriver`, emits a Schema-v1 `Summary` JSON.
- `tools/evolution/seeds/dae/seed_controller.py` is the **Python
  seed** with `// EVOLVE-BLOCK-START/END` markers isolating
  `sample_strategy(rng)`. The OpenEvolve adapter edits only this
  block.
- `tools/evolution/configs/dae.yaml` pins the OpenEvolve config
  (islands + checkpointing, Ollama backend).
- `tools/evolution/openevolve_adapter.py` is the thin adapter that
  bridges OpenEvolve's "candidate = file" workflow to
  `bdf_evaluator`'s "candidate = JSON" workflow.
- `tools/evolution/orchestrate_openevolve.py` is the unbounded ≥200-
  generation driver documented for the follow-up runner.
- `tools/evolution/orchestrate_bounded.py` is the in-session
  bounded driver (deterministic Sobol-style random search).
- `tests/bdf_golden_traces_regression.rs` locks the byte-equivalent
  baseline trace (37 total Newton iterations across 5 circuits).
- `tools/evolution/results/dae/golden/baseline.json` is the
  canonical golden Summary.

## Call-graph mapping (issue task #1)

| Production path                  | Role                                                 |
|----------------------------------|------------------------------------------------------|
| `method_selector.rs`             | 5R1C/CTF/FD envelope selector (NOT BDF). Out of BDF. |
| `solver_manager.rs`              | Wraps 5R1C/CTF/FD `HeatConductionSolver` trait obj.   |
| `solver_registry.rs`             | Plug-in registry for envelope solvers.                |
| `src/physics/bdf_engine.rs`     | BDF DAE engine + Newton + adaptive step. **Library primitive** — no production driver wired in yet. |
| `fluxion-fluid/`                 | Component / Pantelides source for HVAC circuits.      |

**Key finding:** the BDF engine is a *library primitive*, not a
production driver. `method_selector`/`solver_manager`/
`solver_registry` route to envelope solvers (5R1C/CTF/FD), not to
BDF. The BDF evolution lands the integration point and the
benchmark harness; the production wiring is the follow-up issue.

## Benchmark circuits (issue task #3)

| # | Name                              | Dim | Stiffness mechanism          |
|---|-----------------------------------|-----|------------------------------|
| 1 | `mixing_valve_closure`            | 2   | 1-second valve ramp; algebraic constraint pulse |
| 2 | `pump_freq_ramp`                  | 1   | continuous forcing; transient ramp |
| 3 | `cooling_coil_wet`                | 2   | dehumidification latent-load pulse (T = T_dew singularity) |
| 4 | `decoupling_loop_demand`          | 2   | tight 2-state coupling under step demand |
| 5 | `heatpump_entering_fluid_step`    | 1   | C⁰ COP curve discontinuity at T_biv |

## Fitness & invariant wiring (issue task #4)

`bdf_evaluator` aggregates a `DriverStats` across the 5 circuits
and folds them into a Schema-v1 Summary:

- **Fitness**: `1 / (1 + total_newton_iterations × 1e-4 + total_accepted_steps × 1e-4)`. Higher = better.
- **Hard invariants**: each circuit's `ConservationProbe.junction_violates()` returns true if any junction exceeds the 1e-7 relative band, NaN/Inf present, or the driver didn't converge. The Summary forces `fitness = 0.0` on any violation.
- **NaN/Inf rejection**: counted per circuit; non-zero forces zero fitness.
- **Failed-step aborts**: zero — rejected-and-retried steps are counted in `steps_rejected` and penalised, but never abort the run.

## Golden transient traces (issue task)

`golden/baseline.json` is committed and pinned by
`tests/bdf_golden_traces_regression.rs`. The test confirms that
running `bdf_evaluator` with the byte-equivalent baseline strategy
yields the **exact** locked iteration counts:

| Circuit                       | newton_iterations | steps_accepted |
|-------------------------------|-------------------|----------------|
| `cooling_coil_wet`            | 8                 | 6              |
| `decoupling_loop_demand`      | 8                 | 6              |
| `heatpump_entering_fluid_step`| 8                 | 6              |
| `mixing_valve_closure`        | 6                 | 4              |
| `pump_freq_ramp`              | 7                 | 5              |
| **TOTAL**                     | **37**            | **27**         |

Two successive runs of the binary are **byte-identical** in
per-circuit metrics (the `determinism_digest` is excluded — it
hashes candidate_id and the strategy bytes, which vary per
candidate but are stable per candidate_id × strategy_text pair).

## Bounded-campaign outcome (this run)

`python3 tools/evolution/orchestrate_bounded.py --candidate-count 16 --seed 3339`
populates this directory:

- `generation_log.jsonl` — one JSON line per candidate.
- `summary.json` — aggregate campaign stats.
- `winner.json` — the strict-winner's Summary.

Results (`summary.json`):

```text
n_candidates: 16, n_compiled: 16, n_invariants_passed: 15
fitness_stats: mean=0.929, max=0.994 (baseline itself), min=0.0
iteration_stats: min=0, max=148, mean=59.4
baseline_iterations: 37, baseline_accepted_steps: 27
winner: gold-baseline-0000 (fitness=0.994, iters=37, acc=27)
improvement_vs_baseline_iterations_pct: 0.0
```

### Reading the table

- **`mode=0` (fixed-damping baseline)** always scores 0.994
  fitness, 37 Newton iterations. Reflects the issue's
  byte-equivalent golden baseline.
- **`mode=1` (residual-ratio-aware) is HARMFUL on these short
  stiff benchmarks.** Every mode=1 sample in the bounded run had
  MORE Newton iterations than baseline (61, 83, 104, 107, 114,
  148). The pattern: high `tight_threshold` (≥ 1.0) combined with
  low `floor` (≤ 0.5) makes Newton step in tiny increments when
  the residual ratio starts close to 1.0, blowing up iteration
  counts.
- One mode=1 sample hit a conservation violation (gen=13), giving
  fitness=0.0. The pattern there was extreme `floor=0.27` plus
  `aggressiveness=2.2`, which destabilises the conservation
  probe when the stiffness pulse lands mid-integration.

This is a meaningful bounded-campaign result: **the seeded
residual-ratio schedule is over-conservative on these
toy benchmarks**, while a real OpenEvolve run with LLM-driven
mutation would likely rediscover (a) lower `tight_threshold`,
(b) `floor` closer to `baseline_factor`, or (c) a reset-to-baseline
branch for the first few iterations.

### Why this maps to `Refs`, not `resolve`

The issue's acceptance criteria:

> ≥ 25% reduction in Newton iterations vs. the fixed-heuristic
> baseline **across the 5-circuit suite**, with all conservation
> invariants intact on every circuit

- 25% reduction requires ≤ 37 × 0.75 ≈ 27 total Newton iterations.
- Best non-baseline candidate in the bounded run: **61
  iterations** (gen=9). That's 65% more, not 25% less.
- 15/16 candidates passed conservation invariants — that gate is
  met — but the Newton-iteration bar isn't.

The issue's `Refs` decision path applies:

> If bounded campaign cannot complete or meet all criteria, use
> Refs (keep-open) and document the bounded run + full-run
> instructions. The PR still lands harness integration, seeds,
> fixtures, configs, golden traces, and bounded-campaign artifacts.

## Full-run command (the issue's ≥200-generation campaign)

```bash
# Bring up the local LLM:
ollama pull qwen3.5:4b
ollama serve                            # serves http://localhost:11434/v1

# Build the fitness oracle once (required):
cargo build --release --bin bdf_evaluator -p fluxion

# Run the campaign:
source .venv-eval/bin/activate
python3 tools/evolution/orchestrate_openevolve.py \
    --config tools/evolution/configs/dae.yaml \
    --seed   tools/evolution/seeds/dae/seed_controller.py \
    --out    tools/evolution/results/dae/full/ \
    --iterations 200
```

**Wall-time projection** (per `tools/evolution/README.md`):

- Local LLM (`qwen3.5:4b`, ~6 tok/s on CPU) × ~500 prompts per
  generation × ~50–80 s per generation ≈ **4.5 hours** for 200
  generations.
- Smaller pilot: 30 generations ≈ 40 minutes.

If the full campaign finds a winner that beats 27 total Newton
iterations AND keeps all conservation invariants clean, this
issue can be re-opened with the winner candidate spliced back
into `NewtonRaphsonConfig::default()`'s `damping` field and
re-tested under the existing `bdf_golden_traces_regression.rs`
guard.

## Validation gates (all green at merge)

- `cargo test --lib -p fluxion physics::bdf` → 12 passed (9 BDF
  engine + 3 `bdf_benchmarks`).
- `cargo test --test bdf_golden_traces_regression` → 1 passed.
- `cargo test -p fluxion-evaluator` → 1 passed.
- `cargo clippy --workspace --all-targets --exclude fluxion-tauri
  -- -D warnings` → clean.
- `cargo fmt -- --check` → clean.
- `fluxion-core` and the workspace duplicate-version budget
  untouched.
