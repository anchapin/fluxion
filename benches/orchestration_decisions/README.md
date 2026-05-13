# Orchestration Decision Benchmark Harness

Benchmarks and quality scoring for fluxion's 5 simulation orchestration decision types.

Implements the **TDQS (Temporal Decision Quality Score)** metric from Issue #708.

---

## Directory Layout

```
benches/orchestration_decisions/
├── tdqs.rs               — TDQS metric implementation (Rust, unit-tested)
├── decision_recorder.rs  — Decision recording middleware + mock decision functions
├── benchmark_runner.rs   — Criterion benchmark entry ([[bench]] in Cargo.toml)
├── dataset/
│   ├── labeled_decisions.json   — 195 ASHRAE 140 labeled decisions (ground truth)
│   ├── ashrae140_replay.json    — Full ASHRAE 140 replay metadata
│   └── generate_dataset.py     — Regenerate dataset from simulation logs
├── metrics/
│   └── tdqs.py                 — Python cross-check of TDQS formula
├── baselines/
│   ├── rule_based_baseline.json — Current rule-based system baseline (TDQS ≈ 0.70)
│   ├── random_baseline.json     — Chance-performance reference (TDQS = 0.50)
│   └── current_tdqs.json        — Written by benchmark run, read by CI regression check
└── README.md
```

---

## TDQS Formula

```
TDQS = Σᵢ [ correct(dᵢ) × w(dᵢ) × cost_avoided(dᵢ) ]
       ───────────────────────────────────────────────
       Σᵢ [ w(dᵢ) × cost_available(dᵢ) ]
```

| Decision Type      | Weight | Max cost saved (s) | Known gap                          |
|--------------------|--------|--------------------|------------------------------------|
| Solver selection   | 3.0    | 300                | Issue #726: CTF on 900-series      |
| Adaptive timestep  | 1.5    | 45                 | ✅ Working correctly               |
| Surrogate routing  | 2.0    | 2                  | Not yet deployed (v2.1+)           |
| Constraint warning | 1.0    | 30                 | Post-hoc only; no pre-flight check |
| HVAC horizon       | 1.5    | 10                 | Fixed 24h; 72h/6h not implemented  |

**Current baseline TDQS: ~0.70** (rule-based system)

Expected after Issue #726 fix (Wave 1): TDQS → ~0.83

---

## Quick Start

```bash
# Regenerate the 195-decision labeled dataset
python3 benches/orchestration_decisions/dataset/generate_dataset.py

# Run Python cross-check
python3 benches/orchestration_decisions/metrics/tdqs.py

# Run Criterion benchmarks (builds and times all 5 decision types + TDQS computation)
cargo bench --bench orchestration_decisions

# Run CI regression check manually
python3 scripts/check_tdqs_regression.py \
  --current  benches/orchestration_decisions/baselines/current_tdqs.json \
  --baseline benches/orchestration_decisions/baselines/rule_based_baseline.json \
  --threshold 0.05
```

---

## CI Integration

`.github/workflows/tdqs_regression.yml` runs on every push/PR:

1. Regenerates the labeled dataset
2. Runs `cargo bench --bench orchestration_decisions`
3. Calls `scripts/check_tdqs_regression.py` — fails CI if TDQS drops > 5 pp on any type
4. Posts PR comment with per-type delta table
5. Uploads `current_tdqs.json` as a 90-day artifact

Regression is **warn-only on PRs** and **hard-fail on main**.

---

## Building Scientist Handoff

The mock decision functions in `decision_recorder.rs` need to be replaced with real engine hooks once `src/orchestration/decision_types.rs` exists.

Required interface:

```rust
// In src/orchestration/decision_types.rs (Building Scientist to implement)
pub enum OrchestrationDecisionKind {
    SolverSelection,
    AdaptiveTimestep,
    SurrogateRouting,
    ConstraintWarning,
    HvacHorizon,
}

pub struct OrchestrationDecision {
    pub kind: OrchestrationDecisionKind,
    pub chosen: String,         // e.g. "ctf", "fd", "trigger", "surrogate"
    pub features: serde_json::Value, // decision-site features
}
```

And at each call site, emit a `tracing::info!` span:

```rust
tracing::info!(
    decision_type = "solver_selection",
    input_density = layer.density,
    input_thickness = layer.thickness,
    chosen_solver = %solver_type,
    "Solver selection decision"
);
```

Once these are in place, replace the mock functions in `decision_recorder.rs` with real calls
and update `benchmark_runner.rs` to import from `src/orchestration/`.

---

## Interpreting TDQS

| TDQS  | Interpretation                              |
|-------|---------------------------------------------|
| 1.0   | All decisions correct, max savings captured |
| ~0.75 | Expected rule-based system baseline         |
| 0.5   | Chance performance                          |
| < 0.5 | Systematic bias present                     |

**Target for v1.3 release: TDQS ≥ 0.85**
