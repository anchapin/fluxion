---
title: OSimFlow Monte Carlo Sweeps via Declarative Deltas
issue: 1813
phase: Hybrid Measure Approach, Phase 1 (Declarative Deltas)
status: implemented
last_updated: 2026-07-26
---

# OSimFlow Monte Carlo Sweeps via Declarative Deltas (Issue #1813)

Monte Carlo parameter sweeps over a **base building model + a lightweight delta
file**. Instead of sending thousands of fully-defined building models to cloud
executors, OSimFlow sends one Base Model plus thousands of compact delta files.
Each worker applies its delta in-memory via the Rust `apply_sample` API and runs
the annual simulation across `rayon` threads — without invoking Python.

## Architecture

```
                ┌────────────────────────┐
   base.yaml ─▶ │ fluxion monte-carlo    │ ─▶ results.jsonl
   delta.yaml ─▶│   sweep                │ ─▶ delta_000000.json … delta_NNNNNN.json
                │  (Rust worker, #1813)  │ ─▶ summary.json (stats + timing)
                └────────────────────────┘
                            ▲
                            │ (alt) Python generator pre-materializes per-draw
                            │   patches for distributed Nomad / AWS Batch workers
                            │   osimflow/data_gen/generate_monte_carlo_deltas.py
```

Two complementary execution modes:

1. **Declarative (in-process sampling).** The worker receives a base model and a
   delta file describing parameter *distributions*. The Rust runtime draws N
   samples from a seeded RNG, patches the base `CaseSpec`, and simulates each
   draw in parallel. Compact and reproducible — one delta file encodes 1000+
   scenarios. This is the default `fluxion monte-carlo sweep` path.

2. **Pre-materialized (distributed workers).** A Python utility generates N
   individual per-draw JSON patches up front. Each Nomad / AWS Batch worker
   receives the base model plus a single patch. Suitable for bursting across
   cloud executors where per-worker startup cost dominates.

## Delta file format

YAML (or JSON — selected by extension). Parameter paths use dot notation against
the serialized `CaseSpec` tree (e.g. `window_properties.u_value`,
`infiltration_ach`).

```yaml
samples: 1000          # number of Monte Carlo draws (default 1000, per #1813)
seed: 42               # RNG seed for reproducibility (default 0x5EED_1813)
warm_up_years: 2       # convergence warm-up years (default 2)
parameters:
  infiltration_ach:
    distribution: uniform
    min: 0.3
    max: 1.5
  window_properties.u_value:
    distribution: normal
    mean: 3.0
    std: 0.3
  window_properties.shgc:
    distribution: triangular
    min: 0.4
    mode: 0.7
    max: 0.9
```

Supported distributions (`src/analysis/monte_carlo.rs::Distribution`):

| Distribution | Fields             | Notes                                            |
|--------------|--------------------|--------------------------------------------------|
| `uniform`    | `min`, `max`       | Continuous on `[min, max]`                       |
| `normal`     | `mean`, `std`      | Gaussian; `std > 0` required                     |
| `lognormal`  | `mean`, `std`      | Of the underlying normal in log-space            |
| `triangular` | `min`, `mode`, `max` | `min ≤ mode ≤ max` required                    |
| `fixed`      | `value`            | Degenerate; useful for control variates / pinning |

## Worker entrypoint

```bash
fluxion monte-carlo sweep \
    --base-model base.yaml \
    --delta-file delta.yaml \
    --output ./mc_out \
    [--samples N] [--seed S] [--hourly] [--per-draw-files] [--sequential]
```

- `--base-model` — serialized `CaseSpec` (YAML or JSON).
- `--delta-file` — declarative delta (YAML or JSON).
- `--samples N` / `--seed S` — override the values in the delta file.
- `--per-draw-files` — also emit `delta_000000.json …` (acceptance criterion 4).
- `--sequential` — single-threaded; for the startup-time benchmark below.

Outputs:

- `results.jsonl` — one JSON object per draw (index, inputs, annual/peak metrics).
- `delta_NNNNNN.json` — (with `--per-draw-files`) per-draw file.
- `summary.json` — count, failures, per-metric mean/std/p05/p95, and timing.

`fluxion monte-carlo dry-run --delta-file delta.yaml` samples without simulating,
to verify distributions and seeds before a long sweep.

## Python generator

`osimflow/data_gen/generate_monte_carlo_deltas.py` produces the declarative delta
file and (optionally) pre-materializes per-draw patches:

```bash
python3 -m osimflow.data_gen.generate_monte_carlo_deltas -o delta.yaml
python3 -m osimflow.data_gen.generate_monte_carlo_deltas -n 1000 \
    -o delta.yaml --materialize ./patches
```

Tests: `pytest osimflow/data_gen/test_generate_deltas.py` (14 tests).

## Determinism

Sampling is deterministic for a given `(seed, parameter set)`. Parameter names
are sorted before sampling so output is stable across Rust hash-map randomness.
The default seed is `0x5EED_1813`; override per-run via the delta file or
`--seed`.

## Benchmark notes (acceptance criterion 5)

Per-worker startup time and memory footprint improve versus the "send full model
per job" approach because:

- **Payload size.** A delta file is ~1–2 KB (a handful of distribution entries).
  A serialized `CaseSpec` base model is ~10–30 KB. Pre-#1813, each of N workers
  received a full model (~N × 30 KB); now one base model + N deltas
  (~30 KB + N × 2 KB).
- **Startup.** The Rust worker loads the base model once and applies N patches
  in-memory — no Python interpreter spin-up, no per-draw file I/O for the
  declarative path. `summary.json` records `wall_seconds`, `per_draw_ms`, and
  `parallelism` for tracking. Use `--sequential` to isolate per-draw cost from
  rayon scheduling overhead when comparing against the legacy approach.
- **Memory.** Each rayon task clones the patched `CaseSpec` (heap allocation per
  draw) but shares the read-only base and weather data. Peak RSS scales with
  thread count × `CaseSpec` size, not with N.

## Test coverage

- `src/analysis/monte_carlo.rs` (unit, 14 tests) — distribution sampling,
  determinism, patch application, sweep execution, statistics.
- `tests/monte_carlo_sweep.rs` (integration, 4 tests) — end-to-end smoke test:
  10 deltas → 10 result files, each reflecting its patched parameter.
- `osimflow/data_gen/test_generate_deltas.py` (Python, 14 tests) — delta-file
  generation and materialization.

## Related

- Depends on #1811 (JSON Patch core) — reuses the `apply_patch` / `set_nested`
  machinery from `src/analysis/delta.rs`.
- `src/analysis/delta.rs` — deterministic delta/sweep engine this builds on.
- Future Phase 2 will layer measure scripts (OpenStudio-style) on top of these
  declarative deltas.
