# Issue #3338 — Solar / radiation SIMD/cache-blocked evolution

## Profile-first baseline evidence

Baseline measurements from
`tools/evolution/results/solar_simd/baseline_evidence.json`
(`cargo run --release --example solar_simd_profile`, default
features, 200 000 iterations per loop after 5 000 warmup,
`fe3ced3c0105d161` inputs hash, host: `rustc 1.98.0`):

| Loop                                | Median (ns) | IQR (ns) | Samples |
|-------------------------------------|------------:|---------:|--------:|
| `perez_diffuse_tilted`              |        90.0 |     30.0 | 200 000 |
| `calculate_surface_irradiance`      |       160.0 |     60.0 | 200 000 |
| `surface_radiative_exchange`        |        20.0 |     10.0 | 200 000 |
| `net_lw_floor_pair`                 |        30.0 |     10.0 | 200 000 |
| `sky_radiation_net_flux`            |        20.0 |     10.0 | 200 000 |

These are the loops the issue's revised premise names as the
optimization surface. The numbers are quoted at `target/release`
optimization level.

## What ships in this PR

1. **Profile-first evidence** — `examples/solar_simd_profile.rs`
   writes a deterministic JSON file (`baseline_evidence.json`,
   committed). Use `cargo run --release --example
   solar_simd_profile -- --output <file>` for re-collection;
   reproduce the numbers above in CI without the Criterion
   plotters TTY-quirk trap.

2. **Seed kernels** — `tools/evolution/seeds/solar_simd/` has
   two seeds today:
   * `perez_diffuse_tilted.rs` — Pérez 1990 sky model.
   * `stefan_boltzmann_pair.rs` — single-pair Stefan-Boltzmann.

   Each seed carries the documented `// EVOLVE-BLOCK-START / END`
   markers around the canonical reduction, an explicit
   equivalence contract (default-feature build is bit-identical),
   and a self-test against the canonical scalar value.

3. **Edge-case fixture** — `tools/evolution/edge_cases/solar_simd.json`
   (6 cases, regenerated via `examples/regenerate_simd_edge_cases.rs`
   on demand — never hand-edited). Per-case tolerance defaults
   (`1e-9`) vs. `simd-kernels` (`1e-6`) are tracked in the
   fixture so the harness can switch on the build feature.

4. **OpenEvolve config** — `tools/evolution/configs/solar_simd.yaml`
   documents the islands, population, mutation grammar, invariant
   battery, and the regression test contract that any
   `fluxion-evaluator`-driven campaign must satisfy. The
   OpenEvolve adapter itself is out-of-tree (issue #3336
   rationale).

5. **Bounded re-run** — `tools/evolution/scripts/run_bounded_campaign.py`
   drives the in-tree `fluxion-evaluator` recompile path with three
   deterministic mutations per seed (`identity`,
   `soa_pack_4_lane`, `unroll_4x`). All 6 candidates pass
   the invariant battery; per-candidate summaries are at
   `tools/evolution/results/solar_simd/bounded_run/`.

6. **`simd-kernels` feature** — non-default, opt-in. Adds
   `src/physics/simd_kernels.rs`, which routes through a
   runtime-dispatched wrapper. Default-feature builds are
   *byte-identical* to today; the wrapper resolves to a scalar
   passthrough. Under `--features simd-kernels` the wrapper
   reaches the runtime-detected SIMD path. No changes to
   `fast-math` boundaries.

7. **In-tree smoke test** — `tests/solar_simd_evolution.rs`
   drives both seeds through `fluxion_evaluator::invariant::run_battery`
   against the per-edge fixture. The test passes under both
   default and `--features simd-kernels`.

## Acceptance checklist

- [x] Profile-first evidence posted (above).
- [x] Default-feature `cargo test --workspace` green — see the
      3913 unit-test pass + 19 ASHRAE 600/900 energy-balance gate
      passes (executed locally).
- [x] `--features simd-kernels` suite green (in-tree
      `tests/solar_simd_evolution.rs` passes 6/6 under both
      default and feature configurations).
- [x] No exact-equality asserts anywhere the evolved kernels can
      reach — tolerances: `1e-9` default, `1e-6` `simd-kernels`.
- [ ] ≥20 % throughput improvement on the hot
      `solar_kernel_bench` groups with `simd-kernels` enabled —
      **NOT MET**. The bounded campaign produced winners that
      match the canonical scalar reduction to 1 ulp (correct
      contract); it did not produce a semantic SIMD rewrite,
      because the OpenEvolve adapter is out-of-tree and the
      bounded 4-lane pack mutator has `n=1` inputs. The
      per-call overhead at the dispatch wrapper (default-feature
      scalar passthrough vs. `simd-kernels` runtime-detected SIMD
      stub) was not measured under a real batch caller; the
      OpenEvolve campaign is the only path that produces a
      semantic improvement.
- [x] No changes to conduction / zone-balance solvers (`fast-math`
      do-not-use list untouched).
- [x] `cargo fmt -- --check`, `cargo clippy --workspace
      --all-targets --exclude fluxion-tauri -- -D warnings`,
      `cargo clippy --features simd-kernels -- -D warnings` —
      all clean.
- [x] Tests pass for `fluxion` and `fluxion-evaluator`.

## Decision on closure: `Refs` (keep open)

Per the issue's "Acceptance / decision on closure" section:
> If a bounded campaign cannot complete or meet all criteria,
> use Refs (keep-open) and document. The PR still lands seeds,
> fixture, configs, profile-first evidence, bounded-campaign
> artifacts.

The bounded re-run landed and the harness invariant battery
passed all 6 candidates, but the **≥20 % throughput
improvement acceptance gate was NOT MET** because the scalar
input shape (`n = 1` per call) limits the SIMD win without a
batch caller to amortize per-call overhead. The OpenEvolve
adapter (out-of-tree by design — issue #3336) is the only
realistic path to producing a semantic improvement, and that
requires ≥200 generations of LLM-driven mutation which exceeds
session bounds.

We keep this issue open with the harness / seeds / fixture /
config / bounded re-run / simd-kernels gating landed, so the
OpenEvolve follow-up PR has a turnkey starting point.

## Cross-platform determinism

The wrappers in `src/physics/simd_kernels.rs` use
`is_x86_feature_detected!`-style runtime detection; under
`aarch64` / Windows-ARM the dispatch falls through to a portable
scalar path that is bit-identical to the default-feature build.
Per-issue #2549 follow-up CI runs (Mac/Windows/Linux runners)
are part of the out-of-tree OpenEvolve full-run block.

## Re-using the harness from a future PR

The bounded runner's wrapper template
(`_CANDIDATE_WRAPPER_TEMPLATE`) is byte-stable across mutations;
the runner emits a Schema v1 Summary per candidate. The
OpenEvolve adapter (out-of-tree) reads the same YAML config
(`tools/evolution/configs/solar_simd.yaml`), drives
`openevolve` against `target/release/fluxion-evaluator`, and
emits an identical contract.

```text
$ python3 tools/evolution/scripts/run_bounded_campaign.py
$ python3 tools/evolution/scripts/run_openevolve_campaign.py \
    --config tools/evolution/configs/solar_simd.yaml \
    --generations 200 --population 32 --islands 8
```

(The second script is currently a stub because OpenEvolve is
out-of-tree; the bounded runner is the trust artifact.)
