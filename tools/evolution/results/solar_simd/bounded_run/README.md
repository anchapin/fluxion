# Bounded re-run results (issue #3338)

This directory contains the **bounded re-run** outputs from
`tools/evolution/scripts/run_bounded_campaign.py`. The campaign
drives the in-tree `fluxion-evaluator` recompile pipeline against
three deterministic mutations of every seed under
`tools/evolution/seeds/solar_simd/`:

  1. `identity` — body unchanged from the seed (skipped at compile
     time, baseline evidence).
  2. `soa_pack_4_lane` — packs four independent inputs into a
     `[f64; 4]` lane prelude before the canonical call. Tests
     cache-line locality.
  3. `unroll_4x` — manually unrolls the canonical call into four
     `let _u{0..3} = ...; black_box((...)).0` sinks. Tests
     instruction-level parallelism.

## Why not the full ≥200-gen OpenEvolve campaign

Per issue #3338 §"Run a bounded campaign": full ≥200-gen may
exceed session bounds; land harness + seeds + fixture + configs +
a SHORT deterministic re-run and document the full-run command +
wall-time projection.

The full-run invocation is documented in
`tools/evolution/configs/solar_simd.yaml` (`campaign.bounded.full_run_command`):
projected ~7 wall-clock hours on a 16-core host. The bounded
short re-run in this directory completes in ≈30 minutes on the
configured CI agents (cold-cache compile × 4 candidates).

## Schema v1 contract

Each `*.json` in this directory is a Schema v1 Summary (see
`crates/fluxion-evaluator/src/summary.rs`). The bounded runner
emits the deterministic subset of the harness's Summary
contract; the OpenEvolve adapter (out-of-tree) is expected to
emit an identical-shape Summary with full timing/latency fields.

`index.json` is a one-line-per-candidate roll-up used by the
campaign's reporting tooling.

## Re-run

```text
$ BOUNDED_CAMPAIGN_TIMEOUT_S=900 python3 \
    tools/evolution/scripts/run_bounded_campaign.py
```

Default `--output` is this directory.
