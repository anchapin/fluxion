# fluxion-evaluator

Deterministic headless evaluator harness for evolutionary kernel search. This
crate is the **in-tree foundation** that any evolver (OpenEvolve, AlphaEvolve,
FunSearch, …) programs against; the evolver itself stays out-of-tree
([issue #3336](https://github.com/anchapin/fluxion/issues/3336)).

## Why this crate exists

Alpha-evolve-style automated kernel search needs a **scoring oracle**: given a
candidate Rust implementation of a seeded kernel, compile it hermetically, run
a fixed battery of physics edge cases, and emit a structured fitness score
combining accuracy, invariant violations, and CPU runtime. Every reported
score must be reproducible from the committed candidate + harness, independent
of the campaign that produced it.

The harness **does not** contain the evolver — it only contains the contract.
The OpenEvolve adapter (Python shim, out-of-tree) drives this binary via
stdin/stdout JSON.

## Recompilation (default mode, hermetic)

The evaluator copies the candidate source into a fresh tempdir with a
harness-generated `Cargo.toml` that pins the workspace dep set, runs
`cargo build --target-dir <tempdir>/target` in a subprocess, and dispatches
the compiled artifact against the edge-case battery. The subprocess is
isolated via [`crate::sandbox`]: wall-clock cap (default 60 s), no network,
fresh `cwd`.

## Dynamic loading (`dynamic` feature, opt-in, never used in CI)

A prebuilt `cdylib` implementing the documented ABI can be loaded in place of
recompilation. **The feature is currently a stub** — enabling it does NOT add
`libloading` because that would require a new third-party crate and the
project is at zero headroom on the cargo-deny duplicate-version budget
([issue #3310](https://github.com/anchapin/fluxion/issues/3310)). Every load
returns `DynamicLoadError::NotImplementedInThisBuild`; the ABI is documented
in [`src/dynamic.rs`](src/dynamic.rs) so a follow-up PR can swap in the real
plumbing without breaking callers.

## Threat model

Candidate code is **untrusted**. The harness's only line of defense is the
sandbox ([`src/sandbox.rs`](src/sandbox.rs)):

| Capability | Threat | Mitigation |
|------------|--------|------------|
| Arbitrary Rust source | Compile-time resource exhaustion | Fresh `target/`, no debug-info, wall-clock cap |
| Panic in candidate | Crash the harness | Subprocess isolation; exit code surfaced |
| Infinite loop | Hang the harness | Wall-clock cap (60 s default) |
| Memory exhaustion | OOM the runner | Best-effort platform-dependent cap (advisory) |
| Network access | Exfiltrate source | `CARGO_NET_OFFLINE=true` (opt-out: `FLUXION_EVAL_ALLOW_NET=1`) |

## Schema v1

Every evaluation emits exactly one `Summary` JSON object:

```json
{
  "schema_version": 1,
  "candidate_id": "ctf-seed-0042",
  "generation": 137,
  "fitness": 0.9842,
  "compiled": true,
  "invariants_passed": true,
  "max_error": 0.00012,
  "eval_latency_ns": 412,
  "eval_latency_spread_ns": 18,
  "determinism_digest": "sha256:b94d…",
  "outcome": "evaluated",
  "invariant_violations": [],
  "min_invariant_margin": 0.999999
}
```

Schema versioning: bumping policy is documented at the top of
[`src/summary.rs`](src/summary.rs). `schema_version: 1` is the only version
this build emits.

## Exit codes

| Code | Meaning |
|------|---------|
| 0    | Evaluation succeeded (consult `fitness`) |
| 2    | Compile failure |
| 3    | Invariant hard-fail (fitness forced to 0.0) |
| 4    | Timeout / resource cap hit |

## Acceptance

- `cargo test -p fluxion-evaluator` is green.
- Lockfile additions: **zero** new entries (the crate uses only existing
  workspace deps: `serde`, `serde_json`, `thiserror`, `sha2`).
- `cargo clippy --workspace --all-targets -- -D warnings` is clean.
- `cargo fmt -- --check` is clean.

## Reference

- Issue #3336 — original design + acceptance criteria.
- Issue #3310 — duplicate-version budget (zero headroom).
- Issue #3321 — MSRV pinned at 1.98.0 (also matches root toolchain).
- OpenEvolve — the recommended out-of-tree evolver
  ([`algorithmicsuperintelligence/openevolve`](https://github.com/algorithmicsuperintelligence/openevolve));
  the OpenEvolve adapter (Python shim) lives under `tools/evolution/` in a
  follow-up PR.