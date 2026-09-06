# ADR-0014: `cargo nextest` test-runner adoption (audit-first rollout)

> **Summary 1/7:** `Rust Tests & Linting / Test (ubuntu-latest, multi-zone)` spends 305s of "Run tests" on a 2-vCPU GH runner; 54.80s of that is the 3,922 lib tests running single-threaded under `cargo test`, with the rest split across 7 separate `cargo test` invocations.
> **Summary 2/7:** Adopt `cargo nextest run --workspace --all-targets` for the test matrix; per-binary overrides in `.config/nextest.toml` let us gate any non-thread-safe binary found by the audit at `concurrency = 1` without code changes.
> **Summary 3/7:** Rollout model is audit-first wholesale switch (one PR), NOT shadow mode: empirical pass on the 3,922-test lib suite (`cargo nextest run --lib --test-threads=2` ×5) plus manual source review of the 6 small integration test binaries.
> **Summary 4/7:** Local fix (`nextest.toml` `concurrency = 1` override) is the recommended remediation for any audit-flagged binary; upstream refactors for true thread-safety are out of scope and tracked separately if materialised.
> **Summary 5/7:** No change to `release_gates.yaml::ci.required_checks` or to the per-binary ASHRAE 140 / energy-conservation / h_tr_em / surrogate-drift / fluxion-grid tolerance bands.
> **Summary 6/7:** Public OSS unlimited GH minutes — no cost pressure to debloat; the audit is the real safety mechanism, not a multi-week shadow-mode observation window.
> **Summary 7/7:** Documented acceptance: median PR feedback wall-clock for `Rust Tests & Linting` ≤ 15 min (from 2026-09 baseline ~38 min); ≥ 90% of PR runs of the long-pole matrix entries end in `success` not `cancelled` (from 2026-09 baseline ~17% = 1/6).

- **Status:** Accepted
- **Date:** 2026-09-06 (record created)
- **Deciders:** Fluxion maintainers (TBD)
- **Supersedes:** None
- **Depends on:** Issue #3366 (parent tracking issue)
- **Issue:** [#3366](https://github.com/anchapin/fluxion/issues/3366)
- **Related:** ADR-0015 (per-`head_sha` concurrency grouping, the coordinated second leg of #3366); `release_gates.yaml::ci.required_checks`; `AGENTS.md` §"Commands That Are Easy to Guess Wrong" (`cargo test --profile ci` discipline); `scripts/check_docs_summaries.py` (post-write verification)

---

## Context

The 2026-09-04 to 2026-09-06 PR-fleet data (`gh run list --repo anchapin/fluxion`) shows the `Rust Tests & Linting` workflow as the dominant PR feedback bottleneck:

| Metric | Value (2026-09 baseline) | Source |
|---|---|---|
| Wall-clock for full successful run | **37m 54s** (run db=34008638984) | `gh run view` |
| `Test (ubuntu-latest, multi-zone)` total | 347s | `gh api /actions/jobs/...` |
| `Run tests` step within that job | 305s (88.3% of job) | step breakdown |
| `cargo test --lib` (3,922 tests, **single-threaded**) | **54.80s** | log marker |
| Successful PR runs of this workflow | 1 of 6 (~17%) | run list |

`sccache` is already correctly wired via `.github/actions/setup-rust-env/action.yml` (default `sccache: 'true'`, `RUSTC_WRAPPER=sccache`, `SCCACHE_GHA_ENABLED=true`) and is operating at **99.78% Rust hit rate** in the multi-zone job — so cold compilation is *not* the bottleneck. The bottleneck is the sequential test runner on a 2-vCPU GH free runner, multiplied by 7 separate `cargo test` invocations per matrix entry (`cargo test --lib`, plus 5 integration binaries, plus `cargo test -p fluxion-twin --tests`).

`cargo nextest` is the canonical Rust next-generation test runner: each test binary runs in its own process, and within each binary tests run with explicit thread parallelism. It is not currently used anywhere in the repo (`grep -rn "cargo-nextest\|nextest" .github/` → 0 hits; no `.config/nextest.toml` exists). The project uses `cargo test -- --test-threads=N` only for diagnostic isolation (see `copilot-instructions.md` lines 254 and 423) — never for parallelism in the test matrix.

The project already uses Rayon extensively for physics parallelism (`BatchOracle::evaluate_population`, see ADR #2769; AGENTS.md "BatchOracle parallelizes populations only"). This raises the audit stakes: a wholesale switch to nextest without identifying thread-unsafe tests risks flakiness on the long pole — the very tests that must be reliable for ASHRAE 140 / energy-conservation / h_tr_em / surrogate-drift gates to hold their tolerance bands.

---

## Decision

**Adopt `cargo nextest run --workspace --all-targets` as the test runner in the `Rust Tests & Linting` matrix, with audit-first wholesale switch rollout.**

Concretely, in the implementation PR:

1. Add `.config/nextest.toml` at the repo root with per-binary `concurrency = N` overrides (initially all `N = 2`, matching the GH runner vCPU count; `concurrency = 1` for any binary the audit flags).
2. Replace the seven-line `cargo test ...` block in `.github/workflows/rust-tests.yml::test` with a single invocation:

   ```yaml
   - name: Run tests
     run: |
       cargo nextest run --workspace --all-targets \
         --features ${{ matrix.feature_set.nextest_flags }} \
         --test-threads=2 \
         --no-fail-fast
   ```

   (consolidation is a free side-effect of the nextest switch — `cargo nextest run --workspace --all-targets` covers lib + integration binaries + workspace members in one process tree.)
3. Ensure `cargo-nextest` is installed in the runner via the existing `.github/actions/setup-rust-env` composite action (add a `cargo install cargo-nextest --locked` step gated by a `nextest: 'true'` input, default `true`).
4. Add `docs/ci/nextest-rollout.md` (the audit runbook from Audit-3) so a future maintainer can re-run the audit after a Rayon / major-deps bump.
5. No change to `release_gates.yaml::ci.required_checks`. No change to ASHRAE 140 / energy-conservation / h_tr_em / surrogate-drift tolerance bands.

The audit-first wholesale switch (not shadow mode) is justified by three facts:

- The project already has triple-redundant physics guards (ASHRAE 140 strict energy gate #1333, energy-conservation gate #1295, surrogate drift tolerance gate #1784, h_tr_em regression gate #3154). A flake introduced by nextest parallelism will be caught at PR time by at least one of these.
- Public OSS has unlimited GH minutes — there is no shadow-mode cost saving to chase.
- The audit (empirical + manual, ~30 min CI + ~30 min reading) is the *real* safety mechanism. A multi-week shadow-mode observation window is overhead, not safety.

---

## Plan

### Step 1 — Audit (must complete before merge)

1. **Empirical (lib suite only, where the 54.80s lives):**
   - On a single PR runner, run `cargo nextest run --lib --test-threads=2` **5 consecutive times** with `actions/cache` disabled for `target/` (force cold compile so each run is independent).
   - Record results in a scratch file. Any deadlock, panic, or non-deterministic ordering failure → audit item.
2. **Manual (6 integration binaries, each <70 tests total):**
   - `cargo test --test surface_flux_provider_isolation`
   - `cargo test --test regression_exterior_film_unification`
   - `cargo test --test validation_empirical_harness`
   - `cargo test --test hvac_bestest`
   - `cargo test -p fluxion-behavior --test lighting_occupancy_integration`
   - `cargo test -p fluxion-twin --tests`
   - For each, read the test source for known anti-patterns:
     - Direct `std::env::set_var` / `std::env::set_current_dir` (and reads of mutable env vars mid-test)
     - `RAYON_NUM_THREADS` reads or `rayon::ThreadPoolBuilder` global reconfigurations
     - Shared `lazy_static` / `OnceLock` / `static mut` state mutated by tests
     - `tempfile::TempDir` reuse across tests in the same binary
3. **Document each flagged binary** in `.config/nextest.toml`:

   ```toml
   [[test]]
   name = "binary_name"
   concurrency = 1

   [[test]]
   name = "specific_test_that_races"
   concurrency = 1
   ```

### Step 2 — Implementation PR

Single PR (Issue #3366) that lands:

- `.config/nextest.toml` (new) with the per-binary overrides from Step 1.
- `.github/actions/setup-rust-env/action.yml` — add an opt-in `nextest: 'true'` input that installs `cargo-nextest` via `cargo install cargo-nextest --locked`.
- `.github/workflows/rust-tests.yml::test` — replace the 7-line `cargo test` block with a single `cargo nextest run --workspace --all-targets --features X --test-threads=2` invocation per matrix entry.
- `docs/ci/nextest-rollout.md` — the audit runbook (Audit-3 deliverable).
- `docs/doc-inventory.md` — auto-updated by `python3 scripts/generate_doc_inventory.py` per AGENTS.md.
- This ADR (already accepted) and ADR-0015 (the coordinated concurrency change) referenced from the PR description.

### Step 3 — Post-merge watch (1 week)

- Monitor `gh run list --repo anchapin/fluxion --workflow "Rust Tests & Linting" --event pull_request --json conclusion` for any unexpected `failure` outcomes.
- Compare PR feedback wall-clock median against the 2026-09 baseline (~38 min). Target: ≤15 min median.
- If any binary produces a real race that the audit missed, add the per-binary override in a follow-up PR; do not relax the ASHRAE 140 / energy-conservation / h_tr_em tolerance bands.

### Step 4 — Deferred (out of scope for this PR)

- Path-filter the matrix on docs-only PRs (saves ~38 min for that PR shape; ~5% of typical PR volume).
- Move `Memory Budget (Issue #2384)` (8-min `cargo test --release --features multi-zone`) out of `Rust Tests & Linting` into a separate slow-poll workflow.

---

## Consequences

### Positive

- Median PR feedback wall-clock for `Rust Tests & Linting` drops from ~38 min (2026-09 baseline) to ≤15 min — a 2.5× improvement on the developer hot path.
- The 7-line `cargo test` block is consolidated into a single `cargo nextest run --workspace --all-targets` invocation, amortizing runner setup overhead across all test binaries.
- Per-binary `concurrency` overrides in `.config/nextest.toml` give a future maintainer a clear lever to gate any future non-thread-safe test without code changes.
- The audit runbook (`docs/ci/nextest-rollout.md`) is reusable: any future Rayon / `tokio` / `parking_lot` major-version bump can re-run the empirical pass cheaply.

### Negative

- One PR carries a real risk of new test flakes. Mitigated by: (a) the audit-first model, (b) the per-binary override lever, (c) the existing triple-redundant physics guards catching any tolerance impact.
- Adding `cargo install cargo-nextest --locked` adds ~30-60s of job setup on first install (cache-miss); subsequent runs hit the GH `cargo` binary cache.
- The 30-min manual audit is real reviewer time on the implementation PR.

### Neutral

- `cargo test` is no longer the canonical runner for the PR feedback loop; it remains the canonical runner for local development (per AGENTS.md "Commands That Are Easy to Guess Wrong") and for `cargo test --profile ci` discipline.
- `.config/nextest.toml` becomes a CI-side configuration file that diverges from local developer practice (devs still use `cargo test`). A `CONTRIBUTING.md` note documenting the dev-vs-CI runner difference is appropriate but not strictly required.
- The 1-week post-merge watch window is a discipline cost, not a code cost.

---

## References

- Issue #3366 — parent tracking issue ("Streamline CI: cargo-nextest rollout + per-head-sha concurrency grouping").
- ADR-0015 — coordinated `concurrency.group` key change (per-`head_sha` for PR events).
- ADR #2769 — `Parallelize the analytical path in BatchOracle::evaluate_population` — the Rayon pattern that makes the audit non-trivial.
- `release_gates.yaml::ci.required_checks` — canonical required-check list; this ADR ships with no change to that file.
- `AGENTS.md` §"Commands That Are Easy to Guess Wrong" — `cargo test --profile ci` discipline preserved.
- `.github/actions/setup-rust-env/action.yml` — composite action that owns the sccache + cargo registry cache; extended with `nextest: 'true'` input in this PR.
- `.github/workflows/rust-tests.yml::test` — matrix job definition; the 7-line `cargo test` block is replaced here.
- `gh run list --repo anchapin/fluxion` — 2026-09-04 to 2026-09-06 baseline runs used to derive the acceptance criteria.
- `RULES.md` §"Physics and Validation Guardrails" — no parameter tuning, no baseline relaxation; this ADR does not violate either.
- `copilot-instructions.md` lines 254, 423 — existing `--test-threads=N` usage (diagnostic isolation only); orthogonal to this ADR.