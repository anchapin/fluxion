# `cargo nextest` rollout runbook — Issue #3366 / ADR-0014

> **Summary 1/7:** `Rust Tests & Linting` switched from `cargo test` to `cargo nextest run --workspace --all-targets --test-threads=2` (ADR-0014, Issue #3366); the 3,922 lib tests now run with `--test-threads=2` per binary, replacing the single-threaded `cargo test --lib` long pole that ran ~54.80s sequentially.
> **Summary 2/7:** The audit (Step 1, this document) is the *real* safety mechanism: 5 consecutive `cargo nextest run --lib --test-threads=2` runs on a cold target, plus manual source review of the 6 small integration binaries, produced **zero** deadlock / panic / non-deterministic ordering findings.
> **Summary 3/7:** Default `concurrency = 2` in `.config/nextest.toml` (matching the GH free-runner vCPU count). No per-binary overrides required at merge time. The file is a CI-side configuration; local developer workflow continues to use `cargo test` per AGENTS.md §"Commands That Are Easy to Guess Wrong".
> **Summary 4/7:** Re-audit triggers: Rayon / `tokio` / major-deps bumps, any new test binary that touches `std::env::set_var`, `rayon::ThreadPoolBuilder`, `lazy_static` / `OnceLock` / `static mut` globals, or `TempDir` reuse across tests.
> **Summary 5/7:** Empirical 5× rerun baseline (2026-09-06, `CARGO_TARGET_DIR=/tmp/nextest-audit-target`): medians 55.38s–58.86s, 3,922 tests, all PASSED.
> **Summary 6/7:** Acceptance per ADR-0014 §"Consequences": median PR feedback wall-clock for `Rust Tests & Linting` ≤ 15 min (from ~38 min baseline). Post-merge 1-week watch window is the discipline cost — see Step 4.
> **Summary 7/7:** Per-binary concurrency overrides in `.config/nextest.toml` are the recommended remediation for any race surface post-merge. Do **not** relax any of: ASHRAE 140 tolerance bands, energy-conservation invariant, `h_tr_em` regression gate, surrogate drift tolerance gate, `fluxion-grid` integration tests (RULES.md §"Physics and Validation Guardrails").

- **Status:** Audit runbook — read after any Rayon / `tokio` / major-deps bump
- **Issue:** [#3366](https://github.com/anchapin/fluxion/issues/3366)
- **ADRs:** [ADR-0014](../adr/0014-cargo-nextest-test-runner-adoption.md) (this PR's first leg), [ADR-0015](../adr/0015-pr-concurrency-per-head-sha.md) (second leg)
- **Companion files:** [`.config/nextest.toml`](../../.config/nextest.toml), [`scripts/update_concurrency_keys.py`](../../scripts/update_concurrency_keys.py), [`scripts/check_concurrency_keys.py`](../../scripts/check_concurrency_keys.py)

---

## Context

ADR-0014 §"Context" documents the 2026-09-04 to 2026-09-06 PR-fleet data showing `Rust Tests & Linting` as the dominant PR feedback bottleneck:

| Metric | 2026-09 baseline | Source |
|---|---|---|
| Wall-clock for full successful run | **37m 54s** (run db=34008638984) | `gh run view` |
| `Test (ubuntu-latest, multi-zone)` total | 347s | `gh api /actions/jobs/...` |
| `Run tests` step within that job | 305s (88.3% of job) | step breakdown |
| `cargo test --lib` (3,922 tests, single-threaded) | **54.80s** | log marker |
| Successful PR runs of this workflow | 1 of 6 (~17%) | run list |

`sccache` is operating at 99.78% Rust hit rate; the bottleneck is test execution, not compilation. The wholesale switch to `cargo nextest run --workspace --all-targets --test-threads=2` consolidates the 7 separate `cargo test` invocations per matrix entry into one process tree and runs each binary's tests in 2 threads (matching GH free-runner vCPU count).

The audit-first model (not shadow-mode observation) is justified by three facts (ADR-0014 §"Decision"):

1. The project has triple-redundant physics guards (ASHRAE 140 strict energy gate #1333, energy-conservation gate #1295, surrogate drift tolerance gate #1784, `h_tr_em` regression gate #3154) that catch any tolerance impact at PR time.
2. Public OSS has unlimited GH minutes — no shadow-mode cost saving to chase.
3. The audit (this document, ~30 min CI + ~30 min reading) is the *real* safety mechanism.

---

## Step 1 — Audit (completed 2026-09-06)

### Step 1a — Empirical (lib suite, where the 54.80s lives)

Run on this branch, on a fresh `CARGO_TARGET_DIR` (force cold compile so each run is independent):

```bash
mkdir -p /tmp/nextest-audit-target
export CARGO_TARGET_DIR=/tmp/nextest-audit-target
cd /home/alex/Projects/fluxion

for i in 1 2 3 4 5; do
  cargo nextest run --lib --test-threads=2 --no-fail-fast 2>&1 | tee /tmp/audit-run-$i.log
  echo "--- run $i done ---"
done
```

**Result (2026-09-06):**

| Run | Wall-clock | Tests | Outcome |
|---|---|---|---|
| 1 | 58.86s | 3,922 (5 skipped) | ALL PASSED |
| 2 | 55.49s | 3,922 (5 skipped) | ALL PASSED |
| 3 | 55.38s | 3,922 (5 skipped) | ALL PASSED |
| 4 | 56.10s | 3,922 (5 skipped) | ALL PASSED |
| 5 | 56.23s | 3,922 (5 skipped) | ALL PASSED |

**Median: 56.10s** (slightly higher than the 54.80s single-threaded baseline because of `cargo nextest`'s own startup overhead amortised across the test set; the wall-clock improvement on the matrix entry comes from parallelising the `cargo test` invocations and consolidating them into a single process tree).

**Findings:** None. No deadlock, no panic, no non-deterministic ordering failure observed across 5 runs.

### Step 1b — Manual source review (6 small integration binaries)

For each binary, grep for known thread-unsafe anti-patterns and read the source for shared-state mutations across tests:

| Anti-pattern | Why it races under `--test-threads=2` |
|---|---|
| Direct `std::env::set_var` / `std::env::set_current_dir` (mid-test) | Process-local but visible across the test binary's process (nextest re-uses the binary process per test thread) |
| `RAYON_NUM_THREADS` reads / `rayon::ThreadPoolBuilder` global reconfigurations | Changes the global Rayon pool, visible across all tests in the binary |
| Shared `lazy_static` / `OnceLock` / `static mut` state mutated by tests | Read-modify-write race across threads |
| `tempfile::TempDir` reuse across tests in the same binary | Path collisions; some tests depend on a specific path existing |

| Binary | Anti-patterns found | Verdict |
|---|---|---|
| `tests/surface_flux_provider_isolation.rs` (1,138 LoC) | None — pure physics trait-object tests; each test builds its own `WallSpec` locally | ✅ Safe at default concurrency |
| `tests/regression_exterior_film_unification.rs` (461 LoC) | None — only reads `src/` files via `std::fs::read_to_string`; no mutable global state | ✅ Safe at default concurrency |
| `tests/validation_empirical_harness.rs` → `tests/validation/empirical.rs` (244 LoC) | None — pure in-memory `EmpiricalCaseRegistry`; tests instantiate their own registry | ✅ Safe at default concurrency |
| `tests/hvac_bestest.rs` → `tests/hvac_bestest_validation.rs` (178 LoC) | None — pure computation over `run_hvac_bestest()` results | ✅ Safe at default concurrency |
| `fluxion-behavior/tests/lighting_occupancy_integration.rs` (852 LoC) | None — deterministic `MASTER_SEED` and `deterministic_state()` calls; tests build their own `MarkovOccupancyGenerator` / `LightingModel` instances | ✅ Safe at default concurrency |
| `crates/fluxion-twin/tests/*.rs` (4 files, 1,094 LoC total) | None — UKF tests use `StdRng::seed_from_u64(NOISE_SEED)` with fixed seeds; MQTT tests use deterministic synthetic payloads | ✅ Safe at default concurrency |

**Audit conclusion:** all 6 audit-flagged binaries are safe at the default `concurrency = 2`. No per-binary overrides required at merge time.

### Step 1c — `.config/nextest.toml` layout

```toml
# Default: 2 threads per test binary (matches GH free runner vCPU count)
concurrency = 2

# Per-binary overrides (Audit findings) — none at merge time.
# Add a block here if a future race surfaces:
#   [[test]]
#   name = "flagged_binary_name"
#   concurrency = 1
```

The default `concurrency = 2` matches the GH free-runner vCPU count. The `[[test]]` block template is preserved in the file's leading comment for any future re-audit.

---

## Step 2 — Implementation (this PR's first leg)

Single PR (Issue #3366) lands:

- [`.config/nextest.toml`](../../.config/nextest.toml) (new) — default `concurrency = 2`, no per-binary overrides at merge.
- [`.github/actions/setup-rust-env/action.yml`](../../.github/actions/setup-rust-env/action.yml) — added opt-in `nextest: 'true'` input (default `'true'`); installs `cargo-nextest` via `taiki-e/install-action@<pinned-SHA>` with the same 3-attempt retry pattern as `dtolnay/rust-toolchain` and `mozilla-actions/sccache-action`.
- [`.github/workflows/rust-tests.yml`](../../.github/workflows/rust-tests.yml) — replaced the 7-line `cargo test ...` block with a single `cargo nextest run --workspace --all-targets --features ${{ matrix.feature_set.nextest_flags }} --test-threads=2 --no-fail-fast` invocation.
- [`docs/ci/nextest-rollout.md`](./nextest-rollout.md) (this document) — audit runbook.
- [`docs/doc-inventory.md`](../doc-inventory.md) — auto-updated by `python3 scripts/generate_doc_inventory.py`.

The matrix `feature_set` was extended to add `nextest_flags` (parallel to the legacy `flags` field, which is retained for the main-only `test-full` and `build-release` jobs further down `rust-tests.yml` that still use `cargo test` / `cargo build`).

---

## Step 3 — Post-merge watch (1 week)

- Monitor `gh run list --repo anchapin/fluxion --workflow "Rust Tests & Linting" --event pull_request --json conclusion` for any unexpected `failure` outcomes.
- Compare PR feedback wall-clock median against the 2026-09 baseline (~38 min). Target: ≤15 min median.
- If any binary produces a real race that the audit missed, add a per-binary override in `.config/nextest.toml` in a follow-up PR; do **not** relax the ASHRAE 140 / energy-conservation / `h_tr_em` / surrogate-drift tolerance bands.

The 1-week watch window runs in parallel to ADR-0015's `concurrency.group` per-`head_sha` rollout — both halves of Issue #3366 share the same acceptance criteria envelope.

---

## Step 4 — When to re-audit

Re-run Step 1 (at minimum Step 1a empirical) after any of:

| Trigger | Why |
|---|---|
| Rayon major version bump (`rayon 1.10 → 2.x`) | New global pool semantics; existing `BatchOracle` parallelism may surface new races |
| `tokio` major version bump (`tokio 1.40 → 2.x`) | Runtime / test-harness integration changed |
| `parking_lot` major version bump | Lock primitives changed |
| New test binary added to `Rust Tests & Linting` | Run Step 1b manual review on the new binary |
| Any change to `std::env::set_var` / `rayon::ThreadPoolBuilder` / `lazy_static` / `OnceLock` / `static mut` in test code | Direct thread-safety signal |
| A flake report: `gh run list --workflow "Rust Tests & Linting" --json conclusion` shows ≥ 2 `failure` outcomes with the same test in 24h | Audit failed; tighten or override |

To re-audit Step 1a:

```bash
rm -rf /tmp/nextest-audit-target  # force cold target
export CARGO_TARGET_DIR=/tmp/nextest-audit-target
cd /home/alex/Projects/fluxion
for i in 1 2 3 4 5; do
  cargo nextest run --lib --test-threads=2 --no-fail-fast 2>&1 | tee /tmp/audit-re-run-$i.log
done
```

Any deadlock, panic, or non-deterministic ordering failure → add a per-binary override to `.config/nextest.toml`:

```toml
[[test]]
name = "flagged_binary_name"
concurrency = 1
```

---

## Step 5 — Out of scope (deferred)

- Path-filter the matrix on docs-only PRs (saves ~38 min for that PR shape; ~5% of typical PR volume). Tracked by #3367.
- Move `Memory Budget (Issue #2384)` (8-min `cargo test --release --features multi-zone`) out of `Rust Tests & Linting` into a separate slow-poll workflow. Tracked by #3368.
- WASM / Python / Node binding workflow consolidations — each binding has its own test surface and is out of scope for this PR.

---

## References

- **Issue:** #3366 — parent tracking issue ("Streamline CI: cargo-nextest rollout + per-head-sha concurrency grouping")
- **ADR-0014** — coordinated `cargo nextest` test-runner adoption (this document's source-of-truth)
- **ADR-0015** — coordinated `concurrency.group` key change (per-`head_sha` for PR events)
- **AGENTS.md** §"Commands That Are Easy to Guess Wrong" — `cargo test --profile ci` discipline preserved; local developers continue to use `cargo test`, CI uses `cargo nextest`
- **RULES.md** §"Physics and Validation Guardrails" — no parameter tuning, no baseline relaxation; this PR does not violate either
- **`.github/actions/setup-rust-env/action.yml`** — composite action that owns the sccache + cargo registry cache; extended with `nextest: 'true'` input in this PR
- **`.github/workflows/rust-tests.yml::test`** — matrix job definition; the 7-line `cargo test` block is replaced here
- **`gh run list --repoanchapin/fluxion`** — 2026-09-04 to 2026-09-06 baseline runs used to derive the acceptance criteria
- **`scripts/check_concurrency_keys.py`** — companion CI guard that verifies the per-`head_sha` pattern in every workflow
- **`scripts/check_required_checks_sync.py`** — existing CI guard for required-check / workflow-name drift (extended to verify the ADR-0015 shape via `check_concurrency_keys.py`)
- **`copilot-instructions.md` lines 254, 423** — existing `--test-threads=N` usage (diagnostic isolation only); orthogonal to this rollout