# Perf Lane (Nightly TIMING) — Runbook

**Issue:** #3957 (triage: #3952)  
**Date:** 2026-09-26  
**Status:** ACTIVE — advisory nightly lane (ADR-0016 Lane 3). NOT a required check; not in `release_gates.yaml::ci.workflow_index` (scheduled-only workflows cannot block PRs).  
**Workflow:** `.github/workflows/perf_lane.yml`  
**Schedule:** `0 8 * * *` UTC daily + `workflow_dispatch`

## Summary

The nightly release-mode perf lane for the TIMING test cohort — the 8
`tests/all_tests/` modules whose assertions are absolute
throughput/latency thresholds calibrated for release-mode measurement.
Seven of the cohort's tests are `#[ignore]`-quarantined out of the debug
lanes (#3952 triage: they fail deterministically in debug builds even in
isolation); this lane is their designated measurement home. See
`tests/QUARANTINE.md` §"TIMING suites" for the per-test disposition.

## Purpose

Debug-mode absolute thresholds are environment roulette (GH 2-vCPU
runners vs dev boxes) and carry no stable regression signal. Per the #3952
guardrails, thresholds are **never relaxed to compensate for
environment** — instead, timing assertions live in a lane that measures
in the mode the thresholds were written for (release), on a fixed runner
class (`ubuntu-24.04` pinned, never `ubuntu-latest`).

## What runs

`cargo nextest run --release -p fluxion --test all_tests` with:

- `-E 'test(<module>::) | ...'` over the 8 TIMING modules:
  `performance_regression_test`, `quantum_topology_stress`,
  `throughput_benchmark`, `api_concurrent_throughput`,
  `hybrid_perf_regression`, `performance_ci_test`,
  `test_batch_oracle_throughput`, `security_rate_limit`
- `--run-ignored=all` — ignore-stripping: the 7 `#[ignore]`-quarantined
  TIMING tests execute here while staying quarantined in the debug lanes.
  (No attribute churn; precedent: `performance_dashboard.yml` runs
  `test_performance_regression` via `-- --ignored`.)
- `--test-threads=2` (ADR-0014; matches the #3952 evidence runs)
- `--no-fail-fast` so every module reports

The 2 contention-sensitive tests
(`api_concurrent_throughput::concurrent_throughput_smoke`,
`security_rate_limit::rate_limiter_lru_cap_respected_under_concurrent_cold_flood`)
run with the `.config/nextest.toml` `retries = 1` override (default
profile) — they are load-sensitive, not broken.

## Runner class

`ubuntu-24.04` pinned in the workflow. Do NOT substitute
`vars.FLUXION_LINUX_RUNNER` or `ubuntu-latest` here: changing the runner
class silently changes the measurement class the thresholds are
calibrated against, which is exactly the environment-roulette failure
mode this lane exists to avoid.

## Reading results

- The lane is **informational**: it never blocks a PR. A red nightly
  means "investigate", not "revert".
- Triage order for a red run: (1) check whether the failure is one of
  the 7 ignored-in-debug tests now failing in release — that is a real
  regression signal; (2) compare against the previous night's artifact
  (`perf-lane-results-<run_id>`, 30-day retention) to distinguish a step
  change from noise; (3) for `test_performance_regression`, check the
  `_meta.enforcement` of `tests/perf_baseline.json` — `report-only`
  baselines warn, `hard-gate` baselines panic.
- Do NOT "fix" a red lane by relaxing a threshold. Per #3952: an
  unstable test is flagged for follow-up (new issue), not re-thresholded.

## Threshold provenance and re-baselining record (2026-09-26)

**Margin policy.** Every absolute threshold is evaluated against the
median of ≥3 isolated release-mode runs on the reference environment
below. Thresholds are **never relaxed to make an environment pass**. A
test whose median sits within 10% of its threshold (margin < 1.1×) is
flagged **marginal** for follow-up — it stays in the lane, keeps its
threshold, and the nightly trend on the pinned runner decides whether
the threshold or the test needs work. A test that fails spuriously only
under sibling-load contention (parallel `cargo test`) but passes
stably in isolation is documented as contention-sensitive, not
re-thresholded: the lane's `--test-threads=2` bounds that load.

**Reference environment** (this re-baselining; `tests/perf_baseline.json`
`_meta.machine` carries the machine-readable subset):

- CPU: AMD EPYC 9D25 126-Core Processor (2 vCPUs visible in this VM)
- RAM: 7 GB
- OS: Ubuntu 24.04.5 LTS
- rustc: 1.98.0 (stable)
- Date: 2026-09-26
- Command: `python3 scripts/generate_perf_baseline.py tests/perf_baseline.json 7`
  (median of 7 runs; same harness CI uses)

**Measured release-mode values vs thresholds** (isolated runs,
`-- --ignored --exact`, 2026-09-26; VM: 2 vCPU, 7 GB RAM):

| Test | Threshold (release) | Measured | Median | Margin | Verdict |
|------|--------------------|----------|--------|--------|---------|
| `test_batch_oracle_throughput_1000` | ≥ 150 cfg/s | 147.9 (miss), 150.6, 162.7 | 150.6 | 1.00× | **MARGINAL — flagged** |
| `test_batch_oracle_throughput_100` | ≥ 50 cfg/s | 185.4, 174.3, 166.6 | 174.3 | 3.49× | OK |
| `test_throughput_analytical_1000_configs_sec` | ≥ 100 cfg/s | 174.6, 177.0, 161.3 | 174.6 | 1.75× | OK (stable) |
| `test_multi_zone_throughput` | ≥ 10 cfg/s | 33.33 | 33.33 | 3.33× | OK |
| `test_performance_regression` | within 5% of `tests/perf_baseline.json` | ✓ "No regression detected" vs 153.0 baseline | — (relative) | — | OK |
| `test_performance_smoke_test` | > 150 cfg/s, < 10 ms/cfg | 132 (miss), 171, 170; lat 7.57/5.86/5.89 ms | 170 / 5.89 ms | 1.13× / 1.70× | OK (one noisy miss) |
| `stress_20_zone_qubo_encoding_performance_k16` | max < 5000 µs, mean < 4000 µs | 3× pass (assertions internal, no printed values) | — | — | OK |

**Contention note.** Running all 7 together under parallel `cargo
test` (default thread count) produced two spurious misses
(`test_throughput_analytical_1000_configs_sec` at 80.4 cfg/s,
`test_batch_oracle_throughput_1000` at 144.9 cfg/s) that both pass in
isolation — sibling-load contention, not regressions. The lane's
`--test-threads=2` bounds this, but the two 150 cfg/s-gated tests are
inherently tight on 2-vCPU-class hardware: `test_batch_oracle_throughput_1000`
straddles its gate even in isolation here (147.9/150.6/162.7). Per the
#3952 guardrails the 150 gate is **not** relaxed; the test is flagged
marginal and the nightly trend on the pinned `ubuntu-24.04` runner
(4 vCPU, faster than this VM) is the authority on whether the
threshold or the test needs work. The 150 cfg/s floor was originally
calibrated on GH `ubuntu` runners (see `performance_dashboard.yml`
summary), not on 2-vCPU containers.

**Baseline regeneration.** `tests/perf_baseline.json` was regenerated with
`scripts/generate_perf_baseline.py` (median-of-7, `report-only`
enforcement, `runner_class: dev-local` — correct for a dev-machine
baseline; pass `--hard-gate` only when generating on the same
`ubuntu-24.04` runner class the lane uses). This also repaired the
tracked file, which had a stray Rust `use` line prepended to the JSON
(unparseable → `test_performance_regression` fail-loud panics). The
generator script's harness command was fixed in the same change
(post-#3764 the standalone `performance_regression_test` binary no
longer exists; it now targets `--test all_tests ... -- --ignored`).

## Operator notes

- Manual run: `gh workflow run perf_lane.yml --repo anchapin/fluxion --ref develop`
  (or the Actions tab → "Perf Lane (Nightly TIMING)" → Run workflow).
- To re-baseline locally: `python3 scripts/generate_perf_baseline.py
  tests/perf_baseline.json 7` from the repo root (requires a release
  build; ~10 min on a 2-vCPU box). Commit the regenerated
  `tests/perf_baseline.json` — it is a tracked input to
  `test_performance_regression`.
- The lane does not touch `release_gates.yaml`: it adds no required
  check. If a future change promotes any perf-lane job to required,
  `scripts/check_required_checks_sync.py` will fail until the
  `workflow_index` entry is added (exact `jobs.<id>.name` match).
