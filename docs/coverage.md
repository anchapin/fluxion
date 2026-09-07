# Code Coverage Tracking

Tracks line and branch coverage across the four ARCHITECTURE.md critical physics paths.
Runs `cargo-llvm-cov` in CI (`.github/workflows/code-coverage.yml`) on every PR and `develop` push.
Coverage is bucketed per critical path by `scripts/coverage_critical_paths.py` and enforced via a one-way ratchet gate.
Policy: enforced ratchet + min_branch_floor today; v1.3 targets (80%/85%/75%) aspirational and reported per-run (see §Targets — enforced vs aspirational, Issue #3401).
Related: release_gates.yaml (required check), validation/coverage_baseline.json (ratchet floor), docs/KNOWN_ISSUES.md.

## Overview

The Code Coverage Gate (Issue #1932) replaces the previous tarpaulin-based
informational job with a cargo-llvm-cov pipeline that:

1. **Collects** line + branch coverage for the library crate (`--lib
   --features wiring-tracing`), matching the `docs/CONTRIBUTING.md`
   "Coverage Measurement" workflow.
2. **Buckets** the results into the four critical physics paths defined
   in `ARCHITECTURE.md`:

   | Path | Files |
   |------|-------|
   | Weather → Solar | `fluxion-core/src/weather/**`, `src/sim/solar.rs`, `src/sim/solar_gain_distribution.rs` |
   | Weather → Ventilation | `fluxion-core/src/weather/**`, `src/sim/ventilation.rs` |
   | Conduction → Zone Balance | `src/physics/**`, `src/sim/thermal_model*.rs`, `src/sim/thermal_model_data/**`, `src/sim/per_surface_conduction.rs` |
   | HVAC → Zone Balance | `src/sim/hvac/**`, `src/sim/thermal_model_solvers.rs`, `src/sim/hvac_controller.rs` |

   Files may contribute to more than one path (e.g. `weather/` is on both
   the solar and ventilation paths) — this mirrors the real data flow.

3. **Enforces** a one-way ratchet: any path whose baseline
   (`validation/coverage_baseline.json`) has a non-zero value will fail the
   build if its line coverage drops more than 1% relative to the baseline.
   The baseline only ever moves upward.

## Reproducing a coverage run locally

```bash
# Install once
rustup component add llvm-tools
cargo install cargo-llvm-cov

# Match what CI runs
cargo llvm-cov --lib --features wiring-tracing \
  --lcov --output-path target/llvm-cov/lcov.info \
  --ignore-filename-pattern '/tests/|/benches/|/target/|/fluxion-mcp/'

# Print the per-critical-path table
python3 scripts/coverage_critical_paths.py --lcov target/llvm-cov/lcov.info

# Check the gate (non-zero exit = regression)
python3 scripts/coverage_critical_paths.py \
  --lcov target/llvm-cov/lcov.info \
  --baseline validation/coverage_baseline.json \
  --gate
```

## Activating the ratchet

The committed baseline starts with all values at `0.0`, which means
*unenforced*. The gate passes regardless of coverage until a maintainer
records real numbers:

```bash
# After a green develop CI run, download the lcov.info artifact then:
python3 scripts/coverage_baseline.py --update \
  --lcov target/llvm-cov/lcov.info \
  --baseline validation/coverage_baseline.json

git add validation/coverage_baseline.json
git commit -m "ci(coverage): record baseline for #1932 ratchet gate"
```

Once committed, every subsequent PR is held to the recorded floor. Re-run
the command after coverage improvements to bump the ratchet upward.

## Targets — enforced vs aspirational (Issue #3401)

The table below distinguishes what the CI gate **enforces today** from what it **reports**. The v1.3 targets (80% / 85% / 75%) are aspirational and REPORTED per-run (`v1_3_target_branch` in `scripts/coverage_critical_paths.py`) — they do not fail CI while actual metrics sit materially below them (current baseline: 79.80% overall; critical-path branch coverage 61–68%). Each becomes a hard release gate only once the measured metrics approach it, per the promotion criterion documented in the script (`# becomes a hard release gate once the metrics approach it`).

| Metric | v1.3 target (aspirational) | Enforced today |
|--------|---------------------------|----------------|
| Overall line coverage | >80% | Ratchet: no regression below `validation/coverage_baseline.json` |
| Per-critical-path line coverage | >85% | Ratchet: no regression below baseline |
| Per-critical-path branch coverage | >75% | Ratchet (#2533) + absolute `min_branch_floor` hard floor (#2710) |

Promotion path (per Issue #3401): when a metric's measured value is within 2 percentage points of its target, flip the corresponding baseline entry from reported to enforced — one config change in `validation/coverage_baseline.json`, no script change needed. Until then, a materially under-target run that holds the ratchet is green **by design**, and the printed gap is the tracking signal.

## v1.3 branch-coverage floor and target (#2710)

The regression ratchet only prevents coverage from *dropping* — it never
drives it *up*, so a 60–68% branch gap on the critical physics paths
could persist forever. Issue #2710 adds two independent per-path policy
levers to `validation/coverage_baseline.json`:

- **`min_branch_floor`** — an absolute hard floor. The gate FAILS when
  current branch coverage is below it, independent of the ratchet
  baseline. Set at or slightly below current values so the gate passes
  today; maintainers raise it over time. `0.0` = unenforced.
- **`v1_3_target_branch`** — the v1.3 release target (75% branch). The
  gate REPORTS the gap every run but does not yet fail; it becomes a
  hard release gate once the metrics approach it.

Together they ensure the branch gap is tracked and pressured toward a
goal rather than silently locked in by the one-way ratchet.

## Related

- [CONTRIBUTING.md](CONTRIBUTING.md) §Coverage Measurement — documented workflow
- [release_gates.yaml](../release_gates.yaml) — registers the gate as a required check
- [KNOWN_ISSUES.md](KNOWN_ISSUES.md) — open limitations, including baseline-collection status
- [tests/physics/README.md](../tests/physics/README.md) — physics test catalog
