# Python-bindings CI coverage runbook — Issue #3755

> **Summary 1/7:** The `Rust Tests & Linting` workflow (`rust-tests.yml`) now compiles `--features python-bindings` and runs the `python::` lib tests on every PR via a new step in the existing `test` job — "Python bindings feature tests (Issue #3755)".
> **Summary 2/7:** Root cause of the original breakage: API drift between `src/python/bindings.rs` call sites and their targets — the #1773 annealer-trait addition (`max_variables` / `hardware_constraints`) and an `hvac` field path change — produced 10 compile errors on `develop` that CI never saw.
> **Summary 3/7:** The blind spot existed because no PR-path workflow built the feature set; the maturin/pytest matrices in `python-bindings.yml` run only on `main` pushes and nightly (issue #3567), so drift could land on `develop` unmerged-detected for weeks.
> **Summary 4/7:** The mechanical compile fix landed separately as PR #3782 (issue #3749); this runbook documents the CI-side closure: what the gate runs, why it is pinned to one matrix leg (Linux + `no-default`), and how to triage failures.
> **Summary 5/7:** Gate scope: `cargo nextest run -p fluxion --lib --features python-bindings python:: --test-threads=2 --no-fail-fast` (~161 tests). The lib-test compile covers every `#[cfg(feature = "python-bindings")]` module in `src/python/` regardless of the test filter.
> **Summary 6/7:** Environment mirrors the established `python-bindings.yml` pattern: workflow-level `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` (already in `rust-tests.yml`) plus step-level `RUST_MIN_STACK=33554432` (32 MiB build-thread stack for the PyO3 link step).
> **Summary 7/7:** The `python-bindings,dwave` combo is compile-verified locally (`cargo test -p fluxion --lib --features python-bindings,dwave --no-run`) but intentionally NOT wired into this step — issue #3755 acceptance names only `--features python-bindings`; extending coverage to the dwave combo is future scope.

## What the gate is

A step inside the existing `test` job of
[`.github/workflows/rust-tests.yml`](../../.github/workflows/rust-tests.yml),
running on exactly one matrix leg (`runner.os == 'Linux' && matrix.feature_set.name == 'no-default'`)
so the extra cargo build is paid once per PR / main push, not once per
feature-set × OS matrix entry:

```yaml
- name: Python bindings feature tests (Issue #3755)
  if: runner.os == 'Linux' && matrix.feature_set.name == 'no-default'
  env:
    RUST_MIN_STACK: 33554432
  run: |
    cargo nextest run \
      -p fluxion \
      --lib \
      --features python-bindings \
      python:: \
      --test-threads=2 \
      --no-fail-fast
```

The positional filter `python::` (nextest substring filter, same pattern as
the `surface_flux_provider_isolation::` filter in the same job) selects the
feature-gated lib tests; compilation of the `--lib` target itself is the
primary regression signal because every module under `src/python/` is
`#[cfg(feature = "python-bindings")]`-gated. No new required check was
added — the step folds into the already-required `Test (ubuntu-latest,
no-default)` matrix leg, so branch protection and
`release_gates.yaml::ci.required_checks` needed no changes.

## Root cause (issue #3755, acceptance criterion 3)

Discovered by the #3720 wave-1 implementation sub-agent
(auto-improvement-loop session `loop-fluxion-2026-09-13`): the
`--features python-bindings` build of the root crate failed to compile on
unmodified `develop` with 10 errors, none of which any CI workflow could
see because no PR-path job built the feature set (the same class of blind
spot as T1/#3724 for `gauge-solver`):

- `src/python/bindings.rs` — 7x E0061 (argument-count drift between
  call sites and the function at the target) and 1x E0609 (`hvac` field
  moved on the struct path the binding read).
- `src/python/model_bindings/model.rs` — 2x E0609 (`hvac` field path
  change).
- 2x E0046 in `src/quantum/dwave_client.rs` (`MockAnnealer` /
  `GreedyAnnealer` mock impls missing the `max_variables` /
  `hardware_constraints` trait methods added by #1773) — surfaced only
  under `python-bindings,dwave`, fixed on `fix/issue-3720-dwave-missing-token`.

The drift vector was **API drift between the bindings call sites and their
targets**: the #1773 annealer-trait addition and an `hvac` field path
change landed without updating `src/python/` because nothing compiled it.
PR #3782 (issue #3749) mechanically fixed the 10 errors; this gate exists
so the same drift class fails the PR that introduces it.

Impact of the blind spot: no test gated behind `python-bindings` executed
anywhere (including the pre-existing `dwave_client` tests compiled under
the feature umbrella), and the maturin wheel build would have failed.

## Local reproduction and verification

Run before touching `src/python/` or the PyO3 surface (both feature combos
from issue #3755):

```bash
cargo test -p fluxion --lib --features python-bindings --no-run   # compile gate
cargo test -p fluxion --lib --features python-bindings,dwave --no-run
cargo test -p fluxion --lib --features python-bindings python::   # ~161 tests
```

The CI-equivalent invocation is the nextest command above. For workflow
YAML edits, also run the local `act` suite per
[`local-validation.md`](local-validation.md) and the pinning gates
(`scripts/check_action_pinning.py`, `scripts/check_workflow_pin.py`,
`scripts/check_concurrency_keys.py`).

## Relationship to `python-bindings.yml`

[`python-bindings.yml`](../../.github/workflows/python-bindings.yml)
remains the heavyweight matrix: maturin wheel builds (3 OS x 4 Python) and
the PyO3 pytest suite, running on `main` pushes and nightly only (issue
#3567 moved heavy validation off the PR path). This gate is the cheap
PR-path canary: it catches *compile drift* and exercises the Rust-side
`python::` unit tests minutes after the offending PR is opened, instead of
at the next nightly wheel build. The two are complementary; if you change
the PyO3 surface (e.g. `#[pyclass]` / `#[pyfunction]` signatures), also
run `python3 scripts/check_pyi_drift.py` for the `fluxion.pyi` stub.

## Environment variables

| Variable | Where | Why |
|---|---|---|
| `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` | `rust-tests.yml` workflow `env` (pre-existing) | abi3 wheels built against a newer Python run on older interpreters; established repo-wide setting for PyO3 builds. |
| `RUST_MIN_STACK=33554432` | step-level on this gate (32 MiB) | Enlarges the build thread stack for the PyO3 link step; mirrors `python-bindings.yml` (see `docs/FEATURES.md`). |

## When this step fails

1. Read the compile error path — it is almost always bindings-call-site
   drift (E0061 argument count, E0609 field path) after a refactor of the
   underlying Rust API, the exact #3755 pattern.
2. Fix the call sites in `src/python/` (and `fluxion.pyi` if the exported
   surface changed) — do **not** widen the feature gating or relax the
   step.
3. Re-run the three local commands above; check
   [`../KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) before classifying a
   validation-looking failure as gate noise (see also
   [`nextest-rollout.md`](nextest-rollout.md) for runner-semantics
   background).
