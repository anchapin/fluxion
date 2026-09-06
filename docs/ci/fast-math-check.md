# Fast-Math vs IEEE-754 ASHRAE 600/900 Regression Gate — Issue #3326

**Issue:** #3326, #3358  
**Date:** 2026-09-06  
**Status:** REQUIRED — promoted to `release_gates.yaml::ci.required_checks` via the `(GH)` listener pattern (Issue #3358); live `develop` branch-protection activation gates on the 4-week stability window sign-off tracked in Issue #3358 comments.

## Summary

This document is the operator + reviewer guide for the `fast_math_check.yml`
GitHub Actions workflow, the advisory CI gate that runs ASHRAE 140 Case 600
and Case 900 annual simulations twice on the same commit — once with default
features and once with `--features fast-math` — and asserts the two runs
agree within ±0.05% relative on per-case annual heating/cooling totals while
the energy-conservation invariant residual under `fast-math` stays ≤ 1e-5 W.
The job is intentionally separate from `determinism_check.yml` because the
algebraic float methods introduced in #3322 are **non-deterministic by
specification** and would poison the bit-identical cross-platform determinism
contract (issues #1297, #2549). ASHRAE strict gates and cross-platform
determinism CI are unaffected — they keep running default features only.

## Why a separate workflow

The original design draft for #3326 considered adding a step to
`determinism_check.yml`. That workflow's contract is **bit-identical
cross-platform output hashing** with pinned `RUSTFLAGS`; algebraic FP is
non-deterministic by spec and would poison that signal. So the issue body
explicitly forbids that placement:

> Design revision vs. the original draft: do **not** add a step to
> `determinism_check.yml`. That workflow's contract is *bit-identical
> cross-platform output hashing* with pinned `RUSTFLAGS` (issues #1297,
> #2549); algebraic FP is non-deterministic by spec and would poison that
> signal. Use a separate job instead.

The same separation protects the existing ASHRAE strict gates
(`ashrae_140_strict_energy_gate.yml`, `ashrae_validation.yml`, and friends)
— they continue to run under default features and never touch the
`fast-math` build path.

## What the workflow runs

```
              ┌─────────────────────┐    ┌─────────────────────┐
              │ probe-default       │    │ probe-fastmath      │
              │ cargo test --test   │    │ cargo test --test   │
              │ fast_math_probe     │    │ fast_math_probe     │
              │ --release           │    │ --release           │
              │                     │    │ --features fast-math│
              │ (default features   │    │ (algebraic FP via   │
              │  = IEEE-754)        │    │  f*/algebraic_*)    │
              └──────────┬──────────┘    └──────────┬──────────┘
                         │                          │
                         └────────────┬───────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │     compare      │
                            │  ──────────────  │
                            │ • parse both     │
                            │   FAST_MATH_     │
                            │   PROBE_V1 lines │
                            │ • 4 load deltas  │
                            │   within ±0.05%  │
                            │ • residual ≤     │
                            │   1e-5 W (under   │
                            │   fast-math)     │
                            └──────────────────┘
```

The probe is `tests/fast_math_probe.rs` — a dedicated integration-test
binary (auto-discovered by `cargo`) that runs Case 600 + Case 900 with a
14-day warmup + 8760 hourly steps, tracks annual heating/cooling totals
and the `InvariantChecker` max-residual, and emits one machine-parseable
line per run:

```text
FAST_MATH_PROBE_V1|<c600_h_mwh>|<c600_c_mwh>|<c900_h_mwh>|<c900_c_mwh>|<max_residual_w>|<violation_count>|<total_checks>
```

## Assertion contract

| Assertion | Threshold | Source |
|-----------|-----------|--------|
| `case600_heating_mwh` (default ↔ fast-math) | ≤ ±0.05% relative | Issue #3326 acceptance |
| `case600_cooling_mwh` (default ↔ fast-math) | ≤ ±0.05% relative | Issue #3326 acceptance |
| `case900_heating_mwh` (default ↔ fast-math) | ≤ ±0.05% relative | Issue #3326 acceptance |
| `case900_cooling_mwh` (default ↔ fast-math) | ≤ ±0.05% relative | Issue #3326 acceptance |
| `max_residual_w` under fast-math | ≤ 1e-5 W | Issue #3326 acceptance; same machinery as #1295 / `tests/zone_balance_eplus_isolation.rs` |

The residual ceiling is automatically satisfied because the energy-balance
path is **forbidden** from importing `fp_algebraic` (see the do-not-use
list in `src/physics/fp_algebraic.rs` module docs — heat-conduction
solvers, zone-balance / thermal assembly, and the thermal model solvers
are all on the banned list). Any future regression that lifts that
restriction will trip the 1e-5 W ceiling immediately, by construction.

## Why the threshold is 0.05% (not 0.0%)

The `fp_algebraic` helpers route to `f32/f64::algebraic_*` which permit
operand reassociation, contraction, and loop vectorization comparable to
`-ffast-math` (#3322). They are non-deterministic by spec — last-ulp drift
across compiler versions / targets is allowed. A 0.0% tolerance would
trip on every PR; a ±0.05% tolerance absorbs the legitimate last-ulp
noise while still catching any hot-path edit that perturbs the physics
by more than ~1 part in 2000 (e.g. a 1% off-by-one in heat transfer,
a stray `+ 1.0` in a denominator, or a reassociation that flips a sign
in a round-off-sensitive sum). The 1e-5 W residual ceiling adds the
second-axis protection: even if the load totals coincidentally stay
within 0.05%, any kernel that violates energy conservation trips
immediately.

## Trigger schedule

| Trigger | Cron | Notes |
|---------|------|-------|
| `push` to `main`, `develop` | — | Same-commit guarantee |
| `pull_request` to `main`, `develop` | — | Same-commit guarantee |
| `workflow_dispatch` | — | Manual `gh workflow run fast_math_check.yml` |
| `schedule` (nightly) | `0 4 * * *` | 04:00 UTC — off-peak slot, no collision with the other nightly jobs (nightly_validation=00, perf_dashboard=02, pgo=02:30, ashrae_140_validation=02, architecture_drift=03, rust-tests=03:17, loom=03 Sun, gauge=06, required-checks-sync-cron=06, mutation=07, known-issues-stale=09 Mon, rumqttc=10 Mon) |

## Status: required (promoted)

The job **is** now a branch-protection required check on `develop`
as of #3358 (YAML-side promotion). The live `develop` branch-protection
contexts array pickup is the *separate* activation step — gated on the
4-week stability window sign-off that is tracked in the Issue #3358
comments using the #3286 β-soak convention.

Initial advisory state (Issue #3326, PR #3349, 2026-09-03):

> Keep the job advisory initially: do not add it to
> `release_gates.yaml → ci.required_checks`; if/when promoting it, follow
> the required-checks sync discipline (see the #3129 gap and the #3142
> workflow-only rationale in `docs/ci/`).

## Promotion criteria (now met)

Per the #3142 sync-discipline pattern documented in
`docs/ci/branch-protection-strict-mode.md`:

1. ≥ 14 days of green nightly runs on the seeded ASHRAE 600/900 cases
   (no flake on the comparison job or the residual-ceiling check) —
   tracked in the Issue #3358 comments.
2. Confirm no false-positive from the `cargo test` rebuilds — the
   workflow uses a separate cache key per `--features fast-math` so a
   partial rebuild can't serve stale artifacts across modes.
3. Update `release_gates.yaml::ci.workflow_index` with the new check
   name (job name: `"Fast-Math vs IEEE-754 ASHRAE 600/900 Regression
   Gate (GH)"`) — **done in #3358**.
4. Add the check to **both** `ci.required_checks` (for code-changing
   PRs) **and** `ci.required_checks_workflow_only` (for workflow-only
   PRs) — **done in #3358**.
5. Run `python3 scripts/check_required_checks_sync.py` to verify
   drift-free — **passes in #3358** (30 required_check(s), 31
   workflow_index entr(ies), in sync with 46 workflow file(s)).

## Activation step (gated on stability window sign-off)

The YAML side already emits the exact required-check name. To
*activate* the gate on the live `develop` branch-protection, run
once the 4-week stability window closes:

```bash
# Snapshot the current contexts
EXISTING=$(gh api /repos/anchapin/fluxion/branches/develop/protection/required_status_checks \
  | jq -c '.contexts')

# Append the new check name
NEW=$(echo "${EXISTING}" | jq '. + ["Fast-Math vs IEEE-754 ASHRAE 600/900 Regression Gate (GH)"]')

# PATCH preserves existing contexts (PUT would replace them)
gh api --method PATCH /repos/anchapin/fluxion/branches/develop/protection/required_status_checks \
  -H "Content-Type: application/json" \
  --input - <<EOF
{"contexts": ${NEW}}
EOF
```

Verify with `gh api /repos/anchapin/fluxion/branches/develop/protection/required_status_checks`
before and after. The `FLUXION_CHECK_LIVE_PROTECTION=1` invocation of
`scripts/check_required_checks_sync.py` will then pass end-to-end.

For the full operator runbook including a rollback procedure for the
false-positive class, see
[`docs/ci/fast-math-stability-window.md`](fast-math-stability-window.md).

## Self-validation evidence (Issue #3326 acceptance)

Two rounds of self-validation were performed in the worktree before
commit:

1. **Assertion-logic dry-run** — six synthetic scenarios fed through
   the Python verifier with the captured default-mode probe output
   (`4.076183|3.840097|1.128324|2.083407|0.000000e0|0|18192`):

   | # | Scenario | Expected | Got |
   |---|----------|----------|-----|
   | 1 | Identical outputs (default ↔ fast-math) | PASS | PASS |
   | 2 | c600_heating perturbed +0.1% (>0.05% tol) | FAIL | FAIL |
   | 3 | max_residual perturbed to 1e-4 W (>1e-5 ceiling) | FAIL | FAIL |
   | 4 | max_residual exactly at 1e-5 W ceiling | PASS | PASS |
   | 5 | Load delta exactly at 0.05% tolerance | PASS | PASS |
   | 6 | Load delta just over 0.05% tolerance | FAIL | FAIL |

2. **End-to-end perturbation** — a deliberate +0.1% bias was added to
   `tests/fast_math_probe.rs`'s positive heating energy (mirroring a
   hypothetical fast-math hot-path drift), the probe was re-run, and
   the resulting `FAST_MATH_PROBE_V1` line was fed to the same Python
   verifier with the unperturbed default-mode output. The verifier
   flagged `Case 600 heating` (+0.100%) and `Case 900 heating`
   (+0.100%) as FAIL, exactly as a real workflow run would. The bias
   was then reverted; the post-revert probe output is bit-identical to
   the original (`4.076183|3.840097|1.128324|2.083407|0.000000e0|0|18192`),
   confirming the perturbation did not leak into the committed test.

## See Also

- `src/physics/fp_algebraic.rs` — the algebraic-FP helper layer (#3322)
- `.github/workflows/determinism_check.yml` — the bit-identical cross-platform determinism contract that this workflow is **explicitly separate from** (issues #1297, #2549)
- `.github/workflows/ashrae_140_strict_energy_gate.yml` — the ASHRAE strict ±15% annual-energy tolerance gate (#1333)
- `tests/zone_balance_eplus_isolation.rs` — the InvariantChecker machinery the residual ceiling reuses (#1295)
- `docs/ci/branch-protection-strict-mode.md` — the workflow-only required-checks rationale (#3142)
- `release_gates.yaml::ci.required_checks` — where this check will be promoted when ready
- Issue #3322 — the `fast-math` feature flag + `fp_algebraic.rs` helper layer
- Issue #3324 — solar / irradiance reductions (planned consumer of the helpers)
- Issue #3325 — AI batch metric reductions (planned consumer of the helpers)
- Issue #3142 — required-checks sync discipline
- Issue #3129 — the original required_checks drift gap that the sync script closes