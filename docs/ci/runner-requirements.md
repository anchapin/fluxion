# Linux Runner Requirements for Actions v5/v6/v7 (node24) — Issue #3312

**Issue:** #3312 (primary); refreshed for #3579 (alex-workstation runner)  
**Date:** 2026-09-01; last refreshed 2026-09-09  
**Status:** Verified — GitHub-hosted routing for the `fluxion-ci` pool; one
diagnostic `alex-workstation` runner registered.

This doc verifies that every Linux `runs-on:` label used in
`.github/workflows/*.yml` resolves to a runner whose Actions agent meets
the v2.327.1 minimum required by node24 action majors (checkout v7,
cache v6, upload-artifact v7). It is the Issue #3312 acceptance doc
and was refreshed by Issue #3579 to reflect PR #3569, which added the
diagnostic `.github/workflows/alex-dev.yml` running on the local
`alex-workstation` self-hosted runner.

## Why this matters

PR #3294 (`688fe78`, merged 2026-09-01) bumped pinned action SHAs across 4
workflow files, including cross-major upgrades:

- `actions/checkout` v4 → v7.0.1
- `actions/cache` v4 → v6.1.0
- `actions/upload-artifact` v4 → v7.0.1

The v5/v6/v7 action lines execute on the **node24 runtime**, which requires
**GitHub Actions Runner ≥ 2.327.1**. GitHub-hosted runners always satisfy
this; only self-hosted runners can lag behind and fail at step-bootstrap
time with errors like `An error occurred ... node24 ... requires a minimum
version of 2.327.1`.

## What `FLUXION_LINUX_RUNNER` resolves to (verified 2026-09-01)

All consumer workflows route Linux jobs through a single repository
variable with a hosted fallback, e.g.
`.github/workflows/h_tr_em_regression_gate.yml`:

```yaml
runs-on: ${{ vars.FLUXION_LINUX_RUNNER || 'ubuntu-latest' }}
```

Verified state of the repository, last refreshed 2026-09-09:

| Check | Command | Result |
|-------|---------|--------|
| Repo variables | `gh api repos/anchapin/fluxion/actions/variables` | `{"variables":[],"total_count":0}` — `FLUXION_LINUX_RUNNER` is **unset** |
| Self-hosted runners (Hetzner `fluxion-ci` pool) | `gh api repos/anchapin/fluxion/actions/runners?label=fluxion-ci` | `{"total_count":0}` — **zero** registered (Hetzner pool is documented but not active; see `docs/self-hosted-runners.md`) |
| Self-hosted runners (local `alex-workstation`) | `gh api repos/anchapin/fluxion/actions/runners?label=alex-workstation` | `{"total_count":1,"runners":[{"name":"alex-workstation","os":"Linux","status":"online","version":"2.337.0","labels":["self-hosted","Linux","X64","alex-workstation"]}]}` — **one** registered (added by PR #3569, merged 2026-09-09) |
| Diagnostic workflow `runs-on:` matrix | `.github/workflows/alex-dev.yml` (`diagnose` job) | `runs-on: [self-hosted, alex-workstation, linux, x64]` — fixed (no fallback) so the diagnostic always requires the registered `alex-workstation` runner |

> **Note:** `FLUXION_LINUX_RUNNER` (the repository variable consumed by
> `vars.FLUXION_LINUX_RUNNER || 'ubuntu-latest'` in heavy CI workflows)
> is **intentionally separate** from the per-workflow `runs-on:` matrix
> in `alex-dev.yml`. The variable gates the Hetzner `fluxion-ci` pool;
> `alex-dev.yml` targets the local workstation label directly. Setting
> `FLUXION_LINUX_RUNNER` does **not** route `alex-dev.yml` anywhere new,
> and the absence of the variable does not block `alex-dev.yml`.

**Verdict:** with `FLUXION_LINUX_RUNNER` unset, every
`|| 'ubuntu-latest'` fallback resolves to the **GitHub-hosted** label —
no runner service to upgrade on that path. The
`alex-workstation` self-hosted runner is online at Actions agent
**v2.337.0**, comfortably above the v2.327.1 node24 minimum. (The
Hetzner provisioning path is documented in `docs/self-hosted-runners.md`;
it is not currently active.)

## Post-upgrade green-run evidence

The `h_tr_em_regression_gate` workflow (the only workflow in #3294 with
multiple action bumps) ran on `develop` at the exact merge commit of
#3294 — i.e. with checkout v7.0.1 / cache v6.1.0 / upload-artifact v7.0.1
already in effect:

- Run: <https://github.com/anchapin/fluxion/actions/runs/33480132457>
- Head SHA: `688fe78163e3f49ef536b7d00a3a03ff2ac0a7dd` (#3294 merge commit)
- Conclusion: **success** (2026-09-01T07:00:53Z → 07:07:37Z)

This confirms the new action majors execute cleanly end-to-end on the
GitHub-hosted fleet.

## Operator action (only if self-hosted routing is ever enabled)

If `FLUXION_LINUX_RUNNER` is later set to a self-hosted label (e.g.
`fluxion-ci`), the operator **must** upgrade each runner service to
≥ 2.327.1 **before** the next scheduled run, or affected jobs fail at
step bootstrap:

1. Download a runner ≥ 2.327.1 from
   <https://github.com/actions/runner/releases> (or later — current
   releases are far newer).
2. Stop the runner service, replace the runner installation directory
   contents, and restart (or use the runner's built-in auto-update, which
   keeps self-hosted runners current automatically).
3. Confirm with `./config.sh --version` and one green `push`-event run.

## Acceptance criteria status (Issue #3312; refreshed for #3579)

| Criterion | Status |
|-----------|--------|
| Confirm hosted vs self-hosted resolution for `FLUXION_LINUX_RUNNER` | ✅ GitHub-hosted (variable unset, 0 `fluxion-ci` runners) |
| Upgrade self-hosted runner ≥ 2.327.1 (Hetzner pool) | N/A — pool is not active |
| Green post-upgrade `h_tr_em_regression_gate` run | ✅ Run 33480132457, success on `688fe78` |
| Confirm diagnostic `alex-workstation` runner meets node24 minimum | ✅ v2.337.0 (≥ 2.327.1) on PR #3569 workflow |
| `alex-dev.yml` `runs-on:` matrix documented in verification table | ✅ (this doc, refreshed 2026-09-09) |

## See Also

- `docs/self-hosted-runners.md` — Hetzner provisioning, the variable's
  routing semantics, and the local workstation runner pattern
  (`.github/workflows/alex-dev.yml`)
- PR #3294 — the action-SHA bump that motivated this verification
- PR #3569 — diagnostic workflow that registers the `alex-workstation`
  runner (`alex-dev.yml`)
- <https://github.com/actions/runner/releases> — runner version history
