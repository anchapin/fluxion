# Linux Runner Requirements for Actions v5/v6/v7 (node24) — Issue #3312

**Issue:** #3312  
**Date:** 2026-09-01  
**Status:** Verified — GitHub-hosted routing, nothing to upgrade

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

Verified state of the repository on 2026-09-01:

| Check | Command | Result |
|-------|---------|--------|
| Repo variables | `gh api repos/anchapin/fluxion/actions/variables` | `{"variables":[],"total_count":0}` — `FLUXION_LINUX_RUNNER` is **unset** |
| Self-hosted runners | `gh api repos/anchapin/fluxion/actions/runners` | `{"total_count":0,"runners":[]}` — **zero** registered |

**Verdict:** with the variable unset, every `|| 'ubuntu-latest'` fallback
resolves to the **GitHub-hosted** label. No self-hosted runner exists, so
there is no runner service to upgrade. (The Hetzner provisioning path is
documented in `docs/self-hosted-runners.md`; it is not currently active.)

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

## Acceptance criteria status (Issue #3312)

| Criterion | Status |
|-----------|--------|
| Confirm hosted vs self-hosted resolution | ✅ GitHub-hosted (variable unset, 0 self-hosted runners) |
| Upgrade self-hosted runner ≥ 2.327.1 | N/A — no self-hosted runner exists |
| Green post-upgrade `h_tr_em_regression_gate` run | ✅ Run 33480132457, success on `688fe78` |

## See Also

- `docs/self-hosted-runners.md` — Hetzner provisioning and the variable's
  routing semantics
- PR #3294 — the action-SHA bump that motivated this verification
- <https://github.com/actions/runner/releases> — runner version history
