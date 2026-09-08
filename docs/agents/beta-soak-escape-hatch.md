# β-soak escape hatch — operational safety valve (Issue #3285)

> **Summary 1/7:** Operational mechanism for bypassing the §LIMIT-21 β-soak gate ONLY when an explicit, allowlisted human operator authorises the bypass via env var `BETA_SOAK_ESCAPE_AUTHORIZED_BY=<handle>` — never silent.
>
> **Summary 2/7:** Trigger: a gauge-default-flip PR (Phase A8 follow-up) must merge while the §LIMIT-21 air-trajectory cohort is still open AND the β-soak streak cannot reach 30/30 (currently 0/30).
>
> **Summary 3/7:** Approval: requires `BETA_SOAK_ESCAPE_AUTHORIZED_BY` set to a GitHub handle listed in `scripts/beta_soak_admin_allowlist.txt`; the CLI `--escape` flag enables recognition.
>
> **Summary 4/7:** Recording: every bypass emits a structured log line including the authorised handle, the closing streak, and the linked tracking issue — visible in CI logs and the PR conversation.
>
> **Summary 5/7:** Recovery: each bypass carries an expiry (default: one PR cycle, 14 days) and an automatic follow-up issue linking to §LIMIT-21 closure; the gate is NOT weakened or removed.
>
> **Summary 6/7:** This is dormant by design — encoded so it can be invoked without code change, but no PR in the current wave uses it; first-use must be filed and approved separately.
>
> **Summary 7/7:** Related: `scripts/check_beta_soak_gate.py` (`--escape` flag, `BETA_SOAK_ESCAPE_AUTHORIZED_BY` env var), `scripts/beta_soak_admin_allowlist.txt` (admin allowlist), `docs/KNOWN_ISSUES.md` §LIMIT-21, `docs/adr/0007-gauge-solver-structural-work.md` ADR-0007, Issue #3285 (this issue), Issue #3286 (gate contract).

## Why this exists

The β-soak gate (Issue #3286, `scripts/check_beta_soak_gate.py`,
`docs/agents/beta-soak-state-schema.md`) is the production-path gate that
prevents the Phase A8 default-flip from merging before the seven ADR-0007
acceptance criteria have been green for 30 consecutive nightly runs. While
§LIMIT-21 (Issue #3297) is open the streak is pinned at 0/30, which means a
follow-up PR that legitimately needs to merge a structural gauge change
(e.g. a #1465 / #1462 fix that has narrowed the §LIMIT-21 cohort but not yet
closed it) is blocked by the gate even though the change is itself the
*unblocker*.

The escape hatch is the operational safety valve for that scenario. It is
**NOT**:

- A way to weaken or skip the gate permanently.
- A way to reset the streak counter (the gate-failure protocol in
  `docs/agents/beta-soak-gate-failure-no-reset.md` already forbids resets).
- A way to ignore §LIMIT-21 or raise any reference-data baseline
  (`RULES.md` / `AGENTS.md` / ADR-0001 are binding).
- An automated bypass. A human operator MUST explicitly authorise it.

It IS a structured, auditable, expiring bypass that lets one well-reasoned
follow-up PR merge while §LIMIT-21 is still closing, with the bypass itself
acting as a forcing function to land the §LIMIT-21 unblocker.

## Trigger conditions

The escape hatch may be invoked when **all** of the following hold:

1. The β-soak streak is below target (`streak < 30`; currently `0/30` per
   `docs/agents/beta-soak-gate-failure-no-reset.md`).
2. A PR legitimately requires the gauge default to change as a prerequisite
   for the §LIMIT-21 unblocker (a structural fix from #1465 / #1462 /
   #3059, NOT a constant or baseline change).
3. The PR's CI is otherwise green (only the β-soak `enforce` step is
   blocking).
4. An open tracking issue links the bypass to the §LIMIT-21 unblocker — the
   bypass is the lever, not the answer.
5. No PR in the current wave has already used the escape hatch (one
   bypass per tracking issue unless a human operator explicitly reopens it).

## Approval workflow

A bypass requires THREE independent signals. Missing any one is a fail-closed
diagnostic from `scripts/check_beta_soak_gate.py`:

1. **Operator handle (env var).** The operator invoking the bypass MUST set
   `BETA_SOAK_ESCAPE_AUTHORIZED_BY=<github-handle>` in the CI environment
   for the PR's `beta-soak-gate` job. The handle is the operator's GitHub
   username; it is recorded in the bypass log.
2. **Allowlist membership (config file).** The handle MUST appear in
   `scripts/beta_soak_admin_allowlist.txt` (one handle per line; `#`
   comments and blank lines allowed; case-insensitive match). The
   allowlist is curated by repo owners — adding a handle is itself a
   tracked change (audit trail via git blame).
3. **Explicit CLI flag.** The gate script MUST be invoked with
   `--escape`. Without the flag, the env var is ignored (defence in
   depth: setting the env var alone cannot open the gate).

On bypass success the gate exits 0 with the streak / target / handle /
linked-issue printed to stdout in machine-readable form (suitable for the
PR's CI log).

## Recovery

Every bypass has a built-in expiry:

- **Default expiry window:** 14 days from PR merge (`expires_at` field in
  the bypass audit trail). After the window the bypass becomes historical
  record only — it does NOT auto-revert merged code.
- **Forcing function.** The PR that triggered the bypass MUST cite this
  issue (#3285), the §LIMIT-21 unblocker issue (#1465 / #1462 / #3059),
  and add a follow-up issue linking to §LIMIT-21 closure. The follow-up
  issue is the only way the bypass ages out: once §LIMIT-21 closes and
  the streak reaches 30, the gate is open naturally and the escape hatch
  is unused.
- **No code reverts.** The bypass is a one-time decision, not an ongoing
  policy. The escape hatch machinery itself stays dormant after use; it
  is NOT removed.

## CI wire-up (proposed — out of scope for this PR)

The escape-hatch CI step is a separate, follow-up issue (out of scope for
#3285 — this PR only encodes the mechanism in the gate script + the
operator docs). When that lands:

```yaml
- name: β-soak gate (escape hatch allowed for approved PRs)
  env:
    BETA_SOAK_ESCAPE_AUTHORIZED_BY: ${{ github.event.pull_request.body }}
  run: |
    python3 scripts/check_beta_soak_gate.py \
      --state beta-soak-state.json \
      --gate enforce \
      ${{ github.event.pull_request.body contains '#3285-escape-approved' && '--escape' || '' }}
```

The `contains` check requires the PR description to literally carry the
`#3285-escape-approved` marker (a documented convention — see the
escape-hatch log). Without the marker the step runs in default mode.

## Related

- `scripts/check_beta_soak_gate.py` — the gate script; `--escape` flag
  added by this PR.
- `scripts/beta_soak_admin_allowlist.txt` — admin handle allowlist (new in
  this PR).
- `scripts/ci/test_check_beta_soak_gate.py` — pytest coverage of the
  `--escape` flag (added in this PR).
- `docs/agents/beta-soak-state-schema.md` — gate schema and contract.
- `docs/agents/beta-soak-gate-failure-no-reset.md` — gate-failure
  protocol (Issue #3354).
- `docs/KNOWN_ISSUES.md` §LIMIT-21 — the cohort the escape hatch exists
  to live alongside.
- `docs/adr/0007-gauge-solver-structural-work.md` — ADR-0007 acceptance
  criteria; the escape hatch does NOT amend them.
- Issue #3285 — this issue (the escape hatch itself).
- Issue #3286 — β-soak gate contract.
- Issue #3291 / PR #3482 — Phase A8 default flip (the gate's subject).
- Issue #3297 — §LIMIT-21 cohort owner (the gate's reason for being
  closed).