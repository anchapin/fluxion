# β-soak Criterion 2 tracker — ADR-0007 energy-conservation failures (Issue #3359)

> **Summary 1/7:** Tracking surface for the nightly `Criterion 2 — zone_balance_eplus_isolation` failures (ADR-0007 acceptance criteria, Issue #3286 gate). The β-soak streak sits at **0/30** because the workflow fires the gate-failure protocol as designed; this document is the canonical per-night summary of the two consistently-failing tests and their upstream tracking chain.
> **Summary 2/7:** Failing tests (consistent across every nightly run since 2026-09-01): `test_physics_thermal_model_eplus_case_600_reference_csv` (`tests/zone_balance_eplus_isolation.rs:298:5`, wild oscillation, `max ΔT = 34.007 °C`) and `test_free_floating_case_900ff_isolation` (`tests/zone_balance_eplus_isolation.rs:445:5`, divergence to `T_min = −∞` / `T_max = +∞`).
> **Summary 3/7:** Both tests belong to the §LIMIT-21 cohort (`docs/KNOWN_ISSUES.md` — pre-existing gauge β-path air-trajectory failures, owner Issue #3297). The cohort is verified **identical** at `fd7ef13^` = `832b0fe` and HEAD `0b54606` on 2026-09-03, i.e. NOT introduced by any recent PR.
> **Summary 4/7:** Cross-references: parent Issue #3354 (workflow diagnostic), umbrella #3291 (Phase A8), cohort owner #3297 (§LIMIT-21), unblockers #1465 / #1462, escape hatch #3285, gate contract #3286, original infra #3284, meta #3072 (GaugeSolver structural-work ADR), §LIMIT-22 (exact-CN proxy aftermath), ADR-0007 acceptance criteria.
> **Summary 5/7:** **No silent counter reset** is being performed — the streak `0/30` is the workflow's stateless computation and is correct. Silently resetting would violate `AGENTS.md` / `RULES.md` / ADR-0001. A human operator is the appropriate party to reset for non-physics reasons.
> **Summary 6/7:** The β-soak escape hatch (`docs/agents/beta-soak-escape-hatch.md`, Issue #3285, dormant by design) is the operational safety valve for follow-up PRs that need to merge a structural gauge change while §LIMIT-21 is still open — NOT a way to weaken the gate or reset the streak.
> **Summary 7/7:** Recovery steps: close §LIMIT-21 by landing the air-trajectory fidelity program (#1465 / #1462 / #3059 — NOT a constant or baseline change); once the 2 nightly Criterion 2 tests pass for 30 consecutive nights, the streak reaches `30/30` and the gauge default-flip is unblocked via the `beta-soak-gate (enforce)` step in CI.

## Status

| Field | Value |
|-------|-------|
| Streak | `0/30` |
| Last-run criterion 2 | ❌ red (energy conservation) |
| Criteria 1, 7 | ✅ green |
| Criteria 3–6 | ⏭ skipped (abort on Criterion 2 failure — `set -euo pipefail`) |
| Counter | not manually reset (stateless workflow computation) |
| Tracker doc | this file (`docs/agents/beta-soak-criterion-2-tracker.md`) |

## Failing tests (canonical list)

The structured failure list is also emitted by
`python3 scripts/check_beta_soak_gate.py --criterion-2-failures` (a
machine-readable summary for the nightly workflow's JSON artifact). The
two tests below are the **consistent, nightly-failing** Criterion 2
residuals. Other §LIMIT-21-cohort tests across other binaries
(`ashrae_140_case_600_series`, `known_issues_regression`, `ashrae_140_case_960_sunspace`,
`ashrae_140_blind_validation`) belong to nightly Criteria 4, 5, and 6
respectively; this tracker focuses on Criterion 2 only.

| # | Test | File | Panic site | Symptom |
|---|------|------|------------|---------|
| 1 | `test_physics_thermal_model_eplus_case_600_reference_csv` | `tests/zone_balance_eplus_isolation.rs:187` | `:298:5` | `max ΔT = 34.007 °C` step-to-step oscillation; stdout: `T_zone mean=-12.59°C, T_zone min=-32.52°C, T_zone max=7.15°C, \|mean-20\|=32.594°C` |
| 2 | `test_free_floating_case_900ff_isolation` | `tests/zone_balance_eplus_isolation.rs:430` | `:445:5` | numerical divergence to non-finite (`T_min = −∞`, `T_max = +∞`); assertion `min_900.is_finite() && max_900.is_finite()` |

Both failures are listed verbatim in `docs/KNOWN_ISSUES.md`
**§LIMIT-21** (Issue #3297) — the "Gauge β-path pre-existing
air-trajectory failure cohort (Case 600 / 900FF / 600-series / 960 /
950FF) — β-soak blockers". The §LIMIT-21 entry notes the failure set is
**verified identical** at `fd7ef13^` = `832b0fe` and at HEAD
`0b54606` on 2026-09-03 — i.e. **pre-existing, NOT introduced by any
recent PR**.

By `AGENTS.md` / `RULES.md`, the §LIMIT-21 cohort is deliberately **not**
`#[ignore]`-quarantined — quarantining would let the nightly soak go
green with the underlying physics gap hidden, in violation of the
gate-failure contract (Issue #3286).

## Why the gate is closed

Per Issue #3286: 30 consecutive green nightly runs are required to open
the gauge default-flip gate. Per ADR-0007 / ADR-0001 / `RULES.md` /
`AGENTS.md`: the only legitimate closure path is a structural physics
fix (the air-trajectory fidelity program), not a constant, baseline, or
tolerance change. The 2 nightly Criterion 2 tests above remain red
because the underlying physics gap (§LIMIT-21) is still open. The
streak `0/30` is therefore the **correct** current state and the gate
must stay closed.

## Recovery steps

1. **Close §LIMIT-21** by landing the air-trajectory fidelity program:
   - #1465 — Phase 3 GaugeSolver validation against ASHRAE 140 Case 900.
   - #1462 — Phase 1b shadow-mode GaugeSolver (`GaugeSolver::step` in
     `physics_adapter.rs`).
   - #3059 / #3060 / #3061 / #3062 / #3063 / #3070 — cohort follow-ups
     (Case 195 / 600 / 620 / 940 / 960 residuals, all gated on the
     gauge path landing).
2. **Verify nightly Criterion 2 goes green** by running
   `cargo test --locked --features gauge-solver --test
   zone_balance_eplus_isolation` locally and matching the
   `.github/workflows/nightly-ashrae-140-gauge.yml` step-2 command.
3. **Wait for 30 consecutive green runs** in the nightly workflow; the
   `Compute β-soak streak` step computes the new streak from
   `gh run list --workflow nightly-ashrae-140-gauge.yml --branch
   develop --status completed --limit 60` and writes
   `beta-soak-state.json`.
4. **PR-time gate trips open** when a PR touching the gauge default
   runs `python3 scripts/check_beta_soak_gate.py --gate enforce` and
   the streak is `>= 30` (i.e. `unblocked = true`).

Until step 1 lands, the streak will stay at `0/30` and the gate will
remain closed. **No silent reset.**

## Escape hatch (operational safety valve)

If a structural gauge-default PR (e.g. a #1465 / #1462 / #3059
follow-up that itself narrows but does not yet close §LIMIT-21) needs
to merge while the gate is closed, the **escape hatch** mechanism
(`docs/agents/beta-soak-escape-hatch.md`, Issue #3285) is the
appropriate operational lever:

- Requires `BETA_SOAK_ESCAPE_AUTHORIZED_BY=<handle>` env var in CI.
- Requires the handle to appear in `scripts/beta_soak_admin_allowlist.txt`.
- Requires `--escape` CLI flag on the gate invocation (Issue #3285
  owns the `--escape` flag — currently dormant; not added by this PR).

The escape hatch is **not** a way to reset the streak, weaken the gate,
or relax a baseline; it is an expiring (14-day default) auditable
bypass that always cites a §LIMIT-21 unblocker.

## Cross-references

- **Issue #3354** — parent diagnostic ("workflow firing as designed").
- **Issue #3359** — this issue / this tracker doc.
- **Issue #3286** — β-phase 30-day soak window gate contract.
- **Issue #3284** — original nightly CI workflow infrastructure.
- **Issue #3285** — β-soak escape hatch (operational safety valve).
- **Issue #3291 / PR #3482** — Phase A8 umbrella (default flip is
  gated on this gate).
- **Issue #3297** — §LIMIT-21 cohort owner (mass-state proxy aftermath).
- **Issue #1465** — Phase 3 GaugeSolver validation (closed).
- **Issue #1462** — Phase 1b shadow-mode GaugeSolver (closed).
- **Issue #3059 / #3060 / #3061 / #3062 / #3063 / #3070** — cohort
  follow-ups (Case 195 / 600 / 620 / 940 / 960 residuals).
- **Issue #3072** — meta-issue (ADR-0007).
- **`docs/KNOWN_ISSUES.md` §LIMIT-21** — pre-existing air-trajectory
  cohort (the canonical structural-gap documentation).
- **`docs/KNOWN_ISSUES.md` §LIMIT-22** — exact-CN proxy aftermath
  (3 feature-gated `#[cfg_attr(feature = "gauge-solver", ignore)]`
  quarantines; default-build green).
- **`docs/adr/0007-gauge-solver-structural-work.md`** — ADR-0007
  acceptance criteria (Criterion 2 = line 101).
- **`scripts/check_beta_soak_gate.py`** — gate validator; the
  `--criterion-2-failures` CLI option emits this tracker doc's failing
  tests as a JSON summary for the nightly artifact.
- **`.github/workflows/nightly-ashrae-140-gauge.yml`** — the nightly
  workflow; step "Criterion 2 — zone_balance_eplus_isolation" is the
  Criterion 2 check, and the new
  "Emit Criterion 2 failures summary" step uploads the JSON.
- **`docs/agents/beta-soak-gate-failure-no-reset.md`** — the
  gate-failure protocol note (Issue #3354).
- **`docs/agents/beta-soak-state-schema.md`** — gate state schema.
- **`docs/agents/beta-soak-escape-hatch.md`** — escape hatch mechanism
  (Issue #3285).
- **ADR-0001** — No-Parameter-Tuning Rule (why no baseline/constant
  changes).

## Definition of done

This tracker (Issue #3359) closes when **any one** of:

1. The §LIMIT-21 air-trajectory program lands and the 2 nightly
   Criterion 2 tests start passing (resolves the gate as a side effect).
2. The nightly Criterion 2 failure mode is formally migrated to a new
   documented ADR (e.g. ADR-0014) that supersedes §LIMIT-21 for the
   β-soak context.
3. A human owner explicitly closes this issue with rationale for
   retiring the tracking surface (e.g., the β-soak program is paused
   pending §LIMIT-21 closure).
