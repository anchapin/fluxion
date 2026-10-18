# ADR-0025: β-Soak Tracked-Cohort Design for Criterion 4 (Case 600 Series)

- **Status:** 🟡 Proposed — **decision pending (Alex)**. This ADR changes gate semantics for the β-soak (Issue #3286); it must not be implemented without Alex's explicit approval.
- **Date:** 2026-10-18
- **Deciders:** Alex (gate-semantics owner)
- **Supersedes:** None (amends the *interpretation* of one acceptance criterion in [ADR-0007](0007-gauge-solver-structural-work.md); ADR-0007's acceptance criteria themselves are untouched)
- **Depends on:** ADR-0007 (β-soak acceptance criteria), Issue #3286 (soak gate contract), Issue #3297 (§LIMIT-21), Issue #4473 (canonical-weather restore — made the 600-series cohort observable)
- **Related:** `docs/agents/beta-soak-criterion-2-tracker.md` (Issue #3359 — the Criterion 2 precedent), `scripts/check_beta_soak_gate.py` (gate contract source of truth), `.github/workflows/nightly-ashrae-140-gauge.yml`, Issue #4281 (output-diff gate — the drift-check pattern reused here)

> Summary 1/7: The β-soak has **never recorded a green night**. Criteria 1–3 pass, but Criterion 4 (Case 600 series under `--features gauge-solver`) aborts the job on 9 known band-gap failures, so the streak file is never written and the soak cannot accumulate a streak at all.
> Summary 2/7: This ADR proposes a **tracked-cohort** definition for Criterion 4, mirroring the pattern the strict energy gate already uses for its KNOWN-FAIL cohort — known structural gaps are tracked and drift-checked, not silently ignored, and any *new* failure still breaks the streak.
> Summary 3/7: Key verified premise: the soak cohort is measured under the **gauge-solver feature build**, which is a different dispatch path from the default-build strict gate. The two cohorts do **not** coincide (e.g. Case 630 annual cooling PASSes in the default build and FAILs in the gauge build).
> Summary 4/7: Three design options are laid out with tradeoffs; the recommendation is **Option C** — a hard gate on currently-green tests plus a drift-checked tracked cohort with a recorded baseline.
> Summary 5/7: Per RULES.md / ADR-0001, nothing here tunes a band or hides a gap: the tracked cohort is enumerated, measured values stay recorded, and closure of the soak still requires the underlying physics work (§LIMIT-21 / the cohort's owning issues) — the tracked cohort shrinks only when tests actually pass.
> Summary 6/7: The streak semantics stay conservative: a night counts only when the hard gates are green **and** the tracked cohort matches its recorded baseline; any new failure, any cohort member turning green (re-pin required), or any drift beyond epsilon resets the streak.
> Summary 7/7: **This document changes no workflow or script.** Implementation (a workflow edit + a baseline file + checker changes) is a separate PR that waits for Alex's decision on this ADR.

## Problem

The nightly soak (`.github/workflows/nightly-ashrae-140-gauge.yml`) runs the seven ADR-0007 acceptance criteria under `set -euo pipefail`. Criterion 2 (energy conservation) is already allowed-red by design — its step captures the failure and the §LIMIT-21 cohort-drift detector (Issue #3744) checks that the failing set matches the canonical list. Criterion 4 has no such handling.

Since Issue #4473 restored the canonical weather assets, the Criterion 4 suite (`ashrae_140_case_600_series::`, 27 executed tests + 12 ignored) actually executes end-to-end, and the job aborts there. Verified failing set on the 2026-10-18 dispatch run (`39139200510`, develop):

| # | Failing test | Metric |
|---|--------------|--------|
| 1 | `case_600ff::test_max_temperature` | 600FF Tmax |
| 2 | `case_630::test_annual_heating` | 630 annual H |
| 3 | `case_630::test_annual_cooling` | 630 annual C |
| 4 | `case_630::test_peak_cooling` | 630 peak C |
| 5 | `case_640::test_annual_cooling` | 640 annual C |
| 6 | `case_650::test_annual_cooling` | 650 annual C |
| 7 | `case_650::test_peak_cooling` | 650 peak C |
| 8 | `case_650ff::test_min_temperature` | 650FF Tmin |
| 9 | `case_650ff::test_max_temperature` | 650FF Tmax |

6 tests pass; 12 are `#[ignore]`-quarantined (and by the quarantine rule stay excluded — the soak must not count them green). Because the job aborts at Criterion 4, criteria 5–7 never run and — the operational blocker — **the streak file is never written**, so the soak cannot even accumulate a streak of partially-green nights.

### Verified premise: the soak cohort is not the strict-gate cohort

The strict ±15% annual-energy gate runs the **default build** (legacy selector). The soak runs `--features gauge-solver`. Same metric names, different dispatch path, different results. Cross-checked on the same day, same SHA (`37eb811f`):

| Metric | Default build (strict gate) | Gauge build (soak criterion 4) |
|--------|------------------------------|-------------------------------|
| 630 annual cooling | **PASS** (2.976 in [2.400, 3.247]) | **FAIL** |
| 630 annual heating | KNOWN-FAIL | FAIL (consistent) |
| 650 annual cooling | **PASS** | **FAIL** |
| 650FF Tmin/Tmax | KNOWN-FAIL | FAIL (consistent) |
| 600FF Tmax | KNOWN-FAIL | FAIL (consistent) |
| 640 annual cooling | **PASS** | **FAIL** |

Consequence: a tracked-cohort definition for Criterion 4 **cannot be borrowed from the strict-gate baseline**. It needs its own recorded baseline measured in the gauge build. This also means the soak's tracked cohort carries real information the default-build gates cannot see: it is the only per-night instrument on the gauge dispatch path.

## Design goals and constraints

1. The soak must be able to accumulate a streak without the §LIMIT-21-family gaps going green first — otherwise the soak measures nothing until the structural physics work lands, and the "30 consecutive green nights" contract is unreachable by construction.
2. No silent weakening: every excluded-from-gate test is *named*, its measured value recorded, and any new failure or drift resets the streak (RULES.md; ADR-0001; the Criterion 2 precedent's "no silent counter reset" discipline).
3. Cohort membership is closed under evidence: a test leaves the tracked cohort only by passing (with re-pin of its measured value), and a test enters only via an explicit ADR amendment after a diagnosed, pre-existing failure — never by an undiagnosed new failure.
4. The gate stays fail-closed on infrastructure: a missing log line, a dropped test, or a changed test set is a gate failure, not a pass.

## Options

### Option A — Allow-red criterion 4, mirroring Criterion 2 verbatim

Wrap the Criterion 4 step in `|| true`, and extend the existing cohort-drift detector to assert the Criterion 4 failing set equals a canonical list of the 9 tests above.

- ✅ Smallest change; reuses a proven pattern (Issue #3744 / Issue #3359 tracker).
- ✅ The failing set is still verified every night — a test that *stops* failing is caught (the detector compares sets, not just counts).
- ❌ No per-value measurement: the detector compares test names only, so a tracked test whose measured error doubles (e.g. 650FF Tmin drifting from −18.7 °C to −25 °C against the same band) still counts green. The soak would certify a degrading gauge path as stable.
- ❌ The 9-entry list is hand-maintained prose in a tracker doc; nothing binds the workflow's list to the tests' actual identity.

### Option B — Hard gate on criterion 4, wait for physics

Leave the soak as-is; the streak stays unwritten until the 600-series gaps close under the gauge build.

- ✅ Maximally strict; zero gate semantics change.
- ❌ The soak never runs criteria 5–7, so the other four criteria's regression protection is a dead letter for the entire (multi-month) life of the structural work. A gauge-build regression in Case 900 harness or the BatchOracle clone contract would be invisible.
- ❌ The gate's own purpose — detect regressions in the production-path candidate over 30 nights — is defeated: you cannot regress a streak that never starts.

### Option C — Hard gate + drift-checked tracked cohort (recommended)

Criterion 4 becomes two tiers:

- **Hard tier:** the 6 currently-passing tests plus the 12 ignored ones must remain in their current state (pass / still-ignored). Any new failure fails the night.
- **Tracked tier:** the 9 known-gap tests run with `|| true`; their measured values (the exact numbers the tests print) are parsed into a committed baseline file (`tests/reference_data/gauge_soak/criterion4_tracked_cohort_baseline.csv`, with dated SHA provenance, same convention as the #4281 output-diff baseline) and diffed nightly at the same 0.1% relative epsilon. Drift beyond epsilon fails the night.

Streak semantics: a night counts only if every criterion's hard tier is green, the tracked cohort matches its baseline, and criteria 5–7 pass. First tracked test to flip green is celebrated and **re-pinned** (it leaves the cohort; its value becomes a hard gate).

- ✅ The soak finally measures something real: per-night, "no change anywhere in the gauge-build 600-series, and everything else green."
- ✅ Both failure directions are caught: a new red (hard tier) and a value drift in a known red (tracked tier). A test that turns green stops the night and forces a re-pin — the cohort can only shrink with evidence.
- ✅ Baseline membership is machine-checked against the workflow (fail-closed on metric-set change), so the list cannot rot the way a prose tracker can.
- ❌ Most implementation work of the three options: a baseline file, parser/writer script work, a workflow edit, and the criteria 5–7 steps must also be made abort-safe (criterion 2 is already allowed-red; the streak computation must skip it consistently).
- ❌ A genuinely noisy tracked test could flap the streak via epsilon drift; mitigation is that epsilon matches the proven #4281 gate, which has held bit-identical across recent rounds — if the gauge build is *that* unstable, the soak should say so.

## Recommendation

**Option C.** It preserves everything the gate contract actually protects (regression detection on the production-path candidate) while acknowledging — with recorded evidence, per the standing fidelity-first direction — that the 600-series band gaps are known structural gaps owned by the §LIMIT-21-family issues, not nightly noise to re-litigate. Option A is the fallback if the implementation cost of C is judged not worth it before the structural work lands; Option B is rejected because it silently disables four of the seven acceptance criteria indefinitely.

## What this ADR does NOT do

1. It does not change any workflow, script, test, tolerance, or baseline. Implementation waits for Alex's approval of this design.
2. It does not touch the default-build strict gate, its baseline, or its KNOWN-FAIL tracking.
3. It does not quarantine or `#[ignore]` anything; the tracked cohort runs every night and its values are recorded.
4. It does not declare any test passing, and it does not close or re-scope any §LIMIT-* issue.

## Decision

**Pending — Alex.** On approval, implementation lands as its own PR: the baseline recorded from a fresh gauge-build run with provenance, the checker extended, the workflow edited, and the first counted night starting only after that PR's own nightly run is green under the new semantics.
