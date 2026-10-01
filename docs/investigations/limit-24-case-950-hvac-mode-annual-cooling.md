# LIMIT-24 — investigation history

Narrative history for **LIMIT-24**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-24` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-24: Case 950 HVAC-mode annual cooling ~14× UNDER band — docs-only structural LIMIT entry, companion to §LIMIT-17 (Issue #3551)

- **Description:** Per the 2026-08-16 ASHRAE 140 snapshot, Case 950
  (HVAC mode) annual cooling measures **33.08 kWh vs the ASHRAE 140
  reference band 390–920 kWh** — **~14× UNDER** (the band is ~91 %
  above the current value). All four reported Case 950 (HVAC mode)
  metrics on the 84-metric scorecard are fail-rows:

  | Metric | Engine | Reference band | Δ vs band |
  |---|---:|---|---|
  | annual_heating (kWh) | 0.00 | [0.00, 0.00] | in-band (edge) |
  | annual_cooling (kWh) | **33.08** | [390, 920] | **−91.5 % UNDER** (~14× below lower bound) |
  | peak_heating (kW) | 0.00 | [0.00, 0.00] | in-band (edge) |
  | peak_cooling (kW) | 0.39 | [0.70, 0.90] | −44 % UNDER |

  The two heating / peak metrics are formally in-band only because the
  ASHRAE 140 reference band is degenerate (`[0.00, 0.00]`); the annual
  cooling 33.08 kWh and peak cooling 0.39 kW are the load-bearing
  failures.

  This failure has **no dedicated structural LIMIT entry** prior to
  #3551. §LIMIT-17 / #3058 documents the Case 950 **free-floating (FF)**
  companion gap (min free-floating temperature −23.92 °C vs band
  [−20.20, −17.80] °C) and records an explicit **regression-avoidance
  clause** for the HVAC mode: any future solver change that closes
  the Case 950FF gap **must preserve Case 950 (HVAC mode) annual
  cooling in the 390–920 kWh band**. But the actual current HVAC-mode
  state (33.08 kWh) is already ~14× outside that band — the
  "preserved-HVAC" target is far from the present HVAC state, and the
  failure has no separate structural LIMIT entry to track it.

- **Why the structural fix cannot close both at once (per AGENTS.md /
  RULES.md / ADR-0001):** Closing Case 950 (HVAC) annual cooling UP
  toward the 390–920 kWh band and closing Case 950FF min free-floating
  temperature DOWN toward the −20.20 to −17.80 °C band requires the
  **same single solver change** to push the cooling load into the air
  node (HVAC mode) while reducing the raw-outdoor forcing on the
  mass node (FF mode). The two signatures are **bidirectionally
  coupled**:

  1. The `MAX_CONVECTIVE_TO_AIR_MULTIPLIER = 2.0×` cap introduced by
     PR #3041 closed Case 650 (and other 600-series cooling OVERs)
     because it forces more of the solar + envelope gain through the
     air node at the cooling setpoint, but it did **not** close Case
     950 (HVAC) annual cooling — Case 950's night-flush path
     (`h_ve_night ≈ 570.8 W/K`, 18:00–07:00 per
     `tests/ashrae_140_blind_validation.rs:2171`) deposits the cooling
     load into the **mass node (multi-node path)** rather than the
     **air node (5R1C path)**, where the HVAC controller reads the
     setpoint signal.
  2. The derived-`h_tr_3` path that limits Case 950FF winter-min
     over-prediction (per §LIMIT-17 root-cause analysis, ~8×
     `h_ve_night / h_tr_em_wall` ratio) also deflects the summer peak
     away from the air node, suppressing the cooling load the HVAC
     system measures.
  3. No parameter adjustment to `h_ve_night`,
     `MAX_CONVECTIVE_TO_AIR_MULTIPLIER`, or `solar_distribution_to_air`
     can satisfy both regressions simultaneously: each moves one
     signature in the right direction while moving the other in the
     wrong direction. Per AGENTS.md / RULES.md / ADR-0001 ("no
     parameter tuning", "fix the underlying math", "must-never
     hardcode results") such an adjustment is explicitly forbidden.

- **Cross-references:**
  - **§LIMIT-17 / #3058** — Case 950 (free-floating) min
    free-floating temperature is −23.92 °C vs band
    [−20.20, −17.80] °C (3.72 °C outside). The
    regression-avoidance clause in §LIMIT-17 ("any future solver
    change must preserve Case 950 (HVAC mode) annual cooling in the
    390–920 kWh band") is the formal acceptance constraint for the
    future PR — but the clause refers to a band the current HVAC-mode
    state (33.08 kWh) does not occupy.
  - **§LIMIT-05 UPDATE (#2453)** — 900-series bidirectional
    annual-energy over-prediction (Cases 900 / 910 / 920 / 930 / 940).
    The Case 950 HVAC-mode annual cooling 14× UNDER is the same class
    of structural 5R1C + 9R4C single-lumped-mass-node pathology,
    routed to the GaugeSolver rework.
  - **#1465 / #1462** — the GaugeSolver architectural rework
    (treats solar + envelope heat transfer as geometric curvature
    rather than per-timestep energy injection). Both issues are
    individually closed; the **production-path switchover** is staged
    via **#3291 / PR #3482** (Phase A8 default flip —
    `ThermalSelector::default() = ZoneSolverKind::Gauge`, gated on the
    `gauge-solver` cargo feature and §LIMIT-21 β-soak closure).
  - **#3059 / §LIMIT-16** — Cases 610 / 630 / 650 peak cooling OVER
    cohort that motivated the `MAX_CONVECTIVE_TO_AIR_MULTIPLIER`
    cap. The cap closed the 600-series OVERs but did not transfer to
    Case 950 (HVAC) because Case 950's cooling load is deposited on
    the mass node (multi-node path), not the air node (5R1C path)
    where the cap's structural effect lives.

- **Affected Tests:**
  - `tests/diagnostics/case_950_hvac_mode_seasonal_attribution.rs` —
    per-month attribution diagnostic (`#[ignore]`-quarantined, runs
    `--ignored --nocapture`), wired into CI per the Issue #3551
    acceptance criterion. Prints the hourly contribution split (HVAC
    condensation vs mass release vs infiltration) for the Case 950
    HVAC-mode 90-day cooling season. Follow-up implementation is the
    post-#3551 PR.
  - `tests/ashrae_140_blind_validation.rs::test_case_950_5r1c_free_float_uses_night_vent_overrides_issue_1422`
    — the §LIMIT-09 / #3071 quarantine (sibling Case 950 5R1C
    night-vent test, currently `#[ignore]`'d).
  - `tests/ashrae_140_blind_validation.rs::test_case_950_mass_temperature_precooled_issue_1422`
    — the §LIMIT-22 / #3297 gauge-build-only quarantine
    (`cfg_attr(feature = "gauge-solver", ignore = "...")`). The
    default-build assertion (overnight ΔT > 2 °C) remains live and
    passing — this is the regression target for any future
    `h_ve_night` split.

- **Affected Metrics:** Case 950 (HVAC mode) annual cooling (kWh)
  and peak cooling (kW) — the load-bearing metrics. Case 950FF min
  free-floating temperature (°C) is the **co-regression target** per
  §LIMIT-17's regression-avoidance clause — both signatures must be
  closed by the **same single solver change** (the Issue #3551
  acceptance criterion).

- **Severity:** High. Closes a long-standing Case 950 (HVAC) annual
  cooling band failure (~91 % UNDER) and unblocks the §LIMIT-17
  FF-mode companion fix (the regression-avoidance clause in
  §LIMIT-17 has no current HVAC-mode value to preserve). The Cohort
  (Cases 600 / 620 / 900 / 920 / 940 / 950 / 960) is tracked by
  Issue #3072 (aggressive-baseline cohort).

- **GitHub Issue:** [#3551](https://github.com/anchapin/fluxion/issues/3551)
  (this entry), with related issues **#3058 / §LIMIT-17** (Case 950FF
  companion + regression-avoidance clause), **#3059 / §LIMIT-16**
  (Cases 610 / 630 / 650 peak cooling OVER — `MAX_CONVECTIVE_TO_AIR_MULTIPLIER`
  PR #3041 cohort), **#3041** (the PR that introduced the
  `MAX_CONVECTIVE_TO_AIR_MULTIPLIER = 2.0×` cap), **#2453**
  (900-series bidirectional annual-energy cohort), **#1898** (the
  PR that originally introduced `h_ve_night`), **#1422** (Case 950
  5R1C night-vent override tracking), **#1465 / #1462** (GaugeSolver
  implementation and validation harness — both closed individually;
  production-path switchover staged via **#3291 / PR #3482**),
  **#3072** (aggressive-baseline cohort tracking).
  Long-term fix routed to GaugeSolver rework **#1465 / #1462**;
  per-case parameter tuning to close this gap is explicitly out of
  scope (per AGENTS.md / RULES.md / ADR-0001).

- **Status:** 🟡 **Documentation/tracking only — no solver-code change
  in this PR.** The bidirectional signature cannot be closed by
  parameter tuning (closes one regression while opening the other)
  per AGENTS.md / RULES.md / ADR-0001; the architectural fix is the
  GaugeSolver rework #1465 / #1462. The structural decision is
  tracked in **`docs/adr/0011-case-950ff-night-vent-split.md`** (per
  §LIMIT-17's ADR pointer) and the broader Cohort tracking is owned
  by Issue #3072 (aggressive-baseline cohort). No physics-code change;
  no `h_ve_night`, `MAX_CONVECTIVE_TO_AIR_MULTIPLIER`, or
  `solar_distribution_to_air` adjustment; no
  `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`
  raise (per AGENTS.md "strict-energy-gate baseline must NEVER be
  raised to hide a regression").

- **Acceptance for the future structural PR (mirrors §LIMIT-17):**
  1. Case 950 (HVAC mode) annual cooling is within 390–920 kWh on
     the post-#3551 validator path.
  2. Case 950FF min free-floating temperature remains within
     [−20.20, −17.80] °C (the §LIMIT-17 regression-avoidance clause).
  3. Both signatures are closed by the **same single solver change**,
     i.e. the bidirectional fix requires GaugeSolver-style path
     splitting (multi-node mass vs air-node separation), not a
     per-parameter tuning.
  4. Case 950 (HVAC) peak cooling remains within [0.70, 0.90] kW
     (no regression on the secondary metric).
