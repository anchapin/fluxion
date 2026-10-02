# LIMIT-16 — investigation history

Narrative history for **LIMIT-16**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-16` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-16: Cases 610/630/650 peak cooling OVER — 5R1C + 9R4C air-mass distribution structural gap (Issue #3059)

- **Description:** PR #3041 (Issue #2871 partial fix) introduced
  `MAX_CONVECTIVE_TO_AIR_MULTIPLIER = 2.0×` cap on the cooling-mode governor
  symmetric to its heating counterpart and clamped the ACH-driven multiplier
  path on the 9R4C multi-node branch. Cases 620 and 640 closed into their
  ASHRAE 140-2023 reference bands, but Cases 610 / 630 / 650 stayed over the
  peak-cooling bands with the same underlying structural signature:

  | Case | Pre-#3041 peak | Post-#3041 peak | Ref band    | Verdict |
  |------|----------------|-----------------|-------------|---------|
  | 610  | 4.30 kW        | ~unchanged      | 2.20–2.90 kW | **OVER (+48 %)** |
  | 620  | over           | in band         | 3.2–5.0 kW  | PASS    |
  | 630  | over           | ~unchanged      | 2.2–2.7 kW  | **OVER (+39 %)** |
  | 640  | over           | in band         | 3.0–4.4 kW  | PASS    |
  | 650  | 4.30 kW        | ~unchanged      | 2.2–2.7 kW  | **OVER (+92 %)** |

  The three OVER cases form a coherent structural group (3/3 OVER with the
  same magnitude class and the same +48 % / +39 % / +92 % signature) that
  coincides exactly with the Cases 610 / 630 / 650 cohort flagged in the
  §LIMIT-05 UPDATE (#1457 revisit, 2026-07-10) per-case table (where Case
  610 is +48.3 % OVER, Case 630 is +39.2 % OVER, Case 650 is +92.4 % OVER).
  Cases 620 / 640 were brought into band by the cooling-mode governor
  symmetric + ACH-multiplier cap; Cases 610 / 630 / 650 cannot be brought
  into band by the same mechanism because the residual is the structural
  discrete-node air-mass distribution pathology, not a governor asymmetry.

- **Root cause (per Issue #3059):** The 5/5 OVER signature on the post-
  #3041 engine is structural — Fluxion's 5R1C `step_physics_5r1c` and 9R4C
  `step_physics_9r4c` paths use a single lumped thermal-mass node integrated
  on a 1-hour weather timestep (`dt/τ ≈ 3.6`, so the air node is ~98 %
  equilibrated within each step). The forced-convection term from the
  night-ventilation ACH (Case 650 has ACH = 13.14; this drives `h_tr_is`
  to peak ≈ 2.91 × at the cooling peak) dumps pulsed charging into the air
  node via forced convection on the morning ramp, on the 1-hour timestep.
  Sub-agent report on PR #3041 noted that `step_physics_5r1c` deliberately
  does NOT apply the ACH multiplier to `h_tr_is`, contradicting the issue's
  "h_tr_is to peak 0.84·ACH^0.8 ≈ 2.91×" diagnosis — the 5R1C code path does
  not produce that value in production. The OVER is **upstream of the
  multiplier**, in the lumped-mass integration, and cannot be closed by
  the `MAX_CONVECTIVE_TO_AIR_MULTIPLIER` cap that PR #3041 introduced.

  This is the same discrete-node solar-injection pathology documented in
  the §LIMIT-05 UPDATE (#1522) "air-node capacitance INFEASIBLE at 1 h
  timestep" investigation and the §LIMIT-05 UPDATE (#2300) "sub-hour air-
  node sub-stepping BLOCKED by architectural dependency" entry, both of
  which explicitly routed the structural fix to the GaugeSolver rework
  (#1465 / #1462). The Case 650 forced-convection contribution is a
  coupled manifestation of the same single-lumped-mass pathology.

- **Affected cases and metrics:** Case 610 peak_cooling (4.30 kW vs ref
  2.20–2.90 kW; +48 % OVER), Case 630 peak_cooling (3.34 kW vs ref
  1.80–2.40 kW; +39 % OVER), Case 650 peak_cooling (4.81 kW vs ref
  1.90–2.50 kW; +92 % OVER). Annual heating, annual cooling, peak
  heating, and free-floating temperatures for these cases are unchanged
  from the §LIMIT-05 UPDATE (#1457 revisit) table.

- **Severity:** High (strict ±15 % pass-rate gate does not currently admit
  Cases 610 / 630 / 650 peak_cooling), with no parameter-tuning escape
  hatch that closes the structural 5/5 OVER.

- **Implementation options and risk analysis (per Issue #3059 "Recommended
  Direction" + AGENTS.md / RULES.md / ADR-0001):**
  1. **Raise the `MAX_CONVECTIVE_TO_AIR_MULTIPLIER` cap above 2.0× — rejected.**
     Increasing the cap would re-introduce the pre-#3041 asymmetry that
     drove the bulk of Case 620's over-prediction and would also lift
     Cases 620 / 640 back into OVER. The Issue #3059 acceptance criterion
     explicitly forbids this ("do NOT raise baseline — RULES.md 'no
     parameter tuning' rule"). Per **AGENTS.md** ("fix the underlying
     math"), this is an anti-pattern.
  2. **Lower the ACH-driven multiplier saturation, or widen the
     reference-band tolerance for Cases 610 / 630 / 650 — rejected.** Per
     **RULES.md** ("no parameter tuning", "must-never hardcode
     results") and **ADR-0001** (No-Parameter-Tuning Rule), widening a
     band to absorb the OVER, OR removing the `MAX_CONVECTIVE_TO_AIR_
     MULTIPLIER` cap to widen `h_tr_is`, OR raising the strict-energy-
     gate baseline in `tests/reference_data/zone_balance/
     strict_energy_gate_baseline.json` is **parameter tuning** and is
     explicitly forbidden. The "do NOT raise baseline" clause in the
     Issue #3059 acceptance criterion enforces this for the gate, and
     the analogous principle applies to any band or multiplier widening
     that hides the structural OVER.
  3. **Complete the GaugeSolver production-path switchover — required
     structural route.** Per Issue #3059's "Recommended Direction":
     *"GaugeSolver #1465/#1462 needs to land first — that's the
     structural fix that turns the lumped mass into a true multi-node
     representation. Without GaugeSolver, the 5R1C + 9R4C air-mass
     distribution cannot accumulate enough back-zone cooling demand at
     the 27 °C cooling setpoint through inter-zone coupling alone (same
     root cause as #2858 partial fix)."* This is the **same** fix route
     as LIMIT-10 / LIMIT-11 / LIMIT-12 / LIMIT-13 / LIMIT-14 / LIMIT-15.
     The Issues #3059 / #3058 cohort (the Case 950FF night-vent mass-
     coupling F_sky correction) and #2858 / #3061 (Case 960 sunspace)
     are sister limitations with the identical architectural unblocker.
     The Phase A8 default flip (Issue #3291 / PR #3482) wires the gauge
     dispatch unconditionally in `src/sim/thermal_model_physics/step_dispatcher.rs`
     but intentionally retains the `gauge-solver` cargo feature as the
     production-path gate pending §LIMIT-21 closure; the **default
     build (feature OFF) continues to route the `Gauge` selector to
     5R1C/9R4C** until the β-soak program trips (Issue #3286), at
     which point this LIMIT's structural root cause lives inside the
     gauge dispatch path.

- **Status:** 🔄 **Documentation/tracking only; blocked on Issue #3059 and
  the GaugeSolver production-path work (#1465 / #1462).** No physics,
  validation, test, reference-data, ARCHITECTURE.md, or RULES.md change is
  part of this entry. The existing GaugeSolver cohort tracking stub in
  `docs/adr/0007-gauge-solver-structural-work.md` already covers Cases
  610 / 630 / 650 via the `63506-PASS / 65406-OVER` family, so no
  duplicate ADR is needed for this documentation-only update.

- **Acceptance for the future structural PR:**
  1. Case 610 peak_cooling within ±15 % of 2.55 kW reference midpoint
     (i.e. inside the 2.20–2.90 kW band), pre-#3041 baseline 4.30 kW.
  2. Case 630 peak_cooling within ±15 % of 2.10 kW reference midpoint
     (i.e. inside the 1.80–2.40 kW band), pre-#3041 baseline ~3.34 kW.
  3. Case 650 peak_cooling ≤ 2.5 kW (per the issue acceptance criterion
     "Case 650 peak cooling ≤ 2.5 kW (vs current 4.30 kW)") AND within
     ±15 % of the 2.20 kW reference midpoint. The 4.30 kW Case 650
     measurement is from the on-the-day PR #3041 sub-agent report; the
     #1457 revisit table reports 4.81 kW. Both are ≥ +92 % OVER band.
  4. Strict ±15 % annual-energy baseline
     (`tests/reference_data/zone_balance/strict_energy_gate_baseline.json`)
     is **NOT raised** to hide a regression (per the issue acceptance
     criterion "do NOT raise baseline — RULES.md 'no parameter tuning'
     rule").
  5. Energy-balance, cross-case ASHRAE 140, architecture-drift, and
     cycle guards remain green without altering any reference band or
     multiplier cap.

- **Linkage and provenance:**
  - Issue **#2871** — origin: Cases 610 / 620 / 630 / 640 / 650 peak
    cooling investigation that delivered the `MAX_CONVECTIVE_TO_AIR_
    MULTIPLIER = 2.0×` cap and the cooling-mode governor symmetric
    through PR #3041.
  - PR **#3041** — partial fix (closed #2871 for Cases 620 / 640);
    exposed the residual structural OVER on Cases 610 / 630 / 650.
  - Issue **#3058** — companion (Case 950FF night-vent mass coupling,
    same 5R1C structural limitation).
  - Issue **#3059** — this entry's origin; architectural unblocker
    coordinating the 5R1C / 9R4C air-mass-distribution replacement
    through GaugeSolver.
  - Issues **#1465 / #1462** — GaugeSolver validation and `GaugeSolver`
    implementation; the Phase A8 production-path switchover is staged
    via Issue #3291 / PR #3482 — `ThermalSelector::default() =
    ZoneSolverKind::Gauge` is wired but the `gauge-solver` cargo
    feature remains the production-path gate pending §LIMIT-21
    closure per `docs/adr/0007-gauge-solver-structural-work.md`
    §"Status of the underlying work".
  - §LIMIT-10 / Issue #3065 — sister Case 960 free-floating sunspace
    temperature limitation with the same architectural unblocker.
  - §LIMIT-11 / Issue #3064 — sister Case 195 high-mass walls zero-
    energy assertion with the same architectural unblocker.
  - §LIMIT-12 / Issue #3062 — sister Case 940 setback thermostat with
    the same architectural unblocker.
  - §LIMIT-13 / Issue #3063 — sister `h_tr_em` time-invariance with
    the same architectural unblocker.
  - §LIMIT-14 / Issue #3061 — sister Case 960 sunspace annual cooling
    with the same architectural unblocker.
  - §LIMIT-15 / Issue #3060 — sister Case 195 weather-data artefact
    with related-but-different architectural decision space.
  - §LIMIT-05 UPDATE (#1457 revisit) — the per-case 14-metric table
    that produced Cases 610 (4.30 kW) / 630 (3.34 kW) / 650 (4.81 kW)
    peak_cooling numbers, all +48 % / +39 % / +92 % OVER.
  - §LIMIT-05 UPDATE (#1522) and §LIMIT-05 UPDATE (#2300) — the
    air-node capacitance and sub-stepping investigations that already
    routed the structural fix to GaugeSolver #1465 / #1462.
  - `docs/adr/0007-gauge-solver-structural-work.md` — architectural
    unblocker for the cohort (proposed; not yet recorded).
  - `docs/ASHRAE140_RESULTS.md` §"Structural Blockers (Issue #3072)" —
    current pass-rate snapshot for the wider aggressive-baseline
    cohort (Cases 195 / 600 / 620 / 940 / 960 + sibling LIMIT-16
    Cases 610 / 630 / 650).

- **What this PR ships (documentation/tracking scaffolding):**
  1. **This LIMIT-16 entry** — categorises the Cases 610 / 630 / 650
     peak cooling OVER, links to #2871 / #3041 / #3058 / #3059 /
     #1465 / #1462, and gives the future implementer the per-case
     acceptance criteria.
  2. **`docs/ASHRAE140_RESULTS.md` §"Structural Blockers (Issue
     #3072)"** — a new LIMIT-16 sub-section row in the existing
     structural-blockers table that cross-references this entry and
     the GaugeSolver unblocker.

- **What this PR does NOT do (and why):**
  1. **It does NOT modify physics code.** Per **AGENTS.md** ("do NOT
     modify physics code without checking `ARCHITECTURE.md` first"),
     the actual structural fix requires the multi-node GaugeSolver
     implementation that turns the lumped mass into a true multi-node
     representation. That is out of scope for a single sub-agent
     without (a) deep physics expertise, (b) bit-identical baseline
     snapshot discipline (per ADR-0008), and (c) coordination with the
     GaugeSolver rework (#1465 / #1462 per #3059). This entry is
     documentation/tracking only — it does NOT propose, suggest, or
     hint at a tuning fix.
  2. **It does NOT modify
     `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`.**
     Per AGENTS.md ("the baseline must NEVER be raised to hide a
     regression") and the Issue #3059 acceptance criterion ("do NOT
     raise baseline — RULES.md 'no parameter tuning' rule"), this
     file is left untouched.
  3. **It does NOT widen any reference band.** Per **RULES.md**
     ("no parameter tuning") and **ADR-0001** (No-Parameter-Tuning
     Rule), widening the Case 610 / 630 / 650 peak-cooling bands to
     absorb the OVER would be band-space parameter tuning and is
     forbidden. The bands in `src/validation/benchmark.rs` and
     `tests/ashrae_140_blind_validation.rs` remain at the ASHRAE
     140-2023 inter-program envelope.
  4. **It does NOT modify `MAX_CONVECTIVE_TO_AIR_MULTIPLIER`** or any
     5R1C / 9R4C constant. Per ADR-0001 and the issue acceptance
     criterion, raising this constant would re-introduce the pre-
     #3041 asymmetry that drove Cases 620 / 640 into OVER.
  5. **It does NOT modify ARCHITECTURE.md or RULES.md.** Those are
     source-of-truth documents; this stub references them.
  6. **It does NOT record an architectural decision.** The GaugeSolver
     production-path switchover decision is already tracked by ADR-
     0007 (§"Status of the underlying work" — Phase 1b shadow mode
     ships; production-path switchover pending).

- **Why this is NOT a fixable tuning change (per AGENTS.md / RULES.md /
  ADR-0001):**
  1. The 5/5 OVER signature on the post-#3041 engine is the textbook
     discrete-node solar-injection pathology at `dt/τ ≈ 3.6`. Per
     §LIMIT-05 UPDATE (#1522), damping reduces BOTH peaks equally
     because it smooths the air-temperature swing symmetrically — no
     air-node damping can reduce the cooling peak while
     simultaneously increasing the heating peak. The architectural
     trade-off is **structurally infeasible at 1 h timestep**.
  2. Per §LIMIT-05 UPDATE (#2300) (issue #2300 investigation), the
     sub-hour air-node sub-stepping alternative requires a major
     architectural change to the `step_physics_5r1c` call path,
     weather timestep dispatch, scratch buffer management, and HVAC
     coupling — comparable in scope to the GaugeSolver rework.
  3. The genuinely architectural fix is the GaugeSolver rework
     (#1465 / #1462), which treats solar as geometric curvature rather
     than per-timestep energy injection. Both #1462 (Phase 1b shadow-
     mode implementation) and #1465 (Phase 3 ASHRAE 140 Case 900
     validation harness) are **closed** individually, but the
     **production-path switchover is NOT yet landed** — see
     `docs/adr/0007-gauge-solver-structural-work.md` §"Status of the
     underlying work".
  4. Per issue acceptance criterion, the strict ±15 % annual-energy
     baseline must NOT be raised; per ADR-0001, no 5R1C / 9R4C
     constant may be tuned; per AGENTS.md, the underlying math must
     be fixed; per RULES.md, results must not be hardcoded. All four
     constraints together leave only the structural GaugeSolver
     route.

- **Path forward (out of scope for this PR):**
  1. Ship the GaugeSolver production-path switchover (lands #1465
     validation harness outputs into `step_physics_5r1c` /
     `step_physics_9r4c`) per `docs/adr/0007-gauge-solver-structural-
     work.md`.
  2. Re-run `cargo test --test all_tests ashrae_140_case_600_series::` against
     the post-switchover engine.
3. When Cases 610 / 630 / 650 peak_cooling move into band, retire
      this LIMIT-16 entry and un-quarantine the per-case tests, in
      coordination with the `tests/known_issues_regression.rs::
      issue_1457_case_600_series_tracking::
      test_issue1457_remaining_600_series_metrics` reverse guard.
