# LIMIT-14 — investigation history

Narrative history for **LIMIT-14**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-14` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-14: Case 960 sunspace annual cooling and peak heating below band — GaugeSolver-blocked air-mass distribution gap (Issue #3061)

- **Description:** PR #3052 delivered the partial fix requested by Issue #2858:
  common-wall bulk conduction and the ground-reflected inter-zone gain path now
  couple the conditioned back-zone to the free-floating sunspace. The post-fix
  raw annual heating moved into band, but annual cooling and peak heating remain
  below the ASHRAE 140-2023 Case 960 inter-program reference envelope:

  | Metric | Post-#3052 result | Reference band | Verdict |
  |--------|-------------------|----------------|---------|
  | Annual cooling (raw) | 0.63 MWh | 1.55–2.78 MWh | **BELOW** |
  | Peak heating | 1.17 kW | 2.0–8.0 kW | **BELOW** |
  | Cooling validator (COP-adjusted) | 0.10 MWh | 1.55–2.78 MWh | **BELOW** |

  The same run reports raw annual heating at 2.14 MWh within the
  1.65–2.45 MWh band, confirming that PR #3052 improved the inter-zone path
  without closing the remaining load-distribution gap. Reference bands are
  maintained in `validation::benchmark` and summarised in
  `docs/ASHRAE140_MULTI_ZONE_RESULTS.md` §"Case 960 Reference Data".

- **Root cause:** The 5R1C + 9R4C air-mass distribution cannot accumulate
  enough back-zone cooling demand at the 27 °C cooling setpoint through
  inter-zone coupling alone. The free-floating sunspace receives the solar
  forcing, but the current air-to-mass distribution buffers and redistributes
  that forcing before enough of it reaches the conditioned back-zone air node. The same topology smooths the winter load response and
  leaves peak heating below band. This is the Case 960 manifestation of the
  structural limitation documented in §LIMIT-05 and coordinated by Issue
  #3059; it is not a missing common-wall conductance term after PR #3052.

- **Affected case and metrics:** Case 960 annual cooling (raw and
  COP-adjusted validator output) and peak heating. Peak cooling remains in its
  0–4 kW band; this entry does not alter any validation assertion or reference
  range.

- **Severity:** High for ASHRAE 140 compliance (two Case 960 reference-band
  metrics remain below band), with no safe case-local correction in the
  current solver topology.

- **Implementation options and risk analysis:**
  1. **Add a sunspace-side mechanical cooling setpoint — rejected.** The Case
     960 sunspace is specified as free-floating; adding mechanical cooling
     would simulate a different building and hide the coupling limitation
     behind a control that the benchmark does not contain.
  2. **Lower the sunspace-side `convective_to_air_factor` — rejected.** This
     would tune the solar gain split until more energy reaches the back-zone
     through conduction, without deriving a new distribution from first
     principles. It is parameter tuning to pass a system test, explicitly
     forbidden by RULES.md, AGENTS.md, and ADR-0001, and risks regressions in
     other multi-zone and solar-distribution cases.
  3. **Complete the GaugeSolver production-path switchover — required
     structural route.** Issue #3059 coordinates this unblocker through the
     GaugeSolver work in #1465 / #1462 (and the Phase A8 default flip in
     Issue #3291 / PR #3482, which wires `ThermalSelector::default() =
     ZoneSolverKind::Gauge` but intentionally retains the `gauge-solver`
     cargo feature as the production-path gate pending §LIMIT-21 closure).
     Those issues have shipped `GaugeSolver` shadow→production wiring
     and validation infrastructure, but the production `step_physics_5r1c` /
     `step_physics_9r4c` dispatch is not yet backed by the gauge path for the
     default build; the gauge code change landed unconditionally in
     `src/sim/thermal_model_physics/step_dispatcher.rs` but the cargo feature
     remains off-by-default until the β-soak gate (Issue #3286) trips. This
     option has broad solver, energy-balance, and cross-case regression risk,
     so it requires a dedicated architecture-reviewed physics PR rather than
     a Case 960 constant change.

- **Status:** 🔄 **Documentation/tracking only; blocked on Issue #3059 and the
  GaugeSolver production-path work (#1465 / #1462).** No physics, validation,
  test, reference-data, ARCHITECTURE.md, or RULES.md change is part of this
  entry. The existing GaugeSolver cohort tracking stub in
  `docs/adr/0007-gauge-solver-structural-work.md` already covers Case 960, so
  no duplicate ADR is needed for this documentation-only update.

- **Acceptance for the future structural PR:**
  1. Case 960 raw annual cooling is within 1.55–2.78 MWh.
  2. Case 960 peak heating is within 2.0–8.0 kW.
  3. The COP-adjusted validator cooling result is within its reference band.
  4. Energy-balance, cross-case ASHRAE 140, architecture-drift, and cycle
     guards remain green without changing
     `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`.

- **Linkage and provenance:**
  - Issue #2858 — origin of the Case 960 inter-zone coupling work.
  - PR #3052 — partial fix: common-wall bulk conduction and ground-reflected
    inter-zone gain path; exposed the residual cooling/heating gap above.
  - Issue #3059 — architectural unblocker coordinating the 5R1C/9R4C
    air-mass-distribution replacement through GaugeSolver.
  - Issue #1456 — removed the broken Case 960 6R2C override and exposed the
    default 5R1C/9R4C path on which this limitation occurs.
  - Issues #1465 / #1462 — GaugeSolver validation and `GaugeSolver`
    implementation; the Phase A8 production-path switchover is staged
    (Issue #3291 / PR #3482) — `ThermalSelector::default() =
    ZoneSolverKind::Gauge` is wired but the `gauge-solver` cargo feature
    remains the production-path gate pending §LIMIT-21 closure.
  - §LIMIT-10 / Issue #3065 — sister Case 960 free-floating sunspace
    temperature limitation with the same architectural unblocker.
  - `docs/adr/0007-gauge-solver-structural-work.md` — existing cohort-level
    tracking stub for the eventual architecture decision.
