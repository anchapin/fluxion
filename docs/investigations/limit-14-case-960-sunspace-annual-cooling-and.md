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

## Round 23 (2026-10-12): heating gap closed — implicit inter-zone coupling + physics common-wall fraction

**Item A — the suspected gain-path defect is FALSE (measured).** Round 22 noted a suspected pre-existing defect: per-window irradiance (W/m² of window) multiplied by zone floor area, a 12/16 over-collection on the sunspace aperture. Instrumented probe (`FLX_PROBE_GAIN`, throwaway, never committed) at Case 960 zone 1, timestep 12: `calculate_zone_solar_gain` returns the true window-area-weighted gain 2071.15 W (irradiance × 12 m² window × SHGC); `calc_analytical_loads` divides by the zone floor area (16 m²) purely to store W/m²-floor, and `step_physics_9r4c` re-multiplies by the same `setpoints.zone_area` — probe shows stored 129.4472 W/m² × 16 m² = 2071.15 W injected, to all printed digits. The round trip cancels exactly; there is no over-collection on the live gain path. (The diagnostics paths in `ashrae_140_validator/mod.rs` multiply the stored value by floor area for reporting — consistent, not a gain path.)

**Item B — stable coupling integration.** The explicit (lagged) inter-zone term `q_iz = h_iz·(T_j^old − T_i^old)` diverges at the physics conductance: instrumented trace at COMMON_WALL_FRACTION = 1.0 shows the free-floating sunspace oscillating +91.7 → −67.2 → +125.3 °C from hour 21, growing ≈ 1.4×/step until the HVAC capacity clamps. Fix: de-lag — the coupling is solved implicitly in both air-node estimates the engine commits (the 5R1C `t_i_free` network and the multi-node `t_i_free_mn` air balance), as a small dense linear system with SYMMETRIC pairwise conductances w_ij = (h_iz_i + h_iz_j)/2 (implicit Euler on the coupling term: unconditionally stable at any dt; symmetric weights make the pairwise exchange exactly energy-conserving, which the per-zone-weighted lagged form did not guarantee for heterogeneous zones). Single zone or zero h_tr_iz reduces to the previous uncoupled division, bit-identical.

**COMMON_WALL_FRACTION 0.25 → 1.0.** With stable coupling the physics value is usable. Strict gate (Golden-NREL): 960 H 2.517 → 2.085 MWh, IN BAND [1.742, 2.357] (gap 7.81 → 0 pp); C 0.414 → 0.511 (gap 65.87 → 61.39 pp, cooling-chain family gap stands). 970 H 3.300 → 2.004, C 2.026 → 0.869 (both further below the internal target of unverified origin; recorded in §LIMIT-23). All 15 single-zone annual cases and 4 free-float observables bit-identical.

**Superseded:** the round-22 fraction sweep (0.25/0.50/0.75 → H 2.517/2.362/2.341) is history; those values were never adopted.

## Round 24 (2026-10-13): spec-construction split — back zone lightweight per §5.2.2.2.7

**Interrupted-run note.** This round's first run was interrupted mid-execution by the service; state review found a coherent uncommitted diff in `/tmp/r24-wt` (the `zone_constructions` mechanism: `CaseSpec::zone_constructions`, builder `with_zone_constructions`, length validation, per-zone surface/capacitance build in `from_spec_with_selector`, serde-default `None`) plus the round-24 decomposition harness — mechanism complete, no measurements recorded, no 960 spec change, nothing pushed. The diff was adopted as-is, probes reverted, and the round continued from it.

**Decomposition (harness `zz_r24`, blind 8760-step, Golden-NREL; baseline reproduced exactly: 600 6.758/4.301, 900 1.680/1.684, 960 2.085/0.511, 970 2.004/0.869 MWh H/C):**

| Variant | H | C | Reading |
|---|---|---|---|
| E1 baseline (develop 76762935) | 2.085 | 0.511 | recorded |
| E2 both zones low-mass | 13.577 | 0.000 | not a candidate (whole-case swap) |
| E3 remove back-zone south window | 2.085 | 0.511 | bit-identical — the window already lands on the shared partition (round-21 zone-origin machinery), no solar aperture |
| E5 back zone low-mass only | 2.508 | 1.037 | **the 960-specific term** |
| E6 remove door entry | 2.085 | 0.511 | bit-identical — the door has no modeled air/conduction path since round 22 |
| E8 E5+E3+E6 (spec-corrected) | 2.508 | 1.037 | adopted |

**Spec finding.** 140-2017 §5.2.2.2.7 preamble: "The back zone shall be of lightweight construction, and the sun zone shall be of heavyweight construction"; §5.2.2.2.7.1a makes the back zone exactly Case 600 thermally. The shipped spec applied the Case 900 high-mass construction to BOTH zones. Frame/FF/900FF share nothing here — the residual was 960-specific (the 900-family cooling-chain gap remains as the under-band mechanism, now measured through a spec-compliant case).

**Fix.** `CaseSpec.zone_constructions` (optional, serde-default `None` → bit-identical legacy for every other case; length validated against `num_zones`); the 960 factory carries [Case-600 low-mass, Case-900 high-mass]. Inert-as-built entries removed as spec hygiene (E3/E6 bit-identical): the back-zone 12 m² south-window entry (§5.2.2.2.7.1a — replaced by the common wall) and the door entry (§5.2.2.2.7 specifies no door; §5.2.2.2.7.6 no inter-zonal exchange).

**Strict-gate re-record (Golden-NREL, both filters):** 960 H 2.085 → 2.508 MWh — heating LEAVES its band [1.742, 2.357] high (gap 0 → 7.37 pp of mid), recorded as known-fail with dated provenance (`_doc_round24`); C 0.511 → 1.037 MWh (gap 61.39 → 37.09 pp, improvement lowered in the same PR). Checker 9 PASS / 33 KNOWN-FAIL / 0 REGRESSION. 600/900/970 and all single-zone observables unchanged.

**Honest-cost statement.** The round-23 heating in-band result was reached with a spec-noncompliant both-high-mass back zone; the standard-mandated low-mass back zone costs 7.37 pp of heating gap. Per the fidelity-first standing direction the correct physics is adopted and the gap recorded, never tuned away.

**Rubber-duck follow-up (round 24 review, item 3):** the 5R1C longwave path (`step_5r1c.rs` ≈ line 797) broadcasts the GLOBAL `setpoints.wall_u_value/roof_u_value/floor_u_value` scalars to every zone, so the Case 960 low-mass back zone's LW surface exchange uses the sun-zone high-mass U. Pre-existing multi-zone simplification (Case 970 carries it too); round-24 numbers include it. Follow-up scope: thread per-zone U through the LW exchange buffers.

**Closes when:** the §5.2.2.2.7 construction split is landed with the re-record (this round) AND the remaining cooling under-band — the shared 900-family cooling-chain gap (§LIMIT-05/§LIMIT-17) — closes for the 960 geometry.
