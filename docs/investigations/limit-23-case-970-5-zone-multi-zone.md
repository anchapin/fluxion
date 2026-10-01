# LIMIT-23 — investigation history

Narrative history for **LIMIT-23**, moved verbatim out of `docs/KNOWN_ISSUES.md` by Issue #4278. The current state of this limitation is the `LIMIT-23` row in that document's status table; this file is the provenance behind it.

Blocks preserved: 1. No wording was changed, softened or deleted.

---

#### LIMIT-23: Case 970 5-zone multi-zone cross-coupling annual heating + cooling OVER and peak heating/cooling UNDER band — GaugeSolver-blocked air-mass distribution gap (Issue #3552)

- **Description:** Case 970 is the only multi-zone high-mass
  cross-coupling test in the ASHRAE 140 suite — an 8 m × 6 m × 2.7 m
  high-mass concrete envelope divided into 5 zones by 7 interior
  partitions (1 west core + 4 east-strip, 200 mm concrete interior
  partitions, 5×5 symmetric inter-zone conductance matrix per
  `sim::multi_zone_network::MultiZoneAirflowNetwork`). The Case 970
  reference envelope is established in Issue #1446 (closed via #1467)
  and emitted into `tests/reference_data/zone_balance/
  case_970_energy_reference.csv` by
  `tests/reference_data/zone_balance/generate_case_970_energy.py`.
  `tests/ashrae_140_case_970_validation.rs::test_case_970_multi_zone_network_e2e_conservation`
  verifies the 5×5 N-zone algebraic identity `Σ q_iz[i] ≈ 0 W`
  end-to-end across 8 760 hourly `solve_step` calls (`Issue #1348`,
  1e-6 W tolerance).

  The 2026-08-16 ASHRAE 140 snapshot reports **4 / 4** Case 970 reference-band
  metrics failing on the post-#1407 / post-#1446 engine:

  | Metric             | Engine output (2026-08-16 snapshot) | ASHRAE 140-2017 §B6.7 / 140-2023 Annex B8-3 reference band | Verdict |
  |--------------------|------------------------------------:|------------------------------------------------------------:|---------|
  | Annual heating     | **18.58 MWh**                       | [10.54, 14.26] MWh                                          | **OVER** (+30 % to +76 % above band) |
  | Annual cooling     | **21.07 MWh**                       | [7.39, 10.00] MWh                                           | **OVER** (+110 % to +185 % above band) |
  | Peak heating       | **3.80 kW**                         | [4.00, 8.00] kW                                             | **UNDER** (5 % below band low edge) |
  | Peak cooling       | **2.58 kW**                         | [2.50, 5.50] kW                                             | **UNDER** (at band low edge) |

  The bidirectional annual OVER signature (heating AND cooling both
  +30 % to +185 % above their respective bands simultaneously on the
  same 5-zone topology) is consistent with the §LIMIT-05 UPDATE (#2453)
  900-series bidirectional annual-energy over-prediction mechanism:
  the 5R1C/9R4C single lumped mass node cannot capture the
  per-zone, per-orientation air-mass distribution across a 5×5
  inter-zone conductance matrix, and the same solar mass-node
  over-charge that drives the 900-series OVER is amplified by the
  inter-zone coupling topology. The peak heating/cooling UNDER
  signature is the same loss-of-amplitude-side signal as §LIMIT-14
  (Case 960 sunspace annual cooling and peak heating below band) and
  §LIMIT-16 (Cases 610/630/650 peak cooling OVER). Per AGENTS.md /
  RULES.md / ADR-0001 ("no parameter tuning", "must-never hardcode
  results"), none of the four failure metrics can be closed by
  adjusting `inter_zone_conductance`, `solar_distribution_to_air`,
  or `h_ms_coeff`; the bidirectional trade-off is structurally
  infeasible at `dt/τ ≈ 3.6` per the §LIMIT-05 UPDATE (#1522)
  air-node capacitance conclusion. This entry is
  **documentation/tracking only** — it does not propose, suggest, or
  hint at a tuning fix.

- **Affected case and metrics:** Case 970 annual heating, annual
  cooling, peak heating, and peak cooling (4 / 4 reference-band
  metrics failing). The Case 970 multi-zone network conservation
  identity `Σ q_iz[i] ≈ 0 W` (`Issue #1348`) is satisfied end-to-end
  in `tests/ashrae_140_case_970_validation.rs`; this entry does not
  alter any validation assertion, reference range, or the
  `case_970_energy_reference.csv` band.

- **Severity:** High for ASHRAE 140 compliance (four Case 970
  reference-band metrics outside the band — annual heating + cooling
  both OVER, peak heating/cooling both UNDER), with no safe case-local
  correction in the current 5R1C/9R4C + 5×5 inter-zone solver
  topology. The bidirectional OVER + UNDER signature is a single
  structural gap, not four independent failures.

- **Implementation options and risk analysis:**
  1. **Adjust the Case 970 5×5 inter-zone conductance matrix entries
     — rejected.** The matrix entries are derived from the ASHRAE
     140-2017 §B6.7 / 140-2023 Annex B8-3 common-wall U-values and
     partition geometry (200 mm concrete, 4.05 m² zone-0↔east-strip
     walls, 10.8 m² adjacent-east-strip walls); adjusting them to
     absorb the OVER is parameter tuning to pass a system test and is
     explicitly forbidden by RULES.md, AGENTS.md, and ADR-0001.
  2. **Lower the per-zone `convective_to_air_factor` /
     `solar_distribution_to_air` to reduce air-mass distribution
     amplitude — rejected.** Same rationale as §LIMIT-14 option 2 —
     this tunes the solar gain split without deriving a new
     distribution from first principles and risks regressions in
     Cases 600 / 900 / 940 / 950 / 960 (the §LIMIT-05 / LIMIT-14 /
     LIMIT-16 / LIMIT-17 cohort).
  3. **Raise `tests/reference_data/zone_balance/case_970_energy_reference.csv`
     bands to absorb the OVER — rejected.** The reference CSV is the
     ASHRAE 140 inter-program range across EnergyPlus 25.2.0, TRNSYS,
     ESP-r, DOE-2, BSIMAC, CSE, and DeST (per `docs/ASHRAE140_MULTI_ZONE_RESULTS.md`
     §"Case 970 Reference Data"). Raising it to absorb a known engine
     OVER would constitute "parameter tuning in band space" and is
     explicitly forbidden by AGENTS.md / RULES.md / ADR-0001.
  4. **Complete the GaugeSolver production-path switchover — required
     structural route.** Issue #3059 (Case 960 / 610 / 630 / 650 /
     950FF cohort) and the #1465 / #1462 (Phase 1b `GaugeSolver`
     implementation + Phase 3 ASHRAE 140 Case 900 validation harness)
     program coordinate this unblocker. The Phase A8 default flip
     (Issue #3291, PR #3482) wires `ThermalSelector::default() =
     ZoneSolverKind::Gauge` but intentionally retains the
     `gauge-solver` cargo feature as the production-path gate pending
     §LIMIT-21 closure (the β-soak program, Issue #3286). The Case 970
     N-zone air-trajectory fidelity depends on #1465 / #1462 — the
     multi-zone air-mass distribution is structurally identical to
     the Case 600 / 900 air-mass distribution that #1465 / #1462 is
     scoped to repair. This option has broad solver, energy-balance,
     and cross-case regression risk, so it requires a dedicated
     architecture-reviewed physics PR rather than a Case 970 constant
     change.

- **Status:** 🔄 **Documentation/tracking only; blocked on Issue #3059
  and the GaugeSolver production-path work (#1465 / #1462).** No
  physics, validation, test, reference-data, ARCHITECTURE.md, or
  RULES.md change is part of this entry. Per Issue #3552 acceptance
  criteria and the explicit scope guard, this entry does NOT add
  Case 970 to the §LIMIT-05 / #3072 aggressive-baseline cohort
  table (Cases 195 / 600 / 620 / 940 / 960) — the cohort list is
  #3072's purview; Case 970 is tracked separately under this
  §LIMIT-23 entry until #3072 explicitly expands the cohort scope.
  The per-zone attribution diagnostic test stub
  `tests/diagnostics/case_970_multi_zone_seasonal_attribution.rs`
  (Issue #3552 acceptance criterion (c)) is added in this PR as an
  `#[ignore]`-quarantined placeholder with a `// TODO: implement`
  marker; the full per-month, per-zone attribution implementation is
  a follow-up PR.

- **Acceptance for the future structural PR:**
  1. Case 970 annual heating is within [10.54, 14.26] MWh.
  2. Case 970 annual cooling is within [7.39, 10.00] MWh.
  3. Case 970 peak heating is within [4.00, 8.00] kW.
  4. Case 970 peak cooling is within [2.50, 5.50] kW.
  5. The Case 970 multi-zone network conservation identity
     `Σ q_iz[i] ≈ 0 W` (`Issue #1348`, 1e-6 W tolerance) remains
     satisfied end-to-end across all 8 760 hourly `solve_step` calls.
  6. Energy-balance, cross-case ASHRAE 140, architecture-drift, and
     cycle guards remain green without changing
     `tests/reference_data/zone_balance/case_970_energy_reference.csv`,
     `tests/reference_data/zone_balance/strict_energy_gate_baseline.json`,
     or any of `inter_zone_conductance`, `solar_distribution_to_air`,
     `h_ms_coeff`.

- **Linkage and provenance:**
  - Issue #1446 — Case 970 reference + MultiZoneNetwork e2e
    validation (closed via PR #1467). Establishes the 5×5 symmetric
    conductance matrix and the `tests/reference_data/zone_balance/
    case_970_energy_reference.csv` envelope used by
    `validation::ashrae_140_multi_zone::Case970Reference`.
  - Issue #1348 — N-zone network algebraic identity `Σ q_iz[i] ≈ 0 W`
    for a symmetric conductance matrix; the conservation contract
    that `tests/ashrae_140_case_970_validation.rs::
    test_case_970_multi_zone_network_e2e_conservation` enforces
    end-to-end.
  - Issue #3059 — architectural unblocker coordinating the
    5R1C/9R4C air-mass-distribution replacement through GaugeSolver.
    §LIMIT-23 does not duplicate #3059; it documents the Case 970
    specific bidirectional signature and routes the unblocker through
    the same GaugeSolver program.
  - Issues #1465 / #1462 — GaugeSolver validation and `GaugeSolver`
    implementation; the Phase A8 production-path switchover is
    staged (Issue #3291 / PR #3482) — `ThermalSelector::default() =
    ZoneSolverKind::Gauge` is wired but the `gauge-solver` cargo
    feature remains the production-path gate pending §LIMIT-21
    closure.
  - §LIMIT-05 UPDATE (#2453) — 900-series bidirectional
    annual-energy over-prediction; the §LIMIT-23 bidirectional annual
    OVER signature is the Case 970 multi-zone analog of this
    mechanism.
  - §LIMIT-14 / Issue #3061 — Case 960 sunspace annual cooling and
    peak heating below band (2-zone cross-coupling); §LIMIT-23 is the
    Case 970 5-zone cross-coupling extension (same architectural
    unblocker, distinct test scaffolding).
  - §LIMIT-16 / Issue #3059 — Cases 610 / 630 / 650 peak cooling
    OVER (single-zone 5R1C air-mass distribution limitation); same
    cohort root cause as §LIMIT-23.
  - §LIMIT-05 / #3072 cohort — Cases 195 / 600 / 620 / 940 / 960;
    §LIMIT-23 is intentionally NOT added to this cohort per Issue
    #3552 acceptance criteria ("tracked separately; the cohort list
    is #3072's purview"). The #3072 cohort external-references
    section notes §LIMIT-23 as the Case 970 tracking entry.
  - `docs/adr/0007-gauge-solver-structural-work.md` — existing
    cohort-level tracking stub for the eventual architecture decision
    (covers the gauge-solver scope; Case 970 falls under the same
    GaugeSolver program).
  - `tests/diagnostics/case_970_multi_zone_seasonal_attribution.rs`
    (this PR) — `#[ignore]`-quarantined per-month, per-zone
    attribution diagnostic stub (Issue #3552 acceptance criterion
    (c)); full implementation deferred to a follow-up PR.
